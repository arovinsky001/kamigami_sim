import os
import numpy as np

from dm_control.composer import Task, Environment, Entity, variation
from dm_control.composer.variation import distributions
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas.floors import Floor as Arena
from dm_control.utils.transformations import euler_to_quat, quat_to_euler
from dm_control import mjcf

from kamigami_robot import KamigamiRobot

PHYS_TS = 0.02
CTRL_TS = 0.4


class UniformCircle(variation.Variation):
    """ A uniformly sampled pose within a within a circle centered at the origin"""
    def __init__(self, radius, inner_radius=0., z=0.05):
        self._radius = distributions.Uniform(inner_radius, radius)
        self._theta = distributions.Uniform(0, 2*np.pi)
        self._heading = distributions.Uniform(0, 2*np.pi)
        self._z = z

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        radius, theta, heading = variation.evaluate(
            (self._radius, self._theta, self._heading),
            random_state=random_state
        )

        pos = np.array([radius * np.cos(theta), radius * np.sin(theta), self._z])

        euler_vec = np.array([0., 0., heading])
        quat = euler_to_quat(euler_vec)

        return pos, quat


class SimTaskSingleRobot(Task):
    zero_steps = int(0.8 * CTRL_TS / PHYS_TS)
    init_steps = int(1. / PHYS_TS)

    def __init__(self, name='walker'):
        super().__init__()
        self._name = name

        self._arena = Arena()
        self._robot = KamigamiRobot(name=name)
        self._arena.attach(self._robot).add("freejoint")

        self._target = self.root_entity.mjcf_model.worldbody.add(
            'site',
            name='target',
            type='sphere',
            pos=(0., 0., 0.),
            size=(0.1,),
            rgba=(0.9, 0.6, 0.6, 1.0)
        )
        self._robot_init_pose = UniformCircle(3.)

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        def robot_xy_heading(physics):
            pos, quat = [p.copy() for p in self._robot.get_pose(physics)]
            _, _, heading = quat_to_euler(quat)
            return np.array([pos[0], pos[1], heading])

        def is_healthy(physics):
            _, quat = [p.copy() for p in self._robot.get_pose(physics)]
            roll, pitch, _ = quat_to_euler(quat)

            roll, pitch = roll % (2 * np.pi), pitch % (2 * np.pi)
            roll = 2 * np.pi - roll if roll > np.pi else roll
            pitch = 2 * np.pi - pitch if pitch > np.pi else pitch

            healthy = (roll < np.pi / 6) and (pitch < np.pi / 6)
            return healthy

        self._task_observables = {}
        self._task_observables['robot_state'] = observable.Generic(robot_xy_heading)
        self._task_observables['is_healthy'] = observable.Generic(is_healthy)

        for obs in self._task_observables.values():
            obs.enabled = True

        self.set_timesteps(physics_timestep=PHYS_TS, control_timestep=CTRL_TS)

    def initialize_episode(self, physics, random_state):
        self._robot.initialize_episode(physics, random_state)

        self._physics_variator.apply_variations(physics, random_state)
        robot_pos, robot_quat = variation.evaluate((self._robot_init_pose,), random_state=random_state)[0]
        self._robot.set_pose(physics, position=robot_pos, quaternion=robot_quat)

        self.after_step(physics, random_state, n_steps=self.init_steps)

    def before_step(self, physics, action, random_state):
        self._robot.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state, n_steps=None):
        self._robot.apply_action(physics, np.zeros(2), random_state)

        for _ in range(n_steps or self.zero_steps):
            physics.step()

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def get_reward(self, physics):
        robot_xyz = self._robot.get_pose(physics)[0].copy()
        reward = -np.linalg.norm(robot_xyz[:2])

        return reward


class SimTaskMultiRobot(SimTaskSingleRobot):
    def __init__(self, name='walker', n_robots=2):
        Task.__init__(self)
        self._name = name
        self._n_robots = n_robots

        self._arena = Arena()
        self._robots = [KamigamiRobot(name=name) for _ in range(n_robots)]
        for robot in self._robots:
            self._arena.attach(robot).add("freejoint")

        self._target = self.root_entity.mjcf_model.worldbody.add(
            'site',
            name='target',
            type='sphere',
            pos=(0., 0., 0.),
            size=(0.1,),
            rgba=(0.9, 0.6, 0.6, 1.0)
        )
        self._robot_init_pose = UniformCircle(3.)

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        def robot_xy_heading(physics):
            robot_states = np.empty((self._n_robots, 3))

            for robot, state in zip(self._robots, robot_states):
                pos, quat = [p.copy() for p in robot.get_pose(physics)]
                _, _, heading = quat_to_euler(quat)
                state[:] = [pos[0], pos[1], heading]

            return robot_states

        def is_healthy(physics):
            for robot in self._robots:
                _, quat = [p.copy() for p in robot.get_pose(physics)]
                roll, pitch, _ = quat_to_euler(quat)

                roll, pitch = roll % (2 * np.pi), pitch % (2 * np.pi)
                roll = 2 * np.pi - roll if roll > np.pi else roll
                pitch = 2 * np.pi - pitch if pitch > np.pi else pitch

                healthy = (roll < np.pi / 6) and (pitch < np.pi / 6)
                if not healthy:
                    return False

            return True

        self._task_observables = {}
        self._task_observables['robot_states'] = observable.Generic(robot_xy_heading)
        self._task_observables['is_healthy'] = observable.Generic(is_healthy)

        for obs in self._task_observables.values():
            obs.enabled = True

        self.set_timesteps(physics_timestep=PHYS_TS, control_timestep=CTRL_TS)

    def initialize_episode(self, physics, random_state):
        for robot in self._robots:
            robot.initialize_episode(physics, random_state)

            self._physics_variator.apply_variations(physics, random_state)
            robot_pos, robot_quat = variation.evaluate((self._robot_init_pose,), random_state=random_state)[0]
            robot.set_pose(physics, position=robot_pos, quaternion=robot_quat)

        self.after_step(physics, random_state, n_steps=self.init_steps)

    def before_step(self, physics, action, random_state):
        actions_per_robot = np.split(action, self._n_robots)

        for robot, action in zip(self._robots, actions_per_robot):
            robot.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state, n_steps=None):
        for robot in self._robots:
            robot.apply_action(physics, np.zeros(2), random_state)

        for _ in range(n_steps or self.zero_steps):
            physics.step()

    def get_reward(self, physics):
        robots_xyz = np.empty((self._n_robots, 3))

        for robot, xyz in zip(self._robots, robots_xyz):
            xyz[:] = robot.get_pose(physics)[0].copy()

        reward = -np.linalg.norm(robots_xyz[:, :2], axis=-1).sum()

        return reward


class SimTaskMultiRobotObject(SimTaskMultiRobot):
    class BlockObject(Entity):
        """A composer entity representing a block object."""
        _OBJECT_XML_PATH = os.path.join(os.path.dirname(__file__), "object.xml")

        def _build(self, name='block'):
            """Initializes the object.

            Args:
            name: String, the name of this object. Used as a prefix in the MJCF name
                name attributes.
            """
            self._mjcf_root = mjcf.from_path(self._OBJECT_XML_PATH)
            if name:
                self._mjcf_root.model = name

            # Find MJCF elements that will be exposed as attributes.
            self._bodies = self.mjcf_model.find_all('body')

        @property
        def mjcf_model(self):
            """Returns the `mjcf.RootElement` object corresponding to this robot."""
            return self._mjcf_root

    def __init__(self, name='walker', n_robots=2, object_init_radius=0.5, goal_clip_dist=np.inf):
        Task.__init__(self)
        self._name = name
        self._n_robots = n_robots
        self._object_init_radius = object_init_radius
        self._goal_clip_dist = goal_clip_dist

        self._arena = Arena()
        self._robots = [KamigamiRobot(name=name) for _ in range(n_robots)]
        for robot in self._robots:
            self._arena.attach(robot).add("freejoint")

        self._object = self.BlockObject()
        self._arena.attach(self._object).add("freejoint")

        self._target = self.root_entity.mjcf_model.worldbody.add(
            'site',
            name='target',
            type='sphere',
            pos=(0., 0., 0.),
            size=(0.05,),
            rgba=(0.9, 0.6, 0.6, 1.0)
        )
        self._robot_init_pose = UniformCircle(0.4, inner_radius=0.27, z=0.)
        self._object_init_pose = UniformCircle(1.2, inner_radius=object_init_radius)

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        def robot_xy_heading(physics):
            robot_states = np.empty((self._n_robots, 3))

            for robot, state in zip(self._robots, robot_states):
                pos, quat = [p.copy() for p in robot.get_pose(physics)]
                _, _, heading = quat_to_euler(quat)
                state[:] = [pos[0], pos[1], heading]

            return robot_states

        def object_xy_heading(physics):
            pos, quat = self._object.get_pose(physics)
            _, _, heading = quat_to_euler(quat)
            return np.array([pos[0], pos[1], heading])

        def is_healthy(physics):
            healthy_lim = 35
            entities = self._robots + [self._object]

            for entity in entities:
                _, quat = [p.copy() for p in entity.get_pose(physics)]
                roll, pitch, _ = quat_to_euler(quat)

                roll, pitch = roll % (2 * np.pi), pitch % (2 * np.pi)
                roll = 2 * np.pi - roll if roll > np.pi else roll
                pitch = 2 * np.pi - pitch if pitch > np.pi else pitch

                healthy = (roll < healthy_lim * np.pi / 180) and (pitch < healthy_lim * np.pi / 180)
                if not healthy:
                    return False

            return True

        self._task_observables = {}
        self._task_observables['robot_states'] = observable.Generic(robot_xy_heading)
        self._task_observables['object_state'] = observable.Generic(object_xy_heading)
        self._task_observables['is_healthy'] = observable.Generic(is_healthy)

        for obs in self._task_observables.values():
            obs.enabled = True

        self.set_timesteps(physics_timestep=PHYS_TS, control_timestep=CTRL_TS)

    def initialize_episode(self, physics, random_state):
        self._object.initialize_episode(physics, random_state)
        for robot in self._robots:
            robot.initialize_episode(physics, random_state)

        self._physics_variator.apply_variations(physics, random_state)
        object_pos, object_quat = variation.evaluate((self._object_init_pose,), random_state=random_state)[0]
        self._object.set_pose(physics, position=object_pos, quaternion=object_quat)

        robot_pos_lst = np.empty((self._n_robots, 3))

        for i, robot in enumerate(self._robots):
            self._physics_variator.apply_variations(physics, random_state)
            robot_pos, robot_quat = variation.evaluate((self._robot_init_pose,), random_state=random_state)[0]

            if i > 0:
                while np.any(np.linalg.norm((robot_pos_lst[:i] - robot_pos)[:2], axis=-1) < 0.2):
                    self._physics_variator.apply_variations(physics, random_state)
                    robot_pos, robot_quat = variation.evaluate((self._robot_init_pose,), random_state=random_state)[0]

            robot_pos += object_pos
            robot.set_pose(physics, position=robot_pos, quaternion=robot_quat)

            robot_pos_lst[i] = robot_pos.copy()

        self.after_step(physics, random_state, n_steps=self.init_steps)

    def before_step(self, physics, action, random_state):
        actions_per_robot = np.split(action, self._n_robots)

        for robot, action in zip(self._robots, actions_per_robot):
            robot.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state, n_steps=None):
        for robot in self._robots:
            robot.apply_action(physics, np.zeros(2), random_state)

        for _ in range(n_steps or self.zero_steps):
            physics.step()

    def get_reward(self, physics):
        object_xyz = self._object.get_pose(physics)[0].copy()
        object_dist_to_goal = np.linalg.norm(object_xyz[:2])
        # object_dist_to_goal = object_dist_to_goal.clip(0., self._goal_clip_dist)

        robot_xys = np.empty((self._n_robots, 2))
        for i, robot in enumerate(self._robots):
            robot_xyz = robot.get_pose(physics)[0].copy()
            robot_xys[i] = robot_xyz[:2]
        robots_dist_to_object = np.linalg.norm(object_xyz[:2] - robot_xys, axis=-1).sum()

        reward = -object_dist_to_goal -0.1 * robots_dist_to_object
        # reward = 0.

        return reward


def create_sim_environment(n_robots=1, use_object=False):
    if use_object:
        task = SimTaskMultiRobotObject(n_robots=n_robots)
    elif n_robots == 1:
        task = SimTaskSingleRobot()
    else:
        task = SimTaskMultiRobot(n_robots=n_robots)

    return Environment(task=task, strip_singleton_obs_buffer_dim=True)


if __name__ == "__main__":
    from dm_control import viewer

    env = create_sim_environment()

    # import pdb;pdb.set_trace()

    def differential_drive_policy(time_step):
        state = time_step.observation['robot_state'].squeeze()
        goal = np.zeros(3)

        xy_vector_to_goal = (goal - state)[:2]
        target_heading = np.arctan2(xy_vector_to_goal[1], xy_vector_to_goal[0])

        signed_heading_error = (target_heading - state[2] + 3 * np.pi) % (2 * np.pi) - np.pi
        abs_heading_error = np.abs(signed_heading_error)
        heading_cost = min(abs_heading_error, np.pi - abs_heading_error) / (np.pi / 2)

        forward = (abs_heading_error < np.pi / 2)

        left = (signed_heading_error > 0)
        left = left if forward else not left

        forward = forward * 2 - 1
        left = left * 2 - 1

        weight = 1.5
        scale = 1.2

        target_angular_vel = heading_cost * left
        target_linear_vel = (1. - heading_cost) * forward

        left_action = (target_linear_vel - target_angular_vel * weight) * scale
        right_action = (target_linear_vel + target_angular_vel * weight) * scale

        action = np.array([left_action, right_action])
        action /= np.clip(np.abs(action).max(), 1., None)

        print(action)
        return action

    viewer.launch(env, policy=differential_drive_policy)
