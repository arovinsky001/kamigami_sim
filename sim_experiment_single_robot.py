import numpy as np
from tqdm import tqdm, trange

from sim_experiment_base import ExperimentBase
from train_utils import train_from_buffer

np.set_printoptions(precision=4, suppress=True)


def differential_drive_policy(obs):
    state = obs['robot_state'].squeeze()
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

    return action


class Experiment(ExperimentBase):
    def run(self):
        for episode in trange(self.n_episodes, desc='Episode', position=0, leave=True):
            _, _, _, obs = self.env.reset()
            while not obs['is_healthy']:
                _, _, _, obs = self.env.reset()

            episode_rewards = []
            episode_images = []

            for step in trange(self.episode_length, desc='Step', position=1, leave=True):
                if self.overall_step == self.random_steps and self.pretrain_agent:
                    tqdm.write(f'\n\nPRE-TRAIN REPLAY BUFFER SIZE: {self.replay_buffer.size}\n')
                    train_from_buffer(self.agent, self.replay_buffer, pretrain_samples=self.replay_buffer.size)

                state = obs['robot_state']

                if self.overall_step >= self.random_steps:
                    action, next_state_prediction = self.agent.get_action(state, self.goals)
                    if step % int(self.episode_length / 5) == 0:
                        tqdm.write(f'ACTION: {action}')
                else:
                    # action = np.random.uniform(-1., 1., size=2*self.n_robots)
                    action = self.fixed_actions[step]
                    # action = differential_drive_policy(obs)

                _, reward, _, next_obs = self.env.step(action)

                next_state = next_obs['robot_state']
                self.replay_buffer.add(state, action, next_state)

                if self.is_update_step(self.overall_step) and self.overall_step >= self.random_steps:
                    for model in self.agent.models:
                        for _ in range(self.steps_per_update):
                            model.update(*self.replay_buffer.sample(self.batch_size))

                # tqdm.write(f'Reward: {reward}')
                episode_rewards.append(reward)
                if episode % self.log_episode_freq == 0:
                    image = self.env.physics.render(camera_id=self.n_robots+self.use_object)
                    episode_images.append(image)

                obs = next_obs

                self.log_step(reward, action, episode)
                self.overall_step += 1

            tqdm.write(f'\nFinal Reward: {reward}')
            tqdm.write(f'Replay Buffer Size: {self.replay_buffer.size}')
            self.all_rewards.append(episode_rewards)

            if episode % self.log_episode_freq == 0:
                self.log_episode(episode_rewards, episode_images, episode, reward_desc='-robot_dist_to_goal')


if __name__ == '__main__':
    experiment = Experiment()
    experiment.run()

# python sim_experiment_single_robot.py -buffer_capacity=1000000 -n_episodes=40000 -episode_length=25 -batch_size=250 -lr=3e-4 -ensemble_size=1 -n_particles=1 -discount_factor=1. -mpc_method=mppi -mpc_horizon=3 -mpc_samples=200 -noise_std=0.9 -beta=1. -gamma=50. -refine_iters=5 -n_best=20 -alpha=0.8 -hidden_dim=500 -hidden_depth=2 -update_freq=0 -steps_per_update=10 -train_epochs=100 -dist -scale -robot_ids 0 -random_steps=36 -cost_distance_weight=1. -cost_std_weight=0. -exp_name=single_robot_test -pretrain_agent -robot_goals