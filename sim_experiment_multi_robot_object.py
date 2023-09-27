import numpy as np
from tqdm import tqdm, trange
import wandb
from cpprb import MPReplayBuffer

from sim_experiment_base import ExperimentBase, ExperimentMultiprocessBase
from train_utils import train_from_buffer

np.set_printoptions(precision=4, suppress=True)


class Experiment(ExperimentBase):
    def obs_to_state(self, obs):
        robot_states = obs['robot_states'].reshape(3*self.n_robots)
        object_state = obs['object_state']
        return np.concatenate([robot_states, object_state], axis=0)

    def run(self):
        for episode in trange(self.n_episodes, desc='Episode', position=0, leave=True):
            _, _, _, obs = self.env.reset()
            while not obs['is_healthy']:
                _, _, _, obs = self.env.reset()

            episode_rewards = []
            episode_images = []

            # if episode % self.eval_episode_freq == max(self.eval_episode_freq - 1, 2):
            #     self.agent.cost_std_weight = 0.1
            #     self.agent.mppi_gamma *= 2.
            # elif episode % self.eval_episode_freq == 0:
            #     self.agent.cost_std_weight = self.cost_std_weight
            #     self.agent.mppi_gamma = self.mppi_gamma

            for step in trange(self.episode_length, desc='Step', position=1, leave=True):
                if self.overall_step == self.random_steps and self.pretrain_agent:
                    tqdm.write(f'\n\nPRE-TRAIN REPLAY BUFFER SIZE: {self.replay_buffer.get_stored_size()}\n')
                    train_from_buffer(self.agent, self.replay_buffer, pretrain_samples=self.replay_buffer.get_stored_size())
                    # train_from_buffer(self.agent, self.replay_buffer, pretrain_samples=50000)

                state = self.obs_to_state(obs)

                if self.overall_step >= self.random_steps:
                    # if self.overall_step - self.random_steps < 1000:
                    #     self.agent.cost_std_weight = -4.
                    #     self.agent.gamma = 7.
                    # else:
                    #     self.agent.cost_std_weight = 0.2
                    #     self.agent.gamma = 9.

                    action, next_state_prediction = self.agent.get_action(state, self.goals)
                    if step % int(self.episode_length / 5) == 0:
                        tqdm.write(f'ACTION: {action}')
                else:
                    action = np.random.uniform(-1., 1., size=2*self.n_robots)
                    # beta = 0.7
                    # action = np.random.beta(beta, beta, size=2*self.n_robots) * 2. - 1.

                _, reward, _, next_obs = self.env.step(action)

                # tqdm.write(f'Reward: {reward}')
                episode_rewards.append(reward)

                if self.overall_step % self.log_step_freq == 0 or step == self.episode_length - 1:
                    self.log_step(reward, action, episode)

                if episode % self.eval_episode_freq == 0 and episode % self.log_video_freq == 0:
                    image = self.env.physics.render(height=120, width=160, camera_id=self.n_robots+self.use_object)
                    # image = self.env.physics.render(height=480, width=640, camera_id=self.n_robots+self.use_object)
                    episode_images.append(image)

                if not next_obs['is_healthy']:
                    tqdm.write('\n\nNOT HEALTHY\n')
                    while not next_obs['is_healthy']:
                        _, reward, _, next_obs = self.env.reset()

                    obs = next_obs
                    self.overall_step += 1
                    continue

                next_state = self.obs_to_state(next_obs)
                self.replay_buffer.add(state=state, action=action, next_state=next_state)

                if self.is_update_step(self.overall_step) and self.overall_step >= self.random_steps:
                    model_losses = []

                    for model in self.agent.models:
                        for _ in range(self.steps_per_update):
                            samples = self.replay_buffer.sample(self.batch_size)
                            states, actions, next_states = [samples[i] for i in self.env_dict]
                            model_loss = model.update(states, actions, next_states)

                        model_losses.append(model_loss)

                    self.data_dict.update({f'agent/model_loss_{i}': loss for i, loss in enumerate(model_losses)})
                    self.data_dict.update({f'agent/model_min_std_{i}': model.min_logvar.exp().sqrt().mean() for i, model in enumerate(self.agent.models)})
                    self.data_dict.update({f'agent/model_max_std_{i}': model.max_logvar.exp().sqrt().mean() for i, model in enumerate(self.agent.models)})
                    wandb.log(self.data_dict, step=self.overall_step)
                    self.data_dict = {}

                obs = next_obs
                self.overall_step += 1

            tqdm.write(f'\nFinal Reward: {reward}')
            tqdm.write(f'Replay Buffer Size: {self.replay_buffer.get_stored_size()}')
            self.all_rewards.append(episode_rewards)

            if episode % self.log_episode_freq == 0:
                self.log_episode(episode_rewards, episode_images, episode, reward_desc='-object_dist_to_goal')

            if self.overall_step >= self.random_steps:
                if self.save_agent:
                    self.agent.dump()

            if self.save_buffer and episode % self.buffer_save_episode_freq == 0:
                np.save(self.buffer_save_path, self.replay_buffer.get_all_transitions())

    def log_step(self, reward, action, episode, *args):
        object_pos = self.env.task._object.get_pose(self.env._physics)[0]
        robots_pos = np.array([robot.get_pose(self.env._physics)[0] for robot in self.env.task._robots])

        object_dist_to_goal = np.linalg.norm(object_pos[:2])
        robots_dist_to_object = np.linalg.norm((object_pos - robots_pos)[:, :2], axis=-1)

        robots_dist_to_object_dict = {f'env/robot{i} dist to object': dist for i, dist in enumerate(robots_dist_to_object)}

        self.data_dict.update({
            'env/object dist to goal': object_dist_to_goal,
            **robots_dist_to_object_dict
        })

        super().log_step(reward, action, episode, *args)


class ExperimentMultiprocess(ExperimentMultiprocessBase):
    def __init__(self):
        super().__init__(experiment_cls=Experiment)

        n_entities = self.n_robots + self.use_object
        self.params.env_dict = {
            'state': {'shape': 3*n_entities},
            'action': {'shape': 2*self.n_robots},
            'next_state': {'shape': 3*n_entities},
        }

        self.replay_buffer = MPReplayBuffer(size=self.buffer_capacity, env_dict=self.params.env_dict)

        if self.load_buffer:
            buffer_dict = np.load(self.buffer_restore_path, allow_pickle=True).item()
            self.replay_buffer.add(**buffer_dict)


if __name__ == '__main__':
    experiment = ExperimentMultiprocess()
    experiment.run()

# python sim_experiment_multi_robot_object.py -buffer_capacity=1000000 -n_episodes=40000 -episode_length=50 -batch_size=250 -lr=3e-4 -ensemble_size=3 -n_particles=12 -discount_factor=0.98 -mpc_method=mppi -mpc_horizon=7 -mpc_samples=200 -mppi_noise_std=0.9 -mppi_beta=1. -mppi_gamma=2. -cem_refine_iters=3 -cem_n_best=20 -cem_alpha=0.8 -hidden_dim=500 -hidden_depth=2 -update_freq=50 -steps_per_update=10 -train_epochs=50 -dist -scale -robot_ids 0 1 -use_object -random_steps=1000 -cost_distance_weight=250. -cost_separation_weight=50. -cost_std_weight=-0.2 -exp_name=testbig -save_buffer -save_agent -pretrain_agent -cost_object_delta_weight=0. -cost_goal_angle_weight=50. -log_episode_freq=5 -commit_episode_freq=1
