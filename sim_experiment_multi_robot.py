import argparse
import numpy as np
from tqdm import tqdm, trange
import cv2

from task import create_sim_environment
from replay_buffer import ReplayBuffer
from agents import RandomShootingAgent, CEMAgent, MPPIAgent
from utils import SetParamsAsAttributes
from train_utils import train_from_buffer

np.set_printoptions(precision=4, suppress=True)


class Experiment(SetParamsAsAttributes):
    def __init__(self, **params):
        super().__init__(params)

        self.env = create_sim_environment(n_robots=self.n_robots, use_object=self.use_object)
        self.replay_buffer = ReplayBuffer(params)

        if self.mpc_method == 'shooting':
            self.agent = RandomShootingAgent(params)
        elif self.mpc_method == 'cem':
            self.agent = CEMAgent(params)
        elif self.mpc_method == 'mppi':
            self.agent = MPPIAgent(params)
        else:
            raise NotImplementedError

        self.goals = np.zeros((self.mpc_horizon, 3))
        self.all_rewards = []

        # action_range = np.linspace(-1, 1, np.floor(np.sqrt(self.random_steps)).astype("int"))
        # left_actions, right_actions = np.meshgrid(action_range, action_range)
        # self.fixed_actions = np.stack((left_actions, right_actions)).transpose(2, 1, 0).reshape(-1, 2)

    def obs_to_state(self, obs):
        return obs['robot_states'].reshape(3*self.n_robots)

    def run(self):
        for episode in trange(self.n_episodes, desc='Episode', position=0, leave=True):
            _, reward, _, obs = self.env.reset()
            episode_rewards = [reward]
            episode_images = []

            for step in trange(self.episode_length, desc='Step', position=1, leave=True):
                if episode * self.episode_length + step == self.random_steps and self.pretrain_agent:
                    train_from_buffer(self.agent, self.replay_buffer, pretrain_samples=self.random_steps)

                state = self.obs_to_state(obs)

                if episode * self.episode_length + step >= self.random_steps:
                    action, next_state_prediction = self.agent.get_action(state, self.goals)
                    # tqdm.write(f'ACTION: {action}')
                else:
                    action = np.random.uniform(-1., 1., size=2*self.n_robots)
                    # action = self.fixed_actions[step]

                _, reward, _, next_obs = self.env.step(action)

                next_state = self.obs_to_state(next_obs)
                self.replay_buffer.add(state, action, next_state)

                if episode * self.episode_length + step >= self.random_steps and self.update_online:
                    for _ in range(self.utd_ratio):
                        for model in self.agent.models:
                            model.update(*self.replay_buffer.sample(self.batch_size))

                # tqdm.write(f'Reward: {reward}')
                episode_rewards.append(reward)
                episode_images.append(self.env.physics.render(camera_id=2))
                obs = next_obs

            tqdm.write(f'Final Reward: {reward}')
            self.all_rewards.append(episode_rewards)
            log_video(episode_images, f'/Users/Obsidian/Desktop/dm_control/videos/multi_episode{episode}.mp4')

def log_video(imgs, filepath, fps=7):
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for img in tqdm(imgs, desc="Saving Video"):
        video.write(img)

    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-random_steps', type=int, default=100)
    parser.add_argument('-n_particles', type=int, default=20)

    parser.add_argument('-mpc_method', type=str, default='mppi')

    parser.add_argument('-alpha', type=float, default=0.8)
    parser.add_argument('-n_best', type=int, default=30)
    parser.add_argument('-refine_iters', type=int, default=5)

    parser.add_argument('-gamma', type=float, default=50.)
    parser.add_argument('-beta', type=float, default=0.5)
    parser.add_argument('-noise_std', type=float, default=2.)

    # generic
    parser.add_argument('-robot_ids', nargs='+', type=int, default=[0])
    parser.add_argument('-object_id', type=int, default=3)
    parser.add_argument('-use_object', action='store_true')
    parser.add_argument('-exp_name', type=str, default=None)

    parser.add_argument('-n_episodes', type=int, default=3)
    parser.add_argument('-tolerance', type=float, default=0.04)
    parser.add_argument('-episode_length', type=int, default=150)

    parser.add_argument('-meta', action='store_true')
    parser.add_argument('-pretrain_samples', type=int, default=500)
    parser.add_argument('-train_epochs', type=int, default=200)
    parser.add_argument('-validation', action='store_true')
    parser.add_argument('-training_plot', action='store_true')

    parser.add_argument('-save_agent', action='store_true')
    parser.add_argument('-load_agent', action='store_true')

    # agent
    parser.add_argument('-ensemble_size', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=10000)
    parser.add_argument('-update_online', action='store_true')
    parser.add_argument('-sample_recent_ratio', type=float, default=0.)
    parser.add_argument('-utd_ratio', type=int, default=3)
    parser.add_argument('-discount_factor', type=float, default=0.9)

    parser.add_argument('-mpc_horizon', type=int, default=5)
    parser.add_argument('-mpc_samples', type=int, default=200)
    parser.add_argument('-robot_goals', action='store_true')
    parser.add_argument('-close_radius', type=float, default=0.4)
    parser.add_argument('-pretrain_agent', action='store_true')

    # model
    parser.add_argument('-hidden_dim', type=int, default=200)
    parser.add_argument('-hidden_depth', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)

    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-dist', action='store_true')

    # replay buffer
    parser.add_argument('-save_freq', type=int, default=50) # TODO implement this
    parser.add_argument('-buffer_capacity', type=int, default=10000)
    parser.add_argument('-buffer_save_dir', type=str, default='~/kamigami_data/replay_buffers/online_buffers/')
    parser.add_argument('-buffer_restore_dir', type=str, default='~/kamigami_data/replay_buffers/meshgrid_buffers/')

    parser.add_argument('-cost_distance_weight', type=float, default=1.)
    parser.add_argument('-cost_distance_bonus_weight', type=float, default=0.)
    parser.add_argument('-cost_distance_penalty_weight', type=float, default=0.)
    parser.add_argument('-cost_big_distance_penalty_weight', type=float, default=0.)
    parser.add_argument('-cost_separation_weight', type=float, default=0.)
    parser.add_argument('-cost_std_weight', type=float, default=0.)
    parser.add_argument('-cost_goal_angle_weight', type=float, default=0.)
    parser.add_argument('-cost_realistic_weight', type=float, default=0.)
    parser.add_argument('-cost_action_weight', type=float, default=0.)
    parser.add_argument('-cost_to_object_heading_weight', type=float, default=0.)
    parser.add_argument('-cost_object_delta_weight', type=float, default=0.)

    args = parser.parse_args()
    args.n_robots = len(args.robot_ids)

    experiment = Experiment(**vars(args))
    experiment.run()
