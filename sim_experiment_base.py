import os
import abc
import argparse
import copy
import numpy as np
import warnings
from multiprocessing import Process
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import wandb
from cpprb import MPReplayBuffer

from task import create_sim_environment
from replay_buffer import ReplayBuffer
from agents import RandomShootingAgent, CEMAgent, MPPIAgent
from utils import SetParamsAsAttributes
from train_utils import train_from_buffer

np.set_printoptions(precision=4, suppress=True)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-random_steps', type=int, default=100)
    parser.add_argument('-n_particles', type=int, default=20)

    parser.add_argument('-mpc_method', type=str, default='mppi')

    parser.add_argument('-cem_alpha', type=float, default=0.8)
    parser.add_argument('-cem_n_best', type=int, default=30)
    parser.add_argument('-cem_refine_iters', type=int, default=5)

    parser.add_argument('-mppi_gamma', type=float, default=50.)
    parser.add_argument('-mppi_beta', type=float, default=0.5)
    parser.add_argument('-mppi_noise_std', type=float, default=2.)

    # generic
    parser.add_argument('-robot_ids', nargs='+', type=int, default=[0])
    parser.add_argument('-object_id', type=int, default=3)
    parser.add_argument('-use_object', action='store_true')
    parser.add_argument('-exp_name', type=str, default=None)
    parser.add_argument('-project_name', type=str, default=None)

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
    parser.add_argument('-log_locally', action='store_true')
    parser.add_argument('-offline', action='store_true')
    parser.add_argument('-log_step_freq', type=int, default=10)
    parser.add_argument('-log_episode_freq', type=int, default=10)
    parser.add_argument('-log_video_freq', type=int, default=10)
    parser.add_argument('-commit_episode_freq', type=int, default=1)
    parser.add_argument('-eval_episode_freq', type=int, default=10)
    parser.add_argument('-n_proc', type=int, default=1)

    # agent
    parser.add_argument('-ensemble_size', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=10000)
    parser.add_argument('-sample_recent_ratio', type=float, default=0.)
    parser.add_argument('-update_freq', type=int, default=1)
    parser.add_argument('-steps_per_update', type=int, default=1)
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
    parser.add_argument('-buffer_save_episode_freq', type=int, default=5)
    parser.add_argument('-buffer_capacity', type=int, default=10000)
    parser.add_argument('-save_buffer', action='store_true')
    parser.add_argument('-load_buffer', action='store_true')
    parser.add_argument('-load_buffer_exp_name', type=str, default=None)
    # parser.add_argument('-buffer_save_dir', type=str, default='~/kamigami_data/replay_buffers/online_buffers/')
    # parser.add_argument('-buffer_restore_dir', type=str, default='~/kamigami_data/replay_buffers/meshgrid_buffers/')

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

    # rl
    parser.add_argument('-actor_update_freq', type=int, default=1)
    parser.add_argument('-critic_target_update_freq', type=int, default=1)
    parser.add_argument('-critic_target_tau', type=float, default=1.)
    parser.add_argument('-target_entropy', type=float, default=None)
    parser.add_argument('-init_temperature', type=float, default=0.1)
    parser.add_argument('-rnd_bonus_weight', type=float, default=1.)
    parser.add_argument('-goal_clip_dist', type=float, default=1.)
    parser.add_argument('-n_critics', type=int, default=2)

    parser.add_argument('-use_rnd', action='store_true')
    parser.add_argument('-layernorm', action='store_true')

    args = parser.parse_args()
    args.n_robots = len(args.robot_ids)

    buffer_exp_name = args.exp_name if args.load_buffer_exp_name is None else args.load_buffer_exp_name
    args.buffer_save_dir = os.path.join(os.path.dirname(__file__), 'data', args.project_name, args.exp_name, 'replay_buffers')
    args.buffer_restore_dir = os.path.join(os.path.dirname(__file__), 'data', args.project_name, buffer_exp_name, 'replay_buffers')

    if args.save_buffer:
        args.buffer_save_path = os.path.join(args.buffer_save_dir, 'replay_buffer.npy')
        if not os.path.exists(args.buffer_save_dir):
            os.makedirs(args.buffer_save_dir)

    if args.load_buffer:
        args.buffer_restore_path = os.path.join(args.buffer_restore_dir, 'replay_buffer.npy')

    return args


class ExperimentBase(SetParamsAsAttributes):
    def __init__(self, params=None):
        if params is None:
            params = parse_args()

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

        wandb.init(
            # set the wandb project where this run will be logged
            entity='kamigami',
            project=self.project_name,
            name=self.exp_name,

            # track hyperparameters and run metadata
            config=params,
            mode='disabled' if self.offline else 'online',
        )

        self.reward_plot_dir_path = os.path.join(os.path.dirname(__file__), 'data', self.project_name, self.exp_name, 'reward')
        self.video_dir_path = os.path.join(os.path.dirname(__file__), 'data', self.project_name, self.exp_name, 'videos')

        if self.load_buffer:
            self.replay_buffer.restore(restore_path=os.path.join(os.path.dirname(__file__), 'data', self.project_name, self.exp_name, 'replay_buffer.npz'))
            self.random_steps = 0

        if self.load_agent:
            self.agent.restore()
            train_from_buffer(self.agent, self.replay_buffer, set_scale_only=True)

        # single robot fixed actions
        action_range = np.linspace(-1, 1, np.floor(np.sqrt(self.random_steps)).astype("int"))
        left_actions, right_actions = np.meshgrid(action_range, action_range)
        self.fixed_actions = np.stack((left_actions, right_actions)).transpose(2, 1, 0).reshape(-1, 2)

        self.overall_step = 0
        self.data_dict = {}

    def is_update_step(self, step):
        return self.update_freq != 0 and np.isfinite(self.update_freq) and step % self.update_freq == 0

    def is_eval_episode(self, episode):
        return (episode % self.eval_episode_freq == 0)

    @abc.abstractmethod
    def run(self):
        pass

    def log_step(self, reward, action, episode, *args):
        if self.offline:
            return

        self.data_dict.update({'reward': reward})

        action_per_robot = np.split(action, len(action) // 2) if self.n_robots > 1 else action[None, :]
        for i, robot_action in enumerate(action_per_robot):
            self.data_dict.update({f'robot{i}_action_norm': np.linalg.norm(robot_action)})

        if self.is_eval_episode(episode):
            self.data_dict.update({f'eval/{k}': v for k, v in self.data_dict.items()})

        wandb.log(self.data_dict, step=self.overall_step)
        self.data_dict = {}

    def log_episode(self, rewards, images, episode, reward_desc='', fps=10):
        if self.offline:
            return

        if self.log_locally and self.overall_step < self.random_steps:
            return
        elif not self.log_locally:
            self.data_dict.update({
                'reward improvement': rewards[-1] - rewards[0],
                'final reward': rewards[-1],
                'episode': episode,
            })

            if episode % self.log_video_freq == 0:
                self.data_dict.update({'video': wandb.Video(np.array(images).transpose(0, 3, 1, 2), fps=fps, format="gif")})

            if self.is_eval_episode(episode):
                self.data_dict.update({f'eval/{k}': v for k, v in self.data_dict.items()})

            commit = (episode % self.commit_episode_freq == 0)

            wandb.log(self.data_dict, step=self.overall_step-1, commit=commit)
            self.data_dict = {}
            return

        if not os.path.exists(self.reward_plot_dir_path):
            os.makedirs(self.reward_plot_dir_path)

        if not os.path.exists(self.video_dir_path):
            os.makedirs(self.video_dir_path)

        fname = f'{self.n_robots}robot_episode{episode}'

        # log reward
        fig, ax = plt.subplots()
        ax.plot(rewards)

        ax.set_title(f'Episode {episode} Reward')
        ax.set_xlabel('Step')
        ax.set_ylabel(f'Reward ({reward_desc})')
        ax.set_ylim([-self.env.task._object_init_radius * 4., 0])

        fpath = os.path.join(self.reward_plot_dir_path, fname + '.png')
        fig.savefig(fpath)

        plt.close(fig)

        # log video
        height, width = images[0].shape[0], images[0].shape[1]
        fpath = os.path.join(self.video_dir_path, fname + '.mp4')
        video = cv2.VideoWriter(fpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for img in tqdm(images, desc="Saving Video"):
            video.write(img)

        video.release()


def run_single_experiment(params, replay_buffer, experiment_cls, i):
    params = copy.copy(params)

    if params.load_buffer:
        params.load_buffer = False
        params.random_steps = 0
    else:
        params.random_steps = int(params.random_steps / params.n_proc)

    load_agent = params.load_agent
    params.load_agent = False

    if i == 0:
        params.offline = params.original_offline
    else:
        params.offline = True
        params.save_agent = False
        params.save_buffer = False
        params.eval_episode_freq = np.inf

    experiment = experiment_cls(params)
    experiment.replay_buffer = replay_buffer

    if load_agent:
        experiment.agent.restore()
        train_from_buffer(experiment.agent, experiment.replay_buffer, set_scale_only=True)

    experiment.run()


class ExperimentMultiprocessBase(SetParamsAsAttributes):
    def __init__(self, experiment_cls):
        params = parse_args()

        params.original_offline = params.offline
        params.offline = True

        super().__init__(params)
        self.experiment_cls = experiment_cls

        n_entities = self.n_robots + self.use_object
        self.params.env_dict = {
            'state': {'shape': 3*n_entities},
            'next_state': {'shape': 3*n_entities},
            'goal': {'shape': 3},
            'next_goal': {'shape': 3},
            'action': {'shape': 2*self.n_robots},
            'reward': {'shape': 1},
        }

        self.replay_buffer = MPReplayBuffer(size=self.buffer_capacity, env_dict=self.params.env_dict)

    def run(self):
        if self.n_proc == 1:
            run_single_experiment(self.params, self.replay_buffer, self.experiment_cls, 0)
            return

        processes = []

        for i in range(self.n_proc):
            p = Process(target=run_single_experiment, args=(self.params, self.replay_buffer, self.experiment_cls, i))
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()