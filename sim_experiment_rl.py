import numpy as np
import torch
from tqdm import tqdm, trange
from cpprb import MPReplayBuffer

from sim_experiment_base import ExperimentMultiprocessBase
from sim_experiment_multi_robot_object import Experiment as ExperimentMultiRobotObject
from rl_agent import Agent
from utils import dcn

np.set_printoptions(precision=4, suppress=True)


class Experiment(ExperimentMultiRobotObject):
    def __init__(self, params=None):
        super().__init__(params=params)

        self.agent = Agent(self.params)

    def run(self):
        for episode in trange(self.n_episodes, desc='Episode', position=0, leave=True):
            _, _, _, obs = self.env.reset()
            while not obs['is_healthy']:
                _, _, _, obs = self.env.reset()

            episode_rewards = []
            episode_images = []

            for step in trange(self.episode_length, desc='Step', position=1, leave=True):
                state = self.obs_to_state(obs)

                if self.overall_step >= self.random_steps:
                    sample = not self.is_eval_episode(episode)
                    action = self.agent.get_action(state, self.goals[0], sample=sample)
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
                    self.log_step(reward, action, episode, state)

                if episode % self.log_episode_freq == 0:
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
                self.replay_buffer.add(state=state, next_state=next_state, goal=self.goals[0], next_goal=self.goals[0], action=action, reward=reward)

                if self.is_update_step(self.overall_step) and self.overall_step >= self.random_steps:
                    for i in range(self.steps_per_update):
                        self.agent.update(self.replay_buffer, self.overall_step, first_update=(i==0))

                obs = next_obs
                self.overall_step += 1

            tqdm.write(f'\nFinal Reward: {reward}')
            tqdm.write(f'Replay Buffer Size: {self.replay_buffer.get_stored_size()}')
            self.all_rewards.append(episode_rewards)

            if episode % self.log_episode_freq == 0:
                self.log_episode(episode_rewards, episode_images, episode, reward_desc='-object_dist_to_goal-0.5*robots_dist_to_object')

            if self.overall_step >= self.random_steps:
                if self.save_agent:
                    self.agent.dump()

            if self.save_buffer and episode % self.buffer_save_episode_freq == 0:
                np.save(self.buffer_save_path, self.replay_buffer.get_all_transitions())

    def log_step(self, reward, action, episode, state, *args):
        step_qs = self.agent.critic(state, action, self.goals[0], target=False)
        step_q = dcn(torch.cat(step_qs).mean())

        self.data_dict.update({
            'agent/step_q': step_q,
            'replay_buffer/size': self.replay_buffer.get_stored_size(),
        })

        if self.use_rnd:
            rnd_bonus = self.agent.rnd.get_score(state, action, self.goals[0]).squeeze()

            self.data_dict.update({
                'agent/rnd_bonus': rnd_bonus,
            })

        super().log_step(reward, action, episode, state, *args)


class ExperimentMultiprocess(ExperimentMultiprocessBase):
    def __init__(self):
        super().__init__(Experiment)

        n_entities = self.n_robots + self.use_object
        self.params.env_dict = {
            'state': {'shape': 3*n_entities},
            'goal': {'shape': 3},
            'next_state': {'shape': 3*n_entities},
            'next_goal': {'shape': 3},
            'action': {'shape': 2*self.n_robots},
            'reward': {'shape': 1},
        }

        self.replay_buffer = MPReplayBuffer(size=self.buffer_capacity, env_dict=self.params.env_dict)


if __name__ == '__main__':
    experiment = ExperimentMultiprocess()
    experiment.run()

# sim_experiment_rl.py -buffer_capacity=2000000 -n_episodes=100000 -episode_length=50 -batch_size=256 -lr=3e-4 -discount_factor=0.97 -hidden_dim=256 -hidden_depth=2 -update_freq=1 -steps_per_update=20 -actor_update_freq=1 -eval_episode_freq=10 -critic_target_update_freq=2 -critic_target_tau=0.01 -init_temperature=0.1 -robot_ids 0 1 -use_object -random_steps=1000 -project_name=big_mp_rl_online -exp_name=batch256_ac1_crit20_rnd0_temp01_lr3e4_proc3 -save_agent -save_buffer -log_episode_freq=1 -log_video_freq=2 -log_step_freq=5 -commit_episode_freq=1 -buffer_save_episode_freq=5 -n_proc=3 -use_rnd -rnd_bonus_weight=0
