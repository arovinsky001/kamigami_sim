import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd
import wandb

from utils import DataUtils, SetParamsAsAttributes, create_initialize_network, soft_update_params, as_tensor, dcn
from rnd import RND


STATE_DIM = 4
ACTION_DIM = 2


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class Actor(SetParamsAsAttributes, nn.Module):
    def __init__(self, params):
        super().__init__(params)
        self.dtu = DataUtils(params)

        input_dim = STATE_DIM * (self.n_robots + self.use_object)
        output_dim = ACTION_DIM * self.n_robots

        self.net = create_initialize_network(input_dim, output_dim*2, self.hidden_depth, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.log_alpha = nn.parameter.Parameter(torch.log(torch.tensor(self.init_temperature)))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        if self.target_entropy is None:
            self.target_entropy = -ACTION_DIM * self.n_robots / 2.

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, states, goals, sample=False, ret_dist=False, ret_dist_params=False, ret_logvar=False):
        states = as_tensor(states)

        if len(states.shape) == 1:
            states = states[None, :]
        if len(goals.shape) == 1:
            goals = goals[None, :]

        # state comes in as (x, y, theta)
        input_states = self.dtu.state_to_model_input(states)
        input_goals = self.dtu.relative_goal_xysc(states, goals)

        if input_states is None:
            inputs = input_goals.float()
        else:
            inputs = torch.cat([input_states, input_goals], dim=1).float()

        # standard forward pass
        outputs = self.net(inputs)

        mean, logvar = torch.chunk(outputs, 2, dim=1)
        std = logvar.exp().sqrt()

        if ret_dist_params:
            if ret_logvar:
                return mean, logvar
            else:
                return mean, std

        dist = SquashedNormal(mean, std)
        if ret_dist:
            return dist

        pred_actions = dist.rsample() if sample else mean

        return pred_actions


class Critic(SetParamsAsAttributes, nn.Module):
    def __init__(self, params):
        super().__init__(params)
        self.dtu = DataUtils(params)

        input_dim = STATE_DIM * (self.n_robots + self.use_object) + ACTION_DIM * self.n_robots
        output_dim = 1

        self.Qs = [create_initialize_network(input_dim, output_dim, self.hidden_depth, self.hidden_dim, layernorm=self.layernorm) for _ in range(self.n_critics)]
        self.target_Qs = [copy.deepcopy(Q) for Q in self.Qs]

        self.optimizer = torch.optim.Adam([param for Q in self.Qs for param in Q.parameters()], lr=self.lr)

    def forward(self, states, actions, goals, target=False):
        states, actions, goals = as_tensor(states, actions, goals)

        if len(states.shape) == 1:
            states = states[None, :]
        if len(actions.shape) == 1:
            actions = actions[None, :]
        if len(goals.shape) == 1:
            goals = goals[None, :]

        # state comes in as (x, y, theta)
        input_states = self.dtu.state_to_model_input(states)
        input_goals = self.dtu.relative_goal_xysc(states, goals)

        if input_states is None:
            inputs = torch.cat([actions, input_goals], dim=1).float()
        else:
            inputs = torch.cat([input_states, actions, input_goals], dim=1).float()

        # standard forward pass
        if target:
            subset_idx = torch.multinomial(torch.ones(self.n_critics) / self.n_critics, num_samples=2, replacement=False)
        else:
            subset_idx = range(self.n_critics)

        pred_qs = []
        qfs = self.target_Qs if target else self.Qs
        for idx in subset_idx:
            pred_qs.append(qfs[idx](inputs))

        return pred_qs


class Agent(SetParamsAsAttributes, nn.Module):
    def __init__(self, params):
        super().__init__(params)
        self.dtu = DataUtils(params)

        self.actor = Actor(params)
        self.critic = Critic(params)

        if self.use_rnd:
            self.rnd = RND(params)

        self.state_dict_save_dir = os.path.join(os.path.dirname(__file__), 'data', self.project_name, self.exp_name, 'agents')
        self.save_path = os.path.join(self.state_dict_save_dir, f"robot_{'_'.join([str(id) for id in self.robot_ids])}_object_{self.use_object}_state_dict.pt")

    def update(self, replay_buffer, step, first_update=True):
        self.train(True)
        samples = replay_buffer.sample(self.batch_size)
        states, goals, next_states, next_goals, actions, rewards = as_tensor(*[samples[i] for i in self.env_dict])

        log_dict = {}

        if self.use_rnd and first_update:
            rnd_loss = self.rnd.update(states, actions, goals)
            log_dict.update({'agent/rnd loss': rnd_loss})

        critic_loss, mean_q = self.update_critic(states, goals, next_states, next_goals, actions, rewards)
        log_dict.update({
            'agent/critic loss': critic_loss,
            'agent/mean_q': mean_q,
        })

        # if step % self.critic_target_update_freq == 0 and first_update:
        for Q, target_Q in zip(self.critic.Qs, self.critic.target_Qs):
            soft_update_params(Q, target_Q, self.critic_target_tau)

        if step % self.actor_update_freq == 0 and first_update:
        # if step % self.actor_update_freq == 0:
            actor_loss, alpha_loss, alpha, entropy = self.update_actor_and_alpha(states, goals)
            log_dict.update({
                'agent/actor loss': actor_loss,
                'agent/alpha loss': alpha_loss,
                'agent/alpha': alpha,
                'agent/entropy': entropy,
            })

        wandb.log(log_dict, step=step, commit=False)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def get_action(self, state, goal, sample=False):
        self.train(False)
        with torch.no_grad():
            action = self.actor(state, goal, sample=sample)
            action = action.squeeze().clamp(-1., 1.)
            return dcn(action)

    def update_critic(self, states, goals, next_states, next_goals, actions, rewards):
        with torch.no_grad():
            if self.use_rnd:
                rewards += self.rnd_bonus_weight * self.rnd.get_score(states, actions, goals)[:, None]

            dist = self.actor(next_states, next_goals, ret_dist=True)

            next_actions = dist.rsample()
            log_prob = dist.log_prob(next_actions).sum(dim=-1, keepdim=True)

            target_next_qs = self.critic(next_states, next_actions, next_goals, target=True)
            min_q = torch.stack(target_next_qs).min(dim=0)[0]
            target_v = min_q - self.actor.alpha.detach() * log_prob
            target_q = rewards + self.discount_factor * target_v

        # get current Q estimates
        current_qs = self.critic(states, actions, goals, target=False)

        critic_loss = 0.
        for q in current_qs:
            critic_loss += F.mse_loss(q, target_q)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss, torch.cat(current_qs).mean()

    def update_actor_and_alpha(self, states, goals):
        dist = self.actor(states, goals, ret_dist=True)

        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1)

        actor_qs = self.critic(states, actions, goals)
        actor_q = torch.stack(actor_qs).mean(dim=0)

        actor_loss = (self.actor.alpha.detach() * log_prob - actor_q).mean()

        # optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        alpha_loss = (self.actor.alpha * (-log_prob - self.actor.target_entropy).detach()).mean()
        alpha = self.actor.alpha.clone()

        self.actor.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.actor.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, alpha, -log_prob.mean()

    def dump(self):
        if not os.path.exists(self.state_dict_save_dir):
            os.makedirs(self.state_dict_save_dir)

        torch.save(self.state_dict(), self.save_path)

    def restore(self, restore_dir=None, recency=1):
        self.load_state_dict(torch.load(self.save_path))
