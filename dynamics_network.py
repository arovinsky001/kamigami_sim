import torch
from torch import nn
from torch.nn import functional as F

from utils import DataUtils, SetParamsAsAttributes, as_tensor, dcn, create_initialize_network


STATE_DIM = 4
ACTION_DIM = 2


class DynamicsNetwork(SetParamsAsAttributes, nn.Module):
    def __init__(self, params):
        super().__init__(params)

        self.dtu = DataUtils(params)
        self.update_lr = nn.parameter.Parameter(torch.tensor(self.lr))

        input_dim = ACTION_DIM * self.n_robots + STATE_DIM * (self.n_robots + self.use_object - 1)
        output_dim = STATE_DIM * (self.n_robots + self.use_object)
        if self.dist:
            self.max_logvar = nn.parameter.Parameter(torch.ones(output_dim) / 2.)
            self.min_logvar = nn.parameter.Parameter(torch.ones(output_dim) * -10.)
            output_dim *= 2

        self.net = create_initialize_network(input_dim, output_dim, self.hidden_depth, self.hidden_dim)

        if self.dist:
            self.optimizer = torch.optim.Adam([*self.net.parameters(), self.max_logvar, self.min_logvar], lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.lr_optimizer = torch.optim.Adam([self.update_lr], lr=self.lr)

        self.input_mean = 0
        self.input_std = 1

        self.output_mean = 0
        self.output_std = 1

    def forward(self, states, actions, sample=False, ret_dist=False, delta=False, params=None, ret_dist_params=False, ret_logvar=False):
        states, actions = as_tensor(states, actions)

        if len(states.shape) == 1:
            states = states[None, :]
        if len(actions.shape) == 1:
            actions = actions[None, :]

        # state comes in as (x, y, theta)
        input_states = self.dtu.state_to_model_input(states)
        if input_states is None:
            inputs = actions.float()
        else:
            inputs = torch.cat([input_states, actions], dim=1).float()

        if self.scale:
            inputs = self.standardize_input(inputs)

        if params is not None:
            # metalearning forward pass
            x = inputs
            for i in range(self.hidden_depth + 1):
                w, b = params[2*i], params[2*i+1]
                x = F.relu(F.linear(x, w, b))

            w, b = params[-2], params[-1]
            outputs = F.linear(x, w, b)
        else:
            # standard forward pass
            outputs = self.net(inputs)

        if self.dist:
            mean, logvar = torch.chunk(outputs, 2, dim=1)
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            std = logvar.exp().sqrt()

            if ret_dist_params:
                if ret_logvar:
                    return mean, logvar
                else:
                    return mean, std

            dist = torch.distributions.normal.Normal(mean, std)
            if ret_dist:
                return dist

            state_deltas_model = dist.rsample() if sample else mean
        else:
            state_deltas_model = outputs

        if delta:
            return state_deltas_model

        if self.scale:
            state_deltas_model = self.unstandardize_output(state_deltas_model)

        next_states_model = self.dtu.next_state_from_relative_delta(states, state_deltas_model)
        return next_states_model

    def update(self, state, action, next_state):
        self.train()
        state, action, next_state = as_tensor(state, action, next_state)
        state_delta = self.dtu.compute_relative_delta_xysc(state, next_state)

        if self.scale:
            state_delta = self.standardize_output(state_delta)

        if self.dist:
            dist = self(state, action, ret_dist=True)
            loss = -dist.log_prob(state_delta.detach()).mean()
            loss += 0.01 * (self.max_logvar.mean() - self.min_logvar.mean())
        else:
            pred_state_delta = self(state, action, delta=True)
            loss = F.mse_loss(pred_state_delta, state_delta, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return dcn(loss)

    def update_meta(self, state, action, next_state):
        self.train()
        state, action, next_state = as_tensor(state, action, next_state)
        state_delta = self.dtu.compute_relative_delta_xysc(state, next_state)

        # chunk into "tasks" i.e. batches continuous in time
        tasks = 10
        task_steps = 20
        meta_update_steps = 2

        first_idxs = torch.randint(len(state) - 2 * task_steps + 1, size=(tasks,))
        tiled_first_idxs = first_idxs.reshape(-1, 1).repeat(1, task_steps)
        all_idxs = tiled_first_idxs + torch.arange(task_steps)

        batched_states = state[all_idxs]
        batched_actions = action[all_idxs]
        batched_state_deltas = state_delta[all_idxs]

        test_loss_sum = 0

        if self.dist and self.scale:
            batched_state_deltas = self.standardize_output(batched_state_deltas)

        for i in range(tasks):
            train_states, test_states = batched_states[i].chunk(2, dim=0)
            train_actions, test_actions = batched_actions[i].chunk(2, dim=0)
            train_state_deltas, test_state_deltas = batched_state_deltas[i].chunk(2, dim=0)

            params = list(self.net.parameters())

            for _ in range(meta_update_steps):
                # compute prediction and corresponding training loss
                if self.dist:
                    dist = self(train_states, train_actions, ret_dist=True, params=params)
                    loss = -dist.log_prob(train_state_deltas).mean()
                else:
                    pred_state_deltas = self(train_states, train_actions, delta=True, params=params)
                    loss = F.mse_loss(pred_state_deltas, train_state_deltas, reduction='mean')

                # compute gradient and simulate gradient step
                grad = torch.autograd.grad(loss, params)
                params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))

            # only consider last-step losses for main network update
            if self.dist:
                test_dist = self(test_states, test_actions, ret_dist=True, params=params)
                test_loss_sum -= test_dist.log_prob(test_state_deltas).mean()
            else:
                pred_state_deltas = self(test_states, test_actions, delta=True, params=params)
                test_loss_sum += F.mse_loss(pred_state_deltas, test_state_deltas)

        loss = test_loss_sum / tasks

        # update network
        self.optimizer.zero_grad()
        # loss.backward()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # update learning rate
        self.lr_optimizer.zero_grad()
        loss.backward()
        self.lr_optimizer.step()

        return dcn(test_loss_sum)

    def set_scalers(self, state, action, next_state):
        state, action, next_state = as_tensor(state, action, next_state)
        input_state = self.dtu.state_to_model_input(state)

        if input_state is None:
            inp = action
        else:
            inp = torch.cat([input_state, action], axis=1)
        state_delta = self.dtu.compute_relative_delta_xysc(state, next_state)

        self.input_mean = inp.mean(dim=0)
        self.input_std = inp.std(dim=0)

        self.output_mean = state_delta.mean(dim=0)
        self.output_std = state_delta.std(dim=0)

        n_entities = self.n_robots + self.use_object

        if n_entities != 1:
            sin_cos_idx = torch.arange(STATE_DIM*(n_entities-1))
            sin_cos_idx = sin_cos_idx.reshape((n_entities-1, STATE_DIM))[:, -2:].reshape((n_entities-1)*2)
            # sin_cos_idx = torch.arange(STATE_DIM*n_entities)
            # sin_cos_idx = sin_cos_idx.reshape((n_entities, STATE_DIM))[:, -2:].reshape(n_entities*2)
            self.input_mean[sin_cos_idx] = 0.
            self.input_std[sin_cos_idx] = 1.

        sin_cos_idx = torch.arange(STATE_DIM*n_entities)
        sin_cos_idx = sin_cos_idx.reshape((n_entities, STATE_DIM))[:, -2:].reshape(n_entities*2)
        self.output_mean[sin_cos_idx] = 0.
        self.output_std[sin_cos_idx] = 1.

    def standardize_input(self, model_input):
        return (model_input - self.input_mean) / self.input_std

    def standardize_output(self, model_output):
        return (model_output - self.output_mean) / self.output_std

    def unstandardize_output(self, model_output):
        return model_output * self.output_std + self.output_mean
