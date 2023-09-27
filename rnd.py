import torch
from torch import nn
from torch.nn import functional as F

from utils import DataUtils, SetParamsAsAttributes, as_tensor, dcn, initialize_weights


STATE_DIM = 4
ACTION_DIM = 2


class RND(SetParamsAsAttributes, nn.Module):
    def __init__(self, params):
        super(RND, self).__init__(params)

        self.dtu = DataUtils(params)

        input_dim = STATE_DIM * (self.n_robots + self.use_object)# + ACTION_DIM * self.n_robots
        output_dim = 512

        assert self.hidden_depth >= 1

        def build_network():
            hidden_layers = []
            for _ in range(self.hidden_depth):
                hidden_layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
            input_layer = [nn.Linear(input_dim, self.hidden_dim), nn.ReLU()]
            output_layer = [nn.Linear(self.hidden_dim, output_dim)]

            layers = input_layer + hidden_layers + output_layer
            net = nn.Sequential(*layers)
            net.apply(initialize_weights)

            return net

        self.target_net, self.predictor_net = build_network(), build_network()
        self.predictor_optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=self.lr)

        self.input_mean = 0.
        self.input_std = 1.

    def forward(self, states, actions, goals, use_predictor=False):
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
        # inputs = torch.cat([input_states, actions, input_goals], dim=1).float()
        inputs = torch.cat([input_states, input_goals], dim=1).float()

        if self.scale:
            inputs = self.standardize_input(inputs)

        if use_predictor:
            outputs = self.predictor_net(inputs)
        else:
            outputs = self.target_net(inputs)

        return outputs

    def update(self, state, action, goal):
        self.train()
        state, action, goal = as_tensor(state, action, goal)

        predictor_output = self(state, action, goal, use_predictor=True)

        with torch.no_grad():
            target_output = self(state, action, goal, use_predictor=False)

        loss = F.mse_loss(predictor_output, target_output, reduction='mean')

        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()

        return dcn(loss)

    def get_score(self, state, action, goal):
        self.eval()
        state, action, goal = as_tensor(state, action, goal)

        with torch.no_grad():
            predictor_output = self(state, action, goal, use_predictor=True)
            target_output = self(state, action, goal, use_predictor=False)

        score = F.mse_loss(predictor_output, target_output, reduction='none').mean(dim=-1)
        return score

    def set_scalers(self, state, action, goal):
        state, action, goal = as_tensor(state, action, goal)
        input_state = self.dtu.state_to_model_input(state)
        input_goal = self.dtu.relative_goal_xysc(state, goal)

        # inp = torch.cat([input_state, action, input_goal], axis=1)
        inp = torch.cat([input_state, input_goal], axis=1)

        self.input_mean = inp.mean(dim=0)
        self.input_std = inp.std(dim=0)

    def standardize_input(self, model_input):
        return (model_input - self.input_mean) / self.input_std
