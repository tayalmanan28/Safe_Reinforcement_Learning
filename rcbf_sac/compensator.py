import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from rcbf_sac.utils import scale_action, to_tensor, to_numpy

criterion = nn.MSELoss()


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class CompensatorModel(nn.Module):

    def __init__(self, state_dim, action_dim, hidden1=30, hidden2=20, init_w=3e-3):
        super(CompensatorModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = state
        out = self.fc1(state)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        action = torch.tanh(self.fc3(out))  # TODO:Not sure if they have the tanh here in Cheng's paper
        return action


class Compensator:

    def __init__(self, state_dim, action_dim, action_lb, action_ub, args):

        if args.seed > 0:
            self.seed(args.seed)

        self.state_dim = state_dim
        self.action_dim = action_dim

        # # Create Actor and Critic Network
        # net_cfg = {
        #     'hidden1': args.hidden1,
        #     'hidden2': args.hidden2,
        #     'init_w': args.init_w
        # }
        # self.actor = CompensatorModel(state_dim, action_dim, **net_cfg)
        self.comp_actor = CompensatorModel(state_dim, action_dim)
        self.comp_actor_optim = Adam(self.comp_actor.parameters(), lr=args.comp_rate)

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.comp_actor.to(self.device)

        # Action Lower and Upper Bounds
        self.action_lb = to_tensor(action_lb, torch.FloatTensor, self.device)
        self.action_ub = to_tensor(action_ub, torch.FloatTensor, self.device)

        # If its never been trained then we don't want to use it (useful for eval only)
        self.is_trained = False


    def __call__(self, observation):
        action = self.comp_actor(observation) * self.is_trained
        return scale_action(action, self.action_lb, self.action_ub)

    def train(self, rollouts, epochs=1):
        """

        Parameters
        ----------
        rollout : list
            List of dicts with each dict containing the keys ['obs'], ['u_nom'],
            ['u_safe'], ['u_comp']. Each of those keys has a ndarray value where the rows are steps.
            Note here state refers to observation not internal state of the dynamics.

        Returns
        -------

        """

        self.is_trained = True  # only want to use it if its been trained

        for epoch in range(epochs):

            # Compensator Actor update
            self.comp_actor.zero_grad()

            # Stack everything appropriately
            all_obs = np.zeros((0, self.state_dim))
            all_u_comp = np.zeros((0, self.action_dim))
            all_u_safe = np.zeros((0, self.action_dim))

            for rollout in rollouts:
                all_obs = np.vstack((all_obs, rollout['obs']))
                all_u_comp = np.vstack((all_u_comp, rollout['u_comp']))
                all_u_safe = np.vstack((all_u_safe, rollout['u_safe']))

            all_obs = to_tensor(all_obs, torch.FloatTensor, self.device)
            all_u_comp = to_tensor(all_u_comp, torch.FloatTensor, self.device)
            all_u_safe = to_tensor(all_u_safe, torch.FloatTensor, self.device)
            target = scale_action(all_u_comp + all_u_safe, self.action_lb, self.action_ub)
            comp_actor_loss = criterion(self.comp_actor(all_obs), target)

            comp_actor_loss.backward()
            self.comp_actor_optim.step()

    def load_weights(self, output):
        if output is None: return

        self.comp_actor.load_state_dict(
            torch.load('{}/comp_actor.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.comp_actor.state_dict(),
            '{}/comp_actor.pkl'.format(output)
        )

    def seed(self, s):

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)










