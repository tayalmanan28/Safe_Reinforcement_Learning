import torch
import torch.nn.functional as F
from torch.optim import Adam
from rcbf_sac.utils import soft_update, hard_update
from rcbf_sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
from rcbf_sac.diff_cbf_qp import CBFQPLayer
from rcbf_sac.utils import to_tensor
from rcbf_sac.compensator import Compensator
from copy import deepcopy

class RCBF_SAC(object):

    def __init__(self, num_inputs, action_space, env, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.action_space = action_space
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # CBF layer
        self.env = env
        self.cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
        self.diff_qp = args.diff_qp

        # compensator
        if args.use_comp:
            self.compensator = Compensator(num_inputs, action_space.shape[0], action_space.low, action_space.high, args)
        else:
            self.compensator = None

    def select_action(self, state, dynamics_model, evaluate=False, warmup=False):

        state = to_tensor(state, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)

        if warmup:
            batch_size = state.shape[0]
            action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
            for i in range(batch_size):
                action[i] = torch.from_numpy(self.action_space.sample()).to(self.device)
        else:
            if evaluate is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)

        if self.compensator:
            action_comp = self.compensator(state)
            action += action_comp

        safe_action = self.get_safe_action(state, action, dynamics_model)

        if not self.compensator:
            return safe_action.detach().cpu().numpy()[0] if expand_dim else safe_action.detach().cpu().numpy()
        else:
            action_cbf = safe_action - action
            action_comp = action_comp.detach().cpu().numpy()[0] if expand_dim else action_comp.detach().cpu().numpy()
            action_cbf = action_cbf.detach().cpu().numpy()[0] if expand_dim else action_cbf.detach().cpu().numpy()
            safe_action = safe_action.detach().cpu().numpy()[0] if expand_dim else safe_action.detach().cpu().numpy()
            return safe_action, action_comp, action_cbf

    def update_parameters(self, memory, batch_size, updates, dynamics_model, memory_model=None, real_ratio=None):
        """

        Parameters
        ----------
        memory : ReplayMemory
        batch_size : int
        updates : int
        dynamics_model : GP Dynamics' Disturbance model D(x) in x_dot = f(x) + g(x)u + D(x)
        memory_model : ReplayMemory, optional
                If not none, perform model-based RL.
        real_ratio : float, optional
                If performing model-based RL, then real_ratio*batch_size are sampled from the real buffer, and the rest
                is sampled from the model buffer.
        Returns
        -------

        """

        # Model-based vs regular RL
        if memory_model and real_ratio:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=int(real_ratio*batch_size))
            state_batch_m, action_batch_m, reward_batch_m, next_state_batch_m, mask_batch_m, t_batch_m, next_t_batch_m = memory_model.sample(
                batch_size=int((1-real_ratio) * batch_size))
            state_batch = np.vstack((state_batch, state_batch_m))
            action_batch = np.vstack((action_batch, action_batch_m))
            reward_batch = np.hstack((reward_batch, reward_batch_m))
            next_state_batch = np.vstack((next_state_batch, next_state_batch_m))
            mask_batch = np.hstack((mask_batch, mask_batch_m))
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            if self.diff_qp:  # Compute next safe actions using Differentiable CBF-QP
                next_state_action = self.get_safe_action(next_state_batch, next_state_action, dynamics_model)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute Actions and log probabilities
        pi, log_pi, _ = self.policy.sample(state_batch)
        if self.diff_qp:  # Compute safe action using Differentiable CBF-QP
            pi = self.get_safe_action(state_batch, pi, dynamics_model)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For Comet.ml logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For Comet.ml logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def update_parameters_compensator(self, comp_rollouts):

        if self.compensator:
            self.compensator.train(comp_rollouts)

    # Save model parameters
    def save_model(self, output):
        print('Saving models in {}'.format(output))
        torch.save(
            self.policy.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        if self.compensator:
            self.compensator.save_model(output)

    # Load model parameters
    def load_weights(self, output):
        if output is None: return
        print('Loading models from {}'.format(output))

        self.policy.load_state_dict(
            torch.load('{}/actor.pkl'.format(output), map_location=torch.device(self.device))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output), map_location=torch.device(self.device))
        )
        if self.compensator:
            self.compensator.load_weights(output)

    def load_model(self, actor_path, critic_path):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    def get_safe_action(self, obs_batch, action_batch, dynamics_model):
        """Given a nominal action, returns a minimally-altered safe action to take.

        Parameters
        ----------
        obs_batch : torch.tensor
        action_batch : torch.tensor
        dynamics_model : DynamicsModel

        Returns
        -------
        safe_action_batch : torch.tensor
            Safe actions to be taken (cbf_action + action).
        """

        state_batch = dynamics_model.get_state(obs_batch)
        mean_pred_batch, sigma_pred_batch = dynamics_model.predict_disturbance(state_batch)

        safe_action_batch = self.cbf_layer.get_safe_action(state_batch, action_batch, mean_pred_batch, sigma_pred_batch)

        return safe_action_batch