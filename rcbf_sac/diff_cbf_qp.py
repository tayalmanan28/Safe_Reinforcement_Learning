import argparse
import numpy as np
import torch
from rcbf_sac.dynamics import DYNAMICS_MODE
from rcbf_sac.utils import to_tensor, prRed
from time import time
from qpth.qp import QPFunction


class CBFQPLayer:

    def __init__(self, env, args, gamma_b=100, k_d=1.5, l_p=0.03):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b

        if self.env.dynamics_mode not in DYNAMICS_MODE:
            raise Exception('Dynamics mode not supported.')

        if self.env.dynamics_mode == 'Unicycle':
            self.num_cbfs = len(env.hazards_locations)
            self.k_d = k_d
            self.l_p = l_p
        elif self.env.dynamics_mode == 'SimulatedCars':
            self.num_cbfs = 2

        self.action_dim = env.action_space.shape[0]
        self.num_ineq_constraints = self.num_cbfs + 2 * self.action_dim

    def get_safe_action(self, state_batch, action_batch, mean_pred_batch, sigma_batch):
        """
        Parameters
        ----------
        state_batch : torch.tensor or ndarray
        action_batch : torch.tensor or ndarray
            State batch
        mean_pred_batch : torch.tensor or ndarray
            Mean of disturbance
        sigma_batch : torch.tensor or ndarray
            Standard deviation of disturbance

        Returns
        -------
        final_action_batch : torch.tensor
            Safe actions to take in the environment.
        """

        # batch form if only a single data point is passed
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            action_batch = action_batch.unsqueeze(0)
            state_batch = state_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)

        start_time = time()
        Ps, qs, Gs, hs = self.get_cbf_qp_constraints(state_batch, action_batch, mean_pred_batch, sigma_batch)
        build_qp_time = time()
        safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
        # prCyan('Time to get constraints = {} - Time to solve QP = {} - time per qp = {} - batch_size = {} - device = {}'.format(build_qp_time - start_time, time() - build_qp_time, (time() - build_qp_time) / safe_action_batch.shape[0], Ps.shape[0], Ps.device))
        # The actual safe action is the cbf action + the nominal action
        final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))

        return final_action if not expand_dims else final_action.squeeze(0)

    def solve_qp(self, Ps: torch.Tensor, qs: torch.Tensor, Gs: torch.Tensor, hs: torch.Tensor):
        """Solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        Ps : torch.Tensor
            (batch_size, n_u+1, n_u+1)
        qs : torch.Tensor
            (batch_size, n_u+1)
        Gs : torch.Tensor
            (batch_size, num_ineq_constraints, n_u+1)
        hs : torch.Tensor
            (batch_size, num_ineq_constraints)
        Returns
        -------
        safe_action_batch : torch.tensor
            The solution of the qp without the last dimension (the slack).
        """
        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)
        sol = self.cbf_layer(Ps, qs, Gs, hs, solver_args={"check_Q_spd": False, "maxIter": 100000, "notImprovedLim": 10, "eps": 1e-4})
        safe_action_batch = sol[:, :-1]
        return safe_action_batch

    def cbf_layer(self, Qs, ps, Gs, hs, As=None, bs=None, solver_args=None):
        """
        Parameters
        ----------
        Qs : torch.Tensor
        ps : torch.Tensor
        Gs : torch.Tensor
            shape (batch_size, num_ineq_constraints, num_vars)
        hs : torch.Tensor
            shape (batch_size, num_ineq_constraints)
        As : torch.Tensor, optional
        bs : torch.Tensor, optional
        solver_args : dict, optional

        Returns
        -------
        result : torch.Tensor
            Result of QP
        """

        if solver_args is None:
            solver_args = {}

        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        result = QPFunction(verbose=0, **solver_args)(Qs.double(), ps.double(), Gs.double(), hs.double(), As, bs).float()

        if torch.any(torch.isnan(result)):
            prRed('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
            raise Exception('QP Failed to solve')
        return result

    def get_cbf_qp_constraints(self, state_batch, action_batch, mean_pred_batch, sigma_pred_batch):
        """Build up matrices required to solve qp
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Each control barrier certificate is of the form:
            dh/dx^T (f_out + g_out u) >= -gamma^b h_out^3 where out here is an output of the state.

        In the case of SafetyGym_point dynamics:
        state = [x y θ v ω]
        state_d = [v*cos(θ) v*sin(θ) omega ω u^v u^ω]

        Quick Note on batch matrix multiplication for matrices A and B:
            - Batch size should be first dim
            - Everything needs to be 3-dimensional
            - E.g. if B is a vec, i.e. shape (batch_size, vec_length) --> .view(batch_size, vec_length, 1)

        Parameters
        ----------
        state_batch : torch.tensor
            current state (check dynamics.py for details on each dynamics' specifics)
        action_batch : torch.tensor
            Nominal control input.
        mean_pred_batch : torch.tensor
            mean disturbance prediction state, dimensions (n_s, n_u)
        sigma_pred_batch : torch.tensor
            standard deviation in additive disturbance after undergoing the output dynamics.
        gamma_b : float, optional
            CBF parameter for the class-Kappa function

        Returns
        -------
        P : torch.tensor
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        q : torch.tensor
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        G : torch.tensor
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, n_u + 1)
        h : torch.tensor
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        """

        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(
            sigma_pred_batch.shape) == 2, print(state_batch.shape, action_batch.shape, mean_pred_batch.shape,
                                                sigma_pred_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)
        mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
        sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)

        if self.env.dynamics_mode == 'Unicycle':

            num_cbfs = self.num_cbfs
            hazards_radius = self.env.hazards_radius
            hazards_locations = to_tensor(self.env.hazards_locations, torch.FloatTensor, self.device)
            collision_radius = 1.2 * hazards_radius  # add a little buffer
            l_p = self.l_p

            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)

            # p(x): lookahead output (batch_size, 2)
            ps = torch.zeros((batch_size, 2)).to(self.device)
            ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
            ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

            # p_dot(x) = f_p(x) + g_p(x)u + D_p where f_p(x) = 0,  g_p(x) = RL and D_p is the disturbance

            # f_p(x) = [0,...,0]^T
            f_ps = torch.zeros((batch_size, 2, 1)).to(self.device)

            # g_p(x) = RL where L = diag([1, l_p])
            Rs = torch.zeros((batch_size, 2, 2)).to(self.device)
            Rs[:, 0, 0] = c_thetas
            Rs[:, 0, 1] = -s_thetas
            Rs[:, 1, 0] = s_thetas
            Rs[:, 1, 1] = c_thetas
            Ls = torch.zeros((batch_size, 2, 2)).to(self.device)
            Ls[:, 0, 0] = 1
            Ls[:, 1, 1] = l_p
            g_ps = torch.bmm(Rs, Ls)  # (batch_size, 2, 2)

            # D_p(x) = g_p [0 D_θ]^T + [D_x1 D_x2]^T
            mu_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            mu_theta_aug[:, 1, :] = mean_pred_batch[:, 2, :]
            mu_ps = torch.bmm(g_ps, mu_theta_aug) + mean_pred_batch[:, :2, :]
            sigma_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            sigma_theta_aug[:, 1, :] = sigma_pred_batch[:, 2, :]
            sigma_ps = torch.bmm(torch.abs(g_ps), sigma_theta_aug) + sigma_pred_batch[:, :2, :]

            # hs (batch_size, hazards_locations)
            ps_hzds = ps.repeat((1, num_cbfs)).reshape((batch_size, num_cbfs, 2))

            hs = 0.5 * (torch.sum((ps_hzds - hazards_locations.view(1, num_cbfs, -1)) ** 2, axis=2) - collision_radius ** 2)  # 1/2 * (||x - x_obs||^2 - r^2)

            dhdps = (ps_hzds - hazards_locations.view(1, num_cbfs, -1))  # (batch_size, n_cbfs, 2)
                                                                          # (batch_size, 2, 1)
            n_u = action_batch.shape[1]  # dimension of control inputs
            num_constraints = num_cbfs + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            # Add inequality constraints
            G[:, :num_cbfs, :n_u] = -torch.bmm(dhdps, g_ps)  # h1^Tg(x)
            G[:, :num_cbfs, n_u] = -1  # for slack
            h[:, :num_cbfs] = gamma_b * (hs ** 3) + (torch.bmm(dhdps, f_ps + mu_ps) - torch.bmm(torch.abs(dhdps), sigma_ps) + torch.bmm(torch.bmm(dhdps, g_ps), action_batch)).squeeze(-1)
            ineq_constraint_counter += num_cbfs

            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(torch.tensor([1.e0, 1.e-2, 1e5])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)

        elif self.env.dynamics_mode == 'SimulatedCars':

            n_u = action_batch.shape[1]  # dimension of control inputs
            num_constraints = self.num_cbfs + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)
            collision_radius = 3.5

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            # Current State
            pos = state_batch[:, ::2, 0]
            vels = state_batch[:, 1::2, 0]

            # Action (acceleration)
            vels_des = 30.0 * torch.ones((state_batch.shape[0], 5)).to(self.device)  # Desired velocities
            # vels_des[:, 0] -= 10 * torch.sin(0.2 * t_batch)
            accels = self.env.kp * (vels_des - vels)
            accels[:, 1] -= self.env.k_brake * (pos[:, 0] - pos[:, 1]) * ((pos[:, 0] - pos[:, 1]) < 6.0)
            accels[:, 2] -= self.env.k_brake * (pos[:, 1] - pos[:, 2]) * ((pos[:, 1] - pos[:, 2]) < 6.0)
            accels[:, 3] = 0.0  # Car 4's acceleration is controlled directly
            accels[:, 4] -= self.env.k_brake * (pos[:, 2] - pos[:, 4]) * ((pos[:, 2] - pos[:, 4]) < 13.0)

            # f(x)
            f_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
            f_x[:, ::2] = vels
            f_x[:, 1::2] = accels

            # f_D(x) - disturbance in the drift dynamics
            fD_x = torch.zeros((state_batch.shape[0], state_batch.shape[1])).to(self.device)
            fD_x[:, 1::2] = sigma_pred_batch[:, 1::2, 0].squeeze(-1)

            # g(x)
            g_x = torch.zeros((state_batch.shape[0], state_batch.shape[1], 1)).to(self.device)
            g_x[:, 7, 0] = 50.0  # Car 4's acceleration

            # h1
            h13 = 0.5 * (((pos[:, 2] - pos[:, 3]) ** 2) - collision_radius ** 2)
            h15 = 0.5 * (((pos[:, 4] - pos[:, 3]) ** 2) - collision_radius ** 2)

            # dh1/dt = Lfh1
            h13_dot = (pos[:, 3] - pos[:, 2]) * (vels[:, 3] - vels[:, 2])
            h15_dot = (pos[:, 3] - pos[:, 4]) * (vels[:, 3] - vels[:, 4])

            # Lffh1
            dLfh13dx = torch.zeros((batch_size, 10)).to(self.device)
            dLfh13dx[:, 4] = (vels[:, 2] - vels[:, 3])  # Car 3 pos
            dLfh13dx[:, 5] = (pos[:, 2] - pos[:, 3])  # Car 3 vel
            dLfh13dx[:, 6] = (vels[:, 3] - vels[:, 2])
            dLfh13dx[:, 7] = (pos[:, 3] - pos[:, 2])
            Lffh13 = torch.bmm(dLfh13dx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
            LfDfh13 = torch.bmm(torch.abs(dLfh13dx.view(batch_size, 1, -1)), fD_x.view(batch_size, -1, 1)).squeeze()

            dLfh15dx = torch.zeros((batch_size, 10)).to(self.device)
            dLfh15dx[:, 8] = (vels[:, 4] - vels[:, 3])  # Car 5 pos
            dLfh15dx[:, 9] = (pos[:, 4] - pos[:, 3])  # Car 5 vels
            dLfh15dx[:, 6] = (vels[:, 3] - vels[:, 4])
            dLfh15dx[:, 7] = (pos[:, 3] - pos[:, 4])
            Lffh15 = torch.bmm(dLfh15dx.view(batch_size, 1, -1), f_x.view(batch_size, -1, 1)).squeeze()
            LfDfh15 = torch.bmm(torch.abs(dLfh15dx.view(batch_size, 1, -1)), fD_x.view(batch_size, -1, 1)).squeeze()

            # Lfgh1
            Lgfh13 = torch.bmm(dLfh13dx.view(batch_size, 1, -1), g_x)
            Lgfh15 = torch.bmm(dLfh15dx.view(batch_size, 1, -1), g_x)

            # print('state = {}'.format(state_batch))
            # print('f_x = {}'.format(f_x))
            # print('g_x = {}'.format(g_x))
            # print('h13 = {}'.format(h13))
            # print('h15 = {}'.format(h15))
            # print('gamma_b = {}'.format(self.gamma_b))
            # print('h13_dot = {}'.format(h13_dot))
            # print('h15_dot = {}'.format(h15_dot))
            # print('Lffh13 = {}'.format(Lffh13))
            # print('Lffh15 = {}'.format(Lffh15))
            # print('Lgfh13 = {}'.format(Lgfh13))
            # print('Lgfh15 = {}'.format(Lgfh15))

            # Inequality constraints (G[u, eps] <= h)
            h[:, 0] = Lffh13 - LfDfh13 + (gamma_b + gamma_b) * h13_dot + gamma_b * gamma_b * h13 + torch.bmm(Lgfh13, action_batch).squeeze()
            h[:, 1] = Lffh15 - LfDfh15 + (gamma_b + gamma_b) * h15_dot + gamma_b * gamma_b * h15 + torch.bmm(Lgfh15, action_batch).squeeze()
            G[:, 0, 0] = -Lgfh13.squeeze()
            G[:, 1, 0] = -Lgfh15.squeeze()
            G[:, :self.num_cbfs, n_u] = -2e2  # for slack
            ineq_constraint_counter += self.num_cbfs

            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(torch.tensor([0.1, 1e1])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)

        else:
            raise Exception('Dynamics mode unknown!')

        # Second let's add actuator constraints
        n_u = action_batch.shape[1]  # dimension of control inputs

        for c in range(n_u):

            # u_max >= u_nom + u ---> u <= u_max - u_nom
            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

            # u_min <= u_nom + u ---> -u <= u_min - u_nom
            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = -self.u_min[c] + action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

        return P, q, G, h

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max


if __name__ == "__main__":

    from build_env import build_env
    from rcbf_sac.dynamics import DynamicsModel
    from copy import deepcopy
    from rcbf_sac.utils import to_numpy, prGreen


    def simple_controller(env, state, goal):
        goal_xy = goal[:2]
        goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
        v = 0.02 * goal_dist
        relative_theta = 1.0 * np.arctan2(goal_xy[1], goal_xy[0])
        omega = 1.0 * relative_theta

        return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)


    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="SafetyGym", help='Options are Unicycle or SafetyGym')
    parser.add_argument('--robot_xml', default='xmls/point.xml',
                        help="SafetyGym Currently only supporting xmls/point.xml")
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--l_p', default=0.03, type=float)
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--cuda', action='store_true', help='run on CUDA (default: False)')
    args = parser.parse_args()
    # Environment
    env = build_env(args)

    device = torch.device('cuda' if args.cuda else 'cpu')


    def to_def_tensor(ndarray):

        return to_tensor(ndarray, torch.FloatTensor, device)


    diff_cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
    dynamics_model = DynamicsModel(env, args)

    obs = env.reset()
    done = False

    ep_ret = 0
    ep_cost = 0
    ep_step = 0

    for i_step in range(3000):

        if done:
            prGreen('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
            ep_ret, ep_cost, ep_step = 0, 0, 0
            obs = env.reset()

        state = dynamics_model.get_state(obs)

        print('state = {}, dist2hazards = {}'.format(state[:2],
                                                     np.sqrt(np.sum((env.hazards_locations - state[:2]) ** 2, 1))))

        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)

        action = simple_controller(env, state, obs[-3:])  # TODO: observations last 3 indicated
        # action = 2*np.random.rand(2) - 1.0
        assert env.action_space.contains(action)
        final_action = diff_cbf_layer.get_safe_action(to_def_tensor(state), to_def_tensor(action),
                                                      to_def_tensor(disturb_mean), to_def_tensor(disturb_std))
        final_action = to_numpy(final_action)

        # Env Step
        observation2, reward, done, info = env.step(final_action)
        observation2 = deepcopy(observation2)

        # Update state and store transition for GP model learning
        next_state = dynamics_model.get_state(observation2)
        if ep_step % 2 == 0:
            dynamics_model.append_transition(state, final_action, next_state)

        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_step += 1
        # env.render()

        obs = observation2
        state = next_state