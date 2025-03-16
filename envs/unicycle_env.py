import numpy as np
import gym
from gym import spaces

from envs.utils import to_pixel


class UnicycleEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(UnicycleEnv, self).__init__()

        self.dynamics_mode = 'Unicycle'
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(7,))
        self.bds = np.array([[-3., -3.], [3., 3.]])
        self.hazards_radius = 0.6
        self.hazards_locations = np.array([[0., 0], [-1., 1.], [-1., -1.], [1., -1.], [1., 1.]]) * 1.5 #np.random.rand(5,2)*3-1.5  #
        self.dt = 0.02
        self.max_episode_steps = 1000
        self.reward_goal = 1.0
        self.goal_size = 0.3
        # Initialize Env
        self.state = None
        self.episode_step = 0
        self.goal_pos = np.array([2.5, 2.5])

        self.reset()
        # Get Dynamics
        self.get_f, self.get_g = self._get_dynamics()
        # Disturbance
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20

        # Viewer
        self.viewer = None

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        action = np.clip(action, -1.0, 1.0)
        state, reward, done, info = self._step(action)
        return self.get_obs(), reward, done, info

    def _step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        state : ndarray
            New internal state of the agent.
        reward : float
            Reward collected during this transition.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        self.state += self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action)
        self.state -= self.dt * 0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0])  #* np.random.multivariate_normal(self.disturb_mean, self.disturb_covar, 1).squeeze()

        self.episode_step += 1

        info = dict()

        dist_goal = self._goal_dist()
        reward = (self.last_goal_dist - dist_goal)
        self.last_goal_dist = dist_goal
        # Check if goal is met
        if self.goal_met():
            info['goal_met'] = True
            reward += self.reward_goal
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps


        # Include constraint cost in reward
        if np.any(np.sum((self.state[:2] - self.hazards_locations)**2, axis=1) < self.hazards_radius**2):
            if 'cost' in info:
                info['cost'] += 0.1
            else:
                info['cost'] = 0.1
        return self.state, reward, done, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """
        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """
        self.episode_step = 0

        # Re-initialize state
        self.state = np.array([-2.5, -2.5, 0.])

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()

        # TODO: Randomize this
        return self.get_obs()

    def render(self, mode='human', close=False):
        """Render the environment to the screen
         Parameters
         ----------
         mode : str
         close : bool
         Returns
         -------
         """

        if mode != 'human' and mode != 'rgb_array':
            rel_loc = self.goal_pos - self.state[:2]
            theta_error = np.arctan2(rel_loc[1], rel_loc[0]) - self.state[2]
            print('Ep_step = {}, \tState = {}, \tDist2Goal = {}, alignment_error = {}'.format(self.episode_step,
                                                                                              self.state,
                                                                                              self._goal_dist(),
                                                                                              theta_error))

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from envs import pyglet_rendering
            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            # Draw obstacles
            obstacles = []
            for i in range(len(self.hazards_locations)):
                obstacles.append(
                    pyglet_rendering.make_circle(radius=to_pixel(self.hazards_radius, shift=0), filled=True))
                obs_trans = pyglet_rendering.Transform(translation=(
                to_pixel(self.hazards_locations[i][0], shift=screen_width / 2),
                to_pixel(self.hazards_locations[i][1], shift=screen_height / 2)))
                obstacles[i].set_color(1.0, 0.0, 0.0)
                obstacles[i].add_attr(obs_trans)
                self.viewer.add_geom(obstacles[i])

            # Make Goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(0.1, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(translation=(
            to_pixel(self.goal_pos[0], shift=screen_width / 2), to_pixel(self.goal_pos[1], shift=screen_height / 2)))
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)

            # Make Robot
            self.robot = pyglet_rendering.make_circle(radius=to_pixel(0.1), filled=True)
            self.robot_trans = pyglet_rendering.Transform(translation=(
            to_pixel(self.state[0], shift=screen_width / 2), to_pixel(self.state[1], shift=screen_height / 2)))
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(15.0, 0.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)

        if self.state is None:
            return None

        self.robot_trans.set_translation(to_pixel(self.state[0], shift=screen_width / 2),
                                         to_pixel(self.state[1], shift=screen_height / 2))
        self.robot_trans.set_rotation(self.state[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
        """

        rel_loc = self.goal_pos - self.state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal

        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), goal_compass[0], goal_compass[1], np.exp(-goal_dist)])

    def _get_dynamics(self):
        """Get affine CBFs for a given environment.
        Parameters
        ----------
        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        def get_f(state):
            f_x = np.zeros(state.shape)
            return f_x

        def get_g(state):
            theta = state[2]
            g_x = np.array([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [            0, 1.0]])
            return g_x
        return get_f, get_g

    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2])

def get_random_hazard_locations(n_hazards, hazard_radius, bds=None):
    """

    Parameters
    ----------
    n_hazards : int
        Number of hazards to create
    hazard_radius : float
        Radius of hazards
    bds : list, optional
        List of the form [[x_lb, x_ub], [y_lb, y_ub] denoting the bounds of the 2D arena

    Returns
    -------
    hazards_locs : ndarray
        Numpy array of shape (n_hazards, 2) containing xy locations of hazards.
    """

    if bds is None:
        bds = np.array([[-3., -3.], [3., 3.]])

    # Create buffer with boundaries
    buffered_bds = bds
    buffered_bds[0] += hazard_radius
    buffered_bds[1] -= hazard_radius

    hazards_locs = np.zeros((n_hazards, 2))

    for i in range(n_hazards):
        successfully_placed = False
        iter = 0
        while not successfully_placed and iter < 500:
            hazards_locs[i] = (bds[1] - bds[0]) * np.random.random(2) + bds[0]
            successfully_placed = np.all(np.linalg.norm(hazards_locs[:i] - hazards_locs[i], axis=1) > 3*hazard_radius)
            iter += 1

        if iter >= 500:
            raise Exception('Could not place hazards in arena.')

    return hazards_locs

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from rcbf_sac.cbf_qp import CascadeCBFLayer
    from rcbf_sac.dynamics import DynamicsModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="SafetyGym", help='Either SafetyGym or Unicycle.')
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=40, type=float)
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    args = parser.parse_args()

    env = UnicycleEnv()
    dynamics_model = DynamicsModel(env, args)
    cbf_wrapper = CascadeCBFLayer(env, gamma_b=args.gamma_b, k_d=args.k_d)


    def simple_controller(env, state, goal):
        goal_xy = goal[:2]
        goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
        v = 4.0 * goal_dist
        relative_theta = 1.0 * np.arctan2(goal_xy[1], goal_xy[0])
        omega = 5.0 * relative_theta
        return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)

    obs = env.reset()
    done = False
    episode_reward = 0
    episode_step = 0

    # Plot initial state
    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    for i in range(len(env.hazards_locations)):
        ax.add_patch(plt.Circle(env.hazards_locations[i], env.hazards_radius, color='r'))
    ax.add_patch(plt.Circle(env.goal_pos, env.goal_size, color='g'))
    p_pos = ax.scatter(obs[0], obs[1], s=300)
    p_theta = plt.quiver(obs[0], obs[1], obs[0] + .2 * obs[2], .2 * obs[3])
    plt.xlim([-3.0, 3.0])
    plt.ylim([-3.0, 3.0])
    ax.set_aspect('equal', 'box')

    while not done:
        # Plot current state
        p_pos.set_offsets([obs[0], obs[1]])
        p_theta.XY[:, 0] = obs[0]
        p_theta.XY[:, 1] = obs[1]
        p_theta.set_UVC(.2 * obs[2], .2 * obs[3])
        # Take Action and get next state
        # random_action = env.action_space.sample()
        state = dynamics_model.get_state(obs)
        random_action = simple_controller(env, state, obs[-3:])
        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)
        action_safe = cbf_wrapper.get_u_safe(random_action, state, disturb_mean, disturb_std)
        obs, reward, done, info = env.step(random_action + action_safe)
        plt.pause(0.01)
        episode_reward += reward
        episode_step += 1
        print('step {} \tepisode_reward = {}'.format(episode_step, episode_reward))
    plt.show()

