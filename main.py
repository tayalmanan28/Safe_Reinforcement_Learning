# import comet_ml at the top of your file
from comet_ml import Experiment

import argparse
import torch
import numpy as np

from rcbf_sac.sac_cbf import RCBF_SAC
from rcbf_sac.replay_memory import ReplayMemory
from rcbf_sac.dynamics import DynamicsModel
from build_env import *
import os

from rcbf_sac.utils import prGreen, get_output_folder, prYellow
from rcbf_sac.evaluator import Evaluator
from rcbf_sac.generate_rollouts import generate_model_rollouts


def train(agent, env, dynamics_model, args, experiment=None):

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory_model = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    if args.use_comp:
        compensator_rollouts = []
        comp_buffer_idx = 0

    for i_episode in range(args.max_episodes):
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0
        done = False
        obs = env.reset()

        # Saving rollout here to train compensator
        if args.use_comp:
            episode_rollout = dict()
            episode_rollout['obs'] = np.zeros((0, env.observation_space.shape[0]))
            episode_rollout['u_safe'] = np.zeros((0, env.action_space.shape[0]))
            episode_rollout['u_comp'] = np.zeros((0, env.action_space.shape[0]))

        while not done:
            if episode_steps % 10 == 0:
                prYellow('Episode {} - step {} - eps_rew {} - eps_cost {}'.format(i_episode, episode_steps, episode_reward, episode_cost))
            state = dynamics_model.get_state(obs)

            # Generate Model rollouts
            if args.model_based and episode_steps % 5 == 0 and len(memory) > dynamics_model.max_history_count / 3:
                memory_model = generate_model_rollouts(env, memory_model, memory, agent, dynamics_model,
                                                       k_horizon=args.k_horizon,
                                                       batch_size=min(len(memory), 5 * args.rollout_batch_size),
                                                       warmup=args.start_steps > total_numsteps)

            # If using model-based RL then we only need to have enough data for the real portion of the replay buffer
            if len(memory) + len(memory_model) * args.model_based > args.batch_size:

                # Number of updates per step in environment
                for i in range(args.updates_per_step):

                    # Update parameters of all the networks
                    if args.model_based:
                        # Pick the ratio of data to be sampled from the real vs model buffers
                        real_ratio = max(min(args.real_ratio, len(memory) / args.batch_size), 1 - len(memory_model) / args.batch_size)
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                             args.batch_size,
                                                                                                             updates,
                                                                                                             dynamics_model,
                                                                                                             memory_model,
                                                                                                             real_ratio)
                    else:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                           args.batch_size,
                                                                                                           updates,
                                                                                                           dynamics_model)

                    if experiment:
                        experiment.log_metric('loss/critic_1', critic_1_loss, updates)
                        experiment.log_metric('loss/critic_2', critic_2_loss, step=updates)
                        experiment.log_metric('loss/policy', policy_loss, step=updates)
                        experiment.log_metric('loss/entropy_loss', ent_loss, step=updates)
                        experiment.log_metric('entropy_temperature/alpha', alpha, step=updates)
                    updates += 1

            if args.use_comp:
                action, action_comp, action_cbf = agent.select_action(obs, dynamics_model, warmup=args.start_steps > total_numsteps)
            else:
                action = agent.select_action(obs, dynamics_model, warmup=args.start_steps > total_numsteps)  # Sample action from policy

            next_obs, reward, done, info = env.step(action)  # Step
            if 'cost_exception' in info:
                prYellow('Cost exception occured.')
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += info.get('cost', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            memory.push(obs, action, reward, next_obs, mask, t=episode_steps * env.dt, next_t=(episode_steps+1) * env.dt)  # Append transition to memory

            # Update state and store transition for GP model learning
            next_state = dynamics_model.get_state(next_obs)
            if episode_steps % 2 == 0 and i_episode < args.gp_max_episodes:  # Stop learning the dynamics after a while to stabilize learning
                # TODO: Clean up line below, specifically (t_batch)
                dynamics_model.append_transition(state, action, next_state, t_batch=np.array([episode_steps*env.dt]))

            # append comp rollout with step before updating
            if args.use_comp:
                episode_rollout['obs'] = np.vstack((episode_rollout['obs'], obs))
                episode_rollout['u_safe'] = np.vstack((episode_rollout['u_safe'], action_cbf))
                episode_rollout['u_comp'] = np.vstack((episode_rollout['u_comp'], action_comp))

            obs = next_obs

        # Train compensator
        if args.use_comp and i_episode < args.comp_train_episodes:
            if comp_buffer_idx < 50:  # TODO: Turn the 50 into an arg
                compensator_rollouts.append(episode_rollout)
            else:
                comp_buffer_idx[comp_buffer_idx] = episode_rollout
            comp_buffer_idx = (comp_buffer_idx + 1) % 50
            if i_episode % args.comp_update_episode == 0:
                agent.update_parameters_compensator(compensator_rollouts)

        # [optional] save intermediate model
        if i_episode % int(args.max_episodes / 10) == 0:
            agent.save_model(args.output)
            dynamics_model.save_disturbance_models(args.output)

        if experiment:
            # Comet.ml logging
            experiment.log_metric('reward/train', episode_reward, step=i_episode)
            experiment.log_metric('cost/train', episode_cost, step=i_episode)
        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                             round(episode_reward, 2), round(episode_cost, 2)))

        # Evaluation
        if i_episode % 5 == 0 and args.eval is True:
            print('Size of replay buffers: real : {}, \t\t model : {}'.format(len(memory), len(memory_model)))
            avg_reward = 0.
            avg_cost = 0.
            episodes = 5
            for _ in range(episodes):
                obs = env.reset()
                episode_reward = 0
                episode_cost = 0
                done = False
                while not done:
                    if args.use_comp:
                        action, _, _ = agent.select_action(obs, dynamics_model, evaluate=True)
                    else:
                        action = agent.select_action(obs, dynamics_model, evaluate=True)  # Sample action from policy
                    next_obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    episode_cost += info.get('cost', 0)
                    obs = next_obs

                avg_reward += episode_reward
                avg_cost += episode_cost
            avg_reward /= episodes
            avg_cost /= episodes
            if experiment:
                experiment.log_metric('avg_reward/test', avg_reward, step=i_episode)
                experiment.log_metric('avg_cost/test', avg_cost, step=i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Cost: {}".format(episodes, round(avg_reward, 2), round(avg_cost, 2)))
            print("----------------------------------------")


def test(agent, env, dynamics_model, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    dynamics_model.load_disturbance_models(model_path)

    def policy(observation):
        if args.use_comp:
            action, action_comp, action_cbf = agent.select_action(observation, dynamics_model, evaluate=True)
        else:
            action = agent.select_action(observation, dynamics_model,evaluate=True)
        return action

    evaluate(env, policy, dynamics_model=dynamics_model, debug=debug, visualize=visualize, save=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="Unicycle", help='Options is Unicycle')
    # Comet ML
    parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
    parser.add_argument('--comet_key', default='', help='Comet API key')
    parser.add_argument('--comet_workspace', default='', help='Comet workspace')
    # SAC Args
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize env -only in available test mode')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 5 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 12345)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max_episodes', type=int, default=200, metavar='N',
                        help='maximum number of episodes (default: 200)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    # CBF, Dynamics, Env Args
    parser.add_argument('--no_diff_qp', action='store_false', dest='diff_qp', help='Should the agent diff through the CBF?')
    parser.add_argument('--gp_model_size', default=3000, type=int, help='gp')
    parser.add_argument('--gp_max_episodes', default=100, type=int, help='gp max train episodes.')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=20, type=float)
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Look-ahead distance for unicycle dynamics output.")
    # Model Based Learning
    parser.add_argument('--model_based', action='store_true', dest='model_based', help='If selected, will use data from the model to train the RL agent.')
    parser.add_argument('--real_ratio', default=0.3, type=float, help='Portion of data obtained from real replay buffer for training.')
    parser.add_argument('--k_horizon', default=1, type=int, help='horizon of model-based rollouts')
    parser.add_argument('--rollout_batch_size', default=5, type=int, help='Size of initial states batch to rollout from.')
    # Compensator
    parser.add_argument('--comp_rate', default=0.005, type=float, help='Compensator learning rate')
    parser.add_argument('--comp_train_episodes', default=200, type=int, help='Number of initial episodes to train compensator for.')
    parser.add_argument('--comp_update_episode', default=50, type=int, help='Modulo for compensator updates')
    parser.add_argument('--use_comp', type=bool, default=False, help='Should the compensator be used.')
    args = parser.parse_args()

    if args.mode == 'train':
        args.output = get_output_folder(args.output, args.env_name)
    if args.resume == 'default':
        args.resume = os.getcwd() + '/output/{}-run0'.format(args.env_name)
    elif args.resume.isnumeric():
        args.resume = os.getcwd() + '/output/{}-run{}'.format(args.env_name, args.resume)

    if args.cuda:
        torch.cuda.set_device(args.device_num)

    if args.mode == 'train' and args.log_comet:
        project_name = 'sac-rcbf-unicycle-environment' if args.env_name == 'Unicycle' else 'sac-rcbf-sim-cars-env'
        prYellow('Logging experiment on comet.ml!')
        # Create an experiment with your api key
        experiment = Experiment(
            api_key=args.comet_key,
            project_name=project_name,
            workspace=args.comet_workspace,
        )
        # Log args on comet.ml
        experiment.log_parameters(vars(args))
        experiment_tags = ['MB' if args.model_based else 'MF',
                           str(args.batch_size) + '_batch',
                           str(args.updates_per_step) + '_step_updates',
                           'diff_qp' if args.diff_qp else 'reg_qp']
        if args.use_comp:
            experiment_tags.append('use_comp')
        print(experiment_tags)
        experiment.add_tags(experiment_tags)
    else:
        experiment = None

    if args.use_comp and (args.model_based or args.diff_qp):
        raise Exception('Compensator can only be used with model free RL and regular CBF.')

    # Environment
    env = build_env(args)

    # Agent
    agent = RCBF_SAC(env.observation_space.shape[0], env.action_space, env, args)
    dynamics_model = DynamicsModel(env, args)

    # Random Seed
    if args.seed > 0:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dynamics_model.seed(args.seed)

    # If model based, we warm up in the model too
    if args.model_based:
        args.start_steps /= (1 + args.rollout_batch_size)

    if args.mode == 'train':
        train(agent, env, dynamics_model, args, experiment)
    elif args.mode == 'test':
        evaluate = Evaluator(args.validate_episodes, args.validate_steps, args.output)
        test(agent, env, dynamics_model, evaluate, args.resume, visualize=args.visualize, debug=True)

    env.close()

