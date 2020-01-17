from collections import OrderedDict

import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
from gym.wrappers.monitor import Monitor


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    # create a renderer
    env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()
    # get parameters before update
    params_preupdate = OrderedDict(policy.named_parameters())

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    logs = {'tasks': []}
    train_returns, valid_returns = [], []
    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        # get parameters of the learned agent
        params_postupdate = OrderedDict(policy.named_parameters())
        # set task for the environment
        env.unwrapped.reset_task(tasks[0])
        # iterate over pre- and post- update parameters
        for params in [params_preupdate, params_postupdate]:
            # set rewards of the episode
            rewards = 0
            # define episode length
            for _ in np.arange(1000):
                # reset the environment
                observations = env.reset()
                # compute actions from the policy
                with torch.no_grad():
                    observations_tensor = torch.from_numpy(observations).to(device='cpu')
                    actions_tensor = policy(observations_tensor, params=params).sample()
                    actions = actions_tensor.cpu().numpy()
                # perform stepping through the environment
                new_observations, reward, dones, info, = env.step(actions)
                observations, info = new_observations, info
                rewards = rewards + reward
                # print("rewards: ", rewards)
                env.render(mode='human')
            print(f'Total reward aquired: {rewards}')

        logs['tasks'].extend(tasks)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

    logs['train_returns'] = np.concatenate(train_returns, axis=0)
    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

    with open(args.output, 'wb') as f:
        np.savez(f, **logs)

    env.close()


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                 'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str,
                        help='path to the configuration file')
    parser.add_argument('--policy', type=str,
                        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
                            help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
                            help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str,
                      help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
                      help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                      help='number of workers for trajectories sampling (default: '
                           '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
                      help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
                           'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()

    # Manually set the parameters
    args.config = './results/hc-vel-exposed/config.json'
    args.policy = './results/hc-vel-exposed/policy.th'
    args.output = './results_test/hc-vel-exposed/results.npz'
    args.meta_batch_size = 1
    args.num_batches = 1
    args.seed = None
    args.num_workers = 10
    args.use_cuda = False
    args.device = ('cuda' if (torch.cuda.is_available()
                              and args.use_cuda) else 'cpu')

    main(args)