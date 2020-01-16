import gym
import torch
import json
import os
import yaml
from tqdm import trange
import time
import numpy as np

import dowel
from dowel import logger, tabular
import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')
        tb_filename = os.path.join(args.output_folder, 'tensorboard_logir')
        csv_filename = os.path.join(args.output_folder, 'progress.csv')
        log_filename = os.path.join(args.output_folder, 'log.txt')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)
        logger.add_output(dowel.StdOutput())
        logger.add_output(dowel.TextOutput(log_filename))
        logger.add_output(dowel.CsvOutput(csv_filename))
        logger.add_output(dowel.TensorBoardOutput(tb_filename))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()

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

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    num_iterations = 0
    logger.log('Starting up...')
    start_time = time.time()

    for batch in range(config['num-batches']):
        itr_start_time = time.time()
        logger.push_prefix('Itr #{}: '.format(batch))
        logger.log(f'Running training step!')

        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)

            tabular.record('itr', batch)
            tabular.record('tasks', tasks)
            tabular.record('num_iterations', num_iterations)
            tabular.record('train_returns/mean', np.mean(get_returns(train_episodes[0])))
            tabular.record('train_returns/max', np.max(get_returns(train_episodes[0])))
            tabular.record('train_returns/min', np.min(get_returns(train_episodes[0])))
            tabular.record('valid_returns/mean', np.mean(get_returns(valid_episodes)))
            tabular.record('valid_returns/max', np.max(get_returns(valid_episodes)))
            tabular.record('valid_returns/min', np.min(get_returns(valid_episodes)))
            logger.record('total_time', time.time() - start_time)
            logger.record('itr_time', time.time() - itr_start_time)
            logger.log(tabular)

            logger.pop_prefix()
            logger.dump_all()


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                 'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str,
                        help='path to the configuration file.', default='./configs/maml/halfcheetah-vel.yaml')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
                      help='name of the output folder', default='./results')
    misc.add_argument('--seed', type=int, default=None,
                      help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                      help='number of workers for trajectories sampling (default: '
                           '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
                      help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
                           'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()

    # Manually set the parameters
    args.config = './configs/maml/halfcheetah-vel.yaml'
    args.output_folder = './results/hc-vel-exposed'
    args.seed = None
    args.num_workers = 10
    args.use_cuda = False
    args.device = ('cuda' if (torch.cuda.is_available()
                              and args.use_cuda) else 'cpu')

    main(args)
