import argparse

parser = argparse.ArgumentParser(description='RL for navigation')

parser.add_argument('--n_episodes', default=2000, type=int, help='maximum number of training episodes')
parser.add_argument('--max_t', default=1000, type=int, help='maximum number of timesteps per episode')
parser.add_argument('--eps_start', default=1.0, type=float, help='starting value of epsilon,')
parser.add_argument('--eps_end', default=0.01, type=float, help='minimum value of epsilon')
parser.add_argument('--eps_decay', default=0.995, type=float, help='rate to decay epsilon')
parser.add_argument('--success_score', default=13., type=float, help='score condition to success')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--model_path', default='checkpoint.pth', type=str, help='model checkpoint path')

opt = parser.parse_args(args=[])
