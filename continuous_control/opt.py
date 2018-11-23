import argparse

parser = argparse.ArgumentParser(description='RL for continuous control')

parser.add_argument('--n_episodes', default=2000, type=int, help='maximum number of training episodes')

parser.add_argument('--success_score', default=30., type=float, help='score condition to success')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--model_path', default='checkpoint.pth', type=str, help='model checkpoint path')

parser.add_argument('--buffer_size', default=1e5, type=int, help='replay buffer size')
parser.add_argument('--batch_size', default=128, type=int, help='mini batch size')
parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
parser.add_argument('--tau', default=1e-3, type=float, help='for soft update of target parameters')
parser.add_argument('--lr_actor', default=1e-4, type=float, help='learning rate of the actor')
parser.add_argument('--lr_critic', default=1e-4, type=float, help='learning rate of the critic')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 weight decay')
#parser.add_argument('--list_fc_dims_actor', default='400,300', type=str, help='FC layer dimensions for actor')
#parser.add_argument('--list_fc_dims_critic', default='400,300', type=str, help='FC layer dimensions for actor')

opt = parser.parse_args(args=[])
