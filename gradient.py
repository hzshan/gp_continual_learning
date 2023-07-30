import numpy as np
import tqdm, utils, theory, pickle, cluster_utils, data, torch, grad_utils, math, sys

UPDATE_FREQ = 5000  # print an update per UPDATE_FREQ steps
CONVERGENCE_THRESHOLD = 1000  # stop training once the loss hasn't decreased for this many steps
NORMALIZATION_SCHEME = 'none'
DIFF_DATA_SEED = False # whether or not to use a different seed for the data; if False, the seed used to sample data is always 0

"""
Updated Nov 1 2022
Do gradient-based simulations (SGD, Langevin etc.) of continual learning.
Saves: all the training/test acc/loss.
For the NN predictions, only the output from the 0th head is saved,
and the corresponding input is always the first test set!!
"""

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()
device = torch.device('cuda') if ON_CLUSTER else torch.device('cpu')

parser = cluster_utils.Args()
parser.add('P', 100, help='size of training set per task')
parser.add('P_test', 500, help='size of test set per task')
parser.add('N', 100, help='hidden layer width')
parser.add('n_tasks', 2, help='number of tasks in the sequence')
parser.add('resample', 1, help='boolean variable')
parser.add('task_type', 'permuted', help='permuted/split')
parser.add('permutation', 1.0, help='how much permutation to use between tasks; 1.0=full permutation')
parser.add('depth', 1, help='num of hidden layers')
parser.add('dataset', 'mnist', help='dataset to use: mnist/cifar/fashion')

parser.add('eta', 1.0, help='learning rate')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 1.0, help='weight variance')
parser.add('minibatch', 1, help='size of minibatch for SGD; -1 = full batch')
parser.add('seed', 0, help='random seed')
parser.add('l2', 0.0, help='l2 regularizer')
parser.add('decay', 0.0, help='weight decay speed')

parser.add('n_epochs', 1, help='number of times to go through the sequence of tasks')
parser.add('n_steps', 50000, help='number of learning steps')
args = parser.parse_args()

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/',
                              run_name=run_name, only_print=not ON_CLUSTER)
logger.log(str(args))
logger.log(f'gradient descent update interval: {UPDATE_FREQ} steps')
logger.log(f'gradient descent convergence: {CONVERGENCE_THRESHOLD} steps')
logger.log(f'network normalization scheme: {NORMALIZATION_SCHEME}')
results = {'args': args}

# Use the same seed for sampling the dataset etc.
if DIFF_DATA_SEED:
    args.data_seed = args.seed
    torch.manual_seed(args.seed)
else:
    args.data_seed = 0
    torch.manual_seed(0)

args.nn_seed = args.seed

if args.task_type == 'permuted':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_permuted_dataset(args.n_tasks, args.P, args.P_test, dataset_name=args.dataset, resample=False,
                                      permutation=args.permutation, data_path=data_path, precision=32)
elif args.task_type == 'split':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_split_dataset(args.P, args.P_test, dataset_name=args.dataset,
                                   data_path=data_path, precision=32, n_tasks=args.n_tasks)
else:
    raise ValueError('task type not understood. Choose between "permuted" and "split"')

# Some formatting
seq_of_train_x = seq_of_train_x.to(device)
seq_of_test_x = seq_of_test_x.to(device)
seq_of_train_y = seq_of_train_y.to(device)
seq_of_test_y = seq_of_test_y.to(device)

# now switch to the args seed
torch.manual_seed(args.seed)

# generate the network
network = grad_utils.MLP(seq_of_train_x.shape[-1], args.N,
                         depth=args.depth, output_dim=1, sigma=args.sigma, normalize=NORMALIZATION_SCHEME)
network = network.to(device)


train_losses, test_losses, train_accs, test_accs, sampled_outputs =\
      grad_utils.train_on_sequence(network, seq_of_train_x, seq_of_test_x,
                                   seq_of_train_y, seq_of_test_y,
                                   learning_rate=args.eta, num_steps=args.n_steps, l2=args.l2,
                                   temp=args.T, update_freq=UPDATE_FREQ, logger=logger,
                                   convergence_threshold=CONVERGENCE_THRESHOLD, decay=args.decay,
                                   minibatch=args.minibatch)


results['test acc'] = test_accs
results['train acc'] = train_accs
results['train loss'] = train_losses
results['test loss'] = test_losses
results['sampled outputs'] = sampled_outputs


if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
