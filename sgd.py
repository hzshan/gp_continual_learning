import numpy as np
import tqdm, utils, theory, pickle, cluster_utils, data, torch, sgd_utils, math, sys
import torch.nn.functional as F

"""
Updated Aug 5 2022
Do gradient-based simulations (SGD, Langevin etc.) of continual learning.
Saves: all the training/test acc/loss. 
For the NN predictions, only the output from the 0th head is saved.
"""

device = torch.device('cuda')
ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()
parser = cluster_utils.Args()
parser.add('P', 10, help='size of training set per task')
parser.add('P_test', 50, help='size of test set per task')
parser.add('N', 20, help='hidden layer width')
parser.add('n_tasks', 2, help='number of tasks in the sequence')
parser.add('eta', 0.1, help='learning rate')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('depth', 3, help='num of hidden layers. setting depth=0 would use the input kernel')
parser.add('seed', 0, help='random seed')
parser.add('l2', 0.01, help='l2 regularizer')
parser.add('dataset', 'mnist', help='dataset to use: mnist/cifar')
parser.add('permutation', 1.0, help='whether or not to permute the dataset')
parser.add('n_epochs', 1, help='number of times to go through the sequence of tasks')
parser.add('n_steps', 5000, help='number of SGD steps')
args = parser.parse_args()

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/',
                              run_name=run_name, only_print=not ON_CLUSTER)
logger.log(str(args))
results = {'args': args}

# Use the same seed for sampling the dataset etc.
torch.manual_seed(0)
seq_of_train_x, seq_of_test_x, seq_of_train_y_digit, seq_of_test_y_digit = \
    data.prepare_sequential_dataset(args.n_tasks, args.P, args.P_test, dataset_name=args.dataset, resample=True,
                                    permutation=bool(args.permutation), data_path=data_path, n_epochs=args.n_epochs,
                                    precision=32)

seq_of_train_y_onehot = [F.one_hot(digit.long(), num_classes=10) for digit in seq_of_train_y_digit]
seq_of_test_y_onehot = [F.one_hot(digit.long(), num_classes=10) for digit in seq_of_test_y_digit]

seq_of_train_x = seq_of_train_x.to(device)
seq_of_test_x = seq_of_test_x.to(device)
seq_of_train_y_onehot = torch.stack(seq_of_train_y_onehot).to(device)
seq_of_test_y_onehot = torch.stack(seq_of_test_y_onehot).to(device)


# now switch to the args seed
torch.manual_seed(args.seed)

# generate the network
network = sgd_utils.MLP(seq_of_train_x.shape[-1], args.N,
                        depth=args.depth, n_heads=10, sigma=args.sigma)
network = network.to(device)

train_losses, test_losses, train_accs, test_accs, sampled_outputs =\
      sgd_utils.train_on_sequence(network, seq_of_train_x, seq_of_test_x,
                                  seq_of_train_y_onehot, seq_of_test_y_onehot,
                                  learning_rate=args.eta, num_steps=args.n_steps, l2=args.l2,
                                  temp=args.T, update_freq=10000, logger=logger)


results['test acc'] = test_accs
results['train acc'] = train_accs
results['train loss'] = train_losses
results['test loss'] = test_losses
results['sampled outputs'] = sampled_outputs


if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
