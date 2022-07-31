import numpy as np
import tqdm, utils, theory, pickle, cluster_utils, data, torch, sys, sgd_utils
import torch.nn.functional as F


device = torch.device('mps')
ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()
parser = cluster_utils.Args()
parser.add('P', 1000, help='size of training set per task')
parser.add('P_test', 500, help='size of test set per task')
parser.add('N', 2000, help='hidden layer width')
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


args.P = np.min([args.P, int(50000 / args.n_tasks)])
args.P_test = np.min([args.P, int(10000 / args.n_tasks)])

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/',
                              run_name=run_name, only_print=not ON_CLUSTER)
logger.log(str(args))
results = {'args': args}

torch.manual_seed(args.seed)
seq_of_train_x, seq_of_test_x, seq_of_train_y_digit, seq_of_test_y_digit = \
    data.prepare_sequential_dataset(args.n_tasks, args.P, args.P_test, dataset_name=args.dataset, resample=True,
                                    permutation=bool(args.permutation), data_path=data_path, n_epochs=args.n_epochs,
                                    precision=32)

seq_of_train_y_digit = [F.one_hot(digit.long(), num_classes=10) for digit in seq_of_train_y_digit]
seq_of_test_y_digit = [F.one_hot(digit.long(), num_classes=10) for digit in seq_of_test_y_digit]


network = sgd_utils.MLP(seq_of_train_x.shape[-1], args.N, depth=args.depth, n_heads=10, sigma=args.sigma)
network = network.to(device)


# train(network, seq_of_train_x[0], F.one_hot(seq_of_train_y_digit[0].long(), num_classes=10), eta=1)
# train(network, seq_of_train_x[1], F.one_hot(seq_of_train_y_digit[1].long(), num_classes=10), eta=1)
# train(network, seq_of_train_x[2], F.one_hot(seq_of_train_y_digit[2].long(), num_classes=10), eta=1)

train_losses, test_losses, train_accs, test_accs =\
    sgd_utils.train_on_sequence(network, seq_of_train_x, seq_of_test_x, seq_of_train_y_digit, seq_of_test_y_digit,
                                learning_rate=args.eta, num_steps=args.n_steps, l2=args.l2,
                                update_freq=50)


results['test acc'] = test_accs
results['train acc'] = train_accs
results['train loss'] = train_losses
results['test loss'] = test_losses


if ON_CLUSTER:

    logger.finish(results)
    sys.exit()