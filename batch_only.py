#%%
import numpy as np
import tqdm, utils, theory, pickle, cluster_utils, data, torch

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()
parser = cluster_utils.Args()
parser.add('P', 25)  # size of each training set
parser.add('P_test', 500)  # size of each testing set
parser.add('n_tasks', 2, help='number of tasks in the sequence')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('depth', 2, help='num of hidden layers. setting depth=0 would use the input kernel')
parser.add('seed', 0, help='random seed')
parser.add('fixed_w', 1, help='whether or not to fix weights. ignored for linear networks')
parser.add('lambda_val', 1e4, help='lambda')
parser.add('dataset', 'cifar', help='dataset to use: mnist/cifar')
parser.add('permutation', 0, help='whether or not to permute the dataset')
parser.add('n_epochs', 1, help='number of times to go through the sequence of tasks')
args = parser.parse_args()

args.P = np.min([args.P, int(50000 / args.n_tasks)])
args.P_test = np.min([args.P, int(10000 / args.n_tasks)])

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/',
                              run_name=run_name, only_print=not ON_CLUSTER)
logger.log(str(args))
# P = 4000
# P_test = 100
# n_tasks = 3
# T = 0
# sigma = 0.2
# student_type = 'relu'
# lamb = 1e4
# SEED = 2

torch.manual_seed(args.seed)
print(data_path)
seq_of_train_x, seq_of_test_x, seq_of_train_y_digit, seq_of_test_y_digit = \
    data.prepare_sequential_dataset(args.n_tasks, args.P, args.P_test, dataset_name=args.dataset, resample=True,
                                    permutation=bool(args.permutation), data_path=data_path, n_epochs=args.n_epochs)

logger.log('starting to compute batch accuracy')
# test accuracy using all the training data in one batch
batch_acc = torch.zeros(args.n_tasks)
for task_ind in range(args.n_tasks):
    merged_train_x = seq_of_train_x[:task_ind+1].reshape(-1, seq_of_train_x.shape[-1])
    merged_train_y = seq_of_train_y_digit[:task_ind+1].flatten()
    merged_test_x = seq_of_test_x[:task_ind+1].reshape(-1, seq_of_test_x.shape[-1])
    merged_test_y = seq_of_test_y_digit[:task_ind+1].flatten()

    batch_acc[task_ind] = theory.multihead_one_task_accuracy(train_x=merged_train_x, train_y=merged_train_y,
                                                             test_x=merged_test_x, test_y=merged_test_y,
                                                             depth=args.depth)

logger.log('finished computing batch accuracy')

if ON_CLUSTER:
    results = {'args': args,
               'batch acc': batch_acc}

    logger.finish(results)
