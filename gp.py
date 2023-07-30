"""
Apr 22 2022
GP-limit continual learning theory for two-way classification problems.
1. EVERY RANDOM SEED CORRESPONDS TO A NEW RANDOM DATA SAMPLE!!!
To save compute, naive theory is always computed with a fixed kernel (INF LAMBDA)

"""
SAVE_SAMPLED_OUTPUTS = False
LARGE_LAMBDA = False

import numpy as np
import theory, cluster_utils, data, torch, sys

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()

parser = cluster_utils.Args()
parser.add('P', 500)  # size of each training set
parser.add('P_test', 200)  # size of each testing set
parser.add('n_tasks', 5, help='number of tasks in the sequence')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('permutation', 1.0, help='permutation strength; 1.0=full permulation')
parser.add('resample', 1, help='boolean variable')
parser.add('depth', 1, help='num of hidden layers. setting depth=0 would use the input kernel')
parser.add('seed', 0, help='random seed')
parser.add('lambda_val', 1e5, help='lambda')
parser.add('task_type', 'permuted', help='permuted/split')
parser.add('naive_gp', 0, help='1/0')
parser.add('dataset', 'mnist', help='mnist/cifar/fashion/cifar100')
parser.add('N0context', 0, help='embedding dimension')
parser.add('context_strength', 1.0, help='magnifying factor for context embedding')
parser.add('save_outputs', 0, help='1/0; decides whether or not to save predictions on datasets; takes up a lot of disk space')
args = parser.parse_args()

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/', run_name=run_name, only_print=not ON_CLUSTER)
logger.log(str(args))
results = {'args': args}

# Use the same seed for sampling the dataset etc.
torch.manual_seed(args.seed)
args.data_seed = args.seed

if args.task_type == 'permuted':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_permuted_dataset(args.n_tasks, args.P, args.P_test, dataset_name=args.dataset,
                                      resample=bool(args.resample),
                                      permutation=args.permutation, data_path=data_path, precision=64)
elif args.task_type == 'split':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_split_dataset(args.P, args.P_test, dataset_name=args.dataset,
                                   data_path=data_path, precision=64, n_tasks=args.n_tasks)
else:
    raise ValueError('task type not understood. Choose between "permuted" and "split"')

seq_of_train_x, seq_of_test_x = data.add_task_embedding(seq_of_train_x, seq_of_test_x, args.N0context, args.context_strength)

training_predictions, test_predictions =\
    theory.compute_mean_predictions(seq_of_train_x=seq_of_train_x, seq_of_train_y=seq_of_train_y,
                                    w_var=args.sigma**2,
                                    lambda_val=args.lambda_val, seq_of_test_x=seq_of_test_x,
                                    large_lambda=LARGE_LAMBDA, depth=args.depth, use_naive_gp=(bool(args.naive_gp)))

results['train loss'], results['test loss'], results['train acc'], results['test acc'] =\
    data.get_loss_acc(training_predictions, test_predictions, seq_of_train_y, seq_of_test_y)

results['train magnitude'] = np.linalg.norm(training_predictions.squeeze(), axis=-1)**2 / args.P


if bool(args.save_outputs):
    results['sampled outputs'] = test_predictions[0, :]  # save output on test set #1 across time

# # compute some OPs for long-term behavior using NTK kernels

if bool(args.naive_gp):
    K1 = theory.arccos_kernel_deep(seq_of_train_x[0], seq_of_train_x[0], depth=args.depth)
    K2 = theory.arccos_kernel_deep(seq_of_train_x[1], seq_of_train_x[1], depth=args.depth)
    K12 = theory.arccos_kernel_deep(seq_of_train_x[0], seq_of_train_x[1], depth=args.depth)
else:
    K1 = theory.k_ntk(seq_of_train_x[0], seq_of_train_x[0], depth=args.depth, lamb=args.lambda_val)
    K2 = theory.k_ntk(seq_of_train_x[1], seq_of_train_x[1], depth=args.depth, lamb=args.lambda_val)
    K12 = theory.k_ntk(seq_of_train_x[0], seq_of_train_x[1], depth=args.depth, lamb=args.lambda_val)


results['Vt_Sigma1_Vt'] = args.P**-1 * seq_of_train_y[0].T @ np.linalg.inv(K1) @ K12 @ K12.T @ np.linalg.inv(K1) @ seq_of_train_y[0]
results['V1_Sigma1_Vt'] = args.P**-1 * seq_of_train_y[0].T @ K12 @ np.linalg.inv(K2) @ seq_of_train_y[1]
results['V1_V1'] = args.P**-1 * seq_of_train_y[0].T @ np.linalg.inv(K1) @ seq_of_train_y[0]
results['tr(P1P2)/P'] = np.trace(K12 @ np.linalg.inv(K2) @ K12.T @ np.linalg.inv(K1)) / args.P

if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
