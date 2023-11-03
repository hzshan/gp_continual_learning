"""
GP-limit continual learning theory for two-way classification problems.
1. EVERY RANDOM SEED CORRESPONDS TO A NEW RANDOM DATA SAMPLE!!!

"""

import numpy as np
import theory, cluster_utils, data, torch, sys

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()
ONLY_FIRST_TASK = True  # only save loss/accuracy of the first task across time
# this is to reduce the size of the output file

parser = cluster_utils.Args()
parser.add('P', 500)  # size of each training set
parser.add('P_test', 200)  # size of each testing set
parser.add('n_tasks', 5, help='number of tasks in the sequence')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('permutation', 1.0,
           help='permutation strength; 1.0=full permulation')
parser.add('resample', 1, help='boolean variable')
parser.add('depth', 1,
           help='num of hidden layers.'
           'setting depth=0 would use the input kernel')
parser.add('seed', 0, help='random seed')
parser.add('lambda_val', 1e5, help='lambda')
parser.add('use_large_lambda_limit', 0,
            help='whether to assume infinite lambda.'
           'this makes calculations substantially faster.')
parser.add('task_type', 'permuted', help='permuted/split')
parser.add('naive_gp', 0, help='1/0')
parser.add('dataset', 'mnist', help='mnist/cifar/fashion/cifar100')
parser.add('N0context', 0, help='embedding dimension')
parser.add('context_strength', 1.0,
            help='magnifying factor for context embedding')
parser.add('save_outputs', 0, help='1/0'
           '; decides whether or not to save predictions on datasets;'
           ' takes up a lot of disk space')
parser.add('whiten', 0, help='1/0. Whether to whiten data first.')
args = parser.parse_args()

# log whether only saving first task performance
args.only_first_task = ONLY_FIRST_TASK

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(
    output_path=f'{output_home_path}{args.BATCH_NAME}/',
    run_name=run_name,
    only_print=not ON_CLUSTER)

logger.log(str(args))
results = {'args': args}

# Use the same seed for sampling the dataset etc.
torch.manual_seed(args.seed)
args.data_seed = args.seed

if args.task_type == 'permuted':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_permuted_dataset(
            args.n_tasks,
            args.P,
            args.P_test,
            dataset_name=args.dataset,
            resample=bool(args.resample),
            permutation=args.permutation,
            data_path=data_path,
            precision=64,
            whitening=args.whiten,)

elif args.task_type == 'split':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_split_dataset(
            args.P,
            args.P_test,
            dataset_name=args.dataset,
            data_path=data_path,
            precision=64,
            n_tasks=args.n_tasks,
            whitening=args.whiten,)
else:
    raise ValueError(
        'task type not understood. Choose between "permuted" and "split"')

seq_of_train_x, seq_of_test_x = data.add_task_embedding(
    seq_of_train_x,
    seq_of_test_x,
    args.N0context,
    args.context_strength)

training_predictions, test_predictions =\
    theory.compute_mean_predictions(
        seq_of_train_x=seq_of_train_x, seq_of_train_y=seq_of_train_y,
        w_var=args.sigma**2,
        lambda_val=args.lambda_val, seq_of_test_x=seq_of_test_x,
        large_lambda=bool(args.use_large_lambda_limit), depth=args.depth,
        use_naive_gp=(bool(args.naive_gp)))

(results['train loss'], results['test loss'],
 results['train acc'], results['test acc']) =\
    data.get_loss_acc(training_predictions,
                      test_predictions,
                      seq_of_train_y,
                      seq_of_test_y, only_first_task=ONLY_FIRST_TASK)


# results['train magnitude'] = np.linalg.norm(
#     training_predictions.squeeze(), axis=-1)**2 / args.P


if bool(args.save_outputs):
    results['sampled fn on train'] = training_predictions[0, :]
    # save output on train set #1 across time
    results['sampled fn on test'] = test_predictions[0, :]
    # save output on test set #1 across time

# # compute some OPs for long-term behavior
trp1p2, v1v2_cos, _ = theory.compute_forgetting_ops(
    x1=seq_of_train_x[0], x2=seq_of_train_x[1],
    y1=seq_of_train_y[0], y2=seq_of_train_y[1],
    depth=args.depth,
)

trp1p2_ntk, v1v2_cos_ntk, _ = theory.compute_forgetting_ops(
    x1=seq_of_train_x[0], x2=seq_of_train_x[1],
    y1=seq_of_train_y[0], y2=seq_of_train_y[1],
    depth=args.depth, use_ntk_kernel=True
)

results['V1-V2'] = 2 - 2 * v1v2_cos
results['tr(P1P2)/P'] = trp1p2

results['V1-V2 ntk'] = 2 - 2 * v1v2_cos_ntk
results['tr(P1P2)/P ntk'] = trp1p2_ntk

if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
