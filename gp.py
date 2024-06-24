"""
GP-limit continual learning theory for two-way classification problems.
1. EVERY RANDOM SEED CORRESPONDS TO A NEW RANDOM DATA SAMPLE!!!

"""

# only save loss/accuracy of the first task across time
# this is to reduce the size of the output file
ONLY_FIRST_TASK = False


import numpy as np
import theory, cluster_utils, data, torch, sys, configs

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()

parser = configs.GPRealDataArgsParser()
args = parser.parse_args()

# log whether only saving first task performance
args.only_first_task = ONLY_FIRST_TASK

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(
    output_path=f'{output_home_path}{args.BATCH_NAME}/',
    run_name=run_name, only_print=not ON_CLUSTER)

logger.log(str(args))
results = {'args': args}

# Use the same seed for sampling the dataset etc.
torch.manual_seed(args.seed)
args.data_seed = args.seed

if args.task_type == 'permuted':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_permutation_sequence(
            args.n_tasks,
            args.P,
            args.P_test,
            dataset_name=args.dataset,
            resample=bool(args.resample),
            permutation=args.manipulation_ratio,
            data_path=data_path,
            precision=64,
            whitening=args.whiten,)

elif args.task_type == 'split':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_split_sequence(
            args.P,
            args.P_test,
            dataset_name=args.dataset,
            data_path=data_path,
            precision=64,
            n_tasks=args.n_tasks,
            split_ratio=args.manipulation_ratio,
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

results['single task test loss'] = theory.get_single_task_test_losses(
    seq_of_train_x, seq_of_train_y, seq_of_test_x, seq_of_test_y,
    depth=args.depth
    )


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
