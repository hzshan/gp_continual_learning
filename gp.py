"""
GP-limit continual learning theory for two-way classification problems.
Every random seed corresponds to a different subset of data.

"""

import theory, cluster_utils, data, torch, sys, configs

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()

parser = configs.GPRealDataArgsParser()
args = parser.parse_args()

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(
    output_path=f'{output_home_path}{args.BATCH_NAME}/',
    run_name=run_name, only_print=not ON_CLUSTER)

logger.log(str(args))  # log the arguments
results = {'args': args}  # store results in a dictionary

# Use the same seed for sampling the dataset etc.
torch.manual_seed(args.seed)
args.data_seed = args.seed  # for consistency with other scripts

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


# add task embedding (if N0context = 0, it does nothing)
seq_of_train_x, seq_of_test_x = data.add_task_embedding(
    seq_of_train_x,
    seq_of_test_x,
    args.N0context,
    args.context_strength)


# compute theory of the mean predictor on all training/test sets
training_predictions, test_predictions =\
    theory.compute_mean_predictions(
        seq_of_train_x=seq_of_train_x, seq_of_train_y=seq_of_train_y,
        w_var=args.sigma**2,
        lambda_val=args.lambda_val, seq_of_test_x=seq_of_test_x,
        large_lambda=bool(args.use_large_lambda_limit), depth=args.depth,
        use_naive_gp=(bool(args.naive_gp)))


# compute MSE loss and classification accuracy; store in dictionary
(results['train loss'], results['test loss'],
 results['train acc'], results['test acc']) =\
    data.get_loss_acc(training_predictions,
                      test_predictions,
                      seq_of_train_y,
                      seq_of_test_y,
                      only_first_task=args.only_first_task)


results['single task test loss'] = theory.get_single_task_test_losses(
    seq_of_train_x, seq_of_train_y, seq_of_test_x, seq_of_test_y,
    depth=args.depth
    )


if bool(args.save_outputs):
    # this saves the exact predictions on the first training set across time.
    # enabling this significantly increases the output file size

    results['sampled fn on train'] = training_predictions[0, :]
    # save output on train set #1 across time
    results['sampled fn on test'] = test_predictions[0, :]
    # save output on test set #1 across time


# compute some OPs for long-term behavior
# The notations are somewhat different from the paper. 
# v1v2_cos = cos(V1, V2) = \gamma_{rule}
# trp1p2 = tr(P1P2)/P = \gamma_{input}

trp1p2, v1v2_cos, _ = theory.compute_forgetting_ops(
    x1=seq_of_train_x[0], x2=seq_of_train_x[1],
    y1=seq_of_train_y[0], y2=seq_of_train_y[1],
    depth=args.depth)

trp1p2_ntk, v1v2_cos_ntk, _ = theory.compute_forgetting_ops(
    x1=seq_of_train_x[0], x2=seq_of_train_x[1],
    y1=seq_of_train_y[0], y2=seq_of_train_y[1],
    depth=args.depth, use_ntk_kernel=True)

dec2024_gamma_rf, dec2024_gamma_r = theory.compute_dec2024_ops(
    x1=seq_of_train_x[0], x2=seq_of_train_x[1],
    y1=seq_of_train_y[0], y2=seq_of_train_y[1],
    depth=args.depth)

def get_gamma_3(x1, x2, y1, depth):
    k1 = theory.arccos_kernel_deep(x1, x1, depth)
    k12 = theory.arccos_kernel_deep(x1, x2, depth)
    k2 = theory.arccos_kernel_deep(x2, x2, depth)

    k1_inv = torch.inverse(k1)

    v_sq_norm = y1.T @ k1_inv @ y1

    return y1.T @ k1_inv @ k12 @ torch.inverse(k2) @ k12.T @ k1_inv @ y1 / v_sq_norm

results['Nov2024 gamma3'] = get_gamma_3(seq_of_train_x[0], seq_of_train_x[1], seq_of_train_y[0], args.depth)

results['V1-V2'] = 2 - 2 * v1v2_cos
results['tr(P1P2)/P'] = trp1p2

results['V1-V2 ntk'] = 2 - 2 * v1v2_cos_ntk
results['tr(P1P2)/P ntk'] = trp1p2_ntk

results['Dec2024 gamma_rf'] = dec2024_gamma_rf
results['Dec2024 gamma_r'] = dec2024_gamma_r

if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
