#%%
"""
Continual learning theory using the student-teacher setup.
1. Be sure to use 64 bit precision
2. Every job contains MULTIPLE random seeds. This streamlines the job submission process.
"""
SAVE_ACCURACY = False

import numpy as np
import theory, cluster_utils, data, torch, sys, configs

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()

parser = configs.StudentTeacherArgsParser()
parser.add('use_large_lambda_limit', 0,
            help='whether to assume infinite lambda.'
           'this makes calculations substantially faster.')
args = parser.parse_args()

large_lambda = bool(args.use_large_lambda_limit)

args.tsim = float(args.tsim / 100)
args.xsim = float(args.xsim / 100)

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(
    output_path=f'{output_home_path}{args.BATCH_NAME}/',
    run_name=run_name, only_print=not ON_CLUSTER)

logger.log(str(args))
results = {'args': args}
 
for key in ['train loss', 'test loss', 'train acc',
            'test acc', 'train loss zero lambda', 'test loss zero lambda',
            'train acc zero lambda', 'test acc zero lambda',
            'tr(P1P2)/P', 'V1-V2']:
    results[key] = []

# Use the same seed for sampling the dataset etc.
for seed in range(args.NSEEDS):
    logger.log(f'starting random seed #{seed}')
    torch.manual_seed(seed)

    (seq_of_train_x,  # T x P x N0
     seq_of_test_x,  # T x P_test x N0
     seq_of_train_y, # T x P x 1
     seq_of_test_y,  # T x P_test x 1
     ) =\
        data.prepare_cluster_dataset(
            num_tasks=args.n_tasks,
            train_p=args.P,
            test_p=args.P_test,
            num_clusters=args.NC,
            input_dim=args.N0,
            hidden_dim=args.Nh,
            relative_radius=args.radius,
            teacher_similarity=args.tsim,
            input_similarity=args.xsim,
            accumulate=False,
            precision=64,
            teacher_change_weights=bool(args.change_w_in_teachers),
            train_data_has_var=bool(args.train_data_has_var),
            )

    seq_of_train_x, seq_of_test_x = data.add_task_embedding(
        seq_of_train_x,
        seq_of_test_x,
        args.N0context,
        args.context_strength)

    training_predictions, test_predictions =\
        theory.compute_mean_predictions(seq_of_train_x=seq_of_train_x,
                                        seq_of_train_y=seq_of_train_y,
                                        w_var=args.sigma**2, 
                                        lambda_val=args.lambda_val,
                                        seq_of_test_x=seq_of_test_x,
                                        large_lambda=large_lambda,
                                        depth=args.depth)

    training_predictions_zero_lambda, test_predictions_zero_lambda =\
        theory.compute_mean_predictions(seq_of_train_x=seq_of_train_x,
                                        seq_of_train_y=seq_of_train_y,
                                        w_var=args.sigma**2, 
                                        lambda_val=0.01,
                                        seq_of_test_x=seq_of_test_x,
                                        large_lambda=False,
                                        depth=args.depth)


    train_loss, test_loss, train_acc, test_acc =\
        data.get_loss_acc(training_predictions, test_predictions,
                        seq_of_train_y, seq_of_test_y)

    (train_loss_zero_lambda, test_loss_zero_lambda,
     train_acc_zero_lambda, test_acc_zero_lambda) =\
        data.get_loss_acc(training_predictions_zero_lambda,
                          test_predictions_zero_lambda,
                          seq_of_train_y, seq_of_test_y)

    results['train loss'].append(train_loss)
    results['test loss'].append(test_loss)
    results['train acc'].append(train_acc)
    results['test acc'].append(test_acc)

    results['train loss zero lambda'].append(train_loss_zero_lambda)
    results['test loss zero lambda'].append(test_loss_zero_lambda)
    results['train acc zero lambda'].append(train_acc_zero_lambda)
    results['test acc zero lambda'].append(test_acc_zero_lambda)

    # compute some order parameters
    trp1p2, v1v2cos, v1v2cos_ref = theory.compute_forgetting_ops(
        x1=seq_of_train_x[0], x2=seq_of_train_x[1],
        y1=seq_of_train_y[0], y2=seq_of_train_y[1],
        depth=args.depth)

    results['tr(P1P2)/P'].append(trp1p2)
    results['V1-V2'].append(2 - 2 * v1v2cos)

for key in ['train loss', 'test loss', 'train acc', 'test acc',
            'train loss zero lambda',
            'test loss zero lambda',
            'train acc zero lambda',
            'test acc zero lambda']:
    results[key] = np.stack(results[key])

    if SAVE_ACCURACY is False:
        if 'acc' in key:
            results[key] = None

if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
