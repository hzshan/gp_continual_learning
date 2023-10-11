#%%
"""
Continual learning theory using the student-teacher setup.
1. Be sure to use 64 bit precision
2. Every job contains MULTIPLE random seeds. This streamlines the job submission process.
"""
import numpy as np
import theory, cluster_utils, data, torch, sys

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()

parser = cluster_utils.Args()
parser.add('P', 50)  # size of each training set
parser.add('P_test', 10)  # size of each testing set
parser.add('n_tasks', 2, help='number of tasks in the sequence')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('N0', 100, help='input_dimension')
parser.add('Nh', 100, help='hidden layer width of teachers')
parser.add('NC', 10, help='number of input clusters')
parser.add('radius', 0.1, help='relative radius of input clusters')
parser.add('tsim', 10, help='similarity between teachers, IN PERCENTAGE')
parser.add('xsim', 10, help='similarity between inputs, IN PERCENTAGE')
parser.add('depth', 1, help='num of hidden layers. setting depth=0 would use the input kernel')
parser.add('NSEEDS', 5, help='number of random seeds')
parser.add('lambda_val', 1e5, help='lambda')
parser.add('N0context', 100, help='embedding dimension')
parser.add('context_strength', 1.0, help='magnifying factor for context embedding')
args = parser.parse_args()

args.tsim = float(args.tsim / 100)
args.xsim = float(args.xsim / 100)

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/',
                              run_name=run_name, only_print=not ON_CLUSTER)
logger.log(str(args))
results = {'args': args}

for key in ['train loss', 'test loss', 'train acc', 'test acc', 'train loss naive', 'test loss naive', 'train acc naive', 'test acc naive',
            'train magnitude', 'train magnitude naive', 'tr(P1P2)/P', 'V1-V2']:
    results[key] = []

# Use the same seed for sampling the dataset etc.
for seed in range(args.NSEEDS):
    logger.log(f'starting random seed #{seed}')
    torch.manual_seed(seed)

    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y =\
        data.prepare_cluster_dataset(num_tasks=args.n_tasks,
                                    train_p=args.P,
                                    test_p=args.P_test,
                                    num_clusters=args.NC,
                                    input_dim=args.N0,
                                    hidden_dim=args.Nh,
                                    relative_radius=args.radius,
                                    teacher_similarity=args.tsim,
                                    input_similarity=args.xsim,
                                    accumulate=False,
                                    precision=64)

    seq_of_train_x, seq_of_test_x = data.add_task_embedding(seq_of_train_x, seq_of_test_x, args.N0context, args.context_strength)

    training_predictions, test_predictions =\
        theory.compute_mean_predictions(seq_of_train_x=seq_of_train_x, seq_of_train_y=seq_of_train_y,
                                        w_var=args.sigma**2, P_test=args.P_test,
                                        lambda_val=args.lambda_val, seq_of_test_x=seq_of_test_x,
                                        large_lambda=False, depth=args.depth)

    training_predictions_naive, test_predictions_naive =\
        theory.compute_mean_predictions(seq_of_train_x=seq_of_train_x, seq_of_train_y=seq_of_train_y,
                                        w_var=args.sigma**2, P_test=args.P_test,
                                        lambda_val=args.lambda_val, seq_of_test_x=seq_of_test_x,
                                        large_lambda=False, depth=args.depth, use_naive_gp=True)


    train_loss, test_loss, train_acc, test_acc =\
        data.get_loss_acc(training_predictions, test_predictions,
                        seq_of_train_y, seq_of_test_y)

    train_loss_naive, test_loss_naive, train_acc_naive, test_acc_naive =\
        data.get_loss_acc(training_predictions_naive, test_predictions_naive,
                        seq_of_train_y, seq_of_test_y)

    results['train loss'].append(train_loss)
    results['test loss'].append(test_loss)
    results['train acc'].append(train_acc)
    results['test acc'].append(test_acc)

    results['train magnitude'].append(np.linalg.norm(training_predictions.squeeze(), axis=-1)**2 / args.P)

    results['train loss naive'].append(train_loss_naive)
    results['test loss naive'].append(test_loss_naive)
    results['train acc naive'].append(train_acc_naive)
    results['test acc naive'].append(test_acc_naive)

    results['train magnitude naive'].append(np.linalg.norm(training_predictions_naive.squeeze(), axis=-1)**2 / args.P)



    # compute some order parameters
    trp1p2, v1v2cos, v1v2cos_ref = theory.compute_forgetting_ops(
        x1=seq_of_train_x[0], x2=seq_of_train_x[1],
        y1=seq_of_train_y[0], y2=seq_of_train_y[1],
        depth=args.depth)

    results['tr(P1P2)/P'].append(trp1p2)
    results['V1-V2'].append(2 - 2 * v1v2cos)


for key in ['train loss', 'test loss', 'train acc', 'test acc',
            'train loss naive', 'test loss naive', 'train acc naive',
            'test acc naive']:
    results[key] = np.stack(results[key])


if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
