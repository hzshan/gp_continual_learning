#%%
import os
TEST_MODE = False
output_home_path = '/n/home11/haozheshan/ContinualLearning2022/outputs/'
data_path = '/n/home11/haozheshan/ContinualLearning2022/'
if TEST_MODE:
    data_path = None
else:
    os.chdir(data_path)


import numpy as np
import tqdm, utils, theory, pickle, cluster_utils, data


parser = cluster_utils.Args()
parser.add('P', 25)  # size of each training set
parser.add('P_test', 500)  # size of each testing set
parser.add('n_tasks', 2, help='number of tasks in the sequence')
parser.add('N0', 1000, help='input dim')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('depth', 2, help='num of hidden layers. setting depth=0 would use the input kernel')
parser.add('seed', 0, help='random seed')
parser.add('fixed_w', 1, help='whether or not to fix weights. ignored for linear networks')
parser.add('rotation', 0.0, help='whether or not to rotate the center of the input distribution')
parser.add('dist', 0.0, help='distance of the center of the input distribution from the origin')
parser.add('teacher_speed', 1.0, help='speed of teacher changing')
parser.add('cycles', 0, help='number of cycles of a periodic teacher')
args = parser.parse_args()

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/',
                              run_name=run_name, only_print=TEST_MODE)
logger.log(str(args))

lamb_values = 10 ** np.linspace(0, 5, 30)
# lamb_values = np.linspace(50, 2000, 30)

test_loss_all_tasks = np.zeros((len(lamb_values), args.n_tasks, args.n_tasks))
# only computes the test loss for each distribution AFTER all tasks are learnt
training_loss_all_tasks = np.zeros((len(lamb_values), args.n_tasks, args.n_tasks))

test_acc_all_tasks = np.zeros((len(lamb_values), args.n_tasks, args.n_tasks))
# lambda_index, seed_index, time_index, time_index


def prepare_student_teacher_data():
    train_x, test_x = utils.generate_rotating_inputs(args.P, args.P_test, args.N0, args.n_tasks,
                                                     rotation=args.rotation, distance=args.dist)
    train_Y, test_Y = utils.get_teacher_labels(train_x, test_x, max_scaler=args.teacher_speed, cycles=args.cycles)
    # train_Y, test_Y = utils.get_linear_teacher_labels(train_x, test_x, max_scaler=teacher_speed)
    return train_x, test_x, train_Y, test_Y

torch.manual_seed(args.seed)

train_x, test_x, train_Y, test_Y = prepare_student_teacher_data()

seed_avg_batch_test = theory.one_task_loss(train_x=np.vstack(train_x), test_x=test_x[-1],
                                           train_y=np.vstack(train_Y), test_y=test_Y[-1], depth=args.depth)
seed_avg_last_task_only_test = theory.one_task_loss(train_x=train_x[-1], test_x=test_x[-1],
                                                    train_y=train_Y[-1], test_y=test_Y[-1], depth=args.depth)

# ============== compute variance using lambda=1 ==========
mean_train_variance = np.zeros((args.n_tasks, args.n_tasks))
mean_test_variance = np.zeros((args.n_tasks, args.n_tasks))

# training_variances, test_variances, = \
#     theory.compute_predictor_variances(seq_of_train_x=train_x,
#                                        w_var=args.sigma ** 2, P_test=args.P_test,
#                                        lambda_val=1,
#                                        seq_of_test_x=test_x,
#                                        fix_weights=args.fixed_w, depth=args.depth)

# for i in range(len(train_x)):
#     for j in range(len(train_x)):
#         mean_train_variance[i, j] = np.mean(np.diag(training_variances[i, j]))
#
# for i in range(test_variances.shape[0]):
#     for j in range(test_variances.shape[1]):
#         mean_test_variance[i, j] = np.mean(np.diag(test_variances[i, j]))
# ============== compute variance using lambda=1 ==========
for sweep_ind in tqdm.trange(len(lamb_values)):

    lamb = lamb_values[sweep_ind]

    # training_predictions is a 4d tensor.
    # the first three indices (i,j,k) run from 1 to n_tasks and refer to f_i(x_j, w_k)
    # test_predictions is a list with length = number of test sets.
    training_predictions, test_predictions =\
        theory.compute_mean_predictions(seq_of_train_x=train_x, seq_of_train_y=train_Y,
                                        w_var=args.sigma**2, P_test=args.P_test,
                                        lambda_val=lamb, seq_of_test_x=test_x,
                                        fix_weights=args.fixed_w, disable_tqdm=True, depth=args.depth)


    for task_ind in range(args.n_tasks):
        if test_Y[task_ind] is not None:
            for time_ind in range(args.n_tasks):
                test_loss_all_tasks[sweep_ind, task_ind, time_ind] =\
                    utils.loss_from_predictions(test_predictions[task_ind][time_ind], test_Y[task_ind])
                test_acc_all_tasks[sweep_ind, task_ind, time_ind] =\
                    np.mean(np.sign(test_predictions[task_ind][time_ind]) == np.sign(test_Y[task_ind]))

    for time_ind in range(args.n_tasks):
        for task_ind in range(args.n_tasks):
            training_loss_all_tasks[sweep_ind, time_ind, task_ind] =\
                utils.loss_from_predictions(training_predictions[task_ind, time_ind], train_Y[task_ind])

if not TEST_MODE:
    results = {'args': args,
               'test loss': test_loss_all_tasks,
               'test var': mean_test_variance,
               'train var': mean_train_variance,
               'train loss': training_loss_all_tasks,
               'test acc': test_acc_all_tasks,
               'avg batch test': seed_avg_batch_test,
               'last task test': seed_avg_last_task_only_test}

    logger.finish(results)
