#%%
import numpy as np
import matplotlib.pyplot as plt
import tqdm, utils, theory, pickle, os, torch, torchvision, cluster_utils
from numpy.linalg import inv as inv

output_path = '/n/home11/haozheshan/ContinualLearning2022/outputs/'


parser = cluster_utils.Args()
parser.add('P', 100)  # size of each training set
parser.add('P_test', 100)  # size of each testing set
parser.add('n_tasks', 10, help='number of tasks in the sequence')
parser.add('N0', 100, help='input dimension')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('student_type', 'relu', help='architecture of the network. choose between linear and relu')
parser.add('NUM_SEEDS', 5, help='number of random seeds')
parser.add('fixed_w', True, help='whether or not to fix weights. ignored for linear networks')
parser.add('teacher_speed', 0.0, help='speed of the shifting teacher')
parser.add('input_rotation', np.pi, help="speed of rotation of the center of the input distribution")
parser.add('input_dist', 100, help='distance of the input distribution from the origin')
args = parser.parse_args()

#%%

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'


lamb_values = 10 ** np.linspace(1, 4, 50)
# lamb_values = np.linspace(50, 2000, 30)

test_loss_all_tasks = np.zeros((len(lamb_values), args.NUM_SEEDS, args.n_tasks))
# only computes the test loss for each distribution AFTER all tasks are learnt
training_loss_all_tasks = np.zeros((len(lamb_values), args.NUM_SEEDS, args.n_tasks, args.n_tasks))
# lambda_index, seed_index, time_index, time_index


def prepare_student_teacher_data():
    train_x, test_x = utils.generate_rotating_inputs(args.P, args.P_test, args.N0, args.n_tasks,
                                                     resample=True, total_angle=args.input_rotation,
                                                     distance=args.input_dist, only_test_last=False)
    train_Y, test_Y = utils.get_teacher_labels(train_x, test_x, max_scaler=args.teacher_speed)
    # train_Y, test_Y = utils.get_linear_teacher_labels(train_x, test_x, max_scaler=args.teacher_speed)
    return train_x, test_x, train_Y, test_Y


# measure the test error ON THE LAST TASK from some special schemes
seed_avg_batch_test = 0
seed_avg_last_task_only_test = 0


for seed in range(args.NUM_SEEDS):
    np.random.seed(seed)

    train_x, test_x, train_Y, test_Y = prepare_student_teacher_data()
    # train_x, test_x, train_Y, test_Y = utils.prepare_permuted_mnist_data(P=P, P_test=P_test, n_tasks=n_tasks)
    # train_x, test_x, train_Y, test_Y = utils.prepare_even_odd_mnist_data(P=P, P_test=P_test)

    seed_avg_batch_test += theory.simple_kernel_loss(train_x=np.vstack(train_x), test_x=test_x[-1],
                                                     train_Y=np.vstack(train_Y), test_Y=test_Y[-1],
                                                     student_type=args.student_type) / args.NUM_SEEDS
    seed_avg_last_task_only_test += theory.simple_kernel_loss(train_x=train_x[-1], test_x=test_x[-1],
                                                              train_Y=train_Y[-1], test_Y=test_Y[-1],
                                                              student_type=args.student_type) / args.NUM_SEEDS

    for sweep_ind in tqdm.trange(len(lamb_values)):

        lamb = lamb_values[sweep_ind]

        # training_predictions is a 4d tensor.
        # the first three indices (i,j,k) run from 1 to n_tasks and refer to f_i(x_j, w_k)
        # test_predictions is a list with length = number of test sets.
        training_predictions, test_predictions =\
            theory.compute_mean_predictions(train_inputs=train_x, train_targets=train_Y,
                                            w_var=args.sigma**2, P_test=args.P_test,
                                            lambda_val=lamb,
                                            test_inputs=test_x,
                                            fix_weights=args.fixed_w,
                                            student_type=args.student_type)

        for task_ind in range(args.n_tasks):
            if test_Y[task_ind] is not None:
                test_loss_all_tasks[sweep_ind, seed, task_ind] =\
                    utils.loss_from_predictions(test_predictions[task_ind][-1], test_Y[task_ind])

        for time_ind in range(args.n_tasks):
            for task_ind in range(args.n_tasks):
                training_loss_all_tasks[sweep_ind, seed, time_ind, task_ind] =\
                    utils.loss_from_predictions(training_predictions[task_ind, time_ind], train_Y[task_ind])


results = {'args': args,
           'lambda': lamb_values,
           'test loss': test_loss_all_tasks,
           'train loss': training_loss_all_tasks}

result_name = output_path + f'{run_name}.results'
pickle.dump(results, open(result_name, 'wb'))

