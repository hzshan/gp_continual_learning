#%%
import numpy as np
import tqdm, utils, theory, pickle, cluster_utils, data, sys, torch

ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()
parser = cluster_utils.Args()
parser.add('P', 500)  # size of each training set
parser.add('P_test', 1000)  # size of each testing set
parser.add('n_tasks', 20, help='number of tasks in the sequence')
parser.add('T', 0.0, help='temperature')
parser.add('sigma', 0.2, help='weight variance')
parser.add('depth', 1, help='num of hidden layers. setting depth=0 would use the input kernel')
parser.add('seed', 1, help='random seed')
parser.add('fixed_w', 0, help='whether or not to fix weights. ignored for linear networks')
parser.add('lambda_val', 1e3, help='lambda')
parser.add('dataset', 'mnist', help='dataset to use: mnist/cifar/fashion')
parser.add('permutation', 1.0, help='strength of permutation. 1.0=full permutation')
parser.add('n_epochs', 1, help='number of times to go through the sequence of tasks')
parser.add('resample', 1, help='whether to sample new data points for each task')
parser.add('interpolate', 1, help='if True, linearly interpolate between two permutation matrices')
args = parser.parse_args()

args.P = np.min([args.P, int(50000 / args.n_tasks)])
args.P_test = np.min([args.P, int(10000 / args.n_tasks)])

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(output_path=f'{output_home_path}{args.BATCH_NAME}/',
                              run_name=run_name, only_print=not ON_CLUSTER)
logger.log(str(args))
results = {'args': args}

torch.manual_seed(args.seed)
print(data_path)
seq_of_train_x, seq_of_test_x, seq_of_train_y_digit, seq_of_test_y_digit = \
    data.prepare_sequential_dataset(args.n_tasks, args.P, args.P_test, dataset_name=args.dataset,
                                    resample=bool(args.resample),
                                    permutation=args.permutation, data_path=data_path, n_epochs=args.n_epochs,
                                    interpolate=bool(args.interpolate))
#
# single_seed_train_accuracy, single_seed_test_accuracy, single_seed_train_loss, single_seed_test_loss = \
#     data.get_multihead_accuracy(seq_of_train_x=seq_of_train_x, seq_of_test_x=seq_of_test_x,
#                                 seq_of_train_y_digit=seq_of_train_y_digit, seq_of_test_y_digit=seq_of_test_y_digit,
#                                 sigma=args.sigma, lamb=args.lambda_val,
#                                 fixed_w=bool(args.fixed_w), depth=args.depth,
#                                 write_fn=logger.log)
#
# results['test acc'] = single_seed_test_accuracy
# results['train acc'] = single_seed_train_accuracy
# results['train loss'] = single_seed_train_loss
# results['test loss'] = single_seed_test_loss


# training_variances, test_variances, =\
#     theory.compute_predictor_variances(seq_of_train_x=seq_of_train_x,
#                                        w_var=args.sigma**2, P_test=args.P_test,
#                                        lambda_val=args.lambda_val,
#                                        seq_of_test_x=seq_of_test_x,
#                                        fix_weights=args.fixed_w, depth=args.depth)
# mean_training_variances = np.zeros((len(seq_of_train_x), len(seq_of_train_x)))
# for i in range(len(seq_of_train_x)):
#     for j in range(len(seq_of_train_x)):
#         mean_training_variances[i, j] = np.mean(np.diag(training_variances[i, j]))
#
# mean_test_variances = np.zeros((test_variances.shape[0], test_variances.shape[1]))
# for i in range(test_variances.shape[0]):
#     for j in range(test_variances.shape[1]):
#         mean_test_variances[i, j] = np.mean(np.diag(test_variances[i, j]))
#
# results['test var'] = mean_test_variances
# results['train var'] = mean_training_variances

logger.log('starting to compute batch accuracy')
# test accuracy using all the training data in one batch
batch_acc = torch.zeros(args.n_tasks)
batch_loss = torch.zeros(args.n_tasks)
for task_ind in range(10):
    merged_train_x = torch.flip(seq_of_train_x, dims=[0])[:task_ind+1].reshape(-1, seq_of_train_x.shape[-1])
    merged_train_y = torch.flip(seq_of_train_y_digit, dims=[0])[:task_ind+1].flatten()
    merged_test_x = seq_of_test_x[-1]
    merged_test_y = seq_of_test_y_digit[-1]

    batch_acc[task_ind], batch_loss[task_ind] =\
        theory.multihead_one_task_accuracy(train_x=merged_train_x, train_y=merged_train_y,
                                           test_x=merged_test_x, test_y=merged_test_y,
                                           depth=args.depth)

logger.log('finished computing batch accuracy')

if ON_CLUSTER:

    logger.finish(results)
    sys.exit()

#%%

#%%

one_task_acc, one_task_loss =\
        theory.multihead_one_task_accuracy(train_x=seq_of_train_x[-1], train_y=seq_of_train_y_digit[-1],
                                           test_x=seq_of_test_x[-1], test_y=seq_of_test_y_digit[-1],
                                           depth=args.depth)
import matplotlib.pyplot as plt
plt.figure()
for i in range(args.n_tasks):
    plt.scatter(i + 1, torch.mean(single_seed_test_loss[-1, i]), color='k')
# plt.axhline(batch_acc[-1])
plt.axhline(batch_loss[0])
plt.axhline(one_task_loss, color='r')
# plt.ylim(0, 1)
plt.title(f'permuted Fashion MNIST, depth={args.depth}')
plt.xlabel('number of tasks learned')
plt.ylabel('test accuracy')
plt.show()

#%%
plt.figure()
plt.plot(single_seed_train_loss[0])
plt.plot(single_seed_test_loss[0])
plt.show()


#%%
