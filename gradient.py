import cluster_utils, data, torch, grad_utils, sys, configs

UPDATE_FREQ = 5000  
# print an update per UPDATE_FREQ steps

CONVERGENCE_THRESHOLD = 1000  
# stop training once the loss hasn't decreased for this many steps

NORMALIZATION_SCHEME = 'none'
# whether or not to use layer norm etc.

DIFF_DATA_SEED = False 
# whether or not to use a different seed for the data; 
# if False, the seed used to sample data is always 0

TARGET_TRAIN_LOSS = 1e-3

"""
Do gradient-based simulations (SGD, Langevin etc.) of continual learning.
Saves: all the training/test acc/loss.
For the NN predictions, only the output from the 0th head is saved,
and the corresponding input is always the first test set!!
"""

# detect whether we are on the cluster or not; use cuda if on cluster
ON_CLUSTER, data_path, output_home_path = cluster_utils.initialize()
device = torch.device('cuda') if ON_CLUSTER else torch.device('cpu')


args = configs.GradientDescentArgsParser().parse_args()

run_name = f'{args.BATCH_NAME}_{args.TRIAL_IND}'

logger = cluster_utils.Logger(
    output_path=f'{output_home_path}{args.BATCH_NAME}/',
    run_name=run_name, only_print=not ON_CLUSTER)

# log some training parameters
logger.log(str(args))
logger.log(f'gradient descent update interval: {UPDATE_FREQ} steps')
logger.log(f'gradient descent convergence: {CONVERGENCE_THRESHOLD} steps')
logger.log(f'network normalization scheme: {NORMALIZATION_SCHEME}')
results = {'args': args}

# Use the same seed for sampling the dataset etc.
if DIFF_DATA_SEED:
    args.data_seed = args.seed
    torch.manual_seed(args.seed)
else:
    args.data_seed = 0
    torch.manual_seed(0)

args.nn_seed = args.seed

if args.task_type == 'permuted':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_permutation_sequence(
            args.n_tasks,
            args.P,
            args.P_test,
            dataset_name=args.dataset,
            resample=False,
            permutation=args.manipulation_ratio,
            data_path=data_path,
            precision=32)

elif args.task_type == 'split':
    seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y = \
        data.prepare_split_sequence(
            args.P,
            args.P_test,
            dataset_name=args.dataset,
            data_path=data_path,
            precision=32,
            n_tasks=args.n_tasks)
else:
    raise ValueError('task type not understood.'
                     'Choose between "permuted" and "split"')

# Some formatting
seq_of_train_x = seq_of_train_x.to(device)
seq_of_test_x = seq_of_test_x.to(device)
seq_of_train_y = seq_of_train_y.to(device)
seq_of_test_y = seq_of_test_y.to(device)

# now switch to the args seed
torch.manual_seed(args.seed)

# generate the network
network = grad_utils.MLP(
    seq_of_train_x.shape[-1],
    args.N,
    depth=args.depth,
    output_dim=1,
    sigma=args.sigma,
    normalize=NORMALIZATION_SCHEME)

network = network.to(device)

(train_losses, test_losses, train_accs, test_accs,
  sampled_fn_on_train, sampled_fn_on_test) =\
      grad_utils.train_on_sequence(
          network, seq_of_train_x, seq_of_test_x,
          seq_of_train_y, seq_of_test_y,
          learning_rate=args.eta, num_steps=args.n_steps, l2=args.l2,
          temp=args.T, update_freq=UPDATE_FREQ, logger=logger,
          convergence_threshold=CONVERGENCE_THRESHOLD, decay=args.decay,
          minibatch=args.minibatch, target_train_loss=TARGET_TRAIN_LOSS)


results['test acc'] = test_accs
results['train acc'] = train_accs
results['train loss'] = train_losses
results['test loss'] = test_losses
results['sampled fn on train'] = sampled_fn_on_train
results['sampled fn on test'] = sampled_fn_on_test

if ON_CLUSTER:

    logger.finish(results)
    sys.exit()
