import cluster_utils


class GPRealDataArgsParser(cluster_utils.Args):

    def __init__(self):
        super().__init__()
        self.add('P', 100)  # size of each training set
        self.add('P_test', 200)  # size of each testing set
        self.add('n_tasks', 2, help='number of tasks in the sequence')
        self.add('depth', 1,
                help='num of hidden layers.'
                'setting depth=0 would use the input kernel')

        self.add('lambda_val', 1e5, help='lambda')
        self.add('use_large_lambda_limit', 0,
                help='whether to assume infinite lambda.'
                'this makes calculations substantially faster.')
        
        # task type configs
        self.add('manipulation_ratio', 0.0,
                help='if using permutation, 1.0=full permulation; if' +
                ' using split, 1.0=full split.')
        self.add('resample', 0, help='boolean variable')
        self.add('task_type', 'permuted', help='permuted/split')
        self.add('dataset', 'mnist', help='mnist/cifar/fashion/cifar100')

        # utility configs
        self.add('seed', 0, help='random seed')
        self.add('save_outputs', 0, help='1/0'
                '; decides whether or not to save predictions on datasets;'
                ' takes up a lot of disk space')

        # These are not expected to differ in current analysis
        self.add('whiten', 1, help='1/0. Whether to whiten data first.')
        self.add('naive_gp', 0, help='1/0')
        self.add('T', 0.0, help='temperature')
        self.add('sigma', 0.2, help='weight variance')
        self.add('N0context', 0, help='embedding dimension')
        self.add('context_strength', 1.0,
                help='magnifying factor for context embedding')
        self.add('only_first_task', 0, help='1/0. Whether to only save loss on the first task')

class StudentTeacherArgsParser(cluster_utils.Args):

    def __init__(self):
        super().__init__()
        self.add('P', 100)  # size of each training set
        self.add('P_test', 100)  # size of each testing set
        self.add('n_tasks', 2, help='number of tasks in the sequence')
        self.add('T', 0.0, help='temperature')
        self.add('sigma', 0.2, help='weight variance')
        self.add('N0', 100, help='input_dimension')
        self.add('Nh', 100, help='hidden layer width of teachers')
        self.add('NC', 100, help='number of input clusters')
        self.add('radius', 0., help='relative radius of input clusters')
        self.add('tsim', 10, help='similarity between teachers, IN PERCENTAGE')
        self.add('xsim', 10, help='similarity between inputs, IN PERCENTAGE')
        self.add('depth', 1,
                    help='num of hidden layers.'
                    'setting depth=0 would use the input kernel')
        self.add('NSEEDS', 10, help='number of random seeds')
        self.add('lambda_val', 1e5, help='lambda')
        self.add('N0context', 0, help='embedding dimension')
        self.add('context_strength', 1.0,
                help='magnifying factor for context embedding')
        self.add('change_w_in_teachers', 0.0,
                help='whether to vary hidden layer weights of teacher NNs')
        self.add('train_data_has_var', 1.0,
                help='whether training data has deviations from cluster center')
        

class GradientDescentArgsParser(cluster_utils.Args):
    def __init__(self):
        super().__init__()
        self.add('P', 100, help='size of training set per task')
        self.add('P_test', 500, help='size of test set per task')
        self.add('N', 100, help='hidden layer width')
        self.add('n_tasks', 2, help='number of tasks in the sequence')
        self.add('resample', 1, help='boolean variable')
        self.add('task_type', 'permuted', help='permuted/split')
        self.add('manipulation_ratio', 0.0,
                help='if using permutation, 1.0=full permulation; if' +
                ' using split, 1.0=full split.')
        self.add('depth', 1, help='num of hidden layers')
        self.add('dataset', 'mnist', help='dataset to use: mnist/cifar/fashion')

        self.add('eta', 1.0, help='learning rate')
        self.add('T', 0.0, help='temperature')
        self.add('sigma', 1.0, help='weight variance')
        self.add('minibatch', 1, help='size of minibatch for SGD; -1 = full batch')
        self.add('seed', 0, help='random seed')
        self.add('l2', 0.0, help='l2 regularizer')
        self.add('decay', 0.0, help='weight decay speed')

        self.add('n_epochs', 1, help='number of times to go through the sequence of tasks')
        self.add('n_steps', 50000, help='number of learning steps')
        self.add('whiten', 1, help='1/0. Whether to whiten data first.')
