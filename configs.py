import cluster_utils


class StudentTeacherArgsParser(cluster_utils.Args):

    def __init__(self):
        super().__init__()
        self.add('P', 50)  # size of each training set
        self.add('P_test', 100)  # size of each testing set
        self.add('n_tasks', 2, help='number of tasks in the sequence')
        self.add('T', 0.0, help='temperature')
        self.add('sigma', 0.2, help='weight variance')
        self.add('N0', 100, help='input_dimension')
        self.add('Nh', 100, help='hidden layer width of teachers')
        self.add('NC', 10, help='number of input clusters')
        self.add('radius', 0.1, help='relative radius of input clusters')
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
        self.add('change_w_in_teachers', 1.0,
                help='whether to vary hidden layer weights of teacher NNs')