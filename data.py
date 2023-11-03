import torchvision, utils, torch, math
import numpy as np
from torchvision import transforms
import tqdm, theory, warnings
import torch.nn.functional as F
from torch.utils.data import DataLoader
"""
Utility functions for working with MNIST, CIFAR-10 etc.
Also includes code for generating Gaussian mixture ("cluster") data.
"""

def add_replay_items(seq_of_x, seq_of_y, p_replay):
    """
    This turns seq_of_x, which is typically a torch tensor object, into a list object, to accomodate the fact that
    different datasets have different numbers of items
    """
    new_x = seq_of_x.clone()
    new_y = seq_of_y.clone()
    for i in range(len(new_x)):
        if i == 0:
            continue
        else:
            rand_inds = np.random.randint(0, seq_of_x.shape[1], p_replay)
            new_x[i][rand_inds] = new_x[i-1][rand_inds]
            new_y[i][rand_inds] = new_y[i-1][rand_inds]
    return new_x, new_y


def add_task_embedding(seq_of_train_x, seq_of_test_x, embedding_dim, strength=100):


    num_tasks = len(seq_of_train_x)
    assert num_tasks == len(seq_of_test_x)
    p_train = seq_of_train_x.shape[1]
    p_test = seq_of_test_x.shape[1]
    

    if embedding_dim > 0:
        strength_factor = np.sqrt(seq_of_train_x.shape[-1] / embedding_dim) * strength / 100
        embeddings = torch.normal(torch.zeros((num_tasks, embedding_dim))) * strength_factor
        new_train_x = [torch.hstack((seq_of_train_x[i], embeddings[i].repeat(p_train, 1))) for i in range(num_tasks)]
        new_test_x = [torch.hstack((seq_of_test_x[i], embeddings[i].repeat(p_test, 1))) for i in range(num_tasks)]
        return utils.normalize_input(torch.stack(new_train_x)), utils.normalize_input(torch.stack(new_test_x))
    else:
        return seq_of_train_x, seq_of_test_x


def get_clustered_input(num_train_per_cluster, num_test_per_cluster,
                        num_cluster, relative_radius, input_dim,
                        num_datasets, input_similarity,
                        share_variability=True):

    def _generate_centers(n_cluster, n0, n_train_per_cluster,
                          n_test_per_cluster, rel_radius):
        centers = torch.normal(torch.zeros((n_cluster, n0)))
        return centers.repeat(n_train_per_cluster, 1) * np.sqrt(1 - rel_radius), \
            centers.repeat(n_test_per_cluster, 1) * np.sqrt(1 - rel_radius)

    ref_tr_center, ref_te_center =\
        _generate_centers(n_cluster=num_cluster, n0=input_dim,
                          n_train_per_cluster=num_train_per_cluster,
                          n_test_per_cluster=num_test_per_cluster,
                          rel_radius=relative_radius)

    train_datasets = []
    test_datasets = []

    if share_variability:
        # in this case, deviations from cluster centers are shared across
        # datasets. This ensures that when input_similarity = 1, all the 
        # datasets are identical.
        deviations_from_center_tr = torch.normal(
            mean=torch.zeros_like(ref_tr_center),
            std=np.sqrt(relative_radius))

        deviations_from_center_te = torch.normal(
            mean=torch.zeros_like(ref_te_center),
            std=np.sqrt(relative_radius))

    for i in range(num_datasets):

        if not share_variability:
            deviations_from_center_tr = torch.normal(
                mean=torch.zeros_like(ref_tr_center),
                std=np.sqrt(relative_radius))

            deviations_from_center_te = torch.normal(
                mean=torch.zeros_like(ref_te_center),
                std=np.sqrt(relative_radius))

        del_tr_center, del_te_center =\
            _generate_centers(n_cluster=num_cluster, n0=input_dim,
                              n_train_per_cluster=num_train_per_cluster,
                              n_test_per_cluster=num_test_per_cluster,
                              rel_radius=relative_radius)

        tr_center = np.sqrt(input_similarity) * ref_tr_center + np.sqrt(1 - input_similarity) * del_tr_center
        te_center = np.sqrt(input_similarity) * ref_te_center + np.sqrt(1 - input_similarity) * del_te_center

        train_datasets.append(tr_center + deviations_from_center_tr)
        test_datasets.append(te_center + deviations_from_center_te)

    train_datasets = [utils.normalize_input(x) for x in train_datasets]
    test_datasets = [utils.normalize_input(x) for x in test_datasets]

    return train_datasets, test_datasets


class ReluTeachers:
    """Set of two-layer ReLU teacher networks.

    The scalar output has a fixed bias, which makes sure that the mean output on a unit Gaussian input is zero.
    
    Attributes:
        input_dim: N0
        hidden_dim: N
        teacher_similarity: pearson correlation between parameters in different teacher networks
        num_teachers: number of teacher networks to generate
        accumulate: whether to accumulate changes in parameters across teachers
        device: torch device
        same_weight: whether to use the same hidden-layer weight for all teachers.
    """
    def __init__(self,
                input_dim,
                hidden_dim,
                teacher_similarity,
                num_teachers=5,
                accumulate=False,
                device=None,
                same_weight=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Ws = []  # list of first-layer weight matrices
        self.As = []  # list of second-layer readout weights
        self.biases = []  # list of scalar biases for teacher outputs.
        self.num_teachers = num_teachers
        self.accumulative_change = accumulate

        # unit Gaussian input, used to compute the mean output and set to zero
        test_input = torch.normal(torch.zeros((input_dim * 5, input_dim)))  

        # the way parameters are generated such that any pair of parameters has the same correlation (not Markovian)
        W_ref = torch.normal(torch.zeros((input_dim, hidden_dim)))
        A_ref = torch.normal(torch.zeros((hidden_dim, 1)))

        for i in range(num_teachers):
            if i == 0 or self.accumulative_change == False:

                if same_weight:
                    self.Ws.append(W_ref)
                else:
                    self.Ws.append(np.sqrt(teacher_similarity) * W_ref +
                                np.sqrt(1 - teacher_similarity) * torch.normal(torch.zeros((input_dim, hidden_dim))))
                self.As.append(
                    np.sqrt(teacher_similarity) * A_ref +
                    np.sqrt(1 - teacher_similarity) * torch.normal(torch.zeros((hidden_dim, 1))))
            else:
                if same_weight:
                    self.Ws.append(W_ref)
                else:
                    self.Ws.append(np.sqrt(teacher_similarity) * self.Ws[-1] +
                                np.sqrt(1 - teacher_similarity) * torch.normal(torch.zeros((input_dim, hidden_dim))))
                self.As.append(
                    np.sqrt(teacher_similarity) * self.As[-1] +
                    np.sqrt(1 - teacher_similarity) * torch.normal(torch.zeros((hidden_dim, 1))))  

        # add a bias term to the output such that the mean output for input on the unit sphere is 0
        for i in range(num_teachers):
            test_output = torch.relu(test_input @ self.Ws[i]) @ self.As[i] / np.sqrt(self.input_dim * self.hidden_dim)
            self.biases.append(torch.mean(test_output))

        if device is not None:
            self.Ws = [w.to(device) for w in self.Ws]
            self.As = [a.to(device) for a in self.As]
            self.biases = [b.to(device) for b in self.biases]

    def teacher(self, x, teacher_ind):
        assert teacher_ind < self.num_teachers, 'Teacher index exceeds the number of teachers generated.'
        return torch.relu(x @ self.Ws[teacher_ind]) @ self.As[teacher_ind] / np.sqrt(self.input_dim * self.hidden_dim) -\
            self.biases[teacher_ind]


def prepare_cluster_dataset(num_tasks: int,
                            train_p: int, test_p: int,
                            num_clusters: int,
                            input_dim: int, hidden_dim: int, relative_radius: float,
                            teacher_similarity: float, input_similarity: float,
                            accumulate: False,
                            precision=64,
                            device=None,
                            input_share_variability=True,
                            teacher_change_weights=False):
    """Generate toy datasets and teacher-generated labels.

    Each dataset has several Gaussian clusters.
    
    num_tasks: number of tasks. Each "task" has a unique teacher.
    train_p: number of training samples per task.
    test_p: number of test samples per task.
    num_clusters: number of cluster centers.
    input_dim: dimension of input (N0).
    hidden_dim: dimension of hidden layer (N).
    relative_radius: radius of each cluster relative to the distance between cluster centers.
    teacher_similarity: pearson correlation between parameters in different teacher networks.
        (accepts non-negative values)
    input_similarity: pearson correlation between cluster centers in different datasets.
        (accepts non-negative values)
    accumulate: whether to accumulate changes in parameters across teachers.
    precision: precision of floating point numbers.
    device: torch device.
    input_share_variability: whether to share the deviations from cluster centers across datasets.
    teacher_change_weights: whether to change the weights of the hidden layer across teachers.
    """
    all_x_train, all_x_test = get_clustered_input(
        num_train_per_cluster=int(np.ceil(train_p / num_clusters)),
        num_test_per_cluster=int(np.ceil(test_p / num_clusters)),
        num_cluster=num_clusters,
        relative_radius=relative_radius,
        input_dim=input_dim,
        num_datasets=num_tasks,
        input_similarity=input_similarity,
        share_variability=input_share_variability)

    if device is not None:
        all_x_train = [x.to(device) for x in all_x_train]
        all_x_test = [x.to(device) for x in all_x_test]

    assert 0 <= teacher_similarity <= 1, 'Teacher similarity must be between 0 and 1.'
    assert 0 <= input_similarity <= 1, 'Input similarity must be between 0 and 1.'

    teachers = ReluTeachers(input_dim,
                            hidden_dim,
                            teacher_similarity,
                            num_teachers=num_tasks,
                            accumulate=accumulate,
                            device=device,
                            same_weight=not teacher_change_weights)

    all_y_train = []
    all_y_test = []
    for i in range(num_tasks):
        raw_teacher_preds_tr = teachers.teacher(all_x_train[i], teacher_ind=i)
        scalar = np.sqrt(train_p) / torch.norm(raw_teacher_preds_tr)  # add a scalar to make the norm of the output 1
        raw_teacher_preds_te = teachers.teacher(all_x_test[i], teacher_ind=i)
        all_y_train.append(raw_teacher_preds_tr * scalar)
        all_y_test.append(raw_teacher_preds_te * scalar)

    return _pack_data(all_x_train, precision),\
        _pack_data(all_x_test, precision),\
        _pack_data(all_y_train, precision),\
        _pack_data(all_y_test, precision)


def get_loss_acc(tr_preds, te_preds, tr_y, te_y, only_first_task=False):
    _n_tasks = len(tr_y)
    tr_loss = np.zeros((_n_tasks, _n_tasks))
    te_loss = np.zeros((_n_tasks, _n_tasks))
    tr_acc = np.zeros((_n_tasks, _n_tasks))
    te_acc = np.zeros((_n_tasks, _n_tasks))
    # only computes the test loss for each distribution AFTER all tasks are learnt
    for _task_ind in range(_n_tasks):
        if te_y[_task_ind] is not None:
            for _time_ind in range(_n_tasks):
                tr_loss[_task_ind, _time_ind] = \
                    utils.loss_from_predictions(tr_preds[_task_ind][_time_ind], tr_y[_task_ind])
                if te_preds is not None:
                    te_loss[_task_ind, _time_ind] = \
                        utils.loss_from_predictions(te_preds[_task_ind][_time_ind], te_y[_task_ind])

                tr_acc[_task_ind, _time_ind] = \
                    torch.mean(
                        (torch.sign(tr_preds[_task_ind, _time_ind]) == tr_y[_task_ind]).float())
                if te_preds is not None:
                    te_acc[_task_ind, _time_ind] = \
                        torch.mean((torch.sign(te_preds[_task_ind, _time_ind]) == te_y[_task_ind]).float())
    
    if only_first_task:
        return tr_loss[0], te_loss[0], tr_acc[0], te_acc[0]
    else:
        return tr_loss, te_loss, tr_acc, te_acc


def digit_to_onehot(digit_target, num_classes=10):
    return torch.unsqueeze(F.one_hot(digit_target.long(), num_classes=num_classes).double(), -1)


def batch_digit_to_onehot(digit_target, num_classes=10):
    """
    digit target should have shape N_TASKS x P
    """
    return torch.stack([digit_to_onehot(subset, num_classes=num_classes) for subset in digit_target])


def get_loss_mat(net_predictions, seq_of_onehot_targets):
    num_tasks = net_predictions.shape[0]
    loss_mat = torch.zeros((num_tasks, num_tasks))
    for task_ind in range(num_tasks):
        for time_ind in range(num_tasks):
            loss_mat[task_ind, time_ind] =\
                torch.mean((net_predictions[task_ind, time_ind].flatten() -
                            seq_of_onehot_targets[task_ind].flatten())**2)
    return loss_mat


def _pack_data(list_of_data_obs: list, precision=32):
    """
    Convert a list of data observations into a tensor with a desired precision
    """
    if precision == 16:
        return torch.stack(list_of_data_obs).half()
    elif precision == 32:
        return torch.stack(list_of_data_obs).float()
    elif precision == 64:
        return torch.stack(list_of_data_obs).double()
    else:
        raise ValueError('Precision is not understood. Need to be 16/32/64')
    

def _load_and_two_classify_dataset(dataset_name, data_path, whitening):
    """
    Load a multiway classification dataset and split it into two classes
    according to the parity of the label index
    """
    all_train_x, all_test_x, all_train_digits, all_test_digits = \
    load_dataset(dataset_name=dataset_name,
                 num_train=200000,
                 num_test=200000,
                 path=data_path,
                 whitening=whitening)

    all_class_1_train_x = all_train_x[all_train_digits % 2 == 1]
    all_class_2_train_x = all_train_x[all_train_digits % 2 == 0]
    all_class_1_test_x = all_test_x[all_test_digits % 2 == 1]
    all_class_2_test_x = all_test_x[all_test_digits % 2 == 0]

    return (all_class_1_train_x, all_class_2_train_x,
            all_class_1_test_x, all_class_2_test_x)


def _generate_permuted_dataset_from_two_classes(all_class_1_train_x, all_class_2_train_x, all_class_1_test_x, all_class_2_test_x,
                                                 permutation, num_tasks, train_p, test_p, resample=False, precision=64):
    seq_of_train_x = []
    seq_of_test_x = []
    seq_of_train_y = []
    seq_of_test_y = []

    N0 = all_class_1_train_x.shape[-1]

    for task_ind in range(num_tasks):
        p_digit_train = int(train_p / 2)
        p_digit_test = int(test_p / 2)

        if resample:
            class1_train_x = all_class_1_train_x[p_digit_train * task_ind:p_digit_train * (1 + task_ind)]
            class2_train_x = all_class_2_train_x[p_digit_train * task_ind:p_digit_train * (1 + task_ind)]
            class1_test_x = all_class_1_test_x[p_digit_test * task_ind:p_digit_test * (1 + task_ind)]
            class2_test_x = all_class_2_test_x[p_digit_test * task_ind:p_digit_test * (1 + task_ind)]

        else:
            class1_train_x = all_class_1_train_x[:p_digit_train]
            class2_train_x = all_class_2_train_x[:p_digit_train]
            class1_test_x = all_class_1_test_x[:p_digit_test]
            class2_test_x = all_class_2_test_x[:p_digit_test]

        assert class1_train_x.shape[0] == p_digit_train
        assert class2_train_x.shape[0] == p_digit_train
        assert class1_test_x.shape[0] == p_digit_test
        assert class2_test_x.shape[0] == p_digit_test

        fused_train_x = utils.normalize_input(torch.vstack((class1_train_x, class2_train_x)))
        fused_test_x = utils.normalize_input(torch.vstack((class1_test_x, class2_test_x)))

        if permutation > 0:
            perm_mat = utils.get_permutation_mat(N0, strength=permutation)
            fused_train_x = fused_train_x @ perm_mat
            fused_test_x = fused_test_x @ perm_mat
        train_y = torch.vstack((torch.ones((p_digit_train, 1)), torch.ones((p_digit_train, 1)) * -1))
        test_y = torch.vstack((torch.ones((p_digit_test, 1)), torch.ones((p_digit_test, 1)) * -1))

        seq_of_train_x.append(fused_train_x)
        seq_of_test_x.append(fused_test_x)
        seq_of_train_y.append(train_y)
        seq_of_test_y.append(test_y)

    return _pack_data(seq_of_train_x, precision), _pack_data(seq_of_test_x, precision), \
           _pack_data(seq_of_train_y, precision), _pack_data(seq_of_test_y, precision)


def prepare_permuted_dataset(num_tasks: int,
                             train_p: int,
                             test_p: int,
                             dataset_name: str,
                             data_path=None,
                             precision=32,
                             resample=True,
                             permutation=1,
                             whitening=False):
    """
    This prepares a binary permuted task. Odd vs even digits
    """

    (all_class_1_train_x,
    all_class_2_train_x,
    all_class_1_test_x,
    all_class_2_test_x) = _load_and_two_classify_dataset(
        dataset_name, data_path, whitening)

    return _generate_permuted_dataset_from_two_classes(all_class_1_train_x,
                                                       all_class_2_train_x,
                                                       all_class_1_test_x,
                                                       all_class_2_test_x,
                                                       permutation=permutation,
                                                       num_tasks=num_tasks,
                                                       train_p=train_p,
                                                       test_p=test_p,
                                                       resample=resample,
                                                       precision=precision)


def _generate_split_dataset_from_loaded_data(all_train_x,
                                            all_test_x,
                                            all_train_digits,
                                            all_test_digits,
                                            n_tasks, train_p,
                                            test_p,
                                            precision=64):

    seq_of_train_x = []
    seq_of_test_x = []
    seq_of_train_y = []
    seq_of_test_y = []

    total_num_class = len(np.unique(all_test_digits))
    num_tasks = np.min([int(total_num_class / 2), n_tasks])

    # this determines the order of the tasks. the first two inds are the first task, second two are the second task etc.
    task_order = torch.arange(total_num_class)
    task_order = task_order[torch.randperm(len(task_order))]
    # print(task_order)

    for task_ind in range(num_tasks):
        # select train/test x with a certain label

        # number of examples per digit
        p_digit_train = int(train_p / 2)
        p_digit_test = int(test_p / 2)

        class1_train_x = all_train_x[all_train_digits == task_order[task_ind * 2]][:p_digit_train]
        class2_train_x = all_train_x[all_train_digits == task_order[task_ind * 2 + 1]][:p_digit_train]
        class1_test_x = all_test_x[all_test_digits == task_order[task_ind * 2]][:p_digit_test]
        class2_test_x = all_test_x[all_test_digits == task_order[task_ind * 2 + 1]][:p_digit_test]

        assert class1_train_x.shape[0] == p_digit_train, f'{class1_train_x.shape[0]} != {p_digit_train} for task {task_ind}'
        assert class2_train_x.shape[0] == p_digit_train, f'{class2_train_x.shape[0]} != {p_digit_train} for task {task_ind}'
        assert class1_test_x.shape[0] == p_digit_test, f'{class1_test_x.shape[0]} != {p_digit_test} for task {task_ind}'
        assert class2_test_x.shape[0] == p_digit_test, f'{class2_test_x.shape[0]} != {p_digit_test} for task {task_ind}'

        fused_train_x = utils.normalize_input(torch.vstack((class1_train_x, class2_train_x)))
        fused_test_x = utils.normalize_input(torch.vstack((class1_test_x, class2_test_x)))
        train_y = torch.vstack((torch.ones((p_digit_train, 1)), torch.ones((p_digit_train, 1)) * -1))
        test_y = torch.vstack((torch.ones((p_digit_test, 1)), torch.ones((p_digit_test, 1)) * -1))

        seq_of_train_x.append(fused_train_x)
        seq_of_test_x.append(fused_test_x)
        seq_of_train_y.append(train_y)
        seq_of_test_y.append(test_y)

    return _pack_data(seq_of_train_x, precision), _pack_data(seq_of_test_x, precision), \
           _pack_data(seq_of_train_y, precision), _pack_data(seq_of_test_y, precision)


def prepare_split_dataset(train_p: int,
                          test_p: int,
                          dataset_name: str,
                          data_path=None,
                          precision=32,
                          n_tasks=5,
                          whitening=False):
    """
    The n_tasks parameter sets the maximum number of tasks to use. The other constraint is the number of classes in the
    full dataset. If there are K classes (e.g., K=100 for CIFAR-100), then at most one can make 50 split tasks.
    """

    # load the entire dataset first
    all_train_x, all_test_x, all_train_digits, all_test_digits = \
        load_dataset(
            dataset_name=dataset_name,
            num_train=200000,
            num_test=200000,
            path=data_path,
            whitening=whitening)

    return _generate_split_dataset_from_loaded_data(
        all_train_x,
        all_test_x,
        all_train_digits,
        all_test_digits,
        n_tasks,
        train_p,
        test_p,
        precision=precision)


def whiten(input_arr, epsilon=0):
    """
    Compute and apply a N0-by-N0 whitening matrix to the input array.

    Args:
        input_arr: an numpy array or torch tensor of shape P-by-N0,
        where P is the number of samples.
        epsilon: regularization parameter for the covariance matrix.
    
    Return:
        whitened input array, and the whitening matrix.
    """
    if input_arr.dtype in [np.float32, np.float64]:
        cov = np.cov(input_arr.T)
        u, s, _ = np.linalg.svd(cov)
        whiten_mat = u @ np.diag((s + epsilon)**-0.5) @ u.T
        return (input_arr @ whiten_mat), whiten_mat
    else:
        # if not a numpy array, try torch
        cov = torch.cov(input_arr.T)
        u, s, _ = torch.svd(cov)
        whiten_mat = u @ torch.diag((s + epsilon)**-0.5) @ u.T
        return (input_arr @ whiten_mat), whiten_mat


def load_dataset(dataset_name: str,
                 num_train: int,
                 num_test: int,
                 path=None,
                 mean_subtraction='image',
                 whitening=False):
    if path is None:
        path = '/Users/haozheshan/Dropbox/codes/gp_continual_learning/datasets'

    assert dataset_name in ['cifar', 'mnist', 'fashion', 'cifar100', 'emnist'],\
          'dataset name not understood'

    train_set = None
    test_set = None

    trans = transforms.Compose([transforms.PILToTensor(),
                        transforms.ConvertImageDtype(torch.float64)])

    if dataset_name == 'cifar':
        train_set = torchvision.datasets.CIFAR10(path, train=True, download=False, transform=trans)
        test_set = torchvision.datasets.CIFAR10(path, train=False, download=False, transform=trans)
    elif dataset_name == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=trans)
        test_set = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=trans)
    elif dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST(path, train=True, download=True, transform=trans)
        test_set = torchvision.datasets.MNIST(path, train=False, download=True, transform=trans)
    elif dataset_name == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(path, train=True, download=True, transform=trans)
        test_set = torchvision.datasets.FashionMNIST(path, train=False, download=True, transform=trans)
    elif dataset_name == 'emnist':
        train_set = torchvision.datasets.EMNIST(path, 'byclass', train=True, download=True, transform=trans)
        test_set = torchvision.datasets.EMNIST(path, 'byclass', train=False, download=True, transform=trans)

    train_loader = DataLoader(train_set, batch_size=num_train, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=num_test, shuffle=False)

    raw_train_data = next(iter(train_loader))
    raw_test_data = next(iter(test_loader))
    

    def _get_x_y(raw_data: tuple, mean_subtraction):
        x, y = raw_data
        x = x.reshape(x.shape[0], -1).float()

        assert mean_subtraction in ['image', 'batch']
        if mean_subtraction == 'batch':
            # subtract the mean pixel value across positions AND images
            # each image would have a different mean
            x -= x.mean()
        else:
            # subtract the mean pixel value across positions.
            # each image would have zero mean
            x -= torch.mean(x, dim=1).reshape(-1, 1)
        y = y.float()
        return x, y

    train_x, train_y = _get_x_y(raw_train_data, mean_subtraction)
    test_x, test_y = _get_x_y(raw_test_data, mean_subtraction)

    if whitening:
        train_x, whitened_mat = whiten(train_x)
        test_x = test_x @ whitened_mat
    train_x = utils.normalize_input(train_x)
    test_x = utils.normalize_input(test_x)

    return train_x, test_x, train_y, test_y
