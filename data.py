import torchvision, utils, torch, math
import numpy as np
from torchvision import transforms
import tqdm, theory, warnings
import torch.nn.functional as F
from torch.utils.data import DataLoader
"""
Code for generating various task sequences, including
* permuted task sequences
* split task sequences
* student-teacher task sequences (Gaussian mixture data)

it also includes some related utility functions, including functionalities 
not studied in the paper, such as adding task embeddings and replay items.
"""

def permute_with_intermediate_task(
        seq_of_train_x, seq_of_test_x, perm_strength):
    """
    Split one permutation operation into several parts. Apply increasingly
    many parts to each task in the sequence.

    To generate a permutation matrix, we first make an identity matrix, 
    randomly select some of its rows, and scramble them. "inds_to_permute"
    selects the rows to exchange. To apply a part of the permutation, we only 
    scramble a subset of these indices. 
    """
    num_tasks = len(seq_of_train_x)
    n0 = seq_of_train_x.shape[-1]  # input dimension
    assert len(seq_of_test_x) == num_tasks

    
    n0_to_perm = int(n0 * perm_strength)  # total number of pixels to permute

    # number of pixels in each permutation "part"
    n0_perm_per_task = int(n0_to_perm / (num_tasks-1))
    if perm_strength > 0:
        assert n0_perm_per_task > 3, 'Too few pixels to permute'

    # select all pixel indices to be permuted b/t first and last tasks
    inds_to_permute = torch.randperm(n0)[:n0_to_perm]

    # function that permutes a segment in a sequence
    def part_perm(array, start_ind, end_ind):
        if end_ind == start_ind:
            return array
        assert end_ind > start_ind
        perm_inds = array.clone()
        perm_inds[start_ind:end_ind] =\
              perm_inds[start_ind:end_ind][torch.randperm(end_ind - start_ind)]
        return perm_inds
    
    perm_inds = [inds_to_permute]
    for i in range(1, num_tasks):
        # for each task, we scramble a set of indices that wasn't scramled
        # in the previous task
        perm_inds.append(part_perm(perm_inds[-1],
                                   n0_perm_per_task * (i-1),
                                   n0_perm_per_task * i)
                                   )


    perm_mats = [torch.eye(n0).double() for _ in range(num_tasks)]
    for i in range(num_tasks):
        perm_mats[i][inds_to_permute] = perm_mats[i][perm_inds[i]]

    # apply perm mats
    for i in range(num_tasks):
        seq_of_train_x[i] = seq_of_train_x[i] @ perm_mats[i]
        seq_of_test_x[i] = seq_of_test_x[i] @ perm_mats[i]
    
    return seq_of_train_x, seq_of_test_x


def _mix_array(arr1, arr2, similarity):
    """
    Returns sqrt(sim) * arr1 + sqrt(1-sim) * arr2.
    If arr2 is None, add N(0,1) Gaussian noise
    """
    if arr2 is None:
        arr2 = torch.normal(torch.zeros_like(arr1), std=1)
    assert arr1.shape == arr2.shape
    return np.sqrt(similarity) * arr1 + np.sqrt(1 - similarity) * arr2


def add_replay_items(seq_of_x, seq_of_y, p_replay):
    """
    This turns seq_of_x, which is typically a torch tensor object,
    into a list object, to accomodate the fact that
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
        strength_factor = np.sqrt(
            seq_of_train_x.shape[-1] / embedding_dim) * strength / 100
        embeddings = torch.normal(
            torch.zeros((num_tasks, embedding_dim))) * strength_factor
        new_train_x = [
            torch.hstack((seq_of_train_x[i], embeddings[i].repeat(p_train, 1)))\
                  for i in range(num_tasks)]
        new_test_x = [
            torch.hstack((seq_of_test_x[i], embeddings[i].repeat(p_test, 1)))\
                  for i in range(num_tasks)]

        return (
            utils.normalize_input(torch.stack(new_train_x)),
            utils.normalize_input(torch.stack(new_test_x))
            )
    else:
        return seq_of_train_x, seq_of_test_x


def get_clustered_input(num_train_per_cluster, num_test_per_cluster,
                        num_cluster, relative_radius, input_dim,
                        num_datasets, input_similarity,
                        train_data_has_var=True,
                        accumulate_changes=False):
    """
    Generated clustered data for the student-teacher task sequences. 

    Terminology: each "cluster" refers to a Gaussian centered on one template
    Args:
        num_train_per_cluster: number of training samples per cluster.
        num_test_per_cluster: number of test samples per cluster.
        num_cluster: number of clusters.
        relative_radius: radius of each cluster relative to the distance between cluster centers.
        input_dim: dimension of input (N0).
        num_datasets: number of datasets to generate.
        input_similarity: pearson correlation between cluster centers in different datasets.
        train_data_has_var: whether to add deviations from cluster centers to training data.
        accumulate_changes: if true, similarity describes relation between first and last task. 
    """

    # check whether training data are sampled around the mean, or they are the
    # mean. If they are the mean, then num_train_per_cluster should be 1. 
    if train_data_has_var is False:
        assert num_train_per_cluster == 1, \
        'If variability_of_train_data is False, num_train_per_cluster must be 1.'

    def _generate_centers(n_cluster, n0, n_train_per_cluster,
                          n_test_per_cluster, rel_radius):
        centers = torch.normal(torch.zeros((n_cluster, n0)),
                               std=np.sqrt(1 - rel_radius))
        
        # the same centers are repeated for training and test sets. the 
        # number of repeats is the number of examples per cluster
        return centers.repeat(n_train_per_cluster, 1), \
            centers.repeat(n_test_per_cluster, 1)

    
    ref_tr_center, ref_te_center =\
        _generate_centers(n_cluster=num_cluster, n0=input_dim,
                          n_train_per_cluster=num_train_per_cluster,
                          n_test_per_cluster=num_test_per_cluster,
                          rel_radius=relative_radius)
    
    if accumulate_changes:
        # generate centers for the last task
        del_tr_centers, del_te_centers =\
            _generate_centers(n_cluster=num_cluster, n0=input_dim,
                                n_train_per_cluster=num_train_per_cluster,
                                n_test_per_cluster=num_test_per_cluster,
                                rel_radius=relative_radius)
        ref_tr_center_last = _mix_array(ref_tr_center, del_tr_centers, input_similarity)
        ref_te_center_last = _mix_array(ref_te_center, del_te_centers, input_similarity)
    train_datasets = []
    test_datasets = []

    for i in range(num_datasets):
        
        tr_center = None
        te_center = None

        # decide centers for this task
        if accumulate_changes:
            tr_center = _mix_array(ref_tr_center, ref_tr_center_last, 1 - i / (num_datasets - 1))
            te_center = _mix_array(ref_te_center, ref_te_center_last, 1 - i / (num_datasets - 1))
        
        else:
            del_tr_center, del_te_center =\
                _generate_centers(n_cluster=num_cluster, n0=input_dim,
                                n_train_per_cluster=num_train_per_cluster,
                                n_test_per_cluster=num_test_per_cluster,
                                rel_radius=relative_radius)

            tr_center = _mix_array(ref_tr_center, del_tr_center, input_similarity)
            te_center = _mix_array(ref_te_center, del_te_center, input_similarity)

        # generate noise to add to centers
        deviations_from_center_tr = torch.normal(
            mean=torch.zeros_like(ref_tr_center),
            std=np.sqrt(relative_radius))

        deviations_from_center_te = torch.normal(
            mean=torch.zeros_like(ref_te_center),
            std=np.sqrt(relative_radius))

        # only add noise to training data if train_data_has_var is True
        if train_data_has_var:
            train_datasets.append(tr_center + deviations_from_center_tr)
        else:
            train_datasets.append(tr_center)
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
                accumulate_changes=False,
                device=None,
                same_weight=True):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Ws = []  # list of first-layer weight matrices
        self.As = []  # list of second-layer readout weights
        self.biases = []  # list of scalar biases for teacher outputs.
        self.num_teachers = num_teachers
        self.accumulate_changes = accumulate_changes

        # unit Gaussian input, used to compute the mean output and set to zero
        test_input = torch.normal(torch.zeros((input_dim * 5, input_dim)))  

        W_ref = torch.normal(torch.zeros((input_dim, hidden_dim)))
        A_ref = torch.normal(torch.zeros((hidden_dim, 1)))

        if self.accumulate_changes:
            # if accumulating changes over time, then teacher similarity decides
            # relation between the first and last teachers
            W_ref_last = _mix_array(W_ref, None, teacher_similarity)
            A_ref_last = _mix_array(A_ref, None, teacher_similarity)

            # weights for each teacher are mixed from x_ref and x_ref_last
            for i in range(num_teachers):
                self.Ws.append(
                    W_ref if same_weight else _mix_array(
                        W_ref, W_ref_last, 1 - i / (num_teachers - 1)))
                self.As.append(
                    _mix_array(A_ref, A_ref_last, 1 - i / (num_teachers - 1)))

        else:
            for i in range(num_teachers):
                self.Ws.append(
                    W_ref if same_weight else _mix_array(
                        W_ref, None, teacher_similarity))
                
                self.As.append(_mix_array(A_ref, None, teacher_similarity))

        # add a bias term to the output such that the mean output for input on the unit sphere is 0
        for i in range(num_teachers):
            test_output = torch.relu(test_input @ self.Ws[i]) @ self.As[i] /\
                  np.sqrt(self.input_dim * self.hidden_dim)
            self.biases.append(torch.mean(test_output))

        if device is not None:
            self.Ws = [w.to(device) for w in self.Ws]
            self.As = [a.to(device) for a in self.As]
            self.biases = [b.to(device) for b in self.biases]

    def teacher(self, x, teacher_ind):
        assert teacher_ind < self.num_teachers, 'Teacher index exceeds the number of teachers generated.'
        return torch.relu(x @ self.Ws[teacher_ind]) @ self.As[teacher_ind] /\
              np.sqrt(self.input_dim * self.hidden_dim) -\
            self.biases[teacher_ind]


def prepare_cluster_dataset(num_tasks: int,
                            train_p: int, test_p: int,
                            num_clusters: int,
                            input_dim: int, hidden_dim: int, relative_radius: float,
                            teacher_similarity: float, input_similarity: float,
                            accumulate: False,
                            precision=64,
                            device=None,
                            teacher_change_weights=False,
                            train_data_has_var=True):
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
    teacher_change_weights: whether to change the weights of the hidden layer across teachers.
    train_data_has_var: whether to add deviations from cluster centers to training data.
    """
    if train_data_has_var is False:
        assert train_p == num_clusters, \
        'If variability_of_train_data is False, num_train_per_cluster must be 1.'

    all_x_train, all_x_test = get_clustered_input(
        num_train_per_cluster=int(np.ceil(train_p / num_clusters)),
        num_test_per_cluster=int(np.ceil(test_p / num_clusters)),
        num_cluster=num_clusters,
        relative_radius=relative_radius,
        input_dim=input_dim,
        num_datasets=num_tasks,
        input_similarity=input_similarity,
        train_data_has_var=train_data_has_var,
        accumulate_changes=accumulate)

    if device is not None:
        all_x_train = [x.to(device) for x in all_x_train]
        all_x_test = [x.to(device) for x in all_x_test]

    assert 0 <= teacher_similarity <= 1, 'Teacher similarity must be between 0 and 1.'
    assert 0 <= input_similarity <= 1, 'Input similarity must be between 0 and 1.'

    teachers = ReluTeachers(input_dim,
                            hidden_dim,
                            teacher_similarity,
                            num_teachers=num_tasks,
                            accumulate_changes=accumulate,
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


def get_loss_acc(tr_preds, te_preds, tr_y, te_y, only_first_task=False, normalize=True):
    """
    Computes MSE loss and classification accuracy for each task and time step.
    All outputs are in 16 bit precision to reduce output file size.
    """
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
                    utils.loss_from_predictions(
                        tr_preds[_task_ind][_time_ind], tr_y[_task_ind], normalize=normalize)
                if te_preds is not None:
                    te_loss[_task_ind, _time_ind] = \
                        utils.loss_from_predictions(
                            te_preds[_task_ind][_time_ind], te_y[_task_ind], normalize=normalize)

                tr_acc[_task_ind, _time_ind] = \
                    torch.mean(
                        (torch.sign(tr_preds[_task_ind, _time_ind]) == tr_y[_task_ind]).float())
                if te_preds is not None:
                    te_acc[_task_ind, _time_ind] = \
                        torch.mean((torch.sign(te_preds[_task_ind, _time_ind]) == te_y[_task_ind]).float())
    
    if only_first_task:
        return tr_loss[0].astype(np.float16), te_loss[0].astype(np.float16), tr_acc[0].astype(np.float16), te_acc[0].astype(np.float16)
    else:
        return tr_loss.astype(np.float16), te_loss.astype(np.float16), tr_acc.astype(np.float16), te_acc.astype(np.float16)


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


def _generate_permutation_sequence_from_loaded_data(
        all_train_x, all_test_x, all_train_digits, all_test_digits,
        permutation, num_tasks, train_p, test_p, resample=False, precision=64):
    
    all_class_1_train_x = utils.shuffle_along_first_axis(
        all_train_x[all_train_digits % 2 == 1])
    all_class_2_train_x = utils.shuffle_along_first_axis(
        all_train_x[all_train_digits % 2 == 0])
    all_class_1_test_x = utils.shuffle_along_first_axis(
        all_test_x[all_test_digits % 2 == 1])
    all_class_2_test_x = utils.shuffle_along_first_axis(
        all_test_x[all_test_digits % 2 == 0])

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


def prepare_permutation_sequence(num_tasks: int,
                             train_p: int,
                             test_p: int,
                             dataset_name: str,
                             data_path=None,
                             precision=32,
                             resample=True,
                             permutation=1,
                             whitening=False):
    """
    This prepares a sequence of binary classification tasks. Different tasks
    use inputs that are permuted versions of the same images.

    Args:
    num_tasks: number of tasks. Can be as many as desired.
    train_p: number of training samples per task.
    test_p: number of test samples per task.
    dataset_name: name of the dataset. Can be 'mnist', 'fashion', 'cifar', 'emnist'.
    data_path: path to the dataset.
    precision: precision of floating point numbers.
    resample: whether to resample the data for each task. If true, each task
    uses a different subset of the data.
    permutation: strength of the permutation. 0 means no permutation, 1 means
    full permutation.
    whitening: whether to whiten the input data.
    """

    all_train_x, all_test_x, all_train_digits, all_test_digits = \
        load_dataset(
            dataset_name=dataset_name,
            path=data_path,
            whitening=whitening)

    return _generate_permutation_sequence_from_loaded_data(all_train_x,
                                                       all_test_x,
                                                       all_train_digits,
                                                       all_test_digits,
                                                       permutation=permutation,
                                                       num_tasks=num_tasks,
                                                       train_p=train_p,
                                                       test_p=test_p,
                                                       resample=resample,
                                                       precision=precision)


def _generate_split_sequence_from_loaded_data(all_train_x,
                                            all_test_x,
                                            all_train_digits,
                                            all_test_digits,
                                            n_tasks,
                                            train_p,
                                            test_p,
                                            split_ratio=1,
                                            precision=64):
    """
    Each split task has inputs coming from two sources. A fraction
    (1-split_ratio) of them are equally divided among all digits used in this
    task sequence. The rest (split_ratio) are equally divided between two
    specific digits assigned to this task.

    Example: the dataset uses digits 0-9. The first task, which has 0 and 1
    assigned to it, will have (1-split_ratio)
    """

    seq_of_train_x = []
    seq_of_test_x = []
    seq_of_train_y = []
    seq_of_test_y = []

    n_digits_in_dataset = len(np.unique(all_test_digits))
    num_tasks = np.min([int(n_digits_in_dataset / 2), n_tasks])

    # this determines the order of the tasks. the first two inds are the first
    # task, second two are the second task etc.
    task_order = torch.arange(n_digits_in_dataset)
    task_order = task_order[torch.randperm(len(task_order))][:num_tasks * 2]
    p_digit_train = int(train_p / 2)
    p_digit_test = int(test_p / 2)

    def _select_digits_for_each_task(all_x, all_digits,
                                     task_order, task_ind, p_digit):
        """
        From all_x, select p_digit examples for the digit specified by 
        task_order[task_ind * 2] and another p_digit examples for each digit
        specified by task_order[task_ind * 2 + 1].

        Returns: input for each task (P x N0) and labels for each task (P x 1)
        
        Args:
        all_x:  n_examples_in_dataset x N0
        all_y: n_examples_in_dataset x 1, natural numbers
        task_order: a sequence of randomly ordered natural numbers
        task_ind: index of current task
        p_digit: number of examples per digit in each task
        """

        class1_x = all_x[
            all_digits == task_order[task_ind * 2]][:p_digit]
        class2_x = all_x[
            all_digits == task_order[task_ind * 2 + 1]][:p_digit]
        
        assert class1_x.shape[0] == p_digit, f'Expected {p_digit} examples, got {class1_x.shape[0]}'
        assert class2_x.shape[0] == p_digit, f'Expected {p_digit} examples, got {class2_x.shape[0]}'
        
        fused_x = utils.normalize_input(
            torch.vstack((class1_x, class2_x)))
        fused_y = torch.vstack((torch.ones((p_digit, 1)),
                                 torch.ones((p_digit, 1)) * -1))
        
        return fused_x, fused_y


    for task_ind in range(num_tasks):
        # select train/test x with a certain label

        # number of examples per digit

        train_x, train_y = _select_digits_for_each_task(
            all_train_x, all_train_digits, task_order, task_ind, p_digit_train)
        test_x, test_y = _select_digits_for_each_task(
            all_test_x, all_test_digits, task_order, task_ind, p_digit_test)

        seq_of_train_x.append(train_x)
        seq_of_test_x.append(test_x)
        seq_of_train_y.append(train_y)
        seq_of_test_y.append(test_y)

    # to create split sequences with partial splitting, mix data here

    def _mix_split_sequence(seq_of_x, split_ratio):
        """
        Each task has P examples, P/2 from each class.
        Within each class (e.g. the one with +1 label),
        (1-split_ratio)P/2 are evenly split between all digits with this label
        that appear in the sequence (there are T in total), so each digit contributes
        'P_per_digit_to_share' examples to the shared part of each task.
        """
        
        T, P, N0 = seq_of_x.shape

        # in each class there are P/2 examples per task...
        P_per_class = int(P / 2)

        # ...within which P/2 * (1-split_ratio) are the examples shared by all 
        # tasks. These examples are evenly split between all digits with this
        # label, and there are T in total. Therefore, each digit contributes
        # P_per_digit_to_share examples to the shared part of each task.
        P_per_digit_to_share = int(P / 2 * (1 - split_ratio) / T)
        
        # reshape sequence such that examples from the two classes are 
        # separated
        reshaped_seq_of_x = seq_of_x.reshape(T, 2, P_per_class, -1)
        mixed_seq_of_x = reshaped_seq_of_x.clone()
    
        # all shared x for class 1
        shared_class1_tr_x = torch.vstack([
            reshaped_seq_of_x[
                i, 0, :P_per_digit_to_share] for i in range(T)])

        shared_class2_tr_x = torch.vstack([
            reshaped_seq_of_x[
                i, 1, :P_per_digit_to_share] for i in range(T)])

        P_shared_per_class, _ = shared_class1_tr_x.shape

        P_digit_unshared = P_per_class - P_shared_per_class

        for i in range(T):
            # all unshared x for class 1
            unshared_class1_tr_x = reshaped_seq_of_x[
                i, 0, P_per_digit_to_share:(
                    P_per_digit_to_share+P_digit_unshared)]
            unshared_class2_tr_x = reshaped_seq_of_x[
                i, 1, P_per_digit_to_share:(
                    P_per_digit_to_share+P_digit_unshared)]

            mixed_seq_of_x[i, 0] = torch.vstack(
                [shared_class1_tr_x, unshared_class1_tr_x])
            mixed_seq_of_x[i, 1] = torch.vstack(
                [shared_class2_tr_x, unshared_class2_tr_x])
        
        return mixed_seq_of_x.reshape(T, P, N0)

    return (
        _mix_split_sequence(
            _pack_data(seq_of_train_x, precision), split_ratio), 
        _mix_split_sequence(
        _pack_data(seq_of_test_x, precision), split_ratio),
        _pack_data(seq_of_train_y, precision),
        _pack_data(seq_of_test_y, precision))


def prepare_split_sequence(train_p: int,
                          test_p: int,
                          dataset_name: str,
                          data_path=None,
                          precision=32,
                          n_tasks=5,
                          split_ratio=1,
                          whitening=False):
    """
    Generates a sequence of binary classification tasks, where each task uses
    two different classes from the source dataset. E.g., if the source dataset
    is MNIST, the first task may use 0 vs 1, the second 2 vs 3, etc.

    The n_tasks parameter sets the maximum number of tasks to use. The other 
    constraint is the number of classes in the full dataset. If there are K 
    classes (e.g., K=100 for CIFAR-100), then at most one can make 50 split
    tasks.

    Args:
    train_p: number of training samples per task.
    test_p: number of test samples per task.
    dataset_name: name of the dataset. Can be 'mnist', 'fashion', 'cifar', 'emnist'.
    data_path: path to the dataset.
    precision: precision of floating point numbers.
    n_tasks: number of tasks. See comment above.
    split_ratio: fraction of examples that are shared between all tasks.
    whitening: whether to whiten the input data.

    """

    # load the entire dataset first
    all_train_x, all_test_x, all_train_digits, all_test_digits = \
        load_dataset(
            dataset_name=dataset_name,
            path=data_path,
            whitening=whitening)
    
    if dataset_name in ['cifar100', 'cifar100_gray']:
        # some processing specific for CIFAR100
        # Since there are only 500 images per class in the default training set,
        # minus a few that are duplicates, we move some data from the test set
        # into the training set. we also merge classes so that we end up with 50
        # classes in total. This is all to ensure that we can have split
        # sequences where each task has two classes and 2000 examples in total.
        sorted_test_x = [all_test_x[all_test_digits == i] for i in range(100)]


        first_half_of_test_x = [x[:len(x)//2] for x in sorted_test_x]
        second_half_of_test_x = [x[len(x)//2:] for x in sorted_test_x]

        digits_for_first_half = [
            i for i in range(100) for _ in range(len(first_half_of_test_x[i]))]
        digits_for_second_half = [
            i for i in range(100) for _ in range(len(second_half_of_test_x[i]))]

        first_half_of_test_x = torch.vstack(first_half_of_test_x)
        second_half_of_test_x = torch.vstack(second_half_of_test_x)
        digits_for_first_half = torch.tensor(digits_for_first_half)
        digits_for_second_half = torch.tensor(digits_for_second_half)

        all_train_x = torch.vstack(
            [all_train_x, first_half_of_test_x])
        remixed_train_digits = torch.hstack(
            [all_train_digits, digits_for_first_half])

        all_test_x = second_half_of_test_x
        remixed_test_digits = digits_for_second_half

        all_train_digits = remixed_train_digits % 50
        all_test_digits = remixed_test_digits % 50

    return _generate_split_sequence_from_loaded_data(
        all_train_x,
        all_test_x,
        all_train_digits,
        all_test_digits,
        n_tasks,
        train_p,
        test_p,
        split_ratio=split_ratio,
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
                 path=None,
                 mean_subtraction='image',
                 whitening=False):
    """
    Does mean subtraction, duplicate removal, whitening, and scale normalization. 
    """
    if path is None:
        path = '/Users/haozheshan/Dropbox/codes/gp_continual_learning/datasets'

    assert dataset_name in [
        'cifar', 'mnist', 'fashion', 'cifar100', 'emnist', 'cifar_gray',
        'cifar100_gray', 'fashion_emnist_mix'],\
          'dataset name not understood'

    train_set = None
    test_set = None

    trans = transforms.Compose([transforms.PILToTensor(),
                        transforms.ConvertImageDtype(torch.float64)])

    if dataset_name in ['cifar', 'cifar_gray']:
        train_set = torchvision.datasets.CIFAR10(path, train=True, download=False, transform=trans)
        test_set = torchvision.datasets.CIFAR10(path, train=False, download=False, transform=trans)
    elif dataset_name in ['cifar100', 'cifar100_gray']:
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
    elif dataset_name == 'fashion_emnist_mix':
        train_set1 = torchvision.datasets.FashionMNIST(path, train=True, download=True, transform=trans)
        test_set1 = torchvision.datasets.FashionMNIST(path, train=False, download=True, transform=trans)
        train_set2 = torchvision.datasets.EMNIST(path, 'byclass', train=True, download=True, transform=trans)
        test_set2 = torchvision.datasets.EMNIST(path, 'byclass', train=False, download=True, transform=trans)

        train_set2.targets += 10
        test_set2.targets += 10

        train_set = torch.utils.data.ConcatDataset([train_set1, train_set2])
        test_set = torch.utils.data.ConcatDataset([test_set1, test_set2])

    # train_loader = DataLoader(train_set, batch_size=num_train, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=num_test, shuffle=False)

    # raw_train_data = next(iter(train_loader))
    # raw_test_data = next(iter(test_loader))
    

    def _get_x_y(dataset, mean_subtraction):
        if type(dataset) is not torch.utils.data.ConcatDataset:
            x = dataset.data
            y = dataset.targets

            if type(x) == np.ndarray:
                x = torch.tensor(x).float()
            else:
                x = x.float()

            if type(y) == np.ndarray or type(y) == list:
                y = torch.tensor(y).float()
            else:
                y = y.float()

            x = x.reshape(x.shape[0], -1)
        
        else:
            x1 = dataset.datasets[0].data
            x2 = dataset.datasets[1].data
            y1 = dataset.datasets[0].targets
            y2 = dataset.datasets[1].targets
            x1 = x1.reshape(x1.shape[0], -1).float()
            x2 = x2.reshape(x2.shape[0], -1).float()
            x = torch.vstack((x1, x2))
            y = torch.hstack((y1, y2))


        assert mean_subtraction in ['image', 'batch']
        if mean_subtraction == 'batch':
            # subtract the mean pixel value across positions AND images
            # each image would have a different mean
            x -= x.mean()
        else:
            # subtract the mean pixel value across positions.
            # each image would have zero mean
            x -= torch.mean(x, dim=1).reshape(-1, 1)



        return x, y

    train_x, train_y = _get_x_y(train_set, mean_subtraction)
    test_x, test_y = _get_x_y(test_set, mean_subtraction)

    # try to remove duplicate images by removing ones with the same norm
    _, unique_train_inds = np.unique(torch.norm(train_x, dim=1),
                                     return_index=True)
    _, unique_test_inds = np.unique(torch.norm(test_x, dim=1),
                                    return_index=True)

    train_x = utils.normalize_input(train_x[unique_train_inds])
    test_x = utils.normalize_input(test_x[unique_test_inds])

    if 'gray' in dataset_name:
        assert dataset_name in ['cifar_gray', 'cifar100_gray']
        # convert RGB to grayscale
        train_x = train_x.reshape(-1, 3, 1024).mean(1)
        test_x = test_x.reshape(-1, 3, 1024).mean(1)

    if whitening:
        train_x, whitened_mat = whiten(train_x)
        test_x = test_x @ whitened_mat

    return train_x, test_x, train_y[unique_train_inds], test_y[unique_test_inds]


def prepare_target_distractor_sequence(num_tasks: int,
                             train_p: int,
                             test_p: int,
                             num_flipped_labels: int,
                             target_frac: float,
                             data_path=None,
                             precision=32,
                             whitening=False):
    """
    TODO add docstring

    Args:
    num_tasks: number of tasks. Can be as many as desired.
    train_p: number of training samples per task.
    test_p: number of test samples per task.
    num_flipped_labels: number of labels to flip in each task.
    target_frac: fraction of the target digits in each task.
    data_path: path to the dataset.
    precision: precision of floating point numbers.
    whitening: whether to whiten the input data.
    """

    all_train_x, all_test_x, all_train_digits, all_test_digits = \
        load_dataset(
            dataset_name='fashion_emnist_mix',
            path=data_path,
            whitening=whitening)

    return _generate_target_distractor_from_loaded_data(
        all_train_x,
        all_test_x,
        all_train_digits,
        all_test_digits,
        num_flipped_labels=num_flipped_labels,
        num_tasks=num_tasks,
        train_p=train_p,
        test_p=test_p,
        target_frac=target_frac,
        precision=precision)


def _generate_target_distractor_from_loaded_data(
        all_train_x, all_test_x, all_train_digits, all_test_digits,
        num_flipped_labels, num_tasks, train_p, test_p, target_frac,
        precision=64, pixel_perm=0.2):

    assert num_tasks <= 20

    num_targets = int(target_frac * train_p)
    num_distractors = train_p - num_targets

    num_targets_te = int(target_frac * test_p)
    num_distractors_te = test_p - num_targets_te


    def _generate_target_x_y(all_x, all_digits, n_targets):

        # the target digits should be greater than 19 (first 20 are distractors)
        target_digits_inds = np.where(all_digits > 19)[0]
        all_target_x = all_x[target_digits_inds]
        all_target_digits = all_digits[target_digits_inds]

        target_perm = np.random.permutation(len(all_target_x))
        target_x = all_target_x[target_perm][:n_targets]
        target_digits = all_target_digits[target_perm][:n_targets].reshape((n_targets, 1))

        # digits in the first row are in one class; those in the second are in the other
        task0_dichotomy = np.vstack((np.arange(20, 46), np.arange(46, 72)))

        target_y = target_digits.clone()
        utils.dichotomize_in_place(target_y,
                                task0_dichotomy[0],
                                task0_dichotomy[1])
        return target_x, target_y, target_digits


    def _generate_distractor_x_y(all_x, all_digits, distractor_digit_ind, n_distractors):
        distractor_x = utils.shuffle_along_first_axis(
            all_x[all_digits == distractor_digit_ind])[:n_distractors]
        distractor_y = torch.zeros(n_distractors, 1)
        return distractor_x, distractor_y


    target_x, _, target_digits = _generate_target_x_y(
        all_train_x, all_train_digits, num_targets)

    target_x_te, _, target_digits_te = _generate_target_x_y(
        all_test_x, all_test_digits, num_targets_te)

    distractor_class_inds = np.random.permutation(20)

    seq_of_train_x = []
    seq_of_test_x = []
    seq_of_train_y = []
    seq_of_test_y = []

    default_dichotomy = np.vstack((np.arange(20, 46), np.arange(46, 72)))

    for task_ind in range(num_tasks):

        distractor_x, distractor_y = _generate_distractor_x_y(
            all_train_x, all_train_digits, distractor_class_inds[task_ind],
        num_distractors)

        distractor_x_te, distractor_y_te = _generate_distractor_x_y(
            all_test_x, all_test_digits, distractor_class_inds[task_ind],
        num_distractors_te)
        
        # flip some of the targets
        targets_to_flip = np.random.choice(26, num_flipped_labels, replace=False)
        task_dichotomy = default_dichotomy.copy()
        task_dichotomy[0, targets_to_flip] = default_dichotomy[1, targets_to_flip]
        task_dichotomy[1, targets_to_flip] = default_dichotomy[0, targets_to_flip]

        task_target_y = target_digits.clone()
        utils.dichotomize_in_place(task_target_y, task_dichotomy[0], task_dichotomy[1])
        
        task_target_y_te = target_digits_te.clone()
        utils.dichotomize_in_place(task_target_y_te, task_dichotomy[0], task_dichotomy[1])
        
        train_x = utils.normalize_input(torch.cat([target_x, distractor_x], axis=0))
        train_y = torch.cat([task_target_y, distractor_y], axis=0)

        test_x = utils.normalize_input(torch.cat([target_x_te, distractor_x_te], axis=0))
        test_y = torch.cat([task_target_y_te, distractor_y_te], axis=0)

        perm_mat = utils.get_permutation_mat(train_x.shape[-1], strength=pixel_perm)
        
        seq_of_train_x.append(train_x @ perm_mat)
        seq_of_test_x.append(test_x @ perm_mat)
        seq_of_train_y.append(train_y)
        seq_of_test_y.append(test_y)
    
    return _pack_data(seq_of_train_x, precision), _pack_data(seq_of_test_x, precision), \
        _pack_data(seq_of_train_y, precision), _pack_data(seq_of_test_y, precision)