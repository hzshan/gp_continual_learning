import torchvision, utils, torch, math
import numpy as np
from torchvision import transforms
import tqdm, theory, warnings
import torch.nn.functional as F
"""
Utility functions for working with MNIST and CIFAR-10.
"""


def digit_to_onehot(digit_target, num_classes=10):
    return torch.unsqueeze(F.one_hot(digit_target.long(), num_classes=num_classes).double(), -1)


def get_multihead_accuracy(seq_of_train_x, seq_of_test_x, seq_of_train_y_digit,
                           seq_of_test_y_digit, sigma, lamb, fixed_w, depth, write_fn=None, tqdm_disable=False):

    """
    Simple wrapper for computing accuracy of multiclass classification over a sequence of tasks.
    """

    warnings.warn('get_multihead_accuracy assumes that there are 10 classes.')
    n_tasks = len(seq_of_train_x)
    P = seq_of_train_x[0].shape[0]
    P_test = seq_of_test_x[0].shape[0]
    seq_of_train_y_onehot =\
        torch.stack([digit_to_onehot(digit) for digit in seq_of_train_y_digit]).double()
    seq_of_test_y_onehot =\
        torch.stack([digit_to_onehot(digit) for digit in seq_of_test_y_digit]).double()

    all_test_predictions = torch.zeros((n_tasks, n_tasks, P_test, 10))  # task_ind * time_ind * P * num_heads
    all_train_predictions = torch.zeros((n_tasks, n_tasks, P, 10))

    for head_ind in tqdm.trange(10, position=0, disable=tqdm_disable):
        if write_fn is not None:
            write_fn(f'starting head {head_ind}')
        training_predictions, test_predictions =\
            theory.compute_mean_predictions(seq_of_train_x=seq_of_train_x,
                                            seq_of_train_y=seq_of_train_y_onehot[:, :, head_ind],
                                            w_var=sigma**2, P_test=P_test,
                                            lambda_val=lamb,
                                            seq_of_test_x=seq_of_test_x,
                                            fix_weights=fixed_w, depth=depth)
        # test_predictions: task_ind * time_ind * test_input_ind
        all_test_predictions[:, :, :, head_ind] = test_predictions.squeeze()
        all_train_predictions[:, :, :, head_ind] = training_predictions.squeeze()

    accuracy_forgetting_matrix_train = get_accuracy_matrix(all_train_predictions, seq_of_train_y_digit)
    accuracy_forgetting_matrix_test = get_accuracy_matrix(all_test_predictions, seq_of_test_y_digit)

    train_loss_mat = get_loss_mat(all_train_predictions, seq_of_train_y_onehot)
    test_loss_mat = get_loss_mat(all_test_predictions, seq_of_test_y_onehot)
    return accuracy_forgetting_matrix_train, accuracy_forgetting_matrix_test, train_loss_mat, test_loss_mat


def get_loss_mat(net_predictions, seq_of_onehot_targets):
    num_tasks = net_predictions.shape[0]
    loss_mat = torch.zeros((num_tasks, num_tasks))
    for task_ind in range(num_tasks):
        for time_ind in range(num_tasks):
            loss_mat[task_ind, time_ind] =\
                torch.mean((net_predictions[task_ind, time_ind].flatten() -
                            seq_of_onehot_targets[task_ind].flatten())**2)
    return loss_mat


def get_accuracy_matrix(net_predictions, target_digit):
    """

    :param net_predictions: # task_ind * time_ind * P * num_heads
    :param target_digit: # task_ind * P
    :return:
    """
    num_tasks = target_digit.shape[0]
    accuracy_mat = torch.zeros((num_tasks, num_tasks))  # task_ind * time_ind
    assert net_predictions.shape[0] == num_tasks
    net_predictions_in_digits = torch.argmax(net_predictions, dim=-1)
    for task_ind in range(num_tasks):
        for time_ind in range(num_tasks):
            _dummy = torch.zeros(target_digit[task_ind].shape)
            _dummy[net_predictions_in_digits[task_ind, time_ind] ==
                                        target_digit[task_ind]] = 1
            accuracy_mat[task_ind, time_ind] =\
                torch.mean(_dummy.float())
    return accuracy_mat


# def digit_to_onehot(digit_targets):
#     warnings.warn('digit_to_onehot will be removed. use torch nn functional instead')
#     onehot_targets = np.zeros((digit_targets.shape[0], 10, 1))
#     for i in range(digit_targets.shape[0]):
#         onehot_targets[i, digit_targets[i], 0] = 1
#     return onehot_targets


def _digit_to_even_odd(digit_y):
    """
    Convert digit-based targets from MNIST to odd/even (marked with +1/-1)
    :param digit_y:
    :return:
    """
    odd_indices = digit_y % 2 == 1
    even_indices = digit_y % 2 == 0

    digit_y[odd_indices] = 1
    digit_y[even_indices] = -1
    return digit_y


def prepare_sequential_dataset(num_tasks, train_p, test_p, resample,
                               dataset_name, data_path=None, permutation=0, n_epochs=1,
                               interpolate=False, precision=64):
    """
    Prepare a sequence of tasks in the format of 6 lists of length num_tasks.
    """
    assert 0 <= permutation <= 1
    seq_of_train_x = []
    seq_of_test_x = []
    seq_of_train_y_digit = []
    seq_of_test_y_digit = []
    if resample is False:
        all_train_x, all_test_x, all_train_y_digit, all_test_y_digit =\
            load_dataset(dataset_name=dataset_name, num_train=train_p, num_test=test_p, path=data_path)
    else:
        all_train_x, all_test_x, all_train_y_digit, all_test_y_digit =\
            load_dataset(dataset_name=dataset_name, num_train=int(train_p * num_tasks),
                         num_test=int(test_p * num_tasks), path=data_path)

    for _task_ind in range(num_tasks):
        if resample:
            train_x = all_train_x[_task_ind*train_p:(_task_ind+1)*train_p]
            test_x = all_test_x[_task_ind*test_p:(_task_ind+1)*test_p]
            train_y_digit = all_train_y_digit[_task_ind*train_p:(_task_ind+1)*train_p]
            test_y_digit = all_test_y_digit[_task_ind*test_p:(_task_ind+1)*test_p]
        else:
            train_x = all_train_x
            test_x = all_test_x
            train_y_digit = all_train_y_digit
            test_y_digit = all_test_y_digit

        seq_of_train_x.append(train_x)
        seq_of_test_x.append(test_x)
        seq_of_train_y_digit.append(train_y_digit)
        seq_of_test_y_digit.append(test_y_digit)

    N0 = seq_of_train_x[0].shape[1]

    if permutation > 0:
        if interpolate:
            # current_perm_mat = torch.eye(N0)
            # for _task_ind in range(num_tasks):
            #     current_perm_mat = current_perm_mat @ utils.get_permutation_mat(N0, strength=permutation)
            #     seq_of_train_x[_task_ind] = seq_of_train_x[_task_ind] @ current_perm_mat
            #     seq_of_test_x[_task_ind] = seq_of_test_x[_task_ind] @ current_perm_mat
            weights = np.linspace(0, 1, num_tasks)[::-1]
            perm_mat1 = utils.get_permutation_mat(N0, strength=permutation)
            perm_mat2 = utils.get_permutation_mat(N0, strength=permutation)
            for _task_ind in range(num_tasks):
                perm_mat = perm_mat1 * weights[_task_ind] + perm_mat2 * (1 - weights[_task_ind])
                seq_of_train_x[_task_ind] = seq_of_train_x[_task_ind] @ perm_mat
                seq_of_test_x[_task_ind] = seq_of_test_x[_task_ind] @ perm_mat
        else:
            for _task_ind in range(num_tasks):
                permutation_mat = utils.get_permutation_mat(N0, strength=permutation)
                seq_of_train_x[_task_ind] = seq_of_train_x[_task_ind] @ permutation_mat
                seq_of_test_x[_task_ind] = seq_of_test_x[_task_ind] @ permutation_mat

    # normalize all train and test input to have the same norm

    seq_of_train_x = torch.tile(torch.stack([utils.normalize_input(task_x) for task_x in seq_of_train_x]),
                                (n_epochs, 1))
    seq_of_test_x = torch.tile(torch.stack([utils.normalize_input(task_x) for task_x in seq_of_test_x]),
                               (n_epochs, 1))

    seq_of_train_y_digit = torch.tile(torch.stack(seq_of_train_y_digit), (n_epochs, 1))
    seq_of_test_y_digit = torch.tile(torch.stack(seq_of_test_y_digit), (n_epochs, 1))

    # if use_torch:
    #     seq_of_train_x = [torch.from_numpy(train_x.astype(np.float32)) for train_x in seq_of_train_x]
    #     seq_of_test_x = [torch.from_numpy(test_x.astype(np.float32)) for test_x in seq_of_test_x]
    #     seq_of_train_y_digit = torch.from_numpy(seq_of_train_y_digit.astype(np.float32))
    #     seq_of_test_y_digit = torch.from_numpy(seq_of_test_y_digit.astype(np.float32))

    assert precision in [16, 32, 64]
    if precision == 64:
        return seq_of_train_x.double(), seq_of_test_x.double(),\
               seq_of_train_y_digit.double(), seq_of_test_y_digit.double()
    elif precision == 32:
        return seq_of_train_x.float(), seq_of_test_x.float(),\
               seq_of_train_y_digit.float(), seq_of_test_y_digit.float()
    elif precision == 16:
        return seq_of_train_x.half(), seq_of_test_x.half(),\
               seq_of_train_y_digit.half(), seq_of_test_y_digit.half()



def load_dataset(dataset_name: str, num_train: int, num_test: int, path=None):
    if path is None:
        path = '/Users/haozheshan/Dropbox/codes/gp_continual_learning'

    assert dataset_name in ['cifar', 'mnist', 'fashion']

    def _shuffle(x, y):
        random_indices = torch.randperm(len(x))
        return x[random_indices], y[random_indices]

    def _get_x_y(dataset):
        if type(dataset.data) == np.ndarray:
            x = torch.from_numpy(dataset.data).float()
        else:
            x = dataset.data.float()
        x = x.reshape(x.shape[0], -1)
        x /= torch.norm(x, dim=1).reshape(-1, 1) / math.sqrt(x.shape[1])
        y = torch.tensor(dataset.targets).float()
        return _shuffle(x, y)

    train_set = None
    test_set = None
    # if dataset_name == 'cifar':
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    #     train_set = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=transform)
    #     test_set = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=transform)
    # else:
    #     transform = torchvision.transforms.Compose(
    #         [torchvision.transforms.ToTensor(),
    #          torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    #     train_set = torchvision.datasets.MNIST(path, train=True, download=True, transform=transform)
    #     test_set = torchvision.datasets.MNIST(path, train=False, download=True, transform=transform)

    if dataset_name == 'cifar':
        train_set = torchvision.datasets.CIFAR10(path, train=True, download=True)
        test_set = torchvision.datasets.CIFAR10(path, train=False, download=True)
    elif dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST(path, train=True, download=True)
        test_set = torchvision.datasets.MNIST(path, train=False, download=True)
    elif dataset_name == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(path, train=True, download=True)
        test_set = torchvision.datasets.FashionMNIST(path, train=False, download=True)
    train_x, train_y = _get_x_y(train_set)
    test_x, test_y = _get_x_y(test_set)

    return train_x[:num_train], test_x[:num_test], train_y[:num_train], test_y[:num_test]
