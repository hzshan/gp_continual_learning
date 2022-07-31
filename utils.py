import numpy as np
from numpy.linalg import inv
import torch
import tqdm, torchvision, math
import torchvision.transforms as transforms
import scipy.optimize
import matplotlib.pyplot as plt


def exp_fit_and_plot_new(xaxis, y, p0, **kwargs):
    init_offset = y[0]
    p_opt, p_cov = scipy.optimize.curve_fit(lambda t,gamma,v: (gamma - gamma**t) * v, xaxis, y, p0=p0)
    gamma, v = p_opt
    print('fitted param variance:', p_cov)
    plt.plot(xaxis, (gamma - gamma**xaxis) * v, **kwargs)
    print(f'gamma:{gamma}, v:{v}')
    return gamma, v


def exp_fit_and_plot(xaxis, y, p0, plot_axis=None, x_offset=0, **kwargs):

    if plot_axis is None:
        plot_axis = xaxis
    xaxis = xaxis + x_offset
    init_offset = y[0]
    offset, time_constant =\
        scipy.optimize.curve_fit(lambda t,a,b: a + (init_offset - a) * np.exp(-(t-x_offset) / b), xaxis, y, p0=p0)[0]
    plt.plot(plot_axis, offset + (init_offset - offset) * np.exp(-(plot_axis-x_offset) / time_constant), **kwargs)
    return offset, time_constant


def normalize_input(input_arr):
    if len(input_arr.shape) == 2:
        input_arr = input_arr / torch.norm(input_arr, dim=1).reshape(-1, 1) * math.sqrt(input_arr.shape[1])
    elif len(input_arr.shape) == 3:
        for i in range(input_arr.shape[0]):
            input_arr[i] =\
                input_arr[i] / torch.norm(input_arr[i], dim=1).reshape(-1, 1) * math.sqrt(input_arr[i].shape[1])
    else:
        raise ValueError('Shape of input tensor not understood.')
    return input_arr


def cos(x1, x2):
    if len(x1.shape) == 1:
        x1 = x1.reshape(1, -1)
        x2 = x2.reshape(1, -1)
    dot_product = x1 @ x2.T
    norm_mat = np.linalg.norm(x1, axis=1).reshape(-1, 1) @ np.linalg.norm(x2, axis=1).reshape(1, -1)
    return dot_product / norm_mat


def get_loss_matrix_single_task(all_predictions, seq_of_target):
    """

    :param all_predictions: task_ind * time_ind * input_ind * 1
    :param seq_of_target:
    :return: loss matrix: task_ind * time_ind
    """
    n_tasks = all_predictions.shape[0]
    assert len(seq_of_target) == n_tasks

    loss_matrix = np.zeros((n_tasks, n_tasks))

    for task_ind in range(n_tasks):
        for time_ind in range(n_tasks):
            if seq_of_target[task_ind] is not None:
                loss_matrix[task_ind, time_ind] = loss_from_predictions(all_predictions[task_ind, time_ind],
                                                                        seq_of_target[task_ind])
    return loss_matrix


def generate_subspace_inputs(p_train, p_test, dim, num_tasks, subspace_dim, speed):
    seq_of_train_x = []
    seq_of_test_x = []
    assert dim >= subspace_dim
    starting_inds = torch.linspace(0, (dim - subspace_dim) * speed, num_tasks)
    for i in range(num_tasks):
        train_base = torch.zeros((p_train, dim))
        test_base = torch.zeros((p_test, dim))
        ind1 = int(starting_inds[i])
        ind2 = ind1 + subspace_dim
        train_base[:, ind1:ind2] = torch.normal(torch.zeros((p_train, subspace_dim)))
        test_base[:, ind1:ind2] = torch.normal(torch.zeros((p_test, subspace_dim)))
        seq_of_train_x.append(train_base)
        seq_of_test_x.append(test_base)
    return normalize_input(torch.stack(seq_of_train_x)), normalize_input(torch.stack(seq_of_test_x))


def generate_rotating_inputs(p_train, p_test, dim, num_tasks, rotation=1, distance=1):

    assert 0 <= rotation <= 1

    offset0 = np.random.normal(0, 1 / np.sqrt(dim), (1, dim)) * distance
    offset1 = np.random.normal(0, 1 / np.sqrt(dim), (1, dim)) * distance

    seq_of_train_x = []
    seq_of_test_x = []

    offset_weights = np.linspace(1, 1 - rotation, num_tasks)
    for i in range(num_tasks):
        curr_offset = np.sqrt(offset_weights[i]) * offset0 + np.sqrt(1 - offset_weights[i]) * offset1
        seq_of_train_x.append(np.random.normal(0, 1, (p_train, dim)) + curr_offset)
        seq_of_test_x.append(np.random.normal(0, 1, (p_test, dim)) + curr_offset)

    return np.array(seq_of_train_x), np.array(seq_of_test_x)


# def get_linear_teacher_labels(training_inputs, test_input, max_scaler=1):
#     num_tasks = len(training_inputs)
#     dim = test_input.shape[1]
#     a_teach1 = np.random.normal(0, 1, (dim, 1))
#
#     a_teach2 = np.random.normal(0, 1, (dim, 1))
#
#     assert max_scaler <= 1
#     w_scalers = np.linspace(0, max_scaler, num_tasks)
#
#     training_targets = []
#
#     for i in range(num_tasks):
#         teacher1_preds = training_inputs[i] @ a_teach1 * dim**-0.5
#         teacher2_preds = training_inputs[i] @ a_teach2 * dim**-0.5
#         training_targets.append(teacher1_preds * (1 - w_scalers[i]) + teacher2_preds * w_scalers[i])
#     teacher1_test = test_input @ a_teach1 * dim**-0.5
#     teacher2_test = test_input @ a_teach2 * dim**-0.5
#     test_target = teacher1_test * (1 - w_scalers[-1]) + teacher2_test * w_scalers[-1]
#     return training_targets, test_target

def get_linear_teacher_labels(training_inputs, test_inputs, max_scaler=1):
    device = training_inputs.device
    num_tasks = len(training_inputs)
    dim = training_inputs[0].shape[1]
    V_teach1 = torch.normal(0, 1, (dim, 1)).to(device)
    V_teach2 = torch.normal(0, 1, (dim, 1)).to(device)

    assert max_scaler <= 1
    w_scalers = torch.linspace(0, max_scaler, num_tasks)

    training_targets = []
    test_targets = []

    V_teach_transient = None
    for i in range(num_tasks):
        V_teach_transient = torch.sqrt(w_scalers[i]) * V_teach2 + torch.sqrt(1 - w_scalers[i]) * V_teach1
        training_targets.append(training_inputs[i] @ V_teach_transient * dim**-0.5)
        if test_inputs[i] is not None:
            test_targets.append(test_inputs[i] @ V_teach_transient * dim**-0.5)
        else:
            test_targets.append(None)

    return torch.stack(training_targets), torch.stack(test_targets)


def get_mix_up_labels(training_inputs, test_inputs, max_scaler=1, teacher_feature_dim=None):
    if teacher_feature_dim is not None:
        teacher_feature_dim = 5000
    num_tasks = len(training_inputs)
    dim = training_inputs[0].shape[1]
    W_teach1 = np.random.normal(0, 1, (dim, teacher_feature_dim))
    a_teach1 = np.random.normal(0, 1, (teacher_feature_dim, 1))

    W_teach2 = np.random.normal(0, 1, (dim, teacher_feature_dim))
    a_teach2 = np.random.normal(0, 1, (teacher_feature_dim, 1))

    assert max_scaler <= 1
    mixing = np.linspace(0, max_scaler, num_tasks)

    training_targets = []
    test_targets = []

    for i in range(num_tasks):
        training_targets.append((1 -mixing[i]) * forward(training_inputs[i], W_teach1, a_teach1) +
                                mixing[i] * forward(training_inputs[i], W_teach2, a_teach2))
        if test_inputs[i] is not None:
            test_targets.append((1 -mixing[i]) * forward(test_inputs[i], W_teach1, a_teach1) +
                                mixing[i] * forward(test_inputs[i], W_teach2, a_teach2))
        else:
            test_targets.append(None)

    return np.array(training_targets), np.array(test_targets)


def get_teacher_labels(training_inputs, test_inputs, max_scaler=1, teacher_feature_dim=None):
    if teacher_feature_dim is None:
        teacher_feature_dim = 5000
    num_tasks = len(training_inputs)
    dim = training_inputs[0].shape[1]
    W_teach1 = torch.normal(0, 1, (dim, teacher_feature_dim))
    a_teach1 = torch.normal(0, 1, (teacher_feature_dim, 1))

    W_teach2 = torch.normal(0, 1, (dim, teacher_feature_dim))
    a_teach2 = torch.normal(0, 1, (teacher_feature_dim, 1))

    # if cycles == 0:
    #     assert max_scaler <= 1
    #     w_scalers = np.linspace(0, max_scaler, num_tasks)
    # else:
    #     angles = np.linspace(0, 2*np.pi*cycles, num_tasks)
    #     w_scalers = (np.cos(angles) + 1) / 2
    w_scalers = torch.linspace(0, max_scaler, num_tasks)
    training_targets = []
    test_targets = []

    W_teach_transient = None
    a_teach_transient = None
    for i in range(num_tasks):
        W_teach_transient = torch.sqrt(w_scalers[i]) * W_teach2 + torch.sqrt(1 - w_scalers[i]) * W_teach1
        a_teach_transient = torch.sqrt(w_scalers[i]) * a_teach2 + torch.sqrt(1 - w_scalers[i]) * a_teach1
        training_targets.append(forward(training_inputs[i], W_teach_transient,
                                        a_teach_transient))
        if test_inputs[i] is not None:
            test_targets.append(forward(test_inputs[i], W_teach_transient,
                                        a_teach_transient))
        else:
            test_targets.append(None)

    return torch.stack(training_targets), torch.stack(test_targets)


# def generate_inputs_and_targets(p_train, p_test, dim, num_tasks, mode, permute_strength, initial_offset):
#
#     teacher_feature_dim = 2000
#     W_teach = np.random.normal(0, 1, (dim, teacher_feature_dim))
#     a_teach = np.random.normal(0, 1, (teacher_feature_dim, 1))
#
#     # if mode == 'stationary':
#     #     # each input is a new sample from a stationary distribution
#     #     train_inputs = [np.random.normal(0, 1, (p_train, dim)) + initial_offset for _i in range(num_tasks)]
#     #
#     # elif mode == 'partial_permute':
#     #     # each input is a partial permutation of the previous one.
#     #     train_inputs = [np.random.normal(0, 1, (p_train, dim)) + initial_offset]
#     #     for i in range(1, num_tasks):
#     #         train_inputs.append(train_inputs[-1] @ get_permutation_mat(dim, strength=permute_strength))
#     #
#     # elif mode == 'rotate':
#     #     rot_angle = np.pi / 2 / num_tasks
#     #     train_inputs = [np.random.normal(0, 1, (p_train, dim)) + initial_offset]
#     #     for i in range(1, num_tasks):
#     #         train_inputs.append(np.random.normal(0, 1, (p_train, dim)) + initial_offset @ get_rotation_mat(rot_angle * (i + 1), dim))
#     #         # train_inputs.append(train_inputs[-1] @ get_rotation_mat(rot_angle, dim))
#
#     # the test distribution is always centered on the center of the last training set
#     test_input = np.random.normal(0, 1, (p_test, dim)) + train_inputs[-1].mean(0)
#
#     train_targets = [forward(x, W_teach, a_teach) for x in train_inputs]
#     test_target = forward(test_input, W_teach, a_teach)
#
#     return train_inputs, train_targets, test_input, test_target



def get_rotation_mat(angle, dim):
    # this makes a N-dim rotation matrix w.r.t the plane spanned by the first two elements
    output = np.eye(dim)
    output[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return output


def get_permutation_mat(dim, strength=None):

    if strength is None:
        strength = 1.0

    assert 0 <= strength <= 1
    # otherwise, do a permutation of some indices. the number of indices = strength
    inds_to_permute = torch.randperm(dim)[:int(dim * strength)]
    shuffled_inds = inds_to_permute[torch.randperm(len(inds_to_permute))]
    output = torch.eye(dim)
    output[inds_to_permute] = output[shuffled_inds]
    return output


def get_kernel(x1, x2, w1, w2):
    return relu(x1.shape[1]**-0.5 * x1 @ w1) @\
           relu(x1.shape[1]**-0.5 * x2 @ w2).T / w1.shape[1]


def loss_from_predictions(predictions, targets):
    return (torch.norm(predictions.flatten() - targets.flatten())**2  / torch.norm(targets.flatten())**2).cpu()


def get_loss(inputs, weights, readout, target, normalized=True):
    raw_loss = np.linalg.norm(forward(inputs, weights, readout) - target) ** 2 / len(target)
    if normalized:
        return raw_loss / (np.linalg.norm(target)**2 / len(target))
    else:
        return raw_loss


def relu(x):
    x_out = x.copy()
    x_out[x < 0] = 0
    return x_out


def forward(inputs, weights, readout):
    return weights.shape[1]**-0.5 * torch.relu(inputs.shape[1]**-0.5 * inputs @ weights) @ readout



