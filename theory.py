import numpy as np
import tqdm, utils, math, torch

import data, warnings


def inverse(torch_tensor):
    """
    This is necessary for running code on MPS because matrix inverse is not yet implemented (as of 05/30/22)
    :param torch_tensor:
    :return:
    """

    # return torch.inverse(torch_tensor.cpu()).to(torch_tensor.device)
    return torch.inverse(torch_tensor.cpu()).to(torch_tensor.device)


def compute_predictor_variances(seq_of_train_x: list, w_var: float, P_test: int, depth: int,
                                lambda_val: float, seq_of_test_x, fix_weights=False):
    """
    Returns:
    train_variances: 4d tensor of size `num_tasks`x`num_tasks`x`P`x`P`. [i,j,:,:] is the variance on training set #i
    after learning task #j. I.e., variance of f_j(x_i)
    test_variances: 4d tensor of size `num_tasks`x`num_tasks`x`P_test`x`P_test`. [i,j,:,:] is the variance on
     testing set #i after learning task #j. I.e., variance of f_j(x_i)
    """
    raise NotImplementedError('need further work on variance calculations')

    def k_tt(t1, t2):
        # kernel between W_{t1} x_{t1} and W_{t2} x_{t2}
        return cross_kernel(seq_of_train_x[t1], seq_of_train_x[t2], t1, t2, weight_covar, depth)

    num_tasks = len(seq_of_train_x)
    P = seq_of_train_x[0].shape[0]
    lamb_factor = lambda_val / (lambda_val + w_var**-1)
    weight_covar = compute_W_var(w_var, lambda_val, num_tasks, fix_w=fix_weights)
    all_self_kernels =\
        [cross_kernel(seq_of_train_x[_i], seq_of_train_x[_i], _i, _i, weight_covar, depth) for _i in range(num_tasks)]
    k_t_inv = [inverse(_M) for _M in all_self_kernels]

    # Create an array of size TxTxTxPxP, its i,j,k th element is <\delta f(a_i,W_j,x_j) \delta f(a_i,W_k,x_k)>
    full_train_variances = torch.zeros((num_tasks, num_tasks, num_tasks, P, P))

    for input_ind1 in range(num_tasks):
        for input_ind2 in range(num_tasks):
            full_train_variances[0, input_ind1, input_ind2] =\
                (k_tt(input_ind1, input_ind2) - k_tt(input_ind1, 0) @ k_t_inv[0] @ k_tt(0, input_ind2)) * w_var

    for t in range(1, num_tasks):
        for s in range(num_tasks):
            for sp in range(num_tasks):
                full_train_variances[t, s, sp] =\
                    (lambda_val + w_var**-1)**-1 * (k_tt(s, sp) - k_tt(s, t) @ k_t_inv[t] @ k_tt(t, sp)) * w_var +\
                    lamb_factor**2 *\
                    (full_train_variances[t-1, s, sp] - k_tt(s, t) @ k_t_inv[t] @ full_train_variances[t - 1, t, sp] -
                     full_train_variances[t-1, s, t] @ k_t_inv[t] @ k_tt(t, sp) +
                     k_tt(s, t) @ k_t_inv[t] @ full_train_variances[t-1, t, t] @ k_t_inv[t] @ k_tt(t, sp))

    # return only the "physical" prediction variances where a and W have the same index
    train_variances = np.zeros((num_tasks, num_tasks, seq_of_train_x[0].shape[0], seq_of_train_x[0].shape[0]))
    for func_ind in range(num_tasks):
        for input_ind in range(num_tasks):
            train_variances[input_ind, func_ind] = full_train_variances[func_ind, input_ind, input_ind]

    test_variances = np.zeros((len(seq_of_test_x), num_tasks, P_test, P_test))
    for test_input_ind in range(len(seq_of_test_x)):
        if seq_of_test_x[test_input_ind] is None:
            test_variances[test_input_ind] = None
        else:
            test_variances[test_input_ind] = \
                compute_variance_on_input(input_x=seq_of_test_x[test_input_ind],
                                         seq_of_train_x=seq_of_train_x,
                                         full_train_vars=full_train_variances,
                                         lambda_val=lambda_val,
                                         k_t_inv=k_t_inv,
                                         weight_covar=weight_covar, depth=depth)

    return train_variances, test_variances


def compute_variance_on_input(input_x, input_time,
                              seq_of_train_x, depth, full_train_vars, lambda_val, k_t_inv, weight_covar):
    """
    Denote T'=input_time
    Compute the variance <delta f_{T'}(X) delta f_{T'}(X)>
    """
    raise NotImplementedError('variance computation is not implemented')
    num_tasks = len(seq_of_train_x)
    assert input_time <= num_tasks - 1
    P_test = input_x.shape[0]
    P = seq_of_train_x[0].shape[0]
    w_var = weight_covar[0, 0]
    lamb_factor = lambda_val / (lambda_val + w_var**-1)

    # the i,j th element of the tensor below is <delta f(a_i,W_j,x_j) delta f(a_i, W_{T'}, X)>
    train_test_covars = torch.zeros((input_time, input_time, P, P_test))

    def k_tt(t1, t2):
        return cross_kernel(seq_of_train_x[t1], seq_of_train_x[t2], t1, t2, weight_covar, depth)

    K_te_te = cross_kernel(input_x, input_x, input_time, input_time, weight_covar, depth)
    K_t_te =\
        [cross_kernel(seq_of_train_x[t], input_x, t, input_time, weight_covar, depth) for t in range(input_time)]

    for j in range(input_time):
        train_test_covars[0, j] = (K_t_te[j] - k_tt(j, 0) @ k_t_inv[0] @ K_t_te[0]) * w_var

    for i in range(1, input_time):
        for j in range(input_time):
            train_test_covars[i, j] = \
                (lambda_val + w_var ** -1) ** -1 * (K_t_te[j] - k_tt(j, i) @ k_t_inv[i] @ K_t_te[i]) + \
                lamb_factor ** 2 * (train_test_covars[i - 1, j] -
                                    k_tt(j, i) @ k_t_inv[i] @ train_test_covars[i - 1, i] -
                                    full_train_vars[i - 1, j, i] @ k_t_inv[i] @ K_t_te[i] +
                                    k_tt(j, i) @ k_t_inv[i] @ full_train_vars[i - 1, i, i] @ k_t_inv[i] @ K_t_te[i])

    # the i th element of the tensor below is <delta f(a_i,W_{T'},X) delta f(a_i, W_{T'}, X)>
    test_vars = torch.zeros((input_time, P_test, P_test))
    for i in range(input_time):
        if i == 0:
            test_vars[i] = (K_te_te - K_t_te[0].T @ k_t_inv[0] @ K_t_te[0]) * w_var
        else:
            test_vars[i] =\
                (lambda_val + w_var ** -1) ** -1 * (K_te_te - K_t_te[i].T @ k_t_inv[i] @ K_t_te[i]) + \
                lamb_factor ** 2 * (
                test_vars[i - 1] - K_t_te[i].T @ k_t_inv[i] @ train_test_covars[i - 1, i] -
                train_test_covars[i - 1, i].T @ k_t_inv[i] @ K_t_te[i] +
                K_t_te[i].T @ k_t_inv[i] @ full_train_vars[i-1, i, i] @ k_t_inv[i] @ K_t_te[i])

    return test_vars[-1]


def multihead_one_task_accuracy(train_x, test_x, train_y, test_y, depth):
    """
    Computes the classification accuracy of a 10-head GP after learning a single task.
    :param train_x: PxN0
    :param test_x: P_test x N0
    :param train_y: Px1
    :param test_y: P_test x 1
    :param depth: depth of the deep ReLU kernel
    :return:
    """
    train_y_onehot = data.digit_to_onehot(train_y) # P x n_heads
    test_y_onehot = data.digit_to_onehot(test_y) # P x n_heads
    all_test_predictions = torch.zeros((10, test_y.shape[0]))

    mat = arccos_kernel_deep(test_x, train_x, depth=depth, var1=1.0) @\
          inverse(arccos_kernel_deep(train_x, train_x, depth=depth, var1=1.0))

    for head_ind in range(10):
        all_test_predictions[head_ind] = (mat @ train_y_onehot[:, head_ind]).squeeze()

    te_loss = torch.mean((all_test_predictions.T.flatten() - test_y_onehot.flatten())**2)
    _dummy = torch.zeros(test_y.squeeze().shape)
    _dummy[torch.argmax(all_test_predictions, dim=0) == test_y.squeeze()] = 1

    return torch.mean(_dummy), te_loss


# def one_task_loss(train_x, test_x, train_y, test_y, depth):
#     """
#     Computes the normalized MSE loss after kernel learning with a single training set.
#     :param train_x: P x N0
#     :param test_x: P_test x N0
#     :param train_Y: P x 1
#     :param test_Y: P_test x 1
#     :return: scalar normalized MSE loss on the test set.
#     """
#     batch_prediction = arccos_kernel_deep(test_x, train_x, 1, depth=depth) @ inverse(
#         arccos_kernel_deep(train_x, train_x, 1, depth=depth)) @ train_y
#     return utils.loss_from_predictions(batch_prediction, test_y)


def compute_mean_predictions_on_input(input_x, seq_of_train_x, seq_of_train_y,
                                      full_train_preds, depth, lamb_factor, self_kernel_inverses,
                                      weight_covar, lamb, sigma, fix_weights=False, use_naive_gp=False):
    """
    return a T x P_{input} x N0 array. Its i-th element is <f_i(X)>
    """
    kernel_fn_to_use = cross_kernel_new
    if use_naive_gp:
        kernel_fn_to_use = cross_kernel

    device = seq_of_train_x[0].device
    num_tasks = len(seq_of_train_x)
    w_indices = 1 if fix_weights else num_tasks

    # create a T x T x P_{input} x N0 array. Its i,j-th element is <f(a_i,W_j,X)>
    full_test_predictions = torch.zeros((num_tasks, w_indices, input_x.shape[0], 1), device=device)

    for j in range(w_indices):
        full_test_predictions[0, j] = kernel_fn_to_use(input_x, seq_of_train_x[0], j, 0, weight_covar, depth,
                                                       lamb, sigma) @\
                                 self_kernel_inverses[0] @ seq_of_train_y[0]

    for i in range(1, num_tasks):
        for j in range(w_indices):
            if fix_weights:
                pred_from_last = full_train_preds[i - 1, i, 0]
            else:
                pred_from_last = full_train_preds[i - 1, i, i]  # this is <f(a_{i-1},W_i,X_i)>
            full_test_predictions[i, j] =\
                kernel_fn_to_use(input_x, seq_of_train_x[i], j, i, weight_covar, depth, lamb, sigma) @\
                                     self_kernel_inverses[i] @\
            (seq_of_train_y[i] - lamb_factor * pred_from_last) + \
                                     lamb_factor * full_test_predictions[i - 1, j]

    # create the output by only taking the "physical" predictions where a and W have the same index
    output_test_predictions = torch.zeros((num_tasks, input_x.shape[0], 1), device=device)
    for param_ind in range(num_tasks):
        if fix_weights:
            output_test_predictions[param_ind] = full_test_predictions[param_ind, 0, :]
        else:
            output_test_predictions[param_ind] = full_test_predictions[param_ind, param_ind, :]

    return output_test_predictions


def compute_mean_predictions(seq_of_train_x, seq_of_train_y, w_var, P_test, depth,
                             lambda_val, seq_of_test_x, fix_weights=False, use_naive_gp=False):
    """
    returns a TxTxPxN0 array where the i,jth element is <f_j(X_i)>, and a TxTxP_{test}xN0 array where the i,jth element
    is <f_j(i-th test set)>.
    """

    kernel_fn_to_use = cross_kernel_new
    if use_naive_gp:
        kernel_fn_to_use = cross_kernel
    device = seq_of_train_x[0].device
    num_tasks = len(seq_of_train_x)
    # lamb_factor = lambda_val / (lambda_val + w_var**-1)
    lamb_factor = 1
    weight_covar = compute_W_var(w_var, lambda_val, num_tasks, fix_w=fix_weights)
    all_self_kernels =\
        [kernel_fn_to_use(seq_of_train_x[_i], seq_of_train_x[_i], _i, _i, weight_covar, depth=depth,
                          lamb=lambda_val, sigma=np.sqrt(w_var))
         for _i in range(num_tasks)]
    all_self_kernel_inverses = [inverse(_M) for _M in all_self_kernels]

    if fix_weights:
        num_weights = 1
    else:
        num_weights = num_tasks

    # create a TxTxTxPxN0 array where its i,j,k th element is <f(a_i,W_k, X_j)>
    full_train_predictions =\
        torch.zeros((num_tasks, num_tasks, num_weights, seq_of_train_x[0].shape[0], 1), device=device)

    for input_ind in range(num_tasks):
        for weight_ind in range(num_weights):
            full_train_predictions[0, input_ind, weight_ind] = \
                kernel_fn_to_use(seq_of_train_x[input_ind], seq_of_train_x[0],
                                 weight_ind, 0, weight_covar, depth, lambda_val, np.sqrt(w_var)) @\
                all_self_kernel_inverses[0] @ seq_of_train_y[0]


    for i in range(1, num_tasks):
        for j in range(num_tasks):
            for k in range(num_weights):

                if fix_weights:
                    pred_from_last = full_train_predictions[i - 1, i, 0]
                else:
                    pred_from_last = full_train_predictions[i - 1, i, i]  # this is <f(a_{i-1},X_j,W_j>)

                full_train_predictions[i, j, k] =\
                    kernel_fn_to_use(seq_of_train_x[j], seq_of_train_x[i], k, i, weight_covar, depth,
                                 lambda_val, np.sqrt(w_var)) @\
                    all_self_kernel_inverses[i] @ \
                    (seq_of_train_y[i] - lamb_factor * pred_from_last) +\
                    lamb_factor * full_train_predictions[i - 1, j, k]

    # only keep the "physical" predictions where a and W have the same index.
    train_predictions = torch.zeros((num_tasks, num_tasks, seq_of_train_x[0].shape[0], 1), device=device)
    for input_ind in range(num_tasks):
        for param_ind in range(num_tasks):
            if fix_weights:
                train_predictions[input_ind, param_ind] =\
                    full_train_predictions[param_ind, input_ind, 0, :]
            else:
                train_predictions[input_ind, param_ind] =\
                    full_train_predictions[param_ind, input_ind, param_ind, :]

    test_predictions = torch.zeros((len(seq_of_test_x), num_tasks, P_test, 1), device=device)
    # task_ind * time_ind * test_input_ind
    for input_ind in range(len(seq_of_test_x)):

        if seq_of_test_x[input_ind] is None:
            test_predictions[input_ind] = torch.nan
        else:
            test_predictions[input_ind] =\
                compute_mean_predictions_on_input(input_x=seq_of_test_x[input_ind],
                                                  seq_of_train_x=seq_of_train_x,
                                                  seq_of_train_y=seq_of_train_y,
                                                  full_train_preds=full_train_predictions,
                                                  lamb_factor=lamb_factor,
                                                  self_kernel_inverses=all_self_kernel_inverses,
                                                  weight_covar=weight_covar,
                                                  fix_weights=fix_weights,
                                                  depth=depth,
                                                  lamb=lambda_val,
                                                  sigma=np.sqrt(w_var),
                                                  use_naive_gp=use_naive_gp)
    return train_predictions, test_predictions


def compute_W_var(var1, lamb, n_tasks, fix_w=False):
    if fix_w:
        # fixing w throughout the tasks. Thus the covariance between any pair of W_t, W_t' is the same
        return torch.ones((n_tasks, n_tasks)) * var1
    else:
        Wcovars = torch.zeros((n_tasks, n_tasks))
        time_inds = torch.arange(n_tasks).reshape(-1, 1)
        time_ind_diff = time_inds - time_inds.T
        ratios = (lamb / (lamb + var1 ** -1)) ** torch.abs(time_ind_diff)
        Wvars = np.zeros(n_tasks)
        Wvars[0] = var1
        for i in range(1, n_tasks):
            Wvars[i] = Wvars[i - 1] * (lamb / (lamb + var1 ** -1)) ** 2 + 1 / (lamb + var1 ** -1)

        for i in range(n_tasks):
            for j in range(n_tasks):
                Wcovars[i, j] = ratios[i, j] * Wvars[np.min([i, j])]
        return Wcovars


def cross_kernel(x1, x2, t1, t2, Wcovar_mat, depth):
    """
    Simple wrapper of the deep arccos kernel function.
    :param x1:
    :param x2:
    :param t1:
    :param t2:
    :param Wcovar_mat:
    :param depth:
    :return:
    """
    return arccos_kernel_deep(x1, x2, var1=Wcovar_mat[t1, t1],
                         var2=Wcovar_mat[t2, t2], covar=Wcovar_mat[t1, t2], depth=depth)


# need var_t, var_t', var_t'-1, lambda, sigma^2
def cross_kernel_new(x1, x2, t1, t2, w_covar_mat, depth, lamb, sigma):
    """
    """
    # convert time indices to start from 1
    t = t1 + 1
    t_prime = t2 + 1

    t_min = np.min([t, t_prime])
    t_min_idx = np.min([t1, t2])
    t_diff = np.abs(t - t_prime)
    lambda_tilde = lamb / (lamb + sigma**-2)
    factor1 = lambda_tilde**(t_diff + 1) / lamb
    factor2_1 = lambda_tilde**(t + t_prime - 2) * sigma**2

    naive_gp_kernel = arccos_kernel_deep(x1, x2, var1=w_covar_mat[t1, t1],
                                         var2=w_covar_mat[t2, t2], covar=w_covar_mat[t1, t2], depth=depth)

    if t_min == 1:
        return sigma**2 * lambda_tilde**t_diff * naive_gp_kernel
    else:

        zero_gp_kernel = arccos_kernel_deep(x1, x2, var1=w_covar_mat[t1, t1],
                                            var2=w_covar_mat[t2, t2],
                                            covar=w_covar_mat[t_min_idx - 1, t_min_idx - 1] *
                                                  lambda_tilde**(t_diff + 2),
                                            depth=depth)

        if t_min < 3:
            factor2_2 = 0
        else:
            factor2_2 = lambda_tilde**(t_diff + 3) / lamb *\
                        (1 - lambda_tilde**(2 * t_min - 4)) / (1 - lambda_tilde**2)
        return factor1 * naive_gp_kernel + (factor2_1 + factor2_2) * (naive_gp_kernel - zero_gp_kernel)


def arccos_kernel_deep(x1, x2, depth, var1=None, var2=None, covar=None):
    warnings.warn('arccosine kernel computation is assuming that all inputs have the same norm.')
    if var2 is None or covar is None:
        var2 = var1
        covar = var1
    kappa = covar / math.sqrt(var1 * var2)

    assert 0 < kappa <= 1
    input_kernel = x1 @ x2.T / x1.shape[1]
    if depth == 0:
        return input_kernel
    normalized_norm_sq = torch.norm(x1[0]) * torch.norm(x2[0]) / x1.shape[1]
    cos_mat = input_kernel / normalized_norm_sq
    cos_mat = torch.clamp(cos_mat, -1, 1)
    # cos_mat -= torch.heaviside(cos_mat - 1, torch.zeros(1).to(cos_mat.device))
    # do truncation on cpu backend because it is not yet implemented on MPS (05/30/22)
    # cpu_cos_mat = cos_mat.cpu()
    # cpu_cos_mat[cpu_cos_mat > 1] = 1
    # cos_mat = cpu_cos_mat.to(cos_mat.device)

    theta_eff = torch.arccos(kappa * cos_mat)
    k = ((torch.pi - theta_eff) * torch.cos(theta_eff) + torch.sin(theta_eff)) * normalized_norm_sq / (2 * torch.pi) *\
        math.sqrt(var1 * var2)
    if depth == 1:
        return k
    else:
        # set up for iterations
        for layer in range(1, depth):
            normalized_norm_sq = normalized_norm_sq * math.sqrt(var1 * var2) / 2
            cos_theta_eff = k / normalized_norm_sq * kappa
            cos_theta_eff[cos_theta_eff > 1] = 1
            theta_eff = torch.arccos(cos_theta_eff)
            k = ((torch.pi - theta_eff) * torch.cos(theta_eff) + torch.sin(theta_eff)) *\
                normalized_norm_sq / (2 * torch.pi) * math.sqrt(var1 * var2)

    return k


def sample_at(last_a, num_samples, target_t, features_t, temp, lambda_t, std_t):
    """
    Sample from the multivariate Gaussian distribution of a_t
    :param last_a: N by num_samples. Samples of the readout for the last task. If this is the first task,
    this can be set to None.
    :param num_samples: number of samples
    :param target_t: target vector (Y_t)
    :param features_t: features (X_t)
    :param temp: temperature (T). must be >= 0
    :param lambda_t: will be set to zero if last_a is None
    :param std_t: sigma
    :return: at_samples (N by num_samples), at_mean (N by 1)
    """
    _N = features_t.shape[1]
    _P = features_t.shape[0]

    if last_a is None:
        last_a = torch.zeros((_N, num_samples))
        lambda_t = 0

    assert last_a.shape[1] == num_samples

    kernel_t = _N**-1 * features_t @ features_t.T
    inv_kernel_t_prime = torch.linalg.inv(kernel_t + temp * (std_t**-2 + lambda_t) * np.eye(_P))

    mean_1of2 = _N**-0.5 * features_t.T @ inv_kernel_t_prime @ target_t
    mean_2of2 = lambda_t / (lambda_t + std_t**-2) *\
        (torch.eye(_N) - _N**-1 * features_t.T @ inv_kernel_t_prime @ features_t) @ last_a
    at_mean = mean_1of2 + mean_2of2
    # print(at_covar)

    raw_noise = None
    if temp > 0:
        # for finite temperature, generating Gaussian flucatuations that are full rank
        at_covar = np.linalg.inv(_N ** -1 * features_t.T @ features_t / temp +
                                 (std_t ** -2 + lambda_t) * np.eye(_N))
        raw_noise = np.random.multivariate_normal(np.zeros_like(at_mean[:, 0]), at_covar,
                                                  size=num_samples).T
    elif temp == 0:
        # for zero temperature, generating Gaussian flucatuations only in the null space of the features
        _, _, v = np.linalg.svd(features_t)
        featt = v[:features_t.shape[0]].T @ v[:features_t.shape[0]]
        raw_covar = np.linalg.inv((std_t ** -2 + lambda_t) * np.eye(_N))
        raw_noise = np.random.multivariate_normal(np.zeros_like(at_mean[:, 0]).flatten(),
                                                  raw_covar, size=num_samples).T
        raw_noise -= featt @ raw_noise
    return at_mean + raw_noise, at_mean