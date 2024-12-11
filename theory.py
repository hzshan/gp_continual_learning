import numpy as np
import tqdm, math, torch, warnings

"""
Last updated and inspected: Apr 22, 2023
"""

def inverse(torch_tensor):
    """
    This is necessary for running code on MPS because matrix inverse is not yet implemented (as of 05/30/22)
    :param torch_tensor:
    :return:
    """

    # return torch.inverse(torch_tensor.cpu()).to(torch_tensor.device)
    # return torch.inverse(torch_tensor.cpu()).to(torch_tensor.device)
    return torch.inverse(torch_tensor)


def compute_mean_predictions_on_input(input_x, seq_of_train_x, seq_of_train_y,
                                      full_train_preds, depth, self_kernel_inverses,
                                      weight_covar, lamb, sigma, large_lambda=False, use_naive_gp=False):
    """
    return a T x P_{input} x N0 array. Its i-th element is <f_i(X)>
    """
    raise DeprecationWarning("This function is deprecated. Use compute_mean_predictions_on_inputs instead.")
    kernel_fn_to_use = None
    if use_naive_gp is True:
        kernel_fn_to_use = cross_kernel
    else:
        kernel_fn_to_use = cross_kernel_new

    device = seq_of_train_x[0].device
    num_tasks = len(seq_of_train_x)

    # create a T x T x P_{input} x N0 array. Its i,j-th element is <f(a_i,W_j,X)>
    full_test_predictions = torch.zeros(
        size=(num_tasks, num_tasks, input_x.shape[0], 1),
        device=device)

    for j in range(num_tasks):
        full_test_predictions[0, j] = kernel_fn_to_use(
            input_x, seq_of_train_x[0],
            j, 0, weight_covar,
            depth, lamb, sigma) @ self_kernel_inverses[0] @ seq_of_train_y[0]

    for i in range(1, num_tasks):
        for j in range(w_indices):
            if large_lambda:
                pred_from_last = full_train_preds[i - 1, i, 0]
            else:
                pred_from_last = full_train_preds[i - 1, i, i]  # this is <f_{i-1,i}(X_i)>
            full_test_predictions[i, j] =\
                kernel_fn_to_use(input_x, seq_of_train_x[i], j, i, weight_covar, depth, lamb, sigma) @ \
                self_kernel_inverses[i] @ \
                (seq_of_train_y[i] - pred_from_last) + full_test_predictions[i - 1, j]

    # create the output by only taking the "physical" predictions where a and W have the same index
    output_test_predictions = torch.zeros((num_tasks, input_x.shape[0], 1), device=device)
    for param_ind in range(num_tasks):
        if large_lambda:
            output_test_predictions[param_ind] = full_test_predictions[param_ind, 0, :]
        else:
            output_test_predictions[param_ind] = full_test_predictions[param_ind, param_ind, :]

    return output_test_predictions


def compute_mean_predictions_old(seq_of_train_x, seq_of_train_y, w_var, depth,
                             lambda_val, seq_of_test_x, large_lambda=False, use_naive_gp=False,
                             P_test=None):
    """
    returns a TxTxPxN0 array where the i,jth element is <f_j(X_i)>,
     and a TxTxP_{test}xN0 array where the i,jth element
    is <f_j(i-th test set)>.
    May 8 2023: The `P_test` parameter is not used but kept temporarily for compatibility with old code.
    """
    raise DeprecationWarning("This function is deprecated. Use compute_mean_predictions instead.")
    kernel_fn_to_use = None
    if use_naive_gp is True:
        kernel_fn_to_use = cross_kernel
    else:
        kernel_fn_to_use = cross_kernel_new
    device = seq_of_train_x[0].device
    num_tasks = len(seq_of_train_x)
    lambda_tilde = lambda_val / (lambda_val + w_var**-1)

    weight_covar = compute_W_var(w_var, lambda_val, num_tasks, fix_w=large_lambda)
    all_self_kernels =\
        [kernel_fn_to_use(seq_of_train_x[_i], seq_of_train_x[_i], _i, _i, weight_covar, depth=depth,
                          lamb=lambda_val, sigma=np.sqrt(w_var))
         for _i in range(num_tasks)]
    all_self_kernel_inverses = [inverse(_M) for _M in all_self_kernels]

    if large_lambda:
        num_weights = 1
    else:
        num_weights = num_tasks

    # create a TxTxTxPxN0 array where its i,j,k th element is <f(a_i,W_k, X_j)>
    full_train_predictions =\
        torch.zeros((num_tasks, num_tasks, num_weights, seq_of_train_x[0].shape[0], 1), device=device,
                    dtype=seq_of_train_x[0].dtype)

    for input_ind in range(num_tasks):
        for weight_ind in range(num_weights):
            full_train_predictions[0, input_ind, weight_ind] = \
                kernel_fn_to_use(seq_of_train_x[input_ind], seq_of_train_x[0],
                                 weight_ind, 0, weight_covar, depth, lambda_val, np.sqrt(w_var)) @\
                all_self_kernel_inverses[0] @ seq_of_train_y[0]
            
    decay_factor = lambda_tilde if use_naive_gp else 1.0

    for i in tqdm.trange(1, num_tasks):
        for j in range(num_tasks):
            for k in range(num_weights):

                if large_lambda:
                    pred_from_last = full_train_predictions[i - 1, i, 0]
                else:
                    pred_from_last = full_train_predictions[i - 1, i, i]  # this is <f_{i-1,i}(X_i)>

                full_train_predictions[i, j, k] =\
                    kernel_fn_to_use(seq_of_train_x[j], seq_of_train_x[i], k, i, weight_covar, depth,
                                     lambda_val, np.sqrt(w_var)) @\
                    all_self_kernel_inverses[i] @ \
                    (seq_of_train_y[i] - pred_from_last * decay_factor) +\
                    full_train_predictions[i - 1, j, k] * decay_factor

    # only keep the "physical" predictions where a and W have the same index.
    train_predictions = torch.zeros((num_tasks, num_tasks, seq_of_train_x[0].shape[0], 1),
                                    device=device, dtype=seq_of_train_x[0].dtype)
    for input_ind in range(num_tasks):
        for param_ind in range(num_tasks):
            if large_lambda:
                train_predictions[input_ind, param_ind] =\
                    full_train_predictions[param_ind, input_ind, 0, :]
            else:
                train_predictions[input_ind, param_ind] =\
                    full_train_predictions[param_ind, input_ind, param_ind, :]

    if seq_of_test_x is not None:
        test_predictions = torch.zeros((len(seq_of_test_x), num_tasks, seq_of_test_x.shape[1], 1),
                                       device=device, dtype=seq_of_train_x[0].dtype)
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
                                                      self_kernel_inverses=all_self_kernel_inverses,
                                                      weight_covar=weight_covar,
                                                      large_lambda=large_lambda,
                                                      depth=depth,
                                                      lamb=lambda_val,
                                                      sigma=np.sqrt(w_var),
                                                      use_naive_gp=use_naive_gp)
    else:
        test_predictions = None
    return train_predictions, test_predictions


def compute_mean_predictions_on_inputs(seq_of_inputs, aux_variable_means, seq_of_train_x, large_lambda, kernel_fn_to_use):
    
    num_inputs = len(seq_of_inputs)
    num_time = len(seq_of_train_x)

    predictions = torch.zeros(
        size=(num_inputs, num_time, seq_of_inputs.shape[1], 1),
        device=seq_of_train_x.device, dtype=seq_of_train_x[0].dtype)

    for input_ind in range(num_inputs):
        for time_ind in range(num_time):
            if large_lambda:
                # this is the large lambda limit; computational cost for tracking the predictor on each dataset is O(T)
                if time_ind == 0:
                    predictions[input_ind, time_ind] = kernel_fn_to_use(seq_of_inputs[input_ind], seq_of_train_x[0], 0, 0) @ aux_variable_means[0]
                else:
                    predictions[input_ind, time_ind] = predictions[input_ind, time_ind - 1] + kernel_fn_to_use(seq_of_inputs[input_ind], seq_of_train_x[time_ind], time_ind, time_ind) @ aux_variable_means[time_ind]
            else:
                for time_ind2 in range(time_ind + 1):
                    predictions[input_ind, time_ind] +=\
                        kernel_fn_to_use(seq_of_inputs[input_ind], seq_of_train_x[time_ind2], time_ind, time_ind2) @ aux_variable_means[time_ind2]
    return predictions


def compute_mean_predictions(seq_of_train_x, seq_of_train_y, w_var, depth,
                             lambda_val, seq_of_test_x,
                             large_lambda=False, use_naive_gp=False):
    """
    returns a TxTxPxN0 array where the i,jth element is <f_j(X_i)>,
     and a TxTxP_{test}xN0 array where the i,jth element.
    is <f_j(i-th test set)>.
    """
    lambda_tilde = lambda_val / (lambda_val + w_var**-1)
    device = seq_of_train_x[0].device
    num_tasks = len(seq_of_train_x)
    N0 = seq_of_train_x.shape[1]

    weight_covar = compute_W_var(w_var, lambda_val, num_tasks, fix_w=False)

    def kernel_fn_to_use(x1, x2, i1, i2):
        if use_naive_gp:
            return cross_kernel(
                x1, x2, i1, i2,
                weight_covar, depth,
                lambda_val, np.sqrt(w_var)) * lambda_tilde**np.abs(i1 - i2)
        else:
            return cross_kernel_new(
                x1, x2, i1, i2,
                weight_covar, depth, lambda_val, np.sqrt(w_var))

    all_self_kernels =\
        [kernel_fn_to_use(seq_of_train_x[_i], seq_of_train_x[_i], _i, _i)
         for _i in range(num_tasks)]
    all_self_kernel_inverses = [inverse(_M) for _M in all_self_kernels]

    aux_variable_means = torch.zeros(
        size=(num_tasks, N0, 1),
        device=device,
        dtype=seq_of_train_x[0].dtype)

    for time_ind in range(num_tasks):
        y_tilde = seq_of_train_y[time_ind].reshape(-1, 1).clone()
        for time_ind_2  in range(time_ind + 1):
            y_tilde -= kernel_fn_to_use(
                seq_of_train_x[time_ind], seq_of_train_x[time_ind_2],
                time_ind, time_ind_2) @\
                  aux_variable_means[time_ind_2]
        aux_variable_means[time_ind] = all_self_kernel_inverses[time_ind] @\
              y_tilde

    train_predictions = compute_mean_predictions_on_inputs(
        seq_of_train_x, aux_variable_means, seq_of_train_x,
        large_lambda=large_lambda, kernel_fn_to_use=kernel_fn_to_use)
    
    if seq_of_test_x is not None:
        test_predictions = compute_mean_predictions_on_inputs(
            seq_of_test_x, aux_variable_means, seq_of_train_x,
            large_lambda=large_lambda, kernel_fn_to_use=kernel_fn_to_use)
    else:
        test_predictions = None
    return train_predictions, test_predictions


def compute_W_var(var1, lamb, n_tasks, fix_w=False):
    """
    This generates a T by T matrix which describes the correlation between weight matrices from different times
    """
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


def cross_kernel(x1, x2, t1, t2, Wcovar_mat, depth, lamb, sigma):
    """
    Simple wrapper of the deep arccos kernel function (naive GP).
    :param x1:
    :param x2:
    :param t1:
    :param t2:
    :param Wcovar_mat:
    :param depth:
    :param lamb: 
    :param sigma: 
    :return:
    """
    lambda_tilde = lamb / (lamb + sigma**-2)
    return arccos_kernel_deep(x1, x2,
                              var1=Wcovar_mat[t1, t1],
                              var2=Wcovar_mat[t2, t2], covar=Wcovar_mat[t1, t2], depth=depth) * lambda_tilde**np.abs(t1 - t2)


def cross_kernel_new(x1, x2, t1, t2, w_covar_mat, depth, lamb, sigma, fixed_kernel=False):
    """
    """
    # convert time indices to start from 1
    t = t1 + 1
    t_prime = t2 + 1

    t_min = np.min([t, t_prime])
    t_min_idx = np.min([t1, t2])
    t_diff = np.abs(t - t_prime)
    lambda_tilde = 1 if fixed_kernel else lamb / (lamb + sigma**-2)
    factor1 = lambda_tilde**t_diff / lamb
    factor2_1 = lambda_tilde**(t + t_prime - 3) * sigma**2 * lamb

    naive_gp_kernel = arccos_kernel_deep(x1, x2, var1=w_covar_mat[t1, t1],
                                         var2=w_covar_mat[t2, t2], covar=w_covar_mat[t1, t2], depth=depth)

    if t_min == 1:
        return lambda_tilde**t_diff * naive_gp_kernel
    else:
        
        if fixed_kernel:
            zero_gp_kernel = arccos_kernel_deep(x1, x2, var1=w_covar_mat[1, 1],
                                    var2=w_covar_mat[1, 1],
                                    covar=w_covar_mat[0, 0] *
                                            lambda_tilde**(t_diff + 2),
                                    depth=depth)
        else:
            zero_gp_kernel = arccos_kernel_deep(x1, x2, var1=w_covar_mat[t1, t1],
                                                var2=w_covar_mat[t2, t2],
                                                covar=w_covar_mat[t_min_idx - 1, t_min_idx - 1] *
                                                    lambda_tilde**(t_diff + 2),
                                                depth=depth)

        if t_min < 3 or fixed_kernel:
            factor2_2 = 0
        else:
            factor2_2 = lambda_tilde**(t_diff + 2) *\
                        (1 - lambda_tilde**(2 * t_min - 4)) / (1 - lambda_tilde**2)
        return factor1 * naive_gp_kernel + (factor2_1 + factor2_2) * (naive_gp_kernel - zero_gp_kernel)


def arccos_kernel_deep(x1, x2, depth, var1=1, var2=None, covar=None):
    warnings.warn(
        'arccosine kernel computation is assuming that all inputs' + 
        ' have the same norm.')

    # check the first three input vectors and see whether they have the same norm
    # (not checking all vectors to save compute)
    if x1.shape[0] > 2:
        assert (torch.norm(x1[0]) - torch.norm(x1[1]))**2 < 1 / x1.shape[1]
        assert (torch.norm(x1[1]) - torch.norm(x1[2]))**2 < 1 / x1.shape[1]



    if var2 is None or covar is None:
        var2 = var1
        covar = var1
    kappa = covar / math.sqrt(var1 * var2)

    if 1 < kappa < 1 + 1e-5:
        kappa = 1
        # kappa > 1 will cause problems below. Sometimes due to floating point problems kappa is slightly bigger than 1.
    assert 0 <= kappa <= 1, f"kappa computed is {kappa}. Needs to be between 0 and 1"
    input_kernel = x1 @ x2.T / x1.shape[1]
    if depth == 0:
        return input_kernel
    normalized_norm_sq = torch.norm(x1[0]) * torch.norm(x2[0]) / x1.shape[1]
    cos_mat = input_kernel / normalized_norm_sq
    cos_mat = torch.clamp(cos_mat, -1, 1)

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


def k_ntk(x1, x2, depth, sigma=1, lamb=1e5):
    """
    This computes the infinite-width ReLU NTK kernel, using the identity between difference-of-GP kernels and NTK at large lambda.
    """
    lamb = lamb
    w_covar = compute_W_var(var1=sigma**2, lamb=lamb, n_tasks=2)
    return cross_kernel_new(x1, x2, 1, 1, w_covar, depth=depth, lamb=lamb, sigma=sigma)


def compute_forgetting_ops(x1, x2, y1, y2, depth, use_ntk_kernel=False):
    """
    Compute forgetting OPs (Oct 10 version). Use NNGP kernels by default.
    
    Returns:
        trp1p2/P = Tr(K1^{-1} K12 K2^{-1} K21) / P
        v1v2_cos = Y1.T K1^{-1} K12 K2^{-1} Y2 / sqrt(Y1.T K1^{-1} Y1) /
        sqrt(Y2.T K2^{-1} Y2))
    """

    # the kernel_fn takes x1, x2, depth
    kernel_fn = arccos_kernel_deep if use_ntk_kernel is False else k_ntk
    K1 = kernel_fn(x1, x1, depth)
    K2 = kernel_fn(x2, x2, depth)
    K12 = kernel_fn(x1, x2, depth)

    y_ref = torch.ones_like(y1) / torch.norm(y1)

    assert K1.shape == K2.shape
    P = K1.shape[0]
    trp1p2 = float(torch.trace(torch.inverse(K1) @ K12 @ \
                                torch.inverse(K2) @ K12.T)) / P

    v1_norm_sq = float(y1.T @ torch.inverse(K1) @ y1)
    v2_norm_sq = float(y2.T @ torch.inverse(K2) @ y2)
    v1v2 = float(y1.T @ torch.inverse(K1) @ K12 @ torch.inverse(K2) @ y2)
    v1v2_cos = v1v2 / np.sqrt(v1_norm_sq * v2_norm_sq)

    v1_ref_norm_sq = float(y_ref.T @ torch.inverse(K1) @ y_ref)
    v2_ref_norm_sq = float(y_ref.T @ torch.inverse(K2) @ y_ref)
    v1v2_ref = float(y_ref.T @ torch.inverse(K1) @ K12 @
                      torch.inverse(K2) @ y_ref)

    v1v2_cos_ref = v1v2_ref / np.sqrt(v1_ref_norm_sq * v2_ref_norm_sq)

    return trp1p2, v1v2_cos, v1v2_cos_ref


def compute_dec2024_ops(x1, x2, y1, y2, depth, use_ntk_kernel=False):
    """
    Compute forgetting OPs (Dec 2024 version). Use NNGP kernels by default.
    They are defined as:
    P_{12} = P_2 C_1 P_2 + P_1 C_2 P_1
    \gamma_{rf} = V_2^T P_{12} V_2 / P
    \gamma_{r} = V_2^T P_{12} V_1 / P

    They can alternatively be expressed in terms of the kernels as:
    \gamma_{rf} = ||K_{12} K_2^{-1} Y_2||^2 + ||K_{12}^T K_1^{-1} K_{12} K_2^{-1} Y_2||^2
    \gamma{r} = Y_2^T K_2^{-1} K_{12}^T K_{12} K_2^{-1} K_{12}^T K_1^{-1} Y_1 + Y_2^T K_2^{-1} K_{12}^T K_1^{-1} K_{12} K_{12}^T K_1^{-1} Y_1
    Returns:
        gamma_rf = ||K12 K2^{-1} Y2||^2 + ||K12^T K1^{-1} K12 K2^{-1} Y2||^2
        gamma_r = Y2^T K2^{-1} K12^T K12 K2^{-1} K12^T K1^{-1} Y1 + Y2^T K2^{-1} K12^T K1^{-1} K12 K12^T K1^{-1} Y1
    """

    # the kernel_fn takes x1, x2, depth
    kernel_fn = arccos_kernel_deep if use_ntk_kernel is False else k_ntk
    K1 = kernel_fn(x1, x1, depth)
    K2 = kernel_fn(x2, x2, depth)
    K12 = kernel_fn(x1, x2, depth)

    K1_inv = torch.inverse(K1)
    K2_inv = torch.inverse(K2)

    assert K1.shape == K2.shape
    P = K1.shape[0]

    gamma_rf = torch.norm(K12 @ K2_inv @ y2)**2 + torch.norm(K12.T @ K1_inv @ K12 @ K2_inv @ y2)**2
    gamma_r = y2.T @ K2_inv @ K12.T @ K12 @ K2_inv @ K12.T @ K1_inv @ y1 +\
        y2.T @ K2_inv @ K12.T @ K1_inv @ K12 @ K12.T @ K1_inv @ y1

    return gamma_rf / P, gamma_r / P

def get_gp_overlap(x1, x2, depth, epsilon=0):
    assert x1.shape == x2.shape
    P = x1.shape[0]
    K1 = arccos_kernel_deep(x1, x1, depth=depth)
    K2 = arccos_kernel_deep(x2, x2, depth=depth)
    K12 = arccos_kernel_deep(x1, x2, depth=depth)
    return torch.trace(torch.inverse(K1 + epsilon * torch.eye(P)) @ K12 @ \
                                        torch.inverse(K2 + epsilon * torch.eye(P)) @ K12.T) / P


def compute_single_task_test_loss(train_x, train_y, test_x, test_y,
                                depth: int):
    k_tr = arccos_kernel_deep(
        train_x, train_x, depth)
    test_preds = arccos_kernel_deep(
        test_x, train_x, depth) @ \
            torch.inverse(k_tr) @ train_y
    return torch.norm(
        test_preds.flatten() - test_y.flatten())**2 / torch.norm(test_y)**2

def get_single_task_test_losses(seq_of_train_x,
                                seq_of_train_y,
                                seq_of_test_x,
                                seq_of_test_y,
                                depth: int):
    """
    Compute test loss on each task from single-task learning.
    Accepts sequences of training/test data. 
    Expect data to have shape: x - (n_tasks, n_samples, n_features),
    y - (n_tasks, n_samples, 1)
    """
    assert len(seq_of_train_x) == len(seq_of_test_x)
    assert len(seq_of_train_x) == len(seq_of_train_y)
    assert len(seq_of_train_x) == len(seq_of_test_y)
    n_tasks = len(seq_of_train_x)

    test_losses = np.zeros(n_tasks)
    for task_ind in range(n_tasks):
        test_losses[task_ind] = compute_single_task_test_loss(
            seq_of_train_x[task_ind], seq_of_train_y[task_ind],
            seq_of_test_x[task_ind], seq_of_test_y[task_ind],
            depth)
    return test_losses