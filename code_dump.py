# def get_multihead_accuracy(seq_of_train_x, seq_of_test_x, seq_of_train_y_targets,
#                            seq_of_test_y_targets, sigma, lamb, depth,
#                            write_fn=None, tqdm_disable=False, n_heads=10, naive_gp=False):

#     """
#     Simple wrapper for computing accuracy of multiclass classification over a sequence of tasks.
#     """

#     warnings.warn('get_multihead_accuracy assumes that there are 10 classes.')
#     n_tasks = len(seq_of_train_x)
#     P = seq_of_train_x[0].shape[0]

#     # seq_of_train_y_onehot =\
#     #     torch.stack([digit_to_onehot(digit) for digit in seq_of_train_y_digit]).double()
#     # seq_of_test_y_onehot =\
#     #     torch.stack([digit_to_onehot(digit) for digit in seq_of_test_y_digit]).double()
#     all_train_predictions = torch.zeros((n_tasks, n_tasks, P, n_heads))
#     if seq_of_test_x is not None:
#         P_test = seq_of_test_x[0].shape[0]
#         all_test_predictions = torch.zeros((n_tasks, n_tasks, P_test, n_heads))  # task_ind * time_ind * P * num_heads
#     else:
#         P_test = 0
#         all_test_predictions = None

#     for head_ind in tqdm.trange(n_heads, position=0, disable=tqdm_disable):
#         if write_fn is not None:
#             write_fn(f'starting head {head_ind}')
#         training_predictions, test_predictions =\
#             theory.compute_mean_predictions(seq_of_train_x=seq_of_train_x,
#                                             seq_of_train_y=seq_of_train_y_targets[:, :, head_ind],
#                                             w_var=sigma**2, P_test=P_test,
#                                             lambda_val=lamb,
#                                             seq_of_test_x=seq_of_test_x, depth=depth, use_naive_gp=naive_gp)
#         # test_predictions: task_ind * time_ind * test_input_ind

#         all_train_predictions[:, :, :, head_ind] = training_predictions.squeeze()

#         if seq_of_test_x is not None:
#             all_test_predictions[:, :, :, head_ind] = test_predictions.squeeze()

#     accuracy_forgetting_matrix_train = get_accuracy_matrix(all_train_predictions, seq_of_train_y_targets)
#     train_loss_mat = get_loss_mat(all_train_predictions, seq_of_train_y_targets)
#     if seq_of_test_x is not None:
#         accuracy_forgetting_matrix_test = get_accuracy_matrix(all_test_predictions, seq_of_test_y_targets)
#         test_loss_mat = get_loss_mat(all_test_predictions, seq_of_test_y_targets)
#     else:
#         accuracy_forgetting_matrix_test = None
#         test_loss_mat = None

#     return accuracy_forgetting_matrix_train, accuracy_forgetting_matrix_test, train_loss_mat, test_loss_mat


# def get_accuracy_matrix(net_predictions, target_vectors):
#     """
#     :param net_predictions: # task_ind * time_ind * P * num_heads
#     :param target_vectors: # task_ind * P * num_heads * 1
#     :return:
#     """
#     num_tasks = target_vectors.shape[0]
#     num_heads = target_vectors.shape[2]
#     accuracy_mat = torch.zeros((num_tasks, num_tasks))  # task_ind * time_ind
#     if num_heads > 1:
#         # if there are more than one heads, use multihead classification scheme
#         target_digits = torch.argmax(target_vectors, dim=2).squeeze()
#         assert net_predictions.shape[0] == num_tasks
#         net_predictions_in_digits = torch.argmax(net_predictions, dim=-1)
#         for task_ind in range(num_tasks):
#             for time_ind in range(num_tasks):
#                 _dummy = torch.zeros(target_digits[task_ind].shape)
#                 _dummy[net_predictions_in_digits[task_ind, time_ind] ==
#                                             target_digits[task_ind]] = 1
#                 accuracy_mat[task_ind, time_ind] =\
#                     torch.mean(_dummy.float())

#     else:
#         # use binary classification scheme (assuming the target vectors are +/-1)
#         for task_ind in range(num_tasks):
#             for time_ind in range(num_tasks):
#                 accuracy_mat[task_ind, time_ind] =\
#                     torch.mean(
#                         torch.heaviside(
#                             net_predictions[task_ind, time_ind].flatten() * target_vectors[task_ind].flatten(),
#                             values=torch.zeros(1).double()))
#     return accuracy_mat



# def digit_to_onehot(digit_targets):
#     warnings.warn('digit_to_onehot will be removed. use torch nn functional instead')
#     onehot_targets = np.zeros((digit_targets.shape[0], 10, 1))
#     for i in range(digit_targets.shape[0]):
#         onehot_targets[i, digit_targets[i], 0] = 1
#     return onehot_targets


# def _digit_to_even_odd(digit_y):
#     """
#     Convert digit-based targets from MNIST to odd/even (marked with +1/-1)
#     :param digit_y:
#     :return:
#     """
#     odd_indices = digit_y % 2 == 1
#     even_indices = digit_y % 2 == 0

#     digit_y[odd_indices] = 1
#     digit_y[even_indices] = -1
#     return digit_y


# def prepare_sequential_dataset(num_tasks, train_p, test_p, resample,
#                                dataset_name, data_path=None, permutation=0, n_epochs=1,
#                                interpolate=False, precision=64):
#     """
#     Prepare a sequence of tasks in the format of 6 lists of length num_tasks.
#     """
#     assert 0 <= permutation <= 1
#     seq_of_train_x = []
#     seq_of_test_x = []
#     seq_of_train_digits = []
#     seq_of_test_digits = []
#     if resample is False:
#         all_train_x, all_test_x, all_train_digits, all_test_digits =\
#             load_dataset(dataset_name=dataset_name, num_train=train_p, num_test=test_p, path=data_path)
#     else:
#         all_train_x, all_test_x, all_train_digits, all_test_digits =\
#             load_dataset(dataset_name=dataset_name, num_train=int(train_p * num_tasks),
#                          num_test=int(test_p * num_tasks), path=data_path)

#     for _task_ind in range(num_tasks):
#         if resample:
#             train_x = all_train_x[_task_ind*train_p:(_task_ind+1)*train_p]
#             test_x = all_test_x[_task_ind*test_p:(_task_ind+1)*test_p]
#             train_y_digit = all_train_digits[_task_ind*train_p:(_task_ind+1)*train_p]
#             test_y_digit = all_test_digits[_task_ind*test_p:(_task_ind+1)*test_p]
#         else:
#             train_x = all_train_x
#             test_x = all_test_x
#             train_y_digit = all_train_digits
#             test_y_digit = all_test_digits

#         seq_of_train_x.append(train_x)
#         seq_of_test_x.append(test_x)
#         seq_of_train_digits.append(train_y_digit)
#         seq_of_test_digits.append(test_y_digit)

#     N0 = seq_of_train_x[0].shape[1]

#     if permutation > 0:
#         if interpolate:
#             # current_perm_mat = torch.eye(N0)
#             # for _task_ind in range(num_tasks):
#             #     current_perm_mat = current_perm_mat @ utils.get_permutation_mat(N0, strength=permutation)
#             #     seq_of_train_x[_task_ind] = seq_of_train_x[_task_ind] @ current_perm_mat
#             #     seq_of_test_x[_task_ind] = seq_of_test_x[_task_ind] @ current_perm_mat
#             weights = np.linspace(0, 1, num_tasks)[::-1]
#             perm_mat1 = utils.get_permutation_mat(N0, strength=permutation)
#             perm_mat2 = utils.get_permutation_mat(N0, strength=permutation)
#             for _task_ind in range(num_tasks):
#                 perm_mat = perm_mat1 * weights[_task_ind] + perm_mat2 * (1 - weights[_task_ind])
#                 seq_of_train_x[_task_ind] = seq_of_train_x[_task_ind] @ perm_mat
#                 seq_of_test_x[_task_ind] = seq_of_test_x[_task_ind] @ perm_mat
#         else:
#             for _task_ind in range(num_tasks):
#                 permutation_mat = utils.get_permutation_mat(N0, strength=permutation)
#                 seq_of_train_x[_task_ind] = seq_of_train_x[_task_ind] @ permutation_mat
#                 seq_of_test_x[_task_ind] = seq_of_test_x[_task_ind] @ permutation_mat

#     # normalize all train and test input to have the same norm

#     seq_of_train_x = torch.tile(torch.stack([utils.normalize_input(task_x) for task_x in seq_of_train_x]),
#                                 (n_epochs, 1))
#     seq_of_test_x = torch.tile(torch.stack([utils.normalize_input(task_x) for task_x in seq_of_test_x]),
#                                (n_epochs, 1))

#     seq_of_train_digits = torch.tile(torch.stack(seq_of_train_digits), (n_epochs, 1))
#     seq_of_test_digits = torch.tile(torch.stack(seq_of_test_digits), (n_epochs, 1))

#     seq_of_train_targets = batch_digit_to_onehot(seq_of_train_digits, num_classes=10)
#     seq_of_test_targets = batch_digit_to_onehot(seq_of_test_digits, num_classes=10)

#     assert precision in [16, 32, 64]
#     if precision == 64:
#         return seq_of_train_x.double(), seq_of_test_x.double(),\
#                seq_of_train_targets.double(), seq_of_test_targets.double()
#     elif precision == 32:
#         return seq_of_train_x.float(), seq_of_test_x.float(),\
#                seq_of_train_targets.float(), seq_of_test_targets.float()
#     elif precision == 16:
#         return seq_of_train_x.half(), seq_of_test_x.half(),\
#                seq_of_train_targets.half(), seq_of_test_targets.half()


def compute_predictor_variances(seq_of_train_x: list, w_var: float, P_test: int, depth: int,
                                lambda_val: float, seq_of_test_x, large_lambda=False):
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
    weight_covar = compute_W_var(w_var, lambda_val, num_tasks, fix_w=large_lambda)
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