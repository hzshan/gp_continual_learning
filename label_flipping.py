import torch
import numpy as np

def dichotomize_in_place(y, class1_digits, class2_digits):
    """Simple function for dichotomizing the labels in place.
    y is assumed to be an array of integer digit labels (e.g., 0-9).
    Digits in class1_digits will be set to +1 and those in class2_digits will be set to -1.
    """
    if len(y) > 0 and len(np.unique(y)) != len(class1_digits) + len(class2_digits):
            raise ValueError(
                f"There is likely elements in digits not in class1_digits nor class2_digits.")
    for d in class1_digits:
        y[y == d] = 1.5
    for d in class2_digits:
        y[y == d] = -1.5
    y /= 1.5
    return y


def make_label_flipped_sequence(
          all_train_x, all_test_x, all_train_digits, all_test_digits,
          p, p_test, num_tasks, num_flipped_labels, max_total_class=None):
    """Generate a sequence of tasks with label flipping.
    Args:
    all_train_x: torch.Tensor of shape (n_train, n_features)
    all_test_x: torch.Tensor of shape (n_test, n_features)
    all_train_digits: torch.Tensor of shape (n_train,)
    all_test_digits: torch.Tensor of shape (n_test,)
    p: int, number of examples per task
    p_test: int, number of examples per task in the test set
    num_tasks: int, number of tasks
    num_flipped_labels: int, max number of labels to flip in each task plus 1
    max_total_class: int, maximum number of classes to use. If None, use all classes.
    """

    # in-place shuffle raw data
    tr_rand_inds = torch.randperm(len(all_train_x))
    te_rand_inds = torch.randperm(len(all_test_x))
    all_train_x = all_train_x[tr_rand_inds]
    all_train_digits = all_train_digits[tr_rand_inds]
    all_test_x = all_test_x[te_rand_inds]
    all_test_digits = all_test_digits[te_rand_inds]

    # cut out p * num_tasks samples for training and p_test * num_tasks samples for testing
    seq_of_train_x = all_train_x[:p * num_tasks].reshape(
        num_tasks, p, -1).double()
    seq_of_test_x = all_test_x[:p_test * num_tasks].reshape(
        num_tasks, p_test, -1).double()
    
    seq_of_train_y = all_train_digits[:p * num_tasks].reshape(
        num_tasks, p, 1).double()
    seq_of_test_y = all_test_digits[:p_test * num_tasks].reshape(
        num_tasks, p_test, 1).double()
    
    # evenly split digits into two classes. E.g. class 1 is 0-4, class 2 is 5-9
    # rule1_classes and rule2_classes are 2 x (n_class/2). First row are digits
    # in class 1, second row are digits in class 2.
    n_class = len(torch.unique(all_train_digits))

    if max_total_class is not None:
        assert max_total_class % 2 == 0, "max_total_class must be even"
        randomly_ordered_classes = torch.randperm(n_class)[:max_total_class]
    else:
        randomly_ordered_classes = torch.randperm(n_class)

    rule1_classes = randomly_ordered_classes.reshape(2, -1)
    rule2_classes = rule1_classes.clone()

    # flip some labels in the second rule
    rule2_classes[0, :num_flipped_labels] = \
        rule1_classes[1, :num_flipped_labels]
    rule2_classes[1, :num_flipped_labels] = \
        rule1_classes[0, :num_flipped_labels]
    rule1_classes = rule1_classes.reshape(2, -1)
    rule2_classes = rule2_classes.reshape(2, -1)

    # convert the labels in seq_of_xxx_y into +1 or -1 according the the rules
    # generated above. In each dataset, some inputs are labeled using one rule
    # and the rest are labeled using the other rule.
    for task_ind in range(num_tasks):
        # ratio of inputs to apply rule #1
        ratio = 1 - task_ind / (num_tasks - 1)
        rule1_p = int(p * ratio)
        rule1_p_test = int(p * ratio)

        dichotomize_in_place(seq_of_train_y[task_ind, :rule1_p],
                    rule1_classes[0], rule1_classes[1])

        dichotomize_in_place(seq_of_test_y[task_ind, :rule1_p_test],
                    rule1_classes[0], rule1_classes[1])
    
        dichotomize_in_place(seq_of_train_y[task_ind, rule1_p:],
                    rule2_classes[0], rule2_classes[1])

        dichotomize_in_place(seq_of_test_y[task_ind, rule1_p_test:],
                    rule2_classes[0], rule2_classes[1])
    
    return (seq_of_train_x, seq_of_train_y, seq_of_test_x, seq_of_test_y)


def replace_intermediate_datasets_with_mixed_data(
          seq_of_train_x, seq_of_train_y):
    """
    For task sequences of more than two tasks, replace the intermediate datasets
    with mixed data from the first and last datasets.
    """

    num_tasks = seq_of_train_x.shape[0]
    p = seq_of_train_x.shape[1]

    ratios = np.linspace(0, 1, num_tasks)
    if num_tasks > 2:
        for task_ind in range(1, num_tasks - 1):

            # zero the corresponding training data for easier debugging
            seq_of_train_x[task_ind] *= 0
            seq_of_train_y[task_ind] *= 0

            # compute the number of examples to take from the first and last tasks
            task2_p = int(p * ratios[task_ind])
            task1_p = p - task2_p
            task1_inds = torch.randperm(p)[:task1_p]
            task2_inds = torch.randperm(p)[:task2_p]

            # the first task1_p examples are from the first dataset
            seq_of_train_x[task_ind, :task1_p] = seq_of_train_x[0, task1_inds]
            seq_of_train_y[task_ind, :task1_p] = seq_of_train_y[0, task1_inds]

            # the rest are from the second dataset
            seq_of_train_x[task_ind, task1_p:] = seq_of_train_x[-1, task2_inds]
            seq_of_train_y[task_ind, task1_p:] = seq_of_train_y[-1, task2_inds]

    return seq_of_train_x, seq_of_train_y