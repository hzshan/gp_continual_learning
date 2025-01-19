import torch
import numpy as np
import tqdm, math

"""
Utility functions for running gradient-based learning (Langevin, SGD etc.)
"""


class MLP(torch.nn.Module):
    def __init__(self, N0, N, depth, output_dim, sigma, normalize='none'):
        super(MLP, self).__init__()
        self.N = N
        self.N0 = N0
        self.depth = depth
        self.output_dim = output_dim
        self.sigma = sigma

        self.Ls = torch.nn.ModuleList()
        self.Ls.append(torch.nn.Linear(N0, N, bias=False))
        for i in range(depth - 1):
            self.Ls.append(torch.nn.Linear(N, N, bias=False))

        self.readout = torch.nn.Linear(N, output_dim, bias=False)

        for p in list(self.parameters()):
            p.data = torch.normal(torch.zeros_like(p.data), std=sigma)

        self.anchor_parameters = None
        self.normalize = normalize

    def forward(self, x):

        for i in range(self.depth):
            if i == 0:
                width = self.N0
            else:
                width = self.N
            x = self.Ls[i](x) / math.sqrt(width)
            if self.normalize == 'none':
                pass
            elif self.normalize == 'layer':
                x -= torch.mean(x, dim=1, keepdim=True).data
                x /= torch.std(x, dim=1, keepdim=True).data + 1e-10
            elif self.normalize == 'full':
                x -= torch.mean(x).data
                x /= torch.std(x).data
            else:
                raise ValueError(
                    'normalization type for the network not understood.'
                    'need to be none/layer/full')
            x = torch.relu(x)

        return self.readout(x) / math.sqrt(self.N)

    def anchor(self, zero_anchor=False):
        if zero_anchor:
            self.anchor_parameters = [
                torch.zeros_like(p) for p in list(self.parameters())]
        else:
            self.anchor_parameters = [
                p.data.clone() for p in list(self.parameters())]


def langevin_step(model: MLP, train_x, train_y, lr, temp, l2, decay=0):
    """
    Implements GD with the option of including Langevin noise (controlled by `temp`)
    When `temp=0`, it is just gradient descent.
    """
    model.zero_grad()
    mse = torch.mean((model(train_x[:]).flatten() - train_y[:].flatten()) ** 2)

    if torch.isnan(mse.data):
        raise RuntimeError('training diverged')
    mse.backward()

    for p, anchor_p in zip(list(model.parameters()), model.anchor_parameters):
        p.data -= lr * (p.grad.data + l2 * (p.data - anchor_p) + decay * p.data)
        if temp > 0:
            p.data -= lr * math.sqrt(temp) * torch.normal(torch.zeros_like(p.data))
    return mse.data


def train(network, train_x, train_y,
         train_x_for_sampling, test_x_for_sampling,
          n_steps=5000,
          eta=0.001,
          l2=1,
          update_freq=1000,
          temp=0,
          convergence_threshold=-1,
          str_output_fn=print,
          decay=0,
          minibatch=-1,
          first_task=True,
          target_train_loss=1e-3):
    str_output_fn(
        f'\n \n ========================= Training starts ==========================')

    network.anchor(zero_anchor=first_task)
    # this saves the current parameters as the "anchors" for L2 regularization
    # if it is the first task, the anchor parameters are zeros, so it's just weight decay

    init_conv_threshold = convergence_threshold

    curr_best_loss = torch.ones(1).to(train_x.device) * 999

    rand_inds = torch.randperm(train_x.shape[0])

    max_dist_from_anchor = torch.zeros(1)

    for step in range(n_steps):

        tr_loss = None
        if minibatch < 0:
            tr_loss = langevin_step(model=network,
                                    train_x=train_x, train_y=train_y,
                                    lr=eta, temp=temp, l2=l2, decay=decay)
        else:
            num_minibatches = np.ceil(train_x.shape[0] / minibatch).astype(np.int64)
            mini_ind = step % num_minibatches
            inds = rand_inds[mini_ind * minibatch:(mini_ind + 1) * minibatch]
            _ = langevin_step(model=network,
                              train_x=train_x[inds], train_y=train_y[inds],
                              lr=eta, temp=temp, l2=l2, decay=decay)
            with torch.no_grad():
                tr_loss = torch.mean((network(train_x[:]).flatten() - train_y[:].flatten()) ** 2)

        with torch.no_grad():
            dist_from_anchor = torch.zeros(1)
            for p, anchor_p in zip(list(network.parameters()), network.anchor_parameters):
                dist_from_anchor += torch.norm(p - anchor_p).cpu()**2
            if dist_from_anchor > max_dist_from_anchor:
                max_dist_from_anchor = dist_from_anchor.clone()
            elif dist_from_anchor < max_dist_from_anchor:
                convergence_threshold -= 1
        
        if convergence_threshold < 0 and not first_task:
            str_output_fn(f'\n ***** training converged (by norm of weight changes). max norm {max_dist_from_anchor.item():.4f}')
            break
            
        if first_task and tr_loss < target_train_loss:
            str_output_fn(f'\n ***** training MSE less than {target_train_loss}.')
            break

        # if tr_loss < curr_best_loss:
        #     curr_best_loss = tr_loss.clone()

        # if tr_loss < target_train_loss:
        #     str_output_fn(f'\n ***** training MSE less than {target_train_loss}.')
        #     break

        # if init_conv_threshold > 0:
        #     if tr_loss.data > curr_best_loss:
        #         convergence_threshold -= 1
        #         if convergence_threshold < 0:
        #             if tr_loss > target_train_loss:
        #                 l2 = l2 * 0.8
        #                 decay = decay * 0.8
        #                 # str_output_fn(f'\n ***** training converged at loss {curr_best_loss:.4f}.'
        #                 #               f' Reducing L2 to {l2:.3E}.')

        #                 convergence_threshold = init_conv_threshold
        #             # str_output_fn(f'\n ***** training converged. best training loss {curr_best_loss:.4f}')
        #             # break

        if step % update_freq == 0:
            str_output_fn(f'{step} steps ||'
                          f' tr MSE:{torch.mean((network(train_x).flatten() - train_y.flatten()) ** 2):.4f}'
                          f' current l2 {l2}, current change norm {dist_from_anchor.item():.8f}')

    str_output_fn(f'\n Training finished for one task. Final training loss {float(tr_loss):.3f}.')
    fn_on_train = network(train_x_for_sampling)[:, 0]
    fn_on_test = network(test_x_for_sampling)[:, 0]

    # compute total parameter change
    total_param_change_norm = torch.zeros(1)
    for p, anchor_p in zip(list(network.parameters()), network.anchor_parameters):
        total_param_change_norm += torch.norm(p - anchor_p).cpu()**2

    str_output_fn(f'\n =========== Training ended after {step+1} steps; l2 {l2}, decay {decay},'
                  f' param change (sq norm): {float(total_param_change_norm.data.numpy()):.5f} =================')
    return fn_on_train, fn_on_test


def test(network, test_x, test_y):
    output_dim = test_y.shape[-1]
    with torch.no_grad():
        network_y = network(test_x.to(network.Ls[0].weight.device))
        test_y = test_y.float().to(network.Ls[0].weight.device)
        test_loss = torch.mean((network_y - test_y)**2)

        if output_dim > 1:
            test_acc = torch.mean(
                (torch.argmax(
                    network_y, dim=1
                    ) == torch.argmax(test_y, dim=1)).float())
        else:
            test_acc = torch.mean(
                (torch.sign(
                    network_y.flatten()
                    ) == torch.sign(test_y.flatten())).float())
    return test_loss, test_acc


def train_on_sequence(network, seq_of_train_x, seq_of_test_x, seq_of_train_y, seq_of_test_y,
                      learning_rate, num_steps, l2, update_freq=1000, temp=0, convergence_threshold=-1,
                      decay=0, logger=None, minibatch=-1, target_train_loss=1e-3):
    """
    network: torch.nn.Module model
    seq_of_train_x, seq_of_test_x: num_tasks * num_examples * input_dim
    seq_of_train_y, seq_of_test_y: num_tasks * num_examples * output_dim
    """

    if logger is None:
        str_output_fn = print
    else:
        str_output_fn = logger.log

    num_tasks = len(seq_of_train_x)
    train_loss_matrix = np.zeros((num_tasks, num_tasks))
    test_loss_matrix = np.zeros((num_tasks, num_tasks))
    train_acc_matrix = np.zeros((num_tasks, num_tasks))
    test_acc_matrix = np.zeros((num_tasks, num_tasks))
    sampled_fn_on_train_list = []
    sampled_fn_on_test_list = []

    for i in tqdm.trange(num_tasks, position=0, leave=True):
        # for the first task, set the l2 regularizer to 0
        fn_on_train, fn_on_test = train(
            network, seq_of_train_x[i],
            seq_of_train_y[i],
            train_x_for_sampling=seq_of_train_x[0],
            test_x_for_sampling=seq_of_test_x[0],
            eta=learning_rate, n_steps=num_steps, l2=0 if i == 0 else l2,
            update_freq=update_freq, temp=temp,
            convergence_threshold=convergence_threshold,
            str_output_fn=str_output_fn,
            first_task=i == 0,
            decay=decay,
            minibatch=minibatch,
            target_train_loss=target_train_loss)

        sampled_fn_on_train_list.append(fn_on_train)
        sampled_fn_on_test_list.append(fn_on_test)

        for j in range(num_tasks):
            test_loss, test_acc = test(network, seq_of_test_x[j], seq_of_test_y[j])
            train_loss, train_acc = test(network, seq_of_train_x[j], seq_of_train_y[j])
            train_loss_matrix[j, i] = train_loss
            test_loss_matrix[j, i] = test_loss
            train_acc_matrix[j, i] = train_acc
            test_acc_matrix[j, i] = test_acc

            if i == 0 and j == 0:
                if train_loss > target_train_loss:
                    print('!!!!!! Training did not appear to converge for the first task.')

        for _i, L in enumerate(network.Ls):
            logger.log(f'avg w_ij sq in layer {_i + 1} is {torch.mean(L.weight.data.cpu() ** 2)}')

    return (train_loss_matrix,
            test_loss_matrix,
            train_acc_matrix,
            test_acc_matrix,
            torch.stack(sampled_fn_on_train_list).cpu().data,
            torch.stack(sampled_fn_on_test_list).cpu().data
            )
