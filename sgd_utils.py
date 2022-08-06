import torch
import numpy as np
import tqdm, math

"""
Utility functions for running gradient-based learning (Langevin, SGD)
"""

TRAIN_MSE_THRESHOLD = 1e-4


class MLP(torch.nn.Module):
    def __init__(self, N0, N, depth, n_heads, sigma):
        super(MLP, self).__init__()
        self.N = N
        self.N0 = N0
        self.depth = depth
        self.n_heads = n_heads
        self.sigma = sigma

        self.Ls = torch.nn.ModuleList()
        self.Ls.append(torch.nn.Linear(N0, N, bias=False))
        for i in range(depth - 1):
            self.Ls.append(torch.nn.Linear(N, N, bias=False))

        self.readout = torch.nn.Linear(N, n_heads, bias=False)

        for p in list(self.parameters()):
            p.data = torch.normal(torch.zeros_like(p.data), std=sigma)

        self.anchor_parameters = None

    def forward(self, x):

        for i in range(self.depth):
            if i == 0:
                width = self.N0
            else:
                width = self.N
            x = self.Ls[i](x)
            x -= torch.mean(x).data
            x /= torch.std(x).data
            x = torch.relu(x)

        return self.readout(x) / math.sqrt(self.N)

    def anchor(self):
        self.anchor_parameters = [p.data.clone() for p in list(self.parameters())]


def langevin_step(model: MLP, train_x, train_y, lr, temp, l2):
    model.zero_grad()
    mse = torch.mean((model(train_x[:]) - train_y[:]) ** 2)

    if torch.isnan(mse.data):
        raise RuntimeError('training diverged')
    mse.backward()
    for p, anchor_p in zip(list(model.parameters()), model.anchor_parameters):
        p.data -= lr * (p.grad.data + l2 * (p.data - anchor_p))
        if temp > 0:
            p.data -= lr * math.sqrt(temp) * torch.normal(torch.zeros_like(p.data))
    return mse.data


def train(network, train_x, train_y, test_x,
          n_steps=5000, eta=0.001, l2=1, update_freq=1000, temp=0, num_samples=0,
          convergence_threshold=-1, str_output_fn=print):

    network.anchor()
    mse = None
    sampled_outputs = torch.zeros((num_samples, test_x.shape[0]))

    curr_best_loss = torch.ones(1).to(train_x.device) * 999
    for step in range(n_steps):
        mse = langevin_step(model=network, train_x=train_x, train_y=train_y, lr=eta, temp=temp, l2=l2)

        if mse < curr_best_loss:
            curr_best_loss = mse.clone()

        if mse < TRAIN_MSE_THRESHOLD:
            str_output_fn(f'\n ***** training MSE less than {TRAIN_MSE_THRESHOLD}. Starting to sample.')
            break

        if convergence_threshold > 0:
            if mse.data > curr_best_loss:
                convergence_threshold -= 1
                if convergence_threshold < 0:
                    if mse > TRAIN_MSE_THRESHOLD:
                        str_output_fn(f'\n ***** training converged at loss {curr_best_loss:.4f}. Reducing L2.')
                        l2 = l2 * 2/3
                    str_output_fn(f'\n ***** training converged. best training loss {curr_best_loss:.4f}')
                    break

        if step % update_freq == 0:
            str_output_fn(f'{step} steps || tr MSE:{torch.mean((network(train_x) - train_y) ** 2):.4f}')

    str_output_fn(f'\n Training finished for one task. Final training loss {float(mse):.3f}. Starting to sample.')
    for sample_ind in range(num_samples):
        for step_ind in range(1):
            _ = langevin_step(model=network, train_x=train_x, train_y=train_y, lr=eta, temp=temp, l2=l2)

        with torch.no_grad():
            sampled_outputs[sample_ind] = network(test_x)[:, 0]


    str_output_fn(f'\n =========== Training ended after {step+1} steps =================')
    return sampled_outputs


def test(network, test_x, test_y):
    with torch.no_grad():
        network_y = network(test_x.to(network.Ls[0].weight.device))
        test_y = test_y.float().to(network.Ls[0].weight.device)
        test_loss = torch.mean((network_y - test_y)**2)
        test_acc = torch.mean((torch.argmax(network_y, dim=1) == torch.argmax(test_y, dim=1)).float())
    # save the test predictions from head #0
    return test_loss, test_acc


def train_on_sequence(network, seq_of_train_x, seq_of_test_x, seq_of_train_y_digit, seq_of_test_y_digit,
                      learning_rate, num_steps, l2, update_freq=1000, temp=0, convergence_threshold=-1, logger=None):

    if logger is None:
        str_output_fn = print
    else:
        str_output_fn = logger.log

    num_tasks = len(seq_of_train_x)
    train_loss_matrix = np.zeros((num_tasks, num_tasks))
    test_loss_matrix = np.zeros((num_tasks, num_tasks))
    train_acc_matrix = np.zeros((num_tasks, num_tasks))
    test_acc_matrix = np.zeros((num_tasks, num_tasks))
    samples_across_seq = []

    for i in tqdm.trange(num_tasks, position=0, leave=True):
        # for the first task, set the l2 regularizer to 0
        print(f'\n =================================================')

        samples_across_seq.append(train(network, seq_of_train_x[i],
                                        seq_of_train_y_digit[i].long(),
                                        test_x=seq_of_test_x[0],
                                        eta=learning_rate, n_steps=num_steps, l2=0 if i == 0 else l2,
                                        update_freq=update_freq, temp=temp, num_samples=1,
                                        convergence_threshold=convergence_threshold, str_output_fn=str_output_fn))

        for j in range(num_tasks):
            test_loss, test_acc = test(network, seq_of_test_x[j], seq_of_test_y_digit[j].long())
            train_loss, train_acc = test(network, seq_of_train_x[j], seq_of_train_y_digit[j].long())
            train_loss_matrix[j, i] = train_loss
            test_loss_matrix[j, i] = test_loss
            train_acc_matrix[j, i] = train_acc
            test_acc_matrix[j, i] = test_acc

            if i == 0 and j == 0:
                if train_loss > TRAIN_MSE_THRESHOLD:
                    print('!!!!!! Training did not appear to converge for the first task.')
            # if j == i:
            #     print(f'train loss{train_loss:.3f}, test loss{test_loss:.3f}')

    return train_loss_matrix, test_loss_matrix, train_acc_matrix, test_acc_matrix, torch.stack(samples_across_seq)