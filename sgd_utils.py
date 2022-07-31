import torch
import numpy as np
import tqdm, math


USE_ADAM = False


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


def train(network, train_x, train_y, n_steps=5000, eta=0.001, l2=1, update_freq=1000):
    train_x = train_x.to(network.Ls[0].weight.device)
    train_y = train_y.float().to(network.Ls[0].weight.device)

    # store pre-training parameters
    pre_train_parameters = [p.data.clone() for p in list(network.parameters())]

    loss = None
    curr_best_loss = torch.ones(1).to(train_x.device) * 999
    convergence_counter = 5

    if USE_ADAM:
        optim = torch.optim.Adam(network.parameters(), lr=eta)
    else:
        optim = None

    curr_best_loss = torch.ones(1).to(train_x.device) * 999
    convergence_counter = 5
    for step in range(n_steps):
        network.zero_grad()
        loss = torch.mean((network(train_x[:]) - train_y[:]) ** 2)

        if USE_ADAM:
            for p, init_p in zip(list(network.parameters()), pre_train_parameters):
                loss += l2 * torch.sum((p - init_p)**2)
            loss.backward()
            optim.step()
        else:
            loss.backward()
            for p, init_p in zip(list(network.parameters()), pre_train_parameters):
                p.data -= eta * (p.grad.data + l2 * (p.data - init_p))

        if torch.isnan(loss.data):
            raise RuntimeError('training diverged')
        else:
            if loss.data < curr_best_loss:
                curr_best_loss = loss.data.clone()

        if loss.data < 0.001:
            print('\n ***** training loss less than 0.001. Breaking.')
            break

            # if loss.data > curr_best_loss:
            #     convergence_counter -= 1
            #     if convergence_counter < 0:
            #         print(f'\n ***** training converged. best training loss {curr_best_loss:.3f}')
            #         break

        if step % update_freq == 0:
            print(f'training loss:{torch.mean((network(train_x) - train_y) ** 2):.3f}')

    sum_of_p_changes = torch.sum(torch.tensor([torch.sum(p - init_p)**2 for p, init_p in
                                               zip(network.parameters(), pre_train_parameters)]))
          f'\n Training finished for one task. Final training loss {float(loss.data):.3f} ')
    for p ,init_p in zip(network.parameters(), pre_train_parameters):
        print(p.shape, f'mean weight change:{float(torch.mean((p - init_p)**2)):.3f}')
    print(f'\n ====================================================')


def test(network, test_x, test_y):
    with torch.no_grad():
        network_y = network(test_x.to(network.Ls[0].weight.device))
        test_y = test_y.float().to(network.Ls[0].weight.device)
        test_loss = torch.mean((network_y - test_y)**2)
        test_acc = torch.mean((torch.argmax(network_y, dim=1) == torch.argmax(test_y, dim=1)).float())
    return test_loss, test_acc


def train_on_sequence(network, seq_of_train_x, seq_of_test_x, seq_of_train_y_digit, seq_of_test_y_digit,
                      learning_rate, num_steps, l2, update_freq=1000):
    num_tasks = len(seq_of_train_x)
    train_loss_matrix = np.zeros((num_tasks, num_tasks))
    test_loss_matrix = np.zeros((num_tasks, num_tasks))
    train_acc_matrix = np.zeros((num_tasks, num_tasks))
    test_acc_matrix = np.zeros((num_tasks, num_tasks))

    for i in tqdm.trange(num_tasks, position=0, leave=True):
        # for the first task, set the l2 regularizer to 0
        print(f'\n ================= Start task {i+1} / {num_tasks} ==================')
        train(network, seq_of_train_x[i],
              seq_of_train_y_digit[i].long(), eta=learning_rate, n_steps=num_steps, l2=0 if i == 0 else l2,
              update_freq=update_freq)
        for j in range(num_tasks):
            test_loss, test_acc = test(network, seq_of_test_x[j], seq_of_test_y_digit[j].long())
            train_loss, train_acc = test(network, seq_of_train_x[j], seq_of_train_y_digit[j].long())
            train_loss_matrix[j, i] = train_loss
            test_loss_matrix[j, i] = test_loss
            train_acc_matrix[j, i] = train_acc
            test_acc_matrix[j, i] = test_acc

            if i == 0 and j == 0:
                if train_loss > 2e-3:
                    raise RuntimeError('Training did not appear to converge for the first task.')
            # if j == i:
            #     print(f'train loss{train_loss:.3f}, test loss{test_loss:.3f}')

    return train_loss_matrix, test_loss_matrix, train_acc_matrix, test_acc_matrix