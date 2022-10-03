import torch
from torch import distributions as dist
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

def plot(distribution, trace_history, xmin, xmax, n_iterations, n_evals=500):

    x_evals = torch.linspace(xmin, xmax, n_evals)
    evals = torch.exp(distribution.log_prob(x_evals))
    
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection='3d')

    ax.plot(torch.arange(n_iterations), trace_history)
    ax.plot(torch.zeros(n_evals), x_evals, evals)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Samples')
    
    plt.show()


def plot_trace(trace_history, n_iterations):

    plt.plot(torch.arange(n_iterations), trace_history)
    plt.show()


def true_distribution(mixture_probs, mus, scales):

    return dist.MixtureSameFamily(
        mixture_distribution=dist.Categorical(probs=mixture_probs),
        component_distribution=dist.Normal(loc=mus, scale=scales)
        )

def metropolis_hasting(x0, tau, mixture, n_iterations, rng_key=None):
    """
    implements the random walk metropolis-hasting algorithm
    """

    x_current = x0

    x_samples = torch.zeros(n_iterations)
    x_samples[0] = x_current
    cnt_acceptance = 0
     
    for n in range(1, n_iterations):
        
        print(f'iteration       = {n}')
        x_candidate = x_current + tau * dist.Normal(loc=0, scale=1).sample()
        print(f'x_candidate     = {x_candidate}')
        p_candidate = torch.exp(mixture.log_prob(x_candidate))
        print(f'p_candidate     = {p_candidate}')
        p_current = torch.exp(mixture.log_prob(x_current))
        print(f'p_current       = {p_current}')
        alpha = p_candidate / p_current
        print(f'alpha           = {alpha}')
        probs_accept = min(1, alpha)
        print(f'probs_accept    = {probs_accept}')
        u = dist.Uniform(0, 1).sample()
        print(f'uniform sample  = {u}', '\n')

        if u >= probs_accept:
            x_current = x_current
        else:
            x_current = x_candidate
            cnt_acceptance += 1

        x_samples[n] = x_current

    acceptence_probs = cnt_acceptance / n_iterations
    print('--- statistics ---')
    print(f'acceptance rate = {acceptence_probs}')

    return x_samples


def main(args):

    x0 = torch.tensor(20.) ## initial parameter value

    mixture_probs = torch.tensor([0.3, 0.7])
    mus = torch.tensor([-20., 20.])
    scales = torch.tensor([10., 10.])

    n_iters = args.iters
    tau = torch.tensor(args.tau)

    mixture_distribution = true_distribution(mixture_probs, mus, scales)
    x_samples = metropolis_hasting(x0, tau, mixture_distribution, n_iters)

    plot(mixture_distribution, x_samples, -100, 100, n_iters)
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='rw-mh')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=1.)
    args = parser.parse_args()

    main(args)