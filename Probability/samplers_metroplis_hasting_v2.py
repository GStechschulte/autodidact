import torch
from torch import distributions as dist
import matplotlib.pyplot as plt


def plot(distribution, trace_history, xmin, xmax, n_iterations, n_evals=500):

    x_evals = torch.linspace(xmin, xmax, n_evals)
    evals = torch.exp(distribution.log_prob(x_evals))
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(x_evals, evals)
    ax[1].plot(torch.arange(n_iterations), trace_history)
    
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

    x_current = x0

    x_samples = torch.zeros(n_iterations)
    x_samples[0] = x_current
     
    for n in range(1, n_iterations):

        x_candidate = x_current + tau + dist.Normal(loc=0, scale=1).sample()
        print(x_candidate)
        p_candidate = torch.exp(mixture.log_prob(x_candidate))
        print(p_candidate)
        p_current = torch.exp(mixture.log_prob(x_current))
        print(p_current)
        alpha = p_candidate / p_current
        print(alpha)
        probs_accept = min(1, alpha)
        print(probs_accept)
        #u = torch.rand(1)
        u = dist.Uniform(0, 1).sample()
        print(u)
        x_current = x_current if u >= probs_accept else x_candidate
        x_samples[n] = x_current


    return x_samples


def main():

    x0 = torch.tensor(20.) ## initial parameter value

    mixture_probs = torch.tensor([0.3, 0.7])
    mus = torch.tensor([-20., 20.])
    scales = torch.tensor([10., 10.])

    n_iters = 1000
    tau = torch.tensor(8.)

    mixture_distribution = true_distribution(mixture_probs, mus, scales)
    x_samples = metropolis_hasting(x0, tau, mixture_distribution, n_iters)

    plot(mixture_distribution, x_samples, -100, 100, n_iters, 500)
    


if __name__ == '__main__':
    main()