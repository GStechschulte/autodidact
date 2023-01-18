import numpy as np
import torch
from torch import distributions as dist
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")


def plot_samples(x_samples_gibbbs, z_samples_gibbs, n_iterations):

    colors = ["tab:blue" if z else "tab:red" for z in z_samples_gibbs]
    fig, axs = plt.subplots()

    axs.scatter(
        np.arange(n_iterations), x_samples_gibbbs, 
        s=3, facecolors="none", edgecolors=colors
        )
    axs.set_xlabel("Sample number")
    axs.set_ylabel("Value of $x$")
    axs.set_title("Trace History of Samples")

    plt.show()


def plot_gibbs(trace_hist, probs, scales, mus, xmin, xmax, n_iters, n_eval=500):
    
    norm_mixture = dist.MixtureSameFamily(
        mixture_distribution=dist.Categorical(probs=probs),
        component_distribution=dist.Normal(loc=mus, scale=scales)
        )

    x_eval = torch.linspace(xmin, xmax, n_eval)
    px = torch.exp(norm_mixture.log_prob(x_eval))
    sample_px = torch.exp(norm_mixture.log_prob(trace_hist))
    sorted_trace, indices = torch.sort(trace_hist)

    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection='3d')

    ax.plot(torch.arange(n_iters), trace_hist)
    ax.plot(torch.zeros(n_eval), x_eval, px, c="tab:red", linewidth=2)
    ax.plot(torch.zeros(n_iters), sorted_trace, sample_px[indices], c="tab:blue")

    ax.set_xlabel("Iterations", fontsize=None)
    ax.set_ylabel("Samples", fontsize=None)
    plt.show()


def gibbs_sampler(x0, z0, kv, probs, mu, scale, n_iterations, rng_key=None):
    """
    implements the gibbs sampling algorithm for known params. of
    a 1d GMM
    """

    x_current = x0
    z_current = z0

    x_samples = torch.zeros(n_iterations)
    z_samples = torch.zeros(n_iterations)
     
    for n in range(1, n_iterations):

        # likelihood --> p(Z = k | X = x)
        probs_z = torch.exp(dist.Normal(loc=mu, scale=scale).log_prob(x_current))
        # prior * likelihood --> p(X = x | Z = k) * p(Z = k)
        probs_z *= probs
        # denom. of Bayes
        probs_z = probs_z / torch.sum(probs_z)
        z_current = kv[-1] if probs_z[-1] > probs[0] else kv[0]
        # draw new sample X conditioned on k
        x_current = dist.Normal(loc=mu[z_current], scale=scale[z_current]).sample()

        x_samples[n] = x_current
        z_samples[n] = z_current


    return x_samples, z_samples


def main(args):

    # initial parameter values
    x0, z0 = torch.tensor(20.), torch.tensor(0.)
    # for indexing
    kv = np.arange(2)

    mixture_probs = torch.tensor([0.3, 0.7])
    mus = torch.tensor([-20., 20.])
    scales = torch.tensor([10., 10.])

    x_samples, z_samples = gibbs_sampler(
        x0, z0, kv, mixture_probs, mus, scales, args.iters
        )

    plot_gibbs(x_samples, mixture_probs, scales, mus, -100, 100, args.iters)
    plot_samples(x_samples, z_samples, args.iters)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='rw-mh')
    parser.add_argument('--iters', type=int, default=1000)
    args = parser.parse_args()

    main(args)