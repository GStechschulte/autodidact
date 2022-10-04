import numpy as np
import torch
from torch import distributions as dist
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")


def plot(x_samples_gibbbs, z_samples_gibbs, n_iterations, n_evals=500):

    colors = ["tab:blue" if z else "tab:red" for z in z_samples_gibbs]
    fig, axs = plt.subplots()

    axs.scatter(
        np.arange(n_iterations), x_samples_gibbbs, 
        s=3, facecolors="none", edgecolors=colors
        )
    axs.set_xlabel("Sample number")
    axs.set_ylabel("Value of $x$")

    plt.show()



def plot_gibbs(trace_hist, probs, scales, mus, xmin, xmax, n_iters, n_eval=500):
    
    norm_mixture = dist.MixtureSameFamily(
        mixture_distribution=dist.Categorical(probs=probs),
        component_distribution=dist.Normal(loc=mus, scale=scales)
        )

    x_eval = torch.linspace(xmin, xmax, n_eval)
    #kde_eval = pml.kdeg(x_eval[:, None], trace_hist[:, None], h)
    px = torch.exp(norm_mixture.log_prob(x_eval))

    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection='3d')

    ax.plot(torch.arange(n_iters), trace_hist)
    ax.plot(torch.zeros(n_eval), x_eval, px, c="tab:red", linewidth=2)
    #ax.plot(np.zeros(n_eval), x_eval, kde_eval, c="tab:blue")

    #ax.set_zlim(0, kde_eval.max() * 1.2)
    ax.set_xlabel("Iterations", fontsize=None)
    ax.set_ylabel("Samples", fontsize=None)

    plt.show()


def true_distribution(mixture_probs, mus, scales):

    return dist.MixtureSameFamily(
        mixture_distribution=dist.Categorical(probs=mixture_probs),
        component_distribution=dist.Normal(loc=mus, scale=scales)
        )

def gibbs_sampler(x0, z0, kv, probs, mu, scale, n_iterations, rng_key=None):
    """
    implements the gibbs sampling algorithm for known params. of
    a GMM
    """

    x_current = x0
    z_current = z0

    x_samples = torch.zeros(n_iterations)
    z_samples = torch.zeros(n_iterations)
     
    for n in range(1, n_iterations):

        # new_x = dist.Normal(curr_y / 2, torch.sqrt(3/4))
        # new_y = dist.Normal(new_x / 2, torch.sqrt(3/4))

        # num. of Bayes
        probs_z = torch.exp(dist.Normal(loc=mu, scale=scale).log_prob(x_current))
        probs_z *= probs
        # denom. of Bayes
        probs_z = probs_z / torch.sum(probs_z)
        #print(probs_z)

        # z_current = np.random.choice(keys[n], kv, probs_z)

        if probs_z[-1] > probs_z[0]:
             z_current =  kv[-1]
        else:
            z_current = kv[0]

        x_current = dist.Normal(loc=mu[z_current], scale=scale[z_current]).sample()

        x_samples[n] = x_current
        z_samples[n] = z_current


    return x_samples, z_samples


def main(args):

    x0, z0 = torch.tensor(20.), torch.tensor(0.) ## initial parameter value
    kv = np.arange(2)

    mixture_probs = torch.tensor([0.3, 0.7])
    mus = torch.tensor([-20., 20.])
    scales = torch.tensor([10., 10.])

    n_iters = args.iters
    tau = torch.tensor(args.tau)

    #mixture_distribution = true_distribution(mixture_probs, mus, scales)
    x_samples, z_samples = gibbs_sampler(x0, z0, kv, mixture_probs, mus, scales, n_iters)

    plot_gibbs(x_samples, mixture_probs, scales, mus, -100, 100, n_iters)
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='rw-mh')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=1.)
    args = parser.parse_args()

    main(args)