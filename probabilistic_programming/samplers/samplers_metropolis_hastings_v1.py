import numpy as np
from scipy import stats
import argparse
import matplotlib.pyplot as plt

"""
Implements the Metropolis-Hastings algorithm for
the beta-binomial model
"""


def plot_trace(trace):
    
    plt.plot(np.arange(0, len(trace['theta'])), trace['theta'])
    plt.xlabel('iteration')
    plt.ylabel('$\\theta$')
    plt.show()


def posterior(theta, y, alpha, beta):
    
    if 0 <= theta <= 1:
        prior = stats.beta(alpha, beta).pdf(theta)
        likelihood = stats.bernoulli(theta).pmf(y).prod()
        probs = likelihood * prior
    else:
        probs = -np.inf
    
    return probs


def main(args):
    
    y = stats.bernoulli(0.7).rvs(20)

    can_sd = 0.05
    alpha = beta = 1
    theta = 0.5
    trace = {'theta': np.zeros(args.iters)}
    
    p2 = posterior(theta, y, alpha, beta)

    for iter in range(args.iters):
        theta_can = stats.norm(theta, can_sd).rvs(1)
        p1 = posterior(theta_can, y, alpha, beta)
        probs_accept = p1 / p2

        if probs_accept > stats.uniform(0, 1).rvs(1):
            theta = theta_can
            p2 = p1
        
        trace['theta'][iter] = theta

    plot_trace(trace)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='metropolis-hastings')
    parser.add_argument('--iters', type=int, default=1000)
    args = parser.parse_args()

    main(args)
