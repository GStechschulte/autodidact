from random import uniform
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    
    square_func = lambda x: x**2

    ## True p(x) ##
    lower, upper = -1, 1
    x_samples = np.linspace(lower, upper, 200)
    probs_x = 1 / (upper - lower) * np.ones(len(x_samples)) ## p(X = x)
    true_y = square_func(x_samples)
    pdf_y = 1 / (2 * np.sqrt(true_y + 1e-2)) ## add small buffer

    ## Approximation p(x) ##
    uniform_samples = np.random.uniform(-1, 1, 1000)
    approx_y = square_func(uniform_samples)

    ## Plotting ##
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].set_title('True Uniform Distribution')
    ax[0].plot(x_samples, probs_x)
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$p(x)$')

    ax[1].set_title('True $y$ PDF')
    ax[1].plot(true_y, pdf_y)
    ax[1].set_xlabel('$y$')
    ax[1].set_ylabel('$p(y)$')

    ax[2].set_title('Approximated $y$ PDF')
    ax[2].hist(approx_y, bins=30, density=True)
    ax[2].set_xlabel('$y$')
    ax[2].set_ylabel('$p(y)$')

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()