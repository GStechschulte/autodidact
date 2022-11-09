import torch
import pyro.distributions as dist
import pyro
import matplotlib.pyplot as plt
import seaborn as sns



def generate_data():

    true_rates = [40, 3, 20, 50]
    true_durations = [10, 20, 5, 35]

    emissions = torch.concatenate(
        [dist.Poisson(rate).expand([steps]).sample() for rate, steps in zip(true_rates, true_durations)]
        )

    plt.plot(torch.arange(0, len(emissions)), emissions)
    plt.show()
    

    


def main():
    
    generate_data()


if __name__ == '__main__':
    main()