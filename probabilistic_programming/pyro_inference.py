import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, NUTS, MCMC



class PyroMCMC:

    def __init__(self, model):
        self.model = model

    def mcmc(self, num_samples, warmup_steps):
        self.kernel = MCMC(
            NUTS(self.model, adapt_step_size=True), 
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=4
        )

        return self
    
    def run(self, X, y):
        self.kernel.run(X, y)

    

def main(model):
    pass


if __name__ == '__main__':
    main()


