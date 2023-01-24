import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, NUTS, MCMC
import torch
import arviz as az

def process_data():

    data_path = "https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike.csv"
    raw_data_df = pd.read_csv(data_path)
    df = raw_data_df.copy().assign(
        date=pd.date_range(
            start=pd.to_datetime('2011-01-01'),
            end=pd.to_datetime('2012-12-31'),
            freq='D'
        )
    )

    #print(df)

    def design_matrix(df):

        # prepare features
        df['cnt_scaled'] = (df['cnt'] - df['cnt'].mean()) / (df['cnt'].std())
        df['temp_scaled'] = (df['temp'] - df['temp'].mean()) / (df['temp'].std())
        df['hum_scaled'] = (df['hum'] - df['hum'].mean()) / (df['hum'].std())
        df['windspeed_scaled'] = (df['windspeed'] - df['windspeed'].mean()) / (df['windspeed'].std())

        df['holiday'] = df['holiday'].astype('category').cat.codes
        df['workingday'] = df['workingday'].astype('category').cat.codes
        df['weathersit'] = df['weathersit'].astype('category').cat.codes
        t = df['days_since_2011'].to_numpy() / df['days_since_2011'].max()
        
        X = {
            'cnt_scaled': torch.tensor(df['cnt_scaled'].values, dtype=torch.float32),
            'temp_scaled': torch.tensor(df['temp_scaled'].values, dtype=torch.float32),
            'hum_scaled': torch.tensor(df['hum_scaled'].values, dtype=torch.float32),
            'windspeed_scaled': torch.tensor(df['windspeed_scaled'].values, dtype=torch.float32),
            'workingday': torch.tensor(df['workingday'].values, dtype=torch.long),
            'weathersit': torch.tensor(df['weathersit'].values, dtype=torch.long),
            't': torch.tensor(t, dtype=torch.float32),
        }
    
        return X
    
    X = design_matrix(df)
    y = torch.tensor(df['cnt_scaled'].values)
    
    return X, y


def base_model(X, y=None):

    workdays_i = len(torch.unique(X['workingday']))
    weather_i = len(torch.unique(X['weathersit']))
    obs_len = list(X.values())[0].shape[0]

    intercept = pyro.sample('intercept', dist.Normal(0, 2))
    b_temp = pyro.sample('b_temp', dist.Normal(0, 2))
    b_hum = pyro.sample('b_hum', dist.Normal(0, 2))
    b_windspeed = pyro.sample('b_windspeed', dist.Normal(0, 2))
    b_t = pyro.sample('b_t', dist.Normal(0, 3))
    nu = pyro.sample('nu', dist.Gamma(8, 2))
    sigma = pyro.sample('sigma', dist.HalfNormal(2))

    with pyro.plate('workday_i', workdays_i):
        b_workday = pyro.sample('b_workday', dist.Normal(0, 2))

    with pyro.plate('weathersit_i', weather_i):
        b_weather = pyro.sample('weather', dist.Normal(0, 2))

    with pyro.plate('rentals', obs_len):
        mu = pyro.deterministic(
            'mu',
            intercept
            + b_t * X['t']
            + b_temp * X['temp_scaled']
            + b_hum * X['hum_scaled']
            + b_windspeed * X['windspeed_scaled']
            + b_workday[X['workingday']] * X['workingday']
            + b_weather[X['weathersit']] * X['weathersit']
        )
        pyro.sample('likelihood', dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)


def time_varying_coef(X, y=None):

    workdays_i = len(torch.unique(X['workingday']))
    weather_i = len(torch.unique(X['weathersit']))
    time = list(X.values())[0].shape[0]
    obs_len = list(X.values())[0].shape[0]

    intercept = pyro.sample('intercept', dist.Normal(0, 2))
    b_hum = pyro.sample('b_hum', dist.Normal(0, 2))
    b_windspeed = pyro.sample('b_windspeed', dist.Normal(0, 2))
    b_t = pyro.sample('b_t', dist.Normal(0, 3))
    nu = pyro.sample('nu', dist.Gamma(8, 2))
    sigma = pyro.sample('sigma', dist.HalfNormal(2))
    sigma_slopes = pyro.sample('sigma_slope', dist.HalfNormal(2))

    with pyro.plate('workday_i', workdays_i):
        b_workday = pyro.sample('b_workday', dist.Normal(0, 2))

    with pyro.plate('weathersit_i', weather_i):
        b_weather = pyro.sample('weather', dist.Normal(0, 2))
    
    with pyro.plate('vary_t', time):
        b_temp = pyro.sample('b_temp_rw', dist.Normal(0, sigma_slopes))
        b_temp_rw = b_temp.cumsum(-1)

    with pyro.plate('rentals', obs_len):
        mu = pyro.deterministic(
            'mu',
            intercept
            + b_t * X['t']
            + b_temp_rw * X['temp_scaled']
            + b_hum * X['hum_scaled']
            + b_windspeed * X['windspeed_scaled']
            + b_workday[X['workingday']] * X['workingday']
            + b_weather[X['weathersit']] * X['weathersit']
        )
        pyro.sample('likelihood', dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)
    
    
def plot_prior_predictive(X, samples):

    prior_mean = torch.flatten(torch.mean(samples['likelihood'], 0, True))
    sns.lineplot(x=X['t'], y=prior_mean)
    plt.show()


def nuts_inference(model, X, y):

    mcmc = MCMC(NUTS(model, adapt_step_size=True), 500, 300, num_chains=4)
    mcmc.run(X, y)

    return mcmc


def main():
    
    X, y = process_data()
    #pyro.render_model(base_model, (X, y), render_distributions=True, filename='base_model.png')
    # pyro.render_model(
    #     time_varying_coef, 
    #     (X, y),
    #     render_distributions=True, 
    #     filename='time_varying.png'
    # )

    # prior predictive analyis
    #prior_samples = Predictive(base_model, {}, num_samples=1000)(X, None)
    prior_samples = Predictive(time_varying_coef, {}, num_samples=250)(X, None)
    #plot_prior_predictive(X, prior_samples)

    # inference
    mcmc = nuts_inference(time_varying_coef, X, y)
    print(mcmc.summary())

    # posterior predictive analysis
    posterior_samples = mcmc.get_samples(1000)
    post_pred_samples = Predictive(time_varying_coef, posterior_samples)(X, None)

    inferece_data = az.from_pyro(
        posterior=mcmc,
        prior=prior_samples,
        posterior_predictive=post_pred_samples
    )

    inferece_data.to_netcdf('time_vary_coef')


if __name__ == '__main__':
    main()