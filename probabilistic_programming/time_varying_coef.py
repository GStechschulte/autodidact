import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyro
import pyro.distributions as dist
import torch


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
            'cnt_scaled': torch.tensor(df['cnt_scaled'].values),
            'temp_scaled': torch.tensor(df['temp_scaled'].values),
            'hum_scaled': torch.tensor(df['hum_scaled'].values),
            'windspeed_scaled': torch.tensor(df['windspeed_scaled'].values),
            'workingday': torch.tensor(df['workingday'].values, dtype=torch.long),
            'weathersit': torch.tensor(df['weathersit'].values, dtype=torch.long),
            't': torch.tensor(t),
        }

        # X = torch.tensor(df[[
        #     'temp_scaled', 'hum_scaled', 'windspeed_scaled',
        #     'holiday', 'workingday', 'weathersit'
        #     ]].values)
        #y = torch.tensor(df['cnt_scaled'].values)
    
        return X
    
    X = design_matrix(df)
    y = torch.tensor(df['cnt_scaled'].values)
    
    return X, y

def base_model(X, y=None):

    workdays_i = torch.unique(X['workingday'])
    weather_i = torch.unique(X['weathersit'])
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

    with pyro.plate('rentals', obs_len):
        pyro.sample('obs', dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)



def main():
    
    X, y = process_data()

    #print(list(X.values())[0].shape)

    #print(torch.unique(X['workingday']))
    
    pyro.render_model(base_model, (X, y), render_distributions=True, filename='base_model.png')


if __name__ == '__main__':
    main()


