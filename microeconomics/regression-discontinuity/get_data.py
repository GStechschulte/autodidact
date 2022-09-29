import pandas as pd

class chicago_ride_hail():

    def __init__(self, samples=1000):
        self.url = 'https://data.cityofchicago.org/resource/m6dm-c72p.json?$limit={}'.format(samples)

    def fetch_data(self):
        self.data = pd.read_json(self.url)
        return self.data

    def datetime(self):
        self.data.trip_start_timestamp = pd.to_datetime(self.data.trip_start_timestamp, utc=True)
        self.data.trip_end_timestamp = pd.to_datetime(self.data.trip_end_timestamp, utc=True)
        return self.data
