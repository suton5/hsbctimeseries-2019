import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import seaborn as sns


class EnergyModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):

        df['day'] = pd.DatetimeIndex(df['day'])

        df = df.rename(columns={'day': 'ds', 'consumption': 'y'})

        return df, None

    def fit(self, X, y):

        forecast_model = Prophet(
            growth='linear',
            weekly_seasonality=7,
            yearly_seasonality=7)
        self.model = forecast_model.fit(X)

    def preprocess_unseen_data(self, df):
        df['day'] = pd.DatetimeIndex(df['day'])

        df = df.rename(columns={'day': 'ds', 'consumption': 'y'})

        return df

    def predict(self, X):

        predictions = self.model.predict(X)
        return predictions['yhat']
