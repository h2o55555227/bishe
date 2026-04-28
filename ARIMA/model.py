import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle


class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.model_fit = None
        self.train_mean = None
        self.train_std = None

    def fit(self, data, trend='n'):
        self.model = ARIMA(data, order=self.order, trend=trend)
        self.model_fit = self.model.fit()
        return self.model_fit

    def predict(self, steps=1):
        if self.model_fit is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def get_forecast(self, steps=1):
        if self.model_fit is None:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        forecast = self.model_fit.get_forecast(steps=steps)
        return forecast

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'model_fit': self.model_fit,
                'train_mean': self.train_mean,
                'train_std': self.train_std
            }, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(order=data['order'])
        model.model_fit = data['model_fit']
        model.train_mean = data['train_mean']
        model.train_std = data['train_std']
        return model


def build_arima_model(order=(1, 1, 1)):
    return ARIMAModel(order=order)
