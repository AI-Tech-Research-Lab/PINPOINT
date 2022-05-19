import copy
import logging
import os

import numpy as np
import pandas as pd
import torch
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

import pmdarima as pm

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

logging.getLogger('prophet').setLevel(logging.WARNING)


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class LogModified:
    """Class corresponding to a custom variant of logarithmic feature transformation.
    In particular, this transformation tries to overcome the traditional issues of a logarithmic transformation, i.e.
    the impossibility to work on negative data and the different behaviour on 0 < x < 1.

    Notes
    -----
    The actual function computed by this transformation is:

    .. math::
        f(x) = sign(x) * log(|x| + 1)

    The inverse, instead, is:

    .. math::
        f^{-1}(x) = sign(x) * e^{(abs(x) - sign(x))}
    """

    def apply(self, data: pd.Series) -> pd.Series:
        return data.apply(lambda x: np.sign(x) * np.log(abs(x) + 1))

    def inverse(self, data: pd.Series) -> pd.Series:
        return data.apply(lambda x: np.sign(x) * np.exp(abs(x)) - np.sign(x))

    def __str__(self):
        return "modified Log"


class Identity:
    """Class corresponding to the identity transformation.
    This is useful because the absence of a data pre-processing transformation would be a particular case for functions
    which compute predictions; instead, using this, that case is not special anymore.

    Notes
    -----
    The actual function computed by this transformation is:

    .. math::
        f(x) = x

    The inverse, instead, is:

    .. math::
        f^{-1}(x) = x
    """

    def apply(self, data: pd.Series) -> pd.Series:
        return data

    def inverse(self, data: pd.Series) -> pd.Series:
        return data

    def __str__(self):
        return "none"


def sliding_windows(data, _seq_length, _forecast_horizon):
    if len(data) <= _seq_length:
        print("WARNING! Can't make sliding windows!")
    x = []
    y = []

    for i in range(len(data)-_seq_length-_forecast_horizon):
        _x = data[i:(i+_seq_length)]
        _y = data[i+_seq_length:i+_seq_length+_forecast_horizon]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def train_model(cnn, train_set, validation_set, num_epochs, learning_rate, seq_length, forecast_horizon):
    # Account for validation being small
    _validation_length = len(validation_set)
    validation_set = np.concatenate((train_set[-seq_length - forecast_horizon:], validation_set))

    train_data = train_set.reshape(len(train_set), 1)
    validation_data = validation_set.reshape(len(validation_set), 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Train values
    train_normalized = scaler.fit_transform(train_data)
    validation_normalized = scaler.transform(validation_data)

    x_train, y_train = sliding_windows(train_normalized, seq_length, forecast_horizon)
    y_train = y_train.reshape(len(y_train), forecast_horizon)

    trainX = Variable(torch.Tensor(x_train).reshape(len(x_train), 1, seq_length))
    trainY = Variable(torch.Tensor(y_train))

    # Validation values
    x_validation, y_validation = sliding_windows(validation_normalized, seq_length, forecast_horizon)
    y_validation = y_validation.reshape(len(y_validation), forecast_horizon)

    assert len(y_validation) == _validation_length, f"{len(y_validation)} != {_validation_length}"

    validationX = Variable(torch.Tensor(x_validation).reshape(len(x_validation), 1, seq_length))
    validationY = Variable(torch.Tensor(y_validation))

    datas = {'train': (trainX, trainY), 'valid': (validationX, validationY)}

    cnn.to(device)

    criterion = torch.nn.L1Loss()  # L1 = MAE
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    best_model_wts = copy.deepcopy(cnn.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            X, Y = datas[phase]
            if phase == 'train':
                cnn.train()  # Set model to training mode
            else:
                cnn.eval()  # Set model to evaluate mode

            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = cnn(X.to(device))

                loss = criterion(outputs, Y.to(device))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # deep copy the model
            if phase == 'valid' and loss < best_loss:
                # print(f"Better loss found at epoch {epoch}: {loss}")
                best_loss = loss
                best_model_wts = copy.deepcopy(cnn.state_dict())

    # load best model weights
    cnn.load_state_dict(best_model_wts)
    return cnn, scaler


def get_prophet_forecast(train: pd.Series, future_points: int) -> pd.DataFrame:
    """
    Train a FBProphet model on the Pandas Series `train` and return
    a Pandas DataFrame containing the forecast for the next time instants.

    The returned DataFrame has length equal to `future_points`, and three columns,
    i.e.: 'yhat', 'yhat_lower', 'yhat_upper'.
    """
    m = Prophet()
    # prophet_train = train.iloc[-60:].reset_index()
    prophet_train = train.reset_index()

    columns_names = prophet_train.columns
    prophet_train = prophet_train.rename(columns={columns_names[0]: 'ds',
                                                  columns_names[1]: 'y'})

    with suppress_stdout_stderr():
        m.fit(prophet_train)

    future = m.make_future_dataframe(periods=future_points, freq=train.index.freq)
    prophet_forecast = m.predict(future)
    prophet_forecast = prophet_forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
    prophet_forecast.index.freq = train.index.freq  # Force forecast to have the same freq of train
    prophet_forecast = prophet_forecast.iloc[-future_points:, :]
    return prophet_forecast.loc[:, 'yhat']


class ARIMA_model:

    def __init__(self):
        self.model = None
        self.last_date = None
        self.arparams = None
        self.maparams = None

    def train(self, train):
        self.model = pm.auto_arima(train, seasonal=False)
        self.last_date = train.index[-1]
        self.arparams = self.model.arparams()
        self.maparams = self.model.maparams()

    def get_forecast(self, train: pd.Series, future_points: int, update=True) -> list:
        if update:
            self.model.update(train.loc[self.last_date:])
        else:
            self.model.update(train.loc[self.last_date:], maxiter=0)           
            # This is the max we can do, since we have to call update.
            # maxiter=0 "minimizes" the updates in AR/MA parameters.
        
        self.last_date = train.index[-1]
        arima_forecast = self.model.predict(n_periods=future_points)
        return arima_forecast
