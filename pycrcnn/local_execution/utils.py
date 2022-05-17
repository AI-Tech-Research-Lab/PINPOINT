import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(data):
    """Data Loader"""
    data_dir = os.path.join(data)

    data = pd.read_csv(data_dir,
                       # infer_datetime_format=True,
                       parse_dates=['date']
                       )

    data.index = data['date']
    data = data.drop('date', axis=1)

    return data

def plot_full(path, data, feature):
    """Plot Full Graph of Sales"""
    data.plot(y=feature, figsize=(16, 8))
    plt.xlabel('Date', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylabel(feature, fontsize=10)
    plt.grid()
    plt.title(feature)
    plt.savefig(os.path.join(path, '{}.png'.format(feature)))
    plt.show()

def split_sequence(sequence, n_steps):
    """Function to split the sequence into samples and labels"""
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > len(sequence)-1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

       
    

def data_loader(train_seq, test_seq, val_seq, train_label, test_label, val_label, batch_size):
    """Prepare data by applying sliding windows and return data loader"""

    # Convert to Tensor #
    train_set = TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_label))
    val_set = TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_label))
    test_set = TensorDataset(torch.from_numpy(test_seq), torch.from_numpy(test_label))

    # Data Loader #
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_lr_scheduler(lr_scheduler, optimizer):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler

def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_percentage_error(y_true, y_pred):
    """Mean Percentage Error"""
    mpe = np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100
    return mpe


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mape = np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
    return mape

def plot_pred_test(pred, actual, path, feature, model, step):
    """Plot Test set Prediction"""
    plt.figure(figsize=(10, 8))

    plt.plot(pred, label='Pred', marker = 'x')
    plt.plot(actual, label='Actual', marker = '.')

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('{}'.format(feature), fontsize=18)

    plt.legend(loc='best')
    plt.grid()

    plt.title('{} prediction using {} and {}'.format(feature, model.__class__.__name__, step), fontsize=18)
    plt.savefig(os.path.join(path, '{} prediction using {} and {}.png'.format(feature, model.__class__.__name__, step)))