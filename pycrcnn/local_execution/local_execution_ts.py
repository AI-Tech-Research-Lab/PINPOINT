import time
import torch
from sklearn.preprocessing import MinMaxScaler
from utils import *
import jsonpickle
import numpy as np
import torch
import pandas as pd
from Pyfhel import Pyfhel
from utils import split_sequence
from pycrcnn.net_builder.encoded_net_builder_ts import build_from_pytorch
from sklearn.preprocessing import MinMaxScaler
from pycrcnn.crypto.crypto import encrypt_matrix, decrypt_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.autograd import Variable
from model import CNN

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def local_execution(data, encoded_net, HE):

    def compute(local_HE, local_data, local_encoded_net, local_return_dict=None, ind=None):
        t0 = time.time()
        encrypted_data = encrypt_matrix(local_HE, local_data)      

        for layer in local_encoded_net:
            encrypted_data = layer(encrypted_data)

        local_result = decrypt_matrix(local_HE, encrypted_data)
        #print(f'Time for encrypted: {time.time()-t0}')

        if local_return_dict is None:
            return local_result
        else:
            local_return_dict[ind] = local_result

    
    ts = data.detach().numpy()    
    result = compute(HE, ts, encoded_net)

    return result

    


def test():
    def sliding_windows(data, _seq_length, _forecast_horizon):
        x = []
        y = []

        for i in range(len(data)-seq_length-_forecast_horizon):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length:i+seq_length+_forecast_horizon]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)

    def train_model(train_set, num_epochs, learning_rate, seq_length):
        train_data = train_set.values.reshape(len(train_set), 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_normalized = scaler.fit_transform(train_data)
        
        x, y = sliding_windows(train_normalized, seq_length, forecast_horizon)
        
        y = y.reshape(len(y), forecast_horizon)
        
        trainX = Variable(torch.Tensor(x).reshape(len(x), 1, seq_length))
        trainY = Variable(torch.Tensor(y))
        
        cnn = CNN(16, 3, seq_length, 10, forecast_horizon)
        cnn.to(device)
        
        criterion = torch.nn.L1Loss()    # L1 = MAE
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            outputs = cnn(trainX.to(device))
            optimizer.zero_grad()            
            loss = criterion(outputs, trainY.to(device))
            loss.backward()
            optimizer.step()
        
        return cnn, scaler
    
    with open("./local_execution/parameters_ts.json", "r") as f:
        params = jsonpickle.decode(f.read())

        # Parameters
        seq_length = 5         # Hyperparameter to change accordingly to the dataset    
        num_epochs = 1000        # Hyperparameter to change accordingly to the dataset
        learning_rate = 0.001   # Hyperparameter to change accordingly to the dataset
        forecast_horizon = 1
        retrain = True
        forecast = np.array([])
        forecast_from_decrypt = np.array([])

        # # Google price closing stock dataset
        #goog_stock = goog_stock = pd.read_csv("./local_execution/Stocks.csv", parse_dates=["date"], index_col="date")
        #goog_stock = goog_stock.loc[:, 'GOOG']
        #goog_stock.index.freq = 'D'
        #entire_ts = goog_stock
        #train = goog_stock.loc[:pd.Timestamp("2020-09-30")]
        #test = goog_stock.loc[pd.Timestamp("2020-10-01"):]
        
        # # Daily deaths dataset
        #daily_deaths = pd.read_csv("./local_execution/Covid19-italy.csv", parse_dates=["date"], index_col="date")
        #daily_deaths = daily_deaths.loc[:pd.Timestamp("2021-02-25"), 'Daily deaths']
        #daily_deaths.index.freq = 'D'
        #entire_ts = daily_deaths
        #train = daily_deaths.loc[:pd.Timestamp("2020-12-18")]
        #test = daily_deaths.loc[pd.Timestamp("2020-12-19"):]
        
        # # Daily cases dataset
        daily_cases = pd.read_csv("./local_execution/Covid19-italy.csv", parse_dates=["date"], index_col="date")
        daily_cases = daily_cases.loc[:, 'Daily cases']
        daily_cases.index.freq = 'D'
        entire_ts = daily_cases
        train = daily_cases.loc[:pd.Timestamp("2020-12-18")]
        test = daily_cases.loc[pd.Timestamp("2020-12-19"):]
        
        # # Monthly milk production dataset
        #milk_production = milk_production = pd.read_csv("./local_execution/monthly-milk-production.csv", parse_dates=["date"], index_col="date")
        #milk_production = milk_production.loc[:, 'Production']
        #milk_production.index.freq = 'MS'
        #entire_ts = milk_production
        #train = milk_production.loc[:pd.Timestamp("1974-01-01")]
        #test = milk_production.loc[pd.Timestamp("1974-02-01"):]

        _train = train.copy()
        _test = test.copy()
        scaler = MinMaxScaler()
        

        # PLAIN EVALUATION #
        print("Evaluation in progress...")

        
        experiment_maes = []   
        experiment_rmse = []
        experiment_from_decrypt_maes = []   
        experiment_from_decrypt_rmse = []

        encryption_parameters = params["encryption_parameters"]

        HE = Pyfhel()
        HE.contextGen(m=encryption_parameters[0]["m"],
                    p=encryption_parameters[0]["p"],
                    sec=encryption_parameters[0]["sec"],
                    base=encryption_parameters[0]["base"], intDigits=32, fracDigits=64)
        HE.keyGen()
        HE.relinKeyGen(20, 5)     
        

         

        for i in range(0, int(len(test) / forecast_horizon)): 
            if retrain:
                plain_net, scaler =  train_model(_train, num_epochs, learning_rate, seq_length)
                start_time = time.time()
                encoded_net = build_from_pytorch(HE, plain_net.main)
                print(f'Time to encode the net: {time.time()-start_time}')
            else:
                if i == 0:
                    plain_net, scaler = train_model(_train, num_epochs, learning_rate, seq_length)
                    start_time = time.time()
                    encoded_net = build_from_pytorch(HE, plain_net.main)
                    print(f'Time to encode the net: {time.time()-start_time}')
    
            plain_net.eval()
            
            inputs = _train.values.reshape(len(_train), 1)
            inputs_normalized = scaler.fit_transform(inputs)
            inputs_normalized = torch.FloatTensor(inputs_normalized[-seq_length:]).to(device)
            
            result = local_execution(inputs_normalized.reshape(1,1,seq_length), encoded_net, HE)
            result = scaler.inverse_transform(result)
            forecast_from_decrypt = np.append(forecast_from_decrypt, result)

            predict = plain_net(inputs_normalized.reshape(1, 1, seq_length))
            predict = scaler.inverse_transform(predict.cpu().detach().numpy())
            forecast = np.append(forecast, predict)        

            for j in range(0, forecast_horizon):
                _train[_train.index[-1] + j * train.index.freq] = _test[0]
                _test = _test.iloc[1:]
        
        forecast = pd.Series(data=forecast, index=test.iloc[:len(forecast)].index)
        forecast_from_decrypt = pd.Series(data=forecast_from_decrypt, index=test.iloc[:len(forecast)].index)

        experiment_maes.append(mean_absolute_error(test.iloc[:len(forecast)], forecast))
        experiment_rmse.append(mean_squared_error(test.iloc[:len(forecast)], forecast, squared=False))
        experiment_from_decrypt_maes.append(mean_absolute_error(test.iloc[:len(forecast_from_decrypt)], forecast_from_decrypt))
        experiment_from_decrypt_rmse.append(mean_squared_error(test.iloc[:len(forecast_from_decrypt)], forecast_from_decrypt, squared=False))
            
        mae = np.mean(np.array(experiment_maes))
        rmse = np.mean(np.array(experiment_rmse))
        mae_decrypt = np.mean(np.array(experiment_from_decrypt_maes))
        rmse_decrypt = np.mean(np.array(experiment_from_decrypt_rmse))
        
        # PRINTING RESULTS #
        print('Table results:')
        print(f'MAE FROM PLAIN DATA: {mae}\tRMSE FROM PLAIN DATA: {rmse}\nMAE FROM DECRYPTED DATA: {mae_decrypt}\tRMSE FROM DECRYPTED DATA: {rmse_decrypt}')
        print(forecast.to_numpy() - forecast_from_decrypt.to_numpy())
    
    
if __name__ == '__main__':
    test()
