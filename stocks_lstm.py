import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pykrx
from pykrx import stock
import os
import sys


# Pykrx
# 한국 주식 데이터 api

# 거래량, 주가, 종가, 시가, 티커, 기타 종목 등의 정보를 얻을 수 있음.

# https://github.com/sharebook-kr/pykrx

# ticker extraction
tickers = stock.get_market_ticker_list()
tickers_kosdaq = stock.get_market_ticker_list(market = 'KOSDAQ')

indexes = stock.get_index_ticker_list()


index_dict = {}

for _ in indexes:
    index_dict[_] = stock.get_index_ticker_name(_)

data = pd.DataFrame()
for i in list(index_dict.keys()):
    stocks = pd.DataFrame(stock.get_index_ohlcv('20180101', '20240501', str(i))['종가'])
    stocks.columns = [str(index_dict[i])]
    data = pd.concat([data, stocks], axis=1)
    
indexes = stock.get_index_ticker_list()

index_dict = {}

for _ in indexes:
    index_dict[_] = stock.get_index_ticker_name(_)
    
# KOSPI info extract by date and tickers
data = pd.DataFrame()
for i in list(index_dict.keys()):
    stocks = pd.DataFrame(stock.get_index_ohlcv('20180101', '20240501', str(i))['종가'])
    stocks.columns = [str(index_dict[i])]
    data = pd.concat([data, stocks], axis=1)
data2 = pd.DataFrame()
for i in list(index_dict.keys()):
    stocks = pd.DataFrame(stock.get_index_ohlcv('20100101', '20171231', str(i))['종가'])
    stocks.columns = [str(index_dict[i])]
    data2 = pd.concat([data2, stocks], axis=1)
    
    
# 같은 원리로 kosdaq에서도 추출
data_kosdaq = pd.DataFrame()
for i in list(kosdaq_dict.keys()):
    stocks = pd.DataFrame(stock.get_index_ohlcv('20180101', '20240501', str(i))['종가'])
    stocks.columns = [str(kosdaq_dict[i])]
    data_kosdaq = pd.concat([data_kosdaq, stocks], axis=1)
    

# Moving Average(이평선) 작성

# 5 / 10 / 20 / 60 / 120

kospi_5 = kospi.rolling(window=5).mean()
kospi_10 = kospi.rolling(window=10).mean()
kospi_20 = kospi.rolling(window=20).mean()
kospi_60 = kospi.rolling(window=60).mean()
kospi_120 = kospi.rolling(window=120).mean()

# MA 기반 골든/데드크로스 일자 추출 함수

def ma_calc(df, short, long, positive = 1, negative = -1, silence=False, buying_delay = 0, sell_delay=0):
    df_copy = copy.deepcopy(df)
    positive = positive
    negative = negative
    now = 0
    data = pd.DataFrame(index = df_copy.index, columns = df_copy.columns)
    data = data.fillna(0)
    for col in df_copy.columns[1:]:
        breaking = False
        p2n = []
        n2p = []
        
        for i in range(len((short - long)[col])):
            if (now == 0) and (not (short - long).loc[i, col] > 0):
                continue
            
            if pd.isnull((short - long).loc[i, col]):
                continue
            else:
                if now == positive:
                    if (short - long).loc[i, col] > negative:
                        continue
                    else:
#                         if len(n2p) and breaking:
#                             now = negative
#                             p2n.append(i)
#                             breaking = True
#                         else:
#                             continue
                        now = negative
                        p2n.append(i)
                        
                else:
                    if (short - long).loc[i, col] < positive:
                        continue
                    else:
                        now = positive
                        n2p.append(i)

        n2p = np.array(n2p)
        n2p += buying_delay
        n2p = list(n2p)
        
        p2n = np.array(p2n)
        p2n += sell_delay
        p2n = list(p2n)
        
        if p2n[0] < n2p[0]:
            p2n.pop(0)
        
        if len(n2p) > len(p2n):
            n2p.pop()
        
        data[col][n2p] = -1
        data[col][p2n] = 1
        
        if not silence:
            print(col)
        
    return data
  
# 모든 MA선에 대하여 적용하여 결과 추출

kospis = {
    5 : [kospi_10, kospi_20, kospi_60, kospi_120],
    10 : [kospi_20, kospi_60, kospi_120],
    20 : [kospi_60, kospi_120],
    60 : [kospi_120]
}

data = pd.DataFrame()
for i in kospis.keys():
    if i == 5:
        for j in range(len(kospis[i])):
            result = pd.DataFrame()
            result =  ma_calc(kospi, kospi_5, kospis[i][j], )
            
            result_5[j] = result
            
            result2 = pd.DataFrame(kospi[result == 1].sum() - kospi[result == -1].sum())
            num = kospis[i][j]['코스피'].isnull().sum()+1
            result2.columns = [f'{i} to {num}']
            
            data = pd.concat([data, result2], axis=1)
            
    elif i == 10:
        for j in range(len(kospis[i])):
            result = pd.DataFrame()
            result =  ma_calc(kospi, kospi_10, kospis[i][j], )
            
            result_10[j] = result
            
            result2 = pd.DataFrame(kospi[result == 1].sum() - kospi[result == -1].sum())
            num = kospis[i][j]['코스피'].isnull().sum()+1
            result2.columns = [f'{i} to {num}']
            
            data = pd.concat([data, result2], axis=1)
            
    elif i == 20:
        for j in range(len(kospis[i])):
            result = pd.DataFrame()
            result =  ma_calc(kospi, kospi_20, kospis[i][j], )
            
            result_20[j] = result
            
            result2 = pd.DataFrame(kospi[result == 1].sum() - kospi[result == -1].sum())
            num = kospis[i][j]['코스피'].isnull().sum()+1
            result2.columns = [f'{i} to {num}']
            
            data = pd.concat([data, result2], axis=1)

    elif i == 60:
        for j in range(len(kospis[i])):
            result = pd.DataFrame()
            result =  ma_calc(kospi, kospi_60, kospis[i][j])
            
            result_60[j] = result
            
            result2 = pd.DataFrame(kospi[result == 1].sum() - kospi[result == -1].sum())
            num = kospis[i][j]['코스피'].isnull().sum()+1
            result2.columns = [f'{i} to {num}']
            
            data = pd.concat([data, result2], axis=1)
            

# LSTM 적용

# Column별로 데이터 분리

for col in kospi_cols:
    for i in range(len(results)):
        for j in range(len(results[i])):
            data_frame = pd.concat([kospi_cols[col], results[i][j][col]], axis=1)
            kospi_cols[col] = data_frame

for i in kospi_cols.keys():
    col = kospi_cols[i].columns[0]
    
    colnames = [col+' 5 to 10', col+' 5 to 20', col+' 5 to 60', col+' 5 to 120', col+' 10 to 20', col+' 10 to 60', col+' 10 to 120', col+' 20 to 60', col+' 20 to 120', col+' 60 to 120']
    
    kospi_cols[i].columns = colnames

# by TensorFlow

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 60
x_seq, y_seq = create_sequences(np.array(target_train[:1800]), seq_length)

x_seq = x_seq.reshape(x_seq.shape[0], seq_length, 1)

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)))
# model.add(LSTM(120, activation='relu', return_sequences=True))
model.add(LSTM(70, activation='relu'))
model.add(Dense(1))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=25)

x_test, y_test = create_sequences(np.array(target_train[1800:2400]), seq_length)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
history = model.fit(
    x_seq, 
    y_seq, 
    epochs=100, 
    batch_size=16, 
    validation_data=(x_test, y_test),
    callbacks = [early_stopping])
predictions = model.predict(x_test)


# by Torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torch import optim as optim

class SequenceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]
        target = self.data[idx+self.seq_length]
        return torch.FloatTensor(seq), torch.FloatTensor([target])


seq_length = 60
train_dataset = SequenceDataset(np.array(target_train[:1800]), seq_length)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_dataset = SequenceDataset(np.array(target_train[1800:2400]), seq_length)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, hidden_layer2_size=70, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_layer_size, hidden_layer2_size, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_layer2_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm1(input_seq)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.relu(lstm_out[:, -1])
        predictions = self.linear(lstm_out)
        return predictions

model = LSTMModel()

learning_rate = 0.01
epochs = 300
patience = 25
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

early_stopping = patience
min_val_loss = np.inf
epochs_no_improve = 0

val_dataset = SequenceDataset(np.array(target_train[1800:2400]), seq_length)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for sequences, targets in train_dataloader:
        optimizer.zero_grad()
        output = model(sequences.unsqueeze(-1))
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_sequences, val_targets in val_dataloader:
            val_output = model(val_sequences.unsqueeze(-1))
            val_loss += criterion(val_output, val_targets).item()

    val_loss /= len(val_dataloader)
    epoch_loss /= len(train_dataloader)
    print(f'Epoch {epoch+1}, Train Loss: {epoch_loss}, Validation Loss: {val_loss}')

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stopping:
        print("Early stopping!")
        break

test_dataset = SequenceDataset(target_train[1800:2400], seq_length)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = []
model.eval()
with torch.no_grad():
    for test_sequences, _ in test_dataloader:
        test_output = model(test_sequences.unsqueeze(-1))
        predictions.append(test_output.item())































