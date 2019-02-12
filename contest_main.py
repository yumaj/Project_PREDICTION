import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pytorch_rnn import pytorch_rnn
from linearR_model import linear_model
from DEAP_GP_model import DEAP_GP_model
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
import random


#load data
dataset_train = pd.read_csv('Contest_data_Q.csv')

dataset_test = pd.read_csv('Contest_data_P.csv')

dataset_solor = pd.read_csv('weather_t.csv')


training_set_q1 = dataset_train['q1'][1:].astype(float)
training_set_q2 = dataset_train['q2'][1:].astype(float)

training_target_s1 = dataset_test['S1[MW]']


training_set_q1 = np.array(training_set_q1)
training_set_q2 = np.array(training_set_q2)
training_target_s1 = np.array(training_target_s1)


training_set_q1_solor_time = dataset_solor['S1'].astype(float)
training_set_q1_solor_time = np.array(training_set_q1_solor_time)



#setting 
INPUT_SIZE = 92
num_epochs = 1
OUT_PUT_SIZE = 1
predict_move = 0

rnnmode = pytorch_rnn(INPUT_SIZE,num_epochs,OUT_PUT_SIZE,)


sc = MinMaxScaler(feature_range = (0, 1))


#normalize
training_set_q1_solor_time_scaled = sc.fit_transform(training_set_q1_solor_time.reshape(-1, 1) )

training_set_q1_scaled = sc.fit_transform(training_set_q1.reshape(-1, 1) )
training_set_q2_scaled = sc.fit_transform(training_set_q2.reshape(-1, 1) )

training_set_target_s1 = sc.fit_transform(training_target_s1.reshape(-1, 1) )

print(training_set_q1_scaled)
print(training_set_target_s1)

#training 
rnnmode.train(training_set_q1_scaled,training_set_q2_scaled,training_set_target_s1,2000,predict_move,training_set_q1_solor_time_scaled)
#predict 
predicted_stock_price = rnnmode.vaild(training_set_q1_scaled,training_set_q2_scaled,46,training_set_q1_solor_time_scaled)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

predicted_stock_price.tofile('rnn_reslut.csv',sep=',')
