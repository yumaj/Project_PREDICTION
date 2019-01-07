import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pytorch_rnn import pytorch_rnn



############### rnn part #############################
#create rnn model 

INPUT_SIZE = 60
num_epochs = 5
rnnmode = pytorch_rnn(INPUT_SIZE,num_epochs)

dataset_train = pd.read_csv('nikkei_data.csv')
training_set = dataset_train.iloc[:, 1:2].values

print(training_set)
# Feature Scaling

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range( INPUT_SIZE, len(training_set)):
    X_train.append( training_set_scaled[i- INPUT_SIZE:i, 0])
    y_train.append( training_set_scaled[i, 0])
X_train,  y_train = np.array( X_train), np.array( y_train)

'''
print("------old X_train start--------")
print(X_train)
print("------old X_train start--------")
'''
# Reshape shape[0] is number of row , shape[1] is number of col
#reshape to [ [[]],[[]],....]  
X_train = np.reshape( X_train, ( X_train.shape[0], 1,  X_train.shape[1]) )
'''
print("------new X_train start--------")
print(X_train)
print("------new X_train start--------")
'''


#train model 
rnnmode.train(X_train,y_train)



#import test data set
dataset_test = pd.read_csv('nikkei_test_data.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


dataset_total = pd.concat((dataset_train['value'], dataset_test['value']), axis = 0)
#input set from data test - input size to end (original data)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - INPUT_SIZE:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
# create test set, 20 sets 
X_test = []
for i in range(INPUT_SIZE, len(inputs)):
    X_test.append(inputs[i-INPUT_SIZE:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# 

X_collect = np.concatenate((X_train, X_test),axis=0)

predicted_stock_price = rnnmode.vaild(X_collect)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

real_stock_price_all = np.concatenate((training_set[INPUT_SIZE:], real_stock_price))

#plot result
plt.figure(1, figsize=(12, 5))
plt.plot(real_stock_price_all, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Prediction')
plt.title('Stock prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
############### rnn part  end#############################
