import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pytorch_rnn import pytorch_rnn
from linearR_model import linear_model
from DEAP_GP_model import DEAP_GP_model
from scipy import linspace, polyval, polyfit, sqrt, stats, randn

############### load data part #########################
dataset_train = pd.read_csv('nikkei_data.csv')
training_set = dataset_train.iloc[:, 1:2].values


dataset_test = pd.read_csv('nikkei_test_data.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

INPUT_SIZE = 60
num_epochs = 20
# Feature Scaling

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

for i in range( INPUT_SIZE, len(training_set)):
    X_train.append( training_set_scaled[i- INPUT_SIZE:i, 0])
    y_train.append( training_set_scaled[i, 0])
X_train,  y_train = np.array( X_train), np.array( y_train)
# Reshape shape[0] is number of row , shape[1] is number of col
#reshape to [ [[]],[[]],....]  
X_train_for_lin = X_train
Y_train_for_lin = y_train
X_train = np.reshape( X_train, ( X_train.shape[0], 1,  X_train.shape[1]) )

#import test data set

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
X_test__for_lin = X_test
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# 

X_collect = np.concatenate((X_train, X_test),axis=0)
X_collect_for_lin = np.concatenate((X_train_for_lin, X_test__for_lin),axis=0)
############### load data part end #########################


############### linear part #########################


lr_model = linear_model()
lr_model.train(X_train_for_lin,Y_train_for_lin)
lr_arr = lr_model.vaild(X_collect_for_lin)

############### linear part end #########################


############### gp part #############################


GP_model = DEAP_GP_model(num_generation=10)

GP_SIZE = 10
X_train_gp = []
y_train_gp = []
move_size = INPUT_SIZE - GP_SIZE
for i in range(move_size+GP_SIZE, len(training_set)):
    X_train_gp.append( training_set_scaled[i- GP_SIZE:i, 0])
    y_train_gp.append( training_set_scaled[i, 0])
X_train_gp,  y_train_gp = np.array( X_train_gp), np.array( y_train_gp)

GP_model.train(X_train_gp,y_train_gp)

X_test_gp = []
for i in range(move_size+GP_SIZE, len(inputs)):
    X_test_gp.append(inputs[i-GP_SIZE:i, 0])
X_test_gp = np.array(X_test_gp)
X_collect_gp = np.concatenate((X_train_gp, X_test_gp),axis=0)

gp_arr = GP_model.vaild(X_collect_gp)
############### gp part end #########################

############### rnn part #############################
#create rnn model 


rnnmode = pytorch_rnn(INPUT_SIZE,num_epochs)


print(training_set)


#train model 

rnnmode.train(X_train,y_train)


############### rnn part  end#############################



############## plot ######################################

predicted_stock_price = rnnmode.vaild(X_collect)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
lr_arr = lr_arr.reshape(-1, 1)
predicted_stock_price_lr = sc.inverse_transform(lr_arr)
gp_arr = np.array(gp_arr)
gp_arr = gp_arr.reshape(-1, 1)
predicted_stock_price_gp = sc.inverse_transform(gp_arr)

real_stock_price_all = np.concatenate((training_set[INPUT_SIZE:], real_stock_price))




print(predicted_stock_price_lr)

#plot result
plt.figure(1, figsize=(12, 5))
plt.plot(real_stock_price_all, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Prediction')
plt.plot(predicted_stock_price_lr, color = 'yellow', label = 'linear Prediction')
plt.plot(predicted_stock_price_gp, color = 'black', label = 'Gp Prediction')

lrmse = sum(sqrt((real_stock_price_all - predicted_stock_price_lr)**2)) 
gpmse = sum(sqrt((real_stock_price_all - predicted_stock_price_gp)**2))
rnnmse = sum(sqrt((real_stock_price_all - predicted_stock_price)**2))

print('linear mse = {}, gp mse = {}, rnn mse = {}'.format(lrmse,gpmse,rnnmse))


plt.title('Stock prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

############## plot end######################################