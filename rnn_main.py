import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pytorch_rnn import pytorch_rnn
from linearR_model import linear_model
from DEAP_GP_model import DEAP_GP_model
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
import random

############### load data part #########################
dataset_train = pd.read_csv('nikkei_data.csv')
training_set = dataset_train.iloc[:, 1:2].values


dataset_test = pd.read_csv('nikkei_test_data.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

INPUT_SIZE = 60
predict_move = 5
# Feature Scaling

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

for i in range( INPUT_SIZE, len(training_set) - predict_move):
    X_train.append( training_set_scaled[i- INPUT_SIZE:i, 0])
    rn = random.randint(0,10)
    if rn <= 1 :
        cn = random.randint(50,59)
        for j in range(0,cn):
            pos = random.randint(0,INPUT_SIZE - 1)
            num =  int(X_train[-1][pos])
            if pos - 1 >= 0 :
                num2 =  int(X_train[-1][pos - 1])
            X_train[-1][pos] =  random.randint(num2,num)
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

GP_SIZE = 60
GP_model = DEAP_GP_model(num_generation=30,input_size=GP_SIZE)


X_train_gp = []
y_train_gp = []
move_size = INPUT_SIZE - GP_SIZE
for i in range(move_size+GP_SIZE, len(training_set) - predict_move):
    X_train_gp.append( training_set_scaled[i- GP_SIZE:i, 0])

    y_train_gp.append( training_set_scaled[i + predict_move, 0])
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
INPUT_SIZE = 60
num_epochs = 1
OUT_PUT_SIZE = 1

rnnmode = pytorch_rnn(INPUT_SIZE,num_epochs,OUT_PUT_SIZE)


print(training_set)


#train model 

rnnmode.train(training_set_scaled,training_set_scaled,1000,predict_move)
predicted_stock_price = rnnmode.vaild(X_collect)

############### rnn part  end#############################



############## plot ######################################


predicted_stock_price = sc.inverse_transform(predicted_stock_price)
lr_arr = lr_arr.reshape(-1, 1)
predicted_stock_price_lr = sc.inverse_transform(lr_arr)
gp_arr = np.array(gp_arr)
gp_arr = gp_arr.reshape(-1, 1)
predicted_stock_price_gp = sc.inverse_transform(gp_arr)

real_stock_price_all = np.concatenate((training_set[INPUT_SIZE:], real_stock_price))

real_stock_price_all = real_stock_price_all[5:]


print(predicted_stock_price_lr)

#plot result
plt.figure(1, figsize=(12, 5))
plt.plot(real_stock_price_all, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Prediction')
plt.plot(predicted_stock_price_lr, color = 'yellow', label = 'linear Prediction')
plt.plot(predicted_stock_price_gp, color = 'black', label = 'Gp Prediction')


lrmse = sum(sqrt((real_stock_price_all - predicted_stock_price_lr)**2))/len(predicted_stock_price_lr)
gpmse = sum(sqrt((real_stock_price_all - predicted_stock_price_gp)**2))/len(predicted_stock_price_gp)
rnnmse = sum(sqrt((real_stock_price_all - predicted_stock_price)**2))/len(predicted_stock_price)




predicted_stock_price_lr.tofile('lrmodel_reslut.csv',sep=',')
predicted_stock_price_gp.tofile('gp_reslut.csv',sep=',')
predicted_stock_price.tofile('rnn_reslut.csv',sep=',')


f3 = open('GP_tree.txt', 'w')
f3.write(str(GP_model.nof_expr[-0]))
f4 = open('linearReg_parameter.txt', 'w')
f4.write(str(lr_model.coef))

print('linear mse = {}, gp mse = {}, rnn mse = {}'.format(lrmse,gpmse,rnnmse))


plt.title('Stock prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

############## plot end######################################