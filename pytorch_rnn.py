import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

class pytorch_rnn():

    def __init__(self,INPUT_SIZE,num_epochs,OUTPUT_SIZE):
        
        torch.manual_seed(0)

        #mode setting
        self.INPUT_SIZE = INPUT_SIZE
        self.HIDDEN_SIZE = 64
        self.NUM_LAYERS = 5
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.learning_rate = 0.001
        self.num_epochs = num_epochs
        self.rnn = None
    def train(self,X_train,x_train_2,y_train,windwos_size,predict_move,ex_data):

        #defined RNN model 
        class RNN(nn.Module):
            def __init__(self, i_size, h_size, n_layers, o_size):
                super(RNN, self).__init__()
                
                self.rnn = nn.LSTM(
                    #need to chnage this value to get more input
                    input_size=i_size*2,
                    hidden_size=h_size,
                    num_layers=n_layers
                )
                self.out = nn.Linear(h_size, o_size)
            def forward(self, x, h_state):
                r_out, hidden_state = self.rnn(x, h_state)
                
                hidden_size = hidden_state[-1].size(-1)
                r_out = r_out.view(-1, hidden_size)
                outs = self.out(r_out)

                return outs, hidden_state
        print(torch.cuda.is_available())

        #torch.backends.cudnn.enabled = False
    
        #torch.backends.cudnn.benchmark = True


        #print("torch = ",torch.cuda.device_count())
        self.rnn = RNN(self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_LAYERS, self.OUTPUT_SIZE)
        #self.rnn.cuda()
        self.rnn.cuda()
        optimiser = torch.optim.Adam(self.rnn.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()


        for epoch in range(self.num_epochs):
            
            hidden_state = None
            for stage  in  range(0, len(X_train) - windwos_size - self.INPUT_SIZE,windwos_size - predict_move):
                X_train_data  = []
                Y_train_Data = []
                X_train_data_r = None
                Y_train_data_r = None
                for i in range( self.INPUT_SIZE + stage, self.INPUT_SIZE + stage + windwos_size):
                    tempdata = []
                    tempdata = np.append( X_train[i - self.INPUT_SIZE:i, 0],x_train_2[i - self.INPUT_SIZE:i, 0])
                    #tempdata = np.append(tempdata, ex_data[i - self.INPUT_SIZE:i, 0])
                    X_train_data.append(tempdata)
                    Y_train_Data.append( y_train[i + predict_move, 0])


               
                X_train_data_r,  Y_train_data_r = np.array( X_train_data), np.array(Y_train_Data)
                X_train_data_r = np.reshape( X_train_data_r, ( X_train_data_r.shape[0], 1,  X_train_data_r.shape[1]) )
                

               
                inputs = Variable(torch.from_numpy(X_train_data_r).float()).cuda()
                labels = Variable(torch.from_numpy(Y_train_data_r).float()).cuda()
              


                output, hidden_state = self.rnn(inputs, hidden_state)

                loss = criterion(output.view(-1), labels)
                optimiser.zero_grad()
                # back propagation
                loss.backward(retain_graph=True)                     
                # update
                optimiser.step()                                     
                
                print('epoch {}, loss {}'.format(epoch,loss.item()))
        return self.rnn


    def vaild(self,x_test,x_test_2,next_p,ex_data1):
        hidden_state = None

        X_train_data  = []
        for i in range( self.INPUT_SIZE,len(x_test)):
            tempdata = []
            tempdata = np.append( x_test[i - self.INPUT_SIZE:i, 0],x_test_2[i - self.INPUT_SIZE:i, 0])
            #tempdata = np.append(tempdata, ex_data1[i - self.INPUT_SIZE:i, 0])
            X_train_data.append(tempdata)   
            
        X_train_data_r = None
        X_train_data_r  = np.array(X_train_data)
        X_train_data_r = np.reshape( X_train_data_r, ( X_train_data_r.shape[0], 1,  X_train_data_r.shape[1]) ) 

        test_inputs = Variable(torch.from_numpy(X_train_data_r).float()).cuda()
        predicted_stock_price, b = self.rnn(test_inputs, hidden_state)

        predicted_stock_price = np.reshape(predicted_stock_price.cpu().detach().numpy(), (test_inputs.cpu().shape[0], 1))

        ox_size = len(x_test)
        #preidect next month , using new data 
        '''
        x_test = np.append(x_test,predicted_stock_price)
        for i in range(ox_size - self.INPUT_SIZE, next_p):
            new_data = []
            new_data.append( x_test[i - self.INPUT_SIZE:i, 0],x_test_2[i - self.INPUT_SIZE:i, 0],ex_data1[i - self.INPUT_SIZE:i, 0])
            new_data_r = None
            new_data_r  = np.array(new_data)
            new_data_r = np.reshape( new_data_r, ( new_data_r.shape[0], 1,  new_data_r.shape[1]) ) 
            test_inputs = Variable(torch.from_numpy(new_data_r).float())
            ouput_v, b = self.rnn(test_inputs, hidden_state)
            
            ouput_v = np.reshape(ouput_v.detach().numpy(), (ouput_v.shape[0], 1))
            x_test = np.append(x_test,ouput_v)
            predicted_stock_price.append(ouput_v)
        '''
        print("shape = ",predicted_stock_price.shape[0])
        print("shape2 = ",predicted_stock_price.shape[1])
        predicted_stock_price.tofile('rnn_reslut_contest2_noex.csv',sep=',')

        return predicted_stock_price


