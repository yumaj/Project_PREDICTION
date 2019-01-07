import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

class pytorch_rnn():

    def __init__(self,INPUT_SIZE,num_epochs):

        #mode setting
        self.INPUT_SIZE = INPUT_SIZE
        self.HIDDEN_SIZE = 64
        self.NUM_LAYERS = 2
        self.OUTPUT_SIZE = 1
        self.learning_rate = 0.001
        self.num_epochs = num_epochs
        self.rnn = None
    def train(self,X_train,y_train):

        #defined RNN model 
        class RNN(nn.Module):
            def __init__(self, i_size, h_size, n_layers, o_size):
                super(RNN, self).__init__()

                self.rnn = nn.LSTM(
                    input_size=i_size,
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

        self.rnn = RNN(self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_LAYERS, self.OUTPUT_SIZE)

        optimiser = torch.optim.Adam(self.rnn.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        hidden_state = None

        for epoch in range(self.num_epochs):
            inputs = Variable(torch.from_numpy(X_train).float())
            labels = Variable(torch.from_numpy(y_train).float())

            output, hidden_state = self.rnn(inputs, hidden_state) 

            loss = criterion(output.view(-1), labels)
            optimiser.zero_grad()
            #back propagation
            loss.backward(retain_graph=True)                     
            #update
            optimiser.step()                                     
            
            print('epoch {}, loss {}'.format(epoch,loss.item()))
        return self.rnn


    def vaild(self,x_test):
        hidden_state = None
        test_inputs = Variable(torch.from_numpy(x_test).float())
        predicted_stock_price, b = self.rnn(test_inputs, hidden_state)
        predicted_stock_price = np.reshape(predicted_stock_price.detach().numpy(), (test_inputs.shape[0], 1))
        return predicted_stock_price


