from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mat
from numpy import genfromtxt
import  numpy as np
from scipy import linalg

from scipy import linspace, polyval, polyfit, sqrt, stats, randn



class linear_model():



    def __init__(self):
        self.interval = 10
        self.predictarr = []
        self.training_size = 10
        self.lr_model = None
        self.coef = None


    def train(self,training_data_set,training_data_y):

        lr = LinearRegression()

        training_data = []
        for i in range(self.interval,len(training_data_set)):
            newdata = training_data_set[i - self.interval:i+self.interval]
          
            training_data.append(newdata)
        training_data = np.array(training_data)

        modelx = lr.fit(training_data_set,training_data_y)
        self.lr_model = modelx
        self.coef = modelx.coef_
        return modelx
        
    def vaild(self,input_y):
      

        output_array = self.lr_model.predict(input_y)
        return output_array
        