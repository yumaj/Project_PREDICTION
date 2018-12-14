from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mat
from numpy import genfromtxt
import  numpy as np
from scipy import linalg

from scipy import linspace, polyval, polyfit, sqrt, stats, randn


lr = LinearRegression()

my_data = genfromtxt('japangdp.csv', delimiter=',')

data_s = my_data

year =  np.array([1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])

newdata = []
interval = 10
predictarr = []
training_size = 10

allarr = []
allarr2 = []
prediction_year = 10
first_msearray = []
sec_msearray = []
tree_A = []
for x in range(0,30):

    sum_mse = 0
    sum_mse2 = 0
    for j in range(0,prediction_year):
        training_data = []
        for i in range(0,training_size):
            newdata = my_data[j+i:j+i+interval]
            #newdata= np.insert(newdata, 0, 1)
            print(newdata)

            training_data.append(newdata)
        traget  = np.array(my_data[j + interval:j + interval + training_size])
        traget = np.transpose([traget])
        training_data = np.array(training_data)

        transposes = training_data.transpose()
        X =  training_data.transpose()
        Y =  np.array(my_data[j + interval:j + interval + training_size])

        modelx = lr.fit(X,Y)

        Xtranspo = X.transpose()
        newbeata = linalg.inv((Xtranspo.dot(X)))

        right = newbeata.dot(Xtranspo)
        print("Y  = ",Y)
        newbeata = newdata.dot(Y)

        newdata = np.array(training_data)
        print("transposes data= ", transposes)
        print("newdate = ",training_data)
        ans = transposes.dot(training_data)
        print("transpose dot ori ",ans)
        print("transpose dot trage",transposes.dot(traget))
        print("traget = ", traget)
        beta = linalg.inv(transposes.dot(training_data)).dot(transposes.dot(traget))

        print("beta",beta)
        print("new beta",newbeata)
        input = my_data[j + interval:j + interval + training_size]
        #input = np.insert(input,0,1)
        predictionans = input.dot(beta)

        print("orignal = ",traget)
        print("predict = ",predictionans)
        print("my_data sieze = ", my_data.size , " year size = ", year.size)
        #print("loss = ",my_data[j+interval+training_size] - predictionans)

        input = input.reshape(1, -1)
        input2 = my_data[j + interval + 1:j + interval + training_size + 1]
        input2 = input2.reshape(1, -1)

        print(input)
        num = modelx.predict(input)
        num2 = modelx.predict(input2)

        vari_data_target = my_data[j + interval + training_size + 1]
        vari_data_target2 = my_data[j + interval + training_size + 2]
        predictarr.append(num)

        tmep_arr = [num,num2]
        allarr.append(num)
        allarr2.append(num2)
        print(num)
        mse = sqrt((vari_data_target - num) ** 2)
        mse2 = sqrt((vari_data_target2 - num2) ** 2)
        first_msearray.append(sum_mse)
        sec_msearray.append(sum_mse2)

    x = year[interval + training_size :interval + training_size + prediction_year  ]
    print(x)
    print(my_data[prediction_year:prediction_year+interval])
    print("arr = " ,predictarr)
    #mat.plot(x,my_data[interval + training_size :interval + training_size + prediction_year],'r-')
    #mat.xticks(x, x)
    print(len(allarr), "  " , x.size)
    #mat.plot(x,allarr,'g--')
    np.insert(allarr2,0,0)
    #mat.plot(x,allarr2,'y--')
    f = open('SKLEARN_first_data.txt', 'a')
    for time in range(0,30):
        f.write(str(first_msearray[time]))
        f.write(",")
    f2 = open('SKLEARN_sec_data.txt', 'a')
    for time in range(0,30):

        f2.write(str(sec_msearray[time]))
        f2.write(",")
    #mat.legend(['original', 'regression first','regression sec'])

    #mat.show()

    print(allarr)
