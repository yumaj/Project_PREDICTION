from scipy import linspace, polyval, polyfit, sqrt, stats, randn
import matplotlib.pyplot as mat
from numpy import genfromtxt
import  numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression

my_data = genfromtxt('japangdp.csv', delimiter=',')
data_s = my_data

year =  np.array([1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])

newdata = []
interval = 10
predictarr = []
training_size = 10

for j in range(0,10):
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
    input = my_data[j+interval:j+interval + training_size]
    #input = np.insert(input,0,1)
    predictionans = input.dot(beta)
    predictarr.append(predictionans[0])
    print("orignal = ",traget)
    print("predict = ",predictionans)
    print("loss = ",my_data[j+interval+training_size] - predictionans)


'''

x = year[interval:10+interval]
print(x)
print(my_data[10:20])
print("arr = " ,predictarr)
mat.plot(x,my_data[10:20],'r-')
mat.xticks(x, x)

mat.plot(x,predictarr,'g--')
mat.legend(['original', 'regression'])
mat.show()


min = 9e+20
min_i = 0
min_xr = []
#set window size 10 to 30
for j in range(10,11):
    endpoint = ( j - 1)
    endpoint_p = endpoint + 1
    for i in range(0,year.size - j - 1):
        for()
        reg = polyfit(year[i:i+endpoint], my_data[i:i+endpoint],5)
        #
        xr = polyval(reg, year[i+endpoint_p])
        loss = (xr - my_data[i+endpoint_p])**2
        print("erro loss ", str(year[i]) , " ~ " , year[i + endpoint] , " predicet " , str(year[i+endpoint_p]) , "loss value = " ,  loss )
        if min > loss:
            min = loss
            min_i = i
            min_xr = reg
# compute the mean square error
print("min_i  = " , min_i , "loss = ",min , "year start = ",year[min_i], " ~  " , year[min_i+endpoint])
print("min_xr = ",min_xr)
xr = polyval(min_xr, year[min_i+endpoint_p:])
print("loss function",sum((xr-my_data[min_i+endpoint_p:])**2), "from ", year[min_i + endpoint_p], " to 2017")
mse = sum((xr-my_data[min_i+endpoint_p:])**2)/(year.size - min_i - endpoint_p )
print("MSE ",mse, "from ", year[min_i + endpoint_p], " to 2017")

#predict the value
mat.title('JAPAN GDP Linear Regression')
mat.plot(year, my_data, 'g.--')
mat.plot(year[min_i+endpoint_p:], xr, 'r.-')
mat.legend(['original', 'regression'])
mat.show()




my_data = genfromtxt('taiwnagdp.csv', delimiter=',')
data_s = my_data

year =  np.array([1951,1952,1953,1954,1955,1956,1957,1958,1959,1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])

min = 9e+20
min_i = 0
min_xr = []
for j in range(10,11):
    endpoint = ( j - 1)
    endpoint_p = endpoint + 1
    for i in range(0,year.size - j - 1):
        reg = polyfit(year[i:i+endpoint], my_data[i:i+endpoint],5)
        #
        xr = polyval(reg, year[i+endpoint_p])
        loss = (xr - my_data[i+endpoint_p])**2
        print("erro loss ", str(year[i]) , " ~ " , year[i + endpoint] , " predicet " , str(year[i+endpoint_p]) , "loss value = " ,  loss )
        if min > loss:
            min = loss
            min_i = i
            min_xr = reg
# compute the mean square error
print("min_i  = " , min_i , "loss = ",min , "year start = ",year[min_i], " ~  " , year[min_i+endpoint])
print("min_xr = ",min_xr)
xr = polyval(min_xr, year[min_i+endpoint_p:])
print("Taiwan loss function",sum((xr-my_data[min_i+endpoint_p:])**2), "from ", year[min_i + endpoint_p], " to 2017")
mse = sum((xr-my_data[min_i+endpoint_p:])**2)/(year.size - min_i - endpoint_p )
print(" Taiwan MSE ",mse, "from ", year[min_i + endpoint_p], " to 2017")

#predict the value
mat.title('Taiwan GDP Linear Regression')
mat.plot(year, my_data, 'g.--')
mat.plot(year[min_i+endpoint_p:], xr, 'r.-')
mat.legend(['original', 'regression'])
mat.show()

'''