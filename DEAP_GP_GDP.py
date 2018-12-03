from scipy import linspace, polyval, polyfit, sqrt, stats, randn
import matplotlib.pyplot as mat
from numpy import genfromtxt
import numpy as np

import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# input the dataset
my_data = genfromtxt('japangdp.csv', delimiter=',')
data_s = my_data

use_year  = []

year = np.array(
    [1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978,
     1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
     1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
     2017])

data_year = dict(zip(year, my_data))


# Define new functions
def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0


# set tree operator and constant terminal
pset = gp.PrimitiveSet("MAIN", 10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#will product nan value
#.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
#pset.addPrimitive(operator.pow, 2)
# sin cos will make some value = 0 will output error message TypeError: only length-1 arrays can be converted to Python scalars
#pset.addPrimitive(math.cos, 1)
#pset.addPrimitive(math.sin, 1)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
#set input arguments
pset.renameArguments(ARG0='data_of_year1')
pset.renameArguments(ARG1='data_of_year2')
pset.renameArguments(ARG2='data_of_year3')
pset.renameArguments(ARG3='data_of_year4')
pset.renameArguments(ARG4='data_of_year5')
pset.renameArguments(ARG5='data_of_year6')
pset.renameArguments(ARG6='data_of_year7')
pset.renameArguments(ARG7='data_of_year8')
pset.renameArguments(ARG8='data_of_year9')
pset.renameArguments(ARG9='data_of_year10')
# set the fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# set tool box , tree and inividual and population
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points,targetpoints):
    # set function
   # print("points = ",points)
    func = toolbox.compile(expr=individual)
    # input points and eval the value
    sqerrors = 0
    icounter= 0

    for x in points:
        print("x = ", func(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]))
        sqerrors+= func(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]) - targetpoints[icounter]
        icounter += 1
    # print("szie of year = " , len(data_year) , "size of points" , len(points))

    return sqerrors / len(points),




def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats=mstats,
                                   halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    window_size = 10
    best_ge = 0
    best_i = 0
    best_window_size = 0
    training_size = 10
    interval = 10
    #for i in range(0,year.size - 1 - window_size):
    for i in range(0, 1):
        mat.figure(i)
        #trainning data
        use_year = my_data[i:i+window_size]

        training_data = []
        targetpoint = year[i + interval:i + interval + training_size]

        print("targertpoint size3 = ", len(targetpoint))
        for j in range(0, training_size):
            newdata = my_data[j + i:j + i + interval]
            # newdata= np.insert(newdata, 0, 1)
            print(newdata)

            training_data.append(newdata)
        print(use_year)

        toolbox.register("evaluate", evalSymbReg, points=training_data,targetpoints = targetpoint)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=10)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        pop, log, nof = main()
        func = toolbox.compile(expr=nof[-0])
        print(use_year)
        print(nof[-1])
        '''
        resultset = func(use_year)
        print(resultset)
        # predict the value
        # predict the value
        mat.title('JAPAN GDP GP version ')
        mat.subplot(2, 1, 1)
        mat.legend(['original'])
        mat.plot(use_year, my_data[i:i+window_size], 'g.--')
        mat.legend(['DATA'])
        mat.subplot(2, 1, 2)
        mat.plot(use_year, resultset, 'r.-')
        mat.legend(['GP Symbolic Regression'])
        mat.show(block=False)
'''