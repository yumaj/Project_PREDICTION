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
import deap


class DEAP_GP_model():

    def __init__(self,num_generation,input_size):

        # Define new functions
        def safeDiv(left, right):
            try:
                return left / right
            except ZeroDivisionError:
                return 0


        self.input_size = input_size

        # set tree operator and constant terminal
        self.pset = gp.PrimitiveSet("MAIN",60)
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        #pset.addPrimitive(operator.mul, 2)
        #will product nan value
        #.addPrimitive(safeDiv, 2)
        self.pset.addPrimitive(operator.neg, 1)
        #pset.addPrimitive(operator.pow, 2)
        # sin cos will make some value = 0 will output error message TypeError: only length-1 arrays can be converted to Python scalars
        #pset.addPrimitive(math.cos, 1)
        #pset.addPrimitive(math.sin, 1)

        self.pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
        #set input arguments
        self.pset.renameArguments(ARG0='data_of_year1')
        self.pset.renameArguments(ARG1='data_of_year2')
        self.pset.renameArguments(ARG2='data_of_year3')
        self.pset.renameArguments(ARG3='data_of_year4')
        self.pset.renameArguments(ARG4='data_of_year5')
        self.pset.renameArguments(ARG5='data_of_year6')
        self.pset.renameArguments(ARG6='data_of_year7')
        self.pset.renameArguments(ARG7='data_of_year8')
        self.pset.renameArguments(ARG8='data_of_year9')
        self.pset.renameArguments(ARG9='data_of_year10')


        # set the fitness function
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # set tool box , tree and inividual and population
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=1, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.nof_expr = None
        self.num_generation = num_generation
    def vaild(self,test_X):
        
        nof = self.nof_expr
        func = self.toolbox.compile(expr=nof[-1])
        ansnum = []
        for i in range(0,len(test_X)):
            ansnum.append(func(*test_X[i]))

        return ansnum





    def evalSymbReg(self,individual, points,targetpoints):
        # set function
        # print("points = ",points)
        func = self.toolbox.compile(expr=individual)
        # input points and eval the value
        sqerrors = 0
        icounter= 0

        for x in points:
            #print("x = ", func(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]))
            sqerrors+= sqrt(((func(*x) - targetpoints[icounter])**2))
            icounter += 1
        # print("szie of year = " , len(data_year) , "size of points" , len(points))

        return sqerrors / len(points),




    def train(self,train_data_x,train_data_y):

        random.seed(318)


        self.toolbox.register("evaluate", self.evalSymbReg, points=train_data_x,targetpoints = train_data_y)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=10)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


        pop = self.toolbox.population(n=100)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        pop, log = algorithms.eaSimple(pop, self.toolbox, 0.5, 0.1, self.num_generation, stats=mstats,
                                    halloffame=hof, verbose=True)
        self.nof_expr = hof
        return pop, log, hof
        
        #muti times start

