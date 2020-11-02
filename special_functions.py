from math import *

def sigmoid(x):
    # try:
        return 1 / (1 + exp(-x))
    # except OverflowError as ex:
    #     if x > 300:
    #         return 1
    #     elif x < 300:
    #         return 0
    #     else:
    #         print "overflow error in sigmoid: " + str(x)
    #         exit()

def sigmoid_derivative(x):
    # try:
        return exp(-x) / ((1 + exp(-x)) ** 2)
    # except OverflowError as ex:
    #     if abs(x) > 300:
    #         return 0
    #     else:
    #         print "overflow error in sigmoid_derivative: " + str(x)
    #         exit()

def identity(x):
    return x

def identity_derivative(x):
    return 1
