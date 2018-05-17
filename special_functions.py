from math import *

def sigmoid(x):
    try:
        return 1 / (1 + exp(-x))
    except OverflowError as ex:
        print "overflow error in sigmoid: " + str(x)
        exit()

def sigmoid_derivative(x):
    return exp(-x) / ((1 + exp(-x)) ** 2)

def identity(x):
    return x

def identity_derivative(x):
    return 1
