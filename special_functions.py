def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    return exp(-x) / ((1 + exp(-x)) ** 2)

def identity(x):
    return x

def identity_derivative(x):
    return 1
