from Example import *
from ANN import *
from read_data import *


import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_data, test_data = read_data(800)
    ann = ANN([28*28, 50, 10], "sigmoid", "L2", 0)
    print "done loading"
    ann.train_GD(train_data, test_data)


# special things:
# converting images into black and white (maybe not so special ? :) )
# ability to store and load the network. (with nice user interface :D)