from Example import *
from ANN import *
from read_data import *

if __name__ == "__main__":
    train_data, test_data = read_data(600)
    ann = ANN([28*28, 50, 10], "sigmoid")
    print "done loading"
    ann.train_SGD(train_data, test_data)


    # ex1 = Example([1,0], 0, 2)
    # ex2 = Example([0,1], 1, 2)
    # train_data = [ex1, ex2]

    # ann = ANN([2, 2, 2], "identity", None, 0)

    # ann.feed_forward(ex1.input)
    # err_vector = vector_subtract(ex1.label_vector, ann.get_output())
    # print "ann out:", ann.get_output()
    # print "label vec:", ex1.label_vector
    # print "error vector: ", err_vector
    # ann.update_weights(err_vector)


    # print "2nd example"
    # ann.feed_forward(ex2.input)
    # err_vector = vector_subtract(ex2.label_vector, ann.get_output())
    # print "ann out:", ann.get_output()
    # print "label vec:", ex2.label_vector
    # print "error vector: ", err_vector
    # ann.update_weights(err_vector)





#TODO: draw loss function

#TODO (debug):
# double layer
# test with sigmoid (?)
# multi example


# twiddles:
# enable regularization term again
# restore learning rate back to 0.05


#TODO:
# consider shor-circuiting first layer in case performance stinks

#special things:
# converting images into black and white (maybe not so special ? :) )