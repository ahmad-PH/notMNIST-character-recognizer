from scipy import misc
from time import sleep
import numpy as np
import glob
import random
import datetime
from special_functions import *
from utility import *


class Example:
    # input should be a integer array of length 28*28
    # label should be index of english letter (e.g. a=0, b=1, ...)
    def __init__(self, input, label):
        self.input = input
        self.label = label

        self.label_vector = [0] * 10
        self.label_vector[label] = 1

class Neuron:
    def __init__(self, n_inputs, act_func, is_first_layer):
        self.n_inputs = n_inputs

        if is_first_layer:
            self.weights = [1] * n_inputs
            self.bias = 0
        else:
            # ONLY FOR TEST
            # self.weights = [1] * n_inputs
            # self.bias = 0
            self.weights = np.random.normal(0, 0.1, n_inputs)
            self.bias = np.random.normal(0, 0.1)

        self.bias = 0
        self.intermediate_output = None
        self.output = None
        self.act_func = act_func
        self.delta = 0 #for backpropagation

    def calculate_output(self, data):
        #temporary, remove after debugging for better performance
        if (len(data) != self.n_inputs):
            # print "len(data) , n_inputs: ", len(data), self.n_inputs, "data itself: ", data
            raise RuntimeError("Neuron::calculate: data len not equal to n_input")

        result = self.bias
        for i in range(len(data)):
            result += data[i] * self.weights[i]
        self.intermediate_output = result
        self.output = self.act_func(self.intermediate_output)

    def dropout(self):
        self.output = 0


class Layer:
    def __init__(self, n_neurons, n_inputs, act_function, is_first_layer):
        if act_function == "sigmoid":
            self.act_function = sigmoid
            self.act_function_derivative = sigmoid_derivative
        elif act_function == "identity":
            self.act_function = identity
            self.act_function_derivative = identity_derivative
        else:
            raise RuntimeError("invalid act_function type: " + act_function)

        self.n_neurons = n_neurons

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs, self.act_function, is_first_layer))

    def get_output_all(self):
        return [neuron.output for neuron in self.neurons]

    def len(self):
        return len(self.neurons)


class ANN:
    def __init__(self, layers_structure, act_function_str, regularization_type = None, dropout_probabilty = 0):
        self.learning_rate = 0.05
        self._lambda = 0.03

        self.layers = []
        num_input = 1
        for i, layer_size in enumerate(layers_structure):
            self.layers.append(Layer(layer_size, num_input,
                                     "identity" if i == len(layers_structure) - 1 else act_function_str,
                                     i == 0))
            num_input = layer_size

        if regularization_type != "L2" and regularization_type is not None:
            raise RuntimeError("invalid regularization_type: " + regularization_type)

        self.regularization_type = regularization_type
        self.dropout_probability = dropout_probabilty

    def feed_forward(self, data):
        current_input = data

        for i, layer in enumerate(self.layers):
            # print "layer number: ", i
            for j, neuron in enumerate(layer.neurons):
                if np.random.uniform(0, 1) < self.dropout_probability:
                    neuron.dropout()
                elif i == 0:
                    # print "calcing first layer neuron : ", current_input[j]
                    neuron.calculate_output([current_input[j]])
                else:
                    neuron.calculate_output(current_input)

                # print "neuron output is", neuron.output

            current_input = layer.get_output_all()

    def get_output(self):
        return self.layers[-1].get_output_all()

    def calculate_accuracy_percent(self, data):
        n_correct_answers = 0
        for datum in data:
            if self.answer(datum) == datum.label:
                n_correct_answers += 1

        return 100 * (n_correct_answers / len(data))

    def answer(self, example):
        self.feed_forward(example.input)
        return max(self.get_output())[0]

    def update_weights(self, err):
        t = Timer()
        # t.record()
        self.calculate_deltas(err)
        # t.print_elapsed("calc deltas")

        t.record()
        #for non-first layers
        # for i, layer in enumerate(self.layers[1:]):
        for i in xrange(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                # bias_increment = self.learning_rate * (neuron.delta) #- self._lambda * neuron.bias)
                # print "bias ", k, " of layer", i, "being incremented by ", bias_increment
                neuron.bias += self.learning_rate * (neuron.delta) #- self._lambda * neuron.bias)
                for j, weight in enumerate(neuron.weights):
                    # weight_increment = self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta) #- self._lambda * weight)
                    # print "weight ", j, k, "of layer: ", i , "being incremented by: ", weight_increment
                    weight += self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta) #- self._lambda * weight)
        t.print_elapsed("rest of update weights")

    def calculate_deltas(self, err):
        # expected_output = example.label_vector

        # calculate delta for last layer
        for i, neuron in enumerate(self.layers[-1].neurons):
            # print "calculating delta for last layer, neuron", i, " : ", err[i] * self.layers[-1].act_function_derivative(neuron.intermediate_output)
            # err = expected_output[i] - neuron.output
            neuron.delta = err[i] * self.layers[-1].act_function_derivative(neuron.intermediate_output)

        # calculate delta for other layers
        for i in xrange(len(self.layers) - 2, 0, -1):
            for j, neuron in enumerate(self.layers[i].neurons):
                sigma = 0
                for next_layer_neuron in self.layers[i + 1].neurons:
                    sigma += next_layer_neuron.weights[j] * next_layer_neuron.delta
                neuron.delta = self.layers[i].act_function_derivative(neuron.intermediate_output) * sigma


    def train_SGD(self, train_data):
        # print "len train_data", len(train_data)
        t = Timer()
        epochs = 10000

        example_index = 0
        for i in xrange(epochs):
            current_example = train_data[example_index]

            t.record()
            self.feed_forward(current_example.input)
            t.print_elapsed("feed forward")

            t.record()
            err_vector = vector_subtract(current_example.label_vector, self.get_output())
            t.print_elapsed("vector subtraction")

            t.record()
            self.update_weights(err_vector)
            t.print_elapsed("update weights")

            t.record()
            example_index = (example_index + 1) % len(train_data)
            t.print_elapsed("len")

            if i % 50 == 0:
                print "epoch no", i

            # print "ex index:", example_index
            if (i % 1000 == 0 and i != 0):
                print "epoch no", i
                print "train accuracy : ", self.calculate_accuracy_percent(train_data)

            # time1 = datetime.datetime.now()
            # print (datetime.datetime.now() - time1)

    def train_GD(self, train_data):
        epochs = 1000

        for i in xrange(epochs):
            err_vector = [0] * len(self.layers[-1])
            for example in train_data:
                self.feed_forward(example.input)
                partial_err_vector = vector_subtract(example.label_vector, self.get_output())
                vector_add(err_vector, partial_err_vector)
            self.update_weights(err_vector)


test_ratio = 0.1
def read_data():
    result = []
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
        for image_path in glob.glob("../data/" + letter + "/*.png"):
            img = misc.imread(image_path).flatten()
            result.append(Example(img, i))

    random.shuffle(result)

    break_index = int(len(result) * test_ratio)

    test_data = result[0:break_index]
    train_data = result[break_index+1:]

    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = read_data()
    ann = ANN([28*28, 150, 10], "sigmoid")
    print "started training"
    ann.train_SGD(train_data)



    # ex1 = Example([1,0], 0)
    # ex2 = Example([0,1], 1)
    # train_data = [ex1, ex2, ex3]
    #
    # ann = ANN([2,2,2],"identity", None, 0)
    #
    # ann.feed_forward(ex1.input)
    # err_vector = vector_subtract(ex1.label_vector, ann.get_output())
    # print "error vector: ", err_vector
    # ann.update_weights(err_vector)
    #
    # ann.feed_forward(ex2.input)
    # err_vector = vector_subtract(ex2.label_vector, ann.get_output())
    # print "error vector: ", err_vector
    # ann.update_weights(err_vector)






#TODO: draw loss function



# twiddles:
# enable regularization term again
