from scipy import misc
import numpy as np
import glob
import random
from math import exp
import datetime

from special_functions import *
from utility import *


class Example:
    # input should be a integer array of length 28*28
    # label should be index of english letter (e.g. a=0, b=1, ...)
    def __init__(self, input, label):
        self.input = input
        self.label = label

class Neuron:
    def __init__(self, n_inputs, act_func, is_first_layer):
        self.n_inputs = n_inputs

        if is_first_layer:
            self.weights = [1] * n_inputs
            self.bias = 0
        else:
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
            raise RuntimeError("invalid act_function type")

        self.n_neurons = n_neurons

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs, self.act_function, is_first_layer))

    def get_output_all(self):
        return [neuron.output for neuron in self.neurons]

    def len(self):
        return len(self.neurons)


def vector_subtract(vec1, vec2):
    result = []
    for i in xrange(len(vec1)):
        result.append(vec1[i] - vec2[i])
    return result

def vector_add(vec1, vec2):
    result = []
    for i in xrange(len(vec1)):
        result.append(vec1[i] + vec2[i])
    return result



class ANN:
    def __init__(self, layers_structure, act_function, regularization_type = None, dropout_probabilty = 0):
        self.learning_rate = 0.05
        self._lambda = 0.03

        self.layers = []
        num_input = 1
        is_first_layer = True
        for layer_size in layers_structure[1:-1]:
            self.layers.append(Layer(layer_size, num_input, act_function, is_first_layer))
            num_input = layer_size
            is_first_layer = False

        self.layers.append(Layer(layers_structure[-1], num_input, "identity", False))

        if regularization_type != "L2" and regularization_type is not None:
            raise RuntimeError("invalid regularization_type: " + regularization_type)

        self.regularization_type = regularization_type
        self.dropout_probability = dropout_probabilty


    def feed_forward(self, data):
        current_input = data

        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                if np.random.uniform(0, 1) < self.dropout_probability:
                    neuron.dropout()
                elif i == 0:
                    print "calcing first layer neuron : ", current_input[j]
                    neuron.calculate_output([current_input[j]])
                else:
                    neuron.calculate_output(current_input)

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
        self.calculate_deltas(err)

        #for non-first layers
        # for i, layer in enumerate(self.layers[1:]):
        for i in xrange(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                print "bias being incremented by ", self.learning_rate * (neuron.delta + self._lambda * neuron.bias)
                neuron.bias += self.learning_rate * (neuron.delta + self._lambda * neuron.bias)
                for j, weight in enumerate(neuron.weights):
                    weight += self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta + self._lambda + weight)

    def calculate_deltas(self, err):
        # expected_output = self.label_to_expected_output(example.label)

        # calculate delta for last layer
        for i, neuron in enumerate(self.layers[-1].neurons):
            # err = expected_output[i] - neuron.output
            neuron.delta = (-1) * err[i] * self.layers[-1].act_function_derivative(neuron.intermediate_output)

        # calculate delta for other layers
        for i in xrange(len(self.layers) - 2, 0, -1):
            for j, neuron in enumerate(self.layers[i].neurons):
                sigma = 0
                for next_layer_neuron in self.layers[i + 1].neurons:
                    sigma += next_layer_neuron.weights[j] * next_layer_neuron.delta
                neuron.delta = self.layers[i].act_function_derivative(neuron.intermediate_output) * sigma


    def calculate_regularization_value(self):
        if regularization_type == "L2":
            regularization_value = 0
            for layer in self.layers:
                for neuron in layer.neurons:
                    for weight in neuron.weights:
                        self.regularization_value += weight ** 2
        else:
            regularization_value = 0
        return regularization_value

    def label_to_expected_output(self, label):
        result = [0] * self.layers[-1].len()
        result[label] = 1
        return result


    def train_SGD(self, train_data):
        # print "len train_data", len(train_data)
        epochs = 100000

        example_index = 0
        for i in xrange(epochs):
            current_example = train_data[example_index]
            self.feed_forward(current_example.input)
            err_vector = vector_subtract(self.label_to_expected_output(current_example.label), self.get_output())
            self.update_weights(err_vector)

            example_index = (example_index + 1) % len(train_data)

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
                partial_err_vector = vector_subtract(self.label_to_expected_output(example.label), self.get_output())
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
    # ann = ANN([28*28, 150, 10], "sigmoid")


    ex1 = Example([1,0,0], 0)
    ex2 = Example([0,1,0], 1)
    ex3 = Example([0,0,1], 2)
    train_data = [ex1, ex2, ex3]

    ann = ANN([3,3],"identity", None, 0)

    ann.feed_forward(ex1.input)

    # ann.train_SGD(train_data)




#TODO: draw loss function




