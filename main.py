from scipy import misc
import numpy as np
import glob
import random
from math import exp

from utility import *

n_examples = 1024
test_ratio = 0.1

data = [0] * n_examples


class Example:
    # input should be a integer array of length 28*28
    # label should be index of english letter (e.g. a=0, b=1, ...)
    def __init__(self, input, label):
        self.input = input
        self.label = label

class Neuron:
    def __init__(self, n_inputs, act_func):
        self.n_inputs = n_inputs + 1 # +1 is for bias
        self.weights = np.random.normal(0, 0.1, n_inputs + 1)
        self.intermediate_output = None
        self.output = None
        self.act_func = act_func
        self.delta = 0 #for backpropagation

    def calculate(self, data):
        #temporary, remove after debugging for better performance
        if (len(data) != self.n_inputs):
            raise RuntimeError("Neuron::calculate: data len not equal to n_input")

        result = 0
        for i in range(len(data)):
            result += data[i] * self.weights[i]
        self.intermediate_output = result
        self.output = self.act_func(self.intermediate_output)

    def dropout(self):
        self.output = 0

class Layer:
    def __init__(self, n_neurons, n_inputs, act_func):
        self.n_neurons = n_neurons

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs, act_func))

    def get_output_all(self):
        return [neuron.output for neuron in self.neurons]

def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    return exp(-x) / ((1 + exp(-x)) ** 2)

class ANN:
    def __init__(self, layers_structure, act_function, regularization_type = None, dropout_probabilty = 0):
        if act_function == "sigmoid":
            self.act_function = sigmoid
            self.act_function_derivative = sigmoid_derivative
        elif act_function == "linear":
            pass
        else:
            raise RuntimeError("invalid act_function type")

        self.learning_rate = 0.05
        self._lambda = 0.03

        self.layers = []
        num_input = 1
        for layer_size in layers_structure:
            self.layers.append(Layer(layer_size, num_input, act_function))
            num_input = layer_size

        if regularization_type != "L2" and regularization_type is not None:
            raise RuntimeError("invalid regularization_type: " + regularization_type)

        self.regularization_type = regularization_type
        self.dropout_probability = dropout_probabilty


    def feed_forward(self, data):
        data.append(1)
        current_input = data

        for layer in self.layers:

            for neuron in layer.neurons:
                if numpy.random.uniform(0, 1) < self.dropout_probability:
                    neuron.dropout()
                else:
                    neuron.calculate(current_input)

            current_input = layer.get_output_all()
            current_input.append(1)

    def get_output(self):
        return self.layers[-1].get_output_all()

    def calculate_accuracy(self, data):
        n_correct_answers = 0
        for datum in data:
            if self.answer(datum) == datum.label:
                n_correct_answers += 1

        return n_correct_answers / len(data)

    def answer(self, example):
        self.feed_forward(example)
        return max(self.get_output())[0]

    def update_weights(self, example):

        regularization_value = self.calculate_regularization_value()
        expected_output = self.label_to_expected_output(example.label)

        self.calculate_deltas()

        #for first layer
        for i, neuron in enumerate(self.layers[0]):
            neuron.weights[0] += self.learning_rate * (example.input[i] * neuron.delta + self._lambda * neuron.weights[0])

        #for other layers
        for i, layer in enumerate(self.layers[1:]):
            for neuron in layer.neurons:
                for j, weight in enumerate(neuron.weights):
                    weight += self.learning_rate * ( self.layers[i-1] * neuron.delta + self._lambda + weight)


    def calculate_deltas(self):
        # calculate delta for last layer
        for i, neuron in enumerate(self.layers[-1].neurons):
            err = (expected_output[i] - neuron.output) ** 2
            neuron.delta = (-1) * err * self.act_function_derivative(neuron.intermediate_output)

        # calculate delta for other layers
        for i in xrange(len(self.layers) - 2, -1, -1):
            for j, neuron in enumerate(self.layers[i].neurons):
                sigma = 0
                for next_layer_neuron in self.layers[i + 1].neurons:
                    sigma += next_layer_neuron.weights[j] * next_layer_neuron.delta
                neuron.delta = self.act_function_derivative(neuron.intermediate_output) * sigma




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
        result = [0] * len(self.layers[-1])
        result[label] = 1
        return result


def read_data():
    result = []
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
        for image_path in glob.glob("./data/" + letter + "/*.png"):
            img = misc.imread(image_path).flatten()
            result.append(Example(img, i))

    random.shuffle(result)

    break_index = len(result) * test_ratio

    test_data = result[0:break_index]
    train_data = result[break_index+1:]

    return train_data, test_data




if __name__ == "__main__":
    train_data, test_data = read_data()
    ANN([28*28, 150, 10], "sigmoid")







