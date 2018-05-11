from scipy import misc
import numpy as np
import glob
import random
from math import exp

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
        self.weights = np.random.normal(0, 0.2, n_inputs + 1)
        self.intermediate_output = None
        self.output = None
        self.act_func = act_func

    def calculate(self, data):
        #temporary, remove after debugging for better performance
        if (len(data) != self.n_inputs):
            raise RuntimeError("Neuron::calculate: data len not equal to n_input")

        result = 0
        for i in range(len(data)):
            result += data[i] * self.weights[i]
        self.intermediate_output = result
        self.output = self.act_func(self.intermediate_output)

class Layer:
    def __init__(self, n_neurons, n_inputs, act_func):
        self.n_neurons = n_neurons

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs, act_func))

    def get_output_all(self):
        return [neuron.output for neuron in self.neurons]

class ANN:
    def __init__(self, layers_structure, act_function):
        self.act_function = act_function

        self.layers = []
        num_input = 1
        for layer_size in layers_structure:
            self.layers.append(Layer(layer_size, 1, act_function))
            num_input = layer_size

    def feed_forward(self, data):
        data.append(1)
        current_input = data

        for layer in self.layers:

            for neuron in layer.neurons:
                neuron.calculate(current_input)

            current_input = layer.get_output_all()
            current_input.append(1)

    def get_output(self):
        return self.layers[-1].get_output_all();


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


def sigmoid(x):
    return 1 / (1 + exp(-x))

if __name__ == "__main__":
    train_data, test_data = read_data()
    ANN([28*28, 150, 10], sigmoid)







