from Neuron import *
from special_functions import *

class Layer:
    def __init__(self, n_neurons, n_inputs, act_function_str, is_first_layer):
        if act_function_str == "sigmoid":
            self.act_function = sigmoid
            self.act_function_derivative = sigmoid_derivative
        elif act_function_str == "identity":
            self.act_function = identity
            self.act_function_derivative = identity_derivative
        else:
            raise RuntimeError("invalid act_function type: \"" + act_function_str + "\"")

        self.act_function_str = act_function_str
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.is_first_layer = is_first_layer

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs, self.act_function, is_first_layer))

    def get_output_all(self):
        return [neuron.output for neuron in self.neurons]

    def len(self):
        return len(self.neurons)

    def __repr__(self):
        return "len: " + str(self.len()) + ", n_inputs: " + str(self.n_inputs) + \
               ", act_func: " + str(self.act_function_str) + ", first_layer: " +  str(self.is_first_layer)