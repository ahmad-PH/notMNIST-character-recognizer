import numpy as np

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
            self.weights = np.random.normal(0, 0.1, n_inputs).tolist()
            self.bias = np.random.normal(0, 0.1)

        self.bias = 0
        self.intermediate_output = None
        self.output = None
        self.act_func = act_func
        self.delta = 0 #for backpropagation

    def calculate_output(self, data):
        #temporary, remove after debugging for better performance
        # if (len(data) != self.n_inputs):
        # raise RuntimeError("Neuron::calculate: data len not equal to n_input")

        result = self.bias
        for i in xrange(len(data)):
            result += data[i] * self.weights[i]

        self.intermediate_output = result
        self.output = self.act_func(self.intermediate_output)

    def calculate_output_first_layer(self, data):
        self.output = self.bias + self.weights[0] * data

    def dropout(self):
        self.output = 0