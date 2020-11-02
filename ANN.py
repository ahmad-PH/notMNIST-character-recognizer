import ast
from time import sleep

from Neuron import *
from Layer import *
from special_functions import *
from utility import *

import matplotlib.pyplot as plt

class ANN:
    def __init__(self, layers_structure, act_function_str, regularization_type = None, dropout_probabilty = 0):
        self.learning_rate = 0.05
        self._lambda = 0.03
        self.act_function_str = act_function_str

        self.layers = []
        num_input = 1
        for i, layer_size in enumerate(layers_structure):
            self.layers.append(Layer(layer_size, num_input,
                                     "identity" if i == len(layers_structure) - 1 or i == 0 else act_function_str,
                                     i == 0))
            num_input = layer_size

        if regularization_type != "L2" and regularization_type is not None:
            raise RuntimeError("invalid regularization_type: " + regularization_type)

        self.regularization_type = regularization_type
        self.dropout_probability = dropout_probabilty

        self.plot_data = {}
        self.reset_plot_data()


    def reset_plot_data(self):
        self.plot_data = {'loss_func': [], 'train_acc': [], 'test_acc': [], 'epochs': []}

    def feed_forward(self, data):

        # first layer:
        for j, neuron in enumerate(self.layers[0].neurons):
            if np.random.uniform(0, 1) < self.dropout_probability:
                neuron.dropout()
            else:
                neuron.calculate_output_first_layer(data[j])

        current_input = self.layers[0].get_output_all()

        # for other layers:
        for i, layer in enumerate(self.layers[1:]):
            for j, neuron in enumerate(layer.neurons):
                if np.random.uniform(0, 1) < self.dropout_probability:
                    neuron.dropout()
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

        return 100.0 * (float(n_correct_answers) / len(data))

    def answer(self, example):
        self.feed_forward(example.input)
        return max(self.get_output())[0]

    def update_weights(self, err_vector):
        self.calculate_deltas(err_vector)

        #for non-first layers
        for i in xrange(1, len(self.layers)):
            for k, neuron in enumerate(self.layers[i].neurons):
                reg_term = -(self._lambda * neuron.bias) if self.regularization_type == "L2" else 0
                neuron.bias += self.learning_rate * (neuron.delta + reg_term)

                for j, weight in enumerate(neuron.weights):
                    reg_term = -(self._lambda * weight) if self.regularization_type == "L2" else 0
                    neuron.weights[j] += self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta + reg_term)

    def calculate_deltas(self, err):
        # expected_output = example.label_vector

        # calculate delta for last layer
        for i, neuron in enumerate(self.layers[-1].neurons):
            neuron.delta = err[i] * self.layers[-1].act_function_derivative(neuron.intermediate_output)

        # calculate delta for other layers
        for i in xrange(len(self.layers) - 2, 0, -1):
            for j, neuron in enumerate(self.layers[i].neurons):
                sigma = 0
                for next_layer_neuron in self.layers[i + 1].neurons:
                    sigma += next_layer_neuron.weights[j] * next_layer_neuron.delta
                neuron.delta = self.layers[i].act_function_derivative(neuron.intermediate_output) * sigma


    def reset_accumulated_update(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.reset_accumulated_update()

    def accumulate_update(self, err_vector):
        self.calculate_deltas(err_vector)

        # for non-first layers
        for i in xrange(1, len(self.layers)):
            for k, neuron in enumerate(self.layers[i].neurons):
                reg_term = -(self._lambda * neuron.bias) if self.regularization_type == "L2" else 0
                neuron.accumulated_bias_update += self.learning_rate * (neuron.delta + reg_term)

                for j, weight in enumerate(neuron.weights):
                    reg_term = -(self._lambda * weight) if self.regularization_type == "L2" else 0
                    neuron.accumulated_weight_update[j] += self.learning_rate * (self.layers[i - 1].neurons[j].output * neuron.delta + reg_term)

    def apply_accumulated_update(self, data_size):
        for k, layer in enumerate(self.layers[1:]):
            for i, neuron in enumerate(layer.neurons):
                # print "setting neuron bias to : ", float(neuron.accumulated_bias_update) / data_size
                neuron.bias += float(neuron.accumulated_bias_update) / data_size
                for j in xrange(len(neuron.weights)):
                    # print "setting neuron weight ", k, i, j , "to:", float(neuron.accumulated_weight_update[j]) / data_size
                    neuron.weights[j] += float(neuron.accumulated_weight_update[j]) / data_size

    def train_SGD(self, train_data, test_data):
        try:
            epochs = 10000

            example_index = 0
            for i in xrange(epochs):
                current_example = train_data[example_index]

                self.feed_forward(current_example.input)
                err_vector = vector_subtract(current_example.label_vector, self.get_output())
                self.plot_data['loss_func'].append(vector_power_two(err_vector))
                self.update_weights(err_vector)

                example_index = (example_index + 1) % len(train_data)

                if i % 100 == 0:
                    print "epoch no:", i

                if i % 400 == 0 and i != 0:
                    print "calculating accuracies ... press ctrl+c to stop before next epoch begins."
                    train_acc = self.calculate_accuracy_percent(train_data)
                    print "train accuracy : ", train_acc
                    test_acc = self.calculate_accuracy_percent(test_data)
                    print "test accuracy : ", test_acc

                    self.plot_data['epochs'].append(i)
                    self.plot_data['train_acc'].append(train_acc)
                    self.plot_data['test_acc'].append(test_acc)
                    # sleep(1)

        except KeyboardInterrupt:
            print ""

        self.show_plots()
        self.save_if_user_confirms()


    def train_GD(self, train_data, test_data):
        try:
            epochs = 1000

            for i in xrange(epochs):
                if i % 1 == 0:
                    print "epoch no", i

                self.reset_accumulated_update()
                for example in train_data:
                    self.feed_forward(example.input)
                    err_vector = vector_subtract(example.label_vector, self.get_output())
                    self.accumulate_update(err_vector)

                vector_div_inplace(err_vector, len(train_data))
                self.plot_data['loss_func'].append(vector_power_two(err_vector))

                self.apply_accumulated_update(len(train_data))

                if i % 1 == 0:
                    print "calculating accuracies ... press ctrl+c to stop before next epoch begins."
                    train_acc = self.calculate_accuracy_percent(train_data)
                    print "train accuracy : ", train_acc
                    test_acc = self.calculate_accuracy_percent(test_data)
                    print "test accuracy : ", test_acc

                    self.plot_data['epochs'].append(i)
                    self.plot_data['train_acc'].append(train_acc)
                    self.plot_data['test_acc'].append(test_acc)
                    # sleep(1)

        except KeyboardInterrupt:
            print ""

        self.show_plots()
        self.save_if_user_confirms()

    def save_if_user_confirms(self):
        user_input = ask_for_user_input("would you like to save the network? (y/n)", ["y", "n"])
        if user_input == "y":
            file_name = ask_for_user_input("enter file name:")
            self.store("../networks/" + file_name)
            print "saved successfully."
        else:
            print "quitting without saving the network."

    def show_plots(self):
        plt.plot(self.plot_data['loss_func'])
        plt.title('loss function')
        plt.show()

        plt.plot(self.plot_data['epochs'], self.plot_data['train_acc'], 'r-',
                 self.plot_data['epochs'], self.plot_data['test_acc'], 'g-')
        plt.show()


    def __repr__(self):
        result = "structure: \n"
        for i, layer in enumerate(self.layers):
            result += "\t layer " + str(i) + ": " + str(layer) + "\n"

        result += "weights: \n"
        for k, layer in enumerate(self.layers[1:]):
            result += "\tlayer " + str(k+1) + ":\n"
            for j, neuron in enumerate(layer.neurons):
                result += "\t\t neuron " + str(j) + ": "
                result += "b: " + str(neuron.bias) + "\t"
                result += "w: " + str([neuron.weights[i] for i in range(self.layers[k].len())]) + "\n"

        return result


    def store(self, filename):
        file = open(filename, "w")
        file.write(str([layer.len() for layer in self.layers]) + "\n")
        file.write(self.act_function_str + "\n")
        file.write(str(self.regularization_type) + "\n")
        file.write(str(self.dropout_probability) + "\n")
        file.write("\n")
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                file.write(str(neuron.bias) + "\n")
                file.write(str(neuron.weights) + "\n")

    @staticmethod
    def load(filename):
        lines = open(filename, "r").read().splitlines()
        layers_structure = ast.literal_eval(lines[0])
        regularization_type = lines[2] if lines[2] != "None" else None
        result = ANN(layers_structure, lines[1], regularization_type, int(lines[3]))

        line_index = 5
        for layer_index in range(len(layers_structure)):
            if layer_index == 0:
                continue
            for neuron_indx in range(layers_structure[layer_index]):
                result.layers[layer_index].neurons[neuron_indx].bias = float(lines[line_index])
                result.layers[layer_index].neurons[neuron_indx].weights = ast.literal_eval(lines[line_index + 1])
                line_index += 2

        return result
