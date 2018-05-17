import ast
from time import sleep

from Neuron import *
from Layer import *
from special_functions import *
from utility import *


class ANN:
    def __init__(self, layers_structure, act_function_str, regularization_type = None, dropout_probabilty = 0):
        self.learning_rate = 0.05
        # self._lambda = 0.03
        self._lambda = 1

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

                # print "neuron output is", neuron.output

            current_input = layer.get_output_all()
            # print "output is: ", layer.get_output_all()

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
        # print "returning: ", max(self.get_output())[0], "for example with label: ", example.label
        return max(self.get_output())[0]

    def update_weights(self, err):
        self.calculate_deltas(err)

        #for non-first layers
        # for i, layer in enumerate(self.layers[1:]):
        for i in xrange(1, len(self.layers)):
            for k, neuron in enumerate(self.layers[i].neurons):
                # print "bias ", k, " of layer", i, "being incremented by ", self.learning_rate * (neuron.delta) #- self._lambda * neuron.bias)
                neuron.bias += self.learning_rate * (neuron.delta) #- self._lambda * neuron.bias)
                for j in xrange(len(neuron.weights)):
                    # print "weight ", j, k, "of layer: ", i , "being incremented by: ", \
                    #     self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta) #- self._lambda * weight)
                    neuron.weights[j] += self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta) #- self._lambda * weight)

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


    def train_SGD(self, train_data, test_data):
        # print "len train_data", len(train_data)
        epochs = 10000

        example_index = 0
        for i in xrange(epochs):
            current_example = train_data[example_index]

            # t.record()
            self.feed_forward(current_example.input)
            # t.print_elapsed()

            # t.record()
            err_vector = vector_subtract(current_example.label_vector, self.get_output())
            # t.print_elapsed("vector subtraction")

            # t.record()
            self.update_weights(err_vector)
            # print "update weight time: ",
            # t.print_elapsed()

            example_index = (example_index + 1) % len(train_data)

            if i % 50 == 0:
                print "epoch no:", i
                # print self

            # print "ex index:", example_index

            if i % 200 == 0 and i != 0:
                print "epoch no", i
                print "train accuracy : ", self.calculate_accuracy_percent(train_data)
                print "test accuracy : ", self.calculate_accuracy_percent(test_data)
                self.store("test.txt")
                break

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
                # print range(self.layers[k].len())
                # print neuron.weights
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
        ann = ANN(layers_structure, lines[1], regularization_type, int(lines[3]))

        line_index = 5
        for layer_index in range(len(layers_structure)):
            if layer_index == 0:
                continue
            for neuron_indx in range(layers_structure[layer_index]):
                ann.layers[layer_index].neurons[neuron_indx].bias = float(lines[line_index])
                # print "reading literal : ", lines[line_index + 1]
                ann.layers[layer_index].neurons[neuron_indx].weights = ast.literal_eval(lines[line_index + 1])
                # print "result : ", ann.layers[layer_index].neurons[neuron_indx].weights
                line_index += 2
        print "end of load print:"
        print ann