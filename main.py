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
    def __init__(self, input, label, label_vector_len = 10):
        self.input = input
        self.label = label

        self.label_vector = [0] * label_vector_len
        self.label_vector[label] = 1

class Neuron:
    def __init__(self, n_inputs, act_func, is_first_layer):
        self.n_inputs = n_inputs

        if is_first_layer:
            self.weights = [1] * n_inputs
            self.bias = 0
        else:
            # ONLY FOR TEST
            self.weights = [1] * n_inputs
            self.bias = 0
            # self.weights = np.random.normal(0, 0.1, n_inputs)
            # self.bias = np.random.normal(0, 0.1)

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

        # self.intermediate_output =  sum([data[i] * self.weights[i] for i in range(len(data))]) + self.bias
        self.intermediate_output = result
        self.output = self.act_func(self.intermediate_output)

    def calculate_output_first_layer(self, data):
        self.output = self.bias + self.weights[0] * data

    def dropout(self):
        self.output = 0


class Layer:
    def __init__(self, n_neurons, n_inputs, act_function_str, is_first_layer):
        if act_function_str == "sigmoid":
            self.act_function = sigmoid
            self.act_function_derivative = sigmoid_derivative
        elif act_function_str == "identity":
            self.act_function = identity
            self.act_function_derivative = identity_derivative
        else:
            raise RuntimeError("invalid act_function type: " + act_function_str)

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


class ANN:
    def __init__(self, layers_structure, act_function_str, regularization_type = None, dropout_probabilty = 0):
        self.learning_rate = 0.05
        # self._lambda = 0.03
        self._lambda = 1

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
                # bias_increment = self.learning_rate * (neuron.delta) #- self._lambda * neuron.bias)
                print "bias ", k, " of layer", i, "being incremented by ", self.learning_rate * (neuron.delta) #- self._lambda * neuron.bias)
                neuron.bias += self.learning_rate * (neuron.delta) #- self._lambda * neuron.bias)
                for j in xrange(len(neuron.weights)):
                    # weight_increment = self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta) #- self._lambda * weight)
                    print "weight ", j, k, "of layer: ", i , "being incremented by: ", \
                        self.learning_rate * (self.layers[i-1].neurons[j].output * neuron.delta) #- self._lambda * weight)
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


    def train_SGD(self, train_data):
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

            # print "ex index:", example_index

            if i % 200 == 0 and i != 0:
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


test_ratio = 0.1
def read_data(max = None):
    result = []
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


    max_images_per_letter = float("+inf")
    if max != None:
        max_images_per_letter = max / len(letters)

    for i, letter in enumerate(letters):

        num_images_read_for_letter = 0
        for image_path in glob.glob("../data/" + letter + "/*.png"):

            img = misc.imread(image_path).flatten()
            vector_div(img, 32)
            result.append(Example(img, i, 10))

            num_images_read_for_letter += 1
            if num_images_read_for_letter >= max_images_per_letter:
                break

    random.shuffle(result)

    break_index = int(len(result) * test_ratio)

    test_data = result[0:break_index]
    train_data = result[break_index+1:]

    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = read_data(1000)
    ann = ANN([28*28, 50, 10], "sigmoid")
    print "done loading"
    ann.train_SGD(train_data)



    # ex1 = Example([1,0], 0, 2)
    # ex2 = Example([0,1], 1, 2)
    # train_data = [ex1, ex2]
    #
    # ann = ANN([2, 2, 2], "identity", None, 0)
    #
    # ann.feed_forward(ex1.input)
    # err_vector = vector_subtract(ex1.label_vector, ann.get_output())
    # print "ann out:", ann.get_output()
    # print "label vec:", ex1.label_vector
    # print "error vector: ", err_vector
    # ann.update_weights(err_vector)
    #
    # print ann


    # print "2nd example"
    # ann.feed_forward(ex2.input)
    # err_vector = vector_subtract(ex2.label_vector, ann.get_output())
    # print "ann out:", ann.get_output()
    # print "label vec:", ex2.label_vector
    # print "error vector: ", err_vector
    # ann.update_weights(err_vector)


    # ann.feed_forward(ex2.input)
    # err_vector = vector_subtract(ex2.get_label_vector(2), ann.get_output())
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