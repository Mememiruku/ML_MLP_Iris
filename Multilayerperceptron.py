import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

class MLP:
    
    def __init__ (self, data, input_size, epoch, learn_rate, hidden_layer_size, output_layer_size):
        self.data = data
        self.input_size = input_size
        self.epoch = epoch
        self.split(70)
        self.learn_rate = learn_rate
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.init_layers()
        self.graph_error = []
        self.graph_accuracy = []
    
    def init_layers(self):
        self.hidden_layer = Layer(self.hidden_layer_size, self.learn_rate)
        self.output_layer = Layer(self.output_layer_size, self.learn_rate)
        self.init_weight()

    def init_weight(self):
        size = self.input_size
        for layer in [self.hidden_layer] + [self.output_layer]:
            layer.init_weight(size)
            size = len(layer.neurons)
    
    def split (self, part):
        np.random.shuffle(self.data)
        self.train_data = self.data[:int((len(self.data)*part/100))]
        self.test_data = self.data[int(len(self.data)*part/100):]
    
    def feed_forward_network(self, inputs):
        h_output = self.hidden_layer.feed_forward(inputs)
        self.output_layer.feed_forward(h_output)
    
    def backpropagate(self, target):
        #get delta for each layer
        for index, neuron in enumerate(self.output_layer.neurons):
            neuron.get_delta(target[index])
        
        for index_hidden, hidden_neuron in enumerate(self.hidden_layer.neurons):
            d_error_wrt_h = 0
            for index_output, _ in enumerate(self.output_layer.neurons):
                d_error_wrt_h += self.output_layer.neurons[index_output].delta * self.output_layer.neurons[index_output].weights[index_hidden]
            hidden_neuron.delta = d_error_wrt_h*hidden_neuron.get_de_sigmoid()
        
        #updating weights in each neuron
        for neuron in self.output_layer.neurons:
            for index, _ in enumerate(neuron.weights):
                derivatives = neuron.delta * neuron.get_de_squashed_output_to_de_weight(index)
                neuron.weights[index] -= self.learn_rate * derivatives
        for neuron in self.hidden_layer.neurons:
            for index, _ in enumerate(neuron.weights):
                derivatives = neuron.delta * neuron.get_de_squashed_output_to_de_weight(index)
                neuron.weights[index] -= self.learn_rate * derivatives
        self.output_layer.update_bias(self.learn_rate)
        self.hidden_layer.update_bias(self.learn_rate) 

    def train(self, data):
        total_sum_square_error = 0
        for feature in data:
            self.feed_forward_network(feature[:-2])
            sum_square_error = 0
            for index, neuron in enumerate(self.output_layer.neurons):
                sum_square_error += neuron.get_error(feature[-2+index])
            self.backpropagate(feature[-2:])
            total_sum_square_error += sum_square_error
        self.graph_error.append([total_sum_square_error, self.cur_epoch])

    def test(self, data):
        predicted = []
        for feature in data:
            self.feed_forward_network(feature[:-2])
            join =  [self.output_layer.outputs] + [feature[-2:]]
            if int(self.activation(join[0][1])) == int(join[1][1]) and int(self.activation(join[0][0])) == int(join[1][0]):
                predicted.append(bool(True))
            else:
                predicted.append(bool(False))
        accuracy = round(predicted.count(True)/len(predicted)*100, 2)
        self.graph_accuracy.append([accuracy, self.cur_epoch])

    def activation(self, x):
        if x > 0.5:
            return 1
        else:
            return 0

    def plot(self):
        plot_index = [val[1] for val in self.graph_error]
        plt.title("Summary Graph (α = {0})".format(self.learn_rate))
        plt.plot(plot_index, [val[0] for val in self.graph_error])
        plt.plot(plot_index, [val[0] for val in self.graph_accuracy])
        plt.gca().legend(('sum_error','accuracy (%)'))
        plt.ylabel("")
        plt.xlabel("epoch")
        plt.show()


    def run (self, epoch):
        self.cur_epoch = 0
        for cur_epoch in range(epoch):
            self.cur_epoch = cur_epoch + 1
            self.train(self.train_data)
            self.test(self.test_data)
        self.plot()

class Layer:

    def __init__ (self, num_neurons, learn_rate):
        self.bias = np.random.random()
        self.neurons = []
        self.learn_rate = learn_rate
        for _ in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
        
    def feed_forward(self, inputs):
        self.outputs = []
        self.saved_inputs = inputs
        for neuron in self.neurons:
            self.outputs.append(neuron.calculate_output(inputs))
        return self.outputs
        
    def init_weight(self, size):
        for neuron in self.neurons:
            for _ in range(size):
                neuron.weights.append(np.random.random())
    
    def update_bias(self, learn_rate):
        self.learn_rate = learn_rate
        dE_bias = 0
        for neuron in self.neurons:
            dE_bias += neuron.delta * (neuron.squashed_output-(1-neuron.squashed_output))
        self.bias = self.bias - (self.learn_rate*dE_bias)

class Neuron:

    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.net_output = 0
        self.squashed_output = 0
        self.delta = 0
        
    def calculate_output(self, inputs):
        self.net_output = 0
        self.saved_inputs = inputs
        for input, weight in zip(inputs,self.weights):
            self.net_output += input * weight
        self.net_output += self.bias
        self.squashed_output = self.sigmoid(self.net_output)
        return self.squashed_output
    
    def sigmoid(self, x):
        if x>= 0:
            z = np.exp(-x)
            return (1/(1+z))
        else:
            z = np.exp(x)
            return (z/(1+z))

    def get_error(self, target):
        error = 0.5*pow(self.squashed_output - target,2)
        return error

    def get_error_margin(self, target):
        error_margin = self.squashed_output - target
        return error_margin

    def get_de_sigmoid(self):
        de_sigmoid = self.squashed_output + (1-self.squashed_output)
        return de_sigmoid
    
    def get_delta(self, target):
        self.delta = self.get_error_margin(target) * self.get_de_sigmoid()
        return self.delta
    
    def get_de_squashed_output_to_de_weight(self, index):
        return self.saved_inputs[index]

    