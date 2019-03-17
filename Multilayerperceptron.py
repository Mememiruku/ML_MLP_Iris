import numpy as np
import math
import matplotlib.pyplot as plt

class MLP(object):
    def __init__(self, data, input_size, hidden_layers_size, output_layer_size, epoch, learn_rate = 0.1):
        self.data = data
        self.input_size = input_size
        self.data_normal_split(80)
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.hidden_layers_size = hidden_layers_size
        self.output_layer_size = output_layer_size
        self.init_layers()
        self.graphof_sum_square_error = []
        self.graphof_accuracy = []

    def init_layers(self):
        self.hidden_layers = Layer(self.hidden_layers_size, self.learn_rate)
        self.output_layer = Layer(self.output_layer_size, self.learn_rate)
        self.init_weights()

    def init_weights(self):
        prev_layer = self.input_size
        for layer in [self.hidden_layers] + [self.output_layer]:
            layer.init_weights(prev_layer)
            prev_layer = len(layer.neurons)
        
    def data_normal_split(self, percentage):
        np.random.shuffle(self.data)
        self.train_data = self.data[:int((len(self.data))*percentage/100)]
        self.test_data = self.data[int((len(self.data))*percentage/100):]

    def feed_forward_all(self, inputs):
        hidden_out = self.hidden_layers.feed_forward(inputs)
        self.output_layer.feed_forward(hidden_out)

    def backpropagate_all(self, target):
        
        for idx, neuron in enumerate(self.output_layer.neurons):
            neuron.get_delta(target[idx])
      
        for idx_hidden, neuron_hidden in enumerate(self.hidden_layers.neurons):
            d_error_to_hl_out = 0
            for idx_out, _ in enumerate(self.output_layer.neurons):
                d_error_to_hl_out += self.output_layer.neurons[idx_out].delta * self.output_layer.neurons[idx_out].weights[idx_hidden]
            neuron_hidden.delta = d_error_to_hl_out*neuron_hidden.get_de_sigmoid()
       
        for neuron in self.output_layer.neurons:
            for idx, _ in enumerate(neuron.weights):
                ratio = neuron.delta * neuron.get_de_clean_out_per_dweight(idx)
                neuron.weights[idx] -= self.learn_rate * ratio
        for neuron in self.hidden_layers.neurons:
            for idx, _ in enumerate(neuron.weights):
                ratio = neuron.delta * neuron.get_de_clean_out_per_dweight(idx)
                neuron.weights[idx] -= self.learn_rate * ratio
        self.output_layer.update_bias(self.learn_rate)
        self.hidden_layers.update_bias(self.learn_rate)
      

    def train(self, data):
        total_sumof_square_error = 0
        for feature in data:
            self.feed_forward_all(feature[:-2]) 
            sumof_square_error = 0
            for idx, neuron in enumerate(self.output_layer.neurons):
                sumof_square_error += neuron.get_error(feature[-2+idx])
            self.backpropagate_all(feature[-2:])
            total_sumof_square_error += sumof_square_error
            log_error = math.log1p(total_sumof_square_error)
        self.graphof_sum_square_error.append([total_sumof_square_error, self.cur_epoch]) 
        print("epoch {0:<4} - Sum error : {1:<15}".format(self.cur_epoch,round(total_sumof_square_error,10)),end=" ")

    def test(self,data):
        predicted = []
        for feature in data:
            self.feed_forward_all(feature[:-2])
            join =  [self.output_layer.outputs] + [feature[-2:]]
            if int(self.activation_func(join[0][1])) == int(join[1][1]) and int(self.activation_func(join[0][0])) == int(join[1][0]):
                predicted.append(bool(True))
            else:
                predicted.append(bool(False))
        accuracy = round(predicted.count(True)/len(predicted)*100, 2)
        self.graphof_accuracy.append([accuracy, self.cur_epoch])
        print("Accuracy  : {0}".format(accuracy))

    def activation_func(self, x):
        if x < 0.5:
            return 0
        else:
            return 1
    

    def run(self):
        self.cur_epoch = 0
        for cur_epoch in range(self.epoch):
            self.cur_epoch = cur_epoch + 1
            self.train(self.train_data)
            self.test(self.train_data)
        self.plot_error()
        self.plot_accuracy()
    
    def plot_error(self):
        #Create error graph
        plot_index = [val[1] for val in self.graphof_sum_square_error]
        plt.title("Error graph (α = {0})".format(self.learn_rate))
        plt.plot(plot_index,[val[0] for val in self.graphof_sum_square_error])
        plt.gca().legend(('log(sum_error)'))
        plt.ylabel("Error")
        plt.xlabel("Epoch")
        plt.show()
    
    def plot_accuracy(self):
        #Create accuracy graph
        plot_index = [val[1] for val in self.graphof_accuracy]
        plt.title("Accuracy graph (α = {0})".format(self.learn_rate))
        plt.plot(plot_index, [val[0] for val in self.graphof_accuracy])
        plt.gca().legend(('Accuracy (%)'))
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Epoch")
        plt.show()
class Neuron(object):
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.clean_out = 0
        self.sigmoid_out = 0
        self.delta = 0

    def calculate_output(self, inputs):
        self.saved_inputs = inputs
        self.clean_out = 0
        for input, weight in zip(inputs,self.weights):
            self.clean_out += input * weight
        self.clean_out += self.bias
        self.sigmoid_out = self.sigmoid(self.clean_out)
        return self.sigmoid_out
        
    def sigmoid(self, x):
        if x >= 0:
            z = math.exp(-x)
            return 1/(1+z)
        else:
            z = math.exp(x)
            return z/(1+z)

    def step_func(self, x):
        if x < 0.5:
            return 0
        else:
            return 1
    
    def get_error(self, target): 
        error = pow(self.sigmoid_out - target,2)/2 
        return error

    def get_margin_of_error(self, target): 
        margin_of_error = self.sigmoid_out - target
        return margin_of_error

    def get_de_sigmoid(self):
        de_sigmoid_out = self.sigmoid_out*(1-self.sigmoid_out)
        return de_sigmoid_out

    def get_delta(self, target):
        self.delta = self.get_margin_of_error(target)*self.get_de_sigmoid()
        return self.delta
    
    def get_de_clean_out_per_dweight(self, index):
        return self.saved_inputs[index]

class Layer(object):
    def __init__(self, size, learn_rate):
        self.bias = np.random.random()
        self.neurons = []
        self.learn_rate = learn_rate
        for _ in range(size):
            self.neurons.append(Neuron(self.bias))
    
    def feed_forward(self, inputs):
        self.outputs = []
        self.saved_inputs = inputs
        for neuron in self.neurons:
            self.outputs.append(neuron.calculate_output(inputs))
        return self.outputs

    def init_weights(self, prev_size):
        for neuron in self.neurons:
            for _ in range(prev_size):
                neuron.weights.append(np.random.random())
    
    def update_bias(self, learn_rate):
        bias_error_derivative = 0 # dE/do * do/dz * dz/db
        for neuron in self.neurons:
            bias_error_derivative += (neuron.delta)*(neuron.sigmoid_out-(1-neuron.sigmoid_out))
        self.bias = self.bias - (learn_rate*bias_error_derivative)