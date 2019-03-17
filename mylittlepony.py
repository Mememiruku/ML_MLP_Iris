import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import Multilayerperceptron

path = Path(__file__).parents[0]
inputfile = str(path) + "\\iris.csv"
iris = np.genfromtxt(inputfile, skip_header=True, delimiter=",")

mlp_init = Multilayerperceptron.MLP(data=iris, input_size=4, epoch=1000, learn_rate=0.1, hidden_layers_size=7, output_layer_size=2).run()

