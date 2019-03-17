import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import Multilayerperceptron

path = Path(__file__).parents[0]
inputfile = str(path) + "\\iris.csv"
iris = np.genfromtxt(inputfile, skip_header=True, delimiter=",")

mlp_init = Multilayerperceptron.MLP(data=iris, input_size=4, epoch=100, learn_rate=0.01, hidden_layer_size=2, output_layer_size=2)
mlp_init.run(100)

