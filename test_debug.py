import numpy as np
import matplotlib.pyplot as plt
from neuron import Neuron
from network import Network

b1, b2 = (1.5, 2)
beta1, beta2 = (2, 2)
alpha1, alpha2 = (beta1*b1, beta2*b2/(beta1*b1))
params_dict = {'physical_dimensions': 2,
               'p_0': 0.5, 'lbd': 2, 
               'gamma': 1, 'avg_coef': 0.9, 
               'reset_v': -2, 
               'alpha1': 0.1, 'alpha2': 1.5, 
               'dt': 0.1, 'update_eq': "LPL"}
        
net = Network(**params_dict)   
net.init_neurons(number_of_neurons=4)
net.init_input_neurons(num_imputs=2,num_outputs=1)

iter=500
w_sums = [0]*iter
w0_=[0]*iter
for i in range(iter):
    net.update_neurons(inputs=np.random.randint(0, 4, (2,)))
    net.update_weights()
    w_sums[i]=np.sum(net.W)
    w0_[i]=net.W[0,0]
plt.plot(w_sums)
plt.plot(w0_)

