import numpy as np
class Neuron:
    def __init__(self, v0, E0, dt, eps, win_size, energy_coefficient, gamma, avg_coef, reset_v, weights_init, position=(0, 0)):
        self.win_size = win_size
        self.v = v0
        self.E_ = np.zeros(win_size)
        self.E_[0] = E0
        self.output = np.zeros(win_size)
        self.dt = dt
        self.t = 0
        self.eps = eps
        self.alpha_1, self.alpha_2 = energy_coefficient
        self.gamma = gamma
        self.N = 1
        self.reset_v = reset_v
        self.position = position
        self.weights=weights_init
        self.avg_coef = avg_coef
        self.inputs_ = None
        self.outputs_ = None
        
    def update_energy(self, o_hat_mean):
        # need to make it more like an iterative update by multiplying with dt and computing dE so that the overall update looks like a mean over the outputs
        o_mean = self.outputs_
        self.E_[1:] = self.E_[:-1]
        self.E_[0] = self.E_[1] + o_hat_mean**self.alpha_1 - o_mean**self.alpha_2 - self.eps
        return self.E_[0]
    def update_weights(self):
        #dw = self.dt*self.alpha_1*(self.E_[0] - self.E_[1])*inputs
        o_mean = self.outputs_ # its self.spatial_average(self.outputs_) but here we only have one neuron so the spatial average is just mean activity of the neuron
        dw = self.dt*self.alpha_1*self.inputs_*(self.alpha_2*o_mean - self.outputs_) #self.dt*inputs*(o_mean - average_presyn_act)
        self.weights += dw.reshape(self.weights.shape)
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def update_v(self, inputs):
        if self.inputs_ is None:
            self.inputs_ = inputs
        else:
            self.inputs_ = self.inputs_*(1-self.avg_coef) + inputs*self.avg_coef 
        dv = self.weights.T@inputs
        v_ = self.v + dv.squeeze()
        self.v = 1.3*self.gamma*self.sigmoid(v_/self.gamma) # /?/
    
    def activation(self, x):
        return np.where(x>self.gamma, 1, 0)
    def fire(self):
        o = self.activation(self.v)
        self.output[1:] = self.output[:-1]
        self.output[0] =  o
        if self.outputs_ == None:
            self.outputs_ = o
        else:
            self.outputs_ = self.outputs_*(1-self.avg_coef) + o*self.avg_coef
        if o>0: 
            self.v = self.reset_v # /?/: refractory period?
        return o
    
