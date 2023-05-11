from tkinter.messagebox import NO
import colorama
import numpy as np
from matplotlib import markers, pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as SM
import powerlaw
from env import Anim, custom_oneDEnv
from matplotlib import animation, rc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
class Network:
    def __init__(self, physical_dimensions=3, p_0=1, lbd=2, gamma=1, leak_current=0.1, avg_coef=0.9, avg_sz_coef=0.9, reset_v=-2, alpha1=1, alpha2=1, dt=0.1, update_eq = None, noise_prob=.0):
        self.neurons = []
        self.output_map = []
        self.physical_dimensions = physical_dimensions
        self.M = None
        self.p_0 = p_0
        self.lbd = lbd
        self.outputs = None
        self.W = None
        self.v = None
        self.gamma = gamma
        self.avg_coef = avg_coef
        self.avg_sz_coef = avg_sz_coef
        self.dt = dt
        self.alpha1, self.alpha2 = (alpha1, alpha2)
        self.presyn_ = None # "_" stands for rolling exponential average
        self.postsyn_ = None # same thing? not really cause seperated by one time-step but...
        self.reset_v = reset_v
        self.D = None
        self.dzdt = None
        self.update_eq = update_eq
        self.input_neurons_indices=None
        self.number_of_neurons=None
        self.num_outputs=None
        self.sig_z = None
        self.indices = None
        self.leak_current = leak_current
        self.max_dw = 10
        self.W_threshold = 0.1
        self.raster=[]
        
    def reset(self, w):
        self.W = w
        self.presyn_ = None # "_" stands for rolling exponential average
        self.postsyn_ = None
        self.dzdt = None
        self.sig_z = None
        self.raster=[]
        self.outputs = np.zeros(self.number_of_neurons)
        self.v = np.zeros(self.v.shape)
    def init_neurons(self, number_of_neurons):
        self.number_of_neurons=number_of_neurons
        self.M = np.zeros((number_of_neurons, number_of_neurons))
        self.D = np.zeros_like(self.M)
        n_neurons_side = int(number_of_neurons**(1/self.physical_dimensions))+1 # /!/
        # get index of neuron number i in self.M on each of the self.physical_dimensions dimensions
        self.indices = np.array(np.unravel_index(np.arange(number_of_neurons), (n_neurons_side,)*self.physical_dimensions)).T # /!/
        self.init_connection_matrix()
        self.v = np.zeros(number_of_neurons)
        self.outputs = np.zeros(number_of_neurons)
    
    def init_neurons_from_D(self, D):
        self.number_of_neurons = D.shape[0]
        self.M = np.zeros((self.number_of_neurons, self.number_of_neurons))
        self.D = D
        self.v = np.zeros(self.number_of_neurons)
        self.outputs = np.zeros(self.number_of_neurons)

        self.init_connection_matrix()

        # in the code self.indices is actually the position of the neurons in the physical space
        # so we need to compute it from the distance matrix
        # perform MDS to get aragement of neurons in physical space from distance matrix
        from sklearn.manifold import MDS
        mds = MDS(n_components=self.physical_dimensions, dissimilarity='precomputed')
        self.indices = mds.fit_transform(D)
        
    def init_connection_matrix(self):
        self.M = np.array([[self._init_connection(i, j, self.p_0) for i in range(self.M.shape[0])] for j in range(self.M.shape[0])])  # /!/
        D_norm = np.zeros(self.D.shape)
        for i in range(self.D.shape[1]):
            D_norm[:,i] = self.D[:,i]/np.sum(self.D[:,i])
        self.D = D_norm
    def _init_connection(self, i, j, c):
        posi = self.indices[i]
        posj = self.indices[j]
        d = np.linalg.norm(posi-posj)
        self.D[i, j] = c*np.exp(-d**2/self.lbd**2) if i != j else 0
        return 1 if np.random.rand() < self.D[i, j] else 0 
    def init_io_neurons_ind(self, in_ind, out_ind, weights_init="random"):
        self.input_neurons_indices = in_ind
        self.output_neurons_indices = out_ind

        in_mat = np.zeros([len(in_ind), self.M.shape[0]])
        in_mat[np.arange(len(in_ind)), in_ind] = 1
        self.M = np.concatenate( [in_mat, self.M], axis=0 )
        if weights_init=="random":
            self.W = np.multiply(self.M, np.random.rand(self.M.shape[0], self.M.shape[1]))
        elif weights_init=="ones":
            self.W = np.multiply(self.M, np.ones((self.M.shape[0], self.M.shape[1])))
        elif weights_init=="inh/ex ones":
            ones_or_minus_ones = np.random.choice([-1,1], size=(self.M.shape[0], self.M.shape[1]))
            self.W = np.multiply(self.M, ones_or_minus_ones)
        else:
            raise NotImplementedError

    def init_io_neurons(self, num_imputs,num_outputs,meth="random"):
        print("initialize inputs and outputs neurons")
        assert num_imputs < self.number_of_neurons, "Number of inputs exceeds the number of neurons"
        if num_imputs>0 :
            if meth=="first_row":
                self.input_neurons_indices = np.random.choice([i for i in range(0,self.number_of_neurons) if self.indices[i][0] ==0]\
                ,num_imputs)
            else: # for now random on first column and same for output for 2D
                self.input_neurons_indices = np.random.randint(0, self.M.shape[0], num_imputs) # random here but we might want to choose them in a certain way


            in_mat = np.zeros([num_imputs, self.M.shape[0]])
            in_mat[np.arange(num_imputs), self.input_neurons_indices] = 1
            self.M = np.concatenate( [in_mat, self.M], axis=0 )
        
        assert num_outputs< self.number_of_neurons-num_imputs, "Number of outputs exceeds the number of remaining neurons"
        if num_outputs >0 :
                self.output_neurons_indices= np.random.choice([i for i in range(0,self.number_of_neurons) if i not in self.input_neurons_indices]\
                ,num_outputs)

        self.W = np.multiply(self.M,np.random.rand(self.M.shape[0], self.M.shape[1])) # random initialisation of weights, could be different for each neuron, especially for the input neurons
    def get_output_neurons_indices(self):
        return self.output_neurons_indices
    
    def get_input_neurons_indices(self):
        return self.input_neurons_indices

    def update_neurons(self, inputs=np.array([])):
        if self.update_eq=="REQ_DIFF":
            true_act = np.concatenate( [inputs, self.outputs], axis=0 )
            if self.presyn_ is None:
                self.presyn_ = true_act
            else: 
                self.presyn_ = self.presyn_*(1-self.avg_coef) + true_act*self.avg_coef
            I = np.dot(self.W.T*self.M.T, true_act)
            # get the lower diagonal weights of W that don't go to input neurons
            dia = self.W[np.arange(len(self.input_neurons_indices), self.W.shape[0]), np.arange(self.W.shape[1])]
            R = np.array(dia)*self.outputs
            dv = (I-R)**2 - self.leak_current*self.v
            self.v += dv.squeeze()*self.dt
            o = self.fire()
            self.v = np.where(self.v>self.gamma, self.reset_v, self.v)
            ouput_activations=o[self.output_neurons_indices]
            self.update_gamma()
            return self.read_out(ouput_activations)
        else:
            true_act = np.concatenate( [inputs, self.outputs], axis=0 )
            if self.presyn_ is None:
                self.presyn_ = true_act
            else: 
                self.presyn_ = self.presyn_*(1-self.avg_coef) + true_act*self.avg_coef
            dv = np.dot(self.W.T, true_act) - self.leak_current*self.v
            v_ = np.dot(self.W.T, self.presyn_)
            
            #self.dzdt = self.activation_derivative(v_)/self.dt # /?/ should we not multiply by dv?
            #self.v = 1.3*self.gamma*self.sigmoid(self.v + dv.squeeze()/self.gamma) # /?/
            self.v = self.v + dv.squeeze()
            o = self.fire()
            self.v = np.where(self.v>self.gamma, self.reset_v, self.v) # /?/: refractory period?
            ouput_activations=o[self.output_neurons_indices]
            return self.read_out(ouput_activations)

    def update_equation(self):
        self.presyn_ = np.where(self.presyn_>1e-3, self.presyn_, 1e-3)
        if self.update_eq == "PA" or self.update_eq == None:
            o_mean = self.spatial_average()
            inputs_ = np.repeat(self.presyn_.reshape([self.W.shape[0]] + [1]), self.W.shape[1], axis=1)
 
            input_mat = np.multiply(self.W, inputs_)
            dw = self.dt*self.alpha1* np.multiply(input_mat, \
                np.repeat((self.alpha2*o_mean - self.postsyn_)\
                    .reshape([1, self.W.shape[1]]), \
                repeats=self.M.shape[0], axis=0)) # /?/ no derivative of the activation function?
        elif self.update_eq == "REQ_DIFF":
            I_hat = np.dot(self.W.T*self.M.T, self.presyn_) # can do np.log(self.presyn_+1e-3) as well but it can diverge, be careful
            dia = self.W[np.arange(len(self.input_neurons_indices), self.W.shape[0]), np.arange(self.W.shape[1])]
            R_hat = np.array(dia)* self.postsyn_#np.log(self.postsyn_+1e-3)
            dw = self.dt*self.alpha1*np.multiply(self.W, (I_hat-R_hat))
            #dEdw = 2*self.presyn_*(I_hat-R_hat)/(self.sig_z**2 + 1e-12) 
            
            E = (I_hat-R_hat)#/(self.sig_z**2 + 1e-12)
            E_mat       = np.repeat(           E.reshape([self.W.shape[1]] + [1]), repeats=self.W.shape[0], axis=0).reshape(self.W.shape)
            inputs_     = np.repeat(self.presyn_.reshape([self.W.shape[0]] + [1]), repeats=self.W.shape[1], axis=1).reshape(self.W.shape)
            input_mat   = np.multiply(self.M, inputs_)
            dw = -self.dt*self.alpha1*2*np.multiply(input_mat, E_mat)
            dw[np.arange(len(self.input_neurons_indices), self.W.shape[0]), np.arange(self.W.shape[1])] *= -1
        else :
            f_prime = self.activation_derivative(np.dot(self.W.T, self.presyn_))
            f_prime_mat = np.repeat(f_prime     .reshape([self.W.shape[1]] + [1]), repeats=self.W.shape[0], axis=0).reshape(self.W.shape)
            inputs_     = np.repeat(self.presyn_.reshape([self.W.shape[0]] + [1]), repeats=self.W.shape[1], axis=1).reshape(self.W.shape)
            input_mat   = np.multiply(self.M, inputs_)
            # actual formula is dW_j/dt = eta*x_j(t)*f'(a(t))*(-dz/dt + (lbda/sigma^2)*(z(t)-z_bar(t)) )
            BCM = ( self.postsyn_ - self.alpha2*self.spatial_average() )# / (self.sig_z**2 + 1e-12) 
            if self.update_eq == "LPL" :
                LPL = np.repeat((BCM - self.dzdt).reshape([1, self.W.shape[1]]), repeats=self.M.shape[0], axis=0)
            elif self.update_eq == "BCM":
                LPL = np.repeat( BCM             .reshape([1, self.W.shape[1]]), repeats=self.M.shape[0], axis=0)
            dw = self.dt * self.alpha1 * input_mat * f_prime_mat * LPL

        return np.clip(dw, -self.max_dw, self.max_dw) # clip so that the weights do not go out of bounds
    def update_gamma(self):
        if np.mean(self.postsyn_)<1e-3*self.gamma:
            self.gamma = self.gamma*0.9
        elif np.mean(self.postsyn_)>2e-3*self.gamma:
            self.gamma = self.gamma*1.1
    def update_weights(self):
        dw = self.update_equation()
        self.W += dw.reshape(self.W.shape)
        #self.W = np.clip(self.W, -1, 1)
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def spatial_average(self):
        return self.postsyn_.T @ self.D
    def activation(self, x):
        return np.where(x>self.gamma, 1.0, 0.0)
    def activation_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x)) # derivative of the sigmoid
    def fire(self, noise=.0):
        o = self.activation(self.v)
        #o[np.random.rand(o.shape[0])<self.noise_prob] = 1
        self.outputs = o
        self.raster.append(self.outputs)
        if self.postsyn_ is None:
            self.postsyn_ = o
            self.dzdt = o # case where postsyn_(t)=0
        else:
            self.dzdt = -self.postsyn_
            self.postsyn_ = self.postsyn_*(1-self.avg_coef) + o*self.avg_coef
            self.dzdt += self.postsyn_ # so we have postsyn_(t+1)-postsyn_(t)
        #o = np.where(np.random.rand(self.M.shape[0])<o, 1, o)
        if self.sig_z is None:
            self.sig_z = (o-self.postsyn_)**2
        else:
            self.sig_z = self.sig_z*(1-self.avg_sz_coef) + (o-self.postsyn_)**2*self.avg_sz_coef # same average coef?
        return o
    def compute_entropy(self, window_length):
        """
        Computes the entropy of a window of the raster plot, which is a binary matrix of size (n_neurons, window_length)
        """
        if len(self.raster)<window_length:
            return 0
        else:
            # compute the probability of firing for each neuron
            p = np.mean(self.raster[-window_length:], axis=1)
            # compute the entropy
            return -np.sum(p*np.log2(p+1e-12))
    def bin_raster_plot(self, window_length):
        """
        Bins the raster plot into a binary matrix of size (n_neurons, window_length)
        """
        num_time_windows = self.raster.shape[0] // window_length
        binned_raster_plot = np.zeros((num_time_windows, self.raster.shape[1]))
        for i in range(num_time_windows):
            binned_raster_plot[i] = np.sum(self.raster[i*window_length:(i+1)*window_length], axis=0)
        return binned_raster_plot
    
    def detect_avalanches(self, window_length):
        """
        Detects avalanches in the binned raster plot
        """
        binned_raster = self.bin_raster_plot(window_length)
        avalanches = []
        current_avalanche = []
        for time_window in binned_raster:
            if np.any(time_window):
                current_avalanche.append(time_window)
            elif current_avalanche:
                avalanches.append(np.array(current_avalanche))
                current_avalanche = []
        return avalanches
    def compute_avalanche_sizes(self, avalanches):
        sizes = []
        for avalanche in avalanches:
            size = np.sum(avalanche)
            sizes.append(size)
        return sizes
    
    def fit_power_law(self, window_length):
        """
        Fits a power law to the avalanche size distribution
        """
        
        avalanches = self.detect_avalanches(window_length)
        sizes = self.compute_avalanche_sizes(avalanches)
        # fit the power law
        powerlaw = powerlaw.Fit(sizes)
        # return exponent and Kolmogorov-Smirnov distance (badness of the fit)
        return powerlaw.alpha, powerlaw.D
    def visualize_network(self):
        """
        Visualize the network from the connectivity matrix self.W, the neurons mean activations self.postsyn_ and the neurons positions in self.indices
        """
        # First create a figure and put some dot at the right positions
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # plot the lines
        for i in range(len(self.input_neurons_indices), self.M.shape[0]): # number of neurons self.number of neurons?
            for j in range(self.M.shape[1]):
                if self.M[i,j] == 1:
                    # set the color of the line based on the strength of the connection from self.W
                    cl = self.W[i,j]/np.max(np.abs(self.W))
                    i_ = i - len(self.input_neurons_indices)
                    plt.plot([self.indices[i_,0], self.indices[j,0]], [self.indices[i_,1], self.indices[j,1]],'-', c=plt.cm.jet(cl), zorder=1)
        for i,idx in enumerate(self.input_neurons_indices):
            cl_=self.W[i,idx]/np.max(np.abs(self.W))
            plt.scatter([self.indices[idx,0]], [self.indices[idx,1]], linewidths=25, color=plt.cm.jet(cl_), zorder=2, alpha=0.5)
            #plt.plot([self.indices[idx,0]-1, self.indices[idx,0]], [self.indices[idx,1], self.indices[idx,1]],'-', c=plt.cm.jet(cl_), zorder=2)

        cs = self.postsyn_/np.max(np.abs(self.postsyn_)) if np.max(np.abs(self.postsyn_))>0 else self.postsyn_ # grey scale
        # have a minimum value of 0.1 to see the inactive neurons
        cs = np.where(abs(cs)>0.1, cs, 0.1)
        cluster=np.zeros((self.number_of_neurons,1))
        cluster[self.input_neurons_indices]=1
        cluster[self.output_neurons_indices]=2
        Posx,Posy=self.indices[:,0],self.indices[:,1]

        plt.scatter(Posx[cluster[:,0]==0], Posy[cluster[:,0]==0], s=100, c=cs[cluster[:,0]==0], norm=Normalize(-1,1),cmap=plt.cm.jet, zorder=3, alpha=0.5)
        plt.scatter(Posx[cluster[:,0]==1], Posy[cluster[:,0]==1],marker="^", s=300, c=cs[cluster[:,0]==1], norm=Normalize(-1,1), cmap=plt.cm.jet, zorder=1, alpha=1)
        plt.scatter(Posx[cluster[:,0]==2], Posy[cluster[:,0]==2],marker="*",s=300, c=cs[cluster[:,0]==2], norm=Normalize(-1,1),cmap=plt.cm.jet, zorder=1, alpha=1)
        
        plt.colorbar(SM(Normalize(np.min(self.W),np.max(self.W)), plt.cm.jet))
        plt.subplot(122)
        # plot a heatmap of the spatial average at each neuron position
        # get the vector of colors for each neuron
        sa = self.spatial_average()
        cs = sa/np.max(np.abs(sa)) if np.max(np.abs(sa))>0 else sa
        # create the matrix that assigns the color to the neuron's positions
        n,k = self.indices[:,0].max(), self.indices[:,1].max()
        H = np.zeros((n+1,k+1))
        for k in range(len(cs)):
            H[self.indices[k,0], self.indices[k,1]] = cs[k]
        
        plt.imshow(H, cmap=plt.cm.jet, interpolation='nearest', origin = 'lower')
        plt.colorbar(SM(Normalize(np.min(sa),np.max(sa)), plt.cm.jet))

        plt.show()
    def raster_plot(self):
        x,y=([],[])
        for i,col in enumerate(self.raster):
            i_=np.arange(self.number_of_neurons)
            y_=list(i_[col!=0])
            x_=[i*self.dt]*len(y_)
            x+=x_
            y+=y_
            
        x += [0, (i+1)*self.dt]
        y += [0, 0]
        return x, y
        
        pass
    def read_out(self,o_activation):
        return o_activation
    def update_connectivity(self):
        self.M = np.where(np.abs(self.W)>self.W_threshold, 1, 0)
        self.W = np.where(np.abs(self.W)>self.W_threshold, self.W, 0)
    def dinstance_matrix_loss(self, D):
        # enforces consistency of the distance matrix
        # symmetry loss: \sum_{i,j} (D_{ij} - D_{ji})^2
        L_sym = np.sum(np.square(D - D.T))
        # triangle inequality loss: \sum_{i,j,k} max(0, D_{ij} + D_{jk} - D_{ik})
        L_tri = 0
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                for k in range(D.shape[0]):
                    L_tri += np.max([0, D[i,j] + D[j,k] - D[i,k]])**2
        # zero diagonal loss: \sum_{i} D_{ii}^2
        L_diag = np.sum(np.square(np.diag(D)))
        # non-negativity loss: \sum_{i,j} max(0, -D_{ij})
        L_nn = np.sum(np.max([0, -D]))
        return L_sym + L_tri + L_diag + L_nn
    def kill_birth_neurons(self):
        pass   
    def navigate_neurons(self):
        pass
    def set_params(self, params_dict):
        """
        change the attribute of the network class (e.g. self.gamma = params_dict['gamma'])
        """
        for key in params_dict.keys():
            setattr(self, key, params_dict[key])
    
    
def write_in(scalar_normalized, s2): # gets spike train according to normalized scalar observation between -1 and 1
        # s1: 
        obs_pos = int(1/(1*scalar_normalized)) if scalar_normalized>0 else 0 # mean spiking period for positive values
        obs_neg = -int(1/(1*scalar_normalized)) if scalar_normalized<0 else 0 # mean spiking period for negative values
        obs_neg_v = np.array([[int(i%obs_neg==0)] for i in range(s2)]) if obs_neg!=0 else np.zeros((s2, 1)) # spike train for negative values
        obs_pos_v = np.array([[int(i%obs_pos==0)] for i in range(s2)]) if obs_pos!=0 else np.zeros((s2, 1)) # spike train for positive valuess
        return np.concatenate([obs_neg_v, obs_pos_v], axis=1)

def run(iterations, env, param_dict, s=10, number_of_neurons=100, initial_network=None, connection_mat=None):
    # s is the scaling factor between environment dynamics and network dynamics, for instance, s=10 means that for every environment step, the network is stepped 10 times
    # the observation are static over the network steps so they are encoded into a spike train of length s
    # a bigger s means that the network has more representational capacity but it is slower to simulate

    # pick random initial position 
    env.set_pos(np.random.rand()*2-1)

    # initialize network with given parameters
    if initial_network is None:
        net = Network(**param_dict)
        net.init_neurons(number_of_neurons=number_of_neurons)
        n_inputs = 3*2+4
        net.init_io_neurons(num_imputs=n_inputs,num_outputs=2) # Input: speed, acceleration, obs --> two neurons for each; Output: acceleration --W two neurons
        if connection_mat is None:
            net.M = connection_mat
            net.W = net.W * net.M
        for j in range(10): # initial weight convergence
            net.update_neurons(inputs=np.array([1]*n_inputs))
            net.update_weights()
    else:
        net = initial_network
        net.raster = []
    
    action= 0.0
    frames = []
    sum_activations = []
    entropies = []
    num_activations = []
    ys = []
    for i in range(iterations):
        inputs = env.step(action) # obs, vel, acc
        obs,vel,acc=inputs
        in_obs = write_in(obs, s)
        in_vel = write_in(vel, s)
        in_acc = write_in(acc, s)
        output_image1 = write_in(2*net.postsyn_[net.output_neurons_indices[0]]-1, s)
        output_image2 = write_in(2*net.postsyn_[net.output_neurons_indices[1]]-1, s)
        
        
        inputs_ = np.concatenate((in_obs, in_vel, in_acc, output_image1, output_image2), axis=1)
        acs = 0
        o = np.zeros(2)
        for j in range(s):
            o += net.update_neurons(inputs=inputs_[j])
            net.update_weights()
            acs += np.sum(net.postsyn_)
        sum_activations.append(acs/s)
        o = (net.postsyn_[net.output_neurons_indices[0]], net.postsyn_[net.output_neurons_indices[1]])
        num_activations.append(np.sum(net.raster[-s:], axis=0))
        #action = (net.postsyn_[net.output_neurons_indices[0]]-net.postsyn_[net.output_neurons_indices[1]])
        action = (o[0]-o[1])/(o[0]+o[1]+1e-6)
        entropies.append(net.compute_entropy(s))
        if i%100==0:
            print("Iteration: {}\t Action: {}".format(i, action))
        y = [obs, vel, acc, action] # inputs to be reconstructed by the linear regression
        ys.append(y)
        frames.append(env.get_frame())
    return frames, sum_activations, net, entropies, np.array(num_activations)/s, np.array(ys)

def create_animation(frames, filename='plots/animation2.gif'):
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    def init_fr():
        return anim.init_frame(frames[0], ax)
    def animate(n):
        return anim.create_frame(frames[n])
    rc('animation', html='html5')
    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                interval=100, blit=True, init_func=init_fr)
    ani.save(filename, writer='imagemagick', fps=60)


def evolve_for_criticality(self, epochs, device='cpu'):
    """
    Optimize physioligical parameters, including network hyperparameters and distance matrix D for criticality
    """
    param_dict = {'physical_dimensions': 2, 
                  'p_0': 1, 'lbd': 2, 
                  'gamma': 10, 'avg_coef': 0.4, 
                  'avg_sz_coef': 0.6, 'reset_v': -1, 
                  'alpha1': 0.5, 'alpha2': 0.5, 
                  'dt': 0.01, 'update_eq': 'LPL', 
                  'leak_current': 0.001}

    phys_params = ['p_0', # base probability of connection p_0*exp(-d/lbd)
                   'lbd', # length constant of the exponential decay of the connection probability
                   'gamma', # spike threshold
                    'avg_coef', # determine the time constant of the rolling average of network activity (a = a*avg_coef + (1-avg_coef)*a_t), equivalent time constant is 1/(1-avg_coef)
                    'avg_sz_coef', # determine the time constant of the rolling average of network std factor in LPL
                    'alpha_1', 'alpha_2', # energy coefficients, determines the relative importance of the two energy terms and thus the individuality of neurons
                    'leak_current'] # leak current
    number_of_neurons = 100
    # initialize network with given parameters
    net = Network(**param_dict) # spiking neural network, NOT optimizable directly
    net.init_neurons(number_of_neurons=number_of_neurons)
    in_neurons_ind = np.arange(0, 3*2+4)
    out_neurons_ind = np.arange(3*2+4, 3*2+4+2)
    net.init_io_neurons_ind(in_neurons_ind, out_neurons_ind)
    # initialize distance matrix D
    D = torch.rand((number_of_neurons, number_of_neurons))*10 # random distance matrix
    D = (D+D.T)/2 # make it symmetric
    D = D - torch.diag(torch.diag(D)) # set diagonal to zero
    D = D + torch.diag(torch.ones(number_of_neurons)*1e-6) # add small constant to diagonal to avoid division by zero
    D = D.to(device)
    D.requires_grad = True
    # optimize so that we meet triangle inequality defined as D_ij <= D_ik + D_kj for all i,j,k
    optimizer = torch.optim.Adam([D], lr=0.1)
    while True:
        optimizer.zero_grad()
        # sum over all i,j,k, relu as max(0, D_ij - D_ik - D_kj)
        loss = torch.sum(torch.relu(D[:, :, None] - D[:, None, :] - D[None, :, :]))
        loss.backward()
        optimizer.step()
        if loss < 1e-3:
            break
    
    # initialize RL agent outputing network parameters and distance matrix to maximize the network's criticality metrics
    class Agent(nn.Module):
        def __init__(self, phys_params, D_init, net_params):
            super(Agent, self).__init__()
            self.phys_params = phys_params
            self.neuralnet = nn.Sequential(
                nn.Linear(net_params['noise_dim'], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            self.D_head = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, number_of_neurons**2),
                nn.ReLU() # distance matrix is positive
            )
            self.params_head = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(phys_params)),
                nn.ReLU()
            )
            self.critic_head = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.ReLU()
            )
        def forward(self, noise):
            x = self.neuralnet(noise)
            D = self.D_head(x)
            D = D.view(number_of_neurons, number_of_neurons)
            params = self.params_head(x)
            params_dict = {}
            for i, param in enumerate(self.phys_params):
                params_dict[param] = params[:, i]
            return D, params_dict
    

    ks_coef = 2 # how much to weight the KS distance in the loss function
    agent = Agent(phys_params, D, param_dict).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    # tqdm is a progress bar
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        noise = torch.randn((1, param_dict['noise_dim'])).to(device)
        D, params = agent(noise)
        net.set_params(params)
        net.init_neurons_from_D(D)
        net.reset(w=net.W)
        frames, sum_activations, net, entropies, num_activations, ys = net.run(1000, 1000)
        # optimize for criticality, get reward from exponent close to 1 and fit quality of power law fit
        pow_exp, ks_dist = net.powerlaw_fit(num_activations)
        power_law_loss = F.mse_loss(pow_exp, 1) + ks_coef*ks_dist
        
        reward = torch.exp(-power_law_loss) # the lower the loss, the higher the reward
        
        # train actor and critic
        actor_loss = -agent.critic_head(torch.mean(sum_activations, dim=0)).mean()
        critic_loss = F.mse_loss(agent.critic_head(torch.mean(sum_activations, dim=0)), reward)
        loss = actor_loss + critic_loss
        loss.backward()
        optimizer.step()
        
        # print some metrics
        if epoch % 10 == 0:
            print('epoch: ', epoch, 'reward: ', reward.item(), 'power law loss: ', power_law_loss.item(), 'actor loss: ', actor_loss.item(), 'critic loss: ', critic_loss.item())
            print('power law exponent: ', pow_exp.item(), 'KS distance: ', ks_dist.item())
            print('mean sum activation: ', torch.mean(sum_activations).item(), 'mean entropy: ', torch.mean(entropies).item())
            print('mean distance: ', torch.mean(D).item(), 'mean weight: ', torch.mean(net.W).item())
            # visualize the network
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(net.W) # visualize the weight matrix
            plt.subplot(2, 2, 2)
            net.plot_activity(frames, ys) # visualize the activity
            plt.subplot(2, 2, 3)
            net.plot_entropy(entropies) # visualize the entropy
            
            plt.show()
            net.visualize_network()

    

    
    
# ----------------------------------------------------------------------------------------------------------------------
# To do:
# - Explicitely define the spatial average, for instance: should it be normalized? 
#            should the neuron's acticity be included in the spatial average at that neuron?
# - A way to measure complexity of the network dynamics (e.g. Lyapunov exponent) should be added
#           This can then become a metric to optimize and track: 
#               complex behaviour can only emerge at the edge of chaos
#               (related to: complex behaviour can only happen at boundaries of degenerate entropic states)
# - Create an "available energy" functional based on the spatial average activation,
#           this energy should be compatible with the LPL paper's formulation and should give an interpretable metric of the network's state
#           We should also have a per-neuron energy that depends on the activity of the neuron and that can be compared with the available energy from the spatial average
# - Define and add a function to create and delete synapses based on the local group energy and neuron activity energy
# - Neurons should be able to die if they have no connections, or if the local energy is too low
# - Neurons should be able to be created in certain conditions (e.g. if the local energy is high enough)
# - Neurons should be able to change position based on the local energy
# - The local energy should include a repulsive factor so that neurons don't collapse into the same position
# - Explicitely define our duality between "mean space" and "instantaneous space" in which our functions live