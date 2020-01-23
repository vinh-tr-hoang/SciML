"""
Solving 2D elasticity problem using neural network

author: Truong-Vinh Hoang
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class LinearElasticity():

    # Initialize the class
    def __init__(self, dim, layers, lb, ub, E=100.0, nu=0.3, f=0.1, u=None):

        self.mu, self.lamda=E/(2.0*(1.0+nu)), E*nu/((1.0+nu)*(1.0-
        2.0*nu))
        self.lb=lb
        self.ub=ub
        self.f=0.1
        self.u=u

        self.layers=layers

        # Initialize NNs
        self.weights, self.biases=self.initialize_NN(layers)

        if dim==1:
            self.net_stress=self._net_stress_1d
            self.strain_mandel=self._strain_mandel_1d
        elif dim==2:
            self.net_stress=self._net_stress_2d
            self.strain_mandel=self._strain_mandel_2d
        elif dim==3:
            self.net_stress=self._net_stress_3d
            self.strain_mandel=self._strain_mandel_3d
        else:
            raise ValueError(dim)

    def initialize_NN(self, layers, sigma = 1.):
        weights=[]
        biases=[]
        num_layers=len(layers)
        for l in range(0, num_layers-1):
            W=self.xavier_init(size=[layers[l], layers[l+1]], sigma = sigma)
            b=tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size, sigma):
        in_dim=size[0]
        out_dim=size[1]
        xavier_stddev=np.sqrt(2./(in_dim+out_dim))*sigma
        dist=tfp.distributions.Normal(loc=0., scale=xavier_stddev)

        return tf.Variable(dist.sample([in_dim, out_dim]), dtype=tf.float64)

    def neural_net(self, X, weights, biases):
        num_layers=len(weights)+1

        H=X # 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W=weights[l]
            b=biases[l]
            H=keras.activations.relu(tf.add(tf.matmul(H, W), b)) # here is the tanh activation function
        W=weights[-1]
        b=biases[-1]
        Y=tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x):
        u=self.neural_net(tf.convert_to_tensor(x), self.weights, self.biases)
        return u

    def net_epsilon(self, x):
        with tf.GradientTape() as g:
            X=tf.convert_to_tensor(x)
            g.watch (X)
            u=self.neural_net(X, self.weights, self.biases)
        epsilon=g.batch_jacobian (u, X)
        return epsilon

    def _strain_mandel_1d(self, epsilon):
        return epsilon

    def _strain_mandel_2d(self, epsilon):
        return tf.stack([epsilon[:, 0, 0], epsilon[:, 1, 1],
                         (epsilon[:, 0, 1]+epsilon[:, 1, 0])/0.5**2], 1)

    def _strain_mandel_3d(self, epsilon):
        return tf.stack([epsilon[:, 0, 0], epsilon[:, 1, 1], epsilon[:, 2, 2],
                         (epsilon[:, 1, 2]+epsilon[:, 2, 1])/0.5**2,
                         (epsilon[:, 0, 2]+epsilon[:, 2, 0])/0.5**2,
                         (epsilon[:, 0, 1]+epsilon[:, 1, 0])/0.5**2], 1)

    def _net_stress_1d(self, epsilon):
        return (3*self.lamda+2*self.mu)*epsilon

    def _net_stress_2d(self, epsilon):
        eps=self._strain_mandel_2d(epsilon) # Mandels notation
        sigma_xx=2*self.mu*eps[:,0]+self.lamda*(eps[:,0]+eps[:,1])
        sigma_yy=2*self.mu*eps[:,1]+self.lamda*(eps[:,0]+eps[:,1])
        sigma_yx=2*self.mu*eps[:,2]
        sigma=tf.stack([sigma_xx, sigma_yy, sigma_yx], 1)
        return sigma

    def _net_stress_3d(self, epsilon):
        eps=self._strain_mandel_2d(epsilon) # Mandels notation
        trace_eps=eps[:,0]+eps[:,1]+eps[:,2]
        sigma=tf.stack([2*self.mu*eps[:,0]+self.lamda*trace_eps,
                        2*self.mu*eps[:,1]+self.lamda*trace_eps,
                        2*self.mu*eps[:,2]+self.lamda*trace_eps,
                        2*self.mu*eps[:,3], 2*self.mu*eps[:,4], 2*self.mu*eps[:,5]], 1)
        return sigma

    def elastic_energy(self, x, xD, uD, xN, fN, verbose=False):#Should rename to potential energy
        epsilon=self.net_epsilon(x)
        strain=self.strain_mandel(epsilon)
        sigma=self.net_stress(epsilon)
#         elastic_energy=0.5*(tf.reduce_mean(sigma_xx*eps_xx)+tf.reduce_mean(sigma_yy*eps_yy)+
#                             2*tf.reduce_mean(sigma_yx*eps_xy))
        elastic_energy=0.5*tf.reduce_mean(sigma*strain)
        if verbose:
            print("elastic energy "+str(elastic_energy.numpy()))

        UD=self.net_u (xD)
        loss_Dirichlet=tf.reduce_mean((uD-UD)**2)*100000.
#         loss_Dirichlet=(tf.reduce_mean ((uD[:, 0]-UD[:, 0])**2)*100000.+
#                           tf.reduce_mean ((uD[:, 1]-UD[:, 1])**2)*100000.)
#
        UN=self.net_u (xN)
        loss_Neuman=(tf.reduce_sum (fN*UN))
#        if verbose:
#            print("loss_Neuman" + str(loss_Neuman.numpy()))
#            print("loss_Dirichlet" + str(loss_Dirichlet.numpy()))
        return elastic_energy+loss_Dirichlet-loss_Neuman

    def train_step(self, x, xD, uD, xN, fN, opt, verbose=False, learning_rate=0.1, min_learning_rate = 1e-8):
        stop_flag=0.
        weights_old=[self.weights[i].read_value() for i in range (len (self.weights))]
        biases_old=[self.biases[i].read_value() for i in range (len (self.biases))]
        with tf.GradientTape() as tape:
            tape.watch (self.weights)
            loss_value=self.elastic_energy(x, xD, uD, xN, fN, verbose=verbose)
        grads=tape.gradient(loss_value, self.weights)
#        with tf.GradientTape() as tapebiases:
#            tapebiases.watch (self.biases)
#            loss_value = self.elastic_energy(x,y,xD,yD, uD, xN, yN, fN, verbose = verbose)
#        gradbiases = tapebiases.gradient(loss_value,self.biases)
        loss_old=loss_value.numpy()

        # self.weights = self.weights - grads*lr
        opt.apply_gradients(zip(grads, self.weights))
        # opt.apply_gradients(zip(gradbiases, self.biases))
        i=1
        if verbose:
            print ('loss value old '+str(loss_old))
            print ('loss_value before first updated (line search)'+str(self.elastic_energy(x, xD, uD, xN, fN).numpy()))

        # implement tree search
        while loss_old<self.elastic_energy(x, xD, uD, xN, fN).numpy() and learning_rate > min_learning_rate:
            i=i+1
            learning_rate=learning_rate/2.
            if verbose:
                print ('learning rate', learning_rate)
            for i in range (len (self.weights)):
                self.weights[i].assign(weights_old[i])
                self.biases[i].assign(biases_old[i])
            opt=tf.keras.optimizers.Adam(learning_rate=learning_rate)
            opt.apply_gradients(zip(grads, self.weights))
            # opt.apply_gradients(zip(gradbiases, self.biases))
        if loss_old<self.elastic_energy(x, xD, uD, xN, fN).numpy():
            stop_flag=1.
            print ('stop')
            for i in range (len (self.weights)):
                self.weights[i].assign(weights_old[i])
                self.biases[i].assign(biases_old[i])
        if verbose:
            print ('loss_value last updated (line search)'+str(self.elastic_energy(x, xD, uD, xN, fN).numpy()))
        self.lr = learning_rate
        return stop_flag
    def stochastic_gradient_descent (self, x, xD, uD, xN, fN, opt, epoch, batchsize, verbose=False, learning_rate=0.1, min_learning_rate = 1e-8):
        nbcolopoint = x.shape[0]
        for i in range (0, epoch):
            ith=np.random.choice(nbcolopoint, size = batchsize, replace=None)
            x_batch=x[ith]
            print ('step ', i)
            if i ==0:
                learning_rate = learning_rate
            else:
                learning_rate = min (self.lr*4, 1.) # addaptive learning rate 
            stop_flag=self.train_step(x_batch, xD, uD, xN, fN, opt, verbose=verbose, learning_rate=learning_rate, min_learning_rate = min_learning_rate)
            if stop_flag: #and i>=10:
                print ('early stoping because learning rate is smaller than minimum value of ' + str(min_learning_rate) )
                break


        
        
        
        # tf.optimizer.apply_gradients(zip(grads, self.weights))


if __name__=="__main__":
    pass
