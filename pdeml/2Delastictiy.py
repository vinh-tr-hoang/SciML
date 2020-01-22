"""
Solving 2D elasticity problem using neural network

author: Truong-Vinh Hoang
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from pyDOE import lhs


class PhysicsInformedNN_solid():

    # Initialize the class
    def __init__(self, layers, lb, ub, E=100.0, nu=0.3, f=0.1, u=None):

        self.mu, self.lamda=E/(2.0*(1.0+nu)), E*nu/((1.0+nu)*(1.0-
        2.0*nu))
        self.lb=lb
        self.ub=ub
        self.f=0.1
        self.u=u

        self.layers=layers

        # Initialize NNs
        self.weights, self.biases=self.initialize_NN(layers)

    def initialize_NN(self, layers):
        weights=[]
        biases=[]
        num_layers=len(layers)
        for l in range(0, num_layers-1):
            W=self.xavier_init(size=[layers[l], layers[l+1]])
            b=tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim=size[0]
        out_dim=size[1]
        xavier_stddev=np.sqrt(2./(in_dim+out_dim))*1.
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
        # epsilon[:, 0,1] = 0.5 *(epsilon[:, 0,1] + epsilon[:, 1,0])
        # epsilon[:, 1,0] = 0.5 *(epsilon[:, 0,1] + epsilon[:, 1,0])
        # gradepsilon = g.batch_jacobian ()
        return epsilon

    def net_stress (self, epsilon):
        eps_xx=epsilon[:, 0, 0]
        eps_yy=epsilon[:, 1, 1]
        eps_xy=epsilon[:, 0, 1]*0.5+epsilon[:, 1, 0]*0.5
        # sigma = []
#        sigma_xx = 2*self.mu*eps_xx + self.lamda* (eps_xx + eps_yy)
#        sigma_yy = 2*self.mu*eps_yy + self.lamda* (eps_xx + eps_yy)
#        sigma_yx = 2*self.mu*eps_xy
#        sigmax = tf.concat([sigma_xx,sigma_yx],1)
#        sigmay = tf.concat([sigma_yx,sigma_yy],1)
#        sigma = tf.concat([sigmax, sigmay],1)
#        return sigma
        sigma_xx=2*self.mu*eps_xx+self.lamda*(eps_xx+eps_yy)
        sigma_yy=2*self.mu*eps_yy+self.lamda*(eps_xx+eps_yy)
        sigma_yx=2*self.mu*eps_xy
        # sigma = tf.concat ([sigma_xx, sigma_yy, sigma_yx],1)
        return sigma_xx, sigma_yy, sigma_yx

    def net_loss_funct (self, x):

        with tf.GradientTape() as g:
            X=tf.convert_to_tensor(x)
            g.watch (X)
            sigma_xx, sigma_yy, sigma_xy=self.net_stress(self.net_epsilon(X))
        print (sigma_xx)
        gradsigma_xx=g.jacobian (sigma_xx, X)
        print (gradsigma_xx)
        gradsigma_yy=g.jacobian (sigma_yy, X)
        gradsigma_xy=g.jacobian (sigma_xy, X)
        loss_stress=((gradsigma_xx[:, 0]+gradsigma_xy[:, 1])**2+
                       (gradsigma_xy[:, 0]+gradsigma_yy[:, 1])**2)
        if x==self.lb:
            loss_Dirichlet=(self.net_u (x))**2
        else:
            loss_Dirichlet=0
        if x==self.ub:
            loss_Neuman=(self.net_u (x)-self.f)**2
        else:
            loss_Neuman=0
        return tf.sqrt(loss_stress+loss_Dirichlet+loss_Neuman)

    def elastic_energy (self, x, xD, uD, xN, fN, verbose=False):
        epsilon=self.net_epsilon(x)
        sigma_xx, sigma_yy, sigma_yx=self.net_stress(epsilon)
        eps_xx=epsilon[:, 0, 0]
        eps_yy=epsilon[:, 1, 1]
        eps_xy=epsilon[:, 0, 1]*0.5+epsilon[:, 1, 0]*0.5
        elastic_energy=0.5*(tf.reduce_mean(sigma_xx*eps_xx)+tf.reduce_mean(sigma_yy*eps_yy)+
                            2*tf.reduce_mean(sigma_yx*eps_xy))
        if verbose:
            print("elastic energy "+str(elastic_energy.numpy()))

        UD=self.net_u (xD)
        loss_Dirichlet=(tf.reduce_mean ((uD[:, 0]-UD[:, 0])**2)*100000.+
                          tf.reduce_mean ((uD[:, 1]-UD[:, 1])**2)*100000.)
#        sigmaN_xx, sigmaN_yy, sigmaN_yx = self.net_stress(xN,yN)
#
#        loss_Neuman = (tf.reduce_mean ((sigmaN_xx-fN[:,0])**2) +
#                       tf.reduce_mean ((sigmaN_yy-fN[:,1])**2) )

        UN=self.net_u (xN)
#        eps_xx = epsilon[:,0,0]
#        eps_yy = epsilon[:,1,1]
#        eps_xy = epsilon[:,0,1]
        loss_Neuman=(tf.reduce_sum (fN*UN))
#        if verbose:
#            print("loss_Neuman" + str(loss_Neuman.numpy()))
#            print("loss_Dirichlet" + str(loss_Dirichlet.numpy()))
        return elastic_energy+loss_Dirichlet-loss_Neuman

    def train_step(self, x, xD, uD, xN, fN, opt, verbose=False, learning_rate=0.1):
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
        while loss_old<self.elastic_energy(x, xD, uD, xN, fN).numpy() and learning_rate>1e-5:
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
        return stop_flag
  # tf.optimizer.apply_gradients(zip(grads, self.weights))


if __name__=="__main__":
    options=dict(N=400,
                 dim_input=2,
                 numberneural=[20, 20, 20, 20],
                 dim_output=2
                 )

    dim=2 # dimension of the problem

    # Doman bounds
    lb=0.
    ub=1.
    x=[np.linspace(lb, ub, 100, dtype='float64') for i in range(dim)]
    X=np.meshgrid(*x) # for plotting np.meshgrid(x[0], x[1])
    X_plot=np.hstack([X[i].reshape(X[0].size, 1) for i in range(dim)])
    layers=[2, 20, 20, 2]

    colocation=lhs(n=dim, samples=2000)
#     colocation[:,1]=np.random.rand(2000)
    # Dirichlet boundary conditions
    xDL=np.zeros(shape=(20, 1))
    yDL=np.linspace(0., 1., 20).reshape(xDL.shape)
    xD=np.hstack([xDL, yDL])
    uD=np.zeros ((20, 2))

    # Neumann boundary conditions
    xN=np.hstack([np.ones(shape=(20, 1)), np.linspace(0., 1., 20).reshape((20, 1))])
    fN=np.ones ((20, 2))*30./20.
    fN[:, 1]=0.

    for globalsearch_i in range (10):
        model=PhysicsInformedNN_solid(layers, lb, ub)
        # model.weights = model.weights*0.
        # model.biases = model.biases*0.
#        u = model.net_u(colocation)
#        epsilon = model.net_epsilon(colocation)
#        net_stress = model.net_stress(colocation)

        elastic_energy=model.elastic_energy(colocation, xD, uD, xN, fN, verbose=True)
        lr=1.e-1
        opt=tf.keras.optimizers.Adam(learning_rate=lr)
        for i in range (0, 100):
            # stochastic gradient descent
            fN1=np.zeros ((10, 2))*10.
            ith=np.random.choice(1000, size=50, replace=None)
            x_batch=colocation[ith]
            if i % 1 == 0:
                print ('step ', i)
                stop_flag=model.train_step(colocation, xD, uD, xN, fN, opt, verbose=True, learning_rate=lr)
                if stop_flag and i>=10:
                    break
            else:
                stop_flag=model.train_step(x_batch, xD, uD, xN, fN, opt, verbose=False, learning_rate=lr)

        if globalsearch_i==0:
            lost_old=model.elastic_energy(colocation, xD, uD, xN, fN, verbose=True)
            model_old=model
        if globalsearch_i>1 and model.elastic_energy(colocation, xD, uD, xN, fN, verbose=False)<lost_old:
            model_old=model
            lost_old=model.elastic_energy(colocation, xD, uD, xN, fN, verbose=True)

    model=model_old
    ###########################################################################
    plt.figure("displacement")
    U=model.net_u(X_plot)
    U=U.numpy().reshape(X[0].shape + (dim,))
    for i in range (X[0].shape[0]):
        plt.plot(X[0][:, i], X[1][:, i], '-k')
        plt.plot(X[0][:, i]+U[:, i, 0], X[1][:, i]+U[:, i, 1], '-b')
    for i in range (0, X[0].shape[1]):
        plt.plot(X[0][i, :], X[1][i, :], '-k')
        plt.plot(X[0][i, :]+U[i, :, 0], X[1][i, :]+U[i, :, 1], '-b')
    plt.show()
    # model.train(0)
