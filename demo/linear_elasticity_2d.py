import numpy as np
import tensorflow as tf
# from tensorflow import keras
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from pyDOE import lhs
from pdeml.mechanics import LinearElasticity

options=dict(N=400,
             dim_input=2,
             numberneural=[20, 20, 20, 20],
             dim_output=2
             )

dim=2 # dimension of the problem

# Domain bounds
lb=0.
ub=1.
x=[np.linspace(lb, ub, 100, dtype='float64') for i in range(dim)]
X=np.meshgrid(*x) # for plotting np.meshgrid(x[0], x[1])
X_plot=np.hstack([X[i].reshape(X[0].size, 1) for i in range(dim)])
nbnr = 5
layers=[2, nbnr, nbnr, 2]


# Dirichlet boundary conditions
xdl = np.zeros(shape=(20, 1))
xdr = np.ones(shape=(20, 1))
xd = np.concatenate((xdl, xdr))
ydl = np.linspace(0., 1., 20).reshape(xdl.shape)
yd = np.concatenate((ydl, ydl))
xD=np.hstack([xd, yd])
uD=np.zeros ((40, 2))
uD[20:,0] = 0.03

# Neumann boundary conditions (no applied forces)
xN=np.hstack([np.ones(shape=(1, 1)), np.linspace(0., 1., 1).reshape((1, 1))])
fN=np.zeros ((1, 2))
#fN[:, 1]=0.
batchsize = 10*(layers[1]*(layers[0]*2 + 1) + layers[2]*(layers[1]*2 + 1) + layers[3]*(layers[2]*2 + 1))
nbcolopoint = batchsize*20
colocation=lhs(n=dim, samples=nbcolopoint)

epoch = 1000
for globalsearch_i in range (1):
    model=LinearElasticity(dim, layers, lb, ub)
    model.initialize_NN(model.layers, sigma = 0.3)
    elastic_energy=model.elastic_energy(colocation, xD, uD, xN, fN, verbose=True)
    lr=5.e-2
    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.stochastic_gradient_descent (colocation, xD, uD, xN, fN, opt, epoch, batchsize, verbose=True, learning_rate= lr)
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