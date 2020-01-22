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
    model=LinearElasticity(layers, lb, ub)
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