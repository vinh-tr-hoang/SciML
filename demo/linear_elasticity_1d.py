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

dim=1 # dimension of the problem

# Domain bounds
lb=0.
ub=1.
X_plot=np.atleast_2d(np.linspace(lb, ub, 100, dtype='float64')).T
layers=[dim, 20, 20, dim]
colocation=lhs(n=dim, samples=2000)

# Dirichlet boundary conditions
xD=np.array([[0.],[1.]], dtype=np.float64)
uD=np.array([[0.],[0.1]], dtype=np.float64)

# Neumann boundary conditions
xN=np.array([[0.]])
fN=np.zeros_like(xN)

for globalsearch_i in range (1):
    model=LinearElasticity(dim, layers, lb, ub)
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
# U=U.numpy().reshape(X_plot)
dU=model.net_epsilon(X_plot)
stress=model.net_stress(dU)
X_plot=X_plot.squeeze()
plt.plot(X_plot, U.numpy().squeeze(), '-k', label='displacement')
plt.plot(X_plot, dU.numpy().squeeze(), '-b', label='strain')
plt.plot(X_plot, dU.numpy().squeeze(), '-r', label='stress')
plt.legend(loc='best')
plt.show()
# model.train(0)