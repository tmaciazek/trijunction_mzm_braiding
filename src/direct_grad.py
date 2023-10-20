import numpy as np
import matplotlib.pyplot as plt
import time

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# suppress warnings
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from tjunction_utils import *
from dnn_utils import *

tfpi = tf.constant(pi, dtype=tf.float64)


N = 55 #number of sites
Delta = -0.55; #superconducting gap
w = 2.; #hopping amplitude
V0=30.1 #peak of the on-site potential
dedge = 5.
mu0 = 1. # background potential
noiseVar = 0.02 # variance of mu0 noise, set to 0.0 for no noise


'''
N = 30  # number of sites is 3N+1
Delta = -400.0  # superconducting gap
w = 800.0  # hopping amplitude
V0 = 1400.0  # peak of the on-site potential
dedge = 3.0  # anyons' positions away from the edge
mu0 = 0.0  # baseline potential
'''

"""
	Adding noise to the background potential
	
	If random.seed is not fixed, the code realises online stochastic gradient descent.
"""
#np.random.seed(42) # uncomment for constant noise training
noise = np.random.normal(0.0, noiseVar, 3 * N + 1)


"""
	Hamiltonian with zero on-site potentials
"""

KitaevH = construct_hamiltonian_zero_diag(N, Delta, w, V0, dedge)

"""
	Hamiltonian at time t=0
"""

s10 = tf.constant([0.0], dtype=tf.float64)
s20 = tf.constant([0.0], dtype=tf.float64)
PotentialProfile0 = potential_profile_I(N, dedge, mu0, noise, V0, s10, s20)
KitaevH0 = tf.tensor_scatter_nd_update(
    KitaevH, [[i, i] for i in range(6 * N + 2)], PotentialProfile0[0]
)
eig0 = tf.linalg.eigh(KitaevH0)

print("Zero mode energies at t=0:")
print(tf.math.real(eig0[0])[3 * N : 3 * N + 2].numpy())
print("Gap at t=0:")
print(
    tf.math.real(eig0[0])[3 * N + 2].numpy() - tf.math.real(eig0[0])[3 * N + 1].numpy()
)


# zero modes are the middle two columns

zeromode10 = eig0[1][:, 3 * N + 1]
zeromode20 = eig0[1][:, 3 * N]

"""
	Hamiltonian at time t=T
"""

s1T = tf.constant([1.0], dtype=tf.float64)
s2T = tf.constant([1.0], dtype=tf.float64)
PotentialProfileT = potential_profile_IV(N, dedge, mu0, noise, V0, s1T, s2T)
KitaevHT = tf.tensor_scatter_nd_update(
    KitaevH, [[i, i] for i in range(6 * N + 2)], PotentialProfileT[0]
)
eigT = tf.linalg.eigh(KitaevHT)
print("Zero mode energies at t=T:")
print(tf.math.real(eigT[0])[3 * N : 3 * N + 2].numpy())
print("Gap at t=T:")
print(
    tf.math.real(eigT[0])[3 * N + 2].numpy() - tf.math.real(eigT[0])[3 * N + 1].numpy()
)

zeromode1 = eigT[1][:, 3 * N + 1]
zeromode2 = eigT[1][:, 3 * N]

#plot_modes(N, zeromode1, zeromode2)
#plt.show()


"""
	EXCHANGE IN 4 STAGES 
	
	Each stage takes DeltaT time and has NT time steps.
"""

DeltaT = 250.0
NT = 2000
#DeltaT = 0.9
#NT = 800

dt = DeltaT / NT


@tf.function
def loss_fn(s12pred):
    (
        s1VarI,
        s2VarI,
        s1VarII,
        s2VarII,
        s1VarIII,
        s2VarIII,
        s1VarIV,
        s2VarIV,
    ) = profiles_from_NN(s12pred)

    zero = tf.constant([0.0], dtype=tf.float64)
    one = tf.constant([1.0], dtype=tf.float64)
    UI = transport_operator(
        N,
        KitaevH,
        mu0,
        noise,
        V0,
        dedge,
        tf.concat([zero, s1VarI, one], -1),
        tf.concat([zero, s2VarI], -1),
        dt,
        "I",
    )
    UII = transport_operator(
        N, KitaevH, mu0, noise, V0, dedge, s1VarII, tf.concat([s2VarII, one], -1), dt, "II"
    )
    UIII = transport_operator(
        N, KitaevH, mu0, noise, V0, dedge, tf.concat([s1VarIII, one], -1), s2VarIII, dt, "III"
    )
    UIV = transport_operator(
        N,
        KitaevH,
        mu0,
        noise,
        V0,
        dedge,
        tf.concat([s1VarIV, one], -1),
        tf.concat([s2VarIV, one], -1),
        dt,
        "IV",
    )
    Utot = tf.matmul(UIV, tf.matmul(UIII, tf.matmul(UII, UI)))

    return infidelity(N, eig0[1], eigT[1], Utot)


"""
	Optimisation directly on profile
"""

tVec = np.linspace(0.0, 1.0, NT, dtype=np.float64).reshape(1, -1)

layer_dims = [400, 1800, 1800, 1200, 8]

ep0 = 120
params = initialize_parameters_from_model(
    layer_dims, file='models/harmonic_trained_'+str(ep0)+'EP'
)
total_params = 0

print("Model summary:")
for layer in range(1, len(layer_dims) + 1):
    print("Layer " + str(layer))
    print(params["W" + str(layer)].shape)
    layer_params = (
        params["W" + str(layer)].shape[0] * params["W" + str(layer)].shape[1]
        + params["b" + str(layer)].shape[0]
    )
    print(layer_params)
    total_params += layer_params
print("Total params:\t" + str(total_params) + "\n")

s12pred, _ = model_forward(tVec, params)
s12 = tf.Variable(np.transpose(s12pred), dtype=tf.float64)

lr = 1e-6
optimizer = Adam(learning_rate=lr)

n_epochs = 60

for ep in range(n_epochs):
    print("Step ", ep)

    noise = np.random.normal(0.0, noiseVar, 3 * N + 1)

    PotentialProfile0 = potential_profile_I(N, dedge, mu0, noise, V0, s10, s20)
    KitaevH0 = tf.tensor_scatter_nd_update(
        KitaevH, [[i, i] for i in range(6 * N + 2)], PotentialProfile0[0]
    )
    eig0 = tf.linalg.eigh(KitaevH0)
    
    PotentialProfileT = potential_profile_IV(N, dedge, mu0, noise, V0, s1T, s2T)
    KitaevHT = tf.tensor_scatter_nd_update(
        KitaevH, [[i, i] for i in range(6 * N + 2)], PotentialProfileT[0]
    )
    eigT = tf.linalg.eigh(KitaevHT)
    
    then = time.time()
    
    loss_grad = infidelity_grad(N, eig0[1], eigT[1], KitaevH, mu0, noise, V0, dedge, s12, dt)
    optimizer.apply_gradients(grads_and_vars=[ (loss_grad[1], s12) ])
    
    print( "Current infidelity: "+str(loss_grad[0].numpy() ) )
    print('Gradient step took: '+ str(time.time()-then)+' seconds\n')
    
    if ep % 20 == 0 and ep > 0: 
    	np.savetxt('profiles/grad_harmonic_'+str(ep0)+'EP_'+str(ep)+'steps.txt', s12.numpy())

print("Final infidelity: " + str(loss_fn(s12).numpy()))

np.savetxt('profiles/grad_harmonic_'+str(ep0)+'EP_'+str(n_epochs)+'steps.txt', s12.numpy())
plot_profiles(NT, [np.transpose(s12pred), s12], ['NN', 'optimised'])
plt.savefig('profiles/grad_harmonic_'+str(ep0)+'EP_'+str(n_epochs)+'steps.pdf')


