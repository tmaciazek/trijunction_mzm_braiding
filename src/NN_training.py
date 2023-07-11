import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tjunction_utils import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

# suppress warnings
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam


N = 30  # number of sites is 3N+1
Delta = -400.0  # superconducting gap
w = 800.0  # hopping amplitude
V0 = 1400.0  # peak of the on-site potential
dedge = 3.0  # anyons' positions away from the edge
mu0 = 0.0  # baseline potential

"""
	Hamiltonian with zero on-site potentials
"""

KitaevH = construct_hamiltonian_zero_diag(N, Delta, w, V0, dedge)

"""
	update the diagonal of the Hamiltonian with suitable on-site potentials
	
"""

"""
	Hamiltonian at time t=0
"""

s10 = tf.constant([0.0], dtype=tf.float64)
s20 = tf.constant([0.0], dtype=tf.float64)
PotentialProfile0 = potential_profile_I(N, dedge, mu0, V0, s10, s20)
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
PotentialProfileT = potential_profile_IV(N, dedge, mu0, V0, s1T, s2T)
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

# PlotModes(N,zeromode1,zeromode2)
# plt.show()

"""
	EXCHANGE IN 4 STAGES 
	
	Stages I and IV take DeltaT time and have NT time steps.
	Stages II and III take 2*DeltaT time and have 2*NT time steps.
"""

DeltaT = 0.9
NT = 1000


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
    UI = transport_operator_map(
        N,
        KitaevH,
        mu0,
        V0,
        dedge,
        tf.concat([zero, s1VarI, one], -1),
        tf.concat([zero, s2VarI], -1),
        dt,
        "I",
    )
    UII = transport_operator_map(
        N, KitaevH, mu0, V0, dedge, s1VarII, tf.concat([s2VarII, one], -1), dt, "II"
    )
    UIII = transport_operator_map(
        N, KitaevH, mu0, V0, dedge, tf.concat([s1VarIII, one], -1), s2VarIII, dt, "III"
    )
    UIV = transport_operator_map(
        N,
        KitaevH,
        mu0,
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
	Build Model
"""

model = Sequential()
model.add(Dense(400, activation="relu"))
NHlayers = 2
for ilayer in range(NHlayers):
    model.add(Dense(1800, activation="relu"))
model.add(Dense(1200, activation="relu"))
model.add(Dense(8, activation="sigmoid"))


# model.load_weights('models/short_sinusoidal_pretrained_3HL')
model.load_weights("models/short_exchange_trained150EP3HL_run4")


tVec = tf.reshape(tf.cast(tf.linspace(0.0, 1.0, NT - 1), dtype=tf.float64), [-1, 1])
s12pred = model.predict(tVec)

model.summary()

then = time.time()

infid = loss_fn(s12pred)
print("Starting infidelity: " + str(infid.numpy()))
print("Transport took: " + str(time.time() - then) + " seconds")


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
plot_profiles(
    NT, s1VarI, s2VarI, s1VarII, s2VarII, s1VarIII, s2VarIII, s1VarIV, s2VarIV
)
plt.show()


"""
	Optimisation
"""

lr = 0.00001
model.compile(optimizer=Adam(learning_rate=lr))
print("Learning Rate: " + str(lr))


@tf.function
def update_loss(model):
    with tf.GradientTape() as tape:
        s12pred = model(tVec)
        L = loss_fn(s12pred)
    grad = tape.gradient(L, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return L


appendix = "3HL"

N_epochs = 50
for i in range(N_epochs):
    print("Step ", i)
    then = time.time()
    Loss_i = update_loss(model)
    print("Gradient descent step {} took: ".format(i), time.time() - then, "seconds")
    print("Current Infidelity:", Loss_i.numpy())


s12pred = model(tVec)
np.savetxt("short_s12_50EP" + appendix + ".txt", s12pred)

model.save_weights("/user/work/kk19347/short_exchange_trained50EP" + appendix)
