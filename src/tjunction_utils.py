#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 20 2023
Utility functions for training a neural net to find optimal exchange protocol.
@author: tmaciazek
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

tfpi = tf.constant(pi, dtype=tf.float64)
zero = tf.constant([0.0], dtype=tf.float64)
one = tf.constant([1.0], dtype=tf.float64)
I = -tf.cast(tf.complex(0.0, 1.0), dtype=tf.complex128)

"""
	POTENTIAL PROFILES OF THE 4 STAGES
	
	s1 and s2 are tensors of shape (k, )
"""


@tf.function
def potential_profile_I(N, dedge, mu0, noise, V0, s1, s2):
    """
    s1 - particle to the left (0 = far left, 1 = middle)
    s2 - particle to the right (0 = far right, 1 = junction)
    """
    xR = (1.0 - s2) * (2.0 * N - dedge) + s2 * N
    xL = dedge + s1 * (N - dedge)
    ones = tf.ones(s1.shape[0], tf.float64)

    profileH = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * (tf.math.sigmoid(x - xR) + tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileV = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * (tf.math.sigmoid(x + ones)),
            elems=tf.range(N, dtype=tf.float64),
        )
    )
    profile = tf.cast(tf.concat([profileH, profileV], 1) + noise, dtype=tf.complex128)

    return tf.concat([profile, -profile], 1)


@tf.function
def potential_profile_grad_I(N, dedge, V0, s1, s2):
    """
    s1 - particle to the left (0 = far left, 1 = middle)
    s2 - particle to the right (0 = far right, 1 = junction)
    """
    xR = (1.0 - s2) * (2.0 * N - dedge) + s2 * N
    xL = dedge + s1 * (N - dedge)
    ones = tf.ones(s1.shape[0], tf.float64)

    profileH_grad_s1 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (N - dedge)
            * tf.math.sigmoid(xL - x)
            * (1.0 - tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )

    profileH_grad_s2 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (N - dedge)
            * tf.math.sigmoid(x - xR)
            * (1.0 - tf.math.sigmoid(x - xR)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )

    profileV_grad = tf.zeros((s1.shape[0], N), tf.float64)
    profile_grad_s1 = tf.cast(
        tf.concat([profileH_grad_s1, profileV_grad], 1), dtype=tf.complex128
    )
    profile_grad_s2 = tf.cast(
        tf.concat([profileH_grad_s2, profileV_grad], 1), dtype=tf.complex128
    )

    return tf.concat([profile_grad_s1, -profile_grad_s1], 1), tf.concat(
        [profile_grad_s2, -profile_grad_s2], 1
    )


@tf.function
def potential_profile_II(N, dedge, mu0, noise, V0, s1, s2):
    """
    s1 - particle on the vertical chain (0 = junction, 1 = bottom)
    s2 - particle to the right (0 = far right, 1 = junction)
    """
    xR = (1.0 - s2) * (2.0 * N - dedge) + s2 * N
    xL = s1 * dedge + (1.0 - s1) * N
    Nones = N * tf.ones(s1.shape[0], tf.float64)

    profile1 = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * (tf.math.sigmoid(x - xR) + tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileHL = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * tf.math.sigmoid(Nones - x),
            elems=tf.range(N, dtype=tf.float64),
        )
    )
    profile = tf.concat(
        [profileHL, profile1[:, N:], tf.reverse(profile1[:, :N], [-1])], 1
    ) + noise

    return tf.cast(tf.concat([profile, -profile], 1), dtype=tf.complex128)


@tf.function
def potential_profile_grad_II(N, dedge, V0, s1, s2):
    """
    s1 - particle on the vertical chain (0 = junction, 1 = bottom)
    s2 - particle to the right (0 = far right, 1 = junction)
    """
    xR = (1.0 - s2) * (2.0 * N - dedge) + s2 * N
    xL = s1 * dedge + (1.0 - s1) * N
    Nones = N * tf.ones(s1.shape[0], tf.float64)

    profile1_grad_s1 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (dedge - N)
            * tf.math.sigmoid(xL - x)
            * (1.0 - tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profile1_grad_s2 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (N - dedge)
            * tf.math.sigmoid(x - xR)
            * (1.0 - tf.math.sigmoid(x - xR)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileHL_grad = tf.zeros((s1.shape[0], N), tf.float64)
    profile_grad_s1 = tf.concat(
        [
            profileHL_grad,
            profile1_grad_s1[:, N:],
            tf.reverse(profile1_grad_s1[:, :N], [-1]),
        ],
        1,
    )
    profile_grad_s2 = tf.concat(
        [
            profileHL_grad,
            profile1_grad_s2[:, N:],
            tf.reverse(profile1_grad_s2[:, :N], [-1]),
        ],
        1,
    )

    return tf.cast(
        tf.concat([profile_grad_s1, -profile_grad_s1], 1), dtype=tf.complex128
    ), tf.cast(tf.concat([profile_grad_s2, -profile_grad_s2], 1), dtype=tf.complex128)


@tf.function
def potential_profile_III(N, dedge, mu0, noise, V0, s1, s2):
    """
    s1 - particle on the vertical chain (0 = bottom, 1 = junction)
    s2 - particle to the left (0 = junction, 1 = far left)
    """
    xL = (1.0 - s1) * dedge + s1 * N
    xR = s2 * (2.0 * N - dedge) + (1.0 - s2) * N
    ones = tf.ones(s1.shape[0], tf.float64)

    profile1 = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * (tf.math.sigmoid(x - xR) + tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileHR = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * tf.math.sigmoid(x + ones),
            elems=tf.range(N, dtype=tf.float64),
        )
    )
    profile = tf.concat(
        [
            tf.reverse(profile1[:, N:], [-1]),
            profileHR,
            tf.reverse(profile1[:, :N], [-1]),
        ],
        1,
    ) + noise

    return tf.cast(tf.concat([profile, -profile], 1), dtype=tf.complex128)


@tf.function
def potential_profile_grad_III(N, dedge, V0, s1, s2):
    """
    s1 - particle on the vertical chain (0 = bottom, 1 = junction)
    s2 - particle to the left (0 = junction, 1 = far left)
    """
    xL = (1.0 - s1) * dedge + s1 * N
    xR = s2 * (2.0 * N - dedge) + (1.0 - s2) * N
    ones = tf.ones(s1.shape[0], tf.float64)

    profile1_grad_s1 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (N - dedge)
            * tf.math.sigmoid(xL - x)
            * (1.0 - tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profile1_grad_s2 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (dedge - N)
            * tf.math.sigmoid(x - xR)
            * (1.0 - tf.math.sigmoid(x - xR)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileHR_grad = tf.zeros((s1.shape[0], N), tf.float64)
    profile_grad_s1 = tf.concat(
        [
            tf.reverse(profile1_grad_s1[:, N:], [-1]),
            profileHR_grad,
            tf.reverse(profile1_grad_s1[:, :N], [-1]),
        ],
        1,
    )
    profile_grad_s2 = tf.concat(
        [
            tf.reverse(profile1_grad_s2[:, N:], [-1]),
            profileHR_grad,
            tf.reverse(profile1_grad_s2[:, :N], [-1]),
        ],
        1,
    )

    return tf.cast(
        tf.concat([profile_grad_s1, -profile_grad_s1], 1), dtype=tf.complex128
    ), tf.cast(tf.concat([profile_grad_s2, -profile_grad_s2], 1), dtype=tf.complex128)


@tf.function
def potential_profile_IV(N, dedge, mu0, noise, V0, s1, s2):
    """
    s1 - particle to the right (0 = junction, 1 = far right)
    s2 - particle to the left (0 = junction, 1 = far left)
    """
    xL = (1.0 - s2) * N + s2 * dedge
    xR = s1 * (2.0 * N - dedge) + (1.0 - s1) * N
    ones = tf.ones(s1.shape[0], tf.float64)

    profileH = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * (tf.math.sigmoid(x - xR) + tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileV = tf.transpose(
        tf.map_fn(
            fn=lambda x: mu0 + V0 * (tf.math.sigmoid(x + ones)),
            elems=tf.range(N, dtype=tf.float64),
        )
    )
    profile = tf.concat([profileH, profileV], 1) + noise

    return tf.cast(tf.concat([profile, -profile], 1), dtype=tf.complex128)


@tf.function
def potential_profile_grad_IV(N, dedge, V0, s1, s2):
    """
    s1 - particle to the right (0 = junction, 1 = far right)
    s2 - particle to the left (0 = junction, 1 = far left)
    """
    xL = (1.0 - s2) * N + s2 * dedge
    xR = s1 * (2.0 * N - dedge) + (1.0 - s1) * N
    ones = tf.ones(s1.shape[0], tf.float64)

    profileH_grad_s1 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (dedge - N)
            * tf.math.sigmoid(x - xR)
            * (1.0 - tf.math.sigmoid(x - xR)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileH_grad_s2 = tf.transpose(
        tf.map_fn(
            fn=lambda x: V0
            * (dedge - N)
            * tf.math.sigmoid(xL - x)
            * (1.0 - tf.math.sigmoid(xL - x)),
            elems=tf.range(2 * N + 1, dtype=tf.float64),
        )
    )
    profileV_grad = tf.zeros((s1.shape[0], N), tf.float64)
    profile_grad_s1 = tf.concat([profileH_grad_s1, profileV_grad], 1)
    profile_grad_s2 = tf.concat([profileH_grad_s2, profileV_grad], 1)

    return tf.cast(
        tf.concat([profile_grad_s1, -profile_grad_s1], 1), dtype=tf.complex128
    ), tf.cast(tf.concat([profile_grad_s2, -profile_grad_s2], 1), dtype=tf.complex128)


def plot_modes(N, mode1, mode2):
    """
    A 4-panel plot of the two given modes. The left plots show abs(mode1) while the
    right plots show abs(mode2).
    -- horizontal sites have indices 0 - 2N from left to right
    -- vertical sites have indices 2N+1 - 3N from top to bottom
    """

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True)
    ax1.plot(range(0, 2 * N + 1), tf.abs(mode1[: 2 * N + 1]))
    ax1.set_title("Horizontal " + str(2 * N + 1) + " sites")
    ax2.plot(range(0, N), tf.abs(mode1[2 * N + 1 : 3 * N + 1]))
    ax2.set_title("Vertical " + str(N) + " sites")
    ax3.plot(range(0, 2 * N + 1), tf.abs(mode2[: 2 * N + 1]))
    ax3.set_title("Horizontal " + str(2 * N + 1) + " sites")
    ax4.plot(range(0, N), tf.abs(mode2[2 * N + 1 : 3 * N + 1]))
    ax4.set_title("Vertical " + str(N) + " sites")


@tf.function
def construct_hamiltonian_zero_diag(N, Delta, w, V0, dedge):
    """
    Constructs Kitaev hamiltonian of the trijunction with zero on-site potentials.
    The horizontal bar/chain has a real superconducting gap. The vertical bar has
    a purely imaginary superconducting gap of the same amplitude.

    The hamiltonian of a single chain consists of four tridiagonal NxN blocks.
    """

    """
		HORIZONTAL BAR
	"""

    diagDeltaH = tf.experimental.numpy.full(2 * N + 1, Delta / 2.0, dtype=tf.complex128)
    diagWH = tf.experimental.numpy.full(2 * N + 1, -w / 2.0, dtype=tf.complex128)
    zerosNH = tf.experimental.numpy.full(2 * N + 1, 0.0, dtype=tf.complex128)
    blockDeltaH = tf.linalg.LinearOperatorTridiag(
        [-diagDeltaH, zerosNH, diagDeltaH], diagonals_format="sequence"
    )
    diagWH = tf.experimental.numpy.full(2 * N + 1, -w / 2.0, dtype=tf.complex128)
    blockMuWH = tf.linalg.LinearOperatorTridiag(
        [diagWH, zerosNH, diagWH], diagonals_format="sequence"
    )
    blockMuWH = blockMuWH.to_dense()
    blockDeltaH = blockDeltaH.to_dense()

    """
		embed the blocks into the horizontal chain hamiltonian
	"""

    zeros2Np1xN = tf.zeros((2 * N + 1, N), dtype=tf.complex128)
    zerosNx6Np2 = tf.zeros((N, 6 * N + 2), dtype=tf.complex128)
    KitaevHHupper = tf.concat(
        [tf.concat([blockMuWH, zeros2Np1xN, blockDeltaH, zeros2Np1xN], 1), zerosNx6Np2],
        0,
    )
    KitaevHHlower = tf.concat(
        [
            tf.concat([-blockDeltaH, zeros2Np1xN, -blockMuWH, zeros2Np1xN], 1),
            zerosNx6Np2,
        ],
        0,
    )
    KitaevHH = tf.concat([KitaevHHupper, KitaevHHlower], 0)

    """
		VERTICAL BAR
		
		purely imaginary Delta
	"""

    diagIDelta = tf.experimental.numpy.full(
        N, tf.complex(0.0, 1.0) * Delta / 2.0, dtype=tf.complex128
    )
    diagW = tf.experimental.numpy.full(N, -w / 2.0, dtype=tf.complex128)
    zerosN = tf.experimental.numpy.full(N, 0.0, dtype=tf.complex128)
    blockIDelta = tf.linalg.LinearOperatorTridiag(
        [-diagIDelta, zerosN, diagIDelta], diagonals_format="sequence"
    )
    diagW = tf.experimental.numpy.full(N, -w / 2.0, dtype=tf.complex128)
    blockMuW = tf.linalg.LinearOperatorTridiag(
        [diagW, zerosN, diagW], diagonals_format="sequence"
    )
    blockMuW = blockMuW.to_dense()
    blockIDelta = blockIDelta.to_dense()

    """
		embed the blocks into the vertical chain hamiltonian
	"""

    zerosNx2Np1 = tf.zeros((N, 2 * N + 1), dtype=tf.complex128)
    zeros2Np1x6Np2 = tf.zeros((2 * N + 1, 6 * N + 2), dtype=tf.complex128)
    KitaevHVupper = tf.concat(
        [
            zeros2Np1x6Np2,
            tf.concat([zerosNx2Np1, blockMuW, zerosNx2Np1, -blockIDelta], 1),
        ],
        0,
    )
    KitaevHVlower = tf.concat(
        [
            zeros2Np1x6Np2,
            tf.concat([zerosNx2Np1, -blockIDelta, zerosNx2Np1, -blockMuW], 1),
        ],
        0,
    )
    KitaevHV = tf.concat([KitaevHVupper, KitaevHVlower], 0)

    """
		HAMILTONIAN OF THE TWO BARS
		
		horizontal sites have indices 0 - 2N from left to right
		vertical sites have indices 2N+1 - 3N from top to bottom
	"""
    KitaevH = tf.add(KitaevHH, KitaevHV)

    # coupling between the bars
    CouplingIndices = [
        [N, 2 * N + 1],
        [2 * N + 1, N],
        [5 * N + 2, 4 * N + 1],
        [4 * N + 1, 5 * N + 2],
        [4 * N + 1, 2 * N + 1],
        [2 * N + 1, 4 * N + 1],
        [5 * N + 2, N],
        [N, 5 * N + 2],
    ]
    IDelta = tf.cast(
        tf.cast(tf.complex(0.0, 1.0), tf.complex128) * Delta / 2, dtype=tf.complex128
    )
    Couplings = [-w / 2, -w / 2, w / 2, w / 2, IDelta, -IDelta, -IDelta, IDelta]

    return tf.tensor_scatter_nd_update(KitaevH, CouplingIndices, Couplings)


@tf.function
def transport_operator(N, Ham0, mu0, noise, V0, dedge, s1Vec, s2Vec, dt, stage):
    """
    Generates the adiabatic evolution operator.

    can be potentially improved by using a more accurate version of the
    Suzuki-Trotter formula
    """
    if stage == "I":
        PotentialProfiles = potential_profile_I(N, dedge, mu0, noise, V0, s1Vec, s2Vec)
    elif stage == "II":
        PotentialProfiles = potential_profile_II(N, dedge, mu0, noise, V0, s1Vec, s2Vec)
    elif stage == "III":
        PotentialProfiles = potential_profile_III(N, dedge, mu0, noise, V0, s1Vec, s2Vec)
    elif stage == "IV":
        PotentialProfiles = potential_profile_IV(N, dedge, mu0, noise, V0, s1Vec, s2Vec)

    """
		evolution calculated via map
	"""
    HamExps = tf.map_fn(
        fn=lambda MuList: tf.linalg.expm(
            tf.math.scalar_mul(
                I * dt,
                tf.tensor_scatter_nd_update(
                    Ham0, [[i, i] for i in range(6 * N + 2)], MuList
                ),
            )
        ),
        elems=PotentialProfiles,
    )
    EvolutionOperator = tf.foldr(tf.matmul, HamExps)
    del HamExps

    return EvolutionOperator


@tf.function
def infidelity(N, EigenmodesInitial, EigenmodesFinal, EvolutionOperator):
    # auxilliary permutation matrices P and P1 to be used for exchanging zero modes
    P1 = tf.cast(
        tf.linalg.diag(
            tf.concat(
                [tf.experimental.numpy.zeros((1)), tf.experimental.numpy.ones((3 * N))],
                -1,
            )
        ),
        dtype=tf.complex128,
    )
    P = tf.concat(
        [
            tf.concat([P1, tf.eye(3 * N + 1, dtype=tf.complex128) - P1], 1),
            tf.concat([tf.eye(3 * N + 1, dtype=tf.complex128) - P1, P1], 1),
        ],
        0,
    )

    modesU_T = EigenmodesFinal[0 : 3 * N + 1, 3 * N + 1 : 6 * N + 2]
    modesV_T = EigenmodesFinal[3 * N + 1 : 6 * N + 2, 3 * N + 1 : 6 * N + 2]

    # switch zero modes
    W_T = tf.concat(
        [
            tf.concat([modesU_T, tf.math.conj(modesV_T)], 1),
            tf.concat([modesV_T, tf.math.conj(modesU_T)], 1),
        ],
        0,
    )
    w_T = tf.matmul(W_T, P)
    modesU_T2 = w_T[0 : 3 * N + 1, 0 : 3 * N + 1]
    modesV_T2 = w_T[3 * N + 1 : 6 * N + 2, 0 : 3 * N + 1]

    ModesEv = tf.matmul(EvolutionOperator, EigenmodesInitial)
    modesU_ev = ModesEv[0 : 3 * N + 1, 3 * N + 1 : 6 * N + 2]
    modesV_ev = ModesEv[3 * N + 1 : 6 * N + 2, 3 * N + 1 : 6 * N + 2]

    Overlap1 = tf.math.abs(
        tf.math.reduce_prod(
            tf.linalg.eigvals(
                tf.matmul(
                    tf.linalg.matrix_transpose(modesU_T, conjugate=True), modesU_ev
                )
                + tf.matmul(
                    tf.linalg.matrix_transpose(modesV_T, conjugate=True), modesV_ev
                )
            )
        )
    )
    Overlap2 = tf.math.abs(
        tf.math.reduce_prod(
            tf.linalg.eigvals(
                tf.matmul(
                    tf.linalg.matrix_transpose(modesU_T2, conjugate=True), modesU_ev
                )
                + tf.matmul(
                    tf.linalg.matrix_transpose(modesV_T2, conjugate=True), modesV_ev
                )
            )
        )
    )

    return 1.0 - tf.math.maximum(Overlap1, Overlap2)


@tf.function
def ham_exp_derivative(Hams, DHam, dt):
    """
    Computes the derivative of Exp(-I*dt*Ham(s)) with respect to s for a batch of Ham which
    have no spectral degeneracy.

    Uses the formula from
    Kalbfleisch and Lawless, J. Amer. Statist. Assoc., 80, 863–871 (1985)

    Input:
    Ham - tensor of shape (batchsize, M, M) containing values of the Hamiltonians at s=s0
    DHam - tensor of the derivatives of -I*Ham at s=s0

    Returns:
    d(Exp(-I*dt*Ham(s)))/ds at s=s0
    """
    # Read off batch size
    batchsize = Hams.shape[0]

    # Diagonalise Hamiltonians
    #eig = tf.linalg.eigh(Hams)   #faster, but uses more memory
    eig = tf.map_fn(
    	fn=lambda ham: tf.linalg.eigh(ham),
    	elems = Hams,
    	fn_output_signature = (tf.complex128, tf.complex128)
	)

    XT = tf.linalg.matrix_transpose(eig[1], conjugate=True)

    # Form matrix of pairwise differences of the eigenvalues (uses broadcasting)
    eigen_diffs = tf.math.scalar_mul(
        I,
        tf.math.subtract(
            tf.reshape(eig[0], (batchsize, -1, 1)),
            tf.reshape(eig[0], (batchsize, 1, -1)),
        ),
    )

    # Exponents of (-eigenvalues*I*dt)
    eigen_exps = tf.math.exp(tf.math.scalar_mul(I * dt, eig[0]))

    # Matrix D from the paper
    # D_ij = (Exp(-dt*I*e_i)-Exp(-dt*I*e_j))/((-I)*(e_i-e_j))
    # D_ii = dt*Exp(dt*e_i)
    D = tf.math.divide_no_nan(
        tf.reshape(eigen_exps, (batchsize, -1, 1))
        - tf.reshape(eigen_exps, (batchsize, 1, -1)),
        eigen_diffs,
    )
    D = D + tf.math.scalar_mul(dt, tf.linalg.diag(eigen_exps))

    # Matrix G from the paper
    G = tf.matmul(tf.matmul(XT, DHam), eig[1])

    return tf.matmul(tf.matmul(eig[1], tf.math.multiply(G, D)), XT)


@tf.function
def evolution_grad_single_stage(
    N, Ham0, mu0, noise, V0, dedge, s12, dt, UI, UII, UIII, UIV, Utot, stage_label="I"
):

    (
        s1I,
        s2I,
        s1II,
        s2II,
        s1III,
        s2III,
        s1IV,
        s2IV,
    ) = profiles_from_NN(s12)

    if stage_label == "I":
        pp = potential_profile_I(
            N,
            dedge,
            mu0,
            noise,
            V0,
            tf.concat([zero, s1I, one], -1),
            tf.concat([zero, s2I], -1),
        )
        D_pp = potential_profile_grad_I(
            N, dedge, V0, tf.concat([zero, s1I, one], -1), tf.concat([zero, s2I], -1)
        )
        hams = tf.map_fn(
            fn=lambda diag: tf.tensor_scatter_nd_update(
                Ham0, [[i, i] for i in range(6 * N + 2)], diag
            ),
            elems=pp,
        )
        D1_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[0][1:-1]))
        D2_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[1][1:]))
        D1_ham_exps = ham_exp_derivative(hams[1:-1], D1_hams, dt)
        D2_ham_exps = ham_exp_derivative(hams[1:], D2_hams, dt)
    elif stage_label == "II":
        pp = potential_profile_II(N, dedge, mu0, noise, V0, s1II, tf.concat([s2II, one], -1))
        D_pp = potential_profile_grad_II(N, dedge, V0, s1II, tf.concat([s2II, one], -1))
        hams = tf.map_fn(
            fn=lambda diag: tf.tensor_scatter_nd_update(
                Ham0, [[i, i] for i in range(6 * N + 2)], diag
            ),
            elems=pp,
        )
        D1_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[0]))
        D2_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[1][:-1]))
        D1_ham_exps = ham_exp_derivative(hams, D1_hams, dt)
        D2_ham_exps = ham_exp_derivative(hams[:-1], D2_hams, dt)
    elif stage_label == "III":
        pp = potential_profile_III(
            N, dedge, mu0, noise, V0, tf.concat([s1III, one], -1), s2III
        )
        D_pp = potential_profile_grad_III(
            N, dedge, V0, tf.concat([s1III, one], -1), s2III
        )
        hams = tf.map_fn(
            fn=lambda diag: tf.tensor_scatter_nd_update(
                Ham0, [[i, i] for i in range(6 * N + 2)], diag
            ),
            elems=pp,
        )
        D1_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[0][:-1]))
        D2_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[1]))
        D1_ham_exps = ham_exp_derivative(hams[:-1], D1_hams, dt)
        D2_ham_exps = ham_exp_derivative(hams, D2_hams, dt)
    else:
        pp = potential_profile_IV(
            N, dedge, mu0, noise, V0, tf.concat([s1IV, one], -1), tf.concat([s2IV, one], -1)
        )
        D_pp = potential_profile_grad_IV(
            N, dedge, V0, tf.concat([s1IV, one], -1), tf.concat([s2IV, one], -1)
        )
        hams = tf.map_fn(
            fn=lambda diag: tf.tensor_scatter_nd_update(
                Ham0, [[i, i] for i in range(6 * N + 2)], diag
            ),
            elems=pp,
        )
        D1_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[0][:-1]))
        D2_hams = tf.math.scalar_mul(I, tf.linalg.diag(D_pp[1][:-1]))
        D1_ham_exps = ham_exp_derivative(hams[:-1], D1_hams, dt)
        D2_ham_exps = ham_exp_derivative(hams[:-1], D2_hams, dt)

    ham_exps = tf.map_fn(
        fn=lambda ham: tf.linalg.expm(tf.math.scalar_mul(I * dt, ham)), elems=hams
    )

    """
		Loop computing the gradient of the evolution operator
	"""
    Nsteps = s12.shape[0]
    grad_ind0 = tf.constant(2)

    if stage_label == "I":
        Ubefore0 = ham_exps[0]
        Uafter0 = tf.matmul(
            Utot, tf.transpose(tf.matmul(ham_exps[1], ham_exps[0]), conjugate=True)
        )
        D1_Utot = tf.matmul(Uafter0, tf.matmul(D1_ham_exps[0], Ubefore0))
        D1_Utot = tf.reshape(D1_Utot, (1, 6 * N + 2, 6 * N + 2))
        D2_Utot = tf.matmul(Uafter0, tf.matmul(D2_ham_exps[0], Ubefore0))
        D2_Utot = tf.reshape(D2_Utot, (1, 6 * N + 2, 6 * N + 2))
        Uafter0 = tf.matmul(Uafter0, tf.transpose(ham_exps[2], conjugate=True))
        Ubefore0 = tf.matmul(ham_exps[1], Ubefore0)
    elif stage_label == "II":
        Ubefore0 = UI
        Uafter0 = tf.matmul(
            Utot, tf.transpose(tf.matmul(ham_exps[0], Ubefore0), conjugate=True)
        )
        D1_Utot = tf.matmul(Uafter0, tf.matmul(D1_ham_exps[0], Ubefore0))
        D1_Utot = tf.reshape(D1_Utot, (1, 6 * N + 2, 6 * N + 2))
        D2_Utot = tf.matmul(Uafter0, tf.matmul(D2_ham_exps[0], Ubefore0))
        D2_Utot = tf.reshape(D2_Utot, (1, 6 * N + 2, 6 * N + 2))
        Uafter0 = tf.matmul(Uafter0, tf.transpose(ham_exps[1], conjugate=True))
        Ubefore0 = tf.matmul(ham_exps[0], Ubefore0)
    elif stage_label == "III":
        Ubefore0 = tf.matmul(UII, UI)
        Uafter0 = tf.matmul(
            Utot, tf.transpose(tf.matmul(ham_exps[0], Ubefore0), conjugate=True)
        )
        D1_Utot = tf.matmul(Uafter0, tf.matmul(D1_ham_exps[0], Ubefore0))
        D1_Utot = tf.reshape(D1_Utot, (1, 6 * N + 2, 6 * N + 2))
        D2_Utot = tf.matmul(Uafter0, tf.matmul(D2_ham_exps[0], Ubefore0))
        D2_Utot = tf.reshape(D2_Utot, (1, 6 * N + 2, 6 * N + 2))
        Uafter0 = tf.matmul(Uafter0, tf.transpose(ham_exps[1], conjugate=True))
        Ubefore0 = tf.matmul(ham_exps[0], Ubefore0)
    else:
        Ubefore0 = tf.matmul(UIII, tf.matmul(UII, UI))
        Uafter0 = tf.matmul(
            Utot, tf.transpose(tf.matmul(ham_exps[0], Ubefore0), conjugate=True)
        )
        D1_Utot = tf.matmul(Uafter0, tf.matmul(D1_ham_exps[0], Ubefore0))
        D1_Utot = tf.reshape(D1_Utot, (1, 6 * N + 2, 6 * N + 2))
        D2_Utot = tf.matmul(Uafter0, tf.matmul(D2_ham_exps[0], Ubefore0))
        D2_Utot = tf.reshape(D2_Utot, (1, 6 * N + 2, 6 * N + 2))
        Uafter0 = tf.matmul(Uafter0, tf.transpose(ham_exps[1], conjugate=True))
        Ubefore0 = tf.matmul(ham_exps[0], Ubefore0)

    c = lambda grad_ind, Uafter, Ubefore, D1_Utot, D2_Utot: tf.less(grad_ind, Nsteps)
    if stage_label == "I":
        calc_step = lambda grad_ind, Uafter, Ubefore, D1_Utot, D2_Utot: (
            grad_ind + 1,
            tf.matmul(Uafter, tf.transpose(ham_exps[grad_ind + 1], conjugate=True)),
            tf.matmul(ham_exps[grad_ind], Ubefore),
            tf.concat(
                [
                    D1_Utot,
                    tf.reshape(
                        tf.matmul(
                            Uafter, tf.matmul(D1_ham_exps[grad_ind - 1], Ubefore)
                        ),
                        (1, 6 * N + 2, 6 * N + 2),
                    ),
                ],
                axis=0,
            ),
            tf.concat(
                [
                    D2_Utot,
                    tf.reshape(
                        tf.matmul(
                            Uafter, tf.matmul(D2_ham_exps[grad_ind - 1], Ubefore)
                        ),
                        (1, 6 * N + 2, 6 * N + 2),
                    ),
                ],
                axis=0,
            ),
        )
    else:
        calc_step = lambda grad_ind, Uafter, Ubefore, D1_Utot, D2_Utot: (
            grad_ind + 1,
            tf.matmul(Uafter, tf.transpose(ham_exps[grad_ind], conjugate=True)),
            tf.matmul(ham_exps[grad_ind - 1], Ubefore),
            tf.concat(
                [
                    D1_Utot,
                    tf.reshape(
                        tf.matmul(
                            Uafter, tf.matmul(D1_ham_exps[grad_ind - 1], Ubefore)
                        ),
                        (1, 6 * N + 2, 6 * N + 2),
                    ),
                ],
                axis=0,
            ),
            tf.concat(
                [
                    D2_Utot,
                    tf.reshape(
                        tf.matmul(
                            Uafter, tf.matmul(D2_ham_exps[grad_ind - 1], Ubefore)
                        ),
                        (1, 6 * N + 2, 6 * N + 2),
                    ),
                ],
                axis=0,
            ),
        )

    loop_result = tf.while_loop(
        c,
        calc_step,
        loop_vars=[grad_ind0, Uafter0, Ubefore0, D1_Utot, D2_Utot],
        shape_invariants=[
            grad_ind0.get_shape(),
            Uafter0.get_shape(),
            Ubefore0.get_shape(),
            tf.TensorShape([None, 6 * N + 2, 6 * N + 2]),
            tf.TensorShape([None, 6 * N + 2, 6 * N + 2]),
        ],
        parallel_iterations=20,
    )

    if stage_label == "I":
        Uafter = loop_result[1]
        Ubefore = loop_result[2]
        D1_Utot = loop_result[3]
        D2_Utot = tf.concat(
            [
                loop_result[4],
                tf.reshape(
                    tf.matmul(
                        Uafter, tf.matmul(D2_ham_exps[loop_result[0] - 1], Ubefore)
                    ),
                    (1, 6 * N + 2, 6 * N + 2),
                ),
            ],
            axis=0,
        )
    elif stage_label == "II":
        Uafter = loop_result[1]
        Ubefore = loop_result[2]
        D1_Utot = tf.concat(
            [
                loop_result[3],
                tf.reshape(
                    tf.matmul(
                        Uafter, tf.matmul(D1_ham_exps[loop_result[0] - 1], Ubefore)
                    ),
                    (1, 6 * N + 2, 6 * N + 2),
                ),
            ],
            axis=0,
        )
        D2_Utot = loop_result[4]
    elif stage_label == "III":
        Uafter = loop_result[1]
        Ubefore = loop_result[2]
        D1_Utot = loop_result[3]
        D2_Utot = tf.concat(
            [
                loop_result[4],
                tf.reshape(
                    tf.matmul(
                        Uafter, tf.matmul(D2_ham_exps[loop_result[0] - 1], Ubefore)
                    ),
                    (1, 6 * N + 2, 6 * N + 2),
                ),
            ],
            axis=0,
        )
    else:
        Uafter = loop_result[1]
        Ubefore = loop_result[2]
        D1_Utot = tf.concat(
            [
                loop_result[3],
                tf.reshape(
                    tf.matmul(
                        Uafter, tf.matmul(D1_ham_exps[loop_result[0] - 1], Ubefore)
                    ),
                    (1, 6 * N + 2, 6 * N + 2),
                ),
            ],
            axis=0,
        )
        D2_Utot = tf.concat(
            [
                loop_result[4],
                tf.reshape(
                    tf.matmul(
                        Uafter, tf.matmul(D2_ham_exps[loop_result[0] - 1], Ubefore)
                    ),
                    (1, 6 * N + 2, 6 * N + 2),
                ),
            ],
            axis=0,
        )

    return D1_Utot, D2_Utot


@tf.function
def infidelity_grad_single_stage(
    N, EigenmodesInitial, EigenmodesFinal, Utot, D_Utot, stage_label=("I", 1)
):
    # auxilliary permutation matrix P for exchanging zero modes
    perm_vec = (
        [i for i in range(3 * N)]
        + [3 * N + 1, 3 * N]
        + [i + 3 * N + 2 for i in range(3 * N)]
    )
    P = tf.cast(
        tf.linalg.LinearOperatorPermutation(perm_vec).to_dense(), dtype=tf.complex128
    )

    o1 = tf.concat(
        [
            tf.eye(3 * N + 1, dtype=tf.complex128),
            tf.zeros((3 * N + 1, 3 * N + 1), dtype=tf.complex128),
        ],
        0,
    )
    o2 = tf.concat(
        [
            tf.zeros((3 * N + 1, 3 * N + 1), dtype=tf.complex128),
            tf.eye(3 * N + 1, dtype=tf.complex128),
        ],
        0,
    )

    modesU_T = tf.matmul(tf.matmul(tf.transpose(o1), EigenmodesFinal), o2)
    modesV_T = tf.matmul(tf.matmul(tf.transpose(o2), EigenmodesFinal), o2)

    # switch zero modes
    eigenmodes_swapped = tf.matmul(EigenmodesFinal, P)
    modesU_T2 = tf.matmul(tf.matmul(tf.transpose(o1), eigenmodes_swapped), o2)
    modesV_T2 = tf.matmul(tf.matmul(tf.transpose(o2), eigenmodes_swapped), o2)

    ModesEv = tf.matmul(Utot, EigenmodesInitial)
    modesU_ev = tf.matmul(tf.matmul(tf.transpose(o1), ModesEv), o2)
    modesV_ev = tf.matmul(tf.matmul(tf.transpose(o2), ModesEv), o2)

    VTVplusWTW = tf.matmul(
        tf.linalg.matrix_transpose(modesU_T, conjugate=True), modesU_ev
    ) + tf.matmul(tf.linalg.matrix_transpose(modesV_T, conjugate=True), modesV_ev)
    VTVplusWTW2 = tf.matmul(
        tf.linalg.matrix_transpose(modesU_T2, conjugate=True), modesU_ev
    ) + tf.matmul(tf.linalg.matrix_transpose(modesV_T2, conjugate=True), modesV_ev)
    fid = tf.math.abs(tf.linalg.det(VTVplusWTW))
    fid2 = tf.math.abs(tf.linalg.det(VTVplusWTW2))

    D_modes_ev = tf.matmul(D_Utot, EigenmodesInitial)
    D_modesU_ev = tf.matmul(tf.matmul(tf.transpose(o1), D_modes_ev), o2)
    D_modesV_ev = tf.matmul(tf.matmul(tf.transpose(o2), D_modes_ev), o2)
    
    VTVplusWTWinv = tf.linalg.inv(VTVplusWTW)
    VTVplusWTW2inv = tf.linalg.inv(VTVplusWTW2)

    if fid >= fid2:
        VTDVplusWTDW = tf.matmul(
            tf.linalg.matrix_transpose(modesU_T, conjugate=True), D_modesU_ev
        ) + tf.matmul(tf.linalg.matrix_transpose(modesV_T, conjugate=True), D_modesV_ev)
        fid_grad = -fid * tf.math.real(
            tf.linalg.trace(tf.matmul(VTVplusWTWinv, VTDVplusWTDW))
        )
    else:
        VTDVplusWTDW = tf.matmul(
            tf.linalg.matrix_transpose(modesU_T2, conjugate=True), D_modesU_ev
        ) + tf.matmul(
            tf.linalg.matrix_transpose(modesV_T2, conjugate=True), D_modesV_ev
        )
        fid_grad = -fid2 * tf.math.real(
            tf.linalg.trace(tf.matmul(VTVplusWTW2inv, VTDVplusWTDW))
        )

    if stage_label == ("I", 1):
        fid_grad = tf.reshape(tf.concat([fid_grad, zero], axis=0), (-1, 1))
    elif stage_label == ("I", 2):
        fid_grad = tf.reshape(fid_grad, (-1, 1))
    elif stage_label == ("II", 1):
        fid_grad = tf.reshape(fid_grad, (-1, 1))
    elif stage_label == ("II", 2):
        fid_grad = tf.reshape(tf.concat([fid_grad, zero], axis=0), (-1, 1))
    elif stage_label == ("III", 1):
        fid_grad = tf.reshape(tf.concat([fid_grad, zero], axis=0), (-1, 1))
    elif stage_label == ("III", 2):
        fid_grad = tf.reshape(fid_grad, (-1, 1))
    elif stage_label == ("IV", 1):
        fid_grad = tf.reshape(fid_grad, (-1, 1))
    elif stage_label == ("IV", 2):
        fid_grad = tf.reshape(fid_grad, (-1, 1))

    return 1.0 - tf.math.maximum(fid, fid2), fid_grad


@tf.function
def infidelity_grad(
    N, EigenmodesInitial, EigenmodesFinal, Ham0, mu0, noise, V0, dedge, s12, dt
):
    (
        s1I,
        s2I,
        s1II,
        s2II,
        s1III,
        s2III,
        s1IV,
        s2IV,
    ) = profiles_from_NN(s12)
    UI = transport_operator(
        N,
        Ham0,
        mu0,
        noise,
        V0,
        dedge,
        tf.concat([zero, s1I, one], -1),
        tf.concat([zero, s2I], -1),
        dt,
        "I",
    )
    UII = transport_operator(
        N, Ham0, mu0, noise, V0, dedge, s1II, tf.concat([s2II, one], -1), dt, "II"
    )
    UIII = transport_operator(
        N, Ham0, mu0, noise, V0, dedge, tf.concat([s1III, one], -1), s2III, dt, "III"
    )
    UIV = transport_operator(
        N,
        Ham0,
        mu0,
        noise,
        V0,
        dedge,
        tf.concat([s1IV, one], -1),
        tf.concat([s2IV, one], -1),
        dt,
        "IV",
    )
    Utot = tf.matmul(UIV, tf.matmul(UIII, tf.matmul(UII, UI)))

    D1_Utot_I, D2_Utot_I = evolution_grad_single_stage(
        N, Ham0, mu0, noise, V0, dedge, s12, dt, UI, UII, UIII, UIV, Utot, stage_label="I"
    )
    D1_Utot_II, D2_Utot_II = evolution_grad_single_stage(
        N, Ham0, mu0, noise, V0, dedge, s12, dt, UI, UII, UIII, UIV, Utot, stage_label="II"
    )
    D1_Utot_III, D2_Utot_III = evolution_grad_single_stage(
        N, Ham0, mu0, noise, V0, dedge, s12, dt, UI, UII, UIII, UIV, Utot, stage_label="III"
    )
    D1_Utot_IV, D2_Utot_IV = evolution_grad_single_stage(
        N, Ham0, mu0, noise, V0, dedge, s12, dt, UI, UII, UIII, UIV, Utot, stage_label="IV"
    )

    fid, fid_grad1_I = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D1_Utot_I, stage_label=("I", 1)
    )
    fid_grad2_I = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D2_Utot_I, stage_label=("I", 2)
    )[1]

    fid_grad1_II = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D1_Utot_II, stage_label=("II", 1)
    )[1]
    fid_grad2_II = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D2_Utot_II, stage_label=("II", 2)
    )[1]

    fid_grad1_III = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D1_Utot_III, stage_label=("III", 1)
    )[1]
    fid_grad2_III = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D2_Utot_III, stage_label=("III", 2)
    )[1]

    fid_grad1_IV = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D1_Utot_IV, stage_label=("IV", 1)
    )[1]
    fid_grad2_IV = infidelity_grad_single_stage(
        N, EigenmodesInitial, EigenmodesFinal, Utot, D2_Utot_IV, stage_label=("IV", 2)
    )[1]

    fid_grad = tf.concat(
        [
            fid_grad1_I,
            fid_grad2_I,
            fid_grad1_II,
            fid_grad2_II,
            fid_grad1_III,
            fid_grad2_III,
            fid_grad1_IV,
            fid_grad2_IV,
        ],
        axis=1,
    )

    return fid, fid_grad


@tf.function
def profiles_from_NN(s12pred):
    s1I = tf.cast(s12pred[:-1, 0], dtype=tf.float64)
    s2I = tf.cast(s12pred[:, 1], dtype=tf.float64)

    s1II = tf.cast(s12pred[:, 2], dtype=tf.float64)
    s2II = tf.cast(s12pred[:-1, 3], dtype=tf.float64)

    s1III = tf.cast(s12pred[:-1, 4], dtype=tf.float64)
    s2III = tf.cast(s12pred[:, 5], dtype=tf.float64)

    s1IV = tf.cast(s12pred[:, 6], dtype=tf.float64)
    s2IV = tf.cast(s12pred[:, 7], dtype=tf.float64)

    return s1I, s2I, s1II, s2II, s1III, s2III, s1IV, s2IV


def plot_profiles(NT, s12list, labels):

	fig, ((ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax8)) = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(12, 6)
    )

	for s12, label in zip(s12list, labels):
		(s1I, s2I, s1II, s2II, s1III, s2III, s1IV, s2IV) = profiles_from_NN(s12)
		
		ax1.plot(range(NT + 1), tf.concat([zero, s1I, one], -1), label = label)
		ax2.plot(range(NT + 1), tf.concat([zero, s2I], -1), label = label)
		ax3.plot(range(NT), s1II, label = label)
		ax4.plot(range(NT), tf.concat([s2II, one], -1), label = label)
		ax5.plot(range(NT), tf.concat([s1III, one], -1), label = label)
		ax6.plot(range(NT), s2III, label = label)
		ax7.plot(range(NT + 1), tf.concat([s1IV, one], -1), label = label)
		ax8.plot(range(NT + 1), tf.concat([s2IV, one], -1), label = label)
	
	ax1.set_title("Stage I, MZM1 ")
	ax2.set_title("Stage I, MZM2 ")
	ax3.set_title("Stage II, MZM1 ")
	ax4.set_title("Stage II, MZM2 ")
	ax5.set_title("Stage III, MZM1 ")
	ax6.set_title("Stage III, MZM2 ")
	ax7.set_title("Stage IV, MZM1 ")
	ax8.set_title("Stage IV, MZM2 ")
	
	ax1.set_box_aspect(1.0)
	ax2.set_box_aspect(1.0)
	ax3.set_box_aspect(1.0)
	ax4.set_box_aspect(1.0)
	ax5.set_box_aspect(1.0)
	ax6.set_box_aspect(1.0)
	ax7.set_box_aspect(1.0)
	ax8.set_box_aspect(1.0)
	
	legend = fig.legend(labels, bbox_to_anchor=(0.97, 0.8))
	fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85)
	
	fig.text(0.5, 0.03, 'timestep', ha='center', fontsize=15)
	fig.text(0.03, 0.71, r'$s_1^{(*)}$', va='center', rotation='vertical', fontsize=15)
	fig.text(0.03, 0.28, r'$s_2^{(*)}$', va='center', rotation='vertical', fontsize=15)
    


def linear_motion_profiles(NT):
    lin = tf.cast(tf.linspace(0.0, 1.0, NT+1), dtype=tf.float64)

    s1I = tf.Variable(lin[1:])
    s2I = tf.Variable(tf.zeros(NT, dtype=tf.float64))

    s1II = tf.Variable(lin[1:])
    s2II = tf.Variable(lin[1:])

    s1III = tf.Variable(lin[:-1])
    s2III = tf.Variable(lin[:-1])

    s1IV = tf.Variable(lin[:-1])
    s2IV = tf.Variable(tf.ones(NT, dtype=tf.float64))
    
    s1I = tf.reshape(s1I, (-1, 1))
    s2I = tf.reshape(s2I, (-1, 1))
    s1II = tf.reshape(s1II, (-1, 1))
    s2II = tf.reshape(s2II, (-1, 1))
    s1III = tf.reshape(s1III, (-1, 1))
    s2III = tf.reshape(s2III, (-1, 1))
    s1IV = tf.reshape(s1IV, (-1, 1))
    s2IV = tf.reshape(s2IV, (-1, 1))
    
    s12 = tf.concat([s1I, s2I, s1II, s2II, s1III, s2III, s1IV, s2IV], axis = 1)

    return s12


def harmonic_motion_profiles(NT):
    rampupdown = tf.math.square(
        tf.math.sin(tfpi * tf.cast(tf.linspace(0.0, 1.0, NT+1), dtype=tf.float64) / 2.0)
    )

    s1I = tf.Variable(rampupdown[1:])
    s2I = tf.Variable(tf.zeros(NT, dtype=tf.float64))

    s1II = tf.Variable(rampupdown[1:])
    s2II = tf.Variable(rampupdown[1:])

    s1III = tf.Variable(rampupdown[:-1])
    s2III = tf.Variable(rampupdown[:-1])

    s1IV = tf.Variable(rampupdown[:-1])
    s2IV = tf.Variable(tf.ones(NT, dtype=tf.float64))
    
    s1I = tf.reshape(s1I, (-1, 1))
    s2I = tf.reshape(s2I, (-1, 1))
    s1II = tf.reshape(s1II, (-1, 1))
    s2II = tf.reshape(s2II, (-1, 1))
    s1III = tf.reshape(s1III, (-1, 1))
    s2III = tf.reshape(s2III, (-1, 1))
    s1IV = tf.reshape(s1IV, (-1, 1))
    s2IV = tf.reshape(s2IV, (-1, 1))
    
    s12 = tf.concat([s1I, s2I, s1II, s2II, s1III, s2III, s1IV, s2IV], axis = 1)

    return s12
