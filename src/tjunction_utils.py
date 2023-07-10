import matplotlib.pyplot as plt
import numpy as np
from math import pi

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

tfpi=tf.constant(pi, dtype=tf.float64)

'''
	POTENTIAL PROFILES OF THE 4 STAGES
	
	s1 and s2 are tensors of shape (k, )
'''

@tf.function
def potential_profile_I(N,dedge, mu0, V0, s1, s2):
	'''
		s1 - particle to the left (0 = far left, 1 = middle)
		s2 - particle to the right (0 = far right, 1 = junction)
	'''
	xR=(1.-s2)*(2.*N-dedge)+s2*N
	xL=dedge+s1*(N-dedge)
	ones=tf.ones(s1.shape[0], tf.float64)
	
	profileH=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*(tf.math.sigmoid(x-xR)+tf.math.sigmoid(xL-x)), elems= tf.range(2*N+1,dtype=tf.float64)) )
	profileV=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*(tf.math.sigmoid(x+ones)), elems= tf.range(N,dtype=tf.float64)) )
	profile=tf.cast( tf.concat( [ profileH, profileV ], 1), dtype=tf.complex128 )

	return tf.concat([profile, -profile ], 1 )

@tf.function	
def potential_profile_II(N, dedge, mu0, V0, s1, s2):
	'''
		s1 - particle on the vertical chain (0 = junction, 1 = bottom)
		s2 - particle to the right (0 = far right, 1 = junction)
	'''
	xR=(1.-s2)*(2.*N-dedge)+s2*N
	xL=s1*dedge+(1.-s1)*N
	Nones=N*tf.ones(s1.shape[0], tf.float64)
	
	profile1=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*(tf.math.sigmoid(x-xR)+tf.math.sigmoid(xL-x)), elems= tf.range(2*N+1,dtype=tf.float64)) )
	profileHL=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*tf.math.sigmoid(Nones-x), elems= tf.range(N,dtype=tf.float64)) )
	profile=tf.concat( [ profileHL, profile1[:,N:], tf.reverse(profile1[:,:N], [-1]) ], 1)
	
	return tf.cast( tf.concat([profile, -profile ], 1 ), dtype=tf.complex128 )

@tf.function	
def potential_profile_III(N,dedge, mu0, V0, s1, s2):
	'''
		s1 - particle on the vertical chain (0 = bottom, 1 = junction)
		s2 - particle to the left (0 = junction, 1 = far left)
	'''
	xL=(1.-s1)*dedge+s1*N
	xR=s2*(2.*N-dedge)+(1.-s2)*N
	ones=tf.ones(s1.shape[0], tf.float64)
	
	profile1=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*(tf.math.sigmoid(x-xR)+tf.math.sigmoid(xL-x)), elems= tf.range(2*N+1,dtype=tf.float64)) )
	profileHR=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*tf.math.sigmoid(x+ones), elems= tf.range(N,dtype=tf.float64)) )
	profile=tf.concat( [ tf.reverse(profile1[:,N:], [-1]), profileHR, tf.reverse(profile1[:,:N], [-1]) ], 1)
	
	return tf.cast( tf.concat([profile, -profile ], 1 ), dtype=tf.complex128 )

@tf.function	
def potential_profile_IV(N, dedge, mu0, V0, s1, s2):
	'''
		s1 - particle to the right (0 = junction, 1 = far right)
		s2 - particle to the left (0 = junction, 1 = far left)
	'''
	xL=(1.-s2)*N+s2*dedge
	xR=s1*(2.*N-dedge)+(1.-s1)*N
	ones=tf.ones(s1.shape[0], tf.float64)
	
	profileH=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*(tf.math.sigmoid(x-xR)+tf.math.sigmoid(xL-x)), elems= tf.range(2*N+1,dtype=tf.float64)) )
	profileV=tf.transpose( tf.map_fn(fn=lambda x:mu0+V0*(tf.math.sigmoid(x+ones)), elems= tf.range(N,dtype=tf.float64)) )
	profile=tf.concat( [ profileH, profileV ], 1)

	return tf.cast( tf.concat([profile, -profile ], 1 ), dtype=tf.complex128 )

	
def plot_modes(N,mode1,mode2):
	'''
		A 4-panel plot of the two given modes. The left plots show abs(mode1) while the
		right plots show abs(mode2).
		-- horizontal sites have indices 0 - 2N from left to right
		-- vertical sites have indices 2N+1 - 3N from top to bottom
	'''
	
	fig,((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True)
	ax1.plot(range(0,2*N+1), tf.abs(mode1[:2*N+1]))
	ax1.set_title("Horizontal "+str(2*N+1)+" sites")
	ax2.plot(range(0,N), tf.abs(mode1[2*N+1:3*N+1]))
	ax2.set_title("Vertical "+str(N)+" sites")
	ax3.plot(range(0,2*N+1), tf.abs(mode2[:2*N+1]))
	ax3.set_title("Horizontal "+str(2*N+1)+" sites")
	ax4.plot(range(0,N), tf.abs(mode2[2*N+1:3*N+1]))
	ax4.set_title("Vertical "+str(N)+" sites")
	
@tf.function	
def construct_hamiltonian_zero_diag(N,Delta,w,V0,dedge):
	'''
		Constructs Kitaev hamiltonian of the trijunction with zero on-site potentials.
		The horizontal bar/chain has a real superconducting gap. The vertical bar has 
		a purely imaginary superconducting gap of the same amplitude.
		
		The hamiltonian of a single chain consists of four tridiagonal NxN blocks.
	'''
	
	'''
		HORIZONTAL BAR
	'''

	diagDeltaH=tf.experimental.numpy.full(2*N+1,Delta/2.,dtype=tf.complex128)
	diagWH=tf.experimental.numpy.full(2*N+1,-w/2.,dtype=tf.complex128)
	zerosNH=tf.experimental.numpy.full(2*N+1,0.,dtype=tf.complex128)
	blockDeltaH=tf.linalg.LinearOperatorTridiag([-diagDeltaH, zerosNH, diagDeltaH],diagonals_format='sequence')
	diagWH=tf.experimental.numpy.full(2*N+1,-w/2.,dtype=tf.complex128)
	blockMuWH=tf.linalg.LinearOperatorTridiag([diagWH, zerosNH, diagWH],diagonals_format='sequence')
	blockMuWH=blockMuWH.to_dense()
	blockDeltaH=blockDeltaH.to_dense()
	
	'''
		embed the blocks into the horizontal chain hamiltonian
	'''

	zeros2Np1xN=tf.zeros((2*N+1,N),dtype=tf.complex128)
	zerosNx6Np2=tf.zeros((N,6*N+2),dtype=tf.complex128)
	KitaevHHupper=tf.concat([tf.concat([blockMuWH, zeros2Np1xN, blockDeltaH, zeros2Np1xN], 1 ), zerosNx6Np2], 0 )
	KitaevHHlower=tf.concat([tf.concat([-blockDeltaH, zeros2Np1xN, -blockMuWH, zeros2Np1xN], 1), zerosNx6Np2], 0 )
	KitaevHH=tf.concat([KitaevHHupper,KitaevHHlower], 0 )

	'''
		VERTICAL BAR
		
		purely imaginary Delta
	'''

	diagIDelta=tf.experimental.numpy.full(N,tf.complex(0.,1.)*Delta/2.,dtype=tf.complex128)
	diagW=tf.experimental.numpy.full(N,-w/2.,dtype=tf.complex128)
	zerosN=tf.experimental.numpy.full(N,0.,dtype=tf.complex128)
	blockIDelta=tf.linalg.LinearOperatorTridiag([-diagIDelta, zerosN, diagIDelta],diagonals_format='sequence')
	diagW=tf.experimental.numpy.full(N,-w/2.,dtype=tf.complex128)
	blockMuW=tf.linalg.LinearOperatorTridiag([diagW, zerosN, diagW],diagonals_format='sequence')
	blockMuW=blockMuW.to_dense()
	blockIDelta=blockIDelta.to_dense()

	'''
		embed the blocks into the vertical chain hamiltonian
	'''

	zerosNx2Np1=tf.zeros((N,2*N+1),dtype=tf.complex128)
	zeros2Np1x6Np2=tf.zeros((2*N+1,6*N+2),dtype=tf.complex128)
	KitaevHVupper=tf.concat([zeros2Np1x6Np2,tf.concat([zerosNx2Np1,blockMuW,zerosNx2Np1, -blockIDelta], 1 )], 0 )
	KitaevHVlower=tf.concat([zeros2Np1x6Np2,tf.concat([zerosNx2Np1,-blockIDelta,zerosNx2Np1, -blockMuW], 1)], 0 )
	KitaevHV=tf.concat([KitaevHVupper,KitaevHVlower], 0 )
	
	'''
		HAMILTONIAN OF THE TWO BARS
		
		horizontal sites have indices 0 - 2N from left to right
		vertical sites have indices 2N+1 - 3N from top to bottom
	'''
	KitaevH=tf.add(KitaevHH,KitaevHV)

	#coupling between the bars
	CouplingIndices=[[N,2*N+1],[2*N+1,N],[5*N+2,4*N+1],[4*N+1,5*N+2],[4*N+1,2*N+1],[2*N+1,4*N+1],[5*N+2,N],[N,5*N+2]]
	IDelta=tf.cast(tf.cast(tf.complex(0.,1.),tf.complex128)*Delta/2,dtype=tf.complex128)
	Couplings=[-w/2,-w/2,w/2,w/2,IDelta,-IDelta,-IDelta,IDelta]
	
	return tf.tensor_scatter_nd_update(KitaevH, CouplingIndices, Couplings)
	
@tf.function	
def transport_operator_map(N,Ham0,mu0,V0,dedge,s1Vec,s2Vec,dt,stage):
	'''
		generates the adiabatic evolution operator
		
		can be potentially improved by using a more accurate version of the
		Suzuki-Trotter formula
	'''
	
	Idt=-dt*tf.cast(tf.complex(0.,1.),dtype=tf.complex128)
	
	if stage=='I':
		PotentialProfiles=potential_profile_I(N, dedge, mu0, V0, s1Vec, s2Vec)
	elif stage=='II':
		PotentialProfiles=potential_profile_II(N, dedge, mu0, V0, s1Vec, s2Vec)
	elif stage=='III':
		PotentialProfiles=potential_profile_III(N, dedge, mu0, V0, s1Vec, s2Vec)
	elif stage=='IV':
		PotentialProfiles=potential_profile_IV(N, dedge, mu0, V0, s1Vec, s2Vec)
	
	'''
		evolution calculated via map
	'''
	HamExps=tf.map_fn(fn=lambda MuList: tf.linalg.expm(tf.math.scalar_mul(Idt,tf.tensor_scatter_nd_update( Ham0, [[i,i] for i in range(6*N+2)], MuList ))), 
	elems=PotentialProfiles)
	EvolutionOperator=tf.foldr(tf.matmul,HamExps)
	del HamExps
	
	return EvolutionOperator
	
@tf.function	
def transport_operator_loop(N,Ham0,mu0,V0,dedge,s1Vec,s2Vec,dt,stage,ParallelIterations):
	'''
		generates the adiabatic evolution operator
		
		can be potentially improved by using a more accurate version of the
		Suzuki-Trotter formula
	'''
	
	Idt=-dt*tf.cast(tf.complex(0.,1.),dtype=tf.complex128)
	
	if stage=='I':
		PotentialProfiles=potential_profile_I(N, dedge, mu0, V0, s1Vec, s2Vec)
	elif stage=='II':
		PotentialProfiles=potential_profile_II(N, dedge, mu0, V0, s1Vec, s2Vec)
	elif stage=='III':
		PotentialProfiles=potential_profile_III(N, dedge, mu0, V0, s1Vec, s2Vec)
	elif stage=='IV':
		PotentialProfiles=potential_profile_IV(N, dedge, mu0, V0, s1Vec, s2Vec)
	
	'''
		evolution calculated via loop - memory saving
	'''
	Nsteps=s1Vec.shape[0]
	c = lambda istep,U: tf.less(istep, Nsteps)
	U=tf.linalg.expm( tf.math.scalar_mul(Idt,tf.tensor_scatter_nd_update( Ham0, [[i,i] for i in range(6*N+2)], PotentialProfiles[0] )) )
	dtEvolution=lambda istep, U: (tf.add(istep, 1), tf.linalg.matmul(tf.linalg.expm(tf.math.scalar_mul(Idt,tf.tensor_scatter_nd_update( Ham0, [[i,i] for i in range(6*N+2)], PotentialProfiles[istep] ))),U), )
	istep=tf.constant(1)
	
	return tf.while_loop(c, dtEvolution, [istep,U], parallel_iterations=ParallelIterations)[1]
	

@tf.function	
def infidelity(N, EigenmodesInitial, EigenmodesFinal, EvolutionOperator):
	# auxilliary permutation matrices P and P1 to be used for exchanging zero modes
	P1 = tf.cast(tf.linalg.diag(tf.concat([tf.experimental.numpy.zeros((1)),tf.experimental.numpy.ones((3*N))],-1)),dtype=tf.complex128)
	P = tf.concat([tf.concat([P1,tf.eye(3*N+1,dtype=tf.complex128)-P1],1),tf.concat([tf.eye(3*N+1,dtype=tf.complex128)-P1,P1],1)],0)
	
	modesU_T=EigenmodesFinal[0:3*N+1,3*N+1:6*N+2]
	modesV_T=EigenmodesFinal[3*N+1:6*N+2,3*N+1:6*N+2]

	#switch zero modes
	W_T = tf.concat([tf.concat([modesU_T,tf.math.conj(modesV_T)], 1),tf.concat([modesV_T,tf.math.conj(modesU_T)], 1)],0)
	w_T = tf.matmul(W_T,P)
	modesU_T2 = w_T[0:3*N+1,0:3*N+1]
	modesV_T2  = w_T[3*N+1:6*N+2,0:3*N+1]
	
	ModesEv=tf.matmul(EvolutionOperator,EigenmodesInitial)
	modesU_ev = ModesEv[0:3*N+1,3*N+1:6*N+2]
	modesV_ev  = ModesEv[3*N+1:6*N+2,3*N+1:6*N+2]
		
	Overlap1=tf.math.abs(tf.math.reduce_prod(tf.linalg.eigvals(tf.matmul(tf.linalg.matrix_transpose(modesU_T,conjugate=True),modesU_ev)+tf.matmul(tf.linalg.matrix_transpose(modesV_T,conjugate=True),modesV_ev))))
	Overlap2=tf.math.abs(tf.math.reduce_prod(tf.linalg.eigvals(tf.matmul(tf.linalg.matrix_transpose(modesU_T2,conjugate=True),modesU_ev)+tf.matmul(tf.linalg.matrix_transpose(modesV_T2,conjugate=True),modesV_ev))))
	
	return 1.-tf.math.maximum(Overlap1,Overlap2)

@tf.function
def profiles_from_NN(s12pred):
	s1I=tf.cast( s12pred[:-1,0], dtype=tf.float64 )
	s2I=tf.cast( s12pred[:,1], dtype=tf.float64 )

	s1II=tf.cast( s12pred[:,2], dtype=tf.float64 )
	s2II=tf.cast( s12pred[:-1,3], dtype=tf.float64 )

	s1III=tf.cast( s12pred[:-1,4], dtype=tf.float64 )
	s2III=tf.cast( s12pred[:,5], dtype=tf.float64 )

	s1IV=tf.cast( s12pred[:,6], dtype=tf.float64 )
	s2IV=tf.cast( s12pred[:,7], dtype=tf.float64 )
	
	return s1I,s2I,s1II,s2II,s1III,s2III,s1IV,s2IV
	
	
def plot_profiles(NT,s1I,s2I,s1II,s2II,s1III,s2III,s1IV,s2IV):
	zero=tf.constant([0.],dtype=tf.float64)
	one=tf.constant([1.],dtype=tf.float64)
	
	fig,( (ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax8) ) = plt.subplots(2, 4, sharex=True, sharey=True)
	ax1.plot(range(NT), tf.concat([zero,s1I,one],-1) )
	ax1.set_title( "Stage I, MZM1 " )
	ax1.set_box_aspect(1.)
	ax2.plot(range(NT), tf.concat([zero,s2I],-1) )
	ax2.set_title( "Stage I, MZM2 " )
	ax2.set_box_aspect(1.)
	
	ax3.plot(range(NT-1), s1II )
	ax3.set_title( "Stage II, MZM1 " )
	ax3.set_box_aspect(1.)
	ax4.plot(range(NT-1), tf.concat([s2II,one],-1) )
	ax4.set_title( "Stage II, MZM2 " )
	ax4.set_box_aspect(1.)
	
	ax5.plot( range(NT-1), tf.concat([s1III,one],-1) )
	ax5.set_title( "Stage III, MZM1 " )
	ax5.set_box_aspect(1.)
	ax6.plot( range(NT-1), s2III )
	ax6.set_title( "Stage III, MZM2 " )
	ax6.set_box_aspect(1.)
	
	ax7.plot( range(NT), tf.concat([s1IV,one],-1) )
	ax7.set_title( "Stage IV, MZM1 " )
	ax7.set_box_aspect(1.)
	ax8.plot( range(NT), tf.concat([s2IV,one],-1) )
	ax8.set_title( "Stage IV, MZM2 " )
	ax8.set_box_aspect(1.)
	
	
def linear_motion_profiles(NT):
	s1I=tf.Variable( tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)[1:-1] )
	s2I=tf.Variable( tf.zeros(NT,dtype=tf.float64)[1:] )

	s1II=tf.Variable( tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)[1:] )
	s2II=tf.Variable( tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)[1:-1] )

	s1III=tf.Variable( tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)[1:-1] )
	s2III=tf.Variable( tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)[1:] )

	s1IV=tf.Variable( tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)[1:-1] )
	s2IV=tf.Variable( tf.ones(NT,dtype=tf.float64)[1:-1] )
	
	return s1I,s2I,s1II,s2II,s1III,s2III,s1IV,s2IV
	

def sinusoidal_motion_profiles(NT):
	rampup=2.*tf.math.square(tf.math.sin( tfpi*tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)/4. ) )
	rampdown=tf.math.sin( tfpi*tf.cast(tf.linspace(0.,1.,NT),dtype=tf.float64)/2. )
	
	s1I=tf.Variable( rampup[1:-1] )
	s2I=tf.Variable( tf.zeros(NT,dtype=tf.float64)[1:] )
	
	s1II=tf.Variable( rampdown[1:] )
	s2II=tf.Variable( rampup[1:-1] )
	
	s1III=tf.Variable( rampup[1:-1] )
	s2III=tf.Variable( rampdown[1:] )
	
	s1IV=tf.Variable( rampdown[1:] )
	s2IV=tf.Variable( tf.ones(NT,dtype=tf.float64)[1:] )
	
	return s1I,s2I,s1II,s2II,s1III,s2III,s1IV,s2IV
		