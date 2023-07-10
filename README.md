# trijunction_mzm_braiding
Simulation of the topological quantum gate realised by exchanging two Majorana zero modes on a network of coupled p-wave quantum nanowires. Optimisation of the exchange protocol using a neural net.

## Folders and files in this repo


## Physics background and motivation

Some of the leading proposals for the realisation of a topological quantum computer utilise networks of 1D quantum nanowires. Each nanowire is modelled by a so-called p-wave Hamiltonian and may become superconducting when Hamiltonian's parameters are chosen appropriately (for more theory see the seminal paper by A. Kitaev - [arXiv:cond-mat/0010440](https://arxiv.org/abs/cond-mat/0010440)).

In the superconducting regime, there exist two Majorana zero modes (MZMs) localised at the endpoints of the topological region in the nanowire. Crucially, the MZMs can be moved around the nanowire network only by tuning local voltages along the wires -- see the work by J. Alicea et al.[ Nature Physics vol 7, pp 412–417 (2011)](https://www.nature.com/articles/nphys1915). Building up on the ideas from this paper, we simulate the process of the exchange of MZMs on a trijunction as shown in the picture below.

![Image](https://github.com/tmaciazek/trijunction_mzm_braiding/blob/main/pictures/Texchange.png)

Simulating the exchange process on a computer means simulating the time evolution of the respective quantum system. The total time, $T$, of the evolution is fixed and the evolution is discretised in $N_T$ time-steps. Abstractly, in order to find the evolution operator of time-dependent Hamiltonian $H(t)$, we use the Suzuki-Trotter formula

$$ U_{ev}=\prod_{k=1}^{N_T} e^{-i H(k \Delta t) \Delta t}, \quad \Delta t=T/N_T. $$

After computing the evolution operator for the system of two MZMs we evaluate the **fidelity**, $\mathcal{F}$, via the Onishi formula (for more details, see the paper by L. Coopmans et al. [PRX Quantum 2, 020332 (2021)](https://doi.org/10.48550/arXiv.2008.09128)).

Note, that in the adiabatic case we would have $\mathcal{F}=1$. However, because the total time $T$ is finite, we necessarily have $\mathcal{F}<1$.

## Optimisation of the fidelity via a neural net

In order to more conveniently parametrise the positions of the MZMs during the exchange process, we divide the exchange into four stages as shown below.

<img src="https://github.com/tmaciazek/trijunction_mzm_braiding/blob/main/pictures/stages.png" width="600" height="264">

The positions of the MZMs in each stage (I - IV) are described by vectors $\mathbf{s}_1$ and $\mathbf{s}_2$, i.e. the $k$-th element of the vector $\mathbf{s}_j$ determines the position of MZM with label $j$ at timestep $k$ of the given stage. The positions of the MZMs during the exchange determine the exact time dependence of the Hamiltonian and consequently allow us to compute the fidelity via the Suzuki-Trotter and Onishi formulas.

The neural net architecture that has been used for optimising the fidelity is shown below.

<img src="https://github.com/tmaciazek/trijunction_mzm_braiding/blob/main/pictures/nn.png" width="600" height="337">

This way, the positions of the MZMs are functions of the neural net weights. The fidelity is optimised via gradient descent - the automatic differentiation is applied to find the partial derivatives of the fidelity with respect to the neural net weights.

The neural net is first pretrained so that it outputs a motion where the positions of MZMs are described by appropriately scaled sin- and cos- functions of time. Then, we train the neural net for 150 epochs (gradient steps) which results with dropping the infidelity by three orders of magnitude from $0.19$ to $0.00017$. The resulting trajectory is plotted in figure below. 

<img src="https://github.com/tmaciazek/trijunction_mzm_braiding/blob/main/pictures/trained.png" width="600" height="460">

The animations below show the resulting transport protocol together with the corresponding time evolution of the wavefunction.

MZM transport protocol             |  Wavefunction
:-------------------------:|:-------------------------:
![Image](https://github.com/tmaciazek/trijunction_mzm_braiding/blob/main/pictures/dots.gif)  |  ![Image](https://github.com/tmaciazek/trijunction_mzm_braiding/blob/main/pictures/colormap.gif)

Thanks to this simulation we have gained crucial insights into the properties of optimal MZM transport protocols. Namely, the MZMs have to slowly ramp-up and ramp-down their velocities and stop at the junction.

