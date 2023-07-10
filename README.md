# trijunction_mzm_braiding
Simulation of the topological quantum gate realised by exchanging two Majorana zero modes on a network of coupled p-wave quantum nanowires. Optimisation of the exchange protocol using a neural net.

## Physics Background and Motivation

Some of the leading proposals for the realisation of a topological quantum computer utilise networks of 1D quantum nanowires. Each nanowire is modelled by a so-called p-wave Hamiltonian which may become a superconductor when Hamiltonian's parameters are chosen appropriately (for more theory see the seminal paper by A. Kitaev - [arXiv:cond-mat/0010440](https://arxiv.org/abs/cond-mat/0010440)

In the superconducting regime, there exist two Majorana zero modes (MZMs) localised at the endpoints of the topological region in the nanowire. Crucially, the MZMs can be moved around the nanowire network only by tuning local voltages along the wires -- see the work by J. Alicea et al.[ Nature Physics vol 7, pp 412–417 (2011)](https://www.nature.com/articles/nphys1915). Building up on the ideas from this paper, we simulate the process of the exchange of MZMs on a trijunction as shown in the picture below.

![Image](https://github.com/tmaciazek/trijunction_mzm_braiding/blob/main/Texchange.png)
