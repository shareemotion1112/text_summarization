Recent theoretical and experimental results suggest the possibility of using current and near-future quantum hardware in challenging sampling tasks.

In this paper, we introduce free-energy-based reinforcement learning (FERL) as an application of quantum hardware.

We propose a method for processing a quantum annealer’s measured qubit spin configurations in approximating the free energy of a quantum Boltzmann machine (QBM).

We then apply this method to perform reinforcement learning on the grid-world problem using the D-Wave 2000Q quantum annealer.

The experimental results show that our technique is a promising method for harnessing the power of quantum sampling in reinforcement learning tasks.

Reinforcement learning (RL) BID33 ; BID6 has been successfully applied in fields such as engineering BID11 ; BID35 , sociology BID12 ; BID30 , and economics BID22 ; BID31 .

The training samples in reinforcement learning are provided by the interaction of an agent with an ambient environment.

For example, in a motion planning problem in uncharted territory, it is desirable for the agent to learn to correctly navigate in the fastest way possible, making the fewest blind decisions.

That is, neither exploration nor exploitation can be pursued exclusively without either facing a penalty or failing at the task.

Our goal is, therefore, not only to design an algorithm that eventually converges to an optimal policy, but for the algorithm to be able to generate suboptimal policies early in the learning process.

Free-energy-based reinforcement learning (FERL) using a restricted Boltzmann machine (RBM), as suggested by BID27 , relies on approximating a utility function for the agent, called the Q-function, using the free energy of an RBM.

RBMs have the advantage that their free energy can be efficiently calculated using closed formulae.

RBMs can represent any joint distribution over binary variables BID20 ; BID15 ; Le BID19 ; however, this property of universality may require exponentially large RBMs BID20 ; Le BID19 .General Boltzmann machines (GBM) are proposed in an effort to devise universal Q-function approximators with polynomially large Boltzmann networks BID10 .

Traditionally, Monte Carlo simulation is used to perform the computationally expensive tasks of approximating the free energy of GBMs under a Boltzmann distribution.

One way to speed up the approximation process is to represent a GBM by an equivalent physical system and try to find its Boltzmann distribution.

An example of such a physical system is a quantum annealer consisting of a network of pair-wise interacting quantum bits (qubits).

Although quantum annealers have already been used in many areas of computational science, including combinatorial optimization and machine learning, their application in RL has not been explored.

In order to use quantum annealing for RL, we first represent the Q-function as the free energy of a physical system, that is, that of a quantum annealer.

We then slowly evolve the state of the physical system from a well-known initial state toward a state with a Boltzmann-like probability distribution.

Repeating the annealing process sufficiently long can provide us with samples from the Boltzmann distribution so that we can empirically approximate the free energy of the physical system under this distribution.

Finally, approximating the free energy of the system would give us an estimate of the Q-function.

Up until the past few years, studies were limited to the classical Boltzmann machines.

1 Recently, BID10 generalized the classical method toward a quantum or quantum-inspired algorithm for approximating the free energy of GBMs.

Using simulated quantum annealing (SQA) BID10 showed that FERL using a deep Boltzmann machine (DBM) can provide a drastic improvement in the early stages of learning, yet performing the same procedure on an actual quantum device remained a difficult task.

This is because sampling from a quantum system representing a quantum Boltzmann machine is harder than the classical case, since at the end of each anneal the quantum system is in a superposition.

Any attempt to measure the final state of the quantum system is doomed to fail since the superposition would collapse into a classical state that does not carry the entirety of information about the final state.

In this work, we have two main contributions.

We first employ a quantum annealer as a physical device to approximate the free energy of a classical Boltzmann machine.

Second, we generalize the notion of classical Boltzmann machines to quantum Boltzmann machines within the field of RL and utilize a quantum annealer to approximate the free energy of a quantum system.

In order to deal with the issue of superposition mentioned above, we propose a novel stacking procedure in that we attempt to reconstruct the full state of superposition from the partial information that we get from sampling after each anneal.

Finally we report proof-of-concept results using the D-Wave 2000Q quantum processor to provide experimental evidence for the applicability of a quantum annealer in reinforcement learning as predicted by BID10 .

We refer the reader to BID33 and BID37 for an exposition on Markov decision processes (MDP), controlled Markov chains, and the various broad aspects of reinforcement learning.

A Q-function is defined by mapping a tuple pπ, s, aq of a given stationary policy π, a current state s, and an immediate action a of a controlled Markov chain to the expected value of the instantaneous and future discounted rewards of the Markov chain that begins with taking action a at initial state s and continuing according to π: Qpπ, s, aq " Err ps, aqs`E DISPLAYFORM0 Here, rps, aq is a random variable, perceived by the agent from the environment, representing the immediate reward of taking action a from state s, and Π is the Markov chain resulting from restricting the controlled Markov chain to the policy π.

The fixed real number γ P p0, 1q is the discount factor of the MDP.

From Q˚ps, aq " max π Qpπ, s, aq, the optimal policy for the MDP can be retrieved via π˚psq " argmax a Q˚ps, aq.

This reduces the MDP task to that of computing Q˚ps, aq.

Through the Bellman optimality equation BID4 , we get Q˚ps, aq " Err ps, aqs`γ ÿ DISPLAYFORM1 so Q˚is the fixed point of the following operator defined on L 8 pSˆAq:T pQq : ps, aq Þ Ñ Err ps, aqs`γ ż max a 1

In this paper, we focus on the TD(0) Q-learning method, with the Q-function parametrized by neural networks in order to find π˚psq and Q˚ps, aq, which is based on minimizing the distance between T pQq and Q.

A clamped Boltzmann machine is a GBM in which all visible nodes v are prescribed fixed assignments and removed from the underlying graph.

Therefore, the energy of the clamped Boltzmann machine may be written as DISPLAYFORM0 where V and H are the sets of visible and hidden nodes, respectively, and by a slight abuse of notation, the letter v stands both for a graph node v P V and for the assignment v P t0, 1u.

The interactions between the variables represented by their respective nodes are specified by real-valued weighted edges of the underlying undirected graph represented by w vh , and w hh 1 denotes the weights between visible and hidden, or hidden and hidden, nodes of the Boltzmann machine, respectively.

A clamped quantum Boltzmann machine (QBM) has the same underlying graph as a clamped GBM, but instead of a binary random variable, qubits are associated to each node of the network.

The energy function is substituted by the quantum Hamiltonian of an induced transverse field Ising model (TFIM), which is mathematically a Hermitian matrix DISPLAYFORM1 where σ z h represent the Pauli z-matrices and σ x h represent the Pauli x-matrices.

Thus, a clamped QBM with Γ " 0 is equivalent to a clamped classical Boltzmann machine.

This is because, in this case, H v is a diagonal matrix in the σ z -basis, the spectrum of which is identical to the range of the classical Hamiltonian (3).

We note that (4) is a particular instance of a TFIM.

Let us begin with the classical Boltzmann machine case.

Following BID27 , for an assignment of visible variables v, F pvq denotes the equilibrium free energy, and is given via where β " 1 k B T is a fixed thermodynamic beta.

In BID27 , it was proposed to use the negative free energy of a GBM to approximate the Q-function through the relationship Qps, aq «´F ps, aq "´F ps, a; wq for each admissible state-action pair ps, aq P SˆA. Here, s and a are binary vectors encoding the state s and action a on the state nodes and action nodes, respectively, of a GBM.

In RL, the visible nodes of a GBM are partitioned into two subsets of state nodes S and action nodes A. Here, w represents the vector of weights of a GBM as in (3).

Each entry w of w can now be trained using the TD(0) update rule: DISPLAYFORM0 ∆w vh " εpr n ps n , a n q`γQps n`1 , a n`1 q´Qps n , a n qqvxhy and (6) ∆w hh 1 " εpr n ps n , a n q`γQps n`1 , a n`1 q´Qps n , a n qqxhh 1 y ,where xhy and xhh 1 y are the expected values of the variables and the products of the variables, respectively, in the binary encoding of the hidden nodes with respect to the Boltzmann distribution of the classical Hamiltonian (3).

1 k B T be a fixed thermodynamic beta as in the classical case.

As before, for an assignment of visible variables v, F pvq denotes the equilibrium free energy, and is given via DISPLAYFORM1 Here, Z v " trpe´β Hv q is the partition function of the clamped QBM and ρ v is the density matrix DISPLAYFORM2 Hv .

The term´trpρ v ln ρ v q is the entropy of the system.

Note that FORMULA7 is a generalization of (5).

The notation x¨¨¨y is used for the expected value of any observable with respect to the Gibbs measure (i.e., the Boltzmann distribution), in particular, DISPLAYFORM3 This is also a generalization of the weighted sum ř h Pph|vqE v phq in (5).

Inspired by the ideas of BID27 and BID1 , we use the negative free energy of a QBM to approximate the Q-function exactly as in the classical case:Qps, aq «´F ps, a; wq for each admissible state-action pair ps, aq P SˆA. As before, s and a are binary vectors encoding the state s and action a on the state nodes and action nodes, respectively, of a Boltzmann machine.

In RL, the visible nodes of a Boltzmann machine are partitioned into two subsets of state nodes S and action nodes A. Here, w represents the vector of weights of a QBM as in (4).

Each entry w of w can now be trained using the TD(0) update rule:∆w "´εpr n ps n , a n q´γF ps n`1 , a n`1 q`F ps n , a n qq BF Bw .As shown in BID10 , from (8) we obtain ∆w vh " εpr n ps n , a n q (9) γF ps n`1 , a n`1 q`F ps n , a n qqvxσ z h y and ∆w hh 1 " εpr n ps n , a n q (10) γF ps n`1 , a n`1 q`F ps n , a n qqxσ z h σ z h 1 y. This concludes the development of the FERL method using QBMs.

We refer the reader to Algorithm 3 in BID10 for more details.

What remains to be done is to approximate values of the free energy F ps, aq and also the expected values of the observables xσ z h y and xσ z h σ z h 1 y. In this paper, we demonstrate how quantum annealing can be used to address this challenge.

The evolution of a quantum system under a slowly changing, time-dependent Hamiltonian is characterized by BID7 .

The quantum adiabatic theorem (QAT) in BID7 states that a system remains in its instantaneous steady state, provided there is a gap between the eigen-energy of the steady state and the rest of the Hamiltonian's spectrum at every point in time.

QAT motivated BID13 to introduce a paradigm of quantum computing known as quantum adiabatic computation which is closely related to the quantum analogue of simulated annealing, namely quantum annealing (QA), introduced by BID18 .The history of QA and QAT inspired manufacturing efforts towards physical realizations of adiabatic evolution via quantum hardware BID17 .

In reality, the manufactured chips are operated at a non-zero temperature and are not isolated from their environment.

Therefore, the existing adiabatic theory does not cover the behaviour of these machines.

A contemporary investigation in quantum adiabatic theory was therefore initiated to study adiabaticity in open quantum systems BID28 ; BID36 ; Albash et al. FORMULA1 ; BID2 ; BID3 .

These sources prove adiabatic theorems for open quantum systems under various assumptions, in particular when the quantum system is coupled to a thermal bath satisfying the Kubo-Martin-Schwinger condition, implying that the instantaneous steady state is the instantaneous Gibbs state.

This work in progress shows promising opportunities to use quantum annealers to sample from the Gibbs state of a TFIM.

In practice, due to additional complications (e.g., level crossings and gap closure, described in the references above), the samples gathered from the quantum annealer are far from the Gibbs state of the final Hamiltonian.

In fact, BID0 suggests that the distribution of the samples would instead correspond to an instantaneous Hamiltonian at an intermediate point in time, called the freeze-out point.

Unfortunately, this point and, consequently, the strength Γ of the transverse field at this point, is not known a priori, and also depends on the TFIM undergoing evolution.

Our goal is simply to associate a single (average) virual Γ to all TFIMs constructed through FERL.

Another unknown parameter is the inverse temperature β, at which the Gibbs state, the partition function, and the free energy are attained.

In a similar fashion, we wish to associate a single virtual β to all TFIMs encountered.

The quantum annealer used in our experiments is the D-Wave 2000Q, which consists of a chip of superconducting qubits connected to each other according to a sparse adjacency graph called the Chimera graph.

The Chimera graph structure looks significantly different from the frequently used models in machine learning, for example, RBMs and DBMs, which consist of consecutive fully connected bipartite graphs.

FIG0 shows two adjacent blocks of the Chimera graph which consist of 16 qubits, which, in this paper, serve as the clamped QBM used in FERL.Another complication when using a quantum annealer as a QBM is that the spin configurations of the qubits can only be measured along a fixed axis (here the z-basis of the Bloch sphere).

Once σ z is measured, all of the quantum information related to the projection of the spin along the transverse field (i.e., the spin σ x ) collapses and cannot be retrieved.

Therefore, even with a choice of virtual Γ, virtual β, and all of the measured configurations, the energy of the TFIM is still unknown.

We propose a method for overcoming this challenge based on the Suzuki-Trotter expansion of the TFIM, which we call replica stacking, the details of which are explained in §3.4.

In §4, we perform a grid search over values of the virtual parameters β and Γ. The accepted virtual parameters are the ones that result in the most-effective learning for FERL in the early stages of training.

By the Suzuki-Trotter decomposition BID34 , the partition function of the TFIM defined by the Hamiltonian (4) can be approximated using the partition function of a classical Hamiltonian denoted by H eff v and called an effective Hamiltonian, which corresponds to a classical Ising model of one dimension higher.

More precisely, where r is the number of replicas, w`" 1 2β log coth´Γ β r¯, and h k represent spins of the classical system of one dimension higher.

Note that each hidden node's Pauli z-matrices σ z h are represented by r classical spins, denoted by h k , with a slight abuse of notation.

In other words, the original Ising model with a non-zero transverse field represented through non-commuting operators can be mapped to a classical Ising model of one dimension higher.

FIG1 shows the underlying graph of a TFIM on a two-dimensional lattice and a corresponding 10-replica effective Hamiltonian in three dimensions.

DISPLAYFORM0 The intuition behind the Suzuki-Trotter decomposition is that the superposition of the spins in a quantum system is represented classically by replicas in the z-basis.

In other words, the measurement of the quantum system in the z-basis is interpreted as choosing one replica at random.

Note that the probabilities of measuring`1 or´1 for each individual spin are preserved.

This way, each hidden node in the quantum Boltzmann machine carries more information than a classical one; in fact, a classical representation of this system requires r classical binary units via the Suzuki-Trotter decomposition.

Consequently, the connections between the hidden nodes become more complicated in the quantum case as well and can carry more information on the correlations between the hidden nodes.

Note that the coupling strengths between the replicas are not arbitrary, but come from the mathematical decomposition following the Suzuki-Trotter formula.

As a result, the quantum Boltzmann machine can be viewed as an undirected graphical model but in one dimension higher than the classical Boltzmann machine.

In the case of classical GBMs without further restrictions on the graph structure, xhy, xhh 1 y, and Qps, aq «´F ps, a; wq are not tractable.

Consequently, to perform the weight update in (6) one requires samples from the Boltzmann distribution corresponding to energy function (3) to estimate xhy, xhh 1 y, and F ps, a; wq empirically.

To approximate the right-hand side of (9) and FORMULA1 , we sample from the Boltzmann distribution of the energy function represented by the effective Hamiltonian using (Suzuki, 1976, Theorem 6) .

We find the expected values of the observables xσ

One way to sample spin values from the Boltzmann distribution of the effective Hamiltonian is to use the simulated quantum annealing algorithm (SQA) (see (Brabazon et al., 2015, p. 422) for an introduction).

SQA is one of the many flavours of quantum Monte Carlo methods, and is based on the Suzuki-Trotter expansion described above.

This algorithm simulates the quantum annealing phenomena of a TFIM by slowly reducing the strength of the transverse field at finite temperature to the desired target value.

In our implementation, we have used a single spin-flip variant of SQA with a linear transverse-field schedule as in BID21 and BID14 .

Experimental studies have shown similarities in the behaviour of SQA and that of quantum annealing Isakov et al. The classical counterpart of SQA is conventional simulated annealing (SA), which is based on thermal annealing.

This algorithm can be used to sample from Boltzmann distributions that correspond to an Ising spin model in the absence of a transverse field.

Unlike SA, it is possible to use SQA not only to approximate the Boltzmann distribution of a classical Boltzmann machine, but also that of a quantum Hamiltonian in the presence of a transverse field.

This can be done by reducing the strength of the transverse field to the desired value defined by the model, rather than to zero.

It has been proven by BID24 that the spin system defined by SQA converges to the Boltzmann distribution of the effective classical Hamiltonian of one dimension higher that corresponds to the quantum Hamiltonian.

Therefore, it is straightforward to use SQA to approximate the free energy in (12) as well as the observables xσ z h y and xσ z h σ z h 1 y. However, any Boltzmann distribution sampling method based on Markov chain Monte Carlo (MCMC) has the major drawback of being extremely slow and computationally involved.

Actually, it is an NP-hard problem to sample from the Boltzmann distribution.

Another option is to use variational approximation BID26 , which suffers from lack of accuracy and works in practice only in limited cases.

As explained above, quantum annealers have the potential to provide samples from Boltzmann distributions (in the z-basis) corresponding to TFIM in a more efficient way.

In what follows, we explain how to use quantum annealing to approximate the free energy corresponding to an effective Hamiltonian which in turn can be used to approximate the free energy of a QBM. (Suzuki, 1976, Theorem 6 ) and translation invariance, each replica of the effective classical model is an approximation of the spin measurements of the TFIM in the measurement bases σ z .

Therefore, a σ z -configuration sampled by a quantum annealer that operates at a given virtual inverse temperature β, and anneals up to a virtual transverse-field strength Γ, may be viewed as an instance of a classical spin configuration from a replica of the classical effective Hamiltonian of one dimension higher.

This suggests the following method to approximate the free energy from (12) for a TFIM.

We gather a pool C of configurations sampled by the quantum annealer for the TFIM considered, allowing repetitions.

Let r be the number of replicas.

We write c eff " pc 1 , . . .

, c r q to indicate an effective configuration c eff with the classical configurations c 1 to c r as its replicas.

We write c eff to denote the underlying set tc 1 , . . .

, c r u of replicas of c eff (without considering their ordering).

We have P rc eff " pc 1 , . . .

, c r qs " P " c eff " pc 1 , . . .

, c r q|c eff " tc 1 , . . .

, c r u ‰ˆP " c eff " tc 1 , . . .

, c r u ‰ " P " c eff " pc 1 , . . .

, c r q|c eff " tc 1 , . . .

, c r u ‰ˆP " c eff " tc 1 , . . .

, c r u|c eff Ď C ‰ˆP " c eff Ď C ‰ .The argument in the previous paragraph can now be employed to allow the assumption P " c eff Ď C ‰ » 1.

In other words, the probability mass function of the effective configurations is supported in the subset of those configurations synthesized from the elements of C as candidate replicas.

The conditional probability Prc eff " tc 1 , . . .

, c r u|c eff Ď C s can be sampled from by drawing r elements c 1 , . . . , c r from C .

We then sample from P " c eff " pc 1 , . . .

, c r q|c eff " tc 1 , . . .

, c r u ‰ , according to the following distribution over c eff : DISPLAYFORM0 We consider πpc eff q our target distribution and construct the following MCMC method for which the limiting distribution is πpc eff q.

We first attach the r classical spin configurations to the SQA's effective configuration structure uniformly at random.

We then transition to a different arrangement with a Metropolis acceptance probability.

For example, we may choose two classical configurations at random and exchange them with probability DISPLAYFORM1 where Epc eff q " w`ř h´ř r´1 k"1 h c k h c k`1`h c1 h cr¯. Such a stochastic process is known to satisfy the detailed balance condition.

Consequently, the MCMC method allows us to sample from the effective spin configurations.

This procedure of sampling and then performing the MCMC method creates a pool of effective spin configurations, which are then employed in equation FORMULA1 in order to approximate the free energy of the TFIM empirically.

However, we consider a relatively small number of hidden nodes in our experiments, so the number of different σ z -configurations sampled by the quantum annealer is limited.

As a consequence, there is no practical need to perform the MCMC method defined above.

Instead, we attach classical spin configurations from the pool to the SQA effective configuration structure at random.

In other words, in r iterations, a spin configuration is sampled from the pool of classical spin configurations described above and inserted as the next replica of the effective classical Hamiltonian consisting of r replicas.

It is worthwhile to reiterate that this replica stacking technique yields an undirected graphical model.

Specifically, the structure described in FIG1 is an undirected graphical model in the space of hidden nodes, where the node statistics are obtained from the Boltzmann distribution.

One difference between this model and a classical Boltzmann machine is that each hidden node activation is governed by a series of r replicas in one dimension higher, and the undirected, replica-to-replica connections calculated therein.

Moreover, the energy function of this extended model differs from the energy function of the classical Boltzmann machine (compare (11) and FORMULA3 ).

The free energy of the extended graphical model serves as the function approximator to the Q-function.

We benchmark our various FERL methods on a 3ˆ5 grid-world problem BID32 with an agent capable of taking the actions up, down, left, or right, or standing still, on a grid-world with y, andPpc ef f |s i , a i q using Algorithm 2, for (i " 1, 2) calculate F ps i , a i q using (12) for (i " 1, 2) Qps i , a i q Ð´F ps i , a i q for pi " 1, 2q update QBM weights using (9) and (10) πps 1 q Ð argmax a Qps 1 , aq end for return π Algorithm 2 Replica stacking initialize the structure of the effective Hamiltonian in one dimension higher for i " 1, 2, ..., m do for j " 1, 2, ..., r do obtain spin configuration sample in z-basis from QA attach this spin configuration to j-th replica of the i-th effective configuration structure end for perform the MCMC technique described in §3.4 with transition probabilities (13) to obtain the i-th instance of effective spin configurations end for obtain xH eff s i ,a i y from the average energy of the m effective spin configurations obtain xhy and xhh 1 y by averaging over all h and h 1 replicas in each spin configuration gather statistics from Ppc eff |s i , a i q using the m effective spin configurations return xhy, xhh 1 y, xH DISPLAYFORM0 y, and Ppc eff |s i , a i q one deterministic reward, one wall, and one penalty, as shown in FIG5 .

The task is to find an optimal policy, as shown in FIG5 , for the agent at each state in the grid-world.

All of the Boltzmann machines used in our algorithms consist of 16 hidden nodes.

The discount factor, as explained in §2, is set to 0.8.

The agent attains the reward R " 200 in the top-left corner, the neutral value of moving to any empty cell is 100, and the agent is penalized by not receiving any reward if it moves to the penalty cell with value P " 0.For T r independent runs of every FERL method, T s training samples are used.

The fidelity measure at the i-th training sample is defined by fidelitypiq " pT rˆ| S|q´1 DISPLAYFORM1 where π˚denotes the best known policy and Aps, i, lq denotes the action assigned at the l-th run and i-th training sample to the state s.

In our experiments, each algorithm is run 100 times.

An optimal policy for this problem instance can be represented as a selection of directional arrows indicating movement directions.

Fig. 4 demonstrates the performance of a fully connected deep Q-network BID23 consisting of an input layer of 14 state nodes, two layers of eight hidden nodes each, and an output layer of five nodes representing the values of the Q-function for different actions, given a configuration of state nodes.

We use the same number of hidden nodes in the fully connected deep Q-network as in the other networks described in this paper.

We treat the network of superconducting qubits represented in FIG0 as a clamped QBM with two hidden layers, represented using blue and red colours.

The state nodes are considered fully connected to the blue qubits and the action nodes are fully connected to the red qubits.

For a choice of virtual parameters Γ ‰ 0 and β, which appear in FORMULA1 and FORMULA1 , and for each query to the D-Wave 2000Q chip, we construct 150 effective classical configurations of one dimension higher, out of a pool of 3750 reads, according to the replica stacking method introduced in §3.4.

The 150 configurations are, in turn, employed to approximate the free energy of the quantum Hamiltonian.

We conduct 10 independent runs of FERL in this fashion, and find the average fidelity over the 10 runs and over the T s " 300 training samples.

Fig .

6 shows the growth of the average fidelity of the best known policies generated by different FERL methods.

For each method, the fidelity curve is an average over 100 independent runs, each with T s " 500 training samples.

In this figure, the "D-Wave Γ " 0.5, β " 2.0" curve corresponds to the D-Wave 2000Q replica stacking-based method with the choice of the best virtual parameters Γ " 0.5 and β " 2.0, as shown in the heatmap in FIG6 .

The training is based on formulae (9), (10), and (12).

The "SQA Bipartite Γ " 0.5, β " 2.0" and "SQA Chimera Γ " 0.5, β " 2.0" curves are based on the same formulae with the underlying graphs being a bipartite (DBM) and a Chimera graph, respectively, with the same choice of virtual parameters, but the effective Hamiltonian configurations generated using SQA as explained in §3.3.The "SA Bipartite β " 2.0" and "SA Chimera β " 2.0" curves are generated by using SA to train a classical DBM and a classical GBM on the Chimera graph, respectively, using formulae (6), FORMULA6 , and (5).

SA is run with a linear inverse temperature schedule, where β " 2.0 indicates the final value.

The "D-Wave Classical β " 2.0" curve is generated using the same method, but with samples obtained using the D-Wave 2000Q.

The "RBM" curve is generated using the method in BID27 .

We solve the grid-world problem using various Q-learning methods with the Q-function parametrized by different neural networks.

For comparison, we demonstrate the performance of a fully connected deep Q-network method that can be considered state of the art.

This method efficiently processes every training sample, but, as shown in Fig. 4 , requires a very large number of training samples to converge to the optimal policy.

Another conventional method is free-energy-based RL using an RBM.

This method is also very successful at learning the optimal policy at the scale of the RL task considered in our experiment.

Although this method does not outperform other FERL methods that take advantage of a highly efficient sampling oracle, the processing of each training sample is efficient, as it is based on closed formulae.

In fact, for the size of problem considered, the RBM-based FERL outperforms the fully connected deep Q-network method.

The comparison of results in Fig. 6 suggests that replica stacking is a successful method for estimating effective classical configurations obtained from a quantum annealer, given that the spins can only be measured in measurement bases.

For practical use in RL, this method provides a means of treating the quantum annealer as a QBM.

FERL using the quantum annealer, in conjunction with the replica stacking technique, provides significant improvement over FERL using classical Boltzmann machines.

The curve representing SQA-based FERL using a Boltzmann machine on the Chimera graph is almost coincident with the one obtained using the D-Wave 2000Q, whereas the SQA-based FERL using a DBM slightly outperforms it.

This suggests that quantum annealing chips with greater connectivity and more control over annealing time can further improve the performance of the replica stacking method applied to RL tasks.

This is further supported by comparing the performance of SA-based FERL using a DBM versus SA-based FERL using the Chimera graph.

This result shows that DBM is, due to its additional connections, a better choice of neural network compared to the Chimera graph.

For practical reasons, we aim to associate an identical choice of virtual parameters β and Γ to all of the TFIMs constructed using FERL.

BID5 and BID25 provide methods for estimating the effective inverse temperature β for other applications.

However, in both studies, the samples obtained from the quantum annealer are matched to the Boltzmann distribution of a classical Ising model.

In fact, the transverse-field strength is a second virtual parameter that we consider.

The optimal choice Γ " 0.5 corresponds to 2{3 of the annealing time, in agreement with the work of BID0 , who also considers TFIM with 16 qubits.

The agreement of FERL using quantum annealer reads treated as classical Boltzmann samples with that of FERL using SA and classical Boltzmann machines suggests that, at least for this task and this size of Boltzmann machine, the measurements provided by the D-Wave 2000Q can be considered good approximations of Boltzmann distribution samples of classical Ising models.

The extended undirected graphical model developed in this paper using the replica stacking method is not limited to Q-function approximation in RL tasks.

Potentially, this method can be applied to tasks where Boltzmann machines can be used.

This method provides a mechanism for approximating the activations and partition functions of quantum Boltzmann machines that have a significant transverse field.

In this paper, we describe a free-energy-based reinforcement learning algorithm using an existing quantum annealer, namely the D-Wave 2000Q.

Our method relies on the Suzuki-Trotter decomposition and the use of the measured configurations by the D-Wave 2000Q as replicas of an effective classical Ising model of one dimension higher.

The results presented here are first-step proofs of concept of a proposed quantum algorithm with a promising path towards outperforming reinforcement learning algorithms devised for digital hardware.

Given appropriate advances in quantum annealing hardware, future research can employ the proposed principles to solve larger-scale reinforcement learning tasks in the emerging field of quantum machine learning.

<|TLDR|>

@highlight

We train Quantum Boltzmann Machines using a replica stacking method and a quantum annealer to perform a reinforcement learning task.