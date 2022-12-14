We introduce Quantum Graph Neural Networks (QGNN), a new class of quantum neural network ansatze which are tailored to represent quantum processes which have a graph structure, and are particularly suitable to be executed on distributed quantum systems over a quantum network.

Along with this general class of ansatze, we introduce further specialized architectures, namely, Quantum Graph Recurrent Neural Networks (QGRNN) and Quantum Graph Convolutional Neural Networks (QGCNN).

We provide four example applications of QGNN's: learning Hamiltonian dynamics of quantum systems, learning how to create multipartite entanglement in a quantum network, unsupervised learning for spectral clustering, and supervised learning for graph isomorphism classification.

Variational Quantum Algorithms are a promising class of algorithms are rapidly emerging as a central subfield of Quantum Computing (McClean et al., 2016; Farhi et al., 2014; Farhi & Neven, 2018) .

Similar to parameterized transformations encountered in deep learning, these parameterized quantum circuits are often referred to as Quantum Neural Networks (QNNs).

Recently, it was shown that QNNs that have no prior on their structure suffer from a quantum version of the no-free lunch theorem (McClean et al., 2018) and are exponentially difficult to train via gradient descent.

Thus, there is a need for better QNN ansatze.

One popular class of QNNs has been Trotter-based ansatze (Farhi et al., 2014; Hadfield et al., 2019) .

The optimization of these ansatze has been extensively studied in recent works, and efficient optimization methods have been found (Verdon et al., 2019b; Li et al., 2019) .

On the classical side, graph-based neural networks leveraging data geometry have seen some recent successes in deep learning, finding applications in biophysics and chemistry (Kearnes et al., 2016) .

Inspired from this success, we propose a new class of Quantum Neural Network ansatz which allows for both quantum inference and classical probabilistic inference for data with a graph-geometric structure.

In the sections below, we introduce the general framework of the QGNN ansatz as well as several more specialized variants and showcase four potential applications via numerical implementation.

Graph Neural Networks (GNNs) date back to Sperduti & Starita (1997) who applied neural networks to acyclic graphs.

Gori et al. (2005) and Scarselli et al. (2008) developed methods that learned node representations by propagating the information of neighbouring nodes.

Recently, GNNs have seen great breakthroughs by adapting the convolution operator from CNNs to graphs (Bruna et al., 2013; Henaff et al., 2015; Defferrard et al., 2016; Kipf & Welling, 2016; Niepert et al., 2016; Hamilton et al., 2017; Monti et al., 2017) .

Many of these methods can be expressed under the message-passing framework (Gilmer et al., 2017) .

n??n is the adjacency matrix, and X ??? R n??d is the node feature matrix where each node has d features.

where H (k) ??? R n??d are the node representations computed at layer k, P is the message propagation function and is dependent on the adjacency matrix, the previous node encodings and some learnable parameters W (k) .

The initial embedding, H (0) is naturally X. One popular implementation of this framework is the GCN (Kipf & Welling, 2016) which implements it as follows:

where?? = A + I is the adjacency matrix with inserted self-loops,D = j?? ij is the renormalization factor (degree matrix).

Consider a graph G = {V, E}, where V is the set of vertices (or nodes) and E the set of edges.

We can assign a quantum subsystem with Hilbert space H v for each vertex in the graph, forming a global Hilbert space H V ??? v???V H v .

Each of the vertex subsystems could be one or several qubits, a qudit, a qumode (Weedbrook et al., 2012) , or even an entire quantum computer.

One may also define a Hilbert space for each edge and form H E ??? e???E H e .

The total Hilbert space for the graph would then be H E ??? H V .

For the sake of simplicity and feasibility of numerical implementation, we consider this to be beyond the scope of the present work.

The edges of the graph dictate the communication between the vertex subspaces: couplings between degrees of freedom on two different vertices are allowed if there is an edge connecting them.

This setup is called a quantum network (Kimble, 2008; Qian et al., 2019) with topology given by the graph G.

The most general Quantum Graph Neural Network ansatz is a parameterized quantum circuit on a network which consists of a sequence of Q different Hamiltonian evolutions, with the whole sequence repeated P times:??

where the product is time-ordered (Poulin et al., 2011) , the ?? and ?? are variational (trainable) parameters, and the Hamiltonians?? q (??) can generally be any parameterized Hamiltonians whose topology of interactions is that of the problem graph:

Here the W qrjk and B qrv are real-valued coefficients which can generally be independent trainable parameters, forming a collection

are Hermitian operators which act on the Hilbert space of the j th node of the graph.

The sets I jk and J v are index sets for the terms corresponding to the edges and nodes, respectively.

To make compilation easier, we enforce that the terms of a given Hamiltonian?? q commute with one another, but different?? q 's need not commute.

In order to make the ansatz more amenable to training and avoid the barren plateaus (quantum parametric circuit no free lunch) problem (McClean et al., 2018) , we need to add some constraints and specificity.

To that end, we now propose more specialized architectures where parameters are tied spatially (convolutional) or tied over the sequential iterations of the exponential mapping (recurrent).

We define quantum graph recurrent neural networks as ansatze of the form of equation 3 where the temporal parameters are tied between iterations, ?? pq ??? ?? q .

In other words, we have tied the parameters between iterations of the outer sequence index (over p = 1, . . .

, P ).

This is akin to classical recurrent neural networks where parameters are shared over sequential applications of the recurrent neural network map.

As ?? q acts as a time parameter for Hamiltonian evolution under H q , we can view the QGRNN ansatz as a Trotter-based (Lloyd, 1996; Poulin et al., 2011) quantum simulation of an evolution e ???i????? eff under the Hamiltionian?? eff = ??? ???1 q ?? q??q for a time step of size ??? = ?? 1 = q |?? q |.

This ansatz is thus specialized to learn effective quantum Hamiltonian dynamics for systems living on a graph.

In Section 3 we demonstrate this by learning the effective real-time dynamics of an Ising model on a graph using a QGRNN ansatz.

Classical Graph Convolutional neural networks rely on a key feature: that of permutation invariance.

In other words, the ansatz should be invariant under permutation of the nodes.

This is analogous to translational invariance for ordinary convolutional transformations.

In our case, permutation invariance manifests itself as a constraint on the Hamiltonian, which now should be devoid of local trainable parameters, and should only have global trainable parameters.

The ?? parameters thus become tied over indices of the graph: W qrjk ???

W qr and B qrv ??? B qr .

A broad class of graph convolutional neural networks we will focus on is the set of so-called Quantum Alternating Operator Ansatze (Hadfield et al., 2019) , the generalized form of the Quantum Approximate Optimization Algorithm ansatz (Farhi et al., 2014) .

We can take inspiration from the continuous-variable quantum approximate optimization ansatz introduced in Verdon et al. (2019a) to create a variant of the QGCNN: the Quantum Spectral Graph Convolutional Neural Network (QSGCNN).

We show here how it recovers the mapping of Laplacianbased graph convolutional networks (Kipf & Welling, 2016) in the Heisenberg picture, consisting of alternating layers of message passing, node update, and nonlinearities.

Consider an ansatz of the form from equation 3 with four different Hamiltonians (Q = 4) for a given graph.

First, for a weighted graph G with edge weights ?? jk , we define the coupling Hamiltonian a??

The ?? jk here are the weights of the graph G, and are not trainable parameters.

The operators denoted here byx j are quantum continuous-variable position operators, which can be implemented via continuous-variable (analog) quantum computers (Weedbrook et al., 2012) or emulated using multiple qubits on digital quantum computers (Somma, 2015; Verdon et al., 2018) .

After evolving by?? C , which we consider to be the message passing step, one applies an exponential of the kinetic Hamiltonian,?? K ??? 1 2 j???Vp 2 j .

Herep j denotes the continuous-variable momentum (Fourier conjugate) of the position, obeying the canonical commutation relation [x j ,p j ] = i?? jk .

We consider this step as a node update step.

In the Heisenberg picture, the evolution generated by these two steps maps the position operators of each node according to

where L jk = ?? jk v???V ?? jv ??? ?? jk is the Graph Laplacian matrix for the weighted graph G. We can recognize this step as analogous to classical spectral-based graph convolutions.

One difference to note here is that momentum is free to accumulate between layers.

Next, we must add some non-linearity in order to give the ansatz more capacity.

1 The next evolution is thus generated by an anharmonic Hamiltonian?? A = j???V f (x j ), where f is a nonlinear function of degree greater than 2, e.g., a quartic potential of the form f (x j ) = ((x j ??? ??) 2 ??? ?? 2 ) 2 for some ??, ?? hyperparameters.

Finally, we apply another evolution according to the kinetic Hamiltonian.

These last two steps yield an update

which acts as a nonlinear mapping.

By repeating the four evolution steps described above in a sequence of P layers, i.e.,

with variational parameters ?? = {??, ??, ??, ??}, we then recover a quantum-coherent analogue of the node update prescription of Kipf & Welling (2016) in the original graph convolutional networks paper.

Learning the dynamics of a closed quantum system is a task of interest for many applications (Wiebe et al., 2014) , including device characterization and validation.

In this example, we demonstrate that a Quantum Graph Recurrent Neural Network can learn effective dynamics of an Ising spin system when given access to the output of quantum dynamics at various times.

Our target is an Ising Hamiltonian with transverse field on a particular graph,

We are given copies of a fixed low-energy state |?? 0 as well as copies of the state |?? T ??? U (T ) |?? 0 = e ???iT??target for some known but randomly chosen times T ??? [0, T max ].

Our goal is to learn the target Hamiltonian parameters {J jk , Q v } j,k,v???V by comparing the state |?? T with the state obtained by evolving |?? 0 according to the QGRNN ansatz for a number of iterations P ??? T /??? (where ??? is a hyperparameter determining the Trotter step size).

We achieve this by training the parameters via Adam (Kingma & Ba, 2014) gradient descent on the average infidelity

2 averaged over batch sizes of 15 different times T .

Gradients were estimated via finite difference differentiation with step size = 10 ???4 .

The fidelities (quantum state overlap) between the output of our ansatz and the time-evolved data state were estimated via the quantum swap test (Cincio et al., 2018) .

The ansatz uses a Trotterization of a random densely-connected Ising Hamiltonian with transverse field as its initial guess, and successfully learns the Hamiltonian parameters within a high degree of accuracy as shown in Fig. 1a .

A picture of the quantum network topology is inset.

Right: Quantum phase kickback test on the learned GHZ state.

We observe a 7x boost in Rabi oscillation frequency for a 7-node network, thus demonstrating we have reached the Heisenberg limit of sensitivity for the quantum sensor network.

Quantum Sensor Networks are a promising area of application for the technologies of Quantum Sensing and Quantum Networking/Communication (Kimble, 2008; Qian et al., 2019) .

A common task considered where a quantum advantage can be demonstrated is the estimation of a parameter hidden in weak qubit phase rotation signals, such as those encountered when artificial atoms interact with a constant electric field of small amplitude (Qian et al., 2019) .

A well-known method to achieve this advantange is via the use of a quantum state exhibiting multipartite entanglement of the Greenberger-Horne-Zeilinger kind, also known as a GHZ state (Greenberger et al., 1989 ).

Here we demonstrate that, without global knowledge of the quantum network structure, a QGCNN ansatz can learn to prepare a GHZ state.

We use a QGCNN ansatz with?? 1 = {j,k}???E??? j???k and?? 2 = j???VX j .

The loss function is the negative expectation of the sum of stabilizer group generators which stabilize the GHZ state (T??th & G??hne, 2005) , i.e.,

for a network of n qubits.

Results are presented in Fig. 1b .

Note that the advantage of using a QGNN ansatz on the network is that the number of quantum communication rounds is simply proportional to P , and that the local dynamics of each node are independent of the global network structure.

In order to further validate that we have obtained an accurate GHZ state on the network after training, we perform the quantum phase kickback test on the network's prepared approximate GHZ state (Wei et al., 2019) .

3 We observe the desired frequency boost effect for our trained network preparing an approximate GHZ state at test time, as displayed in Figure 2.

As a third set of applications, we consider applying the QSGCNN from Section 2 to the task of spectral clustering (Ng et al., 2002) .

Spectral clustering involves finding low-frequency eigenvalues of the graph Laplacian and clustering the node values in order to identify graph clusters.

In Fig. 3 we present the results for a QSGCNN for varying multi-qubit precision for the representation of the continuous values, where the loss function that was minimized was the expected value of the anharmonic potential L(??) = ?? C +?? A ?? .

Of particular interest to near-term quantum computing with low numbers if qubits is the single-qubit precision case, where we modify the QSGCNN construction asp 2 j ???X j , 3 For this test, one applies a phase rotation j???V e ???i????? j on all the qubits in paralel, then one applies a sequence of CNOT's (quantum adder gates) such as to concentrate the phase shifts onto a single collector node, m ??? V. Given that one had a GHZ state initially, one should then observe a phase shift e ???in?????m where n = |V|.

This boost in frequency of oscillation of the signal is what gives quantum multipartite entanglement its power to increase sensitivity to signals to super-classical levels (Degen et al., 2017) .

configurations, and to their right is the output probability distribution over potential energies.

We see lower energies are most probable and that these configurations have node values clustered.

H A ???

I andx j ??? |1 1| j which transforms the coupling Hamiltonian a??

where |1 1| k = (?? ?????? k )/2.

We see that using a low-qubit precision yields sensible results, thus implying that spectral clustering could be a promising new application for near-term quantum devices.

Recently, a benchmark of the representation power of classical graph neural networks has been proposed (Xu et al., 2018) where one uses classical GCN's to identify whether two graphs are isomorphic.

In this spirit, using the QSGCNN ansatz from the previous subsection, we benchmarked the performance of this Quantum Graph Convolutional Network for identifying isomorphic graphs.

We used the single-qubit precision encoding in order to order to simulate the execution of the quantum algorithms on larger graphs.

Our approach was the following, given two graphs G 1 and G 2 , one applies the single-qubit precision QSGCNN ansatz P j=1 e i??j?? K e i??j?? C with?? K = j???VX j and?? C from equation 5 in parallel according to each graph's structure.

One then samples eigenvalues of the coupling Hamiltonian?? C on both graphs via standard basis measurement of the qubits and computation of the eigenvalue at each sample of the wavefunction.

One then obtains a set of samples of "energies" of this Hamiltonian.

By comparing the energetic measurement statistics output by the QSGCNN ansatz applied with identical parameters ?? = {??, ??} for two different graphs, one can then infer whether the graphs are isomorphic.

We used the Kolmogorov-Smirnoff test (Lilliefors, 1967) on the distribution of energies sampled at the output of the QSGCNN to determine whether two given graphs were isomorphic.

In order to determine the binary classification label deterministically, we considered all KS statistic values above 0.4 to indicate that the graphs were non-isomorphic.

For training and testing purposes, we set the For the dataset, graphs were sampled uniformly at random; to prepare a balanced dataset, we selected isomorphic and non-isomorphic pairs.

In all of our experiments, we had 100 pairs of graphs for training, 50 for validation, 50 for testing, and in all cases there are balanced isomorphic and nonisomorphic pairs.

The networks were trained via Adam gradient-based optimizer with batches of size 50.

Presented in Figure 4 is the training and testing losses for various graph sizes and numbers of energetic samples.

In Tables 1 and 2 , we present the graph isomorphism classification accuracy for the training and testing sets using the trained QGCNN with the previously described thresholded KS statistic as the label.

We see we get highly accurate performance even at low sample sizes.

This seems to imply that the QGCNN is fully capable of identifying graph isomorphism, as desired for graph convolutional network benchmarks.

We leave a comparison to similar scale classical graph convolutional networks to future work.

Results featured in this paper should be viewed as a promising set of first explorations of the potential applications of QGNNs.

Through our numerical experiments, we have shown the use of these QGNN ansatze in the context of quantum dynamics learning, quantum sensor network optimization, unsupervised graph clustering, and supervised graph isomorphism classification.

Given that there is a vast set of literature on the use of Graph Neural Networks and their variants to quantum chemistry, future works should explore hybrid methods where one can learn a graph-based hidden quantum representation (via a QGNN) of a quantum chemical process.

As the true underlying process is quantum in nature and has a natural molecular graph geometry, the QGNN could serve as a more accurate model for the hidden processes which lead to perceived emergent chemical properties.

We seek to explore this in future work.

Other future work could include generalizing the QGNN to include quantum degrees of freedom on the edges, include quantum-optimization-based training of the graph parameters via quantum phase backpropagation (Verdon et al., 2018) , and extending the QSGCNN to multiple features per node.

@highlight

Introducing a new class of quantum neural networks for learning graph-based representations on quantum computers.