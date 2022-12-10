Quantum machine learning methods have the potential to facilitate learning using extremely large datasets.

While the availability of data for training machine learning models is steadily increasing, oftentimes it is much easier to collect feature vectors that to obtain the corresponding labels.

One of the approaches for addressing this issue is to use semi-supervised learning, which leverages not only the labeled samples, but also unlabeled feature vectors.

Here, we present a quantum machine learning algorithm for training Semi-Supervised Kernel Support Vector Machines.

The algorithm uses recent advances in quantum sample-based Hamiltonian simulation to extend the existing Quantum LS-SVM algorithm to handle the semi-supervised term in the loss, while maintaining the same quantum speedup as the Quantum LS-SVM.

Data sets used for training machine learning models are becoming increasingly large, leading to continued interest in fast methods for solving large-scale classification problems.

One of the approaches being explored is training the predictive model using a quantum algorithm that has access to the training set stored in quantum-accessible memory.

In parallel to research on efficient architectures for quantum memory (Blencowe, 2010) , work on quantum machine learning algorithms and on quantum learning theory is under way (see for example Refs. (Biamonte et al., 2017; Dunjko & Briegel, 2018; Schuld & Petruccione, 2018) and (Arunachalam & de Wolf, 2017) for review).

An early example of this approach is Quantum LS-SVM (Rebentrost et al., 2014a) , which achieves exponential speedup compared to classical LS-SVM algorithm.

Quantum LS-SVM uses quadratic least-squares loss and squared-L 2 regularizer, and the optimization problem can be solved using the seminal HHL (Harrow et al., 2009 ) algorithm for solving quantum linear systems of equations.

While progress has been made in quantum algorithms for supervised learning, it has been recently advocated that the focus should shift to unsupervised and semi-supervised setting (Perdomo-Ortiz et al., 2018) .

In many domains, the most laborious part of assembling a training set is the collection of sample labels.

Thus, in many scenarios, in addition to the labeled training set of size m we have access to many more feature vectors with missing labels.

One way of utilizing these additional data points to improve the classification model is through semi-supervised learning.

In semi-supervised learning, we are given m observations x 1 , ..., x m drawn from the marginal distribution p(x), where the l (l m) first data points come with labels y 1 , ..., y l drawn from conditional distribution p(y|x).

Semi-supervised learning algorithms exploit the underlying distribution of the data to improve classification accuracy on unseen samples.

In the approach considered here, the training samples are connected by a graph that captures their similarity.

Here, we introduce a quantum algorithm for semi-supervised training of a kernel support vector machine classification model.

We start with the existing Quantum LS-SVM (Rebentrost et al., 2014a) , and use techniques from sample-based Hamiltonian simulation (Kimmel et al., 2017) to add a semisupervised term based on Laplacian SVM (Melacci & Belkin, 2011) .

As is standard in quantum machine learning (Li et al., 2019) , the algorithm accesses training points and the adjacency matrix of the graph connecting samples via a quantum oracle.

We show that, with respect to the oracle, the proposed algorithm achieves the same quantum speedup as LS-SVM, that is, adding the semisupervised term does not lead to increased computational complexity.

Consider a problem where we are aiming to find predictors h(x) : X → R that are functions from a RKHS defined by a kernel K. In Semi-Supervised LS-SVMs in RKHS, we are looking for a function h ∈ H that minimizes min h∈H,b∈R

where y = (y 1 , ..., y m ) T , 1 = (1, ..., 1) T , 1 is identity matrix, K is kernel matrix, L is the graph Laplacian matrix, γ is a hyperparameter and α = (α 1 , ..., α m )

T is the vector of Lagrangian multipliers.

Quantum computers are devices which perform computing according to the laws of quantum mechanics, a mathematical framework for describing physical theories, in language of linear algebra.

Quantum Systems.

Any isolated, closed quantum physical system can be fully described by a unit-norm vector in a complex Hilbert space appropriate for that system; in quantum computing, the space is always finite-dimensional, C d .

In quantum mechanics and quantum computing, Dirac notation for linear algebra is commonly used.

In Dirac notation, a vector x ∈ C d and its complex conjugate x T , which represents a functional C d → R, are denoted by |x (called ket) and x| (called bra), respectively.

We call {|e i } T , |1 = (0, 1) T and α, β ∈ C, |α| 2 + |β| 2 , is called a quantum bit, or qubit for short.

When both α and β are nonzero, we say |ψ is in a superposition of the computational basis |0 and |1 ; the two superposition states

A composite quantum state of two distinct quantum systems |x 1 ∈ C d1 and |x 2 ∈ C d2 is described as tensor product of quantum states |x 1 ⊗ |x 2 ∈ C d1 ⊗ C d2 .

Thus, a state of an n-qubit system is a vector in the tensor product space C 2 ⊗n = C 2 ⊗ C 2 ⊗ ...

⊗ C 2 , and is written as

i=0 α i |i , where i is expressed using its binary representation; for example for n = 4, we have |2 = |0010 = |0 ⊗ |0 ⊗ |1 ⊗ |0 .

Transforming and Measuring Quantum States.

Quantum operations manipulate quantum states in order to obtain some desired final state.

Two types of manipulation of a quantum system are allowed by laws of physics: unitary operators and measurements.

Quantum measurement, if done in the computational basis, stochastically transforms the state of the system into one of the computational basis states, based on squared magnitudes of probability amplitudes; that is,

will result in |0 and |1 with equal chance.

Unitary operators are deterministic, invertible, normpreserving linear transforms.

A unitary operator U models a transformation of a quantum state |u to |v = U|u .

Note that U|u 1 + U|u 2 = U (|u 1 + |u 2 ), applying a unitary to a superposition of states has the same effect as applying it separately to element of the superposition.

In quantum circuit model unitary transformations are referred to as quantum gates -for example, one of the most common gates, the single-qubit Hadamard gate, is a unitary operator represented in the computational basis by the matrix

Note that H|0 = |+ and H|1 = |− .

Quantum Input Model.

Quantum computation typically starts from all qubits in |0 state.

To perform computation, access to input data is needed.

In quantum computing, input is typically given by a unitary operator that transforms the initial state into the desired input state for the computation -such unitaries are commonly referred to as oracles, and the computational complexity of quantum algorithms is typically measured with access to an oracle as the unit.

For problems involving large amounts of input data, such as for quantum machine learning algorithms, an oracle that abstracts random access memory is often assumed.

Quantum random access memory (qRAM) uses log N qubits to address any quantum superposition of N memory cell which may contains either quantum or classical information.

For example, qRAM allows accessing classical data entries x j i in quantum superposition by a transformation

where |x j i is a binary representation up to a given precision.

Several approaches for creating quantum RAM are being considered (Giovannetti et al., 2008; Arunachalam et al., 2015; Biamonte et al., 2017) , but it is still an open challenge, and subtle differences in qRAM architecture may erase any gains in computational complexity of a quantum algorithm Aaronson (2015) .

Quantum Linear Systems of Equations.

Given an input matrix A ∈ C n×n and a vector b ∈ C n , the goal of linear system of equations problem is finding x ∈ C n such that Ax = b. When A is Hermitian and full rank, the unique solution is x = A −1 b. If A is not a full rank matrix then A −1 is replaced by the Moore-Penrose pseudo-inverse.

HHL algorithm introduced an analogous problem in quantum setting: assuming an efficient algorithm for preparing b as a quantum state b = n i=1 b i |i using log n + 1 qubits, the algorithm applies quantum subroutines of phase estimation, controlled rotation, and inverse of phase estimation to obtain the state

Intuitively and at the risk of over-simplifying, HHL algorithm works as follows: if A has spec-

In general A and A −1 are not unitary (unless all A's eigenvalues have unit magnitude), therefore we are not able to apply A −1 directly on |b .

However, since

is unitary and has the same eigenvectors as A and A −1 , one can implement U and powers of U on a quantum computer by Hamiltonian simulation techniques; clearly for any expected speed-up, one need to enact e iA efficiently.

The HHL algorithm uses the phase estimation subroutine to estimate an approximation of λ i up to a small error.

The Next step computes a conditional rotation on the approximated value of λ i and an auxiliary qubit |0 and outputs

|1 .

The last step involves the inverse of phase estimation and quantum measurement for getting rid of garbage qubits and outputs our desired state

Density Operators.

Density operator formalism is an alternative formulation for quantum mechanics that allows probabilistic mixtures of pure states, more generally referred to as mixed states.

A mixed state that describes an ensemble {p i , |ψ i } is written as

where k i=1 p i = 1 forms a probability distribution and ρ is called density operator, which in a finite-dimensional system, in computational basis, is a semi-definite positive matrix with T r(ρ) = 1.

A unitary operator U maps a quantum state expressed as a density operator ρ to UρU † , where U † is the complex conjugate of the operator U.

Partial Trace of Composite Quantum System.

Consider a two-part quantum system in a state described by tensor product of two density operators ρ ⊗ σ.

A partial trace, tracing out the second part of the quantum system, is defined as the linear operator that leaves the first part of the system in a state Tr 2 (ρ ⊗ σ) = ρ tr (σ), where Tr (σ) is the trace of the matrix σ.

To obtain Kernel matrix K as a density matrix, quantum LS-SVM (Rebentrost et al., 2014b ) relies on partial trace, and on a quantum oracle that can convert, in superposition, each data point

where (x i ) t refers to the tth feature value in data point x i and assuming the oracle is given x i and y i .

Vector of the labels is given in the same fashion as |y = 1 y m i=1 y i |i .

For preparation the normalized Kernel matrix K = 1 tr(K) K where K = X T X, we need to prepare a quantum state combining all data points in quantum

The normalized Kernel matrix is obtained by discarding the training set state,

The approach used above to construct density matrix corresponding to linear kernel matrix can be extended to polynomial kernels (Rebentrost et al., 2014b) .

LMR Technique for Density Operator Exponentiation.

In HHL-based quantum machine learning algorithms , including in the method proposed here, matrix A for the Hamiltonian simulation within the HHL algorithm is based on data.

For example, A can contain the kernel matrix K captured in the quantum system as a density matrix.

Then, one need to be able to efficiently compute e −iK∆t , where K is scaled by the trace of kernel matrix.

Since K is not sparse, a strategy similar to (Lloyd et al., 2014) is adapted for the exponentiation of a non-sparse density matrix:

where S = i,j |i j| ⊗ |j i| is the swap operator and the facts Tr 1 {S(K ⊗ σ)} = Kσ and Tr 1 {(K ⊗ σ)S} = σK are used.

The equation (6) summarizes the LMR technique: approximating e −iK∆t σe iK∆t up to error O(∆t 2 ) is equivalent to simulating a swap operator S, applying it to the state K ⊗ σ and discarding the first system by taking partial trace operation.

Since the swap operator is sparse, its simulation is efficient.

Therefore the LMR trick provides an efficient way to approximate exponentiation of a non-sparse density matrix.

Quantum LS-SVM.

Quantum LS-SVM (Rebentrost et al., 2014b) uses partial trace to construct density operator corresponding to the kernel matrix K. Once the kernel matrix K becomes available as a density operator, the quantum LS-SVM proceeds by applying the HHL algorithm for solving the system of linear equations associated with LS-LSVM, using the LMR technique for performing the density operator exponentiation e −iK∆t where the density matrix K encodes the kernel matrix.

3 QUANTUM SEMI-SUPERVISED LEAST SQUARE SVM.

Semi-Supervised Least Square SVM involves solving the following system of linear equations

In quantum setting the task is to generate |b, α =Â −1 |0, y , where the normalizedÂ = A T r(A) .

The linear system differs from the one in LS-SVM in that instead of K, we have K + KLK.

While this difference is of little significance for classical solvers, in quantum systems we cannot just multiply and then add the matrices and then apply quantum LS-SVM -we are limited by the unitary nature of quantum transformations.

In order to obtain the solution to the quantum Semi-Supervised Least Square SVM, we will use the following steps.

First, we will read in the graph information to obtain scaled graph Laplacian matrix as a density operator.

Next, we will use polynomial Hermitian exponentiation for computing the matrix inverse (K + KLK) −1 .

In the semi-supervised model used here, we assume that we have information on the similarity of the training samples, in a form of graph G that uses n edges to connect similar training samples, represented as m vertices.

We assume that for each sample, G contains its k most similar other samples, that is, the degree of each vertex is d. To have the graph available as a quantum density operator, we observe that the graph Laplacian L is the Gram matrix of the rows of the m × n graph incidence matrix G I , L = G I G T I .

We assume oracle access to the graph adjacency list, allowing us to construct, in superposition, states corresponding to rows of the graph incidence matrix G I

That is, state |v i has probability amplitude

for each edge |t incident with vertex i, and null probability amplitude for all other edges.

In superposition, we prepare a quantum state combining rows of the incidence matrix for all vertices, to obtain

The graph Laplacian matrix L, composed of inner products of the rows of G I , is obtained by discarding the second part of the system,

For computing the matrix inverse (K + KLK) −1 on a quantum computer that runs our quantum machine algorithm and HHL algorithm as a subroutine, we need to efficiently compute e −i(K+KLK)∆t σe i(K+KLK)∆t .

For this purpose we adapt the generalized LMR technique for simulating Hermitian polynomials proposed in (Kimmel et al., 2017) to the specific case of e −i(K+KLK)∆t σe i(K+KLK)∆t .

Simulation of e −iK∆t follows from the original LMR algorithm, and therefore we focus here only on simulation e −iKLK∆t .

The final dynamics (K + KLK) −1 can be obtained by sampling from the two separate output states for e −iKLK∆t and e −iK∆t .

Simulating e iKLK∆t e iKLK∆t e iKLK∆t .

Let D(H) denote the space of density operators associated with state space H. Let K † , K, L ∈ D(H) be the density operators associated with the kernel matrix and the Laplacian, respectively.

We will need two separate systems with the kernel matrix K, to distinguish between them we will denote the first as K † and the second as K; since K is real and symmetric, these are indeed equal.

The kernel and Laplacian matrices K † , K, L are not sparse therefore we adapt the generalized LMR technique for simulating Hermitian polynomials for our specific case

For adapting the generalized LMR technique to our problem we need to generate a quantum state ρ = |0 0| ⊗ ρ + |1 1| ⊗ ρ with T r(ρ + ρ ) = 1, such that

where

is a controlled partial swap in the forward (+S) and backward direction (−S) in time, and

Therefore with one copy of ρ , we obtain the simulation of e −iB∆ up to error O(∆ 2 ).

If we choose the time slice ∆ = δ/t and repeating the above procedure for t 2 /δ times, we are able to simulate e −iBt up to error O(δ) using n = O(t 2 /δ) copies of ρ .

Figure 1 shows the quantum circuit for creating ρ = |0 0| ⊗ ρ + |1 1| ⊗ ρ such that T r(ρ + ρ ) = 1 and B = ρ − ρ = KLK.

The analysis of the steps preformed by the circuit depicted in Fig.1 is as follows.

Let P be the cyclic permutation of three copies of H A that operates as P |j 1 , j 2 , j 3 = |j 3 , j 1 , j 2 .

In operator form it can be written as

The quantum LS-SVM in (Rebentrost et al., 2014b) offers exponential speedup O(log mp) over the classical time complexity for solving SVM as a quadratic problem, which requires time O(log( −1 )poly(p, m)), where is the desired error.

The exponential speedup in p occurs as the result of fast quantum computing of kernel matrix, and relies on the existence of efficient oracle access to data.

The speedup on m is due to applying quantum matrix inversion for solving LS-SVM, which is inherently due to fast algorithm for exponentiation of a resulting non-sparse matrix.

Our algorithm introduces two additional steps: preparing the Laplacian density matrix, and Hamiltonian simulation for KLK.

The first step involves oracle access to a sparse graph adjacency list representation, which is at least as efficient as the oracle access to non-sparse data points.

The Hamiltonian simulation involves simulating a sparse conditional partial swap operator, which results an efficient strategy for applying e −iKLK∆t in timeÕ(log(m)∆t), where the notationÕ hides more slowly growing factors in the simulation (Berry et al., 2007) .

Considerable effort has been devoted into designing fast classical algorithms for training SVMs.

The decomposition-based methods such as SMO (Platt, 1998) are able to efficiently manage problems with large number of features p, but their computational complexities are super-linear in m. Other training strategies (Suykens & Vandewalle, 1999; Fung & Mangasarian, 2005; Keerthi & DeCoste, 2005) are linear in m but scale quadratically in p in the worst case.

The Pegasos algorithm (ShalevShwartz et al., 2011) for non-linear kernel improves the complexity toÕ (m/(λ )), where λ, and are the regularization parameter of SVM and the error of the solution, respectively.

Beyond the classical realm, three quantum algorithms for training linear models have been proposed, the quantum LS-SVM that involves L 2 regularizer (Rebentrost et al., 2014a) , a recently proposed Quantum Sparse SVM which is limited to a linear kernel (Arodz & Saeedi, 2019) , and a quantum training algorithm that solves a maximin problem resulting from a maximum -not average -loss over the training set (Li et al., 2019) .

@highlight

We extend quantum SVMs to semi-supervised setting, to deal with the likely problem of many missing class labels in huge datasets.