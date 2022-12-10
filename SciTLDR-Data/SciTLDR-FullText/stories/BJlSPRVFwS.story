Asynchronous distributed methods are a popular way to reduce the communication and synchronization costs of large-scale optimization.

Yet, for all their success, little is known about their convergence guarantees in the challenging case of general non-smooth, non-convex objectives, beyond cases where closed-form proximal operator solutions are available.

This is all the more surprising since these objectives are the ones appearing in the training of deep neural networks.



In this paper, we introduce the first convergence analysis covering asynchronous methods in the case of general non-smooth, non-convex objectives.

Our analysis applies to stochastic sub-gradient descent methods both with and without block variable partitioning, and both with and without momentum.

It is phrased in the context of a general probabilistic model of asynchronous scheduling accurately adapted to modern hardware properties.

We validate our analysis experimentally in the context of training deep neural network architectures.

We show their overall successful asymptotic convergence as well as exploring how momentum, synchronization, and partitioning all affect performance.

Training parameters arising in Deep Neural Net architectures is a difficult problem in several ways (Goodfellow et al., 2016) .

First, with multiple layers and nonlinear activation functions such as sigmoid and softmax functions, the ultimate optimization problem is nonconvex.

Second, with ReLU activation functions and max-pooling in convolutional structures, the problem is nonsmooth, i.e., it is not differentiable everywhere, although typically the set of non-differentiable points is a set of measure zero in the space of the parameters.

Finally, in many applications it is unreasonable to load the whole sample size in memory to evaluate the objective function or (sub)gradient, thus samples must be taken, necessitating analysis in a probabilistic framework.

The analysis of parallel optimization algorithms using shared memory architectures, motivated by applications in machine learning, was ushered in by the seminal work of Recht et al. (2011) (although precursors exist, see the references therein).

Further work refined this analysis, e.g. (Liu & Wright, 2015) and expanded it to nonconvex problems, e.g. (Lian et al., 2015) .

However, in all of these results, a very simplistic model of asynchronous computation is presented to analyze the problem.

Notably, it is assumed that every block of the parameter, among the set of blocks of iterates being optimized, has a fixed, equal probability of being chosen at every iteration, with a certain vector of delays that determine how old each block is that is stored in the cache relative to the shared memory.

As one can surmise, this implies complete symmetry with regards to cores reading and computing the different blocks.

This does not correspond to asynchronous computation in practice.

In particular, in the common Non-Uniform Memory Access (NUMA) setting, practical experience has shown that it can be effective for each core to control a set of blocks.

Thus, the choice of blocks will depend on previous iterates, which core was last to update, creating probabilistic dependence between the delay vector and the choice of block.

This exact model is formalized in Cannelli et al., which introduced a new probabilistic model of asynchronous parallel optimization and presented a coordinate-wise updating successive convex approximation algorithm.

In this paper, we are interested in studying parallel asynchronous stochastic subgradient descent for general nonconvex nonsmooth objectives, such as the ones arising in the training of deep neural network architectures.

Currently, there is no work in the literature specifically addressing this problem.

The closest related work is given by Zhu et al. (2018) and , which consider asynchronous proximal gradient methods for solving problems of the form f (x) + g(x), where f is smooth and nonconvex, and g(x) is nonsmooth, with an easily computable closed form prox expression.

This restriction applies to the case of training a neural network which has no ReLUs or max pooling in the architecture itself, i.e., every activation is a smooth function, and there is an additional regularization term, such as an 1 .

These papers derive expected rates of convergence.

In the general case, where the activations themselves are nonsmooth-for instance in the presence of ReLUs-there is no such additive structure, and no proximal operator exists to handle away the non-smoothness and remove the necessity of computing and using subgradients explicitly in the optimization procedure.

This general problem of nonsmooth nonconvex optimization is already difficult (see, e.g., Bagirov et al. (2014) ), and the introduction of stochastically uncertain iterate updates creates an additional challenge.

Classically, the framework of stochastic approximation, with stochastic estimates of the subgradient approximating elements in a differential inclusion that defines a flow towards minimization of the objective function, is a standard, effective approach to analyzing algorithms for this class of problems.

Some texts on the framework include Kushner & Yin (2003) , which we shall reference extensively in the paper, and Borkar (2008) .

See also Ermol'ev & Norkin (1998) and Ruszczyński (1987) for some classical results in convergence of stochastic algorithms for nonconvex nonsmooth optimization.

Interest in stochastic approximation has resurfaced recently sparked by the popularity of Deep Neural Network architectures.

For instance, see the analysis of nonconvex nonsmooth stochastic optimization with an eye towards such models in Davis et al. (2018) and Majewski et al. (2018) .

In this paper, we provide the first analysis for nonsmooth nonconvex stochastic subgradient methods in a parallel asynchronous setting, in the stochastic approximation framework.

For this, we employ the state of the art model of parallel computation introduced in Cannelli et al., which we map onto the analysis framework of Kushner & Yin (2003) .

We prove show that the generic asynchronous stochastic subgradient methods considered are convergent, with probability 1, for nonconvex nonsmooth functions.

This is the first result for this class of algorithms, and it combines the state of the art in these two areas, while extending the scope of the results therein.

Furthermore, given the success of momentum methods (see, e.g., Zhang et al. (2017) ), we consider the momentum variant of the classical subgradient method, again presenting the first convergence analysis for this class of algorithms.

We validate our analysis numerically by demonstrating the performance of asynchronous stochastic subgradient methods of different forms on the problem of ResNet deep network training.

We shall consider variants of asynchronous updating with and without write locks and with and without block variable partitioning, showing the nuances in terms of convergence behavior as depending on these strategies and properties of the computational hardware.

Consider the minimization problem min

where f : R n → R is locally Lipschitz continuous (but could be nonconvex and nonsmooth) and furthermore, it is computationally infeasible to evaluate f (x) or an element of the Clarke subdifferential ∂f (x).

The problem (1) has many applications in machine learning, including the training of parameters in deep neural networks.

In this setting, f (x) is loss function evaluated on some model with x as its parameters, and is dependant on input data A ∈ R n×m and target values y ∈ R m of high dimension, i.e., f (x) = f (x; (A, y)), with x a parameter to optimize with respect to the loss function.

In cases of practical interest, f is decomposable in finite-sum form,

where l : R m × R m → R represents the training loss and {(A i , y i )} is a partition of (A, y).

We are concerned with algorithms that solve (1) in a distributed fashion, i.e., using multiple processing cores.

In particular, we are analyzing the following inconsistent read scenario: before computation begins, each core c is allocated a block of variables I c , for which it is responsible to update.

At each iteration the core modifies a block of variables i k , chosen randomly among I c .

Immediately after core c completes its k-th iteration, it updates the shared memory.

A lock is only placed on the shared memory when a core writes to it, thus the process of reading may result in computations of the function evaluated at variable values that never existed in memory, e.g., block 1 is read by core 1, then core 3 updates block 2, then core 1 reads block 2, and now block 1 is operating on a vector with the values in blocks 1 and 2 not simultaneously at their present local values at any point in time in shared memory.

We shall index iterations to indicate when a core writes a new set of values for the variable into memory.

kc n } be the vector of delays for each component of the variable used to evaluate the subgradient estimate, thus the j-th component of x that is used in the computation of the update at k is actually not

In this paper, we are interested in applying stochastic approximation methods, of which the classic stochastic gradient descent forms a special case.

Since f in (1) is in general nonsmooth, we will exploit subgradient methods.

Denote by ξ k the set of mini-batches used to compute an element of

The set of minibatches ξ kc is chosen uniformly at random from (A, y), independently at each iteration.

By the central limit theorem, the error is asymptotically Gaussian as the total size of the data as well as the size of the mini-batches increases.

Asynchronous System Specification.

We consider a shared-memory system with p processes concurrently and asynchronously performing computations on independent compute-cores.

We interchangeably use the terms process and core.

The memory allows concurrent-read-concurrent-write (CRCW) 1 access.

The shared-memory system offers word-sized atomic read and fetch-and-add (faa) primitives.

Processes use faa to update the components of the variable.

We now recall the stochastic subgradient algorithm under asynchronous updating in Algorithm 1, from the perspective of the individual cores.

The update of the iterate performed by

where m is the momentum constant, required to satisfy 0 < m < 1 Sample i from the variables I c corresponding to c. Sample ξ.

Compute a subgradient estimate g kc , local to k c

Write, with a lock, to the shared memory vector partition

Update, with a lock, to the shared memory vector partition

k c = k c + 1 8: end while

For the discrete time probabilistic model of computation introduced in Cannelli et al., we must present the basic requirements that must hold across cores.

In particular, it is reasonable to expect that if some core is entirely faulty, or exponentially deccelerates in its computation, convergence should not be expected to be attained.

Otherwise we wish to make the probabilistic assumption governing the asynchronous update scheduling as general as possible in allowing for a variety of possible architectures.

The details of the probabilistic assumptions are technical and left to the Supplementary Material.

It can be verified that the stochastic approxmation framework discussed in the next section detailing the convergence satisfies these assumptions.

We have the standard assumption about the stochastic sub-gradient estimates.

These assumptions hold under the standard stochastic gradient approach wherein one samples some subset ξ ⊆ {1, ..., M } of mini-batches uniformly from the set of size |ξ| subsets of {1, ..., M }, done independently at each iteration.

This results in independent noise at each iteration being applied to the stochastic subgradient term.

From these mini-batches ξ, a subgradient is taken for each j ∈ ξ and averaged.

Assumption 3.1.

The stochastic subgradient estimates g(x, ξ) satisfy,

where β(x) defines a bias term that is zero if f (·) is continuously differentiable at x.

We provide some more details on the "global" model of asynchronous stochastic updating in the Supplementary material.

In this section, we shall redefine the algorithm and its associated model presented in the previous section in a framework appropriate for analysis from the stochastic approximation perspective.

Consider the Algorithm described as such, for data block i with respective iteration k,

where Y j,i is the estimate of the partial subgradient with respect to block variables indexed by i at local iteration j.

In the context of Algorithm 1, the step size is defined to be the subsequence {γ

l } where l is the iteration index for the core corresponding to block i.

Thus it takes the subsequence of γ k for which i k = i is the block of variables being modified.

to denote a selection of some element of the subgradient, with respect to block

These are standard conditions implied by the sampling procedure in stochastic gradient methods, introduced by the original Robbins-Monro method (Robbins & Monro, 1985) .

In Stochastic Approximation, the standard approach is to formulate a dynamic system or differential inclusion that the sequence of iterates approaches asymptotically.

For this reason, we introduce real time into the model of asynchronous computation, looking at the actual time elapsed between iterations for each block i.

Define δτ k,i to be the real elapsed time between the k-th and k + 1-st iteration for block i.

We let T k,i = k−1 j=0 δτ j,i and define for σ ≥ 0, p l (σ) = min{j : T j,i ≥ σ} the first iteration at or after σ.

We assume now that the step-size sequence comes from an underlying real function, i.e.,

We now define new σ-algebras F k,i and F + k,i defined to measure the random variables {{x 0 }, {Y j−1,i : j, i with T j,i < T k+1,i }, {T j,i : j, i with T j,i ≤ T k+1,i }} , and, {{x 0 }, {Y j−1,i : j, i with T j,i ≤ T k+1,i }, {T j,i : j, i with T j,i ≤ T k+1,i }} , indicating the set of events up to, and up to and including the computed noisy update at k, respectively.

Note that each of these constructions is still consistent with a core updating different blocks at random, with δτ k,i arising from an underlying distribution for δτ k,c(i) .

Let us relate these σ-algebras to those in the previous section.

Note that this takes subsets of random

The form of Y k,i defined above incorporates the random variable d k and i k , as in which components are updated and the age of the information used by where the subgradient is evaluated, as well as ξ k by the presence of the Martingale difference noise.

For any sequence Z k,i we write Z σ k,i = Z pi(σ)+k,i , where p i (σ) is the least integer greater than or equal to σ.

Thus, let δτ σ k,i denote the inter-update times for block i starting at the first update at or after σ, and γ σ k,i the associated step sizes.

.

We introduce piecewise constant interpolations of the vectors in real-time given by,

Now we detail the assumptions on the real delay times.

These ensure that the real-time delays do not grow without bound, either on average, or on relevantly substantial probability mass.

Intuitively, this means that it is highly unlikely that any core deccelerates exponentially in its computation speed.

and there is aū such that for any compact set A,

Assumption 3.4.

It holds that,

This assumption holds if, e.g., the set of x such that f (·) is not continuously differentiable at x is of measure zero, which is the case for objectives of every DNN architecture the authors are aware of.

As mentioned earlier, the primary goal of the previous section is to define a stochastic process that approximates some real-time process asymptotically, with this real-time process defined by dynamics for which at the limit the path converges to a stationary point.

In particular, we shall see that the process defined for the iterate time scale approximates the path of a differential inclusion,

and we shall see that this path defines stationary points of f (·).

We must define the notion of an invariant set for a differential inclusion (DI).

Definition 3.1.

A set Λ ⊂ R n is an invariant set for a DIẋ ∈ g(x) if for all x 0 ∈ Λ, there is a solution x(t), −∞ < t < ∞ that lies entirely in Λ and satisfies x(0) = x 0 .

Now we state our main result.

Its complete proof can be found in the Supplementary Material.

Theorem 3.1.

Let all the stated Assumptions hold.

Then, the following system of differential inclusions,

holds for any u satisfying 3.3.

On large intervals [0, T ],x σ (·) spends nearly all of its time, with the fraction going to one as T → ∞ and σ → ∞ in a small neighborhood of a bounded invariant set of (5).

This Theorem shows weak convergence.

The extension to convergence with probability one is straightforward and described in the Supplementary material.

Finally, we wish to characterize the properties of this invariant set.

From Corollary 5.11 (Davis et al., 2018) , we can conclude that problems arising in training of deep neural network architectures, wherein f (x) = l(y j , a L ) with l(·) one of several standard loss functions, including logistic or Hinge loss, and a i = ρ i (V i (x)a i−1 ) or i = 1, ..., L layers, are activation functions, which are piece-wise defined to be log x, e x , max(0, x) or log(1 + e x ), are such that their set of invariants {x * } for its associated differential inclusion satisfies 0 ∈ ∂f (x * ), and furthermore the values f (x k ) for any iterative algorithm generating {x k } such that x k → x * , an invariant of f (x), converge.

Note that the differential inclusions defined above ensure asymptotic convergence to block-wise stationarity, i.e., 0 ∈ ∂ i f (x) for all i. It is clear, however, that every stationary point is also blockwise stationary, i.e., that 0 ∈ ∂f (x) implies 0 ∈ ∂

i f (x) for all i. In practice, the set of block-wise stationary points which are not stationary is not large.

One can alternatively consider a variant of the algorithm wherein every core updates the entire vector (thus there is no block partitioning) but locks the shared memory whenever it either reads of writes from it.

The same analysis applies to such a procedure.

In particular, this amounts to i k = {1, ..., n} for all k and every limit of x σ (t) as either σ → ∞ or t → ∞ is a critical point of f (x) and, with probability one, asymptotically the algorithm converges to a critical point of f (x) (i.e., x such that 0 ∈ ∂f (x)).

We describe an experimental evaluation comparing the following algorithms:

WIASSM: Write Inconsistent Asynchronous Stochastic Subgradient Method with lock-free read and updates of x k,i .

This procedure applied to smooth strongly-convex and smooth nonconvex f (x) is known as HogWild!

in Recht et al. (2011) and AsySG-incon in Lian et al. (2015) , respectively, in the literature.

Convergence analysis of HogWild! and AsySG-incon additionally required sparsity of x. They have no provable convergence guarantee for nonsmooth nonconvex models.

WCASSM: Write Consistent Asynchronous stochastic subgradient method.

WCASSM differs from WIASSM in its use of locks to update x k,i to make consistent writes.

AsySG-con in Lian et al. (2015) is its counterpart for smooth nonconvex f (x) and sparse x. Figure 1: We plotted the train accuracy and generalization (test) loss and accuracy trajectories for the methods.

SGD runs a single process, whereas the asynchronous methods run 10 concurrent processes.

In this set of experiments we have no momentum correction.

The WIASSM and WCASSM demonstrate better convergence per epoch compared to PASSM.

Note that, the single process executing SGD iterations has a better opportunity to use CUDA threads as there is no concurrent use of GPUs by multiple processes.

The per epoch performance of PASSM matches that of SGD inferring that amount of subgradient updates are almost identical: in the former it is done collectively by all the concurrent processes accessing disjoint set of tensors, whereas, in the latter it is done by a single process using comparable amount of parallization.

We used a momentum = 0.9.

It can be observed that with momentum correction the convergence of PASSM improves significantly.

Mitliagkas et al. Mitliagkas et al. (2016) experimentally showed that the degree of asynchrony directly relates to momentum; our experiments show that the relative gain in terms of convergence per epoch by momentum correction is better for PASSM that exhibits more asynchrony compared to WCASSM, which uses locks for write consistency.

The presented Partitioned Asynchronous Stochastic Subgradient Method.

We read as well as update x k,i lock-free asynchronously.

Hyper-parameters.

For each of the methods, we adopt a decreasing step size strategy γ k,i = (α j × γ)/ √ k, where α j > 0 is a constant for the j th processing core.

γ is fixed initially.

In each of the methods we use an L2 penalty in the form of a weight-decay of 0.0005.

Additionally, we introduced an L1 penalty of 0.0001 that simply gets added to the gradients after it has been put through the L2 penalty.

In accordance with the theory, we explored the effect of momentum correction: we have two sets of benchmarks, one without momentum and another with a constant momentum of 0.9 while checking the convergence with epochs.

In all of the above methods we load the datasets in mini-batches of size 64.

We keep the hyper-parameters, in particular, learning rate and mini-batch-size, identical across methods for the sake of statistical fairness.

In a shared-memory setting, there is not much to exploit on the front of saving on communication cost as some existing works do Goyal et al. (2017) ; and the computing units, see the system setting and the implementation below, are anyway utilized to their entirety by way of efficient data-parallelization.

Dataset and Networks.

We used CIFAR10 data set of RGB images Krizhevsky (2009) .

It contains 50000 labeled images for training and 10000 labeled images for testing.

We trained a well known We used momentum = 0.9 in each of them.

As described, a separate concurrent process keeps on saving a snapshot of the shared model at an interval of 1 minute, simultaneously with the training processes.

Firstly, it can be observed that the convergence of PASSM is faster compared to the other two asynchronous methods for identical number of processes.

This can be understood in terms of block partitioning the model across processes: it helps reducing the synchronization cost and thereby potentially speeds up the data processing per unit time.

Furthermore, we clearly gain in terms of convergence per unit time when we increase the number of processes in PASSM.

In contrast, we note that the use of locks by WCASSM actually slows it down when we increase the number of processes.

This set of experiments demonstrate that PASSM has better convergence with respect to wall-clock time in addition to the scalability with parallel resources.

CNN model Resnet18 He et al. (2016) .

ResNet18 has a blocked architecture -of residual blockstotaling 18 convolution layers.

Each residual block is followed by a ReLU activation causing nonlinearity.

Evidently, training of this neural network offers general nonsmooth nonconvex optimization problems.

System Specification.

We benchmarked the implementations on a NUMA workstation -2 sockets, 10 cores apiece, running at 2.4GHz (Intel(R) Xeon(R) E5-2640), HT enabled 40 logical cores, Linux 4.18.0-0.bpo.1-amd64 (Debian 9) -containing 4 Nvidia GeForce GTX 1080 GPUs.

For a fair evaluation of scalability with cores, we bind the processes restricting their computations -in particular, the cost of data load -to individual CPU cores.

In this setting, to evaluate the scalability with respect to wall-clock-time by increasing the availability of parallel resources, we run the experiments with 5 and 10 processes, which do not migrate across CPU sockets.

For evaluation with respect to time, we employed a separate concurrent process that keeps on saving a snapshot of the shared model at an interval of 1 minute.

Asynchronous Implementation.

We implemented the asynchronous methods using the open-source Pytorch library Paszke et al. (2017) and the multi-processing framework of Python.

Given the multi-GPU environment, which could excellently exploit data-parallel computation, therefore, we used the nn.

DataParallel() module of Pytorch library.

Thereby, a CNN instance is centrally allocated on one of the GPUs and computations -forward pass to compute the model over a computation graph and backward pass to compute the sub-gradients thereon -employ peer GPUs by replicating the central instance on them for each mini-batch in a data-parallel way.

Effectively, the computation of stochastic subgradients happen over each GPU and they are summed and added to the central instance.

Note that, this way of implementation exploits parallel resources while effectively simulating a shared-memory asynchronous environment.

Model Partitioning.

Unlike PASSM, the methods WIASSM, WCASSM and SGD do not partition the model and compute the stochastic subgradients over the entire computation graph of a CNN via backpropagation provided by the autograd module of Pytorch.

PASSM partitions the list of leaves, which are tensors corresponding to the weights and biases, of the computation graph into blocks.

While computing the stochastic subgradients with respect to a block, we switch off the requires_grad flag of the tensors corresponding to other blocks during backpropagation.

This particular implementation component results in some savings in stochastic sub-gradient computation with respect layers relatively closer to the output.

Keeping this in view, we assigned blocks containing s i ≥ L/p parameter components, where L is the model size and p is the number of processes, to the processes P i computing stochastic sub-gradients corresponding to layers closer to output.

Whereas, the process that computes sub-gradient of the layer-block closest to the input is assigned a block containing less than L/p parameter components.

The assignments s i aim to balance computation We plotted test-accuracy in terms of Top1 correct match % vs time (in minutes).

In can be observed that PASSM offers faster convergence per unit time in accuracy as well compared to the other two asynchronous methods.

load, however, it varies across layers depending on the size of the assigned leaves in terms of parameter component.

Nevertheless, a blocked architecture such as ResNet does not allow much scope of computation-cost saving on this count: we observed an insignificant difference in average processing time for the same number of epochs irrespective of switching off the requires_grad flag.

Notice that, this is not a model parallelization and the stochastic subgradient computation with respect to a leaf depends on the computation path leading to the output.

Irrespective of partitioning the model, the multi-GPU-based data-parallel implementation utilizes replication and data partitioning over GPUs while processing a mini-batch.

The experimental observations are described in Figures 1, 2, 3 , and 4.

Summary.

The block partitioning design of PASSM has its efficiency in the following: 1) it reduces the cost of optimization per process, since the parameter is partitioned.

Note that, in case of neural networks, where backpropagation processes almost the entire computation graph irrespective of the location of the leaf, in particular in a blocked architecture such as ResNet, PASSM clearly misses out saving subgradient computation cost by way of parallelization; it can be significantly better if the subgradients with respect to the blocks could be computed independently; and 2) reduces memory traffic and potential write conflicts between processors which we observe in terms of better convergence per unit time.

And finally, it is pertinent to highlight that we also observed that momentum correction improves the convergence per epoch of the block partitioning approach whose performance was way lower if we did not use it.

In this paper we analyzed the convergence theory of asynchronous stochastic subgradient descent.

We found that the state of the art probabilistic model on asynchronous parallel architecture applied to the stochastic subgradient method, with and without the use of momentum, is consistent with standard theory in stochastic approximation and asymptotic convergence with probability one holds for the method under the most general setting of asynchrony.

We presented numerical results that indicate some possible performance variabilities in three types of asynchrony: block partitioning inconsistent read (for which the above convergence theory applies), full-variable-update consistent write (for which the above convergence theory also applies), and full-variable-update inconsistent read/write (for which no convergence theory exists).

Here we give a few more details describing the relation of the probabilistic model of asynchrony to the underlying hardware properties, as modeled in Cannelli et al..

In this section, we present k as a global counter, indicating sequential updates of any block among the variables.

In iteration k, the updated iterate x k+1 i k depends on a random vector ζ

of ζ k depends on the underlying scheduling or message passing protocol.

We use the following formulation, which applies to a variety of architectures.

.., ζ t ) be the stochastic process representing the evolution of the blocks and minibatches used, as well as the iterate delays.

The σ-algebra F is obtained as follows.

Let the cylinder

Consider the conditional distribution of ζ k+1 given ζ 0:k ,

we have the following assumptions on the probabilities of block selection and the delays, Assumption 6.1.

The random variables ζ k satisfy,

1.

There exists a δ such that d

for some p min > 0.

3.

It holds that,

The first condition indicates that there is some maximum possible delay in the vectors, that each element of x used in the computation of x k+1 i k is not too old.

The second is an irreducibility condition that there is a positive probability for any block or minibatch to be chosen, given any state of previous realizations of {ζ k }.

The last assumption indicates that the set of events in Ω that asymptotically go to zero in conditional probability are of measure zero.

In order to enforce global convergence, we wish to use a diminishing step-size.

However, at the same time, as synchronization is to be avoided, there must not be a global counter indicating the rate of decrease of the step-size.

In particular, each core will have its own local step size γ ν(c k ,k) where c k is the core, and, defining the random variable Z k as the component of {1, ...,c} that is active at iteration k, the random variable denoting the number of updates performed by core c k , denoted by ν(k) is given by ν(k)

In addition, noting that it has been observed that in practice, partitioning variable blocks across cores is more efficient than allowing every processor to have the ability to choose across every variable block (Liu & Wright, 2015) .

Thus we partition the blocks of variables across cores.

We can thus denote c k as being defined uniquely by i k , the block variable index updated at iteration k.

for some subsequence, which is antithetical to Assumption 3.1, Part 2.

Thus, note that the stepsizes γ ν(c k ,k) satisfy, where the limit of the sequence is taken in probability,

which is an assumption for the analysis of asynchronous parallel algorithms in Borkar (2008).

We are now ready to present Algorithm 2.

This is presented from the "global" iteration counter perspective.

Input: x 0 .

1: while Not converged and k < k max do 2:

Update

Update

Set k = k + 1 6: end while 7 APPENDIX B: PRELIMINARY ASSUMPTIONS AND LEMMAS

Thus, so is

Proof.

Uniform integrability of {Y k,i , Y σ k,i ; k, i} follows from Assumption 3.2, part 3.

The uniform integrability of

; k, i follows from 0 < m < 1 and the fact that a geometric sum of a uniformly integrable sequence is uniformly integrable.

Now we define some terminology arising in the theory of weak convergence.

We present a result indicating sufficient conditions for a property called tightness.

Theorem 7.1. (Kushner & Yin, 2003, Theorem 7.3.

3) Consider a sequence of processes {A k (·)} with paths in D(−∞, ∞) such that for all δ > 0 and each t in a dense set of (−∞, ∞) there is a compact set K δ,t such that, inf

and for any T > 0,

If a sequence is tight then every weak sense limit process is also a continuous time process.

We say that A k (t) converges weakly to A if,

for any bounded and continuous real-valued function F (·) on R n .

Weak convergence is defined in terms of the Skorohod topology, a technical topology weaker than the topology of uniform convergence on bounded intervals, defined in Billingsley (1968) .

Convergence of a function f n (·) to f (·) in the Skorohod topology is equivalent to uniform convergence on each bounded time interval.

We denote by D j [0, ∞) the j-fold product space of real-valued functions on the interval [0, ∞) that are right continuous with left-hand limits, with the Skorohod topology.

It is a complete and separable metric space.

Much of the proof of the main Theorem can be taken from the analagous result in Chapter 12 of Kushner & Yin (2003) , which considers a particular model of asynchronous stochastic approximation.

As we introduced a slightly different model from the literature, some of the details of the procedure are now different, and furthermore we introduced momentum to the algorithm, and thus in the next section we indicate how to treat the distinctions in the proof and show that the result still holds.

By Theorem 8.6, Chapter 3 in Ethier & Kurtz (2009) a sufficient condition for tightness of a sequence {A n (·)} is that for each δ > 0 and each t in a dense set in (−∞, ∞), there is a compact set K δ,t such that inf n P[A n (t) ∈ K δ,t ] ≥ 1 − δ and for each positive T , lim δ→0 lim sup n sup |τ |≤T, s≤δ E [|A n (τ + s) − A n (τ )|] = 0.

Now since Y k,i is uniformly bounded, and Y σ k,i (·) is its interpolation with jumps only at t being equal to some T k,i , it holds that for all i, .

This implies the Lipschitz continuity of the subsequence limits with probability one, which exist in the weak sense by Prohorov's Theorem, Theorems 6.1 and 6.2 (Billingsley, 2013) .

As σ → ∞ we denote the weakly convergent subsequence's weak sense limits by,

Note that, x i (t) =x i (τ i (t)), x i (t) = x i (N i (t)), N i (τ i (t)) = t.

with a set-valued map S(x, T, φ), and by the noise structure of the assumptions, it can easily be seen thatL exists for all possible values of x, T and φ in the notation of the paper.

One can see that the uniqueness appears once in the beginning of the proof of Theorem 3.1 with the existence of this T 1 such that the trajectory lies in a specific ball around the limit point for t ≥ T 1 .

This can be replaced by the trajectory lying in this ball around the invariant set, for T 1 defined as the supremum of sucĥ T 1 associated with every possible subgradient, i.e., element of the DI.

Since the subgradient is a compact set and is upper semicontinuous, this supremum exists.

Finally, note that Assumption 3.2 is as Assumption 4.1 in Dupuis & Kushner (1989) and thus similarly implies Theorem 4.1 and Theorem 5.3.

This proves that as σ → ∞, w.p.1 x σ (·) converges to an invariant set of the differential inclusion.

@highlight

Asymptotic convergence for stochastic subgradien method with momentum under general parallel asynchronous computation for general nonconvex nonsmooth optimization