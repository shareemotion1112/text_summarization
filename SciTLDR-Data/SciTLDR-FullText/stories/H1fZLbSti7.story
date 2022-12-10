We propose a software framework based on ideas of the Learning-Compression algorithm , that allows one to compress any neural network by different compression mechanisms (pruning, quantization, low-rank, etc.).

By design, the learning of the neural net (handled by SGD) is decoupled from the compression of its parameters (handled by a signal compression function), so that the framework can be easily extended to handle different combinations of neural net and compression type.

In addition, it has other advantages, such as easy integration with deep learning frameworks, efficient training time, competitive practical performance in the loss-compression tradeoff, and reasonable convergence guarantees.

Our toolkit is written in Python and Pytorch and we plan to make it available by the workshop time, and eventually open it for contributions from the community.

the workshop time, and eventually open it for contributions from the community.

With the great success of neural network in solving practical problems in various fields (vision, NLP, 12 etc.) there has been an emergence of research in neural network compression techniques that allows 13 to compress these large models in terms of memory, computation and/or power requirements.

At 14 present many ad-hoc solutions have been proposed that typically solve one specific type of compres-15 sion (binarization and quantization [2, 5, BID10 11, 18, 23, 24] , pruning [9, BID13 15, 16 , 21], low-rank or 16 tensor factorization [6, 7, 12-14, 17, 19, 20, 22] , etc.), as well as several submissions to the present 17 workshop.

Among the various research strands in neural net compression, in our view a fundamental problem 19 is that in practice one does not know what the best type of compression (or combination of compres-20 sion types) may be the best for a given network.

In principle, it would be possible to try different 21 existing algorithms, assuming one can find an implementation for them.

We seek a solution that 22 directly addresses this problem and can potentially allow non-expert end users to compress models 23 easily and effectively BID0 .

It is based on a recently proposed compression framework, the LC algorithm In this paper, we describe our ongoing efforts in building a software implementation that can cap-34 italize on the modularity of the LC algorithm.

At present this handles 1) (C step) various forms 35 of quantization, pruning and low-rank compression, and we will soon add combinations of those 36 and further compression types; and 2) (L step) various types of deep net models.

Our framework is 37 written in Python and Pytorch.

We plan to make it available online as open source by the time of the 38 workshop.

We also hope that interested researchers and developers will eventually contribute their 39 own routines for signal compression or for training of specific neural net architectures.

1 Model compression as a constrained optimization problem

Assume we have a previously trained model with weights w = arg min w L(w).

This is our refer-42 ence model, which represents the best loss we can achieve without compression.

The "Learning-

Compression" paper BID0 defines compression as finding a low-dimensional parameterization ∆(Θ) 44 of w in terms of Q < P parameters Θ. The goal is to find such Θ that its corresponding model has 45 (locally) optimal loss.

Therefore the model compression as a constrained optimization problem is 46 defined as: DISPLAYFORM0 Compression and decompression are usually seen as algorithms, but here they are regarded as math-

ematical mappings in parameter space.

The decompression mapping ∆: Θ ∈ R Q → w ∈ R P maps 49 a low-dimensional parameterization to uncompressed model weights and the compression mapping 50 Π(w) = arg min Θ w − ∆(Θ) 2 behaves as its "inverse".

The problem in FORMULA0 nates two generic steps while slowly driving the penalty parameter µ → ∞: DISPLAYFORM0 2 .

This is a regular training of the un-56 compressed model but with a quadratic regularization term.

This step is independent of the 57 compression type.

• C (compression) step: DISPLAYFORM0 .

This means finding the best

(lossy) compression of w (the current uncompressed model) in the ℓ 2 sense (orthogonal 60 projection on the feasible set), and corresponds to our definition of the compression map-

ping Π. This step is independent of the loss, training set and task.

The LC algorithm defines a continuous path (w(µ), Θ(µ)) which, under some mild assumptions,

converges to a stationary point (typically a minimizer) of the constrained problem.

The beginning 64 of this path, for µ → 0 + , corresponds to training the reference model and then compressing it 65 disregarding the loss (direct compression), a simple but suboptimal approach popular in practice.

Optimizing the L and C steps The L step can be solved by stochastic gradient descent (clipping 67 the step sizes so they never exceed but the top-κ weights (where κ depends on the sparsifying norm used) BID2 .

FIG3 In the pseudo-code there is only one compression mapping ∆(Θ), however, practically we want to 74 mix and match, e.g. compress first layer using quantization and all remaining ones using pruning. and sees its weights as a vector containing P values where P is cardinality of union of all w i , e.g. DISPLAYFORM0 This is achieved by a weights view data structure.

DISPLAYFORM1 The compression mappings assumes a specific structure, e.g. low-rank compression expects a ma-84 trix, while pruning expects a vector.

Therefore, we define a compression view data structure that An implementation of L-step is a Python function which will be invoked with a new value of penalty

(the term ∆(Θ)) and it must return an updated w that minimizes the L(w) DISPLAYFORM0

This step is similar to the reference network learning and requires minimal modifications to include 100 penalty terms.

We also note that it is independent of compression, and needs to be done once for a 101 network, and can be re-used for other compression and combinations.

A list of compression tasks is list of tuples where each tuple contains: compression view of w i and 103 compression function Π i which maps subset of weights w to a specific compression user wants to 104 achieve, e.g. first layer to be quantized, second layer to be pruned.

Library operations.

As soon as user invokes lc.run function with inputs described in previous

@highlight

We propose a software framework based on ideas of the Learning-Compression algorithm , that allows one to compress any neural network by different compression mechanisms (pruning, quantization, low-rank, etc.).

@highlight

This paper presents the design of a software library that makes it easier for the user to compress their networks by hiding away the details of the compression methods.