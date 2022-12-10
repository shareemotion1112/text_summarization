There is a previously identified equivalence between wide fully connected neural networks (FCNs) and Gaussian processes (GPs).

This equivalence enables, for instance, test set predictions that would have resulted from a fully Bayesian, infinitely wide trained FCN to be computed without ever instantiating the FCN, but by instead evaluating the corresponding GP.

In this work, we derive an analogous equivalence for multi-layer convolutional neural networks (CNNs) both with and without pooling layers, and achieve state of the art results on CIFAR10 for GPs without trainable kernels.

We also introduce a Monte Carlo method to estimate the GP corresponding to a given neural network architecture, even in cases where the analytic form has too many terms to be computationally feasible.



Surprisingly, in the absence of pooling layers, the GPs corresponding to CNNs with and without weight sharing are identical.

As a consequence, translation equivariance, beneficial in finite channel CNNs trained with stochastic gradient descent (SGD), is guaranteed to play no role in the Bayesian treatment of the infinite channel limit - a qualitative difference between the two regimes that is not present in the FCN case.

We confirm experimentally, that while in some scenarios the performance of SGD-trained finite CNNs approaches that of the corresponding GPs as the channel count increases, with careful tuning SGD-trained CNNs can significantly outperform their corresponding GPs, suggesting advantages from SGD training compared to fully Bayesian parameter estimation.

@highlight

Finite-width SGD trained CNNs vs. infinitely wide fully Bayesian CNNs. Who wins?

@highlight

The paper establishes a connection between infinite channel Bayesian convolutional neural network and Gaussian processes.