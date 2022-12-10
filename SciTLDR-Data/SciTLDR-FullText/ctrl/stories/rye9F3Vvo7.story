A restricted Boltzmann machine (RBM) learns a probabilistic distribution over its input samples and has numerous uses like dimensionality reduction, classification and generative modeling.

Conventional RBMs accept vectorized data that dismisses potentially important structural information in the original tensor (multi-way) input.

Matrix-variate and tensor-variate RBMs, named MvRBM and TvRBM, have been proposed but are all restrictive by construction.

This work presents the matrix product operator RBM (MPORBM) that utilizes a tensor network generalization of Mv/TvRBM, preserves input formats in both the visible and hidden layers, and results in higher expressive power.

A novel training algorithm integrating contrastive divergence and an alternating optimization procedure is also developed.

directly applicable to matrix and tensor data.

The first RBM designed for tensor inputs is described in

[6], where the visible layer is represented as a tensor but the hidden layer is still a vector.

Furthermore, 24 the connection between the visible and hidden layers is described by a canonical polyadic (CP) tensor 25 decomposition [7] , which constrains the model representation capability [8] .

Another RBM related 26 model that utilizes tensor input is the matrix-variate RBM (MvRBM) [8] .

The visible and hidden 27 layers in an MvRBM are both matrices.

Nonetheless, to limit the number of parameters, an MvRBM 28 models the connection between the visible and hidden layers through two separate matrices, which 29 restricts the ability of the model to capture correlations between different data modes.

All these issues have motivated this work.

Specifically, we propose a matrix product operator (MPO) DISPLAYFORM0 The "building blocks" of the MPO are the 4-way tensors DISPLAYFORM1 , also called the MPO-cores.

DISPLAYFORM2 DISPLAYFORM3 the summations in BID0 and are the key ingredients in being able to express generic weight tensors W.

The storage complexity of an MPORBM with uniform ranks and dimensions is O(dIJR 2 ), which is layer tensors into a d-way tensor, which is then added elementwise with the corresponding bias tensor.

The final step in the computation of the conditional probability is an elementwise application of the 56 logistic sigmoid function on the resulting tensor.

Let Θ = {B, C, W DISPLAYFORM0 with respect to the model parameter Θ. Similar to the standard RBM BID0 , the expression of the 60 gradient of the log-likelihood is: DISPLAYFORM1 We mainly use the contrastive divergence (CD) procedure to train the MPORBM model.

First, a

Gibbs chain is initialized with one particular training sample V (0) = X train , followed by K times

Gibbs sampling which results in the chain {( summing over all edges.

The derivatives of the log-likelihood with respect to the bias tensors B, C are MPO-cores, which we call CD-SU henceforth, will be demonstrated through numerical experiments.

DISPLAYFORM0 DISPLAYFORM1

In the first experiment, we demonstrate the superior data classification accuracy of MPORBM using Finally, we show that an MPORBM is good at generative modeling exemplified by image completion.

We tested this generative task on the binarized MNIST dataset: one half of the image was provided to

<|TLDR|>

@highlight

Propose a general tensor-based RBM model which can compress the model greatly at the same keep a strong model expression capacity