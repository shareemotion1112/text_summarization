We recently observed that convolutional filters initialized farthest apart from each other using offthe- shelf pre-computed Grassmannian subspace packing codebooks performed surprisingly well across many datasets.

Through this short paper, we’d like to disseminate some initial results in this regard in the hope that we stimulate the curiosity of the deep-learning community towards considering classical Grassmannian subspace packing results as a source of new ideas for more efficient initialization strategies.

1. Introduction

Standard initialization methods of neural networks are motivated primarily to prevent activations from vanishing or exploding.

Consider FIG0 focuses on the first convolutional layer of a standard CNN model for MNIST digit classification.

As seen, N = 32 different convolutional filter weights of size 3 × 3, denoted by {w k } N =32k=1 , are applied to an image patch, x ij , to yield the 32 channel values associated with the feature tensor extracted.

The feature values computed for the k th channel would be expressed as DISPLAYFORM0 where f (.) is a non-linear activation function, which is typically RELU or tanh, b k ∈ R is the bias term, and in this specific case, N = 32 and m = 3 × 3 = 9.Initialization of these filter weights has been extensively studied in conjunction with the activation functions and state-of-the-art architectures.

There now exists a plethora Preliminary work.

Under review by the International Conference on Machine Learning (ICML).

Do not distribute.of resources 1 that detail the best practices to be followed with regards to the initialization strategy to be used for the specific architecture chosen.

Most deep-learning frameworks now make available a standard repertoire of initialization strategies available for deeplearning researchers, of which the four most common ones we found available off-the-shelf in almost all frameworks were:1.

Lecun uniform/lecun normal BID9 2.

He-uniform BID6 3.

Glorot/Xavier-uniform BID4 4.

Orthogonal (Saxe et al., 2013) We'd posit that it's now part of machine learning folklore BID8 ) that a practitioner would feault to the choice of Xavier initialization if the activation function of the layer is tanh() and He-initialize if the activation function is RELU ().

It is to be noted that for specific architectures such as deepresidual networks, recent works have questioned on the efficacy of the above initialization strategies.

In (Yang & Schoenholz, 2017) , the authors critique the dependence of the optimal init-variances on the depth of the network and propose a novel mean field residual networks framework.

This 'random initialization on the edge of chaos' idea reoccurs in BID5 where they also re-emphasize the supposed efficacy of using swish activation functions over RELU-like functions.

In the context of Binary neural networks, the authors in BID1 has showcased the efficacy of Hadamard initialization over the other techniques.

We note in passing that the default initializations for specific layers and activation-types vary from one deep learning framework to the other, and is often a point of much debate among the practitioners 2 .

The continued efforts of researchers in this specific subfield of deep-learning highlights that much work needs to be done before unanimity can be achieved as to which is the optimal initialization strategy for what combination of architecture/activation function.

The initial set of observations that we'd like to disseminate through this paper sits squarely into this setting of finding that elusive optimal initialization strategy.

We build on the Grassmannian subspace-packing body of work BID2 in experimental mathematics that has also been successfully used in physical layer wireless communications (See (Love & Heath, 2005) , (Prabhu et al., 2009) ) for limited feedback codebook-based downlink beamforming schemes and sparse signal reconstruction (Malioutov et al., 2005) .

The reason why we developed a hunch for trying out this technique is as follows.

Getting the convolution filters to learn distinctive attributes so that we don't end up with a scenario where the learned filters post-training all look the same has an interesting and chequred history.

Back in 2014, in the highly cited work on Visualizing and Understanding Convolutional Networks (Zeiler & Fergus, 2014) , DISPLAYFORM0 the authors propose a set of best-practices for improving upon Alexnet such as choosing a different kernel size, stridelength and use of feature-scale clipping.

They focus on a set of visualizations that demonstrate how the learned features look like before and after applying their set of techniques.

With regards to the feature scale clipping idea, they highlight how this prevents 'one feature (sic) Kernel' from dominating.

They also showcase how the smaller stride and filter-sizes resulted in more "distinctive features and fewer dead features".

This theme of wanting to ensure that each filter learns something different is rather intuitive as this alludes towards more efficient usage of the computational real estate that the conv-nets dispense at the classification problem.

Our ansatz is that an intuitive way of achieving this inter-filter diversity is to ensure that they are initialized furthest apart upon initialization.

Given that these filters reside in R m , this nows becomes the line-packing ( and in general subspace packing) problem.

Here we'd like to inherit the classical Neil Sloane example of the line-packing problem seeking to answer the question:"How should N laser beams passing through a single point be arranged so as to make the angle between any two of the beams as large as possible?

"

Set theoretic definition: The real Grassmann manifold G(m, k) is define as the set of all k-dimensional (linear) subspaces in R m and the Grassmannian N-subspace packing problem is the problem of finding a set of N k-dimensional subspaces in G(m, k) that maximize the minimum pairwise distance between the constituent subspaces in the set.

In this paper, we consider the special case of k = 1 (also termed as the line-packing scenario).

Let Ω m denote the set of unit vectors in R m .

As shown in (Love & Heath, 2005) , for a given (N, m), arranging N unit vectors, w i ∈ Ω m such that the magnitude correlation between any two vectors is as small as possible yields the line-packings with regards to the sine-distance metric defined to be: DISPLAYFORM0 The final packing is represented by a codebook matrix, W = [w 1 |w 2 |...|w 1 , w N ]; w i ∈ Ω m , characterized by the the minimum distance of packing δ(W), which is defined as, DISPLAYFORM1 The Rankin bound BID0 provides the upper bound for this minimum distance and is given by, DISPLAYFORM2 A normalized invariant measure µ introduced on G(m, 1) by the normalized Haar measure on Ω m allows computation of volumes in G(m, 1), which is in turn used to define the density of a given line-packing matrix W. It was shown in (Love & Heath, 2005) that, DISPLAYFORM3 .There exists pre-computed repositories for the best known packings for both complex 3 and real scenarios.

If the mismatch is with regards to N , we suggest finding the largest N such that the tuple (m, 1, N ) exists in the repository.

Akin to orthogonal matrix initializations (Saxe et al., 2013) , which requires square matrices, we are somewhat limited by the choice of the tuple: (m, 1, N ).

The repository from where we sourced the codebooks is limited up to (m = 16, 1, N = 45) (Sloane, 2004) .

One path ahead is to a priori construct packings in Grassmannian manifolds via the alternating projection method described in BID3 .

In this paper, we explore only those architectures whose filter-sizes (m) and the number of filters (N )3.

Experiments

With the intuition that the first few convolutional kernel filters should capture as much diverse features as possible to prevent 'dead kernels' (Zeiler & Fergus, 2014), we experimented with both shallow CNNs with 2 to 4 convolutional layers, as well as standard ResNet-56 models BID7 on different datasets.

As our baseline, we have 2-layer CNNs and 4-layer CNNs as per 1 with standard default Xavier initialization.

In our experiments, we only change the first layer of convolutional filter kernels, as we wish to capture diverse representations and features from the input image, and allow the rest of the network to learn the different combination and activation of the diverse first-layer features.

In comparison with the baselines, where we introduce Grassmannian initializations without biases as trainable and untrainable (fixed) parameters.

For single-channel inputs such as MNIST, KMNIST and Fashion MNIST and assuming that the first convolutional kernels are of size 3 × 3 = 9, we use line packings of (m, 1, N ) where m = 9 and N is the number of output channels.

This is equivalent to finding the best way of 'stabbing' N lines through a 9-D sphere such that the minimum distance between each line is maximized.

We also ran similar experiments on ResNets, where we use a standard ResNet-56 with standard Xavier initialization and Adam optimizer as baseline.

Given images with 3 input channels and assuming that the first convolutional kernel is of size 3 × 3 = 9, we initialize the weights of the first convolutional kernel to be Grassmannian line packings of (m, 1, N ) where m = 9 is the kernel size squared, with N as the number of output channels and N = 3N since we have input channels of size 3.

We then initialize the first layer using this line packing without biases, and train it under both fixed and trainable conditions.

We ran multiple trials on different datasets using with our approaches, and verified that in our runs we achieve better test accuracies on different datasets on shallow architectures, where both fixed and trainable Grassmannian first-layer initializaitons almost consistently achieve higher first-epoch accuracies.

The optimizer used also has a significant impact on the first-epoch test accuracy of initializations.

While Adam and Adadelta with standard initialization outperforms frozen Grassmannian initialization in first-epoch accuracies, trainable Grassmannians still outperforms standard initializations in both cases.

The improvement gained from Grassmannians are most pronounced with SGD as an optimizer.

For deeper networks such as ResNets, we see faster convergence even with Adam optimizer as per 4, and achieves slight improvement in accuracies on the final test-set classification score after training for 200 epochs.

We also attach our code here for reproducibility purposes, while providing a framework for extracting Grassmannians.

TAB0

@highlight

Initialize weights using off-the-shelf Grassmannian codebooks, get  faster training and better accuracy