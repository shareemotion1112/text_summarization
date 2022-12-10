We exploit a recently derived inversion scheme for arbitrary deep neural networks to develop a new semi-supervised learning framework that applies to a wide range of systems and problems.

The approach reaches current state-of-the-art methods on MNIST and provides reasonable performances on SVHN and CIFAR10.

Through the introduced method, residual networks are for the first time applied to semi-supervised tasks.

Experiments with one-dimensional signals highlight the generality of the method.

Importantly, our approach is simple, efficient, and requires no change in the deep network architecture.

Deep neural networks (DNNs) have made great strides recently in a wide range of difficult machine perception tasks.

They consist of parametric functionals f Θ with internal parameters Θ. However, those systems are still trained in a fully supervised fashion using a large set of labeled data, which is tedious and costly to acquire.

Semi-supervised learning relaxes this requirement by leaning Θ based on two datasets: a labeled set D of N training data pairs and an unlabeled set D u of N u training inputs.

Unlabeled training data is useful for learning as unlabeled inputs provide information on the statistical distribution of the data that can both guide the learning required to classify the supervised dataset and characterize the unlabeled samples in D u hence improve generalization.

Limited progress has been made on semi-supervised learning algorithms for DNNs BID15 ; BID16 BID14 , but today's methods suffer from a range of drawbacks, including training instability, lack of topology generalization, and computational complexity.

In this paper, we take two steps forward in semi-supervised learning for DNNs.

First, we introduce an universal methodology to equip any deep neural net with an inverse that enables input reconstruction.

Second, we introduce a new semi-supervised learning approach whose loss function features an additional term based on this aforementioned inverse guiding weight updates such that information contained in unlabeled data are incorporated into the learning process.

Our key insight is that the defined and general inverse function can be easily derived and computed; thus for unlabeled data points we can both compute and minimize the error between the input signal and the estimate provided by applying the inverse function to the network output without extra cost or change in the used model.

The simplicity of this approach, coupled with its universal applicability promise to significantly advance the purview of semi-supervised and unsupervised learning.

The standard approach to DNN inversion was proposed in BID4 , and the only DNN model with reconstruction capabilities is based on autoencoders BID13 .

While more complex topologies have been used, such as stacked convolutional autoencoder BID11 , thereThe semi-supervised with ladder network approach BID15 employs a per-layer denoising reconstruction loss, which enables the system to be viewed as a stacked denoising autoencoder which is a standard and until now only way to tackle unsupervised tasks.

By forcing the last denoising autoencoder to output an encoding describing the class distribution of the input, this deep unsupervised model is turned into a semi-supervised model.

The main drawback of this method is the lack of a clear path to generalize it to other network topologies, such as recurrent or residual networks.

Also, the per-layer "greedy" reconstruction loss might be too restrictive unless correctly weighted pushing the need for a precise and large cross-validation of hyper-parameters.

The probabilistic formulation of deep convolutional nets presented in BID14 natively supports semi-supervised learning.

The main drawbacks of this approach lies in the requirement that the activation functions be ReLU and that the overall network topology follows a deep convolutional network.

Temporal Ensembling for Semi-Supervised Learning BID8 propose to constrain the representations of a same input stimuli to be identical in the latent space despite the presence of dropout noise.

This search of stability in the representation is analogous to the one of a siamese network BID6 but instead of presenting two different inputs, the same is used through two different models (induced by dropout).

This technique provides an explicit loss for the unsupervised examples leading to the Π model just described and a more efficient method denoted as temporal ensembling.

Distributional Smoothing with Virtual Adversarial Training BID12 proposes also a regularization term contraining the regularity of the DNN mapping for a given sample.

Based on this a semi-supervised setting is derived by imposing for the unlabeled samples to maintain a stable DNN.

Those two last described methods are the closest one of the proposed approach in this paper for which, the DNN stability will be replaced by a reconstruction ability, closely related to the DNN stability.

This paper makes two main contributions: First, we propose a simple way to invert any piecewise differentiable mapping, including DNNs.

We provide a formula for the inverse mapping of any DNN that requires no change to its structure.

The mapping is computationally optimal, since the input reconstruction is computed via a backward pass through the network (as is used today for weight updates via backpropagation).

Second, we develop a new optimization framework for semisupervised learning that features penalty terms that leverage the input reconstruction formula.

A range of experiments validate that our method improves significantly on the state-of-the-art for a number of different DNN topologies.

In this section we review the work of BID1 aiming at interpreting DNNs as linear splines.

This interpretation provides a rigorous mathematical justification for the reconstruction in the context of deep learning.

Recent work in BID1 demonstrated that DNNs of many topologies are or can be approximated arbitrary closely by multivariate linear splines.

The upshot of this theory for this paper is that it enables one to easily derive an explicit input-output mapping formula.

As a result, DNNs can be rewritten as a linear spline of the form To illustrate this point we provide for two common topologies the exact input-output mappings.

For a standard deep convolutional neural network (DCN) with succession of convolutions, nonlinearities, and pooling, one has DISPLAYFORM0 DISPLAYFORM1 where z ( ) (x) represents the latent representation at layer for input x. The total number of layers in a DNN is denoted as L and the output of the last layer z (L) (x) is the one before application of the softmax application.

After application of the latter, the output is denoted byŷ(x).

The product terms are from last to first layer as the composition of linear mappings is such that layer 1 is applied on the input, layer 2 on the output of the previous one and so on.

The bias term results simply from the accumulation of all of the per-layer biases after distribution of the following layers' templates.

For a Resnet DNN, one has DISPLAYFORM2 We briefly observe the differences between the templates of the two topologies.

The presence of an extra term in DISPLAYFORM3 σ C ( ) provides stability and a direct linear connection between the input x and all of the inner representations z ( ) (x), hence providing much less information loss sensitivity to the nonlinear activations.

Based on those findings, and by imposing a simple 2 norm upper bound on the templates, it has been shown that the optimal templates DNNs to perform prediction has templates proportional to the input, positively for the belonging class and negatively for the others BID1 .

This way, the loss cross-entropy function is minimized when using softmax final nonlinearity.

Note that this result is specific to this setting.

For example in the case of spherical softmax the optimal templates become null for the incorrect classes of the input.

Theorem 1.

In the case where all inputs have identity norm ||X n || = 1, ∀n and assuming all templates denoted by DISPLAYFORM4 We now leverage the analytical optimal DNN solution to demonstrate that reconstruction is indeed implied by such an optimum.

Based on the previous analysis, it is possible to draw implications based on the theoretical optimal templates of DNNs.

This is formulated through the corollary below.

First, we propose the following inverse of a DNN as DISPLAYFORM0 Following the analysis from a spline point of view, this reconstruction is leveraging the closest input hyperplane, found through the forward step, to represent the input.

As a result this method provides a reconstruction based on the DNN representation of its input and should be part away from the task of exact input reconstruction which is an ill-posed problem in general.

The bias correction present has insightful meaning when compared to known frameworks and their inverse.

In particular, when using ReLU based nonlinearities we will see that this scheme can be assimilated to a composition of soft-thresholding denoising technique.

We present further details in the next section where we also provide ways to efficiently invert a network as well as describing the semi-supervised application.

We now apply the above inverse strategy to a given task with an arbitrary DNN.

As exposed earlier, all the needed changes to support semi-supervised learning happen in the objective training function by adding extra terms.

In our application, we used automatic differentiation (as in TheanoBergstra et al. (2010) and TensorFlowAbadi et al. (2016) ).

Then it is sufficient to change the objective loss function, and all the updates are adapted via the change in the gradients for each of the parameters.

The efficiency of our inversion scheme is due to the fact that any deep network can be rewritten as a linear mapping BID1 .

This leads to a simple derivation of a network inverse defined as f −1 that will be used to derive our unsupervised and semi-supervised loss function via DISPLAYFORM0 The main efficiency argument thus comes from DISPLAYFORM1 which enables one to efficiently compute this matrix on any deep network via differentiation (as it would be done to back-propagate a gradient, for example).

Interestingly for neural networks and many common frameworks such as wavelet thresholding, PCA, etc., the reconstruction error as ( DISPLAYFORM2 is the definition of the inverse transform.

For illustration purposes, Tab.

1 gives some common frameworks for which the reconstruction error represents exactly the reconstruction loss.

DISPLAYFORM3 We now describe how to incorporate this loss for semi-supervised and unsupervised learning.

We first define the reconstruction loss R as DISPLAYFORM4 While we use the mean squared error, any other differentiable reconstruction loss can be used, such as cosine similarity.

We also introduce an additional "specialization" loss defined as the Shannon entropy of the class belonging probability prediction DISPLAYFORM5 This loss is intuitive and complementary to the reconstruction for the semi-supervised task.

In fact, it will force a clustering of the unlabeled examples toward one of the clusters learned from the supervised loss and examples.

We provide below experiments showing the benefits of this extraterm.

As a result, we define our complete loss function as the combination of the standard cross entropy loss for labeled data denoted by L CE (Y n ,ŷ(X n )), the reconstruction loss, and the entropy loss as DISPLAYFORM6 with α, β ∈ [0, 1] 2 .

The parameters α, β are introduced to form a convex combination of the losses,with α controlling the ratio between supervised and unsupervised loss and β the ratio between the two unsupervised losses.

This weighting is important, because the correct combination of the supervised and unsupervised losses will guide learning toward a better optimum (as we now demonstrated via experiments).

We now present results of our approach on a semi-supervised task on the MNIST dataset, where we are able to obtain reasonable performances with different topologies.

MNIST is made of 70000 grayscale images of shape 28 × 28 which is split into a training set of 60000 images and a test set of 10000 images.

We present results for the case with N L = 50 which represents the number of samples from the training set that are labeled and fixed for learning.

All the others samples form the training set are unlabeled and thus used only with the reconstruction and entropy loss minimization.

We perform a search over (α, β) ∈ {0.3, 0.4, 0.5, 0.6, 0.7} × {0.2, 0.3, 0.5}. In addition, 4 different topologies are tested and, for each, mean and max pooling are tested as well as inhibitor DNN (IDNN) as proposed in BID1 .

The latter proposes to stabilize training and remove biases units via introduction of winner-share-all connections.

As would be expected based on the templates differences seen in the previous section, the Resnet topologies are able to reach the best performance.

In particular, wide Resnet is able to outperform previous state-of-the-art results.

Running the proposed semi-supervised scheme on MNIST leads to the results presented in Tab.

2.

We used the Theano and Lasagne libraries; and learning procedures and topologies are detailed in the appendix.

The column 1000 corresponds to the accuracy after training of DNNs using only the supervised loss (α = 1, β = 0) on 1000 labeled data.

Thus, one can see the gap reached with the same network but with a change of loss and 20 times less labeled data.

We further present performances on CIFAR10 with 4000 labeled data in Tab.

3 and SVHN with 500 labeled data in Tab.

4.

For both tasks we constrain ourselves to a deep CNN models, similar as the LargeCNN of BID14 .

Also, one of the cases correspond to the absence of entropy loss when β = 1.

Furthermore to further present the generalization of the inverse technique we provide results with the leaky-ReLU nonlinearity as well as the sigmoid activation function.

We now present and example of our approach on a supervised task on audio database (1D).

It is the Bird10 dataset distributed online and described in BID5 .

The task is to classify 10 bird species from their songs recorded in tropical forest.

It is a subtask of the BirdLifeClef challenge.

75.28 ± 0.2 73.05 ± 0.4 N L CIFAR4000, CIFAR8000, Improved GAN Salimans et al. (2016) 81.37 ± 2.32 82.28 ± 1.82 LadderNetwork BID15 79.6 ± 0.47 -catGAN BID17 80.42 ± 0.46 -DRMM +KL penalty BID14 76.76 -Triple GAN Li et al. (2017) 83.01 ± 0.36 -Semi-Sup Requires a Bad GAN Dai et al. (2017) 85.59 ± 0.30 -ΠModelLaine & Aila FORMULA0 83.45 ± 0.29 - 88.25 ± 1.12 88.39 ± 0.9 (0.85,1) 80.42 ± 2.4 79.77 ± 1.5 N L SVHN500, SVHN1000, Improved GAN Salimans et al. (2016) 81 .

FORMULA0 92.95 ± 0.3 94.57 ± 0.25 VATMiyato et al. FORMULA0 -75.37We train here networks based on raw audio using CNNs as detailed in the appendix.

We vary (α, β) over 10 runs to demonstrate that the non-regularized supervised model is not optimal.

The maximum validation accuracies on the last 100 epochs FIG3 show that the regularized networks tend to learn more slowly, but always generalize better than the not regularized baseline (α = 1, β = 0).

We have presented a well-justified inversion scheme for deep neural networks with an application to semi-supervised learning.

By demonstrating the ability of the method to best current state-of-theart results on MNIST with different possible topologies support the portability of the technique as well as its potential.

These results open up many questions in this yet undeveloped area of DNN inversion, input reconstruction, and their impact on learning and stability.

Among the possible extensions, one can develop the reconstruction loss into a per-layer reconstruction loss.

Doing so, there is the possibility to weight each layer penalty bringing flexibility as well as meaningful reconstruction.

Define the per layer loss as DISPLAYFORM0 with DISPLAYFORM1 Doing so, one can adopt a strategy in favor of high reconstruction objective for inner layers, close to the final latent representation z (L) in order to lessen the reconstruction cost for layers closer to the input X n .

In fact, inputs of standard dataset are usually noisy, with background, and the object of interest only contains a small energy with respect to the total energy of X n .

Another extension would be to update the weighting while performing learning.

Hence, if we denote by t the position in time such as the current epoch or batch, we now have the previous loss becoming DISPLAYFORM2 (13) One approach would be to impose some deterministic policy based on heuristic such as favoring reconstruction at the beginning to then switch to classification and entropy minimization.

Finer approaches could rely on explicit optimization schemes for those coefficients.

One way to perform this, would be to optimize the loss weighting coefficients α, β, γ ( ) after each batch or epoch by backpropagation on the updates weights.

Define DISPLAYFORM3 as a generic iterative update based on a given policy such as gradient descent.

One can thus adopt the following update strategy for the hyper-parameters as DISPLAYFORM4 and so for all hyper-parameters.

Another approach would be to use adversarial training to update those hyper-parameters where both update cooperate trying to accelerate learning.

EBGAN BID18 ) are GANs where the discriminant network D measures the energy of a given input X. D is formulated such as generated data produce high energy and real data produce lower energy.

Same authors propose the use of an auto-encoder to compute such energy function.

We plan to replace this autoencoder using our proposed method to reconstruct X and compute the energy; hence D(X) = R(X) and only one-half the parameters will be needed for D.Finally, our approach opens the possibility of performing unsupervised tasks such as clustering.

In fact, by setting α = 0, we are in a fully unsupervised framework.

Moreover, β can push the mapping f Θ to produce a low-entropy, clustered, representation or rather simply to produce optimal reconstruction.

Even in a fully unsupervised and reconstruction case (α = 0, β = 1), the proposed framework is not similar to a deep-autoencoder for two main reasons.

First, there is no greedy (per layer) reconstruction loss, only the final output is considered in the reconstruction loss.

Second, while in both case there is parameter sharing, in our case there is also "activation" sharing that corresponds to the states (spline) that were used in the forward pass that will also be used for the backward one.

In a deep autoencoder, the backward activation states are induced by the backward projection and will most likely not be equal to the forward ones.

We thank PACA region and NortekMed, and GDR MADICS CNRS EADM action for their support.

We give below the figures of the reconstruction of the same test sample by four different nets : LargeUCNN (α = 0.5, β = 0.5), SmallUCNN (0.6,0.5), 0.5), 0.5) .

The columns from left to right correspond to: the original image, mean-pooling reconstruction, maxpooling reconstruction, and inhibitor connections.

One can see that our network is able to correctly reconstruct the test sample.

<|TLDR|>

@highlight

We exploit an inversion scheme for arbitrary deep neural networks to develop a new semi-supervised learning framework applicable to many topologies.