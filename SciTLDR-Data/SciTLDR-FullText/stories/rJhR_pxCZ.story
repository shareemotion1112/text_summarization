As deep learning-based classifiers are increasingly adopted in real-world applications, the importance of understanding how a particular label is chosen grows.

Single decision trees are an example of a simple, interpretable classifier, but are unsuitable for use with complex, high-dimensional data.

On the other hand, the variational autoencoder (VAE) is designed to learn a factored, low-dimensional representation of data, but typically encodes high-likelihood data in an intrinsically non-separable way.

We introduce the differentiable decision tree (DDT) as a modular component of deep networks and a simple, differentiable loss function that allows for end-to-end optimization of a deep network to compress high-dimensional data for classification by a single decision tree.

We also explore the power of labeled data in a  supervised VAE (SVAE) with a Gaussian mixture prior, which leverages label information to produce a high-quality generative model with improved bounds on log-likelihood.

We combine the SVAE with the DDT to get our classifier+VAE (C+VAE), which is competitive in both classification error and log-likelihood, despite optimizing both simultaneously and using a very simple encoder/decoder architecture.

While deep learning approaches are very effective in many classification problems, interpretability of the classifier (why a particular classification was made) can be very difficult, yet critical for many applications.

Decision trees are highly interpretable classifiers, so long as the data is encoded such that the classes can be easily separated.

We present a differentiable decision tree (DDT) that we connect to a variational autoencoder (VAE) to learn an embedding of the data that the tree can classify with low expected loss.

The expected loss of the DDT is differentiable, so standard gradient-based methods may be applied in training.

Since we work in a supervised learning setting, it is natural to exploit the label information when training the VAE.

Thus, we employ a supervised VAE (SVAE) that uses a class-specific Gaussian mixture distribution as its prior distribution.

We found that the SVAE was very effective in exploiting label information, resulting in improved log-likelihood due to separation of classes in latent space.

Further, when we combined SVAE with DDT (yielding our Classifier+VAE, or C+VAE), we got a model that is competitive in both classification error and log-likelihood, despite optimizing both simultaneously and using a very simple encoder/decoder architecture.

Further, the resultant decision tree revealed clear semantic meanings in its internal nodes.

Our first contribution is a demonstration of the power of labeled data in autoencoding, using a simple class-based Gaussian mixture model as a VAE's prior, trained with fully labeled data.

Specifically, our VAE objective function regularizes w.r.t.

a class-specific Gaussian mixture model rather than to N (0, I).

Training this supervised VAE (SVAE) requires class labels, but results in better loglikelihood bounds than an unmodified VAE and excellent image generation on MNIST data, very effectively taking advantage of the class information.

Our second contribution is a differentiable decision tree (DDT), which allows us to differentiate the expected loss of a specific tree w.r.t.

a specific distribution.

This is applicable to learning embeddings, allowing us to compute the gradient of expected loss for weight updates.

Our third contribution combines SVAE with DDT to get Classifier+VAE (C+VAE), which learns a latent variable distribution suitable for both classification and generation simultaneously.

Trained on MNIST data, C+VAE produced an encoder that is competitive in both classification error and log-likelihood, using very simple encoders and decoders.

Our final contribution is an analysis of the interpretability of a DDT trained on MNIST.

Each internal node of the tree tests one of the 50 encoded dimensions when making a classification decision.

We examine the values of the MNIST test data in the encoded dimensions used by the tree, analyzing the semantics of each dimension.

We found that the dimensions used by the tree to discriminate correspond to meaningful macro-features learned by the encoder, and that the tree itself effectively summarizes the classification process.

The rest of this paper is organized as follows.

In Section 2 we give relevant background.

Then in Section 3 we describe the SVAE and present our differentiable decision tree and our combined model.

Our experimental results appear in Section 4.

Finally, we present related work in Section 5, and conclude in Section 6 with a discussion of future work.

We begin with a dataset of pairs, {(x 1 , y 1 ), . . .

, (x n , y n )}, where x i ∈ X ⊂ R m is the ith observation and the class label is y i . introduced the variational autoencoder (VAE) as a latent variable model for efficient maximum marginal likelihood learning.

The VAE performs density estimation on p(x, z) where z are latent variables, to maximize the likelihood of the observed training data x: DISPLAYFORM0 Since this marginal likelihood is difficult to work with directly for non-trivial models, instead a parametric inference model q(z | x) is used to optimize the variational lower bound on the marginal log-likelihood: DISPLAYFORM1 where θ indicates the parameters of the encoder p and decoder q models, and KL(·||·) is the Kullback-Leibler divergence BID8 ).

The VAE optimizes the lower bound by reparameterizing q(z | x) ).The first term of L above corresponds to the reconstruction error of the decoder p(x | z), and the second term regularizes the distribution parameterized by the encoder p(z | x) to minimize the K-L divergence from a chosen prior distribution, usually an isotropic, centered Gaussian.

The simplicity of this prior distribution has the downside of restricting the flexibility of latent variable assignment, but allows the VAE to be easily used as a decoder-based generative model as DISPLAYFORM2 While more sophisticated approximate posteriors have been used to improve variational inference BID14 ; BID12 ; BID15 ; BID6 ; BID1 ), for the sake of efficiency and simplicity, in our work we use only the simple approximate posterior q(z | x) stated above.

It is straightforward to combine the more sophisticated approaches with our model.

Now we describe our model.

The main contributions that this model brings in are (1) A differentiable decision tree, where we describe how to compute the expected probability distribution over predicted labels and use this to differentiate the expected loss of the tree (which can be used to optimize learned embeddings to minimize expected loss); (2) an explicit concept of class-derived data (supervised VAE) by modifying the prior p(z) to a mixture of Guassians; and (3) A combined VAE model using modifications (1) and (2) designed to learn a latent variable distribution suitable simultaneously for classifying and generating data (C+VAE).

Previous work has explored the potential of the VAE's encoder to learn the data manifold for nonlinear dimensionality reduction and semi-supervised classification ).

A difficulty arises from the VAE objective as the encoder learns to cluster the latent codes of high-probability data as close as possible to the mean of the prior distribution to minimize the K-L divergence term, KL(q(z | x) p(z)), with the typical choice of unit Gaussian for the prior.

This mixes data of various classes, making them difficult to separate for classification.

We address this issue by changing the prior from an isotropic, unit Gaussian of the standard VAE to a mixture of unit Gaussians.

Specifically, we modify the generation procedure to be class-focused, rather than assuming otherwise undifferentiated data.

To build the notion of class-distributed data into the VAE objective, we use a the prior to the following Gaussian mixture: DISPLAYFORM0 ( 1) where y is a class label, and µ y is the posterior mean of class y, and π is a probability vector which may be pre-computed with the assumption that the labels in the training dataset are iid.

As µ y is calculated empirically from the posterior, it can be initialized to small random values for all classes and updated regularly throughout training.

The objective for this VAE is DISPLAYFORM1 Since we train in a fully supervised model, each training instance is of the form (x, y).

Thus, when y is instantiated as the class variable, we get DISPLAYFORM2 In the remainder of this paper, we refer to a VAE trained with this objective function as the supervised VAE (SVAE).Our approach to utilizing a Gaussian mixture as a prior distribution is similar to that of BID2 .

A key difference between their work and ours is that our use of class labels enhances training, obviating the need to marginalize over all classes to compute the K-L divergence.

This helps avoid the over-regularization problem that they discuss in their paper, while achieving high sample quality in our generated images.

The decision tree is a simple, interpretable model used for non-parametric classification.

Typically, a tree is constructed by an algorithm like CART or C4.5 BID0 BID10 ) that recursively divides the dataset at each node of the tree, greedily minimizing the weighted Gini coefficient or entropy of two subsets by choosing a dividing line in one dimension.

Inference is performed by walking an input from the root to a leaf according to this series of inequalities, and then assigning a class probability vector according to the leaf.

Decision trees often classify well when using data with a few, richly descriptive features, and are very interpretable in their decision making processes.

We are interested in learning a deep network for non-linear dimensionality reduction to allow the decision tree to classify a low-dimensional embedding of data that is normally highdimensional and very structured.

Toward this end, we utilize a probabilistic generalization of decision trees, where each leaf returns a distribution over all classes.

I.e., if instance z lands in leaf of tree T , then T returns a distribution P T (y | ).Fix a decision tree T with leaves L that takes as input instance z and outputs a label y.

We observe that each leaf of T corresponds to a region in an axis-aligned rectilinear partitioning covering the data space whose bounds are defined by the inequalities encoded in the path from the root.

To compute the expected loss of T on instance z drawn according to probability distribution D(z), first consider one leaf , and let R ⊂ R d be the region of T 's input space that is covered by .

Then the probability that randomly drawn instance z falls into leaf is DISPLAYFORM0 Then the probability that randomly drawn instance z is predicted to be class y is DISPLAYFORM1 ] may be replaced with a product of the integrals in each dimension.

Further, each integral may be calculated as the difference of the cumulative distribution function at the upper (r + ,i ) and lower (r − ,i ) bounds of the partition in each dimension i.

The full inference may be re-written as DISPLAYFORM2 where CDF i is the cumulative distribution function of D i (z i ).Given a user-specified loss function loss T (z, y), our goal is to minimize the expected loss L T = E z∼D [loss T (z, y)].

In our work, we use as D the distribution q(z | x) = N (x; µ x , σ x I), where µ x and σ x are the outputs of the encoder on input x. To perform gradient-based optimization of L T , the gradient w.r.t.

each parameter is calculated as DISPLAYFORM3 where PDF i (r | µ x,i , σ x,i ) is the value of the Gaussian PDF of dimension i evaluated at r.

This allows optimization of the distribution parameters for maximum likelihood w.r.t.

an existing decision tree T .

Thus, an embedding of the data may be learned in an EM-style manner, alternately learning a tree on the embedding produced by the parameters of a deep encoder and optimizing the embedding parameters to better fit the class-based partitioning induced by the learned decision tree.

The supervised, Gaussian-mixture-based VAE and decision tree inference can be used with a VAE model to both classify and reconstruct data from the encoded parameters of its latent variable distribution.

Although an embedding could be learned by only optimizing the classification accuracy of the decision tree, the additional reconstruction objective ensures that the learned representation is non-arbitrary and contains more than just class information for downstream use.

Our new architecture C+VAE (Classifier+VAE) uses a deep encoder network to parameterize a Gaussian distribution, which is then used as the input for classifying with the DDT and to reconstruct the encoded data with a deep decoder network.

Generally, the combined modifications can also be applied to existing VAE architectures when label information is available.

The C+VAE training procedure begins by randomly initializing the encoder/decoder parameters and encoding the training data to initialize the decision tree and aggregate posterior class means.

Training then proceeds by running several epochs of gradient updates before re-training the decision tree and updating the aggregate posterior class means until the model converges.

The optimization function of our combined model consists of a linear combination of the objective of the supervised VAE and the expected error of the current decision tree T .

However, since the effect of the DDT gradient is to separate the class means and the supervised VAE K-L divergence term measures w.r.t.

these movable class-based means (rather than the distance from the mean of p(z)), the parameters learned by the encoder could diverge, driving the class means arbitrarily far from the origin.

Thus, an additional regularizing L 2 2 -loss is imposed on the encoded µ posterior value to keep the class means from "drifting" from the origin and encourage the model to learn common factors of variation between classes.

We observed experimentally that this additional drift loss term increases the training stability and classification performance of the model.

The modified VAE objective of the C+VAE to be minimized is DISPLAYFORM0

Our experiments are designed to empirically study the following claims:1.

The supervised VAE very effectively takes advantage of class labels to improve generative performance.2.

The C+VAE classifies competitively with other tree-based embedding methods while simultaneously maintaining a generative model competitive with the literature.3.

The differentiable decision tree is an interpretable classifier that, when used in C+VAE, can learn the semantics of the macro-features learned by the underlying VAE.The MNIST dataset was used for all experiments, as it is widely understood and commonly used for both classification and generation tasks.

We applied the C+VAE modifications to a standard VAE ) with two-layer MLPs of 500 hidden units as encoder and decoder models and a 50-dimensional latent variable z without importance sampling or an autoregressive prior.

The CART algorithm as made available in scikit-learn BID9 ) was used to train the decision tree.

This was regularized by annealing the maximum depth of the decision tree from 1 to 8 as training proceeded, incrementing every 15 epochs, and by setting the minimum proportion of samples in a leaf to be 2% of the training set.

Unless otherwise noted, we used γ = 1000 and λ = 0.1 in the objective function of the C+VAE (Equation FORMULA10 ), and n = 5 epochs of gradient steps between each update of both the decision tree and the aggregate posterior class means.

Adam BID4 ) was used for optimization and the data was not pre-processed or augmented.

TAB0 lists the classification performance of a number of tree-based and VAE-based models.

The M1:SVAE+CART model trains the supervised VAE to convergence, and then trains a standard decision tree with CART to classify its latent code in the style of M1 ).

The intent is to highlight the effect of training without the backpropagated classification loss from the DDT.

C+VAE sans reconstruction zeros the reconstruction loss term of the objective function to highlight the effect of training a model that only learns an embedding suitable for classification with the DDT.

The boundary tree (BT) with embedding is from BID16 , M1+M2 is from , and the Ladder Network is from BID11 .

We first evaluate the efficacy of leveraging label data in a supervised VAE in generation.

I.e., the effect on generation of making the prior distribution a Gaussian mixture and taking advantage of class label information.

This is equivalent to using C+VAE with γ = λ = 0 in Equation (4).

The flexibility of a Gaussian mixture and the fact that the data is clearly multi-modal both contribute to the SVAE log-likelihood of −102.77, which is better than the log-likelihood of −109.56 using our implementation of the VAE of , which uses an unmodified Gaussian prior.

We expect this difference to be the result of using a flexible prior that is more faithful to the true prior.

This flexibility is similar to that seen in techniques like normalizing flows (Rezende & Mohamed FORMULA4 ), but modifies the prior rather than the posterior and uses the additional information provided by label information, rather than adding additional computation.

We next empirically evaluate C+VAE for both classification performance and generative ability.

As a baseline, we first examine how well a standard (non-differentiable) decision tree from CART can classify when the data is encoded by a supervised VAE (but without any error feedback: γ = λ = 0).

This is similar to M1 from with a different VAE.

In TAB0 , row M1:SVAE+CART shows that without the error feedback from the tree, it is unlikely that the embedding will be useful in classification by a decision tree.

This motivates our use of the DDT.To test the benefit of reconstruction in learning an embedding that can be classified well, we ran a test in which we switched off the reconstruction error feedback in learning.

I.e., we removed the first term of Equation (3).

In TAB0 , row C+VAE sans reconstruction shows a significant improvement in classification error over M1:SVAE+CART, but still quite high.

Row C+VAE in TAB0 shows our combined method's performance with γ = 1000 and λ = 0.1.

We see a large improvement in classification error over C+VAE sans reconstruction, demonstrating the importance of both types of feedback in training.

While C+VAE's classification performance is worse than results from the literature, it's still competitive, despite simultaneously optimizing both classification and log-likelihood.

Also, we note that C+VAE's log-likelihood of −110.12 is comparable to the −109.56 from our implementation of the VAE of , which uses the same encoder-decoder pair as C+VAE.

A more powerful encoder or the use of more recent techniques (e.g., normalizing flows, importance weighting, etc.) could conceivably improve both error and log-likelihood even further.

Figure 7 in the appendix presents sample MNIST digits generated by C+VAE.

Each set of digits is generated from one of the empirical aggregated class means.

The final decision tree learned by the C+VAE is shown in FIG0 (a landscape version of the same tree is in Figure 8 in the appendix).

This tree performs feature selection over the 50 available latent dimensions, using only 8 to classify with 98.02% accuracy with one dimension (21) used to split twice.

We were able to leverage the simplicity of the decision tree to assign meaning to the latent dimensions used by the tree to classify inputs.

Each node divides inputs according to a threshold value in a single dimension, which corresponds to detecting the most salient macro-feature that distinguishes the divided subsets.

Dimension 21 of the latent code is the macro-feature used by the decision tree to discriminate between digits '6' and '0', as well as '4' and '7'.

FIG1 visualizes the macro-feature corresponding to dimension 21.

Specifically, the top image (starting with '6') was generated by fixing the other 49 dimensions to be the values of µ 6 and varying the value of dimension 21 in even steps from −1 to 2.

The bottom image was generated the same way, with the other 49 dimensions initialized to values from µ 4 .

In both image sequences, we see that a high value of dimension 21 emphasizes the macro-feature of a flat top bar of a digit whereas a low value removes it.

To illustrate the effect of varying dimension 21 independently of a specific class mean, we generate FIG2 by an identical process, but fix the other 49 dimensions to be the mean of all 10 class means.

The center image is the mean of all 50 dimensions, and is provided for contrast.

The left and right images show the effect of a low or high value of dimension 21 on this 'average digit'.

The clearest effect of this variation is that the flat top macro-feature is present when this value is high, and absent when it is low, just as in FIG1 .We generate Figures 4, 5, and 6 by the same process, varying dimensions 10, 26, and 45, respectively.

The decision tree uses dimension 0 to separate digits '3', '5', and '8' from digits '2', '4', and '7'.

FIG3 shows that a low value of dimension 0 correlates strongly with rounded digits, while a high value creates an emphasized right side and a more angular appearance.

FIG4 shows the effect of varying dimension 26, used by the tree to separate '2' from '4' and '7'.

The most notable impact of a low dimension 26 is the exaggerated lower-left corner, which is absent when that dimension is high.

FIG5 shows the effect of varying dimension 45.

This is the first dimension used by the decision tree, to split '1' from the other nine classes.

A clear vertical line near the center of the digit is emphasized by low values of dimension 1.

Latent codes with high values of dimension 1, a feature common to the other nine classes, lack that central vertical line.

Differentiable Trees Previous work that uses deep networks for representation learning with decision trees is the Deep Neural Decision Forests of BID7 , which stochastically make routing decisions through a decision tree according to the outputs of a deep convolutional network.

The setup achieved good performance at its classification task, but it is not clear how to interpret the proposed classification process, especially when more than one tree is combined into a forest classifier.

As a method of making the decision tree differentiable, our proposed inference method of integrating a probability distribution over the decision regions of the tree is a novel approach.

Another tree-based method uses differentiable boundary trees to learn an embedding suitable for k-nearest neighbor classification BID16 ).

The learned representation allows a small, interpretable boundary tree to classify effectively, similar to our technique.

The classification accuracy of the technique marginally outperforms our combined model, but the C+VAE also acts as a generative model and does not suffer from the significant complexity of having to use dynamically constructed computation graphs.

VAE learning Other work in classifying the latent codes produced by a VAE includes , whose M1 semi-supervised model learns to classify from the latent embedding similarly to our combined classifier.

However, M1 trains the discriminator separately from the VAE and lacks interpretability as the class separation is performed solely by a black-box discriminator.

The M2 model is similar to the supervised VAE, but doesn't change the VAE prior.

BID2 present a Gaussian Mixture Variational Autoencoder to learn a classfocused latent representation.

Our work assumes a supervised, rather than the GMVAE's unsupervised environment.

This allows the classifying modification to the VAE framework to remain simpler and more interpretable, as well as more tractable optimization.

Future work includes applying our approach to other data sets such as CIFAR-10, and using more powerful encoders and decoders to see how performance is affected.

We will also look into extending our approach to handle unlabeled data in applications such as semi-supervised learning and clustering.

A GENERATED MNIST IMAGES Figure 7 : Sample MNIST digits generated by C+VAE.

Each set of digits is generated from one class prior, but the mixture may be sampled from with one extra step.

Figure 8: Landscape version of decision tree learned by C+VAE.

@highlight

We combine differentiable decision trees with supervised variational autoencoders to enhance interpretability of classification. 

@highlight

This paper proposes a hybrid model of a variational autoencoder composed with a differentiable decision tree, and an accompanying training scheme, with experiments demonstrating tree classification performance, neg. log likelihood performance, and latent space interpretability.

@highlight

The paper tries to build an interpretable and accurate classifier via stacking a supervised VAE and a differentiable decision tree