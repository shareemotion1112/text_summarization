This work investigates unsupervised learning of representations by maximizing mutual information between an input and the output of a deep neural network encoder.

Importantly, we show that structure matters: incorporating knowledge about locality in the input into the objective can significantly improve a representation's suitability for downstream tasks.

We further control characteristics of the representation by matching to a prior distribution adversarially.

Our method, which we call Deep InfoMax (DIM), outperforms a number of popular unsupervised learning methods and compares favorably with fully-supervised learning on several classification tasks in with some standard architectures.

DIM opens new avenues for unsupervised learning of representations and is an important step towards flexible formulations of representation learning objectives for specific end-goals.

One core objective of deep learning is to discover useful representations, and the simple idea explored here is to train a representation-learning function, i.e. an encoder, to maximize the mutual information (MI) between its inputs and outputs.

MI is notoriously difficult to compute, particularly in continuous and high-dimensional settings.

Fortunately, recent advances enable effective computation of MI between high dimensional input/output pairs of deep neural networks (Belghazi et al., 2018) .

We leverage MI estimation for representation learning and show that, depending on the downstream task, maximizing MI between the complete input and the encoder output (i.e., global MI) is often insufficient for learning useful representations.

Rather, structure matters: maximizing the average MI between the representation and local regions of the input (e.g. patches rather than the complete image) can greatly improve the representation's quality for, e.g., classification tasks, while global MI plays a stronger role in the ability to reconstruct the full input given the representation.

Usefulness of a representation is not just a matter of information content: representational characteristics like independence also play an important role (Gretton et al., 2012; Hyvärinen & Oja, 2000; Hinton, 2002; Schmidhuber, 1992; Bengio et al., 2013; Thomas et al., 2017) .

We combine MI maximization with prior matching in a manner similar to adversarial autoencoders (AAE, Makhzani et al., 2015) to constrain representations according to desired statistical properties.

This approach is closely related to the infomax optimization principle (Linsker, 1988; Bell & Sejnowski, 1995) , so we call our method Deep InfoMax (DIM).

Our main contributions are the following:• We formalize Deep InfoMax (DIM), which simultaneously estimates and maximizes the mutual information between input data and learned high-level representations.• Our mutual information maximization procedure can prioritize global or local information, which we show can be used to tune the suitability of learned representations for classification or reconstruction-style tasks.• We use adversarial learning (à la Makhzani et al., 2015) to constrain the representation to have desired statistical characteristics specific to a prior.• We introduce two new measures of representation quality, one based on Mutual Information Neural Estimation (MINE, Belghazi et al., 2018 ) and a neural dependency measure (NDM) based on the work by Brakel & Bengio (2017) , and we use these to bolster our comparison of DIM to different unsupervised methods.

There are many popular methods for learning representations.

Classic methods, such as independent component analysis (ICA, Bell & Sejnowski, 1995) and self-organizing maps (Kohonen, 1998) , generally lack the representational capacity of deep neural networks.

More recent approaches include deep volume-preserving maps (Dinh et al., 2014; , deep clustering BID8 Chang et al., 2017) , noise as targets (NAT, Bojanowski & Joulin, 2017) , and self-supervised or co-learning (Doersch & Zisserman, 2017; Dosovitskiy et al., 2016; Sajjadi et al., 2016) .Generative models are also commonly used for building representations BID5 Kingma et al., 2014; Salimans et al., 2016; Rezende et al., 2016; Donahue et al., 2016) , and mutual information (MI) plays an important role in the quality of the representations they learn.

In generative models that rely on reconstruction (e.g., denoising, variational, and adversarial autoencoders, Vincent et al., 2008; Rifai et al., 2012; Kingma & Welling, 2013; Makhzani et al., 2015) , the reconstruction error can be related to the MI as follows:I e (X, Y ) = H e (X) − H e (X|Y ) ≥ H e (X) − R e,d (X|Y ),where X and Y denote the input and output of an encoder which is applied to inputs sampled from some source distribution.

R e,d (X|Y ) denotes the expected reconstruction error of X given the codes Y .

H e (X) and H e (X|Y ) denote the marginal and conditional entropy of X in the distribution formed by applying the encoder to inputs sampled from the source distribution.

Thus, in typical settings, models with reconstruction-type objectives provide some guarantees on the amount of information encoded in their intermediate representations.

Similar guarantees exist for bi-directional adversarial models (Dumoulin et al., 2016; Donahue et al., 2016) , which adversarially train an encoder / decoder to match their respective joint distributions or to minimize the reconstruction error (Chen et al., 2016) .Mutual-information estimation Methods based on mutual information have a long history in unsupervised feature learning.

The infomax principle (Linsker, 1988; Bell & Sejnowski, 1995) , as prescribed for neural networks, advocates maximizing MI between the input and output.

This is the basis of numerous ICA algorithms, which can be nonlinear (Hyvärinen & Pajunen, 1999; BID1 but are often hard to adapt for use with deep networks.

Mutual Information Neural Estimation (MINE, Belghazi et al., 2018) learns an estimate of the MI of continuous variables, is strongly consistent, and can be used to learn better implicit bi-directional generative models.

Deep InfoMax (DIM) follows MINE in this regard, though we find that the generator is unnecessary.

We also find it unnecessary to use the exact KL-based formulation of MI.

For example, a simple alternative based on the Jensen-Shannon divergence (JSD) is more stable and provides better results.

We will show that DIM can work with various MI estimators.

Most significantly, DIM can leverage local structure in the input to improve the suitability of representations for classification.

Leveraging known structure in the input when designing objectives based on MI maximization is nothing new BID3 BID4 BID7 , and some very recent works also follow this intuition.

It has been shown in the case of discrete MI that data augmentations and other transformations can be used to avoid degenerate solutions (Hu et al., 2017) .

Unsupervised clustering and segmentation is attainable by maximizing the MI between images associated by transforms or spatial proximity (Ji et al., 2018) .

Our work investigates the suitability of representations learned across two different MI objectives that focus on local or global structure, a flexibility we believe is necessary for training representations intended for different applications.

Proposed independently of DIM, Contrastive Predictive Coding (CPC, Oord et al., 2018 ) is a MIbased approach that, like DIM, maximizes MI between global and local representation pairs.

CPC shares some motivations and computations with DIM, but there are important ways in which CPC and DIM differ.

CPC processes local features sequentially to build partial "summary features", which are used to make predictions about specific local features in the "future" of each summary feature.

This equates to ordered autoregression over the local features, and requires training separate estimators for each temporal offset at which one would like to predict the future.

In contrast, the basic version of DIM uses a single summary feature that is a function of all local features, and this "global" feature predicts all local features simultaneously in a single step using a single estimator.

Note that, when using occlusions during training (see Section 4.3 for details), DIM performs both "self" predictions and orderless autoregression.

Figure 1 : The base encoder model in the context of image data.

An image (in this case) is encoded using a convnet until reaching a feature map of M × M feature vectors corresponding to M × M input patches.

These vectors are summarized into a single feature vector, Y .

Our goal is to train this network such that useful information about the input is easily extracted from the high-level features.

Figure 2: Deep InfoMax (DIM) with a global MI(X; Y ) objective.

Here, we pass both the high-level feature vector, Y , and the lower-level M ×M feature map (see Figure 1 ) through a discriminator to get the score.

Fake samples are drawn by combining the same feature vector with a M × M feature map from another image.

Here we outline the general setting of training an encoder to maximize mutual information between its input and output.

Let X and Y be the domain and range of a continuous and (almost everywhere) differentiable parametric function, E ψ : X → Y with parameters ψ (e.g., a neural network).

These parameters define a family of encoders, E Φ = {E ψ } ψ∈Ψ over Ψ. Assume that we are given a set of training examples on an input space, X : DISPLAYFORM0 , with empirical probability distribution P. We define U ψ,P to be the marginal distribution induced by pushing samples from P through E ψ .

I.e., U ψ,P is the distribution over encodings y ∈ Y produced by sampling observations x ∼ X and then sampling y ∼ E ψ (x).An example encoder for image data is given in Figure 1 , which will be used in the following sections, but this approach can easily be adapted for temporal data.

Similar to the infomax optimization principle (Linsker, 1988) , we assert our encoder should be trained according to the following criteria:• Mutual information maximization: Find the set of parameters, ψ, such that the mutual information, I(X; E ψ (X)), is maximized.

Depending on the end-goal, this maximization can be done over the complete input, X, or some structured or "local" subset.• Statistical constraints: Depending on the end-goal for the representation, the marginal U ψ,P should match a prior distribution, V. Roughly speaking, this can be used to encourage the output of the encoder to have desired characteristics (e.g., independence).The formulation of these two objectives covered below we call Deep InfoMax (DIM).

Our basic mutual information maximization framework is presented in Figure 2 .

The approach follows Mutual Information Neural Estimation (MINE, Belghazi et al., 2018) , which estimates mutual information by training a classifier to distinguish between samples coming from the joint, J, and the First we encode the image to a feature map that reflects some structural aspect of the data, e.g. spatial locality, and we further summarize this feature map into a global feature vector (see Figure 1) .

We then concatenate this feature vector with the lower-level feature map at every location.

A score is produced for each local-global pair through an additional function (see the Appendix A.2 for details).product of marginals, M, of random variables X and Y .

MINE uses a lower-bound to the MI based on the Donsker-Varadhan representation (DV, Donsker & Varadhan, 1983) of the KL-divergence, DISPLAYFORM0 where T ω : X × Y → R is a discriminator function modeled by a neural network with parameters ω.

At a high level, we optimize E ψ by simultaneously estimating and maximizing I(X, E ψ (X)), DISPLAYFORM1 where the subscript G denotes "global" for reasons that will be clear later.

However, there are some important differences that distinguish our approach from MINE.

First, because the encoder and mutual information estimator are optimizing the same objective and require similar computations, we share layers between these functions, so that DISPLAYFORM2 where g is a function that combines the encoder output with the lower layer.

Second, as we are primarily interested in maximizing MI, and not concerned with its precise value, we can rely on non-KL divergences which may offer favourable trade-offs.

For example, one could define a Jensen-Shannon MI estimator (following the formulation of Nowozin et al., 2016) , DISPLAYFORM3 where x is an input sample, x is an input sampled fromP = P, and sp(z) = log(1+e z ) is the softplus function.

A similar estimator appeared in Brakel & Bengio (2017) in the context of minimizing the total correlation, and it amounts to the familiar binary cross-entropy.

This is well-understood in terms of neural network optimization and we find works better in practice (e.g., is more stable) than the DV-based objective (e.g., see App.

A.3) .

Intuitively, the Jensen-Shannon-based estimator should behave similarly to the DV-based estimator in Eq. 2, since both act like classifiers whose objectives maximize the expected log-ratio of the joint over the product of marginals.

We show in App.

A.1 the relationship between the JSD estimator and the formal definition of mutual information.

Noise-Contrastive Estimation (NCE, Gutmann & Hyvärinen, 2010; 2012) was first used as a bound on MI in Oord et al. (and called "infoNCE", 2018) , and this loss can also be used with DIM by maximizing: DISPLAYFORM4 For DIM, a key difference between the DV, JSD, and infoNCE formulations is whether an expectation over P/P appears inside or outside of a log.

In fact, the JSD-based objective mirrors the original NCE formulation in Gutmann & Hyvärinen (2010) , which phrased unnormalized density estimation as binary classification between the data distribution and a noise distribution.

DIM sets the noise distribution to the product of marginals over X/Y , and the data distribution to the true joint.

The infoNCE formulation in Eq. 5 follows a softmax-based version of NCE (Jozefowicz et al., 2016) , similar to ones used in the language modeling community (Mnih & Kavukcuoglu, 2013; Mikolov et al., 2013) , and which has strong connections to the binary cross-entropy in the context of noise-contrastive learning (Ma & Collins, 2018) .

In practice, implementations of these estimators appear quite similar and can reuse most of the same code.

We investigate JSD and infoNCE in our experiments, and find that using infoNCE often outperforms JSD on downstream tasks, though this effect diminishes with more challenging data.

However, as we show in the App. (A.3), infoNCE and DV require a large number of negative samples (samples fromP) to be competitive.

We generate negative samples using all combinations of global and local features at all locations of the relevant feature map, across all images in a batch.

For a batch of size B, that gives O(B × M 2 ) negative samples per positive example, which quickly becomes cumbersome with increasing batch size.

We found that DIM with the JSD loss is insensitive to the number of negative samples, and in fact outperforms infoNCE as the number of negative samples becomes smaller.

The objective in Eq. 3 can be used to maximize MI between input and output, but ultimately this may be undesirable depending on the task.

For example, trivial pixel-level noise is useless for image classification, so a representation may not benefit from encoding this information (e.g., in zero-shot learning, transfer learning, etc.).

In order to obtain a representation more suitable for classification, we can instead maximize the average MI between the high-level representation and local patches of the image.

Because the same representation is encouraged to have high MI with all the patches, this favours encoding aspects of the data that are shared across patches.

Suppose the feature vector is of limited capacity (number of units and range) and assume the encoder does not support infinite output configurations.

For maximizing the MI between the whole input and the representation, the encoder can pick and choose what type of information in the input is passed through the encoder, such as noise specific to local patches or pixels.

However, if the encoder passes information specific to only some parts of the input, this does not increase the MI with any of the other patches that do not contain said noise.

This encourages the encoder to prefer information that is shared across the input, and this hypothesis is supported in our experiments below.

Our local DIM framework is presented in FIG0 .

First we encode the input to a feature map, DISPLAYFORM0 that reflects useful structure in the data (e.g., spatial locality), indexed in this case by i. Next, we summarize this local feature map into a global feature, E ψ (x) = f ψ • C ψ (x).

We then define our MI estimator on global/local pairs, maximizing the average estimated MI: DISPLAYFORM1 We found success optimizing this "local" objective with multiple easy-to-implement architectures, and further implementation details are provided in the App. (A.2).

Absolute magnitude of information is only one desirable property of a representation; depending on the application, good representations can be compact (Gretton et al., 2012) , independent (Hyvärinen & Oja, 2000; Hinton, 2002; Dinh et al., 2014; Brakel & Bengio, 2017) , disentangled (Schmidhuber, 1992; Rifai et al., 2012; Bengio et al., 2013; Chen et al., 2018; Gonzalez-Garcia et al., 2018) , or independently controllable (Thomas et al., 2017) .

DIM imposes statistical constraints onto learned representations by implicitly training the encoder so that the push-forward distribution, U ψ,P , matches a prior, V. This is done (see Figure 7 in the App.

A.2) by training a discriminator, D φ : Y → R, to estimate the divergence, D(V||U ψ,P ), then training the encoder to minimize this estimate: DISPLAYFORM0 This approach is similar to what is done in adversarial autoencoders (AAE, Makhzani et al., 2015) , but without a generator.

It is also similar to noise as targets (Bojanowski & Joulin, 2017) , but trains the encoder to match the noise implicitly rather than using a priori noise samples as targets.

All three objectives -global and local MI maximization and prior matching -can be used together, and doing so we arrive at our complete objective for Deep InfoMax (DIM): DISPLAYFORM1 where ω 1 and ω 2 are the discriminator parameters for the global and local objectives, respectively, and α, β, and γ are hyperparameters.

We will show below that choices in these hyperparameters affect the learned representations in meaningful ways.

As an interesting aside, we also show in the App. (A.8) that this prior matching can be used alone to train a generator of image data.

We test Deep InfoMax (DIM) on four imaging datasets to evaluate its representational properties: , 2018) .

Note that we take CPC to mean ordered autoregression using summary features to predict "future" local features, independent of the constrastive loss used to evaluate the predictions (JSD, infoNCE, or DV) .

See the App. (A.2) for details of the neural net architectures used in the experiments.

DISPLAYFORM0

Evaluation of representations is case-driven and relies on various proxies.

Linear separability is commonly used as a proxy for disentanglement and mutual information (MI) between representations and class labels.

Unfortunately, this will not show whether the representation has high MI with the class labels when the representation is not disentangled.

Other works (Bojanowski & Joulin, 2017) have looked at transfer learning classification tasks by freezing the weights of the encoder and training a small fully-connected neural network classifier using the representation as input.

Others still have more directly measured the MI between the labels and the representation (Rifai et al., 2012; Chen et al., 2018) , which can also reveal the representation's degree of entanglement.

Class labels have limited use in evaluating representations, as we are often interested in information encoded in the representation that is unknown to us.

However, we can use mutual information neural estimation (MINE, Belghazi et al., 2018) to more directly measure the MI between the input and output of the encoder.

Next, we can directly measure the independence of the representation using a discriminator.

Given a batch of representations, we generate a factor-wise independent distribution with the same per-factor marginals by randomly shuffling each factor along the batch dimension.

A similar trick has been used for learning maximally independent representations for sequential data (Brakel & Bengio, 2017) .

We can train a discriminator to estimate the KL-divergence between the original representations (joint distribution of the factors) and the shuffled representations (product of the marginals, see Figure 12 ).

The higher the KL divergence, the more dependent the factors.

We call this evaluation method Neural Dependency Measure (NDM) and show that it is sensible and empirically consistent in the App.

(A.6).To summarize, we use the following metrics for evaluating representations.

For each of these, the encoder is held fixed unless noted otherwise:• Linear classification using a support vector machine (SVM).

This is simultaneously a proxy for MI of the representation with linear separability.• Non-linear classification using a single hidden layer neural network (200 units) with dropout.

This is a proxy on MI of the representation with the labels separate from linear separability as measured with the SVM above.• Semi-supervised learning (STL-10 here), that is, fine-tuning the complete encoder by adding a small neural network on top of the last convolutional layer (matching architectures with a standard fully-supervised classifier).• MS-SSIM BID6 , using a decoder trained on the L 2 reconstruction loss.

This is a proxy for the total MI between the input and the representation and can indicate the amount of encoded pixel-level information.• Mutual information neural estimate (MINE), I ρ (X, E ψ (x)), between the input, X, and the output representation, E ψ (x), by training a discriminator with parameters ρ to maximize the DV estimator of the KL-divergence.• Neural dependency measure (NDM) using a second discriminator that measures the KL between E ψ (x) and a batch-wise shuffled version of E ψ (x).For the neural network classification evaluation above, we performed experiments on all datasets except CelebA, while for other measures we only looked at CIFAR10.

For all classification tasks, we built separate classifiers on the high-level vector representation (Y ), the output of the previous fully-connected layer (fc) and the last convolutional layer (conv).

Model selection for the classifiers was done by averaging the last 100 epochs of optimization, and the dropout rate and decaying learning rate schedule was set uniformly to alleviate over-fitting on the test set across all models.

In the following experiments, DIM(G) refers to DIM with a global-only objective (α = 1, β = 0, γ = 1) and DIM(L) refers to DIM with a local-only objective (α = 0, β = 1, γ = 0.1), the latter chosen from the results of an ablation study presented in the App. (A.5).

For the prior, we chose a compact uniform distribution on [0, 1] 64 , which worked better in practice than other priors, such as Gaussian, unit ball, or unit sphere.

TAB0 , and 3.

In general, DIM with the local objective, DIM(L), outperformed all models presented here by a significant margin on all datasets, regardless of which layer the representation was drawn from, with exception to CPC.

For the specific settings presented (architectures, no data augmentation for datasets except for STL-10), DIM(L) performs as well as or outperforms a fully-supervised classifier without fine-tuning, which indicates that the representations are nearly as good as or better than the raw pixels given the model constraints in this setting.

Note, however, that a fully supervised classifier can perform much better on all of these benchmarks, especially when specialized architectures and carefully-chosen data augmentations are used.

Competitive or better results on CIFAR10 also exist (albeit in different settings, e.g., Coates et al., 2011; Dosovitskiy et al., 2016) , but to our knowledge our STL-10 results are state-of-the-art for unsupervised learning.

The results in this setting support the hypothesis that our local DIM objective is suitable for extracting class information.

Our results show that infoNCE tends to perform best, but differences between infoNCE and JSD diminish with larger datasets.

DV can compete with JSD with smaller datasets, but DV performs much worse with larger datasets.

For CPC, we were only able to achieve marginally better performance than BiGAN with the settings above.

However, when we adopted the strided crop architecture found in Oord et al. (2018) , both CPC and DIM performance improved considerably.

We chose a crop size of 25% of the image size in width and depth with a stride of 12.5% the image size (e.g., 8 × 8 crops with 4 × 4 strides for CIFAR10, 16 × 16 crops with 8 × 8 strides for STL-10), so that there were a total of 7 × 7 local features.

For both DIM(L) and CPC, we used infoNCE as well as the same "encode-and-dot-product" architecture (tantamount to a deep bilinear model), rather than the shallow bilinear model used in Oord et al. (2018) .

For CPC, we used a total of 3 such networks, where each network for CPC is used for a separate prediction task of local feature maps in the next 3 rows of a summary predictor feature within each column.

2 For simplicity, we omitted the prior term, β, from DIM.

Without data augmentation on CIFAR10, CPC performs worse than DIM(L) with a ResNet-50 (He et al., 2016) type architecture.

For experiments we ran on STL-10 with data augmentation (using the same encoder architecture as TAB1 ), CPC and DIM were competitive, with CPC performing slightly better.

CPC makes predictions based on multiple summary features, each of which contains different amounts of information about the full input.

We can add similar behavior to DIM by computing less global features which condition on 3 × 3 blocks of local features sampled at random from the full 7 × 7 sets of local features.

We then maximize mutual information between these less global features and the full sets of local features.

We share a single MI estimator across all possible 3 × 3 blocks of local features when using this version of DIM.

This represents a particular instance of the occlusion technique described in Section 4.3.

The resulting model gave a significant performance boost to (CPC, Oord et al., 2018) .

These experiments used a strided-crop architecture similar to the one used in Oord et al. (2018) .

For CIFAR10 we used a ResNet-50 encoder, and for STL-10 we used the same architecture as for TAB1 .

We also tested a version of DIM that computes the global representation from a 3x3 block of local features randomly selected from the full 7x7 set of local features.

This is a particular instance of the occlusions described in Section 4.3.

DIM(L) is competitive with CPC in these settings.

DIM for STL-10.

Surprisingly, this same architecture performed worse than using the fully global representation with CIFAR10.

Overall DIM only slightly outperforms CPC in this setting, which suggests that the strictly ordered autoregression of CPC may be unnecessary for some tasks.

TAB4 on linear separability, reconstruction (MS-SSIM), mutual information, and dependence (NDM) with the CIFAR10 dataset.

We did not compare to CPC due to the divergence of architectures.

For linear classifier results (SVC), we trained five support vector machines with a simple hinge loss for each model, averaging the test accuracy.

For MINE, we used a decaying learning rate schedule, which helped reduce variance in estimates and provided faster convergence.

MS-SSIM correlated well with the MI estimate provided by MINE, indicating that these models encoded pixel-wise information well.

Overall, all models showed much lower dependence than BiGAN, indicating the marginal of the encoder output is not matching to the generator's spherical Gaussian input prior, though the mixed local/global version of DIM is close.

For MI, reconstructionbased models like VAE and AAE have high scores, and we found that combining local and global DIM objectives had very high scores (α = 0.5, β = 0.1 is presented here as DIM(L+G)).

For more in-depth analyses, please see the ablation studies and the nearest-neighbor analysis in the App. (A.4, A.5).

Maximizing MI between global and local features is not the only way to leverage image structure.

We consider augmenting DIM by adding input occlusion when computing global features and by adding auxiliary tasks which maximize MI between local features and absolute or relative spatial coordinates given a global feature.

These additions improve classification results (see TAB5 ).For occlusion, we randomly occlude part of the input when computing the global features, but compute local features using the full input.

Maximizing MI between occluded global features and unoccluded local features aggressively encourages the global features to encode information which is shared across the entire image.

For coordinate prediction, we maximize the model's ability to predict the coordinates (i, j) of a local feature c (i,j) = C (i,j) ψ (x) after computing the global features DISPLAYFORM0 To accomplish this, we maximize E[log p θ ((i, j)|y, c (i,j) )] (i.e., minimize the crossentropy).

We can extend the task to maximize conditional MI given global features y between pairs of local features (c (i,j) , c (i ,j ) ) and their relative coordinates (i − i , j − j ).

This objective can be written as DISPLAYFORM1 We use both these objectives in our results.

Additional implementation details can be found in the App. (A.7).

Roughly speaking, our input occlusions and coordinate prediction tasks can be interpreted as generalizations of inpainting (Pathak et al., 2016) and context prediction (Doersch et al., 2015) tasks which have previously been proposed for self-supervised feature learning.

Augmenting DIM with these tasks helps move our method further towards learning representations which encode images (or other types of inputs) not just in terms of compressing their low-level (e.g. pixel) content, but in terms of distributions over relations among higher-level features extracted from their lower-level content.

In this work, we introduced Deep InfoMax (DIM), a new method for learning unsupervised representations by maximizing mutual information, allowing for representations that contain locally-consistent information across structural "locations" (e.g., patches in an image).

This provides a straightforward and flexible way to learn representations that perform well on a variety of tasks.

We believe that this is an important direction in learning higher-level representations.

Here we show the relationship between the Jensen-Shannon divergence (JSD) between the joint and the product of marginals and the pointwise mutual information (PMI).

Let p(x) and p(y) be two marginal densities, and define p(y|x) and p(x, y) = p(y|x)p(x) as the conditional and joint distribution, respectively.

Construct a probability mixture density, m(x, y) = 1 2 (p(x)p(y) + p(x, y)).

It follows that m(x) = p(x), m(y) = p(y), and m(y|x) = 1 2 (p(y) + p(y|x)).

Note that: DISPLAYFORM0 Discarding some constants: DISPLAYFORM1 The quantity inside the expectation of Eqn.

10 is a concave, monotonically increasing function of the ratio p(y|x) p(y) , which is exactly e PMI(x,y) .

Note this relationship does not hold for the JSD of arbitrary distributions, as the the joint and product of marginals are intimately coupled.

We can verify our theoretical observation by plotting the JSD and KL divergences between the joint and the product of marginals, the latter of which is the formal definition of mutual information (MI).

As computing the continuous MI is difficult, we assume a discrete input with uniform probability, p(x) (e.g., these could be one-hot variables indicating one of N i.i.d.

random samples), and a randomly initialized N × M joint distribution, p(x, y), such that M j=1 p(x i , y j ) = 1 ∀i.

For this joint distribution, we sample from a uniform distribution, then apply dropout to encourage sparsity to simulate the situation when there is no bijective function between x and y, then apply a softmax.

As the distributions are discrete, we can compute the KL and JSD between p(x, y) and p(x)p(y).We ran these experiments with matched input / output dimensions of 8, 16, 32, 64, and 128, randomly drawing 1000 joint distributions, and computed the KL and JSD divergences directly.

Our results ( Figure A.1) indicate that the KL (traditional definition of mutual information) and the JSD have an approximately monotonic relationship.

Overall, the distributions with the highest mutual information also have the highest JSD.

Here we provide architectural details for our experiments.

Example code for running Deep Infomax (DIM) can be found at https://github.com/rdevon/DIM.Encoder We used an encoder similar to a deep convolutional GAN (DCGAN, Radford et al., 2015) discriminator for CIFAR10 and CIFAR100, and for all other datasets we used an Alexnet (Krizhevsky et al., 2012) architecture similar to that found in Donahue et al. (2016) .

ReLU activations and batch norm (Ioffe & Szegedy, 2015) were used on every hidden layer.

For the DCGAN architecture, a single hidden layer with 1024 units was used after the final convolutional layer, and for the Alexnet architecture it was two hidden layers with 4096.

For all experiments, the output of all encoders was a 64 dimensional vector.

with discrete inputs and a given randomized and sparse joint distribution, p(x, y).

8 × 8 indicates a square joint distribution with 8 rows and 8 columns.

Our experiments indicate a strong monotonic relationship between M I(x; y) and JSD(p(x, y)||p(x)p(y)) Overall, the distributions with the highest M I(x; y) have the highest JSD(p(x, y)||p(x)p(y)).Mutual information discriminators For the global mutual information objective, we first encode the input into a feature map, C ψ (x), which in this case is the output of the last convolutional layer.

We then encode this representation further using linear layers as detailed above to get E ψ (x).

C ψ (x) is then flattened, then concatenated with E ψ (x).

We then pass this to a fully-connected network with two 512-unit hidden layers (see TAB7 ).

We tested two different architectures for the local objective.

The first ( Figure 5 ) concatenated the global feature vector with the feature map at every location, i.e., {[C DISPLAYFORM0 .

A 1 × 1 convolutional discriminator is then used to score the (feature map, feature vector) pair, DISPLAYFORM1 Fake samples are generated by combining global feature vectors with local feature maps coming from different images, x : DISPLAYFORM2 This architecture is featured in the results of TAB4 , as well as the ablation and nearest-neighbor studies below.

We used a 1 × 1 convnet with two 512-unit hidden layers as discriminator (Table 7) .

Table 7 : Local DIM concat-and-convolve network architecture DISPLAYFORM3 The other architecture we tested ( Figure 6 ) is based on non-linearly embedding the global and local features in a (much) higher-dimensional space, and then computing pair-wise scores using dot products between their high-dimensional embeddings.

This enables efficient evaluation of a large number of pair-wise scores, thus allowing us to use large numbers of positive/negative samples.

Given a sufficiently high-dimensional embedding space, this approach can represent (almost) arbitrary classes of pair-wise functions that are non-linear in the original, lower-dimensional features.

For more information, refer to Reproducing Kernel Hilbert Spaces.

We pass the global feature through a Figure 5 : Concat-and-convolve architecture.

The global feature vector is concatenated with the lower-level feature map at every location.

A 1 × 1 convolutional discriminator is then used to score the "real" feature map / feature vector pair, while the "fake" pair is produced by pairing the feature vector with a feature map from another image.

Figure 6 : Encode-and-dot-product architecture.

The global feature vector is encoded using a fully-connected network, and the lower-level feature map is encoded using 1x1 convolutions, but with the same number of output features.

We then take the dotproduct between the feature at each location of the feature map encoding and the encoded global vector for scores.

Figure 7: Matching the output of the encoder to a prior.

"Real" samples are drawn from a prior while "fake" samples from the encoder output are sent to a discriminator.

The discriminator is trained to distinguish between (classify) these sets of samples.

The encoder is trained to "fool" the discriminator.fully connected neural network to get the encoded global feature, S ω (E ψ (x)).

In our experiments, we used a single hidden layer network with a linear shortcut (See TAB8 ).

We embed each local feature in the local feature map C ψ (x) using an architecture which matches the one for global feature embedding.

We apply it via 1 × 1 convolutions.

Details are in TAB9 .

Finally, the outputs of these two networks are combined by matrix multiplication, summing over the feature dimension (2048 in the example above).

As this is computed over a batch, this allows us to efficiently compute both positive and negative examples simultaneously.

This architecture is featured in our main classification results in TAB0 , and 5.For the local objective, the feature map, C ψ (x), can be taken from any level of the encoder, E ψ .

For the global objective, this is the last convolutional layer, and this objective was insensitive to which layer we used.

For the local objectives, we found that using the next-to-last layer worked best for CIFAR10 and CIFAR100, while for the other larger datasets it was the previous layer.

This sensitivity is likely due to the relative size of the of the receptive fields, and further analysis is necessary to better understand this effect.

Note that all feature maps used for DIM included the final batch normalization and ReLU activation.

Figure 7 shows a high-level overview of the prior matching architecture.

The discriminator used to match the prior in DIM was a fully-connected network with two hidden layers of 1000 and 200 units TAB0 ).

Generative models For generative models, we used a similar setup as that found in Donahue et al. (2016) for the generators / decoders, where we used a generator from DCGAN in all experiments.

All models were trained using Adam with a learning rate of 1 × 10 −4 for 1000 epochs for CIFAR10 and CIFAR100 and for 200 epochs for all other datasets.

Contrastive Predictive Coding For Contrastive Predictive Coding (CPC, Oord et al., 2018) , we used a simple a GRU-based PixelRNN (Oord et al., 2016) with the same number of hidden units as the feature map depth.

All experiments with CPC had the global state dimension matched with the size of these recurrent hidden units.

We found both infoNCE and the DV-based estimators were sensitive to negative sampling strategies, while the JSD-based estimator was insensitive.

JSD worked better (1 − 2% accuracy improvement) by excluding positive samples from the product of marginals, so we exclude them in our implementation.

It is quite likely that this is because our batchwise sampling strategy overestimate the frequency of positive examples as measured across the complete dataset.

infoNCE was highly sensitive to the number of negative samples for estimating the log-expectation term (see FIG4 ).

With high sample size, infoNCE outperformed JSD on many tasks, but performance drops quickly as we reduce the number of images used for this estimation.

This may become more problematic for larger datasets and networks where available memory is an issue.

DV was outperformed by JSD even with the maximum number of negative samples used in these experiments, and even worse was highly unstable as the number of negative samples dropped.

Accuracies shown averaged over the last 100 epochs, averaged over 3 runs, for the infoNCE, JSD, and DV DIM losses.

x-axis is log base-2 of the number of negative samples (0 mean one negative sample per positive sample).

JSD is insensitive to the number of negative samples, while infoNCE shows a decline as the number of negative samples decreases.

DV also declines, but becomes unstable as the number of negative samples becomes too low.

DISPLAYFORM0

In order to better understand the metric structure of DIM's representations, we did a nearest-neighbor analysis, randomly choosing a sample from each class in the test set, ordering the test set in terms of L 1 distance in the representation space (to reflect the uniform prior), then selecting the four with the lowest distance.

Our results in FIG3 show that DIM with a local-only objective, DIM(L), learns a representation with a much more interpretable structure across the image.

However, our result potentially highlights an issue with using only consistent information across patches, as many of the nearest neighbors share patterns (colors, shapes, texture) but not class.

Values calculated are points on the grid, and the heatmaps were derived by bilinear interpolation.

Heatmaps were thresholded at the minimum value (or maximum for NDM) for visual clarity.

Highest (or lowest) value is marked on the grid.

NDM here was measured without the sigmoid function.

Figure 11 : Ablation study on CelebA over the global and local parameters, α and β.

The classification task is multinomial, so provided is the average, minimum, and maximum class accuracies across attibutes.

While the local objective is crucial, the global objective plays a stronger role here than with other datasets.

To better understand the effects of hyperparameters α, β, and γ on the representational characteristics of the encoder, we performed several ablation studies.

These illuminate the relative importance of global verses local mutual information objectives as well as the role of the prior.

The results of our ablation study for DIM on CIFAR10 are presented in FIG5 .

In general, good classification performance is highly dependent on the local term, β, while good reconstruction is highly dependent on the global term, α.

However, a small amount of α helps in classification accuracy and a small about of β improves reconstruction.

For mutual information, we found that having a combination of α and β yielded higher MINE estimates.

Finally, for CelebA (Figure 11 ), where the classification task is more fine-grained (is composed of potentially locally-specified labels, such as "lipstick" or "smiling"), the global objective plays a stronger role than with classification on other datasets (e.g., CIFAR10).

Figure 12 : A schematic of learning the Neural Dependency Measure.

For a given batch of inputs, we encode this into a set of representations.

We then shuffle each feature (dimension of the feature vector) across the batch axis.

The original version is sent to the discriminator and given the label "real", while the shuffled version is labeled as "fake".

The easier this task, the more dependent the components of the representation.

Neural Dependency Measures (NDMs) for various β-VAE BID0 Higgins et al., 2016 ) models (0.1, 0.5, 1.0, 1.5, 2.0, 4.0).

Error bars are provided over five runs of each VAE and estimating NDM with 10 different networks.

We find that there is a strong trend as we increase the value of β and that the estimates are relatively consistent and informative w.r.t.

independence as expected.

β = 1.0 give similar numbers.

In addition, the variance over estimates and models is relatively low, meaning the estimator is empirically consistent in this setting.

Here we present experimental details on the occlusion and coordinate prediction tasks.

Training with occlusion .

With occluded inputs, this loss tends to be highest for local features with receptive fields that overlap the occluded region.

Occlusions.

For the occlusion experiments, the sampling distribution for patches to occlude was ad-hoc.

Roughly, we randomly occlude the input image under the constraint that at least one 10 × 10 block of pixels remains visible and at least one 10 × 10 block of pixels is fully occluded.

We chose 10 × 10 based on the receptive fields of local features in our encoder, since it guarantees that occlusion leaves at least one local feature fully observed and at least one local feature fully unobserved.

FIG2 shows the distribution of occlusions used in our tests.

Absolute coordinate prediction For absolute coordinate prediction, the global features y and local features c (i,j) are sampled by 1) feeding an image from the data distribution through the feature encoder, and 2) sampling a random spatial location (i, j) from which to take the local features c (i,j) .

Given y and c (i,j) , we treat the coordinates i and j as independent categorical variables and measure the required log probability using a sum of categorical cross-entropies.

In practice, we implement the prediction function p θ as an MLP with two hidden layers, each with 512 units, ReLU activations, and batchnorm.

We marginalize this objective over all local features associated with a given global feature when computing gradients.

Relative coordinate prediction For relative coordinate prediction, the global features y and local features c (i,j) /c (i ,j ) are sampled by 1) feeding an image from the data distribution through the feature encoder, 2) sampling a random spatial location (i, j) from which to take source local features c (i,j) , and 3) sampling another random location (i , j ) from which to take target local features c (i ,j ) .

In practice, our predictive model for this task uses the same architecture as for the task described previously.

For each global feature y we select one source feature c (i,j) and marginalize over all possible target features c (i ,j ) when computing gradients.

We show here and in our experiments below that we can use prior objective in DIM (Equation 7) to train a high-quality generator of images by training U ψ,P to map to a one-dimensional mixture of two Gaussians implicitly.

One component of this mixture will be a target for the push-forward distribution of P through the encoder while the other will be a target for the push-forward distribution of the generator, Q θ , through the same encoder.

Let G θ : Z → X be a generator function, where the input z ∈ Z is drawn from a simple prior, p(z) (such as a spherical Gaussian).

Let Q θ be the generated distribution and P be the empirical distribution of the training set.

Like in GANs, we will pass the samples of the generator or the training data through another function, E ψ , in order to get gradients to find the parameters, θ.

However, unlike GANs, we will not play the minimax game between the generator and this function.

Rather E ψ will be trained to generate a mixture of Gaussians conditioned on whether the input sample came from P or Q θ :V P = N (µ P , 1), V Q = N (µ Q , 1), U ψ,P = P#E ψ , U ψ,Q = Q θ #E ψ ,where N (µ P , 1) and N (µ Q , 1) are normal distributions with unit variances and means µ P and µ Q respectively.

In order to find the parameters ψ, we introduce two discriminators, T DISPLAYFORM0 : Y → R, and use the lower bounds following defined by the JSD f-GAN:(ψ,φ P ,φ Q ) = arg min ψ arg max DISPLAYFORM1 The generator is trained to move the first-order moment of E U ψ,Q [y] = E p(z) [E ψ (G θ (z))] to µ P : DISPLAYFORM2 Some intuition might help understand why this might work.

As discussed in BID2 , if P and Q θ have support on a low-dimensional manifolds on X , unless they are perfectly aligned, there exists a discriminator that will be able to perfectly distinguish between samples coming from P and Q θ , which means that U ψ,P and U ψ,Q must also be disjoint.

However, to train the generator, U ψ,P and U ψ,Q need to share support on Y in order to ensure stable and non-zero gradients for the generator.

Our own experiments by overtraining the discriminator FIG8 ) confirm that lack of overlap between the two modes of the discriminator is symptomatic of poor training.

Suppose we start with the assumption that the encoder targets, V P and V Q , should overlap.

Unless P and Q θ are perfectly aligned (which according to BID2 is almost guaranteed not to happen with natural images), then the discriminator can always accomplish this task by discarding information about P or Q θ .

This means that, by choosing the overlap, we fix the strength of the encoder.

For the generator and encoder, we use a ResNet architecture (He et al., 2016) identical to the one found in Gulrajani et al. (2017) .

We used the contractive penalty (found in Mescheder et al. (2018) but first introduced in contractive autoencoders (Rifai et al., 2011) ) on the encoder, gradient clipping on the discriminators, and no regularization on the generator.

Batch norm (Ioffe & Szegedy, 2015) was used on the generator, but not on the discriminator.

We trained on 64 × 64 dimensional LSUN BID10 , CelebA (Liu et al., 2015) , and Tiny Imagenet dataset.

A.10 IMAGES GENERATION a) NS-GAN-CP b) WGAN-GP c) Mapping to two Gaussians Figure 16 : Samples of generated results used to get scores in TAB0 .

For every methods, the sample are generated after 100 epochs and the models are the same.

Qualitative results from these three methods show no qualitative difference.

Here, we train a generator mapping to two Gaussian implicitly as described in Section A.8.

Our results (Figure 16 ) show highly realistic images qualitatively competitive to other methods (Gulrajani et al., 2017; Hjelm et al., 2018) .

In order to quantitatively compare our method to GANs, we trained a non-saturating GAN with contractive penalty (NS-GAN-CP) and WGAN-GP (Gulrajani et al., 2017) with identical architectures and training procedures.

Our results TAB0 show that, while our mehtod did not surpass NS-GAN-CP or WGAN-GP in our experiments, they came reasonably close.

@highlight

We learn deep representation by maximizing mutual information, leveraging structure in the objective, and are able to compute with fully supervised classifiers with comparable architectures