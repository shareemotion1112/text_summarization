Modern neural networks are highly overparameterized, with capacity to substantially overfit to training data.

Nevertheless, these networks often generalize well in practice.

It has also been observed that trained networks can often be ``compressed to much smaller representations.

The purpose of this paper is to connect these two empirical observations.

Our main technical result is a generalization bound for compressed networks based on the compressed size that, combined with off-the-shelf compression algorithms, leads to state-of-the-art generalization guarantees.

In particular, we provide the first non-vacuous generalization guarantees for realistic architectures applied to the ImageNet classification problem.

Additionally, we show that compressibility of models that tend to overfit is limited.

Empirical results show that an increase in overfitting increases the number of bits required to describe a trained network.

A pivotal question in machine learning is why deep networks perform well despite overparameterization.

These models often have many more parameters than the number of examples they are trained on, which enables them to drastically overfit to training data BID39 .

In common practice, however, such networks perform well on previously unseen data.

Explaining the generalization performance of neural networks is an active area of current research.

Attempts have been made at adapting classical measures such as VC-dimension BID14 or margin/norm bounds BID4 , but such approaches have yielded bounds that are vacuous by orders of magnitudes.

Other authors have explored modifications of the training procedure to obtain networks with provable generalization guarantees BID10 .

Such procedures often differ substantially from standard procedures used by practitioners, and empirical evidence suggests that they fail to improve performance in practice BID38 .We begin with an empirical observation: it is often possible to "compress" trained neural networks by finding essentially equivalent models that can be described in a much smaller number of bits; see BID9 for a survey.

Inspired by classical results relating small model size and generalization performance (often known as Occam's razor), we establish a new generalization bound based on the effective compressed size of a trained neural network.

Combining this bound with off-the-shelf compression schemes yields the first non-vacuous generalization bounds in practical problems.

The main contribution of the present paper is the demonstration that, unlike many other measures, this measure is effective in the deep-learning regime.

Generalization bound arguments typically identify some notion of complexity of a learning problem, and bound generalization error in terms of that complexity.

Conceptually, the notion of complexity we identify is: complexity = compressed size − remaining structure.(1) The first term on the right-hand side represents the link between generalization and explicit compression.

The second term corrects for superfluous structure that remains in the compressed representation.

For instance, the predictions of trained neural networks are often robust to perturbations of the network weights.

Thus, a representation of a neural network by its weights carries some irrelevant information.

We show that accounting for this robustness can substantially reduce effective complexity.

Our results allow us to derive explicit generalization guarantees using off-the-shelf neural network compression schemes.

In particular:• The generalization bound can be evaluated by compressing a trained network, measuring the effective compressed size, and substituting this value into the bound.• Using off-the-shelf neural network compression schemes with this recipe yields bounds that are state-of-the-art, including the first non-vacuous bounds for modern convolutional neural nets.

The above result takes a compression algorithm and outputs a generalization bound on nets compressed by that algorithm.

We provide a complementary result by showing that if a model tends to overfit then there is an absolute limit on how much it can be compressed.

We consider a classifier as a (measurable) function of a random training set, so the classifier is viewed as a random variable.

We show that the entropy of this random variable is lower bounded by a function of the expected degree of overfitting.

Additionally, we use the randomization tests of BID39 to show empirically that increased overfitting implies worse compressibility, for a fixed compression scheme.

The relationship between small model size and generalization is hardly new: the idea is a variant of Occam's razor, and has been used explicitly in classical generalization theory BID34 BID5 BID27 BID16 BID33 ).

However, the use of highly overparameterized models in deep learning seems to directly contradict the Occam principle.

Indeed, the study of generalization and the study of compression in deep learning has been largely disjoint; the later has been primarily motivated by computational and storage limitations, such as those arising from applications on mobile devices BID9 .

Our results show that Occam type arguments remain powerful in the deep learning regime.

The link between compression and generalization is also used in work by BID0 , who study compressibility arising from a form of noise stability.

Our results are substantially different, and closer in spirit to the work of BID10 ; see Section 3 for a detailed discussion.

BID39 study the problem of generalization in deep learning empirically.

They observe that standard deep net architectures-which generalize well on real-world data-are able to achieve perfect training accuracy on randomly labelled data.

Of course, in this case the test error is no better than random guessing.

Accordingly, any approach to controlling generalization error of deep nets must selectively and preferentially bound the generalization error of models that are actually plausible outputs of the training procedure applied to real-world data.

Following Langford & Caruana (2002) ; BID10 , we make use of the PAC-Bayesian framework BID29 BID7 BID30 .

This framework allows us to encode prior beliefs about which learned models are plausible as a (prior) distribution π over possible parameter settings.

The main challenge in developing a bound in the PAC-Bayes framework bound is to articulate a distribution π that encodes the relative plausibilities of possible outputs of the training procedure.

The key insight is that, implicitly, any compression scheme is a statement about model plausibilities: good compression is achieved by assigning short codes to the most probable models, and so the probable models are those with short codes.

In this section, we recall some background and notation from statistical learning theory.

Our aim is to learn a classifier using data examples.

Each example (x, y) consists of some features x ∈ X and a label y ∈ Y.

It is assumed that the data are drawn identically and independently from some data generating distribution, DISPLAYFORM0 The goal of learning is to choose a hypothesis h : X → Y that predicts the label from the features.

The quality of the prediction is measured by specifying some loss function L; the value L(h(x), y) is a measure of the failure of hypothesis h to explain example (x, y).

The overall quality of a hypothesis h is measured by the risk under the data generating distribution: DISPLAYFORM1 Generally, the data generating distribution is unknown.

Instead, we assume access to training data S n = {(x 1 , y 1 ), . . .

, (x n , y n )}, a sample of n points drawn i.i.d.

from the data generating distribution.

The true risk is estimated by the empirical risk: DISPLAYFORM2 The task of the learner is to use the training data to choose a hypothesisĥ from among some prespecified set of possible hypothesis H, the hypothesis class.

The standard approach to learning is to choose a hypothesisĥ that (approximately) minimizes the empirical risk.

This induces a dependency between the choice of hypothesis and the estimate of the hypothesis' quality.

Because of this, it can happen thatĥ overfits to the training data:L(ĥ) L(ĥ).

The generalization error L(ĥ) −L(ĥ) measures the degree of overfitting.

In this paper, we consider an image classification problem, where x i is an image and y i the associated label for that image.

The selected hypothesis is a deep neural network.

We mostly consider the 0 -1 loss, that is, L(h(x), y) = 0 if the prediction is correct and L(h(x), y) = 1 otherwise.

We use the PAC-Bayesian framework to establish bounds on generalization error.

In general, a PACBayesian bound attempts to control the generalization error of a stochastic classifier by measuring the discrepancy between a pre-specified random classifier (often called prior), and the classifier of interest.

Conceptually, PAC-Bayes bounds have the form: DISPLAYFORM3 where n is the number of training examples, π denotes the prior, and ρ denotes the classifier of interest (often called posterior).More formally, we write L(ρ) = E h∼ρ [L(h)] for the risk of the random estimator.

The fundamental bound in PAC-Bayesian theory is (Catoni, 2007, Thm.

1.2.6 ): Theorem 2.1 (PAC-Bayes).

Let L be a {0, 1}-valued loss function, let π be some probability measure on the hypothesis class, and let α > 1, > 0.

Then, with probability at least 1 − over the distribution of the sample: DISPLAYFORM4 where we define Φ −1 γ as: DISPLAYFORM5 Remark 2.2.

The above formulation of the PAC-Bayesian theorem is somewhat more opaque than other formulations (e. g., McAllester, 2003; BID30 .

This form is significantly tighter when KL/n is large.

See Bégin et al. (2014) ; BID25 for a unified treatment of PAC-Bayesian bounds.

The quality of a PAC-Bayes bound depends on the discrepancy between the PAC-Bayes prior π-encoding the learned models we think are plausible-and ρ, which is the actual output of the learning procedure.

The main challenge is finding good choices for the PAC-Bayes prior π, for which the value of KL(ρ, π) is both small and computable.3 RELATIONSHIP TO PREVIOUS WORK Generalization.

The question of which properties of real-world networks explain good generalization behavior has attracted considerable attention BID23 BID24 BID16 BID17 BID2 BID8 BID20 BID10 BID36 BID31 BID0 ; see BID1 for a review of recent advances.

Such results typically identify a property of real-world networks, formalize it as a mathematical definition, and then use this definition to prove a generalization bound.

Generally, the bounds are very loose relative to the true generalization error, which can be estimated by evaluating performance on held-out data.

Their purpose is not to quantify the actual generalization error, but rather to give qualitative evidence that the property underpinning the generalization bound is indeed relevant to generalization performance.

The present paper can be seen in this tradition: we propose compressibility as a key signature of performant real-world deep nets, and we give qualitative evidence for this thesis in the form of a generalization bound.

The idea that compressibility leads to generalization has a long history in machine learning.

Minimum description length (MDL) is an early formalization of the idea BID34 .

BID16 applied MDL to very small networks, already recognizing the importance of weight quantization and stochasticity.

More recently, BID0 consider the connection between compression and generalization in large-scale deep learning.

The main idea is to compute a measure of noise-stability of the network, and show that it implies the existence of a simpler network with nearly the same performance.

A variant of a known compression bound (see BID30 for a PAC-Bayesian formulation) is then applied to bound the generalization error of this simpler network in terms of its code length.

In contrast, the present paper develops a tool to leverage existing neural network compression algorithms to obtain strong generalization bounds.

The two papers are complementary: we establish non-vacuous bounds, and hence establish a quantitative connection between generalization and compression.

An important contribution of Arora et al. FORMULA3 is obtaining a quantity measuring the compressibility of a neural network; in contrast, we apply a compression algorithm and witness its performance.

We note that their compression scheme is very different from the sparsity-inducing compression schemes BID9 we use in our experiments.

Which properties of deep networks allow them to be sparsely compressed remains an open question.

To strengthen a naïve Occam bound, we use the idea that deep networks are insensitive to mild perturbations of their weights, and that this insensitivity leads to good generalization behavior.

This concept has also been widely studied (e.g., BID23 BID24 BID16 BID17 BID2 BID8 BID20 BID10 .

As we do, some of these papers use a PAC-Bayes approach BID24 BID10 .

arrive at a bound for non-random classifiers by computing the tolerance of a given deep net to noise, and bounding the difference between that net and a stochastic net to which they apply a PAC-Bayes bound.

Like the present paper, BID24 ; BID10 work with a random classifier given by considering a normal distribution over the weights centered at the output of the training procedure.

We borrow the observation of BID10 that the stochastic network is a convenient formalization of perturbation robustness.

The approaches to generalization most closely related to ours are, in summary: DISPLAYFORM6 Dziugaite & Roy (2017) Perturbation Robustness Perturbation Robustness BID0 Compressibility (from Perturbation Robustness) Present paper Compressibility and Perturbation RobustnessThese represent the best known generalization guarantees for deep neural networks.

Our bound provides the first non-vacuous generalization guarantee for the ImageNet classification task, the de facto standard problem for which deep learning dominates.

It is also largely agnostic to model architecture: we apply the same argument to both fully connected and convolutional networks.

This is in contrast to some existing approaches that require extra analysis to extend bounds for fully connected networks to bounds for convolutional networks BID21 BID0 .Compression.

The effectiveness of our work relies on the existence of good neural network compression algorithms.

Neural network compression has been the subject of extensive interest in the last few years, motivated by engineering requirements such as computational or power constraints.

We apply a relatively simple strategy in this paper in the line of , but we note that our bound is compatible with most forms of compression.

See Cheng et al. (2018) for a survey of recent results in this field.

We first describe a simple Occam's razor type bound that translates the quality of a compression into a generalization bound for the compressed model.

The idea is to choose the PAC-Bayes prior π such that greater probability mass is assigned to models with short code length.

In fact, the bound stated in this section may be obtained as a simple weighted union bound, and a variation is reported in McAllester (2013).

However, embedding this bound in the PAC-Bayesian framework allows us to combine this idea, reflecting the explicit compressible structure of trained networks, with other ideas reflecting different properties of trained networks.

We consider a non-random classifier by taking the PAC-Bayes posterior ρ to be a point mass atĥ, the output of the training (plus compression) procedure.

Recall that computing the PAC-Bayes bound effectively reduces to computing KL(ρ, π).

Theorem 4.1.

Let |h| c denote the number of bits required to represent hypothesis h using some pre-specified coding c. Let ρ denote the point mass at the compressed modelĥ.

Let m denote any probability measure on the positive integers.

There exists a prior π c such that: DISPLAYFORM0 This result relies only on the quality of the chosen coding and is agnostic to whether a lossy compression is applied to the model ahead of time.

In practice, the code c is chosen to reflect some explicit structure-e.g., sparsity-that is imposed by a lossy compression.

Proof.

Let H c ⊆ H denote the set of estimators that correspond to decoded points, and note that h ∈ H c by construction.

Consider the measure π c on H c : DISPLAYFORM1 As c is injective on H c , we have that Z ≤ 1.

We may thus directly compute the KL-divergence from the definition to obtain the claimed result.

Remark 4.2.

To apply the bound in practice, we must make a choice of m. A pragmatic solution is to simply consider a bound on the size of the model to be selected (e.g. in many cases it is reasonable to assume that the encoded model is smaller than 2 64 bytes, which is 2 72 bits), and then consider m to be uniform over all possible lengths.

The simple bound above applies to an estimator that is compressible in the sense that its encoded length with respect to some fixed code is short.

However, such a strategy does not consider any structure on the hypothesis space H. In practice, compression schemes will often fail to exploit some structure, and generalization bounds can be (substantially) improved by accounting for this fact.

We empirically observe that trained neural networks are often tolerant to low levels of discretization of the trained weights, and also tolerant to some low level of added noise in the trained weights.

Additionally, quantization is an essential step in numerous compression strategies .

We construct a PAC-Bayes bound that reflects this structure.

This analysis requires a compression scheme specified in more detail.

We assume that the output of the compression procedure is a triplet (S, C, Q), where S = {s 1 , . . .

, s k } ⊆ {1, . . . , p} denotes the location of the non-zero weights, C = {c 1 , . . .

, c r } ⊆ R is a codebook, and Q = (q 1 , . . .

, q k ), q i ∈ {1, . . . , r} denotes the quantized values.

Most state-of-the-art compression schemes can be formalized in this manner .Given such a triplet, we define the corresponding weight w(S, Q, C) ∈ R p as: DISPLAYFORM0 Following BID24 ; Dziugaite & Roy FORMULA3 , we bound the generalization error of a stochastic estimator given by applying independent random normal noise to the nonzero weights of the network.

Formally, we consider the (degenerate) multivariate normal centered at w: ρ ∼ N (w, σ 2 J), with J being a diagonal matrix such that J ii = 1 if i ∈ S and J ii = 0 otherwise.

Theorem 4.3.

Let (S, C, Q) be the output of a compression scheme, and let ρ S,C,Q be the stochastic estimator given by the weights decoded from the triplet and variance σ 2 .

Let c denote some arbitrary (fixed) coding scheme and let m denote an arbitrary distribution on the positive integers.

Then, for any τ > 0, there is some PAC-Bayes prior π such that: DISPLAYFORM1 Normal(c j , τ 2 ) .Note that we have written the KL-divergence of a distribution with a unnormalized measure (the last term), and in particular this term may (and often will) be negative.

We defer the construction of the prior π and the proof of Theorem 4.3 to the supplementary material.

Remark 4.4.

We may obtain the first term k log r + |S| c + |C| c from the simple Occam's bound described in Theorem 4.1 by choosing the coding of the quantized values Q as a simple array of integers of the correct bit length.

The second term thus describes the adjustment (or number of bits we "gain back") from considering neighbouring estimators.

In this section we present examples combining our theoretical arguments with state-of-the-art neural network compression schemes.

1 Recall that almost all other approaches to bounding generalization error of deep neural networks yield vacuous bounds for realistic problems.

The one exception is BID10 , which succeeds by retraining the network in order to optimize the generalization bound.

We give two examples applying our generalization bounds to the models output by modern neural net compression schemes.

In contrast to earlier results, this leads immediately to non-vacuous bounds on realistic problems.

The strength of the Occam bound provides evidence that the connection between compressibility and generalization has substantive explanatory power.

We report 95% confidence bounds based on the measured effective compressed size of the networks.

The bounds are achieved by combining the PAC-Bayes bound Theorem 2.1 with Theorem 4.3, showing that KL(ρ, π) is bounded by the "effective compressed size".

We note a small technical modification: we choose the prior variance τ 2 layerwise by a grid search, this adds a negligible contribution to the effective size (see Appendix A.1 for the technical details of the bound).LeNet-5 on MNIST.

Our first experiment is performed on the MNIST dataset, a dataset of 60k grayscale images of handwritten digits.

We fit the LeNet-5 (LeCun et al., 1998) network, one of the first convolutional networks.

LeNet-5 has two convolutional layers and two fully connected layers, for a total of 431k parameters.

We apply a pruning and quantization strategy similar to that described in .

We prune the network using Dynamic Network Surgery BID12 , pruning all but 1.5% of the network weights.

We then quantize the non-zero weights using a codebook with 4 bits.

The location of the non-zero coordinates are stored in compressed sparse row format, with the index differences encoded using arithmetic compression.

We consider the stochastic classifier given by adding Gaussian noise to each non-zero coordinate before each forward pass.

We add Gaussian noise with standard deviation equal to 5% of the difference between the largest and smallest weight in the filter.

This results in a negligible drop in classification performance.

We obtain a bound on the training error of 46% (with 95% confidence).

The effective size of the compressed model is measured to be 6.23 KiB.ImageNet.

The ImageNet dataset (Russakovsky et al., 2015) is a dataset of about 1.2 million natural images, categorized into 1000 different classes.

ImageNet is substantially more complex than the MNIST dataset, and classical architectures are correspondingly more complicated.

For example, AlexNet BID22 and VGG-16 BID37 contain 61 and 128 million parameters, respectively.

Non-vacuous bounds for such models are still out of reach when applying our bound with current compression techniques.

However, motivated by computational restrictions, there has been extensive interest in designing more parsimonious architectures that achieve comparable or better performance with significantly fewer parameters BID19 BID18 BID40 .

By combining neural net compression schemes with parsimonious models of this kind, we demonstrate a non-vacuous bounds on models with better performance than AlexNet.

Our simple Occam bound requires only minimal assumptions, and can be directly applied to existing compressed networks.

For example, BID19 introduce the SqueezeNet architecture, and explicitly study its compressibility.

They obtain a model with better performance than AlexNet but that can be written in 0.47 MiB. A direct application of our naïve Occam bound yields non-vacuous bound on the test error of 98.6% (with 95% confidence).

To apply our stronger bound-taking into account the noise robustness-we train and compress a network from scratch.

We consider Mobilenet 0.5 BID18 , which in its uncompressed form has better performance and smaller size than SqueezeNet (Iandola et al., 2016).Zhu & Gupta (2017) study pruning of MobileNet in the context of energy-efficient inference in resource-constrained environments.

We use their pruning scheme with some small adjustments.

In particular, we use Dynamic Network Surgery BID12 as our pruning method but follow a similar schedule.

We prune 67 % of the total parameters.

The pruned model achieves a validation accuracy of 60 %.

We quantize the weights using a codebook strategy .

We consider the stochastic classifier given by adding Gaussian noise to the non-zero weights, with the variance set in each layer so as not to degrade our prediction performance.

For simplicity, we ignore biases and batch normalization parameters in our bound, as they represent a negligible fraction of the parameters.

We consider top-1 accuracy (whether the most probable guess is correct) and top-5 accuracy (whether any of the 5 most probable guesses is correct).

The final "effective compressed size" is 350 KiB.

The

We have shown that compression results directly imply generalization bounds, and that these may be applied effectively to obtain non-vacuous bounds on neural networks.

In this section, we provide a complementary view: overfitting implies a limit on compressibility.

Theory.

We first prove that the entropy of estimators that tend to overfit is bounded in terms of the expected degree of overfitting.

That implies the estimators fail to compress on average.

As previously, consider a sample S n = {(x 1 , y 1 ), . . .

, (x n , y n )} sampled i.i.d.

from some distribution D, and an estimator (or selection procedure)ĥ, which we consider as a (random) function of the training data.

The key observation is: DISPLAYFORM0 That is, the probability of misclassifying an example in the training data is smaller than the probability of misclassifying a fresh example, and the expected strength of this difference is determined by the expected degree of overfitting.

By Bayes' rule, we thus see that the moreĥ overfits, the better it is able to distinguish a sample from the training and testing set.

Such an estimatorĥ must thus "remember" a significant portion of the training data set, and its entropy is thus lower bounded by the entropy of its "memory".Theorem 6.1.

Let L,L, andĥ be as in the text immediately preceeding the theorem.

For simplicity, assume that both the sample space X × Y and the hypothesis set H are discrete.

Then, DISPLAYFORM1 where g denotes some non-negative function (given explicitly in the proof).We defer the proof to the supplementary material.

Experiments.

We now study this effect empirically.

The basic tool is the randomization test of BID39 : we consider a fixed architecture and a number of datasets produced by randomly relabeling the categories of some fraction of examples from a real-world dataset.

If the model has sufficiently high capacity, it can be fit with approximately zero training loss on each dataset.

In this case, the generalization error is given by the fraction of examples that have been randomly relabeled.

We apply a standard neural net compression tool to each of the trained models, and we observe that the models with worse generalization require more bits to describe in practice.

For simplicity, we consider the CIFAR-10 dataset, a collection of 40000 images categorized into 10 classes.

We fit the ResNet BID15 architecture with 56 layers with no pre-processing and no penalization on the CIFAR-10 dataset where the labels are subjected to varying levels of randomization.

As noted in BID39 , the network is able to achieve 100 % training accuracy no matter the level of randomization.

We then compress the networks fitted on each level of label randomization by pruning to a given target sparsity.

Surprisingly, all networks are able to achieve 50 % sparsity with essentially no loss of training accuracy, even on completely random labels.

However, we observe that as the compression level increases further, the scenarios with more randomization exhibit a faster decay in training accuracy, see FIG0 .

This is consistent with the fact that network size controls generalization error.

It has been a long standing observation by practitioners that despite the large capacity of models used in deep learning practice, empirical results demonstrate good generalization performance.

We show that with no modifications, a standard engineering pipeline of training and compressing a network leads to demonstrable and non-vacuous generalization guarantees.

These are the first such results on networks and problems at a practical scale, and mirror the experience of practitioners that best results are often achieved without heavy regularization or modifications to the optimizer BID38 .The connection between compression and generalization raises a number of important questions.

Foremost, what are its limitations?

The fact that our bounds are non-vacuous implies the link between compression and generalization is non-trivial.

However, the bounds are far from tight.

If significantly better compression rates were achievable, the resulting bounds would even be of practical value.

For example, if a network trained on ImageNet to 90% training and 70% testing accuracy could be compressed to an effective size of 30 KiB-about one order of magnitude smaller than our current compression-that would yield a sharp bound on the generalization error.

A PROOF OF THEOREM 4.3 In this section we describe the construction of the prior π and prove the bound on the KL-divergence claimed in Theorem 4.3.

Intuitively, we would like to express our prior as a mixture over all possible decoded points of the compression algorithm.

More precisely, define the mixture component π S,Q,C associated with a triplet (S, Q, C) as: DISPLAYFORM0 We then define our prior π as a weighted mixture over all triplets, weighted by the code length of the triplet: DISPLAYFORM1 where the sum is taken over all S and C which are representable by our code, and all Q = (q 1 , . . .

, q k ) ∈ {1, . . . , r} k .

In practice, S takes values in all possible subsets of {1, . . .

, p}, and C takes values in F r , where F ⊆ R is a chosen finite subset of representable real numbers (such as those that may be represented by IEEE-754 single precision numbers), and r is a chosen quantization level.

We now give the proof of Theorem 4.3.Proof.

We have that: DISPLAYFORM2 where we must have Z ≤ 1 by the same argument as in the proof of Theorem 4.1Suppose that the output of our compression algorithm is a triplet (Ŝ,Q,Ĉ).

We recall that our posterior ρ is given by a normal centered at w(Ŝ,Q,Ĉ) with variance σ 2 , and we may thus compute the KL-divergence: DISPLAYFORM3 We are now left with the mixture term, which is a mixture of r k many terms in dimension k, and thus computationally untractable.

However, we note that we are in a special case where the mixture itself is independent across coordinates.

Indeed, let φ τ denote the density of the univariate normal distribution with mean 0 and variance τ 2 , we note that we may write the mixture as: DISPLAYFORM4 Additionally, as our chosen stochastic estimator ρ is independent over the coordinates, the KLdivergence decomposes over the coordinates, to obtain: DISPLAYFORM5 Plugging the above computation into (13) gives the desired result.

Although Theorem 4.3 contains the main mathematical contents of our bound, applying the bound in a fully correct fashion requires some amount of minutiae and book-keeping we detail in this section.

In particular, we are required to select a number of parameters (such as the prior variances).

We extend the bound to account for such unrestricted (and possibly data-dependent) parameter selection.

Typically, such adjustments have a negligible effect on the computed bounds.

Theorem A.1 (Union Bound for Discrete Parameters).

Let π ξ , ξ ∈ Ξ, denote a family of priors parameterized by a discrete parameter ξ, which takes values in a finite set Ξ. There exists a prior π such that for any posterior ρ and any ξ ∈ Ξ: DISPLAYFORM0 Proof.

We define π as a uniform mixture of the π ξ : DISPLAYFORM1 We then have that: DISPLAYFORM2 but we can note that DISPLAYFORM3 , from which we deduce that: DISPLAYFORM4 We make liberal use of this variant to control a number of discrete parameters which are chosen empirically (such as the quantization resolution at each layer).

We also use this bound to control a number of continuous quantities (such as the prior variances) by discretizing these quantities as IEEE-754 single precision (32 bit) floating point numbers.

B EXPERIMENT DETAILS

We train the baseline model for LeNet-5 using stochastic gradient descent with momentum and no data augmentation.

The batch size is set to 1024, and the learning rate is decayed using an inverse time decay starting at 0.01 and decaying every 125 steps.

We apply a small 2 penalty of 0.005.

We train a total of 20000 steps.

We carry out the pruning using Dynamic Network Surgery BID12 .

The threshold is selected per layer as the mean of the layer coefficients offset by a constant multiple of the standard deviation of the coefficients, where the multiple is piecewise constant starting at 0.0 and ending at 4.0.

We choose the pruning probability as a piecewise constant starting at 1.0 and decaying to 10 −3 .

We train for 30000 steps using the ADAM optimizer.

We quantize all the weights using a 4 bit codebook per layer initialized using k-means.

A single cluster in each weight is given to be exactly zero and contains the pruned weights.

The remaining clusters centers are learned using the ADAM optimizer over 1000 steps.

MobileNets are a class of networks that make use of depthwise separable convolutions.

Each layer is composed of two convolutions, with one depthwise convolution and one pointwise convolution.

We use the pre-trained MobileNet model provided by Google as our baseline model.

We then prune the pointwise (and fully connected) layers only, using Dynamic Network Surgery.

The threshold is set for each weight as a quantile of the absolute values of the coordinates, which is increased according to the schedule given in (Zhu & Gupta, 2017) .

As the lower layers are smaller and more sensitive, we scale the target sparsity for each layer according to the size of the layer.

The target sparsity is scaled linearly between 65% and 75% as a proportion of the number of elements in the layer compared to the largest layer (the final layer).

We use stochastic gradient descent with momentum and decay the learning with an inverse time decay schedule, starting at 10 −3 and decaying by 0.05 every 2000 steps.

We use a minibatch size of 64 and train for a total of 300000 steps, but tune the pruning schedule so that the target sparsity is reached after 200000 steps.

We quantize the weights by using a codebook for each layer with 6 bits for all layers except the last fully connected layer which only has 5 bits.

The pointwise and fully connected codebooks have a reserved encoding for exact 0, whereas the non-pruned depthwise codebooks are fully learned.

We initialize the cluster assignment using k-means and train the cluster centers for 20000 steps with stochastic gradient with momentum with a learning rate of 10 −4 .

Note that we also modify the batch normalization moving average parameters in this step so that it adapts faster, choosing .99 as the momentum parameter for the moving averages.

To witness noise robustness, we only add noise to the pointwise and fully connected layer.

We are able to add Gaussian noise with standard deviation equal to 2% of the difference in magnitude between the largest and smallest coordinate in the layer for the fully connected layer.

For pointwise layers we add noise equal to 1% of the difference scaled linearly by the relative size of the layer compared to the fully connected layer.

These quantities were chosen to minimally degrade the training performance while obtaining good improvements on the generalization bound: in our case, we observe that the top-1 training accuracy is reduced to 65% with noise applied from 67% without noise.

and its entropy is thus lower bounded by the entropy of its "memory".

Quantitatively, we note that the quality ofĥ as a discriminator between the training and testing set is captured by the quantities DISPLAYFORM0 We may interpret p n as the average proportion of false positives and q n as the average proportion of true negatives when viewingĥ as a classifier.

We prove that if those quantities are substantially different from a random classifier, thenĥ must have high entropy.

We formalize this statement and provide a proof below.

Theorem C.1.

Let S = {(x 1 , y 1 ), . . .

, (x n , y n )} be sampled i.i.d.

from some distribution D, and letĥ be a selection procedure, which is only a function of the unordered set S. Let us viewĥ as a random quantity through the distribution induced by the sample S. For simplicity, we assume that both the sample space X × Y and the hypothesis set H are discrete.

We have that: DISPLAYFORM1 where g denotes some non-negative function.

Proof.

Consider a sequence of pairs (s i andĥ have the same distribution as if they were sampled from the procedure described before (19).

Namely, sample S i.i.d.

according to the data generating distribution, and letĥ be the corresponding estimator, B i an independent Bernoulli random variable, and L i = L(ĥ(x), y) where (x, y) is sampled uniformly from S if B i = 0 and according to the data generating distribution if B i = 1.

Note that this distribution does not depend on i due to the assumption thatĥ is measurable with respect to the unordered sample S. By (19), we thus deduce that: DISPLAYFORM2 which yields the desired result by taking expectation over the distribution ofĥ,L(ĥ).Similarly, we may compute the distribution of B i conditional on the event where L 0 i = 0, as P(B i = 0 | L 0 i = 0) = q n .

By definition, we now have that: DISPLAYFORM3 where h b (p) denotes the binary entropy function.

Finally, we apply the chain rule for entropy.

We note that H(B | E,ĥ) = H(B,ĥ | E) − H(ĥ | E),

@highlight

We obtain non-vacuous generalization bounds on ImageNet-scale deep neural networks by combining an original PAC-Bayes bound and an off-the-shelf neural network compression method.