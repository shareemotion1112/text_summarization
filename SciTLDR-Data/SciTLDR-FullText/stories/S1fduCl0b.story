Lifelong learning is the problem of learning multiple consecutive tasks in a sequential manner where knowledge gained from previous tasks is retained and used for future learning.

It is essential towards the development of intelligent machines that can adapt to their surroundings.

In this work we focus on a lifelong learning approach to generative modeling where we continuously incorporate newly observed streaming distributions into our learnt model.

We do so through a student-teacher architecture which allows us to learn and preserve all the distributions seen so far without the need to retain the past data nor the past models.

Through the introduction of a novel cross-model regularizer, the student model leverages the information learnt by the teacher, which acts as a summary of everything seen till now.

The regularizer has the additional benefit of reducing the effect of catastrophic interference that appears when we learn over streaming data.

We demonstrate its efficacy on streaming distributions as well as its ability to learn a common latent representation across a complex transfer learning scenario.

Deep unsupervised generative learning allows us to take advantage of the massive amount of unlabeled data available in order to build models that efficiently compress and learn an approximation of the true data distribution.

It has numerous applications such as image denoising, inpainting, super-resolution, structured prediction, clustering, pre-training and many more.

However, something that is lacking in the modern ML toolbox is an efficient way to learn these deep generative models in a sequential, lifelong setting.

In a lot of real world scenarios we observe distributions sequentially.

Examples of this include streaming data from sensors such as cameras and microphones or other similar time series data.

A system can also be resource limited wherein all of the past data or learnt models cannot be stored.

We are interested in the lifelong learning setting for generative models where data arrives sequentially in a stream and where the storage of all data is infeasible.

Within the stream, instances are generated according to some non-observed distribution which changes at given time-points.

We assume we know the time points at which the transitions occur and whether the latent distribution is a completely new one or one that has been observed before.

We do not however know the underlying identity of the individual distributions.

Our goal is to learn a generative model that can summarize all the distributions seen so far in the stream.

We give an example of such a setting in figure 1(a) using MNIST BID19 , where we have three unique distributions and one that is repeated.

Since we only observe one distribution at a time we need to develop a strategy of retaining the previously learnt knowledge (i.e. the previously learnt distributions) and integrate it into future learning.

To accumulate additional distributions in the current generative model we utilize a student-teacher architecture similar to that in distillation methods BID9 ; BID4 .

The teacher contains a summary of all past distributions and is used to augment the data used to train the student model.

The student model thus receives data samples from the currently observable distribution as well as synthetic data samples from previous distributions.

This allows the student model to learn a distribution that summarizes the current as well as all previously observed distributions.

Once a new distribution shift occurs the existing teacher model is discarded, the student becomes the teacher and a new student is instantiated.

We further leverage the generative model of the teacher by introducing a regularizer in the learning objective function of the student that brings the posterior distribution of the latter close to that of the former.

This allows us to build upon and extend the teacher's generative model in the student each time the latter is re-instantiated (rather than re-learning it from scratch).

By coupling this regularizer with a weight transfer from the teacher to the student we also allow for faster convergence of the student model.

We empirically show that the regularizer allows us to learn a much larger set of distributions without catastrophic interference BID23 .We build our lifelong generative models over Variational Autoencoders (VAEs) BID17 .

VAEs learn the posterior distribution of a latent variable model using an encoder network; they generate data by sampling from a prior and decoding the sample through a conditional distribution learnt by a decoder network.

Using a vanilla VAE as a teacher to generate synthetic data for the student is problematic due to a couple of limitations of the VAE generative process.

1) Sampling the prior can select a point in the latent space that is in between two separate distributions, causing generation of unrealistic synthetic data and eventually leading to loss of previously learnt distributions.

2) Additionally, data points mapped to the posterior that are further away from the prior mean will be sampled less frequently resulting in an unbalanced sampling of the constituent distributions.

Both limitations can be understood by visually inspecting the learnt posterior distribution of a standard VAE evaluated on test images from MNIST as shown in figure 1(b).

To address the VAE's sampling limitations we decompose the latent variable vector into a continuous and a discrete component.

The discrete component is used to summarize the discriminative information of the individual generative distributions while the continuous caters for the remaining sample variability.

By independently sampling the discrete and continuous components we preserve the distributional boundaries and circumvent the two problems above.

This sampling strategy, combined with the proposed regularizer allows us to learn and remember all the individual distributions observed in the past.

In addition we are also able to generate samples from any of the past distributions at will; we call this property consistent sampling.

Past work in sequential learning of generative models has focused on learning Gaussian mixture models BID28 ; BID3 or on variational methods such as Variational EM Ghahramani & Attias (2000) .

Work that is closer to ours is the online or sequential learning of generative models in a streaming setting.

Variational methods have been adapted for a streaming setting, e.g: Streaming Variational Bayes BID0 , Streaming Variational Mixture models Tank et al. (2015) , and the Population Posterior BID24 .

However their learning objectives are very different from ours.

The objective of these methods is to adjust the learnt model such that it reflects the current data distribution as accurately as possible, while forgetting the previously observed distributions.

Instead we want to do lifelong learning and retain all previously observed distributions within our learnt model.

As far as we know our work is the first one that tries to bring generative models, and in particular VAEs, into a lifelong setting where distributions are seen, learnt, and remembered sequentially.

VAEs rely on an encoder and a decoder neural network in order to learn the parameters of the posterior and likelihood.

One of the central problems that arise when training a neural network in an sequential manner is that it causes the model to run into the problem of catastrophic interference BID23 .

Catastrophic interference appears when we train neural networks in a sequential manner and model parameters start to become biased to the most recent samples observed, while forgetting what was learnt from older samples.

This generally happens when we stop exposing the model to past data.

There have been a number of attempts to solve the problem of catastrophic interference in neural networks.

These range from distillation methods such as the original method BID9 and ALTM Furlanello et al. (2016) , to utilizing privileged information BID21 , as well as transfer learning approaches such as Learning Without Forgetting BID20 and methods that relay information from previously learnt hidden layers such as in Progressive Neural Networks BID27 and Deep Block-Modular Neural Networks Terekhov et al. (2015) .

All of these methods necessitate the storage of previous models or data; our method does not.

The recent work of elastic weight consolidation (EWC) BID18 utilizes the Fisher Information matrix (FIM) to avoid the problem of catastrophic interference.

The FIM captures the sensitivity of the log-likelihood with respect to the model parameters; EWC leverages this (via a linear approximation of the FIM) to control the change of model parameter values between varying distributions.

Intuitively, important parameters should not have their values changed, while non-important parameters are left unconstrained.

Since EWC assumes model parameters being distributed under an exponential family, it allows for the utilization of the FIM as a quadratic approximationJeffreys (1946) to the Kullback-Leibler (KL) divergence.

Our model makes no such distributional assumptions about the model parameters.

Instead of constraining the parameters of the model as in EWC, we restrict the posterior representation of the student model to be close to that of the teacher for the previous distributions accumulated by the teacher.

This allows the model parameters to vary as necessary in order to best fit the data.

We consider an unsupervised setting where we observe a sample X of K ??? 1 realizations X = {x (0) , x (1) , ..., x (K) } from an unknown true distribution P * (x) with x ??? R N .

We assume that the data is generated by a random process involving a non-observed random variable z ??? R M .

In order to incorporate our prior knowledge we posit a prior P (z) over z. Our objective is to approximate the true underlying data distribution by a model P ?? (x) such that P ?? (x) ??? P * (x).Given a latent variable model P ?? (x|z)P (z) we obtain the marginal likelihood P ?? (x) by integrating out the latent variable z from the joint distribution.

The joint distribution can in turn be factorized using the conditional distribution P ?? (x|z) or the posterior P ?? (z|x).

DISPLAYFORM0 We model the conditional distribution P ?? (x|z) by a decoder, typically a neural network.

Very often the marginal likelihood P ?? (x) will be intractable because the integral in equation FORMULA0 does not have an analytical form nor an efficient estimator (Kingma (2017)).

As a result the respective posterior distribution, P ?? (z|x), is also intractable.

Variational inference side-steps the intractability of the posterior by approximating it with a tractable distribution Q ?? (z|x) ??? P ?? (z|x).

VAEs use an encoder (generally a neural network) to model the approximate posterior Q ?? (z|x) and optimize the parameters ?? to minimize the reverse KL divergence KL[Q ?? (z|x)||P ?? (z|x)] between the approximate posterior distribution Q ?? (z|x) and the true posterior P ?? (z|x).

Given that Q ?? (z|x) is a powerful model (such that the KL divergence against the true posterior will be close to zero) we maximize the tractable Evidence Lower BOund (ELBO) to the intractable marginal likelihood.

DISPLAYFORM1 By sharing the variational parameters ?? of the encoder across the data points (amortized inference Gershman & Goodman FORMULA0 ), variational autoencoders avoid per-data optimization loops typically needed by mean-field approaches.

The standard setting in maximum-likelihood generative modeling is to estimate the set of parameters ?? that will maximize the marginal likelihood P ?? (x) for data sample X generated IID from a single true data distribution P * (x).

In our work we assume the data are generated from multiple distributions P * DISPLAYFORM0 In classical batch generative modelling, the individual data points are not associated with the specific generative distributions P * i (x).

Instead, the whole sample X is considered to be generated from the mixture distribution P * (x).

Latent variable models P ?? (x, z) = P ?? (x|z)P (z) (such as VAEs) capture the complex structures in P * (x) by conditioning the observed variables x on the latent variables z and combining these in (possibly infinite) mixtures P ?? (x) = P ?? (x|z)P (z)??z.

Our sequential setting is vastly different from the batch approach described above.

We receive a stream of (possibly infinite) data X = {X 1 , X 2 , . . .} where the data samples X i = {x DISPLAYFORM1 } originate from the components P * i (x) of the generative distribution.

At any given time we observe the latest sample X i generated from a single component P * i (x) without access to any of the previous samples generated by the other components of P * (x).

Our goal is to sequentially build an approximation P ?? (x) of the true mixture P * (x) by only observing data from a single component P * i (x) at a time.

Figure 2: Shown above is the relationship of the teacher and the student generative models.

Data generated from the teacher model is used to augment the student model's training data and consistency is applied between posteriors.

Best viewed in color.

To enable lifelong generative learning we propose a dual model architecture based on a student-teacher model.

The teacher and the student have rather different roles throughout the learning process: the teacher's role is to preserve the memory of the previously learned tasks and to pass this knowledge onto the student; the student's role is to learn the distributions over the new incoming data while accommodating for the knowledge obtained from the teacher.

The dual model architecture is summarized in figure 2.The top part represents the teacher model.

At any given time the teacher contains a summary of all previous distributions within the learned parameters of the encoder Q ?? (z|x) and the decoder P ?? (x|z).

The teacher is used to generate synthetic samplesx from these past distributions by decoding samples from the prior??? ??? P (z) through the decoderx ??? P ?? (x|???).

The generated synthetic samplesx are passed onto the student model as a form of knowledge transfer about the past distributions.

The bottom part of figure 2 represents the student, which is responsible for updating the parameters of the encoder Q ?? (z|x) and decoder P ?? (x|z) models over the newly observed data.

The student is exposed to a mixture of learning instances x sampled from x ??? P (??)P (x|??), ?? ??? Ber(??); it sees synthetic instances generated by the teacher P (x|?? = 0) = P ?? (x|z), and real ones sampled from the currently active training distribution P (x|?? = 1) = P * (x).

The mean ?? of the Bernouli distribution controls the sampling proportion of the previously learnt distributions to the current one.

If we have seen k distinct distributions prior to the currently active one then ?? = k k+1 .

In this way we ensure that all the past distributions and the current one are equally represented in the training set used by the student model.

Once a new distribution is signalled, the old teacher is dropped, the student model is frozen and becomes the new teacher (?? ??? ??, ?? ??? ??), and a new student is initiated with the latest weights ?? and ?? from the previous student (the new teacher).

Each new student instantiation uses the input data mix to learn a new approximate posterior Q ?? (z|x).

In addition to being initiated by the new teacher's weights and receiving information about the teacher's knowledge via the synthetic samplesx, we further foster the lifelong learning idea by bringing the latent variable posterior induced by the student model closer to the respective posterior induced by the teacher model.

We enforce the latter constraint only over the synthetic samples, ensuring that the previously learnt latent variable posteriors are preserved over the different models.

In doing so, we alleviate the effect of catastrophic interference.

To achieve this, we complement the classical VAE objective (equation FORMULA1 ) with a term minimizing the KL divergence KL[Q ?? (z|x)||Q ?? (z|x)] between the student's and the teacher's posteriors over the synthetic datax.

The teacher's encoder model, which already has the accumulated knowledge from the previous learning steps, is thus reused within the new student's objective.

Under certain mild assumptions, we show that this objective reparameterizes the student model's posterior, while preserving the same learning objective as a standard VAE (appendix section 7.0.1).

A critical component of our model is the synthetic data generation by the teacher's decoder x ??? P ?? (x|z).

The synthetic samples need to be representative of all the previously observed distributions in order to provide the student with ample information about the learning history.

The teacher generates these synthetic samples by first sampling the latent variable from the prior z ??? P (z) followed by the decoding stepx ??? P ?? (x|???).

As we will describe shortly, the latent variable??? has a categorical component which corresponds to all the past distributions.

This categorical component allows us to uniformly sample synthetic instances from all past distributions.

A simple unimodal prior distribution P (z), such as the isotropic Gaussian typically used in classical VAEs, results in an undersampling of the data points that are mapped to a posterior mean that is further away from the prior mean.

Visualizing the 2d latent posterior of MNIST in figure 1(b) allows us to get a better intuition of this problem.

If for example the prior mean corresponds to a point in latent space between two disparate distributions, the sample generated will not correspond to a sample from the real distribution.

Since we use synthetic samples from the teacher in the student model, this aliased sample corresponding to the prior mean, will be reused over and over again, causing corruption in the learning process.

In addition, we would under represent the respective true distributions in the learning input mix of the student and eventually lead to distribution loss.

We circumvent this in our model by decomposing the latent variable z into a discrete component DISPLAYFORM0 The discrete component z d shall summarise the most discriminative information about each of the true generating distributions P * i (x).

We use the uniform multivariate categorical prior z d ??? Cat( 1 J ) to represent it and the same parametric family for the approximate posterior Q ?? (z|x).

The continuous z c component is the global representation of the distributional variability and we use the multivariate standard normal as the prior z c ??? N (0, I) and the isotropic multivariate normal N (??, ?? 2 I) for the approximate posterior.

When generating synthetic data, the teacher now independently samples from the discrete and continuous priors??? d ??? P (z d ),??? c ??? P (z c ) and uses the composition of these to condition the decoding stepx ??? P ?? (x|??? d ,??? c ).

Since the discrete representation??? d is associated with the true generative distribution components P * i (x), uniformly sampling the discrete prior ensures that that the distributions are well represented in the synthetic mix that the student observes.

In general, the capacity of a categorical distribution is less than that of a continuous normal distribution.

To prevent the VAE's encoder from using primarily the continuous representation while disregarding the discrete one we further complement the learning objective by a term maximising the mutual information between the discrete representation and the data I( DISPLAYFORM1

The final learning objective for each of the student models is the maximization of the ELBO from equation FORMULA1 , augmented by the negative of the cross-model consistency term introduced in section 4.1 and the mutual information term proposed in section 4.2.

DISPLAYFORM0 We sample the training instances x from x ??? P (??)P (x|??), ?? ??? Ber(??) as described in section 4.

Thus they can either be generated from the teacher model (?? = 0) or come from the training set of the currently active distribution (?? = 1).

1(.) is the indicator function which evaluates to 1 if its argument is true and zero otherwise; it makes sure that the consistency regularizer is applied only over the synthetic samples generated by the teacher.

The ?? hyper-parameter controls the importance of the mutual information regularizer.

We present the analytical evaluation of the consistency regularizer in appendix section 7.0.1.

We conducted a set of experiments to explore the behaviour and properties of the method we propose.

We specifically concentrate on the benefits our model brings in the lifelong learning setting which is the main motivation of our work.

We explain the settings of the individual experiments and their focus in the following three sections.

In all the experiments we use the notion of a distributional 'interval': the interval in which we observe samples from a single distribution P * i (x) before the transition to the next distribution P * i+1 (x) occurs.

The length of the intervals is in principle random and we developed a heuristic to generate these.

We provide further details on this together with other technical details related to the network implementation and training common for all the experiments in the appendix.

In this experiment, we seek to establish the performance benefit that our augmented objective formulation in section 4.3 brings into the learning in contrast to the simple ELBO objective 2.

We do so by training two models with identical student-teacher architectures as introduced in section 4, with one using the consistency and mutual information augmented objective (with consistency) and the other using the standard ELBO objective (without consistency).

We also demonstrate the ability of our model to disambiguate distributional boundaries from the distributional variations.

We use Fashion MNIST Xiao et al. (2017) 2 to simulate our sequential learning setting.

We treat each object as a different distribution and present the model with samples drawn from a single distribution at a time.

We sequentially progress over the ten available distributions.

When a distribution transition occurs (new object) we signal the model, make the latest student the new teacher and instantiate a new student model.

We quantify the performance of the generative models by computing the ELBO over the standard Fashion MNIST test set after every distributional transition.

The test set contains objects from all of the individual distributions.

We run this procedure ten times and report the average test ELBO over the ten repetitions in FIG3 .

We see that around the 3rd interval (the 3rd distributional 1 A similar idea is leveraged in InfoGAN Chen et al. (2016) .

2 We do a similar experiment over MNIST in the appendix transition), the negative ELBO of the with consistency model is systematically below (??? 20 nats ) that of the without consistency model.

This confirms the benefits of our new objective formulation for reducing the effects of the catastrophic interference, a crucial property in our lifelong learning setting.

In the same figure we also plot the ELBO of the baseline batch VAE.

The batch VAE will always outperform our model because it has simultaneous access to all of the distributions during training.

After observing and training over all ten distributions we generate samples from the final students of the two models.

We do this by fixing the discrete distribution z d to one-hot vectors over the whole categorical distribution, while randomly sampling the continuous prior z c ??? N (0, I).

We contrast samples generated from the model with consistency ( FIG3 ) to the model without consistency ( FIG3 ).

Our model learns to separate 'style' from the distributional boundaries.

For example, in the last row of our with consistency model, we observe the various styles of shoes.

The without consistency model mixes the distributions randomly.

This illustrates the benefits that our augmented objective has for achieving consistent sampling from the individual distributional components.

In this experiment we dig deeper into the benefits our objective formulation brings for the lifelong learning setting.

We expose the models to a much larger number of distributions and we explore how our augmented objective from 4.3 helps in preserving the previously learned knowledge.

As in section 5.1, we compare models with and without consistency with identical teacher-student architectures.

We measure the ability of the models to recall the previously learned information by looking at the consistency between the posterior of the student and the teacher models over the test data set consistency: #{k : DISPLAYFORM0 We use the MNIST dataset in which we rotate each of the original digit samples by angles ?? = [30 DISPLAYFORM1 We treat each rotation of a single digit family as an individual distribution {P * DISPLAYFORM2 .

Within each distributional interval, we sample the data by first sampling (uniformly with replacement) one of the 70 distributions and then sampling the data instances x from the selected distribution.

FIG4 (b) compares the consistency results of the two tested models throughout the learning process.

Our model with the augmented objective clearly outperforms the model that uses the simple ELBO objective.

This confirms the usefulness of the additional terms in our objective for preserving the previously learned knowledge in accordance with the lifelong learning paradigms.

In addition, similarly as in experiment 5.1, figure 4(a) documents that the model with the augmented objective (thanks to reducing the effects of the catastrophic interference) achieves lower negative test ELBO systematically over the much longer course of learning (??? 30 nats).We also visualise in figure 4(c) how the accumulation of knowledge speeds up the learning process.

For each distributional interval we plot the norms of the model gradients across the learning iterations.

We observe that for later distributional intervals the curves become steeper much quicker, reducing the gradients and reaching (lower) steady states much faster then in the early learning stages.

This suggests that the latter models are able to learn quicker in our proposed architecture.

In this experiment we explore the ability of our model to retain and transfer knowledge across completely different datasets.

We use MNIST and SVHN Netzer et al. (2011) to demonstrate this.

We treat all samples from SVHN as being generated by one distribution P * 1 (x) and all the MNIST 3 samples as generated by another distribution P * 2 (x) (irrespective of the specific digit).

We first train a student model (standard VAE) over the entire SVHN data set.

Once done, we freeze the parameters of the encoder and the decoder and transfer the model into the teacher state (?? ??? ??, ?? ??? ??).

We then use this teacher to aid the learning of the new student over the mix of the teacher-generated synthetic SVHN samplesx and the true MNIST data.

We use the final student model to reconstruct samples from the two datasets by passing them through the learned encoding/decoding flow: DISPLAYFORM0 We visualise examples of the true inputs x and the respective reconstructionsx in FIG5 .

We see that even though the only true data the final model received for training were from MNIST, it can still reconstruct SVHN data.

This confirms the ability of our architecture to transition between complex distributions while still preserving the knowledge learned from the previously observed distributions.

Finally, in figure 5(b) and 5(c) we illustrate the data generated from an interpolation of a 2-dimensional continuous latent space.

For this we specifically trained the models with the continuous latent variable z c ??? R 2 .

To generate the data, we fix the discrete categorical z d to one of the possible values {[0, 1], [1, 0]} and linearly interpolate the continuous z c over the range [???3, 3] .

We then decode these to obtain the samplesx ??? P ?? (x|z d , z c ) .

The model learns a common continuous structure for the two distributions which can be followed by observing the development in the generated samples from top left to bottom right on both figure 5(b) and 5(c).

In this work we propose a novel method for learning generative models over streaming data following the lifelong learning principles.

The principal assumption for the data is that they are generated by multiple distributions and presented to the learner in a sequential manner (a set of observations from a single distribution followed by a distributional transition).

A key limitation for the learning is that the method can only access data generated by the current distribution and has no access to any of the data generated by any of the previous distributions.

The proposed method is based on a dual student-teacher architecture where the teacher's role is to preserve the past knowledge and aid the student in future learning.

We argue for and augment the standard VAE's ELBO objective by terms helping the teacher-student knowledge transfer.

We demonstrate on a series of experiments the benefits this augmented objective brings in the lifelong learning settings by supporting the retention of previously learned knowledge (models) and limiting the usual effects of catastrophic interference.

In our future work we will explore the possibilities to extend our architecture to GAN-like BID8 learning with the prospect to further improve the generative abilities of our method.

GANs, however, do not use a metric for measuring the quality of the learned distributions such as the marginal likelihood or the ELBO in their objective and therefore the transfer of our architecture to these is not straightforward.

Han Xiao, Kashif Rasul, and Roland Vollgraf.

Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.

The analytical derivations of the consistency regularizer show that the regularizer can be interpreted as an a transformation of the standard VAE regularizer.

In the case of an isotropic gaussian posterior, the proposed regularizer scales the mean and variance of the student posterior by the variance of the teacher 7.0.2 and adds an extra 'volume' term.

This interpretation of the consistency regularizer shows that the proposed regularizer preserves the same learning objective as that of the standard VAE.

Below we present the analytical form of the consistency regularizer with categorical and isotropic gaussian posteriors:Corollary 7.0.1 We parameterize the learnt posterior of the teacher by DISPLAYFORM0 and the posterior of the student by DISPLAYFORM1 .

We also redefine the normalizing constants as DISPLAYFORM2 for the teacher and student models respectively.

The reverse KL divergence in equation 8 can now be re-written as: DISPLAYFORM3 where H( ) is the entropy operator and H( , ) is the cross-entropy operator.

Corollary 7.0.2 We assume the learnt posterior of the teacher is parameterized by a centered, isotropic gaussian with DISPLAYFORM4 DISPLAYFORM5 Via a reparameterization of the student's parameters: DISPLAYFORM6 It is also interesting to note that our posterior regularizer becomes the prior if: DISPLAYFORM7

Variational inference BID10 side-steps the intractability of the posterior distribution by approximating it with a tractable distribution Q ?? (z|x); we then optimize the parameters ?? in order to bring this distribution close to P ?? (z|x).

The form of this approximate distribution is fixed and is generally conjugate to the prior P (z).

Variational inference converts the problem of posterior inference into an optimization problem over ??. This allows us to utilize stochastic gradient descent to solve our problem.

To be more concrete, variational inference tries to minimize the reverse Kullback-Leibler (KL) divergence between the variational posterior distribution Q ?? (z|x) and the true posterior P ?? (z|x): DISPLAYFORM0 Rearranging the terms in equation 8 and utilizing the fact that the KL divergence is a measure, we can derive the evidence lower bound L ?? (ELBO) which is the objective function we directly optimize: DISPLAYFORM1 In order to backpropagate it is necessary to remove the dependence on the stochastic variable z. To achieve this, we push the sampling operation outside of the computational graph for the normal distribution via the reparameterization trick BID17 and the gumbel-softmax reparameterization BID22 BID12 for the discrete distribution.

In essence the reparameterization trick allows us to introduce a distribution P ( ) that is not a function of the data or computational graph in order to move the gradient operator into the expectation: DISPLAYFORM2

In this section we provide extra details of our model architecture.

We utilized two different architectures for our experiments.

The first two utilize a standard deep neural network with two layers of 512 to map to the latent representation and two layers of 512 to map back to the reconstruction for the decoder.

We used batch norm BID11 and ELU activations for all the layers barring the layer projecting into the latent representation and the output layer.

The final experiment with the transfer from SVHN to MNIST utilizes a fully convolutional architecture with only strided convolutional layers in the encoder (where the number of filters are doubled at each layer).

The final projection layer for the encoder maps the data to a [C=|z d |, 1, 1] output which is then reparameterized in the standard way.

The decoder utilizes fractional strides for the convolutional-transpose (de-convolution) layers where we reduce the number of filters in half at each layer.

The full architecture can be examined in our code repository [which will be de-anonymized after the review process].

All layers used batch norm BID11 and ELU activations.

We utilized Adam BID16 to optimize all of our problems with a learning rate of 1e-4.

When we utilized weight transfer we re-initialized the accumulated momentum vector of Adam as well as the aggregated mean and covariance of the Batch Norm layers.

Our code is already available online under an MIT license at 4

Since we model our latent variable as a combination of a discrete and a continuous distribution we also use the Gumbel-Softmax reparameterization BID22 BID12 .

The Gumbel-Softmax reparameterization over logits [linear output of the last layer in the encoder] p ??? R M and an annealed temperature parameter ?? ??? R is defined as: DISPLAYFORM0 u ??? R M , g ??? R M .

As the temperature parameter ?? ??? 0, z converges to a categorical.

Multilayer neural networks with sigmoidal activations have a VC dimension bounded between O(?? 2 )Sontag (1998) and O(?? 4 ) BID14 where ?? are the number of parameters.

A model that is able to consistently add new information should also be able to expand its VC dimension by adding new parameters over time.

Our formulation imposes no restrictions on the model architecture: i.e. new layers can be added freely to the new student model.

In addition we also allow the dimensionality of z d ??? R J , our discrete latent representation to grow in order to accommodate new distributions.

This is possible because the KL divergence between two categorical distributions of different sizes can be evaluated by simply zero padding the teacher's smaller discrete distribution.

Since we also transfer weights between the teacher and the student model, we need to handle the case of expanding latent representations appropriately.

In the event that we add a new distribution we copy all the weights besides the ones immediately surrounding the projection into and out of the latent distribution.

These surrounding weights are reinitialized to their standard Glorot initializations BID7 .

In our setting we have the ability to utilize the zero forcing (reverse or mode-seeking) KL or the zero avoiding (forward) KL divergence.

In general, if the true underlying posterior is multi-modal, it is preferable to operate with the reverse KL divergence (Murphy (2012) 21.2.2).

In addition, utilizing the mode-seeking KL divergence generates more realistic results when operating over image data.

In order to validate this, we repeat the experiment in 5.1.

We train two models: one with the forward KL posterior regularizer and one with the reverse.

We evaluate the -ELBO mean and variance over ten trials.

Empirically, we observed no difference between the different measures.

This is demonstrated in figure 6.

Our method derives its sample complexity from standard VAEs.

In practice we evaluate the number of required real and synthetic samples by utilizing early stopping.

When the negative ELBO on the validation set stops decreasing for 50 steps we stop training the current model and transition to the next distribution interval.

Using this and the fact that we keep equal proportions of all observed distributions in our minibatch, we can evaluate the number of synthetic and real samples used during the single distribution interval.

We demonstrate this procedure on experiment 5.1 in figure 7.

We observe a rapid decrease of the number of required real samples as we assimilate more distributions into our model.

In this section we provide an extra experiment run on MNIST as well as some extra images from the rotated MNIST experiment.

In this experiment, we seek to establish the performance benefit that the consistency regularizer brings into the learning process.

We do so by evaluating the ELBO for a model with and without the consistency and mutual information regularizers.

We also demonstrate the ability of the regularizers to disambiguate distributional boundaries and their inter-distributional variations.

I.e. for MNIST this separates the MNIST digits from their inter-class variants (i.e drawing style).We use MNIST to simulate our sequential learning setting.

We treat each digit as a different distribution and present the model with samples drawn from a single distribution at a time.

For the purpose of this experiment we sequentially progress over the ten distributions (i.e. interval sampling involves linearly iterating over all the distributions ).When an interval transition occurs we signal the model, make the student the new teacher and instantiate a new student model.

We contrast this to a model that utilizes the same graphical model, without our consistency and mutual information regularizers.

We quantify the performance of the generative models by computing the ELBO over the standard MNIST test set at every interval.

The test set contains digits from all of the individual distributions.

We run this procedure ten times and report the average ELBO over the test set.

After observing all ten distributions we evaluate samples generated from the final student model.

We do this by fixing the discrete distribution z d , while randomly sampling z c ??? N (0, I).

We contrast samples generated from the model with both regularizers (left-most image in 8) to the model without the regularizers (center image in 8).

Our model learns to separate 'style' from distributional boundaries.

This is demonstrated by observing the digit '2': i.e. different samples of z c produce different styles of writing a '2'.

We provide a larger sized image for the ELBO from experiment 5.2.

We also visualize reconstructions from the rotated MNIST problem (visualized in FIG0 ).

Finally in FIG0 we show the effects on the reconstructions when we do not use the mutual information regularizer.

We believe this is due to the fact that the network utilizes the larger continuous representation to model the discriminative aspects of the observed distribution.

@highlight

Lifelong distributional learning through a student-teacher architecture coupled with a cross model posterior regularizer.