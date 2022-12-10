The maximum mean discrepancy (MMD) between two probability measures P and Q is a metric that is zero if and only if all moments of the two measures are equal, making it an appealing statistic for two-sample tests.

Given i.i.d.

samples from P and Q, Gretton et al. (2012) show that we can construct an unbiased estimator for the square of the MMD between the two distributions.

If P is a distribution of interest and Q is the distribution implied by a generative neural network with stochastic inputs, we can use this estimator to train our neural network.

However, in practice we do not always have i.i.d.

samples from our target of interest.

Data sets often exhibit biases—for example, under-representation of certain demographics—and if we ignore this fact our machine learning algorithms will propagate these biases.

Alternatively, it may be useful to assume our data has been gathered via a biased sample selection mechanism in order to manipulate properties of the estimating distribution Q. In this paper, we construct an estimator for the MMD between P and Q when we only have access to P via some biased sample selection mechanism, and suggest methods for estimating this sample selection mechanism when it is not already known.

We show that this estimator can be used to train generative neural networks on a biased data sample, to give a simulator that reverses the effect of that bias.

Neural networks with stochastic input layers can be trained to approximately sample from an arbitrary probability distribution P based on samples from P BID7 .

Generating simulations from complex distributions has applications in a large number of fields: We can automatically generate illustrations for text BID21 or streams of video BID20 ; we can simulate novel molecular fingerprints to aid scientific exploration BID11 ; and, we can synthesize medical time-series data that can be shared without violating patient privacy BID6 .In this paper, we consider the setting of a feedforward neural network (referred to as the generator) that maps random noise inputs z ∈ R d to some observation space X .

The weights of the neural network are trained to minimize some loss function between the resulting simulations and exemplars of real data.

The general form of the resulting distribution Q over simulations G(z) is determined by the architecture of the generator-which governs the form of the mapping G-and by the loss function used to train the generator.

Generative adversarial networks BID7 use dynamically varying, adversarially learned loss functions specified in terms of the output of a classifier.

Other generative networks use a loss function defined using a distributional distance or divergence between the simulation distribution Q and a target distribution P BID0 BID15 BID22 , requiring the generator to mimic the variance in a collection of data points rather than simply converge to a single mode.

In particular, the maximum mean discrepancy BID8 has demonstrated good performance as a loss function in this setting BID18 BID19 , since it reduces to zero if and only if all moments of two distributions are equal, requiring the generator to reproduce the full range of variation found in the data.

These approaches, like most machine learning methods, assume our data is a representative sample from the distribution of interest.

If this assumption is correct, minimizing the distributional distance between the simulations and the data is equivalent to learning a distribution that is indistinguishable under an appropriate two-sample test from our target distribution.

However, we run into problems if our data is not in fact a representative sample from our target distribution-for instance, if our data gathering mechanism is susceptible to sample selection bias.

The problem of machine learning algorithms replicating and even magnifying human biases is gathering increasing awareness BID2 BID23 , and if we believe our dataset suffers from such biases-for example, if our audio dataset contains primarily male speakers or our image dataset contains primarily white faces -we will typically want to take steps to correct this bias.

Even if our data is representative of the underlying distribution, we might want to generate samples from a modified version of this distribution.

For example, we might want to alter the demographics of characters in a scene to fit a story-line.

In this setting, we can treat our desired modified distribution as our target distribution, and treat our data as if they were sampled from this distribution subject to an appropriately biased sample selection mechanism.

If we know the form of our sample selection bias, we can reformulate our loss function to penalize the generator based on the difference between simulated data and the unbiased distribution of interest, which we will refer to as our target distribution.

After a review of relevant background information in Section 2, we show in Section 3 that, given a function that describes how our observed data deviates from this target distribution, we can construct an estimator of the MMD between the generator and the target distribution.

In practice, we will not know the function linking the target distribution and the empirical data distribution.

However, we can approximate this function based on user-provided examples of data points that are over-and under-represented.

In Section 4, we discuss ways to estimate this function, and in Section 5 we discuss related work in survey sampling statistics and bias reduction.

We demonstrate the efficacy and applicability of our approach in Section 6.

Generative networks are a class of models which take a collection of data lying in some observation space X as input, and aim to generate simulations that are similar to that data -or more generally, whose empirical distribution is close to the distribution underlying the data.

These models do not use or require an explicit probability distribution for the data, but instead rely on the fact that sufficiently complex neural networks have the capacity to approximate any arbitrary function BID9 .

We can therefore construct a method of simulating from an arbitrarily complex probability distribution Q on X by using a neural network generator G : R d → X to transform random ddimensional inputs z into simulations G(z) ∈ X .

In order to minimize the difference between the probability distribution Q over simulations G(z) and the target distribution P, we train the neural network to minimize a loss function between simulations G(z) ∼ Q and data x ∼ P The most common forms of generative network are generative adversarial networks (GANs, BID7 , so called because the loss function is dynamically defined in terms of the output of an adversarially learned classifier.

This classifier-itself a neural network-is trained to differentiate between two classes, data and simulations, and for a given observation returns a score under the two classes (loosely corresponding to a probability of belonging to that class).

The generator's loss function is a function of the score assigned to simulations under the true data class, so that reducing the loss function leads to an increased chance of fooling the classifier.

While this adversarial framework has the advantage of a dynamically evolving and increasingly discriminative loss function, a disadvantage is that the generator can successfully minimize this loss function by mimicking only a subset of the data, leading to a phenomenon known as mode collapse BID17 BID4 .

To avoid this, recent works have incorporated estimators of distributional distances between P and Q, that consider the overall distributions rather than just element-wise distances between samples.

For example, the maximum mean discrepancy BID8 BID18 BID19 between two dis-tributions, which we explore further in Section 2.2, reduces to zero if and only if all moments of the two distributions are the same.

Other distributional distances that have been used include the Wasserstein distance and the Cramer distance BID0 BID1 .

In some cases, these distance-based generative networks include adversarial components in their loss functions; for example the MMD-GAN of adversarially learns parameters of the maximum mean discrepancy metric.

The maximum mean discrepancy (MMD, BID8 projects two distributions P and Q into a reproducing kernel Hilbert space (RKHS) H, and looks at the maximum mean distance between the two projections, i.e. DISPLAYFORM0 If we specify the kernel mean embedding µ P of P as µ P = k(x, ·)dP(x), where k(·, ·) is the characteristic kernel defining the RKHS, then we can write the square of this distance as DISPLAYFORM1 Since we have projected the distributions into an infinite-dimensional space, the distance between the two distributions is zero if and only if all their moments are the same.

In order to be a useful loss function for training a neural network, we must be able to estimate the MMD from data, and also take derivatives of this estimate with respect to the network parameters.

We can construct an unbiased estimator of square of the MMD BID8 using m samples x i ∼ P and n samples y i ∼ Q as DISPLAYFORM2 If P is the distribution underlying our data and Q is an approximating distribution represented using a neural network, we can differentiate the individual kernel terms in Equation 1 with respect to the simulated data y i , and hence (via the chain rule) with respect to the parameters of the neural networks.

The MMD has been successfully used as a loss function in several generative adversarial networks.

BID5 and BID13 propose training a feedforward neural network to minimize the MMD between simulations and data; BID13 also propose minimizing the MMD between simulations and data that has been encoded using a pre-trained autoencoder.

The generator in the MMD-GAN of also aims to reduce the MMD between simulations and data, but learns the characteristic kernel in an adversarial manner by combining the kernel with a dynamically learned autoencoder.

If we have unbiased samples from two distributions P and Q, the estimator described in Equation FORMULA2 gives an unbiased estimate of the MMD between those two distributions.

In a generative network context, we can therefore use this estimator as a loss function in order to modify the generator associated with Q so that the MMD between the two distributions is minimized.

However, this relies on having access to unbiased samples from our target distribution P. In practice, our data may have been gathered using biased sampling practices: A dataset of images of faces may over-represent white faces; datasets generated from medical experiments may over-represent male patients; datasets generated from on-campus studies may over-represent college-aged students.

If our data is a biased sample of our target distribution, this estimator will estimate the difference between our simulations and the biased empirical distribution, so our simulations will recreate the biases therein.

In this section, we propose an estimator for the MMD between two distributions P and Q when we have access to P only via some biased sample selection mechanism.

Concretely, we assume that P is our target of interest, but our observed data are actually sampled from a related distribution T (x)P(x).

We can think of T (x) as an appropriately scaled "thinning function" T (x) = ZT (x) that characterizes the sample selection mechanism.

In other words, we assume that candidate samples x * are sampled from P, and these candidates are selected into our pool with probability T (x * );the normalizing constant Z ensures that T (x)P(x) = DISPLAYFORM0 is a valid probability distribution.

While in the remainder of this paper we will continue to use the language of biased sample selection mechanisms, we note that this framework can also be used if our data are unbiased samples but we want to explicitly modulate our target distribution via some function F (x) so that we generate simulations from F (x)P(x); in this setting we can treat the transformed target as P and let T (x) = 1/F (x).For now, we assume that our thinning function T (x) is known; we discuss ways to estimate or specify it in Section 4.

Our estimation problem becomes an importance sampling problem: We have samples from T (·)P(·), and we want to use these to estimate MMD 2 [P, Q], which is a function of the target distribution P. Importance sampling provides a method for constructing an estimator for the expectation of a function φ(x) with respect to a distribution P, by taking an appropriately weighted sum of evaluations of φ at values sampled from a different distribution P .

If we knew the normalizing constant Z, we could construct an unbiased estimator of the MMD between P and Q by weighting each function evaluation associated with sample x from T (x)P(x) with the likelihood ratio P(x)/T (x)P(x) = 1/T (x), i.e. DISPLAYFORM1 However, the normalizing constant Z depends on both T and P. We will not, in general, know an analytic form for P, so we will not be able to calculate Z. Since we will only be able to evaluate T (x) up to a normalizing constant, we cannot work with Equation 2 directly.

Instead, we can construct a biased estimator M d by using self-normalized importance weights, DISPLAYFORM2 where w( DISPLAYFORM3 .We refer to the estimator M b in Equation 3 as the weighted MMD estimator.

While this estimator is biased due to the self-normalized weights, this bias will decrease as 1/m where m is the number of samples from T (·)P(·).

Further, this biased estimator will often have lower variance than the unbiased estimator in Equation 2 BID16 .

In Section 3, we assumed that our target distribution P is only accessible via a biased sample selection mechanism characterized by a thinning function T (x), so that our samples are actually distributed (up to a normalizing constant) according to T (x)P(x).

If we know the thinning function T (x) that describes our sample selection mechanism, we can use Equation 3 directly in the loss function of our generator.

However, in practice we will not have access to this thinning function.

Rather, we are likely to be in a situation where a practitioner has noticed that one or more classes is over-or under-represented, either in the dataset or in simulations from a generative network trained on the data.

If our dataset is fully labeled, we could either manually re-balance our dataset to match the target distribution, or use these labels to estimateT by comparing the number of data points in various subsets of X with the expected number under our target distribution.

Even if we do not have a fully labeled dataset, we might be able to label a subset of examples and use these to estimate T (x).

For example, assume we have an image dataset with more pictures of men than women, and that we select and label a random subset of this dataset.

A reasonable estimate for our thinning function would be to set T (men) = 1 and approximate T (women) by the sample ratio of men to women in our labeled subset.

In this simple two-class setting, to extrapolate values of T across our observation space X we can run a logistic regression using the labeled subset.

In a more complicated problem, we may need to deploy more sophisticated regression tools, but the problem remains one of function estimation from labeled data.

Given a set of labeled exemplars and user-specified estimates of the thinning function evaluated at those exemplars (which could be based on demographic statistics or domain knowledge, or desired statistics of the simulation distribution), we could learn a thinning function T (x) using techniques such as neural network function estimation or Gaussian process regression.

While it has received little attention in the deep learning literature, the problem of correcting for biased sample selection mechanisms is familiar in the field of survey statistics.

Our estimator is related to inverse probability weighting (IPW), originally proposed by BID10 and still in wide use today (e.g. BID14 .

IPW is used in stratified random sampling study designs to correct parameter estimates for intentional bias in sampling.

Under IPW, each data point is assigned a weight proportionate to the inverse of its selection probability, so that samples from classes which are disproportionately unlikely to be seen in the sample are assigned increased weights to counteract the undersampling of their classes.

This is the same form as the weights in our sampling scheme, and serves a similar purpose, although it is not placed in the context of the MMD between two distributions and assumes that the probability of selection is known for each observation.

This work also follows increased awareness of the effects of biased data on machine learning outcomes, and interest in mitigating these effects.

For example, BID3 and BID2 explore how biases and prejudices expressed in our language manifest in the word embeddings and representations found.

BID23 find that popular image labeling datasets exhibit gender stereotypes and that algorithms trained on these datasets tend to amplify these stereotypes.

We consider two experimental settings: One where the form of the sampling bias is known, and one where it is estimated from data.

In Section 3, we discussed how, for a given thinning function T (x), we can construct an estimator for the MMD between the distribution Q implied by our generator and the underlying unbiased data distribution P. To demonstrate the efficacy of this in practice, we assume P to be a mixture of two Gaussians (Figure 1a) , and let T = 0.5 1+exp(10(x−1)) + 0.5 be a scaled logistic function as shown in Figure 1b .

The resulting data is distributed according to Figure 1c .We construct a simple generator network taking univariate random noise as input, and consisting of six fully-connected layers, each with three nodes and exponential linear unit activations, and with a univariate output with no activation.

We train this network using both the standard unbiased MMD DISPLAYFORM0 (d) Simulations using standard MMD estimator (with PDF of T (x)P(x) for comparison) (e) Simulations using weighted MMD estimator (with PDF of P(x) for comparison) Figure 1 : Output of GANs trained on data sampled according to T (x)P(x).

Simulation plots PDFs are computed using NumPy's random normal sampler, mixed to create the appropriate mode ratios.estimator of Equation 1, and the weighted MMD estimator proposed in Equation 3, using ADAM optimization with a learning rate of 0.001.

Figures 1e and 1d show histograms of samples generated using the two estimators.

We see that, as expected, the standard MMD estimator does a good job of replicating the empirical distribution of the data presented to it.

The weighted MMD estimator, however, is able to replicate the target distribution P, even though it only has access to samples from T (x)P(x).

In practice we are unlikely to know a functional form for T (x), the function that describes the form of the sampling bias.

Here we consider the case where we must estimate this function from data.

We consider as an example a dataset containing zeros and ones from the MNIST dataset.

We assume our target distribution contains 50% zeros and 50% ones, but that due to a biased selection procedure our data contains 80% zeros and 20% ones.

We hypothesise that the practitioner using this dataset is aware that there is a discrepancy, but does not know how to translate this into a functional form for T .

Instead, the practitioner labels examples of each class, and supplies an estimate of how much each class is under-represented.

In our example, we assume the practitioner has labeled 800 zeros and 200 ones, and estimates that ones are undersampled according to a thinning probability of 0.75.

We note that this number can easily be estimated from a data sample if the practitioner knows the global ratio of zeros and ones.

We use an architecture based on the MMD-GAN of , which incorporates an autoencoder, and trains a generator based on the MMD between the encoded representations of the data and of the simulations.

The autoencoder is simultaneously learned in an adversarial manner, and serves to optimize the kernel used in MMD to maximally discriminate between the two distributions.

Our experiments are run on an adapted version of the original MMD-GAN Torch code, which will be made available after publication.

Since the autoencoder provides a low-dimensional embedding of our images, we specify our thinning function T (x) on the space spanned by the encoder.

After each update of the autoencoder, we estimate T with the labeled set, using appropriately scaled logistic regression.

We then replace the standard MMD estimator in the loss function with our weighted estimator, calculating using the estimated T .One could imagine using the weighted estimator in two scenarios.

We might be already aware of the bias in our data, in which case we would simply train our GAN using the weighted MMD estimator.

Alternatively, we might only become aware of the bias after already training a GAN using the standard estimator, in which case we might initialize our network weights to the resulting pretrained values.

To simulate these scenarios, we trained two networks using the weighted estimator: one initialized to a pre-trained MMD-GAN, and one randomly initialized.

FIG1 shows a random subset of the data used to train the GANs.

Due to the biased sample selection method, there are significantly more zeros than ones.

FIG1 shows simulations generated by minimizing the standard MMD estimator using this biased data; as expected, it reflects the greater proportion of zeros compared to ones.

By contrast, the simulations trained by minimizing the weighted MMD estimator, which estimates the MMD from the underlying target distribution as estimated by our thinning function, shows a marked increase in the number of ones, without any obvious difference in simulation quality.

All networks were trained for 10,000 iterations.

To further quantify these results, we used the (unscaled) logistic regression used to specify the thinning function to classify the simulated images as zeros or ones.

TAB0 shows the proportion of ones obtained using the three scenarios (standard MMD estimator, weighted MMD estimator initialized with a pre-trained network, and weighted MMD estimator with random initialization).

We see that, as expected, the network trained using the standard MMD estimator produces around 20% ones, reflecting the proportion in the dataset.

The networks trained using the weighted MMD estimator achieve a significanly higher percent of ones, with the randomly initialized network performing better than the pre-trained network.

While both the networks trained using the weighted MMD estimator simulate a higher proportion of ones than is present in our dataset, neither reaches the 50:50 ratio in our assumed target distribution.

We believe this is due to inability of the network to fully converge to a solution that gives zero loss.

This interpretation is supported by the fact that the randomly initialized network converges to a better solution than the pre-initialized network, despite the architecture and objective function being the same in both cases.

To visualize the changes introduced by using the weighted MMD estimator, in FIG2 we plot the proportion of ones on a network that is trained for 10,000 iterations using the standard MMD estimator, then for 10,000 iterations using the weighted MMD estimator.

We see that the proportion quickly rises after the loss function is altered.

We have presented an asymptotically unbiased estimator for the MMD between two distributions P and Q, for use when we only have access to P via a biased sampling mechanism.

This mechanism can be specified by a known or estimated thinning function T (x), where samples are then assumed to come from a distribution T (x)P(x)/Z. We show that this estimator can be used to manipulate the distribution of simulations learned by a generative network, in order to correct for sampling bias or to explicitly change the distribution according to a user-specified function.

When the thinning function is unknown, it can be estimated from labeled data.

We demonstrate this in an interpretable experiment using partially labeled images, where we jointly estimate the thinning function alongside the generator weights.

An obvious next step is to explore the use of more sophisticated thinning functions appropriate for complex, multimodal settings.

@highlight

We propose an estimator for the maximum mean discrepancy, appropriate when a target distribution is only accessible via a biased sample selection procedure, and show that it can be used in a generative network to correct for this bias.

@highlight

Proposes an importance-weighted estimator of the MMD to estimate the MMD between distributions based on samples biased according to a known or estimated unknown scheme.

@highlight

The authors address the problem of sample selection bias in MMD-GANs and propose an estimate of the MMD between two distributions using weighted maximum mean discrepancy.

@highlight

This paper presents a modification of the objective used to train generative networks with an MMD adversary 