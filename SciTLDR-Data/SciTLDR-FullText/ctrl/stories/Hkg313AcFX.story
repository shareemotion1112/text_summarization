In this paper we propose to view the acceptance rate of the Metropolis-Hastings algorithm as a universal objective for learning to sample from target distribution -- given either as a set of samples or in the form of unnormalized density.

This point of view unifies the goals of such approaches as Markov Chain Monte Carlo (MCMC), Generative Adversarial Networks (GANs), variational inference.

To reveal the connection we derive the lower bound on the acceptance rate and treat it as the objective for learning explicit and implicit samplers.

The form of the lower bound allows for doubly stochastic gradient optimization in case the target distribution factorizes (i.e. over data points).

We empirically validate our approach on Bayesian inference for neural networks and generative models for images.

Bayesian framework and deep learning have become more and more interrelated during recent years.

Recently Bayesian deep neural networks were used for estimating uncertainty BID6 , ensembling BID6 and model compression BID20 .

On the other hand, deep neural networks may be used to improve approximate inference in Bayesian models BID13 .Learning modern Bayesian neural networks requires inference in the spaces with dimension up to several million by conditioning the weights of DNN on hundreds of thousands of objects.

For such applications, one has to perform the approximate inference -predominantly by either sampling from the posterior with Markov Chain Monte Carlo (MCMC) methods or approximating the posterior with variational inference (VI) methods.

MCMC methods provide the unbiased (in the limit) estimate but require careful hyperparameter tuning especially for big datasets and high dimensional problems.

The large dataset problem has been addressed for different MCMC algorithms: stochastic gradient Langevin dynamics BID28 , stochastic gradient Hamiltonian Monte Carlo , minibatch MetropolisHastings algorithms BID15 BID1 .

One way to address the problem of high dimension is the design of a proposal distribution.

For example, for the Metropolis-Hastings (MH) algorithm there exists a theoretical guideline for scaling the variance of a Gaussian proposal BID24 BID25 .

More complex proposal designs include adaptive updates of the proposal distribution during iterations of the MH algorithm BID12 BID7 .

Another way to adapt the MH algorithm for high dimensions is combination of adaptive direction sampling and the multiple-try Metropolis algorithm as proposed in BID17 .

Thorough overview of different extensions of the MH algorithm is presented in BID18 .Variational inference is extremely scalable but provides a biased estimate of the target distribution.

Using the doubly stochastic procedure BID27 BID11 VI can be applied to extremely large datasets and high dimensional spaces, such as a space of neural network weights BID14 BID5 .

The bias introduced by variational approximation can be mitigated by using flexible approximations BID22 and resampling BID9 .Generative Adversarial Networks BID8 ) (GANs) is a different approach to learn samplers.

Under the framework of adversarial training different optimization problems could be solved efficiently BID0 BID21 .

The shared goal of "learning to sample" inspired the connection of GANs with VI BID19 and MCMC BID26 .In this paper, we propose a novel perspective on learning to sample from a target distribution by optimizing parameters of either explicit or implicit probabilistic model.

Our objective is inspired by the view on the acceptance rate of the Metropolis-Hastings algorithm as a quality measure of the sampler.

We derive a lower bound on the acceptance rate and maximize it with respect to parameters of the sampler, treating the sampler as a proposal distribution in the Metropolis-Hastings scheme.

We consider two possible forms of the target distribution: unnormalized density (density-based setting) and a set of samples (sample-based setting).

Each of these settings reveals a unifying property of the proposed perspective and the derived lower bound.

In the density-based setting, the lower bound is the sum of forward and reverse KL-divergences between the true posterior and its approximation, connecting our approach to VI.

In the sample-based setting, the lower bound admit a form of an adversarial game between the sampler and a discriminator, connecting our approach to GANs.

The closest work to ours is of BID26 .

In contrast to their paper our approach (1) is free from hyperparameters; (2) is able to optimize the acceptance rate directly; (3) avoids minimax problem in the density based setting.

Our main contributions are as follows:1.

We introduce a novel perspective on learning to sample from the target distribution by treating the acceptance rate in the Metropolis-Hastings algorithm as a measure of sampler quality.

2.

We derive the lower bound on the acceptance rate allowing for doubly stochastic optimization of the proposal distribution in case when the target distribution factorizes (i.e. over data points).

3.

For sample-based and density-based forms of target distribution we show the connection of the proposed algorithm to variational inference and GANs.

The rest of the paper is organized as follows.

In Section 2 we introduce the lower bound on the AR.

Special forms of target distribution are addressed in Section 3.

We validate our approach on the problems of approximate Bayesian inference in the space of high dimensional neural network weights and generative modeling in the space of images in Section 4.

We discuss results and directions of the future work in Section 5.

In MH algorithm we need to sample from target distribution p(x) while we are only able to sample from proposal distribution q(x | x).

One step of the MH algorithm can be described as follows.

DISPLAYFORM0 If the proposal distribution q(x | x) does not depend on x, i.e. q(x | x) = q(x ), the algorithm is called independent MH algorithm.

The quality of the proposal distribution is measured by acceptance rate and mixing time.

Mixing time defines the speed of convergence of the Markov chain to the stationary distribution.

The acceptance rate of the MH algorithm is defined as DISPLAYFORM1 where DISPLAYFORM2 In case of independent proposal distribution we show that the acceptance rate defines a semimetric in distribution space between p and q (see Appendix A.2).

Although, we can maximize the acceptance rate of the MH algorithm (Eq. 1) directly w.r.t.

parameters φ of the proposal distribution q φ (x | x), we propose to maximize the lower bound on the acceptance rate.

As our experiments show (see Section 4) the optimization of the lower bound compares favorably to the direct optimization of the acceptance rate.

To introduce this lower bound we first express the acceptance rate in terms of total variation distance.

DISPLAYFORM0 where TV is the total variation distance.

The proof of Theorem 1 can be found in Appendix A.1.

This reinterpretation in terms of total variation allows us to lower bound the acceptance rate through the Pinsker's inequality DISPLAYFORM1 The maximization of this lower bound can be equivalently formulated as the following optimization problem DISPLAYFORM2 In the following sections, we show the benefits of this optimization problem in two different settings -when the target distribution is given in a form of unnormalized density and as a set of samples.

In Appendix C.5 and C.1 we provide the empirical evidence that maximization of the proposed lower bound results in the maximization of the acceptance rate.

From now on we consider only optimization problem Eq. 5 but the proposed algorithms can be also used for the direct optimization of the acceptance rate (Eq. 1).To estimate the loss function (Eq. 5) we need to evaluate the density ratio.

In the density-based setting unnormalized density of the target distribution is given, so we suggest to use explicit proposal distribution to compute the density ratio explicitly.

In the sample-based setting, however, we cannot compute the density ratio, so we propose to approximate it via adversarial training BID8 .

The brief summary of constraints for both settings is shown in TAB0 .

DISPLAYFORM0 The following subsections describe the algorithms in detail.

In the density-based setting, we assume the proposal to be an explicit probabilistic model, i.e. the model that we can sample from and evaluate its density at any point up to the normalization constant.

We also assume that the proposal is reparameterisable BID13 BID23 BID4 .During the optimization of the acceptance rate we might face a situation when proposal collapses to the delta-function.

This problem usually arises when we use Markov chain proposal, for example, q φ (x | x) = N (x | x, σ).

For such proposal we can obtain arbitrary high acceptance rate, making the σ small enough.

However, this is not the case for the independent proposal distribution q φ (x | x) = q φ (x ).

In Appendix B.1 we provide more details and intuition on this property of acceptance rate maximization.

We also provide empirical evidence in Section 4 that collapsing to the delta-function does not happen for the independent proposal.

In this paper, we consider two types of explicit proposals: simple parametric family (Section 4.2) and normalizing flows BID22 BID3 ) (Section 4.1).

Rich family of normalizing flows allows to learn expressive proposal and evaluate its density in any point of target distribution space.

Moreover, an invertible model (such as normalizing flow) is a natural choice for the independent proposal due to its ergodicity.

Indeed, choosing the arbitrary point in the target distribution space, we can obtain the corresponding point in the latent space using the inverse function.

Since every point in the latent space has positive density, then every point in the target space also has positive density.

Considering q φ (x ) as the proposal, the objective of optimization problem 5 takes the form DISPLAYFORM0 Explicit form of the proposal q φ (x ) and the target p(x) distributions allows us to obtain density ratios q φ (x)/q φ (x ) and p(x )/p(x) for any points x, x .

But to estimate the loss in Eq. 6 we also need to obtain samples from the target distribution x ∼ p(x) during training.

For this purpose, we use the current proposal q φ and run the independent MH algorithm.

After obtaining samples from the target distribution it is possible to perform optimization step by taking stochastic gradients w.r.t.

φ.

Pseudo-code for the obtained procedure is shown in Algorithm 1.Algorithm 1 Optimization of proposal distribution in density-based case DISPLAYFORM1 approximate loss with finite number of samples DISPLAYFORM2 perform gradient descent step end while return optimal parameters φ Algorithm 1 could also be employed for the direct optimization of the acceptance rate (Eq. 1).

Now we apply this algorithm for Bayesian inference problem and show that during optimization of the lower bound we can use minibatches of data, while it is not the case for direct optimization of the acceptance rate.

We consider Bayesian inference problem for discriminative model on dataset DISPLAYFORM3 , where x i is the feature vector of ith object and y i is its label.

For the discriminative model we know likelihood p(y i | x i , θ) and prior distribution p(θ).

In order to obtain predictions for some object x i , we need to evaluate the predictive distribution DISPLAYFORM4 To obtain samples from posterior distribution p(θ | D) we suggest to learn proposal distribution q φ (θ) and perform independent MH algorithm.

Thus the objective 6 can be rewritten as DISPLAYFORM5 Note that due to the usage of independent proposal, the minimized KL-divergence splits up into the sum of two KL-divergences.

DISPLAYFORM6 Minimization of the first KL-divergence corresponds to the variational inference procedure.

DISPLAYFORM7 The second KL-divergence has the only term that depends on φ.

Thus we obtain the following optimization problem DISPLAYFORM8 The first summand here contains the sum over all objects in dataset D. We follow doubly stochastic variational inference and suggest to perform unbiased estimation of the gradient in problem 11 using only minibatches of data.

Moreover, we can use recently proposed techniques BID15 BID1 that perform the independent MH algorithm using only minibatches of data.

Combination of these two techniques allows us to use only minibatches of data during iterations of algorithm 1.

In the case of the direct optimization of the acceptance rate, straightforward usage of minibatches results in biased gradients.

Indeed, for the direct optimization of the acceptance rate (Eq. 1) we have the product over the all training data inside min function.

In the sample-based setting, we assume the proposal to be an implicit probabilistic model, i.e. the model that we can only sample from.

As in the density-based setting, we assume that we are able to perform the reparameterization trick for the proposal.

In this subsection we consider only Markov chain proposal q φ (x | x), but everything can be applied to independent proposal q φ (x ) by simple substitution q φ (x | x) with q φ (x ).

From now we will assume our proposal distribution to be a neural network that takes x as its input and outputs x .

Considering proposal distribution parameterized by a neural network allows us to easily exclude delta-function from the space of solutions.

We avoid learning the identity mapping by using neural networks with the bottleneck and noisy layers.

For the detailed description of the architectures see Appendix C.8.The set of samples from the true distribution X ∼ p(x) allows for the Monte Carlo estimation of the loss DISPLAYFORM0 To compute the density ratio DISPLAYFORM1 we suggest to use well-known technique of density ratio estimation via training discriminator network.

Denoting discriminator output as D(x, x ), we suggest the following optimization problem for the discriminator.

DISPLAYFORM2 Speaking informally, such discriminator takes two images as input and tries to figure out which image is sampled from true distribution and which one is generated by the one step of proposal distribution.

It is easy to show that optimal discriminator in problem 13 will be DISPLAYFORM3 Note that for optimal discriminator we have D(x, x ) = 1 − D(x , x).

In practice, we have no optimal discriminator and these values can differ significantly.

Thus, we have four ways for density ratio estimation that may differ significantly.

DISPLAYFORM4 To avoid the ambiguity we suggest to use the discriminator of a special structure.

Let D(x, x ) be a convolutional neural network with scalar output.

Then the output of discriminator D(x, x ) is defined as follows.

DISPLAYFORM5 In other words, such discriminator can be described as the following procedure.

For single neural network D(·, ·) we evaluate two outputs D(x, x ) and D(x , x).

Then we take softmax operation for these values.

Summing up all the steps, we obtain algorithm 2.Algorithm 2 Optimization of proposal distribution in sample-based case DISPLAYFORM6 approximate loss with finite number of samples DISPLAYFORM7 perform gradient descent step end for return parameters φ Algorithm 2 could also be employed for direct optimization of the acceptance rate (Eq. 1).

But, in Appendix B.2 we provide an intuition for this setting that the direct optimization of the acceptance rate may struggle from vanishing gradients.

In this section, we provide experiments for both density-based and sample-based settings, showing the proposed procedure is applicable to high dimensional target distributions.

Code for reproducing all of the experiments will be published with the camera-ready version of the paper.

To demonstrate performance of our approach we reproduce the experiment from BID26 .

For target distributions we use synthetic 2d distributions (see Appendix C.3 for densities): ring (a ring-shaped density), mog2 (a mixture of 2 Gaussians), mog6 (a mixture of 6 Gaussians), ring5 (a mixture of 5 distinct rings).

We measure performance of learned samplers using Effective Sample Size (see Appendix C.4 for formulation).

Since the unnormalized densities of target distributions are given, we can learn proposals as suggested in the density-based setting (Section 3.1).To learn the independent proposal we use RealNVP model BID3 ) (see details in Appendix C.2) and compare the performance of proposals after optimization of different objectives: the acceptance rate (AR), our lower bound on the acceptance rate (ARLB), evidence lower bound that corresponds to the variational inference (VI).

We also compare the performance of obtained independent proposals with the performance of Markov chain proposals: A-NICE-MC BID26 , Hamiltonian Monte Carlo (HMC).In Tables 2, 3 we see that our approach has comparable performance with A-NICE-MC BID26 .

However, comparison between A-NICE-MC and learning independent proposal is not the main subject of interest, since A-NICE-MC learns Markov chain proposal.

On the one hand, Markov chain proposal uses more information while generating a new sample, hence can learn more expressive stationary distribution, on the other hand, usage of previous sample increase autocorrelation between samples and reduces ESS.

Thus, the main point of interest is the comparison of two independent proposals: one is learned by maximization of the acceptance rate (or its lower bound), and the second is learned by variational inference procedure, i.e. maximization of evidence lower bound.

In TAB1 we see that both maximization of the acceptance rate and its lower bound outperform variational inference for all target distributions.

Moreover, in FIG0 we show that variational inference fails to cover all the modes of mog6 in contrast to proposals learned via maximization of acceptance rate or its lower bound.

Densities of learned proposals and histograms for all distributions are presented in Appendix C.6.

In density-based setting, we consider Bayesian inference problem for the weights of a neural network.

In our experiments we consider approximation of predictive distribution (Eq. 7) as our main goal.

To estimate the goodness of the approximation we measure negative log-likelihood and accuracy on the test set.

In subsection 3.1 we show that lower bound on acceptance rate can be optimized more efficiently than acceptance rate due to the usage of minibatches.

But other questions arise.1.

Does the proposed objective in Eq. 11 allow for better estimation of predictive distribution compared to the variational inference?2.

Does the application of the MH correction to the learned proposal distribution allow for better estimation of the predictive distribution (Eq. 7) than estimation via raw samples from the proposal?To answer these questions we consider reduced LeNet-5 architecture (see Appendix C.7) for classification task on 20k images from MNIST dataset (for test data we use all of the MNIST test set).

Even after architecture reduction we still face a challenging task of learning a complex distribution in 8550-dimensional space.

For the proposal distribution we use fully-factorized gaussian DISPLAYFORM0 .

For variational inference, we train the model using different initialization and pick the model according to the best ELBO.

For our procedure, we do the same and choose the model by the maximum value of the acceptance rate lower bound.

In Algorithm 1 we propose to sample from the posterior distribution using the independent MH and the current proposal.

It turns out in practice that it is better to use the currently learned proposal q φ (θ) = N (θ | µ, σ) as the initial state for random-walk MH algorithm.

That is, we start with the mean µ as an initial point, and then use random-walk proposal q(θ | θ) = N (θ | θ, σ) with the variances σ of current independent proposal.

This should be considered as a heuristic that improves the approximation of the loss function.

The optimization of the acceptance rate lower bound results in the better estimation of predictive distribution than the variational inference (see FIG1 ).

Optimization of acceptance rate for the same number of epochs results in nearly 30% accuracy on the test set.

That is why we do not report results for this procedure in FIG1 In both procedures we apply the independent MH algorithm to estimate the predictive distribution.

To answer the second question we estimate predictive distribution in two ways.

The first way is to perform 100 accept/reject steps of the independent MH algorithm with the learned proposal q φ (θ) after each epoch, i.e. perform MH correction of the samples from the proposal.

The second way is to take the same number of samples from q φ (θ) without MH correction.

For both estimations of predictive distribution, we evaluate negative log-likelihood on the test set and compare them.

The MH correction of the learned proposal improves the estimation of predictive distribution for the variational inference (right plot of FIG2 ) but does not do so for the optimization of the acceptance rate lower bound (left plot of FIG2 ).

This fact may be considered as an implicit evidence that our procedure learns the proposal distribution with higher acceptance rate.

In the sample-based setting, we estimate density ratio using a discriminator.

Hence we do not use the minibatching property (see subsection 3.1) of the obtained lower bound, and optimization problems for the acceptance rate and for the lower bound have the same efficiency in terms of using data.

That is why our main goal in this setting is to compare the optimization of the acceptance rate and the optimization of the lower bound.

Also, in this setting, we have Markov chain proposal that is interesting to compare with the independent proposal.

Summing up, we formulate the following questions:1.

Does the optimization of the lower bound has any benefits compared to the direct optimization of the acceptance rate?

2.

Do we have mixing issue while learning Markov chain proposal in practice?

3.

Could we improve the visual quality of samples by applying the MH correction to the learned proposal?We use DCGAN architecture for the proposal and discriminator (see Appendix C.8) and apply our algorithm to MNIST dataset.

We consider two optimization problems: direct optimization of the acceptance rate and its lower bound.

We also consider two ways to obtain samples from the approximation of the target distribution -use raw samples from the learned proposal, or perform the MH algorithm, where we use the learned discriminator for density ratio estimation.

In case of the independent proposal, we show that the MH correction at evaluation step allows to improve visual quality of samples -figures 4(a) and 4(b) for the direct optimization of acceptance rate, figures 4(c) and 4(d) for the optimization of its lower bound.

Note that in Algorithm 2 we do not apply the independent MH algorithm during training.

Potentially, one can use the MH algorithm considering any generative model as a proposal distribution and learning a discriminator for density ratio estimation.

Also, for this proposal, we demonstrate the negligible difference in visual quality of samples obtained by the direct optimization of acceptance rate (see Fig. 4(a) ) and by the optimization of the lower bound (see Fig. 4(c) ).

Figure 4 : Samples from the learned independent proposal obtained via optimization: of acceptance rate (4(a), 4(b)) and its lower bound (4(c), 4(d)).

In Fig. 4(b) , 4(d) we show raw samples from the learned proposal.

In Fig. 4(a) , 4(c) we show the samples after applying the independent MH correction to the samples, using the learned discriminator for density ratio estimation.

In the case of the Markov chain proposal, we show that the direct optimization of acceptance rate results in slow mixing (see Fig. 5(a) ) -most of the time the proposal generates samples from one of the modes (digits) and rarely switches to another mode.

When we perform the optimization of the lower bound the proposal switches between modes frequently (see Fig. 5(b) ).

Note that we obtain different distributions of the samples because of conditioning of our proposal.

To show that the learned proposal distribution has the Markov property rather than being totally independent, we show samples from the proposal conditioned on two different points in the dataset (see Fig. 6 ).

The difference in samples from two these distributions ( Fig. 6(a) , 6(a)) reflects the dependence on the conditioning.

Additionally, in Appendix C.9 we present samples from the chain after 10000 accepted images and also samples from the chain that was initialized with noise.

This paper proposes to use the acceptance rate of the MH algorithm as the universal objective for learning to sample from some target distribution.

We also propose the lower bound on the acceptance rate that should be preferred over the direct maximization of the acceptance rate in many cases.

The proposed approach provides many ways of improvement by the combination with techniques from the recent developments in the field of MCMC, GANs, variational inference.

For example• The proposed loss function can be combined with the loss function from BID16 , thus allowing to learn the Markov chain proposal in the density-based setting.• We can use stochastic Hamiltonian Monte Carlo for the loss estimation in Algorithm 1.

• In sample-based setting one can use more advanced techniques of density ratio estimation.

Application of the MH algorithm to improve the quality of generative models also requires exhaustive further exploration and rigorous treatment.

Remind that we have random variables ξ = DISPLAYFORM0 , and want to prove the following equalities.

DISPLAYFORM1 Equality E ξ min{1, ξ} = P{ξ > u} is obvious.

DISPLAYFORM2 Equality P{ξ > u} = 1 − 1 2 E ξ |ξ − 1| can be proofed as follows.

DISPLAYFORM3 where F ξ (u) is CDF of random variable ξ.

Note that F ξ (0) = 0 since ξ ∈ (0, +∞].

Eq. 21 can be rewritten in two ways.

DISPLAYFORM4 To rewrite Eq. 21 in the second way we note that Eξ = 1.

DISPLAYFORM5 Summing equations 22 and 23 results in the following formula DISPLAYFORM6 Using the form of ξ we can rewrite the acceptance rate as DISPLAYFORM7

In independent case we have ξ = DISPLAYFORM0 and we want to prove that E ξ |ξ − 1| is semimetric (or pseudo-metric) in space of distributions.

For this appendix, we denote D(p, q) = E ξ |ξ − 1|.

The first two axioms for metric obviously holds DISPLAYFORM1 There is an example when triangle inequality does not hold.

DISPLAYFORM2 But weaker inequality can be proved.

DISPLAYFORM3 Summing up equations 28, 30 and 32 we obtain DISPLAYFORM4 B OPTIMIZATION OF PROPOSAL DISTRIBUTION

Firstly, let's consider the case of gaussian random-walk proposal q(x | x) = N (x | x, σ).

The optimization problem for the acceptance rate takes the form DISPLAYFORM0 It is easy to see that we can obtain acceptance rate arbitrarly close to 1, taking σ small enough.

In the case of the independent proposal, we don't have the collapsing to the delta-function problem.

In our work, it is important to show non-collapsing during optimization of the lower bound, but the same hold for the direct optimization of the acceptance rate.

To provide such intuition we consider one-dimensional case where we have some target distribution p(x) and independent proposal q(x) = N (x | µ, σ).

Choosing σ small enough, we approximate sampling with the independent MH as sampling on some finite support x ∈ [µ − a, µ + a].

For this support, we approximate the target distribution with the uniform distribution (see FIG5 ).For such approximation, optimization of lower bound takes the form Here N (x | 0, σ, −a, a) is truncated normal distribution.

The first KL-divergence can be written as follows.

DISPLAYFORM1 DISPLAYFORM2 Here Z is normalization constant of truncated log normal distribution and DISPLAYFORM3 Summing up two KL-divergencies and taking derivative w.r.t.

σ we obtain ∂ ∂σ DISPLAYFORM4 To show that the derivative of the lower bound w.r.t.

σ is negative, we need to prove that the following inequality holds for positive x. DISPLAYFORM5 2 /2 dt and noting that 2φ(x) = √ 2π(Φ(x) − Φ(−x)) we can rewrite inequality 47 as DISPLAYFORM6 By the fundamental theorem of calculus, we have DISPLAYFORM7 Hence, DISPLAYFORM8 Or equivalently, DISPLAYFORM9 Using this inequality twice, we obtain DISPLAYFORM10 and DISPLAYFORM11 Thus, the target inequality can be verified by the verification of DISPLAYFORM12 Thus, we show that partial derivative of our lower bound w.r.t.

σ is negative.

Using that knowledge we can improve our loss by taking a bigger value of σ.

Hence, such proposal does not collapse to delta-function.

In this section, we provide an intuition for sample-based setting that the loss function for lower bound has better gradients than the loss function for acceptance rate.

Firstly, we remind that in the sample-based setting we use a discriminator for density ratio estimation.

DISPLAYFORM0 For this purpose we use the discriminator of special structure DISPLAYFORM1 We denote d(x, x ) = D(x, x ) − D(x , x) and consider the case when the discriminator can easily distinguish fake pairs from valid pairs.

So D(x, x ) is close to 1 and d(x, x ) 0 for x ∼ p(x) and x ∼ q(x | x).

To evaluate gradients we consider Monte Carlo estimations of each loss and take gradients w.r.t.

x in order to obtain gradients for parameters of proposal distribution.

We do not introduce the reparameterization trick to simplify the notation but assume it to be performed.

For the optimization of the acceptance rate we have DISPLAYFORM2 DISPLAYFORM3 While for the optimization of the lower bound we have DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 Now we compare Eq. 59 and Eq. 62.

We see that in case of strong discriminator we have vanishing gradients in Eq. 59 due to exp(−d(x, x )), while it is not the case for Eq. 62.

This experiment shows that it is possible to optimize the acceptance rate, optimizing its lower bound.

For the target distribution we consider bimodal Gaussian p(x) = 0.5 · N (x | − 2, 0.5) + 0.5 · N (x | 2, 0.7), for the independent proposal we consider unimodal gaussian q(x) = N (x | µ, σ).

We perform stochastic gradient optimization using Algorithm 1 from the same initialization for both objectives FIG6 and obtain approximately the same local maximums.

For the proposal distribution we use similar architecture to the NICE proposal.

The RealNVP model BID3 use the same strategy for evaluating the Jacobian as the NICE model does.

Each coupling layer define the following function.

Given a D dimensional input x and d < D, the output y is evaluated by the formula DISPLAYFORM0 where the functions s, t can be arbitrary complex, since the structure of the functions doesn't influence the computation of the Jacobian.

For our proposal we use 4 coupling layers with s and t consist of two fully-connected layers with hidden dimension of 256.

For synthetic distributions we consider the same distributions as in BID26 .The analytic form of p(x) for ring is: DISPLAYFORM0 The analytic form of p(x) for mog2 is: DISPLAYFORM1 where DISPLAYFORM2 The analytic form of p(x) for mog6 is: DISPLAYFORM3 where DISPLAYFORM4 DISPLAYFORM5 where DISPLAYFORM6

For the effective sample size formulation we follow BID26 .Assume a target distribution p(x), and a Markov chain Monte Carlo (MCMC) sampler that produces a set of N correlated samples DISPLAYFORM0 .

Suppose we are estimating the mean of p(x) through sampling; we assume that increasing the number of samples will reduce the variance of that estimate.

DISPLAYFORM1 DISPLAYFORM2 where ρ s denotes the autocorrelation under q of x at lag s. We compute the following empirical estimateρ s for ρ s :ρ DISPLAYFORM3 whereμ andσ are the empirical mean and variance obtained by an independent sampler.

Due to the noise in large lags s, we adopt the approach of Hoffman & Gelman FORMULA1 where we truncate the sum over the autocorrelations when the autocorrelation goes below 0.05.

In this section we provide the empirical evidence that maximization of the proposed lower bound on the acceptance rate (ARLB) results in maximization of the acceptance rate (AR).

For that purpose we evaluate ARLB and AR at each iteration during the optimization of ARLB.

After training we evaluate correlation coefficient between ARLB and logarithm of AR.

The curves are shown in FIG9 : plots for the acceptance rate and the acceptance rate lower bound evaluated at every iteration during the optimization of the acceptance rate lower bound.

Correlation coefficient is evaluated between the logarithm of the acceptance rate and the acceptance rate lower bound.

In this section we provide levelplots of learned proposals densities (see FIG0 ).

We also provide 2d histrograms of samples from the MH algorithm using the corresponding proposals (see FIG0 ).

In this section, we show additional figures for Markov chain proposals.

In FIG0 we show samples from the chain that was initialized by the noise.

In FIG0 we show samples from the chain after 10000 accepted samples.

FIG0 : Samples from the chain initialized with noise.

To obtain samples we use the MH algorithm with the learned proposal and the learned discriminator for density ratio estimation.

In Fig. 5 (a) we use proposal and discriminator that are learned during optimization of acceptance rate.

In Fig. 5(b) we use proposal and discriminator that are learned during the optimization of the acceptance rate lower bound.

Samples in the chain are obtained one by one from left to right from top to bottom starting with noise (first image in the figure).

FIG0 : Samples from the chain after 10000 accepted samples.

To obtain samples we use the MH algorithm with the learned proposal and the learned discriminator for density ratio estimation.

In Fig. 5(a) we use proposal and discriminator that are learned during optimization of acceptance rate.

In Fig. 5(b) we use proposal and discriminator that are learned during the optimization of the acceptance rate lower bound.

Samples in chain are obtained one by one from left to right from top to bottom.

<|TLDR|>

@highlight

Learning to sample via lower bounding the acceptance rate of the Metropolis-Hastings algorithm