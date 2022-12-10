We propose a new Integral Probability Metric (IPM) between distributions: the Sobolev IPM.

The Sobolev IPM compares the mean discrepancy of two distributions for functions (critic) restricted to a Sobolev ball defined with respect to a dominant measure mu.

We show that the Sobolev IPM compares two distributions in high dimensions based on weighted conditional Cumulative Distribution Functions (CDF) of each coordinate on a leave one out basis.

The Dominant measure mu plays a crucial role as it defines the support on which conditional CDFs are compared.

Sobolev IPM can be seen as an extension of the one dimensional Von-Mises Cramer statistics to high dimensional distributions.

We show how Sobolev IPM can be used to train Generative Adversarial Networks (GANs).

We then exploit the intrinsic conditioning implied by Sobolev IPM in text generation.

Finally we show that a variant of Sobolev GAN achieves competitive results in semi-supervised learning on CIFAR-10, thanks to the smoothness enforced on the critic by Sobolev GAN which relates to Laplacian regularization.

In order to learn Generative Adversarial Networks BID14 , it is now well established that the generator should mimic the distribution of real data, in the sense of a certain discrepancy measure.

Discrepancies between distributions that measure the goodness of the fit of the neural generator to the real data distribution has been the subject of many recent studies BID1 BID36 BID20 BID30 BID17 , most of which focus on training stability.

In terms of data modalities, most success was booked in plausible natural image generation after the introduction of Deep Convolutional Generative Adversarial Networks (DCGAN) BID39 .

This success is not only due to advances in training generative adversarial networks in terms of loss functions and stable algorithms, but also to the representation power of convolutional neural networks in modeling images and in finding sufficient statistics that capture the continuous density function of natural images.

When moving to neural generators of discrete sequences generative adversarial networks theory and practice are still not very well understood.

Maximum likelihood pre-training or augmentation, in conjunction with the use of reinforcement learning techniques were proposed in many recent works for training GAN for discrete sequences generation BID50 BID40 .

Other methods included using the Gumbel Softmax trick BID23 ) and the use of auto-encoders to generate adversarially discrete sequences from a continuous space BID51 .

End to end training of GANs for discrete sequence generation is still an open problem BID38 .

Empirical successes of end to end training have been reported within the framework of WGAN-GP BID17 , using a proxy for the Wasserstein distance via a pointwise gradient penalty on the critic.

Inspired by this success, we propose in this paper a new Integral Probability Metric (IPM) between distributions that we coin Sobolev IPM.

Intuitively an IPM BID35 between two probability distributions looks for a witness function f , called critic, that maximally discriminates between samples coming from the two distributions: DISPLAYFORM0 Traditionally, the function f is defined over a function class F that is independent to the distributions at hand BID48 .

The Wasserstein-1 distance corresponds for instance to an IPM where the witness functions are defined over the space of Lipschitz functions; The MMD distance corresponds to witness functions defined over a ball in a Reproducing Kernel Hilbert Space (RKHS).We will revisit in this paper Fisher IPM defined in , which extends the IPM definition to function classes defined with norms that depend on the distributions.

Fisher IPM can be seen as restricting the critic to a Lebsegue ball defined with respect to a dominant measure µ. The Lebsegue norm is defined as follows: DISPLAYFORM1 where µ is a dominant measure of P and Q.In this paper we extend the IPM framework to critics bounded in the Sobolev norm: DISPLAYFORM2 In contrast to Fisher IPM, which compares joint probability density functions of all coordinates between two distributions, we will show that Sobolev IPM compares weighted (coordinate-wise) conditional Cumulative Distribution Functions for all coordinates on a leave on out basis.

Matching conditional dependencies between coordinates is crucial for sequence modeling.

Our analysis and empirical verification show that the modeling of the conditional dependencies can be built in to the metric used to learn GANs as in Sobolev IPM.

For instance, this gives an advantage to Sobolev IPM in comparing sequences over Fisher IPM.

Nevertheless, in sequence modeling when we parametrize the critic and the generator with a neural network, we find an interesting tradeoff between the metric used and the architectures used to parametrize the critic and the generator as well as the conditioning used in the generator.

The burden of modeling the conditional long term dependencies can be handled by the IPM loss function as in Sobolev IPM (more accurately the choice of the data dependent function class of the critic) or by a simpler metric such as Fisher IPM together with a powerful architecture for the critic that models conditional long term dependencies such as LSTM or GRUs in conjunction with a curriculum conditioning of the generator as done in BID38 .

Highlighting those interesting tradeoffs between metrics, data dependent functions classes for the critic (Fisher or Sobolev) and architectures is crucial to advance sequence modeling and more broadly structured data generation using GANs.3.

The intrinsic conditioning and the CDF matching make Sobolev IPM suitable for discrete sequence matching and explain the success of the gradient pernalty in WGAN-GP and Sobolev GAN in discrete sequence generation.

4.

We give in Section 5 an ALM (Augmented Lagrangian Multiplier) algorithm for training Sobolev GAN.

Similar to Fisher GAN, this algorithm is stable and does not compromise the capacity of the critic.

5.

We show in Appendix A that the critic of Sobolev IPM satisfies an elliptic Partial Differential Equation (PDE).

We relate this diffusion to the Fokker-Planck equation and show the behavior of the gradient of the optimal Sobolev critic as a transportation plan between distributions.

6.

We empirically study Sobolev GAN in character level text generation (Section 6.1).

We validate that the conditioning implied by Sobolev GAN is crucial for the success and stability of GAN in text generation.

As a take home message from this study, we see that text generation succeeds either by implicit conditioning i.e using Sobolev GAN (or WGAN-GP) together with convolutional critics and generators, or by explicit conditioning i.e using Fisher IPM together with recurrent critic and generator and curriculum learning.

7.

We finally show in Section 6.2 that a variant of Sobolev GAN achieves competitive semisupervised learning results on CIFAR-10, thanks to the smoothness implied by the Sobolev regularizer.

In this Section, we review different representations of probability distributions and metrics for comparing distributions that use those representations.

Those metrics are at the core of training GAN.In what follows, we consider probability measures with a positive weakly differentiable probability density functions (PDF).

Let P and Q be two probability measures with PDFs P(x) and Q(x) defined on X ⊂ R d .

Let F P and F Q be the Cumulative Distribution Functions (CDF) of P and Q respectively.

For x = (x 1 , . . .

, x d ), we have: DISPLAYFORM0 The score function of a density function is defined as: DISPLAYFORM1 In this work, we are interested in metrics between distributions that have a variational form and can be written as a suprema of mean discrepancies of functions defined on a specific function class.

This type of metrics include ϕ-divergences as well as Integral Probability Metrics BID47 and have the following form: DISPLAYFORM2 where F is a function class defined on X and ∆ is a mean discrepancy, ∆ : F → R. The variational form given above leads in certain cases to closed form expressions in terms of the PDFs P, Q or in terms of the CDFs F P , F Q or the score functions s P , s Q .In TAB0 , we give a comparison of different discrepancies ∆ and function spaces F used in the literature for GAN training together with our proposed Sobolev IPM.

We see from TAB0 that Sobolev IPM, compared to Wasserstein Distance, imposes a tractable smoothness constraint on the critic on points sampled from a distribution µ, rather then imposing a Lipschitz constraint on all points in the space X .

We also see that Sobolev IPM is the natural generalization of the Cramér Von-Mises Distance from one dimension to high dimensions.

We note that the Energy Distance, a form of Maximum Mean Discrepancy for a special kernel, was used in BID6 as a generalization of the Cramér distance in GAN training but still needed a gradient penalty in its algorithmic counterpart leading to a mis-specified distance between distributions.

Finally it is worth noting that when comparing Fisher IPM and Sobolev IPM we see that while Fisher IPM compares joint PDF of the distributions, Sobolev IPM compares weighted (coordinate-wise) conditional CDFs.

As we will see later, this conditioning nature of the metric makes Sobolev IPM suitable for comparing sequences.

Note that the Stein metric BID27 uses the score function to match distributions.

We will show later how Sobolev IPM relates to the Stein discrepancy (Appendix A).

DISPLAYFORM3 Q(x) ) BID14 ) BID36 ϕ * Fenchel Conjugate Sinkhorn Divergence BID17 ) (Genevay et al., 2017) MMD ) BID26 BID11 Stein DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 f smooth with zero has a closed form boundary condition in RKHS BID5 boundary condition DISPLAYFORM7 DISPLAYFORM8 (This work) with zero boundary condition where DISPLAYFORM9

Imposing data-independent constraints on the function class in the IPM framework, such as the Lipschitz constraint in the Wasserstein distance is computationally challenging and intractable for the general case.

In this Section, we generalize the Fisher IPM introduced in , where the function class is relaxed to a tractable data dependent constraint on the second order moment of the critic, in other words the critic is constrained to be in a Lebsegue ball.

Fisher IPM.

Let X ⊂ R d and P(X ) be the space of distributions defined on X .

Let P, Q ∈ P(X ), and µ be a dominant measure of P and Q, in the sense that µ(x) = 0 =⇒ P(x) = 0 and Q(x) = 0.We assume µ to be also a distribution in P(X ), and assume µ(x) > 0, ∀x ∈ X .

Let L 2 (X , µ) be the space of µ-measurable functions.

For f, g ∈ L 2 (X , µ), we define the following dot product and its corresponding norm: DISPLAYFORM0 Note that L 2 (X , µ), can be formally defined as follows: DISPLAYFORM1 We define the unit Lebesgue ball as follows: DISPLAYFORM2 Fisher IPM defined in , searches for the critic function in the Lebesgue Ball B 2 (X , µ) that maximizes the mean discrepancy between P and Q. Fisher GAN was originally formulated specifically for µ = 1 2 (P + Q).

We consider here a general µ as long as it dominates P and Q. We define Generalized Fisher IPM as follows: DISPLAYFORM3 Note that: DISPLAYFORM4 .Hence Fisher IPM can be written as follows: DISPLAYFORM5 We have the following result: Theorem 1 (Generalized Fisher IPM).

The Fisher distance and the optimal critic are as follows:1.

The Fisher distance is given by: DISPLAYFORM6 2.

The optimal f χ achieving the Fisher distance F µ (P, Q) is: DISPLAYFORM7 Proof of Theorem 1.

From Equation (2), the optimal f χ belong to the intersection of the hyperplane that has normal n = P−Q µ , and the ball B 2 (X , µ), hence f χ = n n L 2 (X ,µ).

Hence DISPLAYFORM8 We see from Theorem 1 the role of the dominant measure µ: the optimal critic is defined with respect to this measure and the overall Fisher distance can be seen as an average weighted distance between probability density functions, where the average is taken on points sampled from µ. We give here some choices of µ:1.

For µ = 1 2 (P + Q), we obtain the symmetric chi-squared distance as defined in .

2. µ GP , the implicit distribution defined by the interpolation lines between P r and Q θ as in BID17 .

3.

When µ does not dominate P, and Q, we obtain a non symmetric divergence.

For example for µ = P, DISPLAYFORM9 dx.

We see here that for this particular choice we obtain the Pearson divergence.

In this Section, we introduce the Sobolev IPM.

In a nutshell, the Sobolev IPM constrains the critic function to belong to a ball in the restricted Sobolev Space.

In other words we constrain the norm of the gradient of the critic ∇ x f (x).

We will show that by moving from a Lebesgue constraint as in Fisher IPM to a Sobolev constraint as in Sobolev IPM, the metric changes from a joint PDF matching to weighted (ccordinate-wise) conditional CDFs matching.

The intrinsic conditioning built in to the Sobolev IPM and the comparison of cumulative distributions makes Sobolev IPM suitable for comparing discrete sequences.

We will start by recalling some definitions on Sobolev Spaces.

We assume in the following that X is compact and consider functions in the Sobolev space W 1,2 (X , µ): DISPLAYFORM0 We restrict ourselves to functions in W 1,2 (X , µ) vanishing at the boundary, and note this space W 1,2 0 (X , µ).

Note that in this case: DISPLAYFORM1 defines a semi-norm.

We can similarly define a dot product in W 1,2 DISPLAYFORM2 Hence we define the following Sobolev IPM, by restricting the critic of the mean discrepancy to the Sobolev unit ball : DISPLAYFORM3 When compared to the Wasserstein distance, the Sobolev IPM given in Equation FORMULA26 uses a data dependent gradient constraint (depends on µ) rather than a data independent Lipchitz constraint.

Let F P and F Q be the cumulative distribution functions of P and Q respectively.

We have: DISPLAYFORM4 and we define DISPLAYFORM5 high-order partial derivative excluding the variable i.

Our main result is presented in Theorem 2.

Additional theoretical results are given in Appendix A. All proofs are given in Appendix B. Theorem 2 (Sobolev IPM).

Assume that F P , and F Q and its d derivatives exist and are continuous: DISPLAYFORM6 The Sobolev IPM given in Equation (3) has the following equivalent forms:1.

Sobolev IPM as comparison of high order partial derivatives of CDFs.

The Sobolev IPM has the following form: DISPLAYFORM7 2.

Sobolev IPM as comparison of weighted (coordinate-wise) conditional CDFs.

The Sobolev IPM can be written in the following equivalent form: DISPLAYFORM8 3.

The optimal critic f * satisfies the following identity: DISPLAYFORM9 Sobolev IPM Approximation.

Learning in the whole Sobolev space W 1,2 0 is challenging hence we need to restrict our function class to a hypothesis class H , such as neural networks.

We assume in the following that functions in H vanish on the boundary of X , and restrict the optimization to the function space H .

H can be a Reproducing Kernel Hilbert Space as in the MMD case or parametrized by a neural network.

Define the Sobolev IPM approximation in H : DISPLAYFORM10 The following Lemma shows that the Sobolev IPM approximation in H is proportional to Sobolev IPM.

The tightness of the approximation of the Sobolev IPM is governed by the tightness of the approximation of the optimal Sobolev Critic f * in H .

This approximation is measured in the Sobolev sense, using the Sobolev dot product.

Lemma 1 (Sobolev IPM Approximation in a Hypothesis Class).

Let H be a function space with functions vanishing at the boundary.

For any f ∈ H and for f * the optimal critic in W 1,2 0 , we have: DISPLAYFORM11 Note that this Lemma means that the Sobolev IPM is well approximated if the space H has an enough representation power to express ∇ x f * (x).

This is parallel to the Fisher IPM approximation where it is shown that the Fisher IPM approximation error is proportional to the critic approximation in the Lebesgue sense.

Having in mind that the gradient of the critic is the information that is passed on to the generator, we see that this convergence in the Sobolev sense to the optimal critic is an important property for GAN training.

Relation to Fokker-Planck Diffusion.

We show in Appendix A that the optimal Sobolev critic is the solution of the following elliptic PDE (with zero boundary conditions): DISPLAYFORM12 We further link the elliptic PDE given in Equation FORMULA35 and the Fokker-Planck diffusion.

As we illustrate in FIG2 (b) the gradient of the critic defines a transportation plan for moving the distribution mass from Q to P.Discussion of Theorem 2.

We make the following remarks on Theorem 2:1.

From Theorem 2, we see that the Sobolev IPM compares d higher order partial derivatives of the cumulative distributions F P and F Q , while Fisher IPM compares the probability density functions.2.

The dominant measure µ plays a similar role to Fisher: DISPLAYFORM13 the average distance is defined with respect to points sampled from µ.

Comparison of coordinate-wise Conditional CDFs.

We note in the following DISPLAYFORM0 Note that we have: DISPLAYFORM1 (Using Bayes rule) DISPLAYFORM2 is the cumulative distribution of the variable X i given the other variables X −i = x −i , weighted by the density function of X −i at x −i .

This leads us to the form given in Equation 5 .

We see that the Sobolev IPM compares for each dimension i the conditional cumulative distribution of each variable given the other variables, weighted by their density function.

We refer to this as comparison of coordinate-wise CDFs on a leave one out basis.

From this we see that we are comparing CDFs, which are better behaved on discrete distributions.

Moreover, the conditioning built in to this metric will play a crucial role in comparing sequences as the conditioning is important in this context (See section 6.1).

Sobolev IPM / Cramér Distance and Wasserstein-1 in one Dimension.

In one dimension, Sobolev IPM is the Cramér Distance (for µ uniform on X , we note this µ := 1).

While Sobolev IPM in one dimension measures the discrepancy between CDFs, the one dimensional Wasserstein-p distance measures the discrepancy between inverse CDFs: DISPLAYFORM0 Recall also that the Fisher IPM for uniform µ is given by : DISPLAYFORM1 Consider for instance two point masses P = δ a1 and Q = δ a2 with a 1 , a 2 ∈ R. The rationale behind using Wasserstein distance for GAN training is that since it is a weak metric, for far distributions Wasserstein distance provides some signal .

In this case, it is easy to see that DISPLAYFORM2 As we see from this simple example, CDF comparison is more suitable than PDF for comparing distributions on discrete spaces.

See FIG1 , for a further discussion of this effect in the GAN context.

Sobolev IPM between two 2D Gaussians.

We consider P and Q to be two dimensional Gaussians with means µ 1 and µ 2 and covariances Σ 1 and Σ 2 .

Let (x, y) be the coordinates in 2D.

We note F P and F Q the CDFs of P and Q respectively.

We consider in this example µ = P+Q 2 .

We know from Theorem 2 that the gradient of the Sobolev optimal critic is proportional to the following vector field: DISPLAYFORM3 In FIG2 we consider µ 1 = [1, 0], Σ 1 = 1.9 0.8 0.8 1.3 DISPLAYFORM4 In FIG2 (a) we plot the numerical solution of the PDE satisfied by the optimal Sobolev critic given in Equation (8), using MATLAB solver for elliptic PDEs (more accurately we solve DISPLAYFORM5 hence we obtain the solution of Equation FORMULA35 up to a normalization constant ( 1 Sµ(P,Q) )).

We numerically solve the PDE on a rectangle with zero boundary conditions.

We see that the optimal Sobolev critic separates the two distributions well.

In FIG2 (b) we then numerically compute the gradient of the optimal Sobolev critic on a 2D grid as given in Equation 9 (using numerical evaluation of the CDF and finite difference for the evaluation of the partial derivatives).

We plot in FIG2 (b) the density functions of P and Q as well as the vector field of the gradient of the optimal Sobolev critic.

As discussed in Section A.1, we see that the gradient of the critic (wrt to the input), defines on the support of µ = P+Q 2 a transportation plan for moving the distribution mass from Q to P.

Now we turn to the problem of learning GANs with Sobolev IPM.

Given the "real distribution" P r ∈ P(X ), our goal is to learn a generator g θ : Z ⊂ R nz → X , such that for z ∼ p z , the distribution of g θ (z) is close to the real data distribution P r , where p z is a fixed distribution on Z (for instance z ∼ N (0, I nz )).

We note Q θ for the "fake distribution" of g θ (z), z ∼ p z .

Consider DISPLAYFORM0 We consider these choices for µ: DISPLAYFORM1 i.ex ∼ P r orx = g θ (z), z ∼ p z with equal probability 1 2 .

2. µ GP is the implicit distribution defined by the interpolation lines between P r and Q θ as in BID17 i.e : DISPLAYFORM2 Sobolev GAN can be written as follows: min DISPLAYFORM3 For any choice of the parametric function class H p , note the constraint byΩ DISPLAYFORM4 DISPLAYFORM5 2 .

Note that, since the optimal theoretical critic is achieved on the sphere, we impose a sphere constraint rather than a ball constraint.

Similar to we define the Augmented Lagrangian corresponding to Sobolev GAN objective and constraint DISPLAYFORM6 where λ is the Lagrange multiplier and ρ > 0 is the quadratic penalty weight.

We alternate between optimizing the critic and the generator.

We impose the constraint when training the critic only.

Given θ, we solve max p min λ L S (p, θ, λ), for training the critic.

Then given the critic parameters p we optimize the generator weights θ to minimize the objective min θÊ (f p , g θ ).

See Algorithm 1.Algorithm 1 Sobolev GAN Input: ρ penalty weight, η Learning rate, n c number of iterations for training the critic, N batch size Initialize p, θ, λ = 0 repeat for j = 1 to n c do Sample a minibatch DISPLAYFORM7 Remark 1.

Note that in Algorithm 1, we obtain a biased estimate since we are using same samples for the cost function and the constraint, but the incurred bias can be shown to be small and vanishing as the number of samples increases as shown and justified in BID45 .Relation to WGAN-GP.

WGAN-GP can be written as follows: DISPLAYFORM8 The main difference between WGAN-GP and our setting, is that WGAN-GP enforces pointwise constraints on points drawn from µ = µ GP via a point-wise quadratic penalty DISPLAYFORM9 2 ) while we enforce that constraint on average as a Sobolev norm, allowing us the coordinate weighted conditional CDF interpretation of the IPM.

Sobolev IPM has two important properties; The first stems from the conditioning built in to the metric through the weighted conditional CDF interpretation.

The second stems from the diffusion properties that the critic of Sobolev IPM satisfies (Appendix A) that has theoretical and practical ties to the Laplacian regularizer and diffusion on manifolds used in semi-supervised learning BID4 .In this Section, we exploit those two important properties in two applications of Sobolev GAN: Text generation and semi-supervised learning.

First in text generation, which can be seen as a discrete sequence generation, Sobolev GAN (and WGAN-GP) enable training GANs without need to do explicit brute-force conditioning.

We attribute this to the built-in conditioning in Sobolev IPM (for the sequence aspect) and to the CDF matching (for the discrete aspect).

Secondly using GANs in semi-supervised learning is a promising avenue for learning using unlabeled data.

We show that a variant of Sobolev GAN can achieve strong SSL results on the CIFAR-10 dataset, without the need of any form of activation normalization in the networks or any extra ad hoc tricks.

In this Section, we present an empirical study of Sobolev GAN in character level text generation.

Our empirical study on end to end training of character-level GAN for text generation is articulated on four dimensions (loss, critic, generator, µ).

(1) the loss used (GP: WGAN-GP BID17 , S: Sobolev or F: Fisher) (2) the architecture of the critic (Resnets or RNN) (3) the architecture of the generator (Resnets or RNN or RNN with curriculum learning) (4) the sampling distribution µ in the constraint.

Text Generation Experiments.

We train a character-level GAN on Google Billion Word dataset and follow the same experimental setup used in BID17 .

The generated sequence length is 32 and the evaluation is based on Jensen-Shannon divergence on empirical 4-gram probabilities (JS-4) of validation data and generated data.

JS-4 may not be an ideal evaluation criteria, but it is a reasonable metric for current character-level GAN results, which is still far from generating meaningful sentences.

Annealed Smoothing of discrete P r in the constraint µ. Since the generator distribution will always be defined on a continuous space, we can replace the discrete "real" distribution P r with a smoothed version (Gaussian kernel smoothing) P r N (0, σ 2 I d ).

This corresponds to doing the following sampling for P r : x + ξ, x ∼ P r , and ξ ∼ N (0, σ 2 I d ).

Note that we only inject noise to the "real" distribution with the goal of smoothing the support of the discrete distribution, as opposed to instance noise on both "real" and "fake" to stabilize the training, as introduced in BID20 BID1 .

As it is common in optimization by continuation BID32 , we also anneal the noise level σ as the training progresses on a linear schedule.

In this setting, we compare (WGAN-GP,G=Resnet,D=Resnet,µ = µ GP ) to (Sobolev,G=Resnet,D=Resnet,µ) where µ is one of: (1) µ GP , (2) the noise smoothed µ s (σ) = DISPLAYFORM0 or (3) noise smoothed with annealing µ a s (σ 0 ) with σ 0 the initial noise level.

We use the same architectures of Resnet with 1D convolution for the critic and the generator as in BID17 ) (4 resnet blocks with hidden layer size of 512).

In order to implement the noise smoothing we transform the data into one-hot vectors.

Each one hot vector x is transformed to a probability vector p with 0.9 replacing the one and 0.1/(dict size − 1) replacing the zero.

We then sample from a Gaussian distribution N (0, σ 2 ), and use softmax to normalize log p + .

We use algorithm 1 for Sobolev GAN and fix the learning rate to 10 −4 and ρ to 10 −5 .

The noise level σ was annealed following a linear schedule starting from an initial noise level σ 0 (at iteration i, σ i = σ 0 (1 − i M axiter ), Maxiter=30K).

For WGAN-GP we used the open source implementation with the penalty λ = 10 as in BID17 .

Results are given in FIG4 for the JS-4 evaluation of both WGAN-GP and Sobolev GAN for µ = µ GP .

In FIG4 (b) we show the JS-4 evaluation of Sobolev GAN with the annealed noise smoothing µ a s (σ 0 ), for various values of the initial noise level σ 0 .

We see that the training succeeds in both cases.

Sobolev GAN achieves slightly better results than WGAN-GP for the annealing that starts with high noise level σ 0 = 1.5.

We note that without smoothing and annealing i.e using µ = Pr+Q θ 2 , Sobolev GAN is behind.

Annealed smoothing of P r , helps the training as the real distribution is slowly going from a continuous distribution to a discrete distribution.

See Appendix C ( Figure 6 ) for a comparison between annealed and non annealed smoothing.

We give in Appendix C a comparison of WGAN-GP and Sobolev GAN for a Resnet generator architecture and an RNN critic.

The RNN has degraded performance due to optimization difficulties.

We analyze how Fisher GAN behaves under different architectures of generators and critics.

We first fix the generator to be ResNet.

We study 3 different architectures of critics: ResNet, GRU (we follow the experimental setup from BID38 ), and hybrid ResNet+GRU BID41 .

We notice that RNN is unstable, we need to clip the gradient values of critics in [−0.5, 0.5], and the gradient of the Lagrange multiplier λ F to [−10 4 , 10 4 ].

We fix ρ F = 10 −7 and we use µ = µ GP .

We search the value for the learning rate in [10 −5 , 10 −4 ].

We see that for µ = µ GP and G = Resnet for various critic architectures, Fisher GAN fails at the task of text generation ( FIG5 .

Nevertheless, when using RNN critics FIG5 ) a marginal improvement happens over the fully collapsed state when using a resnet critic FIG5 .

We hypothesize that RNN critics enable some conditioning and factoring of the distribution, which is lacking in Fisher IPM.Finally FIG5 shows the result of training with recurrent generator and critic.

We follow BID38 in terms of GRU architecture, but differ by using Fisher GAN rather than WGAN-GP.

We use µ = Pr+Q θ 2i.e.

without annealed noise smoothing.

We train (F, D=RNN,G=RNN, Pr+Q θ 2 ) using curriculum conditioning of the generator for all lengths as done in BID38 : the generator is conditioned on 32 − characters and predicts the remaining characters.

We increment = 1 to 32 on a regular schedule (every 15k updates).

JS-4 is only computed when > 4.

We see in FIG5 that under curriculum conditioning with recurrent critics and generators, the training of Fisher GAN succeeds and reaches similar levels of Sobolev GAN (and WGAN-GP) .

Note that the need of this explicit brute force conditioning for Fisher GAN, highlights the implicit conditioning induced by Sobolev GAN via the gradient regularizer, without the need for curriculum conditioning.

It is important to note that by doing this explicit curriculum conditioning for Fisher GAN, we highlight the implicit conditioning induced by Sobolev GAN, via the gradient regularizer.

A proper and promising framework for evaluating GANs consists in using it as a regularizer in the semi-supervised learning setting BID22 .

As mentioned before, the Sobolev norm as a regularizer for the Sobolev IPM draws connections with the Laplacian regularization in manifold learning BID4 .

In the Laplacian framework of semi-supervised learning, the classifier satisfies a smoothness constraint imposed by controlling its Sobolev norm: et al., 2016) .

In this Section, we present a variant of Sobolev GAN that achieves competitive performance in semi-supervised learning on the CIFAR-10 dataset BID21 without using any internal activation normalization in the critic, such as batch normalization (BN) BID19 , layer normalization (LN) BID3 , or weight normalization BID43 .

DISPLAYFORM0 In this setting, a convolutional neural network Φ ω : X → R m is shared between the cross entropy (CE) training of a K-class classifier (S ∈ R K×m ) and the critic of GAN (See Figure 5) .

We have the following training equations for the (critic + classifer) and the generator: DISPLAYFORM1 where the main IPM objective with N samples: DISPLAYFORM2 Following we use the following "K + 1 parametrization" for the critic (See Figure 5 ) : DISPLAYFORM3 Note that p(y|x) = Softmax( S, Φ ω (x) ) y appears both in the critic formulation and in the CrossEntropy term in Equation (11).

Intuitively this critic uses the K class directions of the classifier S y to define the "real" direction, which competes with another K+1 th direction v that indicates fake samples.

This parametrization adapts the idea of , which was formulated specifically for the classic KL / JSD based GANs, to IPM-based GANs.

We saw consistently better results with the K + 1 formulation over the regular formulation where the classification layer S doesn't interact with the critic direction v. We also note that when applying a gradient penalty based constraint (either WGAN-GP or Sobolev) on the full critic f = f + − f − , it is impossible for the network to fit even the small labeled training set (underfitting), causing bad SSL performance.

This leads us to the formulation below, where we apply the Sobolev constraint only on f − .

Throughout this Section we fix µ = Pr+Q θ 2 .We propose the following two schemes for constraining the K+1 critic f ( DISPLAYFORM4 1) Fisher constraint on the critic: We restrict the critic to the following set: DISPLAYFORM5 This constraint translates to the following ALM objective in Equation FORMULA16 : DISPLAYFORM6 where the Fisher constraint ensures the stability of the training through an implicit whitened mean matching .2) Fisher+Sobolev constraint: We impose 2 constraints on the critic: DISPLAYFORM7 This constraint translates to the following ALM in Equation FORMULA16 : DISPLAYFORM8 Note that the fisher constraint on f ensures the stability of the training, and the Sobolev constraints on the "fake" critic f − enforces smoothness of the "fake" critic and thus the shared CNN Φ ω (x).

This is related to the classic Laplacian regularization in semi-supervised learning BID4 .

Table 2 shows results of SSL on CIFAR-10 comparing the two proposed formulations.

Similar to the standard procedure in other GAN papers, we do hyperparameter and model selection on the validation set.

We present baselines with a similar model architecture and leave out results with significantly larger convnets.

G and D architectures and hyperparameters are in Appendix D. Φ ω is similar to in architecture, but note that we do not use any batch, layer, or weight normalization yet obtain strong competitive accuracies.

We hypothesize that we don't need any normalization in the critic, because of the implicit whitening of the feature maps introduced by the Fisher and Sobolev constraints as explained in .

DISPLAYFORM9 Figure 5: "K+1" parametrization of the critic for semi-supervised learning.

Table 2 : CIFAR-10 error rates for varying number of labeled samples in the training set.

Mean and standard deviation computed over 5 runs.

We only use the K + 1 formulation of the critic.

Note that we achieve strong SSL performance without any additional tricks, and even though the critic does not have any batch, layer or weight normalization.

Baselines with * use either additional models like PixelCNN, or do data augmentation (translations and flips), or use a much larger model, either of which gives an advantage over our plain simple training method.

† is the result we achieved in our experimental setup under the same conditions but without "K+1" critic (see Appendix D), since BID17 does not have SSL results.

Number of labeled examples 1000 2000 4000 8000 ModelMisclassification rate CatGAN (Springenberg, 2015) 19.58 FM 21.83 ± 2.01 19.61 ± 2.09 18.63 ± 2.32 17.72 ± 1.82 ALI 19.98 ± 0.3 19.09 ± 0.15 17.99 ± 0.54 17.05 ± 0.50 Tangents Reg BID22 20.06 ± 0.5 16.78 ± 0.6 Π-model (Laine & Aila, 2016) * 16.55 ± 0.29 VAT BID31 14.87 Bad Gan BID9

We introduced the Sobolev IPM and showed that it amounts to a comparison between weighted (coordinate-wise) CDFs.

We presented an ALM algorithm for training Sobolev GAN.

The intrinsic conditioning implied by the Sobolev IPM explains the success of gradient regularization in Sobolev GAN and WGAN-GP on discrete sequence data, and particularly in text generation.

We highlighted the important tradeoffs between the implicit conditioning introduced by the gradient regularizer in Sobolev IPM, and the explicit conditioning of Fisher IPM via recurrent critics and generators in conjunction with the curriculum conditioning.

Both approaches succeed in text generation.

We showed that Sobolev GAN achieves competitive semi-supervised learning results without the need of any normalization, thanks to the smoothness induced by the gradient regularizer.

We think the Sobolev IPM point of view will open the door for designing new regularizers that induce different types of conditioning for general structured/discrete/graph data beyond sequences.

In this Section we present the theoretical properties of Sobolev IPM and how it relates to distributions transport theory and other known metrics between distributions, notably the Stein distance.

In this Section, we characterize the optimal critic of the Sobolev IPM as a solution of a non linear PDE.

The solution of the variational problem of the Sobolev IPM satisfies a non linear PDE that can be derived using standard tools from calculus of variations BID12 BID0 .

Theorem 3 (PDE satisfied by the Sobolev Critic).

The optimal critic of Sobolev IPM f * satisfies the following PDE: DISPLAYFORM0 Define the Stein Operator: DISPLAYFORM1 ) .

Hence we have the following Transport Equation of P to Q: DISPLAYFORM2 Recall the definition of Stein Discrepancy : DISPLAYFORM3 Theorem 4 (Sobolev and Stein Discrepanices).

The following inequality holds true: DISPLAYFORM4 Stein Good fitness of the model Q w.r.t to µ DISPLAYFORM5 Consider for example µ = P, and sequence Q n .

If the Sobolev distance goes S P (P, Q n ) → 0, the ratio r n (x) = Qn(x) P(x) converges in expectation (w.r.t to Q) to 1.

The speed of the convergence is given by the Stein Discrepancy S(Q n , P).Relation to Fokker-Planck Diffusion Equation and Particles dynamics.

Note that PDE satisifed by the Sobolev critic given in Equation FORMULA16 can be equivalently written: DISPLAYFORM6 written in this form, we draw a connection with the Fokker-Planck Equation for the evolution of a density function q t that is the density of particles X t ∈ R d evolving with a drift (a velocity field) DISPLAYFORM7 The Fokker-Planck Equation states that the evolution of the particles density q t satisfies: DISPLAYFORM8 Comparing Equation FORMULA16 and Equation FORMULA16 , we identify then the gradient of Sobolev critic as a drift.

This suggests that one can define "Sobolev descent" as the evolution of particles along the gradient flow: DISPLAYFORM9 where the density of X 0 is given by q 0 (x) = Q(x), where f * t is the Sobolev critic between q t and P. One can show that the limit distribution of the particles is P. The analysis of "Sobolev descent" and its relation to Stein Descent BID27 is beyond the scope of this paper and will be studied in a separate work.

Hence we see that the gradient of the Sobolev critic defines a transportation plan to move particles whose distribution is Q to particles whose distribution is P (See FIG2 .

This highlights the role of the gradient of the critic in the context of GAN training in term of transporting the distribution of the generator to the real distribution.

Proof of Theorem 2.

Let F P and F Q , be the cumulative distribution functions of P and Q respectively.

We have: DISPLAYFORM0 We note D = In the following we assume that F P , and F Q and its d derivatives exist and are continuous meaning that F P and F Q ∈ C d (X ).

The objective function in Equation FORMULA26 can be written as follows: DISPLAYFORM1 ⊗d the dot product is defined as follows: DISPLAYFORM2 and the norm is given : DISPLAYFORM3 We can write the objective in Equation FORMULA16 in term of the dot product in L 2 (X , µ) ⊗d : DISPLAYFORM4 On the other hand the constraint in Equation (3) can be written in terms of the norm in L 2 (X , µ) ⊗d : DISPLAYFORM5 Replacing the objective and constraint given in Equations FORMULA16 and FORMULA18 in Equation FORMULA26 , we obtain: DISPLAYFORM6 Hence we find also that the optimal critic f * satisfies: DISPLAYFORM7 Proof of Lemma 1.

DISPLAYFORM8 Hence we have: sup DISPLAYFORM9 It follows therefore that: DISPLAYFORM10 We conclude that the Sobolev IPM can be approximated in arbitrary space as long as it has enough capacity to approximate the optimal critic.

Interestingly the approximation error is measured now with the Sobolev semi-norm, while in Fisher it was measured with the Lebesgue norm.

Approximations with Sobolev Semi-norms are stronger then Lebesgue norms as given by the Poincare inequality (||f || L2 ≤ C f W 1,2 0 ), meaning if the error goes to zero in Sobolev sense it also goes to zero in the Lebesgue sense , but the converse is not true.

Proof of Theorem 3.

The proof follows similar arguments in the proofs of the analysis of Laplacian regularization in semi-supervised learning studied by BID0 .

DISPLAYFORM11 Note that this problem is convex in f BID12 .

Writing the lagrangian for equation FORMULA16 we get : DISPLAYFORM12 We denote P(x) − Q(x) as µ 1 (x).To get the optimal f , we need to apply KKT conditions on the above equation.

DISPLAYFORM13 From the calculus of variations: DISPLAYFORM14 Now we apply integration by part and set h to be zero at boundary as in BID0 .

We get : DISPLAYFORM15 Hence, DISPLAYFORM16 The functional derivative of L(f, λ), at any test function h vanishing on the boundary: DISPLAYFORM17 As a result, the solution f * of the partial differential equation given in equation FORMULA18 DISPLAYFORM18 Using the constraint in FORMULA18 we can get the value of λ * : DISPLAYFORM19 Proof of Theorem 4.

Define the Stein operator BID37 : DISPLAYFORM20 This operator was later used in defining the Stein discrepancy BID15 BID8 BID27 .Recall that Barbour generator theory provides us a way of constructing such operators that produce mean zero function under µ. It is easy to verify that: DISPLAYFORM21 Recall that this operator arises from the overdamped Langevin diffusion, defined by the stochastic differential equation: DISPLAYFORM22 where (W t ) t≥0 is a Wiener process.

This is related to plug and play networks for generating samples if the distribution is known, using the stochastic differential equation.

From Theorem 3, it is easy to see that the PDE the Sobolev Critic (f * , λ * = S µ (P, Q)) can be written in term of Stein Operator as follows: DISPLAYFORM23 Taking absolute values and the expectation with respect to Q: DISPLAYFORM24 Recall that the definition of Stein Discrepancy : DISPLAYFORM25 It follows that Sobolev IPM critic satisfies: DISPLAYFORM26 Hence we have the following inequality: DISPLAYFORM27 This is equivalent to: DISPLAYFORM28 Stein Good fitness of the model Q w.r.t to µ S µ (P, Q)Sobolev DistanceSimilarly we obtain: DISPLAYFORM29 µ(x) ≤ 2 S(P, µ)Stein Good fitness of µ w.r.t to P S µ (P, Q)

For instance consider µ = P, we have therefore:1 2 E x∼Q Q(x) P(x) − 1 ≤ S(Q, P)S P (P, Q).Note that the left hand side of the inequality is not the total variation distance.

Hence for a sequence Q n if the Sobolev distance goes S P (P, Q n ) → 0, the ratio r n (x) = Qn(x) P(x) converges in expectation (w.r.t to Q) to 1.

The speed of the convergence is given by the Stein Discrepancy S(Q n , P).One important observation here is that convergence of PDF ratio is weaker than the conditional CDF as given by the Sobolev distance and of the good fitness of score function as given by Stein discrepancy.

Comparison of annealed versus non annealed smoothing of P r in Sobolev GAN. (S, D=res, G=res, µs(σ = 1.

0)) (S, D=res, G=res, µ a s (σ0 = 1.

0)) (S, D=res, G=res, µs(σ = 1. 5)) (S, D=res, G=res, µ a s (σ0 = 1.

5))Figure 6: Comparison of annealed versus non annealed smoothing of P r in Sobolev GAN.

We see that annealed smoothing outperforms the non annealed smoothing experiments.

Sobolev GAN versus WGAN-GP with RNN.

We fix the generator architecture to Resnets.

The experiments of using RNN (GRU) as the critic architecture for WGAN-GP and Sobolev is shown in FIG9 where we used µ = µ GP for both cases.

We only apply gradient clipping to stabilize the performance without other tricks.

We can observe that using RNN degrades the performance.

We think that this is due to an optimization issue and a difficulty in training RNN under the GAN objective without any pre-training or conditioning.

@highlight

We define a new Integral Probability Metric (Sobolev IPM) and show how it can be used for training GANs for text generation and semi-supervised learning.

@highlight

Suggests a novel regularization scheme for GANs based on a Sobolev norm, measuring deviations between L2 norms of derivatives.

@highlight

The authors provide another type of GAN using the typical setup of a GAN but with a different function class, and produce a recipe for training GANs with that sort of function class.

@highlight

The paper proposes a different gradient penalty for GAN critics that forces the expected squared norm of the gradient to be equal to 1