Generative adversarial networks are a learning framework that rely on training a discriminator to estimate a measure of difference between a target and generated distributions.

GANs, as normally formulated, rely on the generated samples being completely differentiable w.r.t.

the generative parameters, and thus do not work for discrete data.

We introduce a method for training GANs with discrete data that uses the estimated difference measure from the discriminator to compute importance weights for generated samples, thus providing a policy gradient for training the generator.

The importance weights have a strong connection to the decision boundary of the discriminator, and we call our method boundary-seeking GANs (BGANs).

We demonstrate the effectiveness of the proposed algorithm with discrete image and character-based natural language generation.

In addition, the boundary-seeking objective extends to continuous data, which can be used to improve stability of training, and we demonstrate this on Celeba, Large-scale Scene Understanding (LSUN) bedrooms, and Imagenet without conditioning.

Generative adversarial networks (GAN, BID7 involve a unique generative learning framework that uses two separate models, a generator and discriminator, with opposing or adversarial objectives.

Training a GAN only requires back-propagating a learning signal that originates from a learned objective function, which corresponds to the loss of the discriminator trained in an adversarial manner.

This framework is powerful because it trains a generator without relying on an explicit formulation of the probability density, using only samples from the generator to train.

GANs have been shown to generate often-diverse and realistic samples even when trained on highdimensional large-scale continuous data BID31 .

GANs however have a serious limitation on the type of variables they can model, because they require the composition of the generator and discriminator to be fully differentiable.

With discrete variables, this is not true.

For instance, consider using a step function at the end of a generator in order to generate a discrete value.

In this case, back-propagation alone cannot provide the training signal, because the derivative of a step function is 0 almost everywhere.

This is problematic, as many important real-world datasets are discrete, such as character-or word-based representations of language.

The general issue of credit assignment for computational graphs with discrete operations (e.g. discrete stochastic neurons) is difficult and open problem, and only approximate solutions have been proposed in the past BID2 BID8 BID10 BID14 BID22 BID40 .

However, none of these have yet been shown to work with GANs.

In this work, we make the following contributions:• We provide a theoretical foundation for boundary-seeking GANs (BGAN), a principled method for training a generator of discrete data using a discriminator optimized to estimate an f -divergence BID29 BID30 .

The discriminator can then be used to formulate importance weights which provide policy gradients for the generator.• We verify this approach quantitatively works across a set of f -divergences on a simple classification task and on a variety of image and natural language benchmarks.•

We demonstrate that BGAN performs quantitatively better than WGAN-GP BID9 in the simple discrete setting.• We show that the boundary-seeking objective extends theoretically to the continuous case and verify it works well with some common and difficult image benchmarks.

Finally, we show that this objective has some improved stability properties within training and without.

In this section, we will introduce boundary-seeking GANs (BGAN), an approach for training a generative model adversarially with discrete data, as well as provide its theoretical foundation.

For BGAN, we assume the normal generative adversarial learning setting commonly found in work on GANs BID7 , but these ideas should extend elsewhere.

Assume that we are given empirical samples from a target distribution, {x DISPLAYFORM0 , where X is the domain (such as the space of images, word-or character-based representations of natural language, etc.).

Given a random variable Z over a space Z (such as [0, 1] m ), we wish to find the optimal parameters,θ ∈ R d , of a function, G θ : Z → X (such as a deep neural network), whose induced probability distribution, Q θ , describes well the empirical samples.

In order to put this more succinctly, it is beneficial to talk about a probability distribution of the empirical samples, P, that is defined on the same space as Q θ .

We can now consider the difference measure between P and Q θ , D(P, Q θ ), so the problem can be formulated as finding the parameters: DISPLAYFORM1 Defining an appropriate difference measure is a long-running problem in machine learning and statistics, and choosing the best one depends on the specific setting.

Here, we wish to avoid making strong assumptions on the exact forms of P or Q θ , and we desire a solution that is scalable and works with very high dimensional data.

Generative adversarial networks (GANs, BID7 fulfill these criteria by introducing a discriminator function, D φ : X → R, with parameters, φ, then defining a value function, DISPLAYFORM2 where samples z are drawn from a simple prior, h(z) (such as U (0, 1) or N (0, 1)).

Here, D φ is a neural network with a sigmoid output activation, and as such can be interpreted as a simple binary classifier, and the value function can be interpreted as the negative of the Bayes risk.

GANs train the discriminator to maximize this value function (minimize the mis-classification rate of samples coming from P or Q θ ), while the generator is trained to minimize it.

In other words, GANs solve an optimization problem: DISPLAYFORM3 Optimization using only back-propogation and stochastic gradient descent is possible when the generated samples are completely differentiable w.r.t.

the parameters of the generator, θ.

In the non-parametric limit of an optimal discriminator, the value function is equal to a scaled and shifted version of the Jensen-Shannon divergence, 2 * D JSD (P||Q θ ) − log 4, 1 which implies the generator is minimizing this divergence in this limit.

f -GAN BID30 generalized this idea over all f -divergences, which includes the Jensen-Shannon (and hence also GANs) but also the Kullback-Leibler, Pearson χ 2 , and squared-Hellinger.

Their work provides a nice formalism for talking about GANs that use f -divergences, which we rely on here.

Definition 2.1 (f -divergence and its dual formulation).

Let f : R + → R be a convex lower semicontinuous function and f : C ⊆ R → R be the convex conjugate with domain C. Next, let T be an arbitrary family of functions, T = {T : X → C}. Finally, let P and Q be distributions that are completely differentiable w.r.t.

the same Lebesgue measure, µ.2 The f -divergence, D f (P||Q θ ), generated by f , is bounded from below by its dual representation BID29 , DISPLAYFORM4 The inequality becomes tight when T is the family of all possible functions.

The dual form allows us to change a problem involving likelihood ratios (which may be intractable) to an maximization problem over T .

This sort of optimization is well-studied if T is a family of neural networks with parameters φ (a.k.a., deep learning), so the supremum can be found with gradient ascent BID30 .Definition 2.2 (Variational lower-bound for the f -divergence).

Let T φ = ν •F φ be a function, which is the composition of an activation function, ν : R → C and a neural network, F φ : X → R. We can write the variational lower-bound of the supremum in Equation 4 as 3 : DISPLAYFORM5 Maximizing Equation 5 provides a neural estimator of f -divergence, or neural divergence BID12 .

Given the family of neural networks, T Φ = {T φ } φ∈Φ , is sufficiently expressive, this bound can become arbitrarily tight, and the neural divergence becomes arbitrarily close to the true divergence.

As such, GANs are extremely powerful for training a generator of continuous data, leveraging a dual representation along with a neural network with theoretically unlimited capacity to estimate a difference measure.

For the remainder of this work, we will refer to T φ = ν •F φ as the discriminator and F φ as the statistic network (which is a slight deviation from other works).

We use the general term GAN to refer to all models that simultaneously minimize and maximize a variational lower-bound, V(P, Q θ , T φ ), of a difference measure (such as a divergence or distance).

In principle, this extends to variants of GANs which are based on integral probability metrics (IPMs, BID36 ) that leverage a dual representation, such as those that rely on restricting T through parameteric regularization or by constraining its output distribution BID37 .

Here we will show that, with the variational lower-bound of an f -divergence along with a family of positive activation functions, ν : R → R + , we can estimate the target distribution, P, using the generated distribution, Q θ , and the discriminator, T φ .Theorem 1.

Let f be a convex function and T ∈ T a function that satisfies the supremum in Equation 4 in the non-parametric limit.

Let us assume that P and Q θ (x) are absolutely continuous w.r.t.

a measure µ and hence admit densities, p(x) and q θ (x).

Then the target density function, p(x), is equal to (∂f /∂T )(T (x))q θ (x).

DISPLAYFORM0 Proof.

Following the definition of the f -divergence and the convex conjugate, we have: DISPLAYFORM1 As f is convex, there is an absolute maximum when DISPLAYFORM2 .

Rephrasing t as a function, T (x), and by the definition of T (x), we arrive at the desired result.

Theorem 1 indicates that the target density function can be re-written in terms of a generated density function and a scaling factor.

We refer to this scaling factor, w (x) = (∂f /∂T )(T (x)), as the optimal importance weight to make the connection to importance sampling 4 .

In general, an optimal discriminator is hard to guarantee in the saddle-point optimization process, so in practice, T φ will define a lower-bound that is not exactly tight w.r.t.

the f -divergence.

Nonetheless, we can define an estimator for the target density function using a sub-optimal T φ .

Definition 2.3 (f -divergence importance weight estimator).

Let f and f , and T φ (x) be defined as in Definitions 2.1 and 2.2 but where ν : DISPLAYFORM3 The non-negativity of ν is important as the densities are positive.

TAB0 provides a set of fdivergences (following suggestions of BID30 with only slight modifications) which are suitable candidates and yield positive importance weights.

Surprisingly, each of these yield the same function over the neural network before the activation function: w(x) = e F φ (x) .

5 It should be noted thatp(x) is a potentially biased estimator for the true density; however, the bias only depends on the tightness of the variational lower-bound: the tighter the bound, the lower the bias.

This problem reiterates the problem with all GANs, where proofs of convergence are only provided in the optimal or near-optimal limit BID7 BID30 BID23 .

As mentioned above and repeated here, GANs only work when the value function is completely differentiable w.r.t.

the parameters of the generator, θ.

The gradients that would otherwise be used to train the generator of discrete variables are zero almost everywhere, so it is impossible to train the generator directly using the value function.

Approximations for the back-propagated signal exist BID2 BID8 BID10 BID14 BID22 BID40 , but as of this writing, none has been shown to work satisfactorily in training GANs with discrete data.

Here, we introduce the boundary-seeking GAN as a method for training GANs with discrete data.

We first introduce a policy gradient based on the KL-divergence which uses the importance weights 4 In the case of the f -divergence used in BID7 , the optimal importance weight equals DISPLAYFORM0 Note also that the normalized weights resemble softmax probabilities Algorithm 1 .

Discrete Boundary Seeking GANs (θ, φ) ← initialize the parameters of the generator and statistic network repeat DISPLAYFORM1 Compute the un-normalized and normalized importance weights (applied uniformly if P and Q θ are multi-variate) DISPLAYFORM2 Optimize the generator parameters until convergence as a reward signal.

We then introduce a lower-variance gradient which defines a unique reward signal for each z and prove this can be used to solve our original problem.

Policy gradient based on importance sampling Equation 7 offers an option for training a generator in an adversarial way.

If we know the explicit density function, q θ , (such as a multivariate Bernoulli distribution), then we can, usingp(x) as a target (keeping it fixed w.r.t.

optimization of θ), train the generator using the gradient of the KL-divergence: DISPLAYFORM3 Here, the connection to importance sampling is even clearer, and this gradient resembles other importance sampling methods for training generative models in the discrete setting BID3 BID33 .

However, we expect the variance of this estimator will be high, as it requires estimating the partition function, β (for instance, using Monte-Carlo sampling).

We address reducing the variance from estimating the normalized importance weights next.

Lower-variance policy gradient Let q θ (x) = Z g θ (x | z)h(z)dz be a probability density function with a conditional density, DISPLAYFORM4

be a Monte-Carlo estimate of the normalized importance weights.

The gradient of the expected conditional KL-divergence w.r.t.

the generator parameters, θ, becomes: DISPLAYFORM0 where we have approximated the expectation using the Monte-Carlo estimate.

Minimizing the expected conditional KL-divergences is stricter than minimizing the KL-divergence in Equation 7, as it requires all of the conditional distributions to match independently.

We show that the KL-divergence of the marginal probabilities is zero when the expectation of the conditional KL-divergence is zero as well as show this estimator works better in practice in the Appendix.

Algorithm 1 describes the training procedure for discrete BGAN.

This algorithm requires an additional M times more computation to compute the normalized importance weights, though these can be computed in parallel exchanging space for time.

When the P and Q θ are multi-variate (such as with discrete image data), we make the assumption that the observed variables are independent conditioned on Z. The importance weights, w, are then applied uniformly across each of the observed variables.

Connection to policy gradients REINFORCE is a common technique for dealing with discrete data in GANs BID4 BID20 .

Equation 9 is a policy gradient in the special case that the reward is the normalized importance weights.

This reward approaches the likelihood ratio in the non-parametric limit of an optimal discriminator.

Here, we make another connection to REINFORCE as it is commonly used, with baselines, by deriving the gradient of the reversed KL-divergence.

Definition 2.4 (REINFORCE-based BGAN).

Let T φ (x) be defined as above where DISPLAYFORM1 .

Consider the gradient of the reversed KL-divergence: DISPLAYFORM2 From this, it is clear that we can consider the output of the statistic network, F φ (x), to be a reward and b = log β = E Q θ [w(x)] to be the analog of a baseline.

6 This gradient is similar to those used in previous works on discrete GANs, which we discuss in more detail in Section 3.

For continuous variables, minimizing the variational lower-bound suffices as an optimization technique as we have the full benefit of back-propagation to train the generator parameters, θ.

However, while the convergence of the discriminator is straightforward, to our knowledge there is no general proof of convergence for the generator except in the non-parametric limit or near-optimal case.

What's worse is the value function can be arbitrarily large and negative.

Let us assume that max T = M < ∞ is unique.

As f is convex, the minimum of the lower-bound over θ is: inf DISPLAYFORM0 In other words, the generator objective is optimal when the generated distribution, Q θ , is nonzero only for the set {x | T (x) = M }.

Even outside this worst-case scenario, the additional consequence of this minimization is that this variational lower-bound can become looser w.r.t.

the f -divergence, with no guarantee that the generator would actually improve.

Generally, this is avoided by training the discriminator in conjunction with the generator, possibly for many steps for every generator update.

However, this clearly remains one source of potential instability in GANs.

Equation 7 reveals an alternate objective for the generator that should improve stability.

Notably, we observe that for a given estimator,p(x), q θ (x) matches when w(x) = (∂f /∂T )(T (x)) = 1.

Definition 2.5 (Continuous BGAN objective for the generator).

Let G θ : Z → X be a generator function that takes as input a latent variable drawn from a simple prior, z ∼ h(z).

Let T φ and w(x) be defined as above.

We define the continuous BGAN objective as:θ = arg min θ (log w(G θ (z))) 2 .

We chose the log, as with our treatments of f -divergences in TAB0 , the objective is just the square of the statistic network output:θ DISPLAYFORM1 This objective can be seen as changing a concave optimization problem (which is poor convergence properties) to a convex one.

On estimating likelihood ratios from the discriminator Our work relies on estimating the likelihood ratio from the discriminator, the theoretical foundation of which we draw from f -GAN BID30 .

The connection between the likelihood ratios and the policy gradient is known in previous literature BID15 , and the connection between the discriminator output and the likelihood ratio was also made in the context of continuous GANs BID26 BID39 .

However, our work is the first to successfully formulate and apply this approach to the discrete setting.

Importance sampling Our method is very similar to re-weighted wake-sleep (RWS, BID3 , which is a method for training Helmholtz machines with discrete variables.

RWS also relies on minimizing the KL divergence, the gradients of which also involve a policy gradient over the likelihood ratio.

Neural variational inference and learning (NVIL, BID25 , on the other hand, relies on the reverse KL.

These two methods are analogous to our importance sampling and REINFORCE-based BGAN formulations above.

GAN for discrete variables Training GANs with discrete data is an active and unsolved area of research, particularly with language model data involving recurrent neural network (RNN) generators BID20 .

Many REINFORCE-based methods have been proposed for language modeling BID20 BID6 which are similar to our REINFORCE-based BGAN formulation and effectively use the sigmoid of the estimated loglikelihood ratio.

The primary focus of these works however is on improving credit assignment, and their approaches are compatible with the policy gradients provided in our work.

There have also been some improvements recently on training GANs on language data by rephrasing the problem into a GAN over some continuous space BID19 BID16 BID9 .

However, each of these works bypass the difficulty of training GANs with discrete data by rephrasing the deterministic game in terms of continuous latent variables or simply ignoring the discrete sampling process altogether, and do not directly solve the problem of optimizing the generator from a difference measure estimated from the discriminator.

Remarks on stabilizing adversarial learning, IPMs, and regularization A number of variants of GANs have been introduced recently to address stability issues with GANs.

Specifically, generated samples tend to collapse to a set of singular values that resemble the data on neither a persample or distribution basis.

Several early attempts in modifying the train procedure (Berthelot et al., 2017; BID35 as well as the identifying of a taxonomy of working architectures BID31 addressed stability in some limited setting, but it wasn't until Wassertstein GANs (WGAN, BID1 were introduced that there was any significant progress on reliable training of GANs.

WGANs rely on an integral probability metric (IPM, BID36 ) that is the dual to the Wasserstein distance.

Other GANs based on IPMs, such as Fisher GAN tout improved stability in training.

In contrast to GANs based on f -divergences, besides being based on metrics that are "weak", IPMs rely on restricting T to a subset of all possible functions.

For instance in WGANs, T = {T | T L ≤ K}, is the set of K-Lipschitz functions.

Ensuring a statistic network, T φ , with a large number of parameters is Lipschitz-continuous is hard, and these methods rely on some sort of regularization to satisfy the necessary constraints.

This includes the original formulation of WGANs, which relied on weight-clipping, and a later work BID9 which used a gradient penalty over interpolations between real and generated data.

Unfortunately, the above works provide little details on whether T φ is actually in the constrained set in practice, as this is probably very hard to evaluate in the high-dimensional setting.

Recently, BID32 introduced a gradient norm penalty similar to that in BID9 without interpolations and which is formulated in terms of f -divergences.

In our work, we've found that this approach greatly improves stability, and we use it in nearly all of our results.

That said, it is still unclear empirically how the discriminator objective plays a strong role in stabilizing adversarial learning, but at this time it appears that correctly regularizing the discriminator is sufficient.

We first verify the gradient estimator provided by BGAN works quantitatively in the discrete setting by evaluating its ability to train a classifier with the CIFAR-10 dataset BID17 .

The "generator" in this setting is a multinomial distribution, g θ (y | x) modeled by the softmax output of a neural network.

The discriminator, T φ (x, y), takes as input an image / label pair so that the variational lower-bound is: DISPLAYFORM0 For these experiments, we used a simple 4-layer convolutional neural network with an additional 3 fully-connected layers.

We trained the importance sampling BGAN on the set of f -divergences given in TAB0 as well as the REINFORCE counterpart for 200 epochs and report the accuracy on the test set.

In addition, we ran a simple classification baseline trained on cross-entropy as well as a continuous approximation to the problem as used in WGAN-based approaches BID9 .

No regularization other than batch normalization (BN, BID13 was used with the generator, while gradient norm penalty BID32 was used on the statistic networks.

For WGAN, we used clipping, and chose the clipping parameter, the number of discriminator updates, and the learning rate separately based on training set performance.

The baseline for the REIN-FORCE method was learned using a moving average of the reward.

Our results are summarized in TAB1 .

Overall, BGAN performed similarly to the baseline on the test set, with the REINFORCE method performing only slightly worse.

For WGAN, despite our best efforts, we could only achieve an error rate of 72.3% on the test set, and this was after a total of 600 epochs to train.

Our efforts to train WGAN using gradient penalty failed completely, despite it working with higher-dimension discrete data (see Appendix).

Image data: binary MNIST and quantized CelebA We tested BGAN using two imaging benchmarks: the common discretized MNIST dataset BID34 ) and a new quantized version of the CelebA dataset (see BID21 , for the original CelebA dataset).For CelebA quantization, we first downsampled the images from 64 × 64 to 32 × 32.

We then generated a 16-color palette using Pillow, a fork of the Python Imaging Project (https://pythonpillow.org).

This palette was then used to quantize the RGB values of the CelebA samples to a one-hot representation of 16 colors.

Our models used deep convolutional GANs (DCGAN, BID31 .

The generator is fed a vector of 64 i.i.d.

random variables drawn from a uniform distribution, [0, 1].

The output nonlinearity was sigmoid for MNIST to model the Bernoulli centers for each pixel, while the output was softmax for quantized CelebA.Our results show that training the importance-weighted BGAN on discrete MNIST data is stable and produces realistic and highly variable generated handwritten digits FIG0 ).

Further quantitative experiments comparing BGAN against WGAN with the gradient penalty (WGAN- GP Gulrajani et al., 2017) showed that when training a new discriminator on the samples directly (keeping the Right: Samples produced from the generator trained as a boundaryseeking GAN on the quantized CelebA for 50 epochs.

Table 3 : Random samples drawn from a generator trained with the discrete BGAN objective.

The model is able to successfully learn many important character-level English language patterns.

And it 's miant a quert could he He weirst placed produces hopesi What 's word your changerg bette " We pait of condels of money wi Sance Jory Chorotic , Sen doesin In Lep Edger 's begins of a find", Lankard Avaloma was Mr. Palin , What was like one of the July 2 " I stroke like we all call on a Thene says the sounded Sunday in The BBC nothing overton and sleaWith there was a passes ipposing About dose and warthestrinds fro College is out in contesting rev And tear he jumped by even a roy generator fixed), the final estimated distance measures were higher (i.e., worse) for WGAN-GP than BGAN, even when comparing using the Wasserstein distance.

The complete experiment and results are provided in the Appendix.

For quantized CelebA, the generator trained as a BGAN produced reasonably realistic images which resemble the original dataset well and with good diversity.

Next, we test BGAN in a natural language setting with the 1-billion word dataset BID5 , modeling at the character-level and limiting the dataset to sentences of at least 32 and truncating to 32 characters.

For character-level language generation, we follow the architecture of recent work BID9 , and use deep convolutional neural networks for both the generator and discriminator.

Training with BGAN yielded stable, reliably good character-level generation (Table 3) , though generation is poor compared to recurrent neural network-based methods BID38 BID24 .

However, we are not aware of any previous work in which a discrete GAN, without any continuous relaxation BID9 , was successfully trained from scratch without pretraining and without an auxiliary supervised loss to generate any sensible text.

Despite the low quality of the text relative to supervised recurrent language models, the result demonstrates the stability and capability of the proposed boundary-seeking criterion for training discrete GANs.

Here we present results for training the generator on the boundary-seeking objective function.

In these experiments, we use the original GAN variational lower-bound from BID7 , only modifying the generator function.

All results use gradient norm regularization BID32 to ensure stability.

We test here the ability of continuous BGAN to train on high-dimensional data.

In these experiments, we train on the CelebA, LSUN BID42 datasets, and the 2012 ImageNet dataset with all 1000 labels BID18 .

The discriminator and generator were both modeled as 4-layer Resnets BID11 ) without conditioning on labels or attributes.

Figure 3 shows examples from BGAN trained on these datasets.

Overall, the sample quality is very good.

Notably, our Imagenet model produces samples that are high quality, despite not being trained Published as a conference paper at ICLR 2018CelebA Imagenet LSUN Figure 3 : Highly realistic samples from a generator trained with BGAN on the CelebA and LSUN datasets.

These models were trained using a deep ResNet architecture with gradient norm regularization BID32 ).

The Imagenet model was trained on the full 1000 label dataset without conditioning.conditioned on the label and on the full dataset.

However, the story here may not be that BGAN necessarily generates better images than using the variational lower-bound to train the generator, since we found that images of similar quality on CelebA could be attained without the boundaryseeking loss as long as gradient norm regularization was used, rather we confirm that BGAN works well in the high-dimensional setting.

As mentioned above, gradient norm regularization greatly improves stability and allows for training with very large architectures.

However, training still relies on a delicate balance between the generator and discriminator: over-training the generator may destabilize learning and lead to worse results.

We find that the BGAN objective is resilient to such over-training.

Stability in training with an overoptimized generator To test this, we train on the CIFAR-10 dataset using a simple DCGAN architecture.

We use the original GAN objective for the discriminator, but vary the generator loss as the variational lower-bound, the proxy loss (i.e., the generator loss function used in BID7 , and the boundary-seeking loss (BGAN).

To better study the effect of these losses, we update the generator for 5 steps for every discriminator step.

Our results (Figure 4) show that over-optimizing the generator significantly degrades sample quality.

However, in this difficult setting, BGAN learns to generate reasonable samples in fewer epochs than other objective functions, demonstrating improved stability.

Following the generator gradient We further test the different objectives by looking at the effect of gradient descent on the pixels.

In this setting, we train a DCGAN BID31 using the proxy loss.

We then optimize the discriminator by training it for another 1000 updates.

Next, we perform gradient descent directly on the pixels, the original variational lower-bound, the proxy, and the boundary seeking losses separately.

Figure 4: Training a GAN with different generator loss functions and 5 updates for the generator for every update of the discriminator.

Over-optimizing the generator can lead to instability and poorer results depending on the generator objective function.

Samples for GAN and GAN with the proxy loss are quite poor at 50 discriminator epochs (250 generator epochs), while BGAN is noticeably better.

At 100 epochs, these models have improved, though are still considerably behind BGAN.Our results show that following the BGAN objective at the pixel-level causes the least degradation of image quality.

This indicates that, in training, the BGAN objective is the least likely to disrupt adversarial learning.

Reinterpreting the generator objective to match the proposal target distribution reveals a novel learning algorithm for training a generative adversarial network (GANs, BID7 .

This proposed approach of boundary-seeking provides us with a unified framework under which learning algorithms for both discrete and continuous variables are derived.

Empirically, we verified our approach quantitatively and showed the effectiveness of training a GAN with the proposed learning algorithm, which we call a boundary-seeking GAN (BGAN), on both discrete and continuous variables, as well as demonstrated some properties of stability.

Starting image (generated) 10k updates GAN Proxy GAN BGAN 20k updates Figure 5 : Following the generator objective using gradient descent on the pixels.

BGAN and the proxy have sharp initial gradients that decay to zero quickly, while the variational lower-bound objective gradient slowly increases.

The variational lower-bound objective leads to very poor images, while the proxy and BGAN objectives are noticeably better.

Overall, BGAN performs the best in this task, indicating that its objective will not overly disrupt adversarial learning.

Berthelot, David, Schumm, Tom, and Metz, Luke.

Began: Boundary equilibrium generative adversarial networks.

arXiv preprint arXiv:1703.10717, 2017.

In these experiments, we produce some quantitative measures for BGAN against WGAN with the gradient penalty (WGAN-GP, BID9 on the discrete MNIST dataset.

In order to use back-propagation to train the generator, WGAN-GP uses the softmax probabilities directly, bypassing the sampling process at pixel-level and problems associated with estimating gradients through discrete processes.

Despite this, WGAN-GP is been able to produce samples that visually resemble the target dataset.

Here, we train 3 models on the discrete MNIST dataset using identical architectures with the BGAN with the JS and reverse KL f -divergences and WGAN-GP objectives.

Each model was trained for 300 generator epochs, with the discriminator being updated 5 times per generator update for WGAN-GP and 1 time per generator update for the BGAN models (in other words, the generators were trained for the same number of updates).

This model selection procedure was chosen as the difference measure (i.e., JSD, reverse KL divergence, and Wasserstein distance) as estimated during training converged for each model.

WGAN-GP was trained with a gradient penalty hyper-parameter of 5.0, which did not differ from the suggested 10.0 in our experiments with discrete MNIST.

The BGAN models were trained with the gradient norm penalty of 5.0 BID32 .Next, for each model, we trained 3 new discriminators with double capacity (twice as many hidden units on each layer) to maximize the the JS and reverse KL divergences and Wasserstein distance, keeping the generators fixed.

These discriminators were trained for 200 epochs (chosen from convergence) with the same gradient-based regularizations as above.

For all of these models, the discriminators were trained using the samples, as they would be used in practical applications.

For comparison, we also trained an additional discriminator, evaluating the WGAN-GP model above on the Wasserstein distance using the softmax probabilities.

Final evaluation was done by estimating difference measures using 60000 MNIST training examples againt 60000 samples from each generator, averaged over 12 batches of 5000.

We used the training set as this is the distribution over which the discriminators were trained.

Test set estimates in general were close and did not diverge from training set distances, indicating the discriminators were not overfitting, but training set estimates were slightly higher on average.

Our results show that the estimates from the sampling distribution from BGAN is consistently lower than that from WGAN-GP, even when evaluating using the Wasserstein distance.

However, when training the discriminator on the softmax probabilities, WGAN-GP has a much lower Wasserstein distance.

Despite quantitative differences, samples from these different models were indistinguishable as far as quality by visual inspection.

This indicates that, though playing the adversarial game using the softmax outputs can generate realistic-looking samples, this procedure ultimately hurts the generator's ability to model a truly discrete distribution.

Here we validate the policy gradient provided in Equation 10 theoretically and empirically.

Theorem 2.

Let the expectation of the conditional KL-divergence be defined as in Equation 9 .

Then E h(z) [D KL (p(x | z) g θ (x | z))] = 0 =⇒ D KL (p(x)||q θ ) = 0.Proof.

As the conditional KL-divergence is has an absolute minimum at zero, the expectation can only be zero when the all of the conditional KL-divergences are zero.

In other words: DISPLAYFORM0 As per the definition ofp(x | z), this implies that α(z) = w(x) = C is a constant.

If w(x) is a constant, then the partition function β = CE Q θ [1] = C is a constant.

Finally, when w(x) β = 1, p(x) = q θ =⇒ D KL (p(x)||q θ ) = 0.In order to empirically evaluate the effect of using an Monte-Carlo estimate of β from Equation 8 versus the variance-reducing method in Equation 10, we trained several models using various sample sizes from the prior, h(z), and the conditional, g θ (x | z).We compare both methods with 64 samples from the prior and 5, 10, and 100 samples from the conditional.

In addition, we compare to a model that estimates β using 640 samples from the prior and a single sample from the conditional.

These models were all run on discrete MNIST for 50 epochs with the same architecture as those from Section 4.2 with a gradient penalty of 1.0, which was the minimum needed to ensure stability in nearly all the models.

Our results FIG1 ) show a clear improvement using the variance-reducing method from Equation 10 over estimating β.

Wall-clock times were nearly identical for methods using the same number of total samples (blue, green, and red dashed and solid line pairs).

Both methods improve as the number of conditional samples is increased. .

α indicates the variance-reducing method, and β is estimating β using Monte-Carlo.

z = indicates the number of samples from the prior, h(z), and x = indicates the number of samples from the conditional, g θ (x | z) used in estimation.

Plotted are the estimated GAN distances (2 * JSD − log 4) from the discriminator.

The minimum GAN distance, − log 4, is included for reference.

Using the variance-reducing method gives a generator with consistently lower estimated distances than estimating β directly.

<|TLDR|>

@highlight

We address training GANs with discrete data by formulating a policy gradient that generalizes across f-divergences