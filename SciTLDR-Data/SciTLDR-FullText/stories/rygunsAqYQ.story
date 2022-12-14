Implicit probabilistic models are models defined naturally in terms of a sampling procedure and often induces a likelihood function that cannot be expressed explicitly.

We develop a simple method for estimating parameters in implicit models that does not require knowledge of the form of the likelihood function or any derived quantities, but can be shown to be equivalent to maximizing likelihood under some conditions.

Our result holds in the non-asymptotic parametric setting, where both the capacity of the model and the number of data examples are finite.

We also demonstrate encouraging experimental results.

Generative modelling is a cornerstone of machine learning and has received increasing attention.

Recent models like variational autoencoders (VAEs) BID32 BID45 and generative adversarial nets (GANs) BID21 BID25 , have delivered impressive advances in performance and generated a lot of excitement.

Generative models can be classified into two categories: prescribed models and implicit models BID12 BID40 .

Prescribed models are defined by an explicit specification of the density, and so their unnormalized complete likelihood can be usually expressed in closed form.

Examples include models whose complete likelihoods lie in the exponential family, such as mixture of Gaussians BID18 , hidden Markov models BID5 , Boltzmann machines BID27 .

Because computing the normalization constant, also known as the partition function, is generally intractable, sampling from these models is challenging.

On the other hand, implicit models are defined most naturally in terms of a (simple) sampling procedure.

Most models take the form of a deterministic parameterized transformation T θ (·) of an analytic distribution, like an isotropic Gaussian.

This can be naturally viewed as the distribution induced by the following sampling procedure:1.

Sample z ∼ N (0, I) 2.

Return x := T θ (z)The transformation T θ (·) often takes the form of a highly expressive function approximator, like a neural net.

Examples include generative adversarial nets (GANs) BID21 BID25 and generative moment matching nets (GMMNs) BID36 BID16 .

The marginal likelihood of such models can be characterized as follows: DISPLAYFORM0 where φ(·) denotes the probability density function (PDF) of N (0, I).In general, attempting to reduce this to a closed-form expression is hopeless.

Evaluating it numerically is also challenging, since the domain of integration could consist of an exponential number of disjoint regions and numerical differentiation is ill-conditioned.

These two categories of generative models are not mutually exclusive.

Some models admit both an explicit specification of the density and a simple sampling procedure and so can be considered as both prescribed and implicit.

Examples include variational autoencoders BID32 BID45 , their predecessors BID38 BID10 and extensions BID11 , and directed/autoregressive models, e.g., BID42 BID6 BID33 van den Oord et al., 2016 ).

Maximum likelihood BID19 BID17 is perhaps the standard method for estimating the parameters of a probabilistic model from observations.

The maximum likelihood estimator (MLE) has a number of appealing properties: under mild regularity conditions, it is asymptotically consistent, efficient and normal.

A long-standing challenge of training probabilistic models is the computational roadblocks of maximizing the log-likelihood function directly.

For prescribed models, maximizing likelihood directly requires computing the partition function, which is intractable for all but the simplest models.

Many powerful techniques have been developed to attack this problem, including variational methods BID31 , contrastive divergence BID26 Welling & Hinton, 2002) , score matching BID30 and pseudolikelihood maximization BID9 , among others.

For implicit models, the situation is even worse, as there is no term in the log-likelihood function that is in closed form; evaluating any term requires computing an intractable integral.

As a result, maximizing likelihood in this setting seems hopelessly difficult.

A variety of likelihood-free solutions have been proposed that in effect minimize a divergence measure between the data distribution and the model distribution.

They come in two forms: those that minimize an f -divergence, and those that minimize an integral probability metric BID41 .

In the former category are GANs, which are based on the idea of minimizing the distinguishability between data and samples (Tu, 2007; BID24 .

It has been shown that when given access to an infinitely powerful discriminator, the original GAN objective minimizes the Jensen-Shannon divergence, the − log D variant of the objective minimizes the reverse KL-divergence minus a bounded quantity , and later extensions BID43 minimize arbitrary f -divergences.

In the latter category are GMMNs which use maximum mean discrepancy (MMD) BID22 as the witness function.

In the case of GANs, despite the theoretical results, there are a number of challenges that arise in practice, such as mode dropping/collapse BID21 , vanishing gradients Sinn & Rawat, 2017) and training instability BID21 .

A number of explanations have been proposed to explain these phenomena and point out that many theoretical results rely on three assumptions: the discriminator must have infinite modelling capacity BID21 , the number of samples from the true data distribution must be infinite Sinn & Rawat, 2017) and the gradient ascent-descent procedure BID4 Schmidhuber, 1992) can converge to a global pure-strategy Nash equilibrium BID21 .

When some of these assumptions do not hold, the theoretical guarantees do not necessarily apply.

A number of ways have been proposed that alleviate some of these issues, e.g., (Zhao et al., 2016; Salimans et al., 2016; BID13 BID15 BID28 Zhu et al., 2017) , but a way of solving all three issues simultaneously remains elusive.

In this paper, we present an alternative method for estimating parameters in implicit models.

Like the methods above, our method is likelihood-free, but can be shown to be equivalent to maximizing likelihood under some conditions.

Our result holds when the capacity of the model is finite and the number of data examples is finite.

The idea behind the method is simple: it finds the nearest sample to each data example and optimizes the model parameters to pull the sample towards it.

The direction in which nearest neighbour search is performed is important: the proposed method ensures each data example has a similar sample, which contrasts with an alternative approach of pushing each sample to the nearest data example, which would ensure that each sample has a similar data example.

The latter approach would permit all samples being similar to one data example.

Such a scenario would be heavily penalized by the former approach.

The proposed method could sidestep the three issues mentioned above: mode collapse, vanishing gradients and training instability.

Modes are not dropped because the loss ensures each data example has a sample nearby at optimality; gradients do not vanish because the gradient of the distance between a data example and its nearest sample does not become zero unless they coincide; training is stable because the estimator is the solution to a simple minimization problem.

By leveraging recent advances in fast nearest neighbour search algorithms BID34 , this approach is able to scale to large, high-dimensional datasets.longer the case, due to recent advances in nearest neighbour search algorithms BID34 .Note that the use of Euclidean distance is not a major limitation of the proposed approach.

A variety of distance metrics are either exactly or approximately equivalent to Euclidean distance in some non-linear embedding space, in which case the theoretical guarantees are inherited from the Euclidean case.

This encompasses popular distance metrics used in the literature, like the Euclidean distance between the activations of a neural net, which is often referred to as a perceptual similarity metric (Salimans et al., 2016; BID14 .

The approach can be easily extended to use these metrics, though because this is the initial paper on this method, we focus on the vanilla setting of Euclidean distance in the natural representation of the data, e.g.: pixels, both for simplicity/clarity and for comparability to vanilla versions of other methods that do not use auxiliary sources of labelled data or leverage domain-specific prior knowledge.

For distance metrics that cannot be embedded in Euclidean space, the analysis can be easily adapted with minor modifications as long as the volume of a ball under the metric has a simple dependence on its radius.

There has been debate BID29 over whether maximizing likelihood of the data is the appropriate objective for the purposes of learning generative models.

Recall that maximizing likelihood is equivalent to minimizing D KL (p data p θ ), where p data denotes the empirical data distribution and p θ denotes the model distribution.

One proposed alternative is to minimize the reverse KLdivergence, D KL (p θ p data ), which is suggested BID29 to be better because it severely penalizes the model for generating an implausible sample, whereas the standard KL-divergence, D KL (p data p θ ), severely penalizes the model for assigning low density to a data example.

As a result, when the model is underspecified, i.e. has less capacity than what's necessary to fit all the modes of the data distribution, minimizing D KL (p θ p data ) leads to a narrow model distribution that concentrates around a few modes, whereas minimizing D KL (p data p θ ) leads to a broad model distribution that hedges between modes.

The success of GANs in generating good samples is often attributed to the former phenomenon .This argument, however, relies on the assumption that we have access to an infinite number of samples from the true data distribution.

In practice, however, this assumption rarely holds: if we had access to the true data distribution, then there is usually no need to fit a generative model, since we can simply draw samples from the true data distribution.

What happens when we only have the empirical data distribution?

Recall that D KL (p q ) is defined and finite only if p is absolutely continuous w.r.t.

q, i.e.: q(x) = 0 implies p(x) = 0 for all x. In other words, D KL (p q ) is defined and finite only if the support of p is contained in the support of q. Now, consider the difference between D KL (p data p θ ) and D KL (p θ p data ): minimizing the former, which is equivalent to maximizing likelihood, ensures that the support of the model distribution contains all data examples, whereas minimizing the latter ensures that the support of the model distribution is contained in the support of the empirical data distribution, which is just the set of data examples.

In other words, maximum likelihood disallows mode dropping, whereas minimizing reverse KL-divergence forces the model to assign zero density to unseen data examples and effectively prohibits generalization.

Furthermore, maximum likelihood discourages the model from assigning low density to any data example, since doing so would make the likelihood, which is the product of the densities at each of the data examples, small.

From the modelling perspective, because maximum likelihood is guaranteed to preserve all modes, it can make use of all available training data and can therefore be used to train high-capacity models that have a large number of parameters.

In contrast, using an objective that permits mode dropping allows the model to pick and choose which data examples it wants to model.

As a result, if the goal is to train a high-capacity model that can learn the underlying data distribution, we would not be able to do so using such an objective because we have no control over which modes the model chooses to drop.

Put another way, we can think about the model's performance along two axes: its ability to generate plausible samples (precision) and its ability to generate all modes of the data distribution (recall).

A model that successfully learns the underlying distribution should score high along both axes.

If mode dropping is allowed, then an improvement in precision may be achieved at the expense of lower recall and could represent a move to a different point on the same precision-recall curve.

As a result, since sample quality is an indicator of precision, improvement in sample quality in this setting may not mean an improvement in density estimation performance.

On the other hand, if mode dropping is disallowed, since full recall is always guaranteed, an improvement in precision is achieved without sacrificing recall and so implies an upwards shift in the precision-recall curve.

In this case, an improvement in sample quality does signify an improvement in density estimation performance, which may explain sample quality historically was an important way to evaluate the performance of generative models, most of which maximized likelihood.

With the advent of generative models that permit mode dropping, however, sample quality is no longer a reliable indicator of density estimation performance, since good sample quality can be trivially achieved by dropping all but a few modes.

In this setting, sample quality can be misleading, since a model with low recall on a lower precision-recall curve can achieve a better precision than a model with high recall on a higher precision-recall curve.

Since it is hard to distinguish whether an improvement in sample quality is due to a move along the same precision-recall curve or a real shift in the curve, an objective that disallows mode dropping is critical tool that researchers can use to develop better models, since they can be sure that an apparent improvement in sample quality is due to a shift in the precision-recall curve.. Let F θ (·) be the cumulative distribution function (CDF) ofr θ and Ψ(z) := min θ E R θ |p θ (0) = z .If P θ satisfies the following:• p θ (x) is differentiable w.r.t.

θ and continuous w.r.t.

x everywhere.• ∀θ, v, there exists θ such that p θ (x) = p θ (x + v) ∀x.• For any θ 1 , θ 2 , there exists θ 0 such that DISPLAYFORM0

MNIST TFD DBN BID7 138 ± 2 1909 ± 66 SCAE BID7 121 ± 1.6 2110 ± 50 DGSN 214 ± 1.1 1890 ± 29 GAN BID21 225 ± 2 2057 ± 26 GMMN BID36 147 ± 2 2085 ± 25 IMLE (Proposed) 257 ± 6 2139 ± 27 Table 1 : Log-likelihood of the test data under the Gaussian Parzen window density estimated from samples generated by different methods.• DISPLAYFORM0 , where B θ * (τ ) denotes the ball centred at θ * of radius τ .• Ψ(z) is differentiable everywhere.• DISPLAYFORM1 . . .

DISPLAYFORM2 Now, we examine the restrictiveness of each condition.

The first condition is satisfied by nearly all analytic distributions.

The second condition is satisfied by nearly all distributions that have an unrestricted location parameter, since one can simply shift the location parameter by v. The third condition is satisfied by most distributions that have location and scale parameters, like a Gaussian distribution, since the scale can be made arbitrarily low and the location can be shifted so that the constraint on p θ (·) is satisfied.

The fourth condition is satisfied by nearly all distributions, whose density eventually tends to zero as the distance from the optimal parameter setting tends to infinity.

The fifth condition requires min θ E R θ |p θ (0) = z to change smoothly as z changes.

The final condition requires the two n-dimensional vectors, one of which can be chosen from a set of d vectors, to be not exactly orthogonal.

As a result, this condition is usually satisfied when d is large, i.e. when the model is richly parameterized.

There is one remaining difficulty in applying this theorem, which is that the quantity 1/Ψ (p θ * (x i ))p θ * (x i ), which appears as an coefficient on each term in the proposed objective, is typically not known.

If we consider a new objective that ignores the coefficients, i.e. n i=1 E R θ i , then minimizing this objective is equivalent to minimizing an upper bound on the ideal objective, DISPLAYFORM3 The tightness of this bound depends on the difference between the highest and lowest likelihood assigned to individual data points at the optimum, i.e. the maximum likelihood estimate of the parameters.

Such a model should not assign high likelihoods to some points and low likelihoods to others as long as it has reasonable capacity, since doing so would make the overall likelihood, which is the product of the likelihoods of individual data points, low.

Therefore, the upper bound is usually reasonably tight.

We trained generative models using the proposed method on three standard benchmark datasets, MNIST, the Toronto Faces Dataset (TFD) and CIFAR-10.

All models take the form of feedforward neural nets with isotropic Gaussian noise as input.

For MNIST, the architecture consists of two fully connected hidden layers with 1200 units each followed by a fully connected output layer with 784 units.

ReLU activations were used for hidden layers and sigmoids were used for the output layer.

For TFD, the architecture is wider and consists of two fully connected hidden layers with 8000 units each followed by a fully connected output layer with 2304 units.

For both MNIST and TFD, the dimensionality of the noise vector is 100.

For CIFAR-10, we used a simple convolutional architecture with 1000-dimensional Gaussian noise as input.

The architecture consists of five convolutional layers with 512 output channels and a kernel size of 5 that all produce 4 × 4 feature maps, followed by a bilinear upsampling layer that doubles the width and height of the feature maps.

There is a batch normalization layer followed by leaky ReLU activations with slope −0.2 after each convolutional layer.

This design is then repeated for each subsequent level of resolution, namely 8 × 8, 16 × 16 and 32 × 32, so that we have 20 convolutional layers, each with output 512 channels.

We then add a final output layer with three output channels on top, followed by sigmoid activations.

We note that this architecture has more capacity than typical architectures used in other methods, like BID44 .

This is because our method aims to capture all modes of the data distribution and therefore needs more modelling capacity than methods that are permitted to drop modes.

sirable properties of the generative model that do not affect performance on the task.

Intrinsic evaluation metrics measure performance without relying on external models or data.

Popular examples include estimated log-likelihood Wu et al., 2016) and visual assessment of sample quality.

While recent literature has focused more on the latter and less on the former, it should be noted that they evaluate different properties -sample quality reflects precision, i.e.: how accurate the model samples are compared to the ground truth, whereas estimated log-likelihood focuses on recall, i.e.: how much of the diversity in the data distribution the model captures.

Consequently, both are important metrics; one is not a replacement for the other.

As pointed out by (Theis et al., 2015) , "qualitative as well as quantitative analyses based on model samples can be misleading about a model's density estimation performance, as well as the probabilistic model's performance in applications other than image synthesis." Two models that achieve different levels of precision may simply be at different points on the same precision-recall curve, and therefore may not be directly comparable.

Models that achieve the same level of recall, on the other hand, may be directly compared.

So, for methods that maximize likelihood, which are guaranteed to preserve all modes and achieve full recall, both sample quality and estimated log-likelihood capture precision.

Because most generative models traditionally maximized likelihood or a lower bound on the likelihood, the only property that differed across models was precision, which may explain why sample quality has historically been seen as an important indicator of performance.

However, in heterogenous experimental settings with different models optimized for various objectives, sample quality does not necessarily reflect how well a model learns the underlying data distribution.

Therefore, under these settings, both precision and recall need to be measured.

While there is not yet a reliable way to measure recall (given the known issues of estimated log-likelihoods in high dimensions), this does not mean that sample quality can be a valid substitute for estimated log-likelihoods, as it cannot detect the lack of diversity of samples.

A secondary issue that is more easily solvable is that samples presented in papers are sometimes cherry-picked; as a result, they capture the maximum sample quality, but not necessarily the mean sample quality.

To mitigate these problems to some extent, we avoid cherry-picking and visualize randomly chosen samples, which are shown in FIG0 .

We also report the estimated log-likelihood in Table 1 .

As mentioned above, both evaluation criteria have biases/deficiencies, so performing well on either of these metrics does not necessarily indicate good density estimation performance.

However, not performing badly on either metric can provide some comfort that the model is simultaneously able to achieve reasonable precision and recall.

As shown in FIG0 , despite its simplicity, the proposed method is able to generate reasonably good samples for MNIST, TFD and CIFAR-10.

While it is commonly believed that minimizing reverse KL-divergence is necessary to produce good samples and maximizing likelihood necessarily leads to poor samples BID23 , the results suggest that this is not necessarily the case.

Even though Euclidean distance was used in the objective, the samples do not appear to be desaturated or overly blurry.

Samples also seem fairly diverse.

This is supported by the estimated log-likelihood results in Table 1 .

Because the model achieved a high score on that metric on both MNIST and TFD, this suggests that the model did not suffer from significant mode dropping.

In FIG3 in the supplementary material, we show samples and their nearest neighbours in the training set.

Each sample is quite different from its nearest neighbour in the training set, suggesting that the model has not overfitted to examples in the training set.

Next, we visualize the learned manifold by walking along a geodesic on the manifold between pairs of samples.

More concretely, we generate five samples, arrange them in arbitrary order, perform linear interpolation in latent variable space between adjacent pairs of samples, and generate an image from the interpolated latent variable.

As shown in FIG2 , the images along the path of interpolation appear visually plausible and do not have noisy artifacts.

In addition, the transition from one image to the next appears smooth, including for CIFAR-10, which contrasts with findings in the literature that suggest the transition between two natural images tends to be abrupt.

This indicates that the support of the model distribution has not collapsed to a set of isolated points and that the proposed method is able to learn the geometry of the data manifold, even though it does not learn a distance metric explicitly.

Finally, we illustrate the evolution of samples as training progresses in FIG1 .

As shown, the samples are initially blurry and become sharper over time.

Importantly, sample quality consistently improves over time, which demonstrates the stability of training.

While our sample quality may not be state-of-the-art, it is important to remember that these results are obtained under the setting of full recall.

So, this does not necessarily mean that our method models the underlying data distribution less accurately than other methods that achieve better sample quality, as some of them may drop modes and therefore achieve less than full recall.

As previously mentioned, this does not suggest a fundamental tradeoff between precision and recall that cannot be overcome -on the contrary, our method provides researchers with a way of designing models that can improve the precision-recall curve without needing to worry that the observed improvements are due to a movement along the curve.

With refinements to the model, it is possible to move the curve upwards and obtain better sample quality at any level of recall as a consequence.

This is left for future work; as this is the initial paper on this approach, its value stems from the foundation it lays for a new research direction upon which subsequent work can be built, as opposed to the current results themselves.

For this paper, we made a deliberate decision to keep the model simple, since non-essential practically motivated enhancements are less grounded in theory, may obfuscate the key underlying idea and could impart the impression that they are critical to making the approach work in practice.

The fact that our method is able to generate more plausible samples on CIFAR-10 than other methods at similar stages of development, such as the initial versions of GAN BID21 and PixelRNN (van den Oord et al., 2016) , despite the minimal sophistication of our method and architecture, shows the promise of the approach.

Later iterations of other methods incorporate additional supervision in the form of pretrained weights and/or make task-specific modifications to the architecture and training procedure, which were critical to achieving state-of-the-art sample quality.

We do believe the question of how the architecture should be refined in the context of our method to take advantage of task-specific insights is an important one, and is an area ripe for future exploration.

In this section, we consider and address some possible concerns about our method.

It has been suggested BID29 that maximizing likelihood leads to poor sample quality because when the model is underspecified, it will try to cover all modes of the empirical data distribution and therefore assign high density to regions with few data examples.

There is also empirical evidence BID23 for a negative correlation between sample quality and log likelihood, suggesting an inherent trade-off between maximizing likelihood and achieving good sample quality.

A popular solution is to minimize reverse KL-divergence instead, which trades off recall for pre-cision.

This is an imperfect solution, as the ultimate goal is to model all the modes and generate high-quality samples.

Note that this apparent trade-off exists that the model capacity is assumed to be fixed.

We argue that a more promising approach would be to increase the capacity of the model, so that it is less underspecified.

As the model capacity increases, avoiding mode dropping becomes more important, because otherwise there will not be enough training data to fit the larger number of parameters to.

This is precisely a setting appropriate for maximum likelihood.

As a result, it is possible that a combination of increasing the model capacity and maximum likelihood training can achieve good precision and recall simultaneously.

When the model has infinite capacity, minimizing distance from data examples to their nearest samples will lead to a model distribution that memorizes data examples.

The same is true if we maximize likelihood.

Likewise, minimizing any divergence measure will lead to memorization of data examples, since the minimum divergence is zero and by definition, this can only happen if the model distribution is the same as the empirical data distribution, whose support is confined to the set of data examples.

This implies that whenever we have a finite number of data examples, any method that learns a model with infinite capacity will memorize the data examples and will hence overfit.

To get around this, most methods learn a parametric model with finite capacity.

In the parametric setting, the minimum divergence is not necessarily zero; the same is true for the minimum distance from data examples to their nearest samples.

Therefore, the optimum of these objective functions is not necessarily a model distribution that memorizes data examples, and so overfitting will not necessarily occur.

Arjovsky et al. (2017) observes that the data distribution and the model distribution are supported on low-dimensional manifolds and so they are unlikely to have a non-negligible intersection.

They point out D KL (p data p θ ) would be infinite in this case, or equivalently, the likelihood would be zero.

While this does not invalidate the theoretical soundness of maximum likelihood, since the maximum of a non-negative function that is zero almost everywhere is still well-defined, it does cause a lot of practical issues for gradient-based learning, as the gradient is zero almost everywhere.

This is believed to be one reason that models like variational autoencoders BID32 BID45 use a Gaussian distribution with high variance for the conditional likelihood/observation model rather than a distribution close to the Dirac delta, so that the support of the model distribution is broadened to cover all the data examples .This issue does not affect our method, as our loss function is different from the log-likelihood function, even though their optima are the same (under some conditions).

As the result, the gradients of our loss function are different from those of log-likelihood.

When the supports of the data distribution and the model distribution do not overlap, each data example is likely far away from its nearest sample and so the gradient is large.

Moreover, the farther the data examples are from the samples, the larger the gradient gets.

Therefore, even when the gradient of log-likelihood can be tractably computed, there may be situations when the proposed method would work better than maximizing likelihood directly.

We presented a simple and versatile method for parameter estimation when the form of the likelihood is unknown.

The method works by drawing samples from the model, finding the nearest sample to every data example and adjusting the parameters of the model so that it is closer to the data example.

We showed that performing this procedure is equivalent to maximizing likelihood under some conditions.

The proposed method can capture the full diversity of the data and avoids common issues like mode collapse, vanishing gradients and training instability.

The method combined with vanilla model architectures is able to achieve encouraging results on MNIST, TFD and CIFAR-10.

Before proving the main result, we first prove the following intermediate results: • There is a bounded set S ⊆ Ω such that bd(S) ⊆ Ω, θ * ∈ S and ∀f i , ∀θ ∈ Ω \ S, f i (θ) > f i (θ * ), where bd(S) denotes the boundary of S. DISPLAYFORM0 • DISPLAYFORM1 . . .

DISPLAYFORM2 Proof.

Let S ⊆ Ω be the bounded set such that bd(S) ⊆ Ω, θ * ∈ S and ∀f i , ∀θ ∈ Ω \ S, f i (θ) > f i (θ * ).

Consider the closure of S := S ∪ bd(S), denoted asS. Because S ⊆ Ω and bd(S) ⊆ Ω, S ⊆ Ω. Since S is bounded,S is bounded.

BecauseS ⊆ Ω ⊆ R d and is closed and bounded, it is compact.

is differentiable on Ω and hence continuous on Ω. By the compactness ofS and the continuity of DISPLAYFORM0 since Φ is strictly increasing.

Because Φ (·) > 0, w i > 0 and so DISPLAYFORM1 At the same time, since θ * ∈ S ⊂S, by definition ofθ, DISPLAYFORM2 .

Combining these two facts yields DISPLAYFORM3 Since the inequality is strict, this implies that θ /∈ Ω \ S, and soθ DISPLAYFORM4 In addition, becauseθ is the minimizer of DISPLAYFORM5 On the other hand, since Φ is differentiable on V and f i (θ) ∈ V for all θ ∈ Ω, Φ (f i (θ)) exists for all θ ∈ Ω. So, DISPLAYFORM6 Combining this with the fact that θ * is the minimizer of DISPLAYFORM7 Because ∀θ ∈ Ω, if θ = θ * , ∃j ∈ Sinceθ is a critical point on Ω, we can conclude that θ * =θ, and so θ * is a minimizer of N i=1 w i Φ(f i (·)) on Ω. Since any other minimizer must be a critical point and θ * is the only critical point, θ * is the unique minimizer.

So, arg min θ∈Ω Let > 0 be arbitrary.

Since p(·) is continuous at x 0 , by definition, ∀˜ > 0 ∃δ > 0 such that ∀u ∈ B x0 (δ), |p(u) − p(x 0 )| <˜ .

Letδ > 0 be such that ∀u ∈ B x0 (δ), p(x 0 ) − < p(u) < p(x 0 ) + .

We choose δ =δ.

Let 0 <h < δ be arbitrary.

Since p(x 0 )− < p(u) < p(x 0 )+ ∀u ∈ B x0 (δ) = B x0 (δ) ⊃ B x0 (h), DISPLAYFORM8

@highlight

We develop a new likelihood-free parameter estimation method that is equivalent to maximum likelihood under some conditions