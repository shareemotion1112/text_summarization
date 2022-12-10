The Wasserstein probability metric has received much attention from the machine learning community.

Unlike the Kullback-Leibler divergence, which strictly measures change in probability, the Wasserstein metric reflects the underlying geometry between outcomes.

The value of being sensitive to this geometry has been demonstrated, among others, in ordinal regression and generative modelling, and most recently in reinforcement learning.

In this paper we describe three natural properties of probability divergences that we believe reflect requirements from machine learning: sum invariance, scale sensitivity, and unbiased sample gradients.

The Wasserstein metric possesses the first two properties but, unlike the Kullback-Leibler divergence, does not possess the third.

We provide empirical evidence suggesting this is a serious issue in practice.

Leveraging insights from probabilistic forecasting we propose an alternative to the Wasserstein metric, the Cramér distance.

We show that the Cramér distance possesses all three desired properties, combining the best of the Wasserstein and Kullback-Leibler divergences.

We give empirical results on a number of domains comparing these three divergences.

To illustrate the practical relevance of the Cramér distance we design a new algorithm, the Cramér Generative Adversarial Network (GAN), and show that it has a number of desirable properties over the related Wasserstein GAN.

In machine learning, the Kullback-Leibler (KL) divergence is perhaps the most common way of assessing how well a probabilistic model explains observed data.

Among the reasons for its popularity is that it is directly related to maximum likelihood estimation and is easily optimized.

However, the KL divergence suffers from a significant limitation: it does not take into account how close two outcomes might be, but only their relative probability.

This closeness can matter a great deal: in image modelling, for example, perceptual similarity is key (Rubner et al., 2000; BID13 .

Put another way, the KL divergence cannot reward a model that "gets it almost right".To address this limitation, researchers have turned to the Wasserstein metric, which does incorporate the underlying geometry between outcomes.

The Wasserstein metric can be applied to distributions with non-overlapping supports, and has good out-of-sample performance BID11 .

Yet, practical applications of the Wasserstein distance, especially in deep learning, remain tentative.

In this paper we provide a clue as to why that might be: estimating the Wasserstein metric from samples yields biased gradients, and may actually lead to the wrong minimum.

This precludes using stochastic gradient descent (SGD) and SGD-like methods, whose fundamental mode of operation is sample-based, when optimizing for this metric.

As a replacement we propose the Cramér distance (Székely, 2002; Rizzo & Székely, 2016) , also known as the continuous ranked probability score in the probabilistic forecasting literature BID14 .

The Cramér distance, like the Wasserstein metric, respects the underlying geometry but also has unbiased sample gradients.

To underscore our theoretical findings, we demonstrate a significant quantitative difference between the two metrics when employed in typical machine learning scenarios: categorical distribution estimation, regression, and finally image generation.

In the latter case, we use a multivariate generalization of the Cramér distance, the energy distance (Székely, 2002) , itself an instantiation of the MMD family of metrics BID16 .

In this section we provide the notation to mathematically distinguish the Wasserstein metric (and later, the Cramér distance) from the Kullback-Leibler divergence and probability distances such as the total variation.

Let P be a probability distribution over R. When P is continuous, we will assume it has density µ P .

The expectation of a function f : R → R with respect to P is E x∼P f (x) := ∞ −∞ f (x)P (dx) = f (x)µ P (x)dx if P is continuous, and f (x)P (x) if P is discrete.

We will suppose all expectations and integrals under consideration are finite.

We will often associate P to a random variable X, such that for a subset of the reals A ⊆ R, we have Pr{X ∈ A} = P (A).

The (cumulative) distribution function of P is then DISPLAYFORM0 Finally, the inverse distribution function of P , defined over the interval (0, 1], is DISPLAYFORM1 P (u) := inf{x : F P (x) = u}.

Consider two probability distributions P and Q over R. A divergence d is a mapping (P, Q) → R + with d(P, Q) = 0 if and only if P = Q almost everywhere.

A popular choice is the KullbackLeibler (KL) divergence DISPLAYFORM0 with KL(P Q) = ∞ if P is not absolutely continuous w.r.t.

Q. The KL divergence, also called relative entropy, measures the amount of information needed to encode the change in probability from Q to P (Cover & BID6 .A probability metric is a divergence which is also symmetric (d(P, Q) = d(Q, P )) and respects the triangle inequality: for any distribution R, d(P, Q) ≤ d(P, R) + d(R, Q).

We will use the term probability distance to mean a symmetric divergence satisfying the relaxed triangle inequality d(P, Q) ≤ c [d(P, R) + d(R, Q)] for some c ≥ 1.We will first study the p-Wasserstein metrics w p BID9 .

For 1 ≤ p < ∞, a practical definition is through the inverse distribution functions of P and Q: DISPLAYFORM1 We will sometimes find it convenient to deal with the p th power of the metric, which we will denote by w p p ; note that w p p is not a metric proper, but is a probability distance.

We will be chiefly concerned with the 1-Wasserstein metric, which is most commonly used in practice.

The 1-Wasserstein metric has a dual form which is theoretically convenient and which we mention here for completeness.

Define F ∞ to be the class of 1-Lipschitz functions.

Then w 1 (P, Q) := sup DISPLAYFORM2 This is a special case of the celebrated Monge-Kantorovich duality (Rachev et al., 2013) , and is the integral probability metric (IPM) with function class F ∞ (Müller, 1997) .

We invite the curious reader to consult these two sources as a starting point on this rich topic.

As noted in the introduction, the fundamental difference between the KL divergence and the Wasserstein metric is that the latter is sensitive not only to change in probability but also to the geometry of possible outcomes.

To capture this notion we now introduce the concept of an ideal divergence.

Consider a divergence d, and for two random variables X, Y with distributions P, Q write d(X, Y ) := d(P, Q).

We say that d is scale sensitive (of order β), i.e. it has property (S), if there exists a β > 0 such that for all X, Y , and a real value c > 0, DISPLAYFORM0 A divergence d has property (I), i.e. it is sum invariant, if whenever A is independent from X, Y DISPLAYFORM1 Following Zolotarev (1976) , an ideal divergence d is one that possesses both (S) and (I).

We can illustrate the sensitivity of ideal divergences to the value of outcomes by considering Dirac functions δ x at different values of x. If d is scale sensitive of order β = 1 then the divergence d(δ 0 , δ 1/2 ) can be no more than half the divergence d(δ 0 , δ 1 ).

If d is sum invariant, then the divergence of δ 0 to δ 1 is equal to the divergence of the same distributions shifted by a constant c, i.e. of δ c to δ 1+c .

As a concrete example of the importance of these properties, BID2 recently demonstrated the importance of ideal metrics in reinforcement learning, specifically their role in providing the contraction property of the distributional Bellman operator.

In particular, the contraction modulus is γ β , where γ ∈ [0, 1) is a discount factor and β is the scale sensitivity order.

In machine learning we often view the divergence d as a loss function.

Specifically, let Q θ be some distribution parametrized by θ, and consider the loss θ → d(P, Q θ ).

We are interested in minimizing this loss, that is finding θ * := arg min θ d(P, Q θ ).

We now describe a third property based on this loss, which we call unbiased sample gradients.

Let X m := X 1 , X 2 , . . .

, X m be independent samples from P and define the empirical distribution DISPLAYFORM0 δ Xi (note thatP m is a random quantity).

From this, define the sample loss θ → d(P m , Q θ ).

We say that d has unbiased sample gradients when the expected gradient of the sample loss equals the gradient of the true loss for all P and m: DISPLAYFORM1 The notion of unbiased sample gradients is ubiquitous in machine learning and in particular in deep learning.

Specifically, if a divergence d does not possess (U) then minimizing it with stochastic gradient descent may not converge, or it may converge to the wrong minimum.

Conversely, if d possesses (U) then we can guarantee that the distribution which minimizes the expected sample loss is Q = P .

In the probabilistic forecasting literature, this makes d a proper scoring rule BID14 .We now characterize the KL divergence and the Wasserstein metric in terms of these properties.

As it turns out, neither simultaneously possesses both (U) and (S).

Proposition 1.

The KL divergence has unbiased sample gradients (U), but is not scale sensitive (S).Proposition 2.

The Wasserstein metric is ideal (I, S), but does not have unbiased sample gradients.

We will provide a proof of the bias in the sample Wasserstein gradients just below; the proof of the rest and later results are provided in the appendix.

In this section we give theoretical evidence of serious issues with gradients of the sample Wasserstein loss.

We will consider a simple Bernoulli distribution P with parameter θ * ∈ (0, 1), which we would like to estimate from samples.

Our model is Q θ , a Bernoulli distribution with parameter θ.

We study the behaviour of stochastic gradient descent w.r.t.

θ over the sample Wasserstein loss, specifically using the p th power of the metric (as is commonly done to avoid fractional exponents).

Our results build on the example given by BID2 , whose result is for θ * = 1 2 and m = 1.

We now show that even in this simplest of settings, this estimate is biased, and we exhibit a lower bound on the bias for any value of m. Hence the Wasserstein metric does not have property (U).

More worrisome still, we show that the minimum of the expected empirical Wasserstein loss θ → E Xm w p p (P m , Q θ ) is not the minimum of the Wasserstein loss θ → w p p (P, Q θ ).

We then conclude that minimizing the sample Wasserstein loss by stochastic gradient descent may in general fail to converge to the minimum of the true loss.

Theorem 1.

LetP m = 1 m m i=1 δ Xi be the empirical distribution derived from m independent samples X m = X 1 , . . .

, X m drawn from a Bernoulli distribution P .

Then for all 1 ≤ p < ∞,• Non-vanishing minimax bias of the sample gradient.

For any m ≥ 1 there exists a pair of Bernoulli distributions P , Q θ for which DISPLAYFORM0 • Wrong minimum of the sample Wasserstein loss.

The minimum of the expected sample lossθ = arg min θ E Xm w p p (P m , Q θ ) is in general different from the minimum of the true Wasserstein loss θ * = arg min θ w p p (P, Q θ ).•

Deterministic solutions to stochastic problems.

For any m ≥ 1, there exists a distribution P with nonzero entropy whose sample loss is minimized by a distribution Qθ with zero entropy.

Taken as a whole, Theorem 1 states that we cannot in general minimize the Wasserstein loss using naive stochastic gradient descent methods.

Although our result does not imply the lack of a stochastic optimization procedure for this loss, 2 we believe our result to be cause for concern.

We leave as an open question whether an unbiased optimization procedure exists and is practical.

Our result is surprising given the prevalence of the Wasserstein metric in empirical studies.

We hypothesize that this bias exists in published results and is an underlying cause of learning instability and poor convergence often remedied to by heuristic means.

For example, BID12 and Montavon et al. (2016) reported the need for a mixed KL-Wasserstein loss to obtain good empirical results, with the latter explicitly discussing the issue of wrong minima when using Wasserstein gradients.

We remark that our result also applies to the dual (2), since the losses are the same.

This dual was recently considered by as an alternative loss to the primal (1).

The adversarial procedure proposed by the authors is a two time-scale process which first maximizes (2) w.r.t f ∈ F ∞ using m samples, then takes a single stochastic gradient step w.r.t.

θ.

Interestingly, this approach does seem to provide unbiased gradients as m → ∞. However, the cost of a single gradient is now significantly higher, and for a fixed m we conjecture that the minimax bias remains.

We are now ready to describe an alternative to the Wasserstein metric, the Cramér distance (Székely, 2002; Rizzo & Székely, 2016) .

As we shall see, the Cramér distance has the same appealing properties as the Wasserstein metric, but also provides us with unbiased sample gradients.

As a result, we believe this underappreciated distance is an appealing alternative to the Wasserstein metric for many machine learning applications.

Recall that for two distributions P and Q over R, their (cumulative) distribution functions are respectively F P and F Q .

The (squared) Cramér distance between P and Q is FORMULA3 is significantly more distant than the two others (0, 1).

Rest.

Distributions minimizing the divergences discussed in this paper, under the constraint Q(1) = Q(10).

Both Wasserstein metric and Cramér distance underemphasize Q(0) to better match the cumulative distribution function.

The sample Wasserstein loss result is for m = 1.

DISPLAYFORM0 The Cramér distance is a Bregman divergence, and is a member of the l p family of divergences DISPLAYFORM1 The l p and Wasserstein metrics are identical at p = 1, but are otherwise distinct.

As the following theorem shows, the Cramér distance possesses unique properties.

Theorem 2.

Consider two random variables X, Y , a random variable A independent of X, Y , and a real value c > 0.

Then for 1 ≤ p ≤ ∞, DISPLAYFORM2 Furthermore, the Cramér distance has unbiased sample gradients.

That is, given X m := X 1 , . . .

, X m drawn from a distribution P , the empirical distributionP DISPLAYFORM3 and of all the l p distances, only the Cramér (p = 2) has this property.

We conclude that the Cramér distance enjoys both the benefits of the Wasserstein metric and the SGD-friendliness of the KL divergence.

Given the close similarity of the Wasserstein and l p metrics, it is truly remarkable that only the Cramér distance has unbiased sample gradients.

To illustrate how the Cramér distance compares to the 1-Wasserstein metric, we consider modelling the discrete distribution P depicted in FIG0 (left).

Since the trade-offs between metrics are only apparent when using an approximate model, we use an underparametrized discrete distribution Q θ which assigns the same probability to x = 1 and x = 10.

That is, DISPLAYFORM0 Figure 1 depicts the distributions minimizing the various divergences under this parametrization.

In particular, the Cramér solution is relatively close to the 1-Wasserstein solution.

Furthermore, the minimizer of the sample Wasserstein loss (m = 1) clearly provides a bad solution (most of the mass is on 0).

Note that, as implied by Theorem 1, the bias shown here would arise even if the distribution could be exactly represented.

To further show the impact of the Wasserstein bias we used gradient descent to minimize either the true or sample losses with a fixed step-size (α = 0.001).

In the stochastic setting, at each step we construct the empirical distributionP m from m samples (a Dirac when m = 1), and take a gradient step.

We measure the performance of each method in terms of the true 1-Wasserstein loss.

small sample sizes stochastic gradient descent fails to find reasonable solutions, and for m = 1 even converges to a solution worse than the KL minimizer.

This small experiment highlights the cost incurred from minimizing the sample Wasserstein loss, and shows that increasing the sample size may not be sufficient to guarantee good behaviour.

We next trained a neural network in an ordinal regression task using either of the three divergences.

The task we consider is the Year Prediction MSD dataset (Lichman, 2013) .

In this task, the model must predict the year a song was written (from 1922 to 2011) given a 90-dimensional feature representation.

In our setting, this prediction takes the form of a probability distribution.

We measure each method's performance on the test set ( The results show that minimizing the sample Wasserstein loss results in significantly worse performance.

By contrast, minimizing the Cramér distance yields the lowest RMSE and Wasserstein loss, confirming the practical importance of having unbiased sample gradients.

Naturally, minimizing for one loss trades off performance with respect to the others, and minimizing the Cramér distance results in slightly higher negative log likelihood than when minimizing the KL divergence FIG6 in appendix).

We conclude that, in the context of ordinal regression where outcome similarity plays an important role, the Cramér distance should be preferred over either KL or the Wasserstein metric.

The energy distance (Székely, 2002) is a natural extension of the Cramér distance to the multivariate case.

Let P, Q be probability distributions over R d and let X, X and Y, Y be independent random variables distributed according to P and Q, respectively.

The energy distance (sometimes called the squared energy distance, see e.g. Rizzo & Székely, 2016 ) is DISPLAYFORM0 Székely showed that, in the univariate case, l 2 2 (P, Q) = 1 2 E(P, Q).

Interestingly enough, the energy distance can also be written in terms of a difference of expectations.

For DISPLAYFORM1 we find that DISPLAYFORM2 The energy distance is closely related to the distances known as maximum mean discrepancies (MMDs; BID16 ; in particular, Sejdinovic et al. (2013) showed that the energy distance is equivalent to the squared MMD with kernel k(x, y) = x 2 + y 2 − x − y 2 .

Finally, we remark that E also possesses properties (I), (S), and (U) (proof in the appendix).

Algorithm 1: Cramér GAN Losses.

Parameter.

Gradient penalty coefficient λ.

DISPLAYFORM3 Interpolate real and generated samples: DISPLAYFORM4 ) 2 Sample surrogate generator loss (13) and critic loss: DISPLAYFORM5 Figure 4: Approximate Wasserstein distances between CelebA test set and the generators.

N u is the number critic updates per generator update.

We now consider the Generative Adversarial Networks (GAN) framework BID15 , in particular issues arising in the Wasserstein GAN , and propose a better GAN based on the Cramér distance.

A GAN is composed of a generative model Q (in our experiments, over images), called the generator, a target source P , and a trainable loss function called a discriminator or critic.

GANs are particularly interesting because we can establish a direct comparison between the two distances.

Our choice of name reflects this fact, and we prefer Cramér GAN to the perhaps more technically correct, but less palatable Energy Distance GAN.

In theory, the Wasserstein GAN algorithm requires training the critic until convergence, but this is rarely achievable: we would require a critic that is a very powerful network to approximate the Wasserstein distance well BID1 .

Simultaneously, training this critic to convergence would overfit the empirical distribution of the training set, which is undesirable.

Our proposed loss function allows for useful learning with imperfect critics by combining the energy distance with a transformation function h : R d → R k , where d is the input dimensionality and k = 256 in our experiments.

The generator then seeks to minimize the energy distance of the transformed variables E(h(X), h(Y )), where X is a real sample and Y is a generated sample.

The critic itself seeks to maximize this same distance by changing the parameters of h, subject to a soft constraint (the gradient penalty used by BID17 .

Specifically, the critic maximizes a surrogate loss whose gradient can be estimated from a single real sample.

The Cramér GAN losses are summarized in Algorithm 1, with additional design choices detailed in Appendix C.We note that MMDs such as the energy distance have in the last year become an appealing tool for training GANs.

Among others, the squared MMD is used within Generative Moment Matching Networks (Li et al., 2015; BID10 ; BID4 trained a model to minimize the energy distance for hand pose estimation.

Our use of the tranformation h(x) reflects our anecdotal finding that the direct minimization of the energy distance over raw images does not work well (see FIG0 in appendix).

Similar findings can be found in the work of Mroueh et al. (2017) and the independently developed MMD GAN (Li et al., 2017) , which additionally uses an auto-encoder loss to make the transformation injective.

The Cramér GAN we present here complements our comparison of the Wasserstein and Cramér distance from previous sections.

At the same time, our experiments also provide novel GAN-related contributions, including the ability to perform conditional modelling using a surrogate generator loss, which lets us train the critic even when only one independent sample from P is available.

We note also that in our experiments, x − y 2 distances were more stable than distances generated by Gaussian or Laplacian kernels.

We now show that, compared to the improved Wasserstein GAN (WGAN-GP) of BID17 , the Cramér GAN leads to more stable learning and increased diversity in the generated samples.

In both cases we train generative models that predict the right half of an image given the left half; samples from unconditional models are provided in the appendix FIG0 ).

The dataset we use here is the CelebA 64 × 64 dataset (Liu et al., 2015) of celebrity faces.

Increased diversity.

In our first experiment, we compare the qualitative diversity of completed faces by showing three sample completions generated by either model given the left half of a validation set image FIG3 ).

We observe that the completions produced by WGAN-GP are almost deterministic.

Our findings echo those of Isola et al. (2016) , who observed that "the generator simply learned to ignore the noise."

By contrast, the completions produced by Cramér GAN are fairly diverse, including different hairstyles, accessories, and backgrounds.

We view this lack of diversity in WGAN-GP as undesirable given that the main requirement of a generative model is that it should provide a variety of outputs.

Theorem 1 provides a clue as to what may be happening here.

We know that minimizing the sample Wasserstein loss will find the wrong minimum.

In particular, when the target distribution has low entropy, the sample Wasserstein minimizer may actually be a deterministic distribution.

But a good generative model of images must lie in this "almost deterministic" regime, since the space of natural images makes up but a fraction of all possible pixel combinations and hence there is little perpixel entropy.

We hypothesize that the increased diversity in the Cramér GAN comes exactly from learning these almost deterministic predictions.

More stable learning.

In a second experiment, we varied the number of critic updates (N u ) per generator update.

To compare performance between the two architectures, we measured the loss computed by an independent WGAN-GP critic trained on the validation set, following a similar evaluation previously done by BID7 .

Figure 4 shows the independent Wasserstein critic distance between each generator and the test set during the course of training.

Echoing our results with the toy experiment and ordinal regression, the plot shows that when a single critic update is used, WGAN-GP performs particularly poorly.

We note that additional critic updates also improve Cramér GAN.

This indicates that it is helpful to keep adapting the h(x) transformation.

There are many situations in which the KL divergence, which is commonly used as a loss function in machine learning, is not suitable.

The desirable alternatives, as we have explored, are the divergences that are ideal and allow for unbiased estimators: they allow geometric information to be incorporated into the optimization problem; because they are scale-sensitive and sum-invariant, they possess the convergence properties we require for efficient learning; and the correctness of their sample gradients means we can deploy them in large-scale optimization problems.

Among open questions, we mention deriving an unbiased estimator that minimizes the Wasserstein distance, and variance analysis and reduction of the Cramér distance gradient estimate.

Proof (Proposition 1 and 2).

The statement regarding (U) for the KL divergence is well-known, and forms the basis of most stochastic gradient algorithms for classification.

BID5 have shown that the total variation does not have property (S); by Pinsker's inequality, it follows that the same holds for the KL divergence.

A proof of (I) and (S) for the Wasserstein metric is given by BID3 , while the lack of (U) is shown in the proof of Theorem 1.

Proof (Theorem 1).

Minimax bias: Consider P = B(θ * ), a Bernoulli distribution of parameter θ * and Q θ = B(θ) a Bernoulli of parameter θ.

The empirical distributionP m is a Bernoulli with parameterθ := 1 m m i=1 X i .

Note that with P and Q θ both Bernoulli distributions, the p th powers of the p-Wasserstein metrics are equal, i.e. w 1 (P, Q θ ) = w p p (P, Q θ ).

This gives us an easy way to prove the stronger result that all p-Wasserstein metrics have biased sample gradients.

The gradient of the loss w DISPLAYFORM0 and similarly, the gradient of the sample loss is, for θ =θ, DISPLAYFORM1 .

Notice that this estimate is biased for any m ≥ 1 since DISPLAYFORM2 which is different from g for any θ * ∈ (0, 1).

In particular for m = 1, E Pĝ = 1 − 2θ * does not depend on θ, thus a gradient descent using a one-sample gradient estimate has no chance of minimizing the Wasserstein loss as it will converge to either 1 or 0 instead of θ * .Now observe that for m ≥ 2, and any θ > DISPLAYFORM3 and therefore DISPLAYFORM4 Taking θ * = m−1 m , we find that DISPLAYFORM5 Thus for any m, there exists P = B(θ * ) and Q θ = B(θ) with θ * = m−1 m < θ < 1 such that the bias g − Eĝ is lower-bounded by a numerical constant.

Thus the minimax bias does not vanish with the number of samples m.

Notice that a similar argument holds for θ * and θ being close to 0.

In both situations where θ * is close to 0 or 1, the bias is non vanishing when |θ * − θ| is of order 1 m .

However this is even worse when θ * is away from the boundaries.

For example chosing θ * = 1 2 , we can prove that the bias is non vanishing even when |θ * − θ| is (only) of order DISPLAYFORM6 Indeed, using the anti-concentration result of Veraar (2010) (Proposition 2), we have that for a sequence Y 1 , . . .

, Y m of Rademacher random variables (i.e. +/ − 1 with equal probability), DISPLAYFORM7 This means that for samples X 1 , . . .

, X m drawn from a Bernoulli B(θ * = 1 2 ) (i.e., Y i = 2X i − 1 are Rademacher), we have thus for 1/2 = θ * < θ < θ * + 1/ √ 8m we have the following lower bound on the bias: DISPLAYFORM8 DISPLAYFORM9 Thus the bias is lower-bounded by a constant (independent of m) when θ * = 1 2 and |θ DISPLAYFORM10 Wrong minimum: From (5), we deduce that a stochastic gradient descent algorithm based on the sample Wasserstein gradient will converge to aθ such that Pr{θ <θ} = 1 2 , i.e.,θ is the median of the distribution overθ, whereas θ * is the mean of that distribution.

Sinceθ follows a (normalized) binomial distribution with parameters m and θ * , we know that the medianθ and the mean θ * do not necessarily coincide, and can actually be as far as .

It follows that the minimum of the expected sample Wasserstein loss (the fixed point of the stochastic gradient descent using the sample Wasserstein gradient) is different from the minimum of the true Wasserstein loss: DISPLAYFORM11 This is illustrated in FIG4 .Notice that the fact that the minima of these losses differ is worrisome as it means that minimizing the sample Wasserstein loss using (finite) samples will not converge to the correct solution.

Deterministic solutions: Consider the specific case where (1/2) 1/n < θ * < 1 (illustrated in the right plot of FIG4 ).

Then the expected sample gradient ∇ E[w p p (P m , Q θ * )] = Eĝ = 1−2(θ * ) n < 0 for any θ, so a gradient descent algorithm will converge to 1 instead of θ * .

Notice that a symmetric argument applies for θ * close to 0.In this simple example, minimizing the sample Wasserstein loss may lead to degenerate solutions (i.e., deterministic) when our target distributions have low (but not zero) entropy.

We provide an additional result here showing that the sample 1-Wasserstein gradient converges to the true gradient as m → ∞.Theorem 3.

Let P and Q θ be probability distributions, with Q θ parametrized by θ.

Assume that the set x ∈ X, such that F P (x) = F Q θ (x) has measure zero, and that for any x ∈ X, the map θ → F Qθ (x) is differentiable in a neighborhood V(θ) of θ with a uniformly bounded derivative (forθ ∈ V(θ) and x ∈ X).

LetP m = 1 m m i=1 δ Xi be the empirical distribution derived from m independent samples X 1 , . . .

, X m drawn from P .

Then lim m→∞ ∇w 1 (P m , Q θ ) = ∇w 1 (P, Q θ ), almost surely.

We note that the measure requirement is strictly to keep the proof simple, and does not subtract from the generality of the result.

Proof.

Let ∇ := ∇ θ .

Since p = 1 the Wasserstein distance w 1 (P, Q) measures the area between the curves defined by the distribution function of P and Q, thus w 1 (P, Q) = l 1 (P, Q) = F P (x) − F Q (x) dx and DISPLAYFORM0 Now since we have assumed that for any x ∈ X, the map θ → F Q θ (x) is differentiable in a neighborhood V(θ) of θ and its derivative is uniformly (over V(θ) and x) bounded by M , we have DISPLAYFORM1 Thus the dominated convergence theorem applies and DISPLAYFORM2 since we have assumed that the set of x ∈ X such that F P (x) = F Q θ (x) has measure zero.

Now, using the same argument for w 1 (P m , Q θ ) we deduce that DISPLAYFORM3 Let us decompose this integral over X as the sum of two integrals, one over X \ Ω m and the other one over Ω m , where DISPLAYFORM4 Now from the strong law of large numbers, we have that for any x, the empirical cumulative distribution function FP m (x) converges to the cumulative distribution F P (x) almost surely.

We deduce that Ω m converges to the set x, F P (x) = F Q θ (x) which has measure zero, thus |Ω m | → 0 and DISPLAYFORM5 Now, since |∇F Q θ (x)| ≤ M , we can use once more the dominated convergence theorem to deduce that DISPLAYFORM6 The following lemma will be useful in proving that the Cramér distance has property (U).Lemma 1.

Let X m := X 1 , . . .

, X m be independent samples from P , and letP m := DISPLAYFORM7 Proof.

Because the X i 's are independent, DISPLAYFORM8 Now, taking the expectation w.r.t.

DISPLAYFORM9 since the X i are identically distributed according to P .

Like the Wasserstein metrics, the l p metrics have dual forms as integral probability metrics (see BID8 , for a proof): DISPLAYFORM0 where F q := {f : f is absolutely continuous, df dx q ≤ 1} and q is the conjugate exponent of p, i.e. p −1 + q −1 = 1.

3 We will use this dual form below.

We will prove that l p has properties (I) and (S) for p ∈ [1, ∞); the case p = ∞ follows by a similar argument.

Begin by observing that DISPLAYFORM1 Then we may rewrite l p p (cX, cY ) as DISPLAYFORM2 where (a) uses a change of variables z = x/c.

Taking both sides to the power 1/p proves that the l p metric possesses property (S) of order 1/p.

For (I), we use the IPM formulation (6): DISPLAYFORM3 where (a) is by independence of A and X, Y , and (b) is by Jensen's inequality.

Next, recall that F q is the set of absolutely continuous functions whose derivative has bounded L q norm.

Hence if f ∈ F q , then also for all a the translate g a (x) := f (x + a) is also in F q .

Therefore, DISPLAYFORM4

Here we make use of the introductory requirement that "all expectations under consideration are finite."

Specifically, we require that the mean under P , E x∼P [x] , is well-defined and finite, and similarly for Q θ .

In this case, DISPLAYFORM0 This mild requirement guarantees that the tails of the distribution function F P are light enough to avoid infinite Cramér distances and expected gradients (a similar condition was set by BID8 ).

Now, by definition, DISPLAYFORM1 , where (a) follows from the hypothesis (7) (the convergence of the squares follows from the convergence of the ordinary values), (b) follows from Lemma 1 and (c) follows from Fubini's theorem, again invoking (7).Finally, we prove that of all the l p p distances (1 ≤ p ≤ ∞) only the Cramér distance, l 2 2 , has the (U) property.

Without loss of generality, let us suppose P is not a Dirac, and further suppose that for any X m ∼ P , DISPLAYFORM2 everywhere.

For example, when Q θ has bounded support we can take P to be a sufficiently translated version of Q θ , such that the two distributions' supports do not overlap.

We have already established that the 1-Wasserstein does not have the (U) property, and is equivalent to l p p for p = 1.

We will thus assume that p > 1, and also that p < ∞, the latter being recovered through standard limit arguments.

Begin with the gradient for l p p (P, Q θ ), DISPLAYFORM3 for φ p (z) = z p−1 ; in (a) we used the same argument as in Theorem 3.

Now, φ p is convex on [0, ∞) when p ≥ 2 and concave on the same interval when 1 < p < 2.

From Jensen's inequality we know that for a convex (concave) function φ and a random variable Z, E φ(Z) is greater than (less than) or equal to φ(E Z), with equality if and only if φ is linear or Z is deterministic.

By our first assumption we have ruled out the latter.

By our second assumption DISPLAYFORM4 , we can apply Jensen's inequality at every x to deduce that DISPLAYFORM5 We conclude that of the l p p distances, only the Cramér distance has unbiased sample gradients.

Proposition 3.

The energy distance E(P, Q) has properties (I), (S), and (U).Proof.

As before, write E(X, Y ) := E(P, Q).

Recall that DISPLAYFORM6 Consider a random variable A independent of X and Y .

First, we want to prove property (I): DISPLAYFORM7 We will use Proposition 2 from Székely & Rizzo (2013) to express the energy distance in terms of characteristic functions φ X , φ Y of d-dimensional random variables X and Y : DISPLAYFORM8 .

The proof then uses properties of characteristic functions (|φ A (t)| ≤ 1 and φ A+X (t) = φ A (t)φ X (t) for independent variables A and X) to show: DISPLAYFORM0 This proves (I).

Next, consider a real value c > 0.

We have DISPLAYFORM1 This proves (S).

Finally, suppose that Y is distributed according to Q θ parametrized by θ.

Let X m = X 1 , . . .

, X m be drawn from P , and letP m := 1 m m i=1 δ Xi .

LetX be the random variable distributed according toP m , andX an independent copy ofX. Then DISPLAYFORM2 The gradient of the true loss w.r.t.

θ is DISPLAYFORM3 Now, taking the gradient of the sample loss w.r.t.

θ, DISPLAYFORM4 Since the second terms of the gradients match, all we need to show is that the first terms are equal, in expectation.

Assuming that ∇ θ and the expectation over X m commute, we write DISPLAYFORM5 by independence of X and Y .

But now we know that the expected empirical distribution is P , that is DISPLAYFORM6 It follows that the first terms of FORMULA61 and FORMULA62 are also equal, in expectation w.r.t.

X m .

Hence we conclude that the energy distance has property (U), that is DISPLAYFORM7 B COMPARISON WITH THE WASSERSTEIN DISTANCE FIG2 (left) provides learning curves for the toy experiment described in Section 4.2.

We compare the different losses on an ordinal regression task using the Year Prediction MSD dataset from (Lichman, 2013) .

The task is to predict the year of a song (taking on values from 1922 to 2011), from 90-dimensional feature representation of the song.

4 Previous work has used this dataset for benchmarking regression performance BID18 , treating the target as a continuous value.

Following Hernández-Lobato & Adams FORMULA3 , we train a network with a single hidden layer with 100 units and ReLU non-linearity, using SGD with 40 passes through the training data, using the standard train-test split for this dataset (Lichman, 2013) .

Unlike BID18 , the network outputs a probability distribution over the years (90 possible years from 1922-2011).We train models using either the 1-Wasserstein loss, the Cramér loss, or the KL loss, the latter of which reduces the ordinal regression problem to a classification problem.

In all cases, we compare performance for three different minibatch sizes, i.e. the number of input-target pairs per gradient step.

Note that the minibatch size only affects the gradient estimation, but has otherwise no direct relation to the number of samples m previously discussed, since each sample corresponds to a different input vector.

We report results as a function of number of passes over the training data so that our results are comparable with previous work, but note that smaller batch sizes get more updates.

The results are shown in FIG2 .

Training using the Cramér loss results in the lowest root mean squared error (RMSE) and the final RMSE value of 8.89 is comparable to regression BID18 which directly optimizes for MSE.

We further observe that minimizing the Wasserstein loss trains relatively slowly and leads to significantly higher KL loss.

Interestingly, larger minibatch sizes do seem to improve the performance of the Wasserstein-based method somewhat, suggesting that there might be some beneficial bias reduction from combining similar inputs.

By contrast, using with the Cramér loss trains significantly faster and is more robust to choice of minibatch size.

As additional supporting material, we provide here the results of experiments on learning a probabilistic generative model on images using either the 1-Wasserstein, Cramér, or KL loss.

We trained a PixelCNN model (Van den Oord et al., 2016) on the CelebA 32x32 dataset (Liu et al., 2015) , which is constituted of 202,599 images of celebrity faces.

At a high level, probabilistic image modelling involves defining a joint probability Q θ over the space of images.

PixelCNN forms this joint probability autoregressively, by predicting each pixel using a histogram distribution conditional on a probability-respecting subset of its neighbours.

This kind of modelling task is a perfect setting to study Wasserstein-type losses, as there is a natural ordering on pixel intensities.

This is also a setting in which full distributions are almost never available, because each prediction is conditioned on very different context; and hence we require a loss that can be optimized from single samples.

Here the true losses are not available.

Instead we report the sample Wasserstein loss, which is an upper bounds on the true loss Bellemare et al. (proof is provided by 2017).

For the KL divergence we report the cross-entropy loss, as is typically done; the KL divergence itself corresponds to the expected cross-entropy loss minus the real distribution's (unknown) entropy.

Figure 8 shows, as in the toy example, that minimizing the Wasserstein distance by means of stochastic gradient fails.

The Cramér distance, on the other hand, is as easily minimized as the KL and in fact achieves lower Wasserstein and Cramér loss.

We note that the resulting KL loss is higher than when directly minimizing the KL, reflecting the very real trade-off of using one loss over another.

We conclude that in the context of learning an autoregressive image model, the Cramér should be preferred to the Wasserstein metric.

C CRAMÉR GAN

Our critic has a special form: DISPLAYFORM0 where Q is the generator and P is the target distribution.

The critic has trainable parameters only inside the deep network used for the transformation h. From (4), we define the generator loss to be DISPLAYFORM1 as in Wasserstein GAN, except that no max f operator is present and we can obtain unbiased sample gradients.

At the same time, to provide helpful gradients for the generator, we train the transformation h to maximize the generator loss.

Concretely, the critic seeks to maximize the generator loss while minimizing a gradient penalty: DISPLAYFORM2 where GP is the gradient penalty from the original WGAN-GP algorithm BID17 (the penalty is given in Algorithm 1).

The gradient penalty bounds the critic's outputs without using a saturating function.

We chose λ = 10 from a short parameter sweep.

Our training is otherwise similar to the improved training of Wasserstein GAN BID17 .In the next two sections, we describe how to practically compute gradients of these losses with respect to the generator and transformation parameters, respectively.

Recall that the energy distance is: DISPLAYFORM0 If Y is generated from the standard normal noise Z ∼ N (0, 1) by a differentiable generator Y = G(Z) and the generator has an integrable gradient, we can use the reparametrization trick (Kingma & Welling, 2014) to compute the gradient with respect to the generator parameters: DISPLAYFORM1 We see that we only need one real sample X to estimate the gradient, because the X − X term does not depend on the generator parameters.

This allows us to define a generator loss usable for situations with only one real sample (e.g., for conditional modeling): DISPLAYFORM2 C.3 GRADIENT ESTIMATES FOR THE TRANSFORMATION As shown in the previous section, we can obtain an unbiased gradient estimate of the generator loss (12) from three samples: two from the generator, and one from the target distribution.

However, to estimate the gradient of the Cramér GAN loss with respect to the transformation parameters we need four independent samples: two from the generator and two from the target distribution.

In many circumstances, for example when learning conditional densities, we do not have access to two independent target samples.

We will instead define a surrogate objective for the critic.

The surrogate critic will have the following form: DISPLAYFORM3 Figure 10: Left.

Generated images from a generator trained to minimize the energy distance of raw images, E(X, Y ).

Right.

Generated images if minimizing the Cramér GAN loss, E(h(X), h(Y )).

Both generators had the same DCGAN architecture (Radford et al., 2015) .which we use to define a surrogate loss L s (X, Y ) similar to (10): DISPLAYFORM4 The surrogate loss emulates an integral probability metric (IPM) (Müller, 1997) and can be used to train the critic.

The maximization of this loss will force E h(X)−h(Y ) 2 and E h(Y )−h(Y ) 2 to be informative about the underlying distributions.

The generator can be then trained to minimize the energy distanceL g (12) of the transformed variables.

It is also possible to obtain training more similar to Wasserstein GAN by training the generator to minimize the surrogate loss (13).

We recommend trying both possibilities, because they were both stable and produced diverse conditional samples.

The whole training procedure is summarized as Algorithm 1.Finally, when estimating the losses in Algorithm 1, we use two independent samples x g , x g from the generator.

However, in constructing the surrogate lossL s , an asymmetry arises.

We reduce variance by averaging the two lossesL s (x g , x g ) andL s (x g , x g ).

The generator architecture is the U-Net (Ronneberger et al., 2015) previously used for Image-toImage translation (Isola et al., 2016) .

We used no batch normalization and no dropout in the generator and in the critic.

The network conditioned on the left half of the image and on extra 12 channels with Gaussian noise.

We generated two independent samples for a given image to compute the Cramér GAN loss.

To be computationally fair to WGAN-GP, we trained WGAN-GP with twice the minibatch size (i.e., the Cramér GAN minibatch size was 64, while the WGAN-GP minibatch size was 128).

Our h(x) transformation is a deep network with 256 outputs (more is better).

The network has the traditional deep convolutional architecture (Radford et al., 2015) .

We do not use batch normalization, as it would conflict with the gradient penalty.

We report the Inception score (Salimans et al., 2016) and the Fréchet Inception Distance (FID) (Heusel et al., 2017) in FIG0 (left), which are commonly used measures of evaluation for GANs.

These evaluation measures have the disadvantage that they are not able to detecting overfitting and account for diversity in generated conditional samples.

For example, a mixture model that overfits to the training set would get a better Inception score and FID than the trained GANs.

We propose a new evaluation for conditional GANs that uses data from the validation set and that is able to detect overfitting.

Our Inception Energy Distance (IED) measures a difference, similar to the genererator loss (12), between features of completed image and features of the corresponding real image.

An unbiased estimator of the IED is: DISPLAYFORM0 where x r is a real sample and x g , x g are two independent generated samples.

in(x) are the features for image x, and is the is the output of the pretrained Inception network 5 (Szegedy et al., 2016) , specifically the output layer pool_3:0 with 2048 features.

The pretrained Inception network allows to objectively compare different GANs.

Our performance measure is similar to the FID, but can be computed with one real sample and monitored online.

We use the Inception Energy Distance only to detect underfitting and overfitting.

FIG0 (right) shows that WGAN-GP is not minimizing IED on the training set.

WGAN-GP produces very deterministic completions and this is detected by the in(x g ) − in(x g ) 2 term in the IED.

We also see that the Cramér GAN is overfitting the training set.

The Cramér GAN is progressively learning the distribution of the training set and obtains a worse IED on the validation set.

This suggests that our optimization is able to successfully train the generator, and that with more data and regularization methods, we will be able to overcome this overfitting.

For example, future work can train on large video datasets and try to minimize the IED directly.

@highlight

The Wasserstein distance is hard to minimize with stochastic gradient descent, while the Cramer distance can be optimized easily and works just as well.

@highlight

The manuscript proposes to use the Cramer distance to act as a loss when optimizing an objective function using stochastic gradient descent because it has unbiased sample gradients.

@highlight

The contribution of the article is related to performance criteria, in particular to the Wasserstein/Mallows metric