We derive an unbiased estimator for expectations over discrete random variables based on sampling without replacement, which reduces variance as it avoids duplicate samples.

We show that our estimator can be derived as the Rao-Blackwellization of three different estimators.

Combining our estimator with REINFORCE, we obtain a policy gradient estimator and we reduce its variance using a built-in control variate which is obtained without additional model evaluations.

The resulting estimator is closely related to other gradient estimators.

Experiments with a toy problem, a categorical Variational Auto-Encoder and a structured prediction problem show that our estimator is the only estimator that is consistently among the best estimators in both high and low entropy settings.

Put replacement in your basement!

We derive the unordered set estimator: an unbiased (gradient) estimator for expectations over discrete random variables based on (unordered sets of) samples without replacement.

In particular, we consider the problem of estimating (the gradient of) the expectation of f (x) where x has a discrete distribution p over the domain D, i.e.

This expectation comes up in reinforcement learning, discrete latent variable modelling (e.g. for compression), structured prediction (e.g. for translation), hard attention and many other tasks that use models with discrete operations in their computational graphs (see e.g. Jang et al. (2016) ).

In general, x has structure (such as a sequence), but we can treat it as a 'flat' distribution, omitting the bold notation, so x has a categorical distribution over D given by p(x), x ∈ D. Typically, the distribution has parameters θ, which are learnt through gradient descent.

This requires estimating the gradient ∇ θ E x∼p θ (x) [f (x)], using a set of samples S. A gradient estimate e(S) is unbiased if

The samples S can be sampled independently or using alternatives such as stratified sampling which reduce variance to increase the speed of learning.

In this paper, we derive an unbiased gradient estimator that reduces variance by avoiding duplicate samples, i.e. by sampling S without replacement.

This is challenging as samples without replacement are dependent and have marginal distributions that are different from p(x).

We further reduce the variance by deriving a built-in control variate, which maintains the unbiasedness and does not require additional samples.

Related work.

Many algorithms for estimating gradients for discrete distributions have been proposed.

A general and widely used estimator is REINFORCE (Williams, 1992) .

Biased gradients based on a continuous relaxations of the discrete distribution (known as Gumbel-Softmax or Concrete) were jointly introduced by Jang et al. (2016) and Maddison et al. (2016) .

These can be combined with the straight through estimator (Bengio et al., 2013) if the model requires discrete samples or be used to construct control variates for REINFORCE, as in REBAR (Tucker et al., 2017) or RELAX (Grathwohl et al., 2018) .

Many other methods use control variates and other techniques to reduce the variance of REINFORCE (Paisley et al., 2012; Ranganath et al., 2014; Gregor et al., 2014; Mnih & Gregor, 2014; Gu et al., 2016; Mnih & Rezende, 2016) .

Some works rely on explicit summation of the expectation, either for the marginal distribution (Titsias & Lázaro-Gredilla, 2015) or globally summing some categories while sampling from the remainder (Liang et al., 2018; Liu et al., 2019) .

Other approaches use a finite difference approximation to the gradient (Lorberbom et al., 2018; 2019) .

Yin et al. (2019) introduced ARSM, which uses multiple model evaluations where the number adapts automatically to the uncertainty.

In the structured prediction setting, there are many algorithms for optimizing a quantity under a sequence of discrete decisions, using (weak) supervision, multiple samples (or deterministic model evaluations), or a combination both (Ranzato et al., 2016; Shen et al., 2016; He et al., 2016; Norouzi et al., 2016; Bahdanau et al., 2017; Edunov et al., 2018; Leblond et al., 2018; Negrinho et al., 2018) .

Most of these algorithms are biased and rely on pretraining using maximum likelihood or gradually transitioning from supervised to reinforcement learning.

Using Gumbel-Softmax based approaches in a sequential setting is difficult as the bias accumulates because of mixing errors (Gu et al., 2018) .

Throughout this paper, we will denote with B k an ordered sample without replacement of size k and with S k an unordered sample (of size k) from the categorical distribution p.

Restricted distribution.

When sampling without replacement, we remove the set C ⊂ D already sampled from the domain and we denote with p D\C the distribution restricted to the domain D \ C:

Ordered sample without replacement B k .

Let B k = (b 1 , ..., b k ), b i ∈ D be an ordered sample without replacement, which is generated from the distribution p as follows: first, sample b 1 ∼ p, then sample b 2 ∼ p D\{b1} , b 3 ∼ p D\{b1,b2} , etc.

i.e. elements are sampled one by one without replacement.

Using this procedure, B k can be seen as a (partial) ranking according to the PlackettLuce model (Plackett, 1975; Luce, 1959) and the probability of obtaining the vector B k is

We can also restrict B k to the domain D \ C, which means that b i ∈ C for i = 1, ..., k:

Unordered sample without replacement.

Let S k ⊆ D be an unordered sample without replacement from the distribution p, which can be generated simply by generating an ordered sample and discarding the order.

We denote elements in the sample with s ∈ S k (so without index) and we write B(S k ) as the set of all k!

permutations (orderings) B k that correspond to (could have generated) S k .

It follows that the probability for sampling S k is given by:

.

(6) The last step follows since B k ∈ B(S k ) is an ordering of S k , such that

, but in Appendix B we show how to compute it efficiently.

When sampling from the distribution restricted to D \ C, we sample S k ⊆

D \ C with probability:

The Gumbel-Top-k trick.

As an alternative to sequential sampling, we can also sample B k and S k by taking the top k of Gumbel variables (Yellott, 1977; Vieira, 2014; Kim et al., 2016) .

Following notation from Kool et al. (2019c) , we define the perturbed log-probability g φi = φ i + g i , where φ i = log p(i) and g i ∼ Gumbel(0).

Then let b 1 = arg max i∈D g φi , b 2 = arg max i∈D\{b1} g φi , etc., so B k is the top k of the perturbed log-probabilities in decreasing order.

The probability of obtaining B k using this procedure is given by equation 4, so this provides an alternative sampling method which is effectively a (non-differentiable) reparameterization of sampling without replacement.

For a differentiable reparameterization, see Grover et al. (2019) .

It follows that taking the top k perturbed log-probabilities without order, we obtain the unordered sample set S k .

This way of sampling underlies the efficient computation of p(S k ) in Appendix B.

In this section, we derive the unordered set policy gradient estimator: a low-variance, unbiased estimator of

based on an unordered sample without replacement S k .

First, we derive the generic (non-gradient) estimator for E[f (x)] as the Rao-Blackwellized version of a single sample Monte Carlo estimator (and two other estimators!).

Then we combine this estimator with REINFORCE (Williams, 1992) and we show how to reduce its variance using a built-in baseline.

A very crude but simple estimator for E[f (x)] based on the ordered sample B k is to only use the first element b 1 , which by definition is a sample from the distribution p.

We define this estimator as the single sample estimator, which is unbiased, since

Discarding all but one sample, the single sample estimator is inefficient, but we can use RaoBlackwellization (Casella & Robert, 1996) to signficantly improve it.

To this end, we consider the distribution B k |S k , which is, knowing the unordered sample S k , the conditional distribution over ordered samples B k ∈ B(S k ) that could have generated

The Rao-Blackwellized version of the single sample estimator computes the inner conditional expectation exactly.

Since B k is an ordering of S k , we have b 1 ∈ S k and we can compute this as

where, in a slight abuse of notation, P (b 1 = s|S k ) is the probability that the first sampled element b 1 takes the value s, given that the complete set of k samples is S k .

Using Bayes' Theorem we find

The step p(S k |b 1 = s) = p D\{s} (S k \ {s}) comes from analyzing sequential sampling without replacement: given that the first element sampled is s, the remaining elements have a distribution restricted to D \ {s}, so sampling S k (including s) given the first element s is equivalent to sampling the remainder S k \ {s} from the restricted distribution, which has probability p D\{s} (S k \ {s}) (see equation 7).

The unordered set estimator.

For notational convenience, we introduce the leave-one-out ratio.

Definition 1.

The leave-one-out ratio of s w.r.t.

the set S is given by R(S k , s) =

shows that the probability of sampling s first, given S k , is simply the unconditional probability multiplied by the leave-one-out ratio.

We now define the unordered set estimator as the Rao-Blackwellized version of the single-sample estimator.

Theorem 1.

The unordered set estimator, given by

is the Rao-Blackwellized version of the (unbiased!)

single sample estimator.

Proof.

The implication of this theorem is that the unordered set estimator, in explicit form given by equation 11, is an unbiased estimator of E[f (x)] since it is the Rao-Blackwellized version of the unbiased single sample estimator.

Also, as expected by taking multiple samples, it has variance equal or lower than the single sample estimator by the Rao-Blackwell Theorem (Lehmann & Scheffé, 1950) .

The unordered set estimator is also the result of Rao-Blackwellizing two other unbiased estimators: the stochastic sum-and-sample estimator and the importance-weighted estimator.

The sum-and-sample estimator.

We define as sum-and-sample estimator any estimator that relies on the identity that for any

For the derivation, see Appendix C.1 or Liang et al. (2018); Liu et al. (2019) .

In general, a sum-andsample estimator with a budget of k > 1 evaluations sums expectation terms for a set of categories C (s.t.

|C| < k) explicitly (e.g. selected by their value f (Liang et al., 2018) or probability p (Liu et al., 2019)), and uses k − |C| (down-weighted) samples from D \ C to estimate the remaining terms.

As is noted by Liu et al. (2019) , selecting C such that

is minimized guarantees to reduce variance compared to a standard minibatch of k samples (which is equivalent to setting C = ∅).

See also Fearnhead & Clifford (2003) for a discussion on selecting C optimally.

The ability to optimize C depends on whether p(c) can be computed efficiently a-priori (before sampling).

This is difficult in high-dimensional settings, e.g. sequence models which compute the probability incrementally while ancestral sampling.

An alternative is to select C stochastically (as equation 13 holds for any C), and we choose C = B k−1 to define the stochastic sum-and-sample estimator:

For simplicity, we consider the version that sums k − 1 terms here, but the following results also hold for a version that sums k − m terms and uses m samples (without replacement) (see Appendix C.3).

Sampling without replacement, it holds that

, so the unbiasedness follows from equation 13 by separating the expectation over B k into expectations over B k−1 and b k |B k−1 :

In general, a sum-and-sample estimator reduces variance if the probability mass is concentrated on the summed categories.

As typically high probability categories are sampled first, the stochastic sum-and-sample estimator sums high probability categories, similar to the estimator by Liu et al.

(2019) which we refer to as the deterministic sum-and-sample estimator.

As we show in Appendix C.2, Rao-Blackwellizing the stochastic sum-and-sample estimator also results in the unordered set estimator.

This even holds for a version that uses m samples and k−m summed terms (see Appendix C.3), which means that the unordered set estimator has equal or lower variance than the optimal (in terms of m) stochastic sum-and-sample estimator, but conveniently does not need to choose m.

The importance-weighted estimator.

The importance-weighted estimator (Vieira, 2017) is

This estimator is based on the idea of priority sampling (Duffield et al., 2007) .

It does not use the order of the sample, but assumes sampling using the Gumbel-Top-k trick and requires access to κ, the (k + 1)-th largest perturbed log-probability, which can be seen as the 'threshold' since g φs > κ ∀s ∈ S k .

q(s, a) = P (g φs > a) can be interpreted as the inclusion probability of s ∈ S k (assuming a fixed threshold a instead of a fixed sample size k).

For details and a proof of unbiasedness, see Vieira (2017) or Kool et al. (2019c) .

As the estimator has high variance, Kool et al. (2019c) resort to normalizing the importance weights, resulting in biased estimates.

Instead, we use Rao-Blackwellization to eliminate stochasticity by κ.

Again, the result is the unordered set estimator (see Appendix D.1), which thus has equal or lower variance.

Writing p θ to indicate the dependency on the model parameters θ, we can combine the unordered set estimator with REINFORCE (Williams, 1992) to obtain the unordered set policy gradient estimator.

Corollary 1.

The unordered set policy gradient estimator, given by

is an unbiased estimate of the policy gradient.

Proof.

Using REINFORCE (Williams, 1992) combined with the unordered set estimator we find:

Variance reduction using a built-in control variate.

The variance of REINFORCE can be reduced by subtracting a baseline from f .

When taking multiple samples (with replacement), a simple and effective baseline is to take the mean of other (independent!)

samples (Mnih & Rezende, 2016) .

Sampling without replacement, we can use the same idea to construct a baseline based on the other samples, but we have to correct for the fact that the samples are not independent.

Theorem 2.

The unordered set policy gradient estimator with baseline, given by

where

is the second order leave-one-out ratio, is an unbiased estimate of the policy gradient.

Proof.

See Appendix E.1.

This theorem shows how to include a built-in baseline based on dependent samples (without replacement), without introducing bias.

By having a built-in baseline, the value f (s) for sample s is compared against an estimate of its expectation E[f (s)], based on the other samples.

The difference is an estimate of the advantage (Sutton & Barto, 2018) , which is positive if the sample s is 'better' than average, causing p θ (s) to be increased (reinforced) through the sign of the gradient, and vice versa.

By sampling without replacement, the unordered set estimator forces the estimator to compare different alternatives, and reinforces the best among them.

Including the pathwise derivative.

So far, we have only considered the scenario where f does not depend on θ.

If f does depend on θ, for example in a VAE (Kingma & Welling, 2014; Rezende et al., 2014) , then we use the notation f θ and we can write the gradient (Schulman et al., 2015) as

(19) The additional second ('pathwise') term can be estimated (using the same samples) with the standard unordered set estimator.

This results in the full unordered set policy gradient estimator:

Equation 20 is straightforward to implement using an automatic differentiation library.

We can also include the baseline (as in equation 17) but we must make sure to call STOP GRADIENT (DETACH in PyTorch) on the baseline (but not on f θ (s)!).

Importantly, we should never track gradients through the leave-one-out ratio R(S k , s) which means it can be efficiently computed in pure inference mode.

We can use the unordered set estimator for any discrete distribution from which we can sample without replacement, by treating it as a univariate categorical distribution over its domain.

This includes sequence models, from which we can sample using Stochastic Beam Search (Kool et al., 2019c) , as well as multivariate categorical distributions which can also be treated as sequence models (see Section 4.2).

In the presence of continuous variables or a stochastic function f , we may separate this stochasticity from the stochasticity over the discrete distribution, as in Lorberbom et al. (2019) .

The computation of the leave-one-out ratios adds some overhead, although they can be computed efficiently, even for large k (see Appendix B).

For a moderately sized model, the costs of model evaluation and backpropagation dominate the cost of computing the estimator.

Relation to Murthy's estimator.

We found out that the 'vanilla' unordered set estimator (equation 11) is actually a special case of the estimator by Murthy (1957) , known in statistics literature for estimation of a population total Θ = i∈D y i .

Using

Murthy's estimator can be used to estimate expectations (see equation 11).

Murthy derives the estimator by 'unordering' a convex combination of Raj (1956) estimators, which, using y i = p(i)f (i), are stochastic sum-and-sample estimators in our analogy.

Murthy (1957) also provides an unbiased estimator of the variance, which may be interesting for future applications.

Since Murthy's estimator can be used with arbitrary sampling distribution, it is straightforward to derive importance-sampling versions of our estimators.

In particular, we can sample S without replacement using q(x) > 0, x ∈ D, and use equations 11, 16, 17 and 20, as long as we compute the leave-one-out ratio R(S k , s) using q.

While part of our derivation coincides with Murthy (1957), we are not aware of previous work using this estimator to estimate expectations.

Additionally, we discuss practical computation of p(S) (Appendix B), we show the relation to the importance-weighted estimator, and we provide the extension to estimating policy gradients, especially including a built-in baseline without adding bias.

Relation to the empirical risk estimator.

The empirical risk loss (Edunov et al., 2018) estimates the expectation in equation 1 by summing only a subset S of the domain, using normalized proba-

s ∈S p θ (s) .

Using this loss, the (biased) estimate of the gradient is given by

The risk estimator is similar to the unordered set policy gradient estimator, with two important differences: 1) the individual terms are normalized by the total probability mass rather than the leave-one-out ratio and 2) the gradient is computed through the normalization factor.

Intuitively, by taking the gradient through the normalization factor, samples are forced to 'compete' for probability mass, such that only the best can be reinforced.

This has the same effect as using a built-in baseline, which we prove in the following theorem.

Theorem 3.

By taking the gradient w.r.t.

the normalization factor into account, the risk estimator has a built-in baseline, which means it can be written as

This theorem highlights the similarity between the biased risk estimator and our unbiased estimator (equation 17), and suggests that their only difference is the weighting of terms.

Unfortunately, the implementation by Edunov et al. (2018) has more sources of bias (e.g. length normalization), which are not compatible with our estimator.

However, we believe that our analysis helps analyze the bias of the risk estimator and is a step towards developing unbiased estimators for structured prediction.

Relation to VIMCO.

VIMCO (Mnih & Rezende, 2016) is an estimator that uses k samples (with replacement) to optimize an objective of the form log

, which is a multi-sample stochastic lower bound in the context of variational inference.

VIMCO reduces the variance by using a local baseline for each of the k samples, based on the other k − 1 samples.

While we do not have a log term, as our goal is to optimize general E[f (x)], we adopt the idea of forming a baseline based on the other samples, and we define REINFORCE without replacement (with built-in baseline) as the estimator that computes the gradient estimate using samples with replacement

This estimator is unbiased, as Kool et al. (2019b) ).

We think of the unordered set estimator as the without-replacement version of this estimator, which weights terms by p θ (s)R(S k , s) instead of 1 k .

This puts more weight on higher probability elements to compensate for sampling without replacement.

If probabilities are small and (close to) uniform, there are (almost) no duplicate samples and the weights will be close to 1 k , so the gradient estimate of the with-and without-replacement versions are similar.

Relation to ARSM.

The ARSM (Yin et al., 2019) estimator also uses multiple evaluations of p θ and f .

It determines a number of 'pseudo-samples', from which duplicates should be removed for efficient implementation.

This can be seen as similar to sampling without replacement, and the estimator also has a built-in control variate.

Compared to ARSM, our estimator allows direct control over the computational cost (through the sample size k) and has wider applicability, for example it also applies to multivariate categorical variables with different numbers of categories per dimension.

Relation to stratified/systematic sampling.

Our estimator aims to reduce variance by changing the sampling distribution for multiple samples by sampling without replacement.

There are alternatives, such as using stratified or systematic sampling (see, e.g. Douc & Cappé (2005) ).

Both partition the domain D into k strata and take a single sample from each stratum, where systematic sampling uses common random numbers for each stratum.

In applications involving high-dimensional or structured domains, it is unclear how to partition the domain and how to sample from each partition.

Additionally, as samples are not independent, it is non-trivial to include a built-in baseline, which we find is a key component that makes our estimator perform well.

We use the code by Liu et al. (2019) to reproduce their Bernoulli toy experiment.

Given a vector p = (0.6, 0.51, 0.48) the goal is to minimize the loss L(η) = E x1,x2,x3∼Bern(σ(η))

Here x 1 , x 2 , x 3 are i.i.d.

from the Bernoulli(σ(η)) distribution, parameterized by a scalar η ∈ R, where σ(η) = (1 + exp(−η)) −1 is the sigmoid function.

We compare different estimators, with and (b) Low entropy (η = −4) Figure 1 : Bernoulli gradient variance (on log scale) as a function of the number of model evaluations (including baseline evaluations, so the sum-and-sample estimators with sampled baselines use twice as many evaluations).

Note that for some estimators, the variance is 0 (log variance −∞) for k = 8.

without baseline (either 'built-in' or using additional samples, referred to as REINFORCE+ in Liu et al. (2019)).

We report the (log-)variance of the scalar gradient ∂L ∂η as a function of the number of model evaluations, which is twice as high when using a sampled baseline (for each term).

As can be seen in Figure 1 , the unordered set estimator is the only estimator that has consistently the lowest (or comparable) variance in both the high (η = 0) and low entropy (η = −4) regimes and for different number of samples/model evaluations.

This suggests that it combines the advantages of the other estimators.

We also ran the actual optimization experiment, where with as few as k = 3 samples the trajectory was indistinguishable from using the exact gradient (see Liu et al. (2019)).

We use the code from Yin et al. (2019) to train a categorical Variational Auto-Encoder (VAE) with 20 dimensional latent space, with 10 categories per dimension (details in Appendix G.1).

To use our estimator, we treat this as a single factorized distribution with 10 20 categories from which we can sample without replacement using Stochastic Beam Search (Kool et al., 2019c) , sequentially sampling each dimension as if it were a sequence model.

We also perform experiments with 10 2 latent space, which provides a lower entropy setting, to highlight the advantage of our estimator.

Measuring the variance.

In Table 1 , we report the variance of different gradient estimators with k = 4 samples, evaluated on a trained model.

The unordered set estimator has the lowest variance in both the small and large domain (low and high entropy) setting, being on-par with the best of the (stochastic 2 ) sum-and-sample estimator and REINFORCE with replacement 3 .

This confirms the toy experiment, suggesting that the unordered set estimator provides the best of both estimators.

In Appendix G.2 we repeat the same experiment at different stages of training, with similar results.

ELBO optimization.

We use different estimators to optimize the ELBO (details in Appendix G.1).

Additionally to the baselines by Yin et al. (2019) we compare against REINFORCE with replacement and the stochastic sum-and-sample estimator.

In Figure 2 we observe that our estimator performs on par with REINFORCE with replacement (and built-in baseline, equation 23) and outperforms other estimators in at least one of the settings.

There are a lot of other factors, e.g. exploration that may explain why we do not get a strictly better result despite the lower variance.

We note some overfitting (see validation curves in Appendix G.2), but since our goal is to show improved optimization, and to keep results directly comparable to Yin et al. (2019) , we consider regularization a separate issue outside the scope of this paper.

These results are using MNIST binarized by a threshold of 0.5.

In Appendix G.2 we report results using the standard binarized MNIST dataset from Salakhutdinov & Murray (2008) .

2 We cannot use the deterministic version by Liu et al. (2019) since we cannot select the top k categories.

3 We cannot compare against VIMCO (Mnih & Rezende, 2016) as it optimizes a different objective.

For reference, we also include the biased risk estimator, either 'sampling' using stochastic or deterministic beam search (as in Edunov et al. (2018) ).

In Figure 3a , we compare training progress (measured on the validation set) as a function of the number of training steps, where we divide the batch size by k to keep the total number of samples equal.

Our estimator outperforms REINFORCE with replacement, the stochastic sum-and-sample estimator and the strong greedy rollout baseline (which uses additional baseline model evaluations) and performs on-par with the biased risk estimator.

In Figure 3b , we plot the same results against the number of instances, which shows that, compared to the single sample estimators, we can train with less data and less computational cost (as we only need to run the encoder once for each instance).

We introduced the unordered set estimator, a low-variance, unbiased gradient estimator based on sampling without replacement, which can be used as an alternative to the popular biased GumbelSoftmax estimator (Jang et al., 2016; Maddison et al., 2016) .

Our estimator is the result of RaoBlackwellizing three existing estimators, which guarantees equal or lower variance, and is closely related to a number of other estimators.

It has wide applicability, is parameter free (except for the sample size k) and has competitive performance to the best of alternatives in both high and low entropy regimes.

In our experiments, we found that REINFORCE with replacement, with multiple samples and a built-in baseline as inspired by VIMCO (Mnih & Rezende, 2016) , is a simple yet strong estimator which has performance similar to our estimator in the high entropy setting.

We are not aware of any recent work on gradient estimators for discrete distributions that has considered this estimator as baseline, while it may be often preferred given its simplicity.

This means that F φ (g) is the CDF and f φ (g) the PDF of the Gumbel(φ) distribution.

Additionally we will use the identities by Maddison et al. (2014):

Also, we will use the following notation, definitions and identities (see Kool et al. (2019c) ):

For a proof of equation 30, see Maddison et al. (2014) .

We can sample the set S k from the Plackett-Luce distribution using the Gumbel-Top-k trick by drawing Gumbel variables G φi ∼ Gumbel(φ i ) for each element and returning the indices of the k largest Gumbels.

If we ignore the ordering, this means we will obtain the set S k if min i∈S k G φi > max i∈D\S k G φi .

Omitting the superscript k for clarity, we can use the Gumbel-Max trick, i.e. that G φ D\S = max i ∈S G φi ∼ Gumbel(φ D\S ) (equation 30) and marginalize over G φ D\S :

Here we have used a change of variables u = F φ D\S (g φ D\S ).

This expression can be efficiently numerically integrated (although another change of variables may be required for numerical stability depending on the values of φ).

Exact computation in O(2 k ).

The integral in equation 31 can be computed exactly using the identity i∈S

Computation of p D\C (S \ C).

When using the Gumbel-Top-k trick over the restricted domain D \ C, we do not need to renormalize the log-probabilities φ s , s ∈ D \ C since the Gumbel-Top-k trick applies to unnormalized log-probabilities.

Also, assuming

This means that we can compute p D\C (S \ C) similar to equation 31:

Computation of R(S k , s).

Note that, using equation 10, it holds that

This means that, to compute the leave-one-out ratio for all s ∈ S k , we only need to compute p D\{s} (S k \ {s}) for s ∈ S k .

When using the numerical integration or summation in O(2 k ), we can reuse computation, whereas using the naive method, the cost is O(k · (k − 1)!) = O(k!), making the total computational cost comparable to computing just p(S k ), and the same holds when computing the 'second-order' leave one out ratios for the built-in baseline (equation 17).

Details of numerical integration.

For computation of the leave-one-out ratio (equation 35) for large k we can use the numerical integration, where we need to compute equation 34 with C = {s}. For this purpose, we rewrite the integral as

Here we have used change of variables v = u exp(−b) and a = b − φ D\S .

This form allows to compute the integrands efficiently, as

where the numerator only needs to computed once, and, since C = {s} when computing equation 35, the denominator only consists of a single term.

The choice of a may depend on the setting, but we found that a = 5 is a good default option which leads to an integral that is generally smooth and can be accurately approximated using the trapezoid rule.

We compute the integrands in logarithmic space and sum the terms using the stable LOGSUMEXP trick.

In our code we provide an implementation which also computes all second-order leave-one-out ratios efficiently.

We show that the sum-and-sample estimator is unbiased for any set C ⊂ D (see also Liang et al. (2018) ; Liu et al. (2019)):

In this section we give the proof that Rao-Blackwellizing the stochastic sum-and-sample estimator results in the unordered set estimator.

Theorem 4.

Rao-Blackwellizing the stochastic sum-and-sample estimator results in the unordered set estimator, i.e.

Proof.

To give the proof, we first prove three Lemmas.

Lemma 1.

Proof.

Similar to the derivation of P (b 1 = s|S k ) (equation 10 in the main paper), we can write:

The step from the first to the second row comes from analyzing the event S k ∩b k = s using sequential sampling: to sample S k (including s) with s being the k-th element means that we should first sample S k \ {s} (in any order), and then sample s from the distribution restricted to D \ (S k \ {s}).

Lemma 2.

Dividing equation 33 by 1 − s ∈S p(s ) on both sides, we obtain Proof.

Multiplying by 1 − s ∈S p(s ) and rearranging terms proves Lemma 2.

Lemma 3.

Proof.

First using Lemma 1 and then Lemma 2 we find

to the estimator, moving the terms independent of B k outside the expectation and using Lemma 3:

As was discussed in Liu et al. (2019) , one can trade off the number of summed terms and number of sampled terms to maximize the achieved variance reduction.

As a generalization of Theorem 4 (the stochastic sum-and-sample estimator with k − 1 summed terms), we introduce here the stochastic sum-and-sample estimator that sums k − m terms and samples m > 1 terms without replacement.

To estimate the sampled term, we use the unordered set estimator on the m samples without replacement, on the domain restricted to D \ B k−m .

In general, we denote the unordered set estimator restricted to the domain D \ C by

where R D\C (S k , s) is the leave-one-out ratio restricted to the domain D \ C, similar to the second order leave-one-out ratio in equation 18:

While we can also constrain S k ⊆ (D \ C), this definition is consistent with equation 18 and allows simplified notation.

Theorem 5.

Rao-Blackwellizing the stochastic sum-and-sample estimator with m > 1 samples results in the unordered set estimator, i.e.

Proof.

Recall that for the unordered set estimator, it holds that

which for the restricted equivalent (with restricted distribution p D\C ) translates into

Now we consider the distribution b k−m+1 |S k , B k−m : the distribution of the first element sampled (without replacement) after sampling B k−m , given (conditionally on the event) that the set of k samples is S k , so we have b k−m+1 ∈ S k and b k−m+1 ∈ B k−m .

This means that its conditional expectation of f (b k−m+1 ) is the restricted unordered set estimator for C = B k−m since e US,D\B

Observing that the definition (equation 42) of the stochastic sum-and-sample estimator does not depend on the actual order of the m samples, and using equation 45, we can reduce the multisample estimator to the stochastic sum-and-sample estimator with k = k − m + 1, such that the result follows from equation 36.

D THE IMPORTANCE-WEIGHTED ESTIMATOR

In this section we give the proof that Rao-Blackwellizing the importance-weighted estimator results in the unordered set estimator.

Theorem 6.

Rao-Blackwellizing the importance-weighted estimator results in the unordered set estimator, i.e.:

Here we have slightly rewritten the definition of the importance-weighted estimator, using that q(s, a) = P (g φs > a) = 1 − F φs (a) , where F φs is the CDF of the Gumbel distribution (see Appendix A).

Proof.

We first prove the following Lemma:

Proof.

Conditioning on S k , we know that the elements in S k have the k largest perturbed logprobabilities, so κ, the (k + 1)-th largest perturbed log-probability is the largest perturbed logprobability in D \ S k , and satisfies κ = max s∈D\S k g φs = g φ D\S k ∼ Gumbel(φ D\S k ).

Computing p(κ|S k ) using Bayes' Theorem, we have

which allows us to compute (using equation 34 with C = {s} and g φ D\S = κ)

Using Lemma 4 we find

For self-containment we include this section, which is adapted from our unpublished workshop paper (Kool et al., 2019b) .

The importance-weighted policy gradient estimator combines REIN-FORCE (Williams, 1992) with the importance-weighted estimator (Duffield et al., 2007; Vieira, 2017) in equation 15 which results in an unbiased estimator of the policy gradient

Recall that κ is the (k + 1)-th largest perturbed log-probability (see Section 3.2).

We compute a lower variance but biased variant by normalizing the importance weights using the normalization

As we show in Kool et al. (2019b) , we can include a 'baseline' B(

q θ,κ (s) f (s) and correct for the bias (since it depends on the complete sample S k ) by weighting individual terms of

For the normalized version, we use the normalization

q θ,κ (s) for the baseline, and

q θ,κ (s) + p θ (s) to normalize the individual terms:

It seems odd to normalize the terms in the outer sum by

, but equation 52 can be rewritten into a form similar to equation 17, i.e. with a different baseline for each sample, but this form is more convenient for implementation (Kool et al., 2019b ).

To prove the unbiasedness of result we need to prove that the control variate has expectation 0: Lemma 5.

Proof.

Similar to equation 10, we apply Bayes' Theorem conditionally on b 1 = s to derive for s = s

For s = s we have R D\{s} (S k , s ) = 1 by definition, so using equation 54 we can show that

Now we can show that the control variate is actually the result of Rao-Blackwellization:

This expression depends only on b 1 and b 2 and we recognize the stochastic sum-and-sample estimator for k = 2 used as 'baseline'.

As a special case of equation 13 for C = {b 1 }, we have

Using this, and the fact that

We show that the RISK estimator, taking gradients through the normalization factor actually has a built-in baseline.

We first use the log-derivative trick to rewrite the gradient of the ratio as the ratio times the logarithm of the gradient, and then swap the summation variables in the double sum that arises:

Published as a conference paper at ICLR 2020

This assumes we can compute the KL divergence analytically.

Alternatively, we can use a sample estimate for the KL divergence, and use equation 56 with equation 19 to obtain ∇ φ L(φ, θ) = E z∼q φ (z|x) [∇ φ ln q φ (z|x)(ln p θ (x|z) + ln p(z) − ln q φ (z|x)) + ∇ φ ln q φ (z|x)] (60) = E z∼q φ (z|x) [∇ φ ln q φ (z|x)(ln p θ (x|z) − ln q φ (z|x))] .

Here we have left out the term E z∼q φ (z|x) [∇ φ ln q φ (z|x)] = 0, similar to Roeder et al. (2017) , and, assuming a uniform (i.e. constant) prior ln p(z), the term E z∼q φ (z|x) [∇ φ ln q φ (z|x) ln p(z)] = 0.

With a built-in baseline, this second term cancels out automatically, even if it is implemented.

Despite the similarity of the equation 56 and equation 57, their gradient estimates (equation 60 and equation 59) are structurally dissimilar and care should be taken to implement the REINFORCE estimator (or related estimators such as ARSM and the unordered set estimator) correctly using automatic differentiation software.

Using Gumbel-Softmax and RELAX, we take gradients 'directly' through the objective in equation 57.

We optimize the ELBO using the analytic KL for 1000 epochs using the Adam (Kingma & Ba, 2015) optimizer.

We use a learning rate of 10 −3 for all estimators except Gumbel-Softmax and RELAX, which use a learning rate of 10 −4 as we found they diverged with a higher learning rate.

For ARSM, as an exception we use the sample KL, and a learning rate of 3 · 10 −4 , as suggested by the authors.

All reported ELBO values are computed using the analytic KL.

Our code is publicly available 6 .

Gradient variance during training.

We also evaluate gradient variance of different estimators during different stages of training.

We measure the variance of different estimators with k = 4 samples during training with REINFORCE with replacement, such that all estimators are computed for the same model parameters.

The results during training, given in Figure 4 , are similar to the results for the trained model in Table 1 , except for at the beginning of training, although the rankings of different estimator are mostly the same.

Negative ELBO on validation set.

Figure 5 shows the -ELBO evaluated during training on the validation set.

For the large latent space, we see validation error quickly increase (after reaching a minimum) which is likely because of overfitting (due to improved optimization), a phenomenon observed before (Tucker et al., 2017; Grathwohl et al., 2018) .

Note that before the overfitting starts, both REINFORCE without replacement and the unordered set estimator achieve a validation error similar to the other estimators, such that in a practical setting, one can use early stopping.

Results using standard binarized MNIST dataset.

Instead of using the MNIST dataset binarized by thresholding values at 0.5 (as in the code and paper by Yin et al. (2019)) we also experiment with the standard (fixed) binarized dataset by Salakhutdinov & Murray (2008) ; Larochelle & Murray (2011) , for which we plot train and validation curves for two runs on the small and large domain in Figure 6 .

This gives more realistic (higher) -ELBO scores, although we still observe the effect of overfitting.

As this is a bit more unstable setting, one of the runs using REINFORCE with replacement diverged, but in general the relative performance of estimators is similar to using the dataset with 0.5 threshold.

The Travelling Salesman Problem (TSP) is a discrete optimization problem that consists of finding the order in which to visit a set of locations, given as x, y coordinates, to minimize the total length of the tour, starting and ending at the same location.

As a tour can be considered a sequence of locations, this problem can be set up as a sequence modelling problem, that can be either addressed using supervised (Vinyals et al., 2015) or reinforcement learning (Bello et al., 2016; Kool et al., 2019a) .

Kool et al. (2019a) introduced the Attention Model, which is an encoder-decoder model which considers a TSP instances as a fully connected graph.

The encoder computes embeddings for all nodes (locations) and the decoder produces a tour, which is sequence of nodes, selecting one note at the time using an attention mechanism, and uses this autoregressively as input to select the next node.

In Kool et al. (2019a) , this model is trained using REINFORCE, with a greedy rollout used as baseline to reduce variance.

We use the code by Kool et al. (2019a) to train the exact same Attention Model (for details we refer to Kool et al. (2019a) ), and minimize the expected length of a tour predicted by the model, using different gradient estimators.

We did not do any hyperparameter optimization and used the exact same training details, using the Adam optimizer (Kingma & Ba, 2015 ) with a learning rate of 10 −4 (no decay) for 100 epochs for all estimators.

For the baselines, we used the same batch size of 512, but for estimators that use k = 4 samples, we used a batch size of 512 4 = 128 to compensate for the additional samples (this makes multi-sample methods actually faster since the encoder still needs to be evaluated only once).

@highlight

We derive a low-variance, unbiased gradient estimator for expectations over discrete random variables based on sampling without replacement