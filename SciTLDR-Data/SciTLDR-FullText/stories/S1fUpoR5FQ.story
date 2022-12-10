Momentum-based acceleration of stochastic gradient descent (SGD) is widely used in deep learning.

We propose the quasi-hyperbolic momentum algorithm (QHM) as an extremely simple alteration of momentum SGD, averaging a plain SGD step with a momentum step.

We describe numerous connections to and identities with other algorithms, and we characterize the set of two-state optimization algorithms that QHM can recover.

Finally, we propose a QH variant of Adam called QHAdam, and we empirically demonstrate that our algorithms lead to significantly improved training in a variety of settings, including a new state-of-the-art result on WMT16 EN-DE.

We hope that these empirical results, combined with the conceptual and practical simplicity of QHM and QHAdam, will spur interest from both practitioners and researchers.

Code is immediately available.

Stochastic gradient descent (SGD) serves as the optimizer of choice for many recent advances in deep learning across domains (Krizhevsky et al., 2012; He et al., 2016a; .

SGD for deep learning is typically augmented with either the "heavy ball" momentum technique of Polyak (1964) or the accelerated gradient of Nesterov (1983) .

In the deterministic setting, these methods provably yield faster convergence in fairly general settings.

In the stochastic setting, these methods lose many theoretical advantages.

However, due to its implicit gradient averaging, momentum can confer the benefit of variance reduction, applying less noisy parameter updates than plain SGD.

Recent work has explicitly shown the use of momentum as a variance reducer (Roux et al., 2018) .Algorithms Starting with gradient variance reduction as an informal and speculative motivation, we introduce the quasi-hyperbolic momentum (QHM) optimization algorithm in Section 3.

Put as simply as possible, QHM's update rule is a weighted average of momentum's and plain SGD's update rule.

We later propose a similar variant of Adam (QHAdam) in Section 5.Connecting the dots QHM is simple yet expressive.

In Section 4, we connect QHM with plain SGD, momentum, Nesterov's accelerated gradient, PID control algorithms (Recht, 2018; , synthesized Nesterov variants (Lessard et al., 2016) , noise-robust momentum (Cyrus et al., 2018) , Triple Momentum (Scoy et al., 2018) , and least-squares acceleration of SGD (Kidambi et al., 2018) .

Such connections yield reciprocal benefits -these algorithms aid in analyzing QHM, and conversely QHM recovers many of these algorithms in a more efficient and conceptually simpler manner.

We then characterize the set of optimization algorithms that QHM recovers.

In Section 6, we empirically demonstrate that QHM and QHAdam provide superior optimization in a variety of deep learning settings.

We provide both comprehensive parameter sweep analyses on smaller models and case studies on large real-world models.

We demonstrate improvements on strong (sometimes state-of-the-art) models simply by swapping out the vanilla algorithms with the QH counterpart.

Notably, taking the WMT16 EN-DE translation model of BID15 , we achieve a 40% improvement in stability, along with a new state-of-the-art result of 29.45 BLEU.

We then offer some practical tips for QHM and QHAdam.

Miscellany We provide errata for Kingma & Ba (2015) , Recht (2018) , and Kidambi et al. (2018) .

We also offer evidence that momentum often yields negligible improvement over plain SGD.We emphasize QHM and QHAdam's efficiency and conceptual simplicity.

QHM has no extra overhead vs. Nesterov's accelerated gradient, and QHAdam has very little overhead vs. Adam.

Also, both algorithms are easily understood as an interpolation between two other well-known algorithms, so they are accessible to practitioners and can be tuned starting with existing practical intuitions.

We believe that this contributes strongly to the algorithms' practical promise.

We begin with notation and a brief review of stochastic gradient descent (SGD) and momentum.

Primitives In this paper, θ ∈ R p denotes a vector of model parameters.

L(θ) : R p → R denotes a loss function to be minimized via θ.

L(θ) : R p → R denotes an approximator of the loss function (e.g. over a minibatch).

∇L denotes the gradient of function L. Unless otherwise specified, all vector operations are element-wise.

We use g, a, s, v, w ∈ R p as auxiliary buffers, and g is typically the "momentum buffer".

θ,L(·), and all buffers are subscriptable by t, the optimization step.

Optimization algorithms We consider optimization algorithms that perform a sequence of steps (indexed by t), updating θ at each step towards minimizing L(θ).

For brevity, we write algorithms as "update rules", which describe the algorithm's behavior during a single step t, rather than as full pseudocode.

Update rules take this basic form (optionally with one or more auxiliary steps): DISPLAYFORM0 Plain SGD The SGD algorithm, parameterized by learning rate α ∈ R, uses the update rule: DISPLAYFORM1 Momentum The momentum algorithm, parameterized by α ∈ R and β ∈ R, uses the update rule: DISPLAYFORM2 (2) where g is commonly called the "momentum buffer".

Note that β = 0 recovers plain SGD.The exponential discount factor β controls how slowly the momentum buffer is updated.

In the stochastic setting, β also controls the variance of a normalized momentum buffer.

A common rule of thumb for momentum is β = 0.9 (Ruder, 2016) .

2 In contrast to common formulations of momentum (Polyak, 1964; Sutskever et al., 2013) , we normalize, or "dampen", the momentum buffer g by (1 − β) in (1).

This serves both to remove dependence of the update step magnitude on β, and to allow the interpretation of g as a weighted average of past gradients (and thus a gradient estimator).

Of course, this also shrinks the updates by a factor of 1 − β vs. common formulations; this is easily reversible with a corresponding increase to α.

In this section, we propose and discuss the quasi-hyperbolic momentum (QHM) algorithm.

QHM, parameterized by α ∈ R, β ∈ R, and ν ∈ R, uses the update rule: DISPLAYFORM0 DISPLAYFORM1 Section 7.1 provides a recommended rule of thumb (ν = 0.7 and β = 0.999).Interpretation QHM introduces the immediate discount factor ν, encapsulating plain SGD (ν = 0) and momentum (ν = 1).

A self-evident interpretation of QHM is as a ν-weighted average of the momentum update step and the plain SGD update step.

QHM vs. momentum Comparing (2) and (4), QHM may seem at first glance identical to momentum with discount factor νβ.

Appendix A.8 analytically demonstrates that this is not the case.

We note that the expressive power of QHM intuitively comes from decoupling the momentum buffer's discount factor (β) from the current gradient's contribution to the update rule (1 − νβ).

In contrast, momentum tightly couples the discount factor (β) and the current gradient's contribution (1 − β).Variance reduction QHM is originally motivated by an informal and speculative variance reduction analysis; for brevity, we provide the full details in Appendix A. 3 In short, the square bracket term in (4) can be viewed as a gradient estimator (modulo initialization bias).

When ν = 1, this is simply the momentum buffer g t+1 .

Increasing β decreases the variance of the momentum buffer, but potentially at the cost of making it unusably "stale" (biased).

QHM allows for the mitigation of this staleness by upweighting the current, unbiased gradient (i.e. setting ν < 1).Efficiency QHM, like momentum, requires 1 auxiliary buffer of memory.

It also requires 1 in-place scalar-vector multiplication and 3 scaled vector additions per update step.

We now present numerous connections between QHM and other optimization algorithms.

The common theme is that QHM recovers almost all of these algorithms, and thus is a highly interpretable and more efficient implementation of these algorithms.

The first few subsections present these connections, 4 TAB0 summarizes these connections, and Section 4.5 provides discussion.

Nesterov (1983)'s accelerated gradient (NAG) can be viewed as a closely related cousin of momentum.

In fact, replacing the g t+1 term in (2) with DISPLAYFORM0 Connection with QHM It follows from (4) that QHM recovers NAG with ν = β.

This sheds light on the somewhat unintuitive NAG algorithm, providing a natural interpretation of NAG's update rule as a β-weighted average between momentum and plain SGD.Efficiency NAG's compute/memory cost is equivalent to that of QHM.

Recht (2018) draws a strong connection between gradient-based optimization and PID control.

We regurgitate the excellent exposition (with minor modifications) in Appendix B.Update rule A PID control optimizer, parameterized by k P , k I , k D ∈ R, uses the update rule: DISPLAYFORM0 Connection with QHM We fully relate QHM and PID in Appendix C.3.

To summarize, PID is a superfamily of QHM.

Viewing β as a constant, QHM imposes a restriction on the ratio between k P and k D .

Viewing β as a free variable, however, QHM can recover nearly all PID coefficients.

Efficiency Recht (2018) provides a transformation of variables that reduces the memory cost to 2 auxiliary buffers, and the compute cost to 1 in-place scalar-vector multiplication and 4 scaled vector additions per update step.

This is still costlier than QHM.Alternative PID setting In Appendix E, we briefly discuss another PID setting by and relate the resulting optimization algorithm to QHM.

In short, the setting is degenerate as the P, I, and D terms are linearly dependent.

Thus, QHM can recover the resulting PID control optimizer.

Section 6 of Lessard et al. (2016) describes a "synthesized Nesterov variant" algorithm, which we call "SNV" for convenience.

This algorithm is used to analyze and improve optimizer robustness under "relative deterministic noise" (i.e. multiplicative noise of the gradient).Update rule SNV, parameterized by γ, β 1 , β 2 ∈ R, uses the update rule: DISPLAYFORM0 Connection with QHM We fully relate QHM and SNV in Appendix C.4.

To summarize, QHM and SNV recover each other.

By extension, QHM recovers the Robust Momentum method, which is a specific parameterization of SNV (Cyrus et al., 2018) .

Moreover, since Robust Momentum recovers the Triple Momentum of Scoy et al. (2018) , QHM also recovers Triple Momentum.

Efficiency SNV is costlier than QHM, requiring 2 auxiliary buffers and 5 scaled vector additions.

Jain et al. (2017) and Kidambi et al. (2018) point out various failures of momentum and NAG in the setting of stochastic least squares optimization.

This motivates their proposal of the AccSGD algorithm, which yields faster convergence over momentum and NAG in certain least-squares regression settings.

Here, we discuss the formulation of Kidambi et al. (2018) .

Update rule AccSGD, parameterized by δ > 0, κ > 1, ξ ≤ √ κ, and < 1, uses the update rule: DISPLAYFORM0 Connection with QHM We fully relate QHM and AccSGD in Appendix C.5.

To summarize, QHM recovers AccSGD.

In the reverse direction, AccSGD does not recover QHM; specifically, we disprove the claim in Kidambi et al. (2018) that AccSGD recovers NAG.

Since QHM recovers NAG, AccSGD cannot fully recover QHM.Efficiency AccSGD, like QHM, requires 1 auxiliary buffer.

Computationally, AccSGD is costlier, requiring 2 in-place scalar-vector multiplications and 4 scaled vector additions per update step.

Theoretical convergence results We note that various convergence results follow simply via these connections.

In the deterministic (full-batch) case, since QHM recovers Triple Momentum, QHM also recovers the global linear convergence rate of 1 − 1/ √ κ for strongly convex, smooth loss functions.6 For first-order methods, this is the fastest known global convergence rate for such functions.

In the stochastic (minibatch) case, QHM's recovery of AccSGD gives QHM the same convergence results as in Kidambi et al. (2018) 's least-squares regression setting, of O( √ κ · log κ · log 1 ) iterations for -approximation of the minimal loss.

Unifying two-state optimization algorithms These connections demonstrate that many two-state optimization algorithms are functionally similar or equivalent to each other.

However, they are often implemented inefficiently and their parameterizations can be inaccessible to practitioners.

QHM yields a highly accessible and efficient version of these algorithms.

Polyak, 1964) subfamily better recovered by QHM with ν = 1 NAG (Nesterov, 1983) subfamily same recovered by QHM with ν = β PID (Recht, 2018) parent worse QHM's β restricts PID's k P /k D PID bijective worse degenerate; either "PI" or "PD" SNV (Lessard et al., 2016) bijective worse used in handling multiplicative noise Robust M. (Cyrus et al., 2018) subfamily worse SNV w/ convergence guarantees Triple M. (Scoy et al., 2018)

subfamily worse "fastest" for str.

convex, smooth L(·) AccSGD (Kidambi et al., 2018) subfamily worse acceleration for least-squares SGD * "subfamily" means that QHM recovers the algorithm but not vice-versa.

"parent" means that the algorithm recovers QHM but not vice-versa.

"bijective" means that the algorithms recover each other.

† Efficiency (compute and/or memory) vs. QHM.In Appendix D, we characterize the set of two-state optimization algorithms recoverable by QHM.

Our hope here is to provide future work with a routine conversion to QHM so that they may leverage the accessibility and efficiency benefits, as well as the many connections to other algorithms.

Many-state optimization algorithms Going beyond a single momentum buffer, it is possible to recover many-state algorithms by linearly combining many momentum buffers (with different discount factors) in the update rule.

However, we found in preliminary experiments that using multiple momentum buffers yields negligible value over using a single slow-decaying momentum buffer and setting an appropriate immediate discount -that is, using QHM with high β and appropriate ν.

We note that the Aggregated Momentum (AggMo) algorithm (Lucas et al., 2018) precisely performs this linear combination of multiple momentum buffers.

While AggMo takes a simple average of the buffers, an extended variant of AggMo allows for other linear combinations.

This extended AggMo can be viewed as a many-state generalization of two-state algorithms (including QHM), recovering them when two buffers are used.

Appendix H provides a supplemental discussion and empirical comparison of QHM and AggMo, corroborating our preliminary experiments' findings.

The Adam optimizer (Kingma & Ba, 2015) has enabled many compelling results in deep learning (Xu et al., 2015; BID12 Yu et al., 2018) .

We propose to replace both of Adam's moment estimators with quasi-hyperbolic terms, and we name the resulting algorithm QHAdam.

QHAdam, parameterized by α, ≥ 0, β 1 , β 2 ∈ [0, 1), and ν 1 , ν 2 ∈ R, uses the update rule: DISPLAYFORM0 Note that only the last expression differs from vanilla Adam.

In fact, QHAdam recovers Adam when ν 1 = ν 2 = 1.

Moreover, modulo bias correction, QHAdam recovers RMSProp (Hinton et al., 2012) when ν 1 = 0 and ν 2 = 1, and NAdam (Dozat, 2016) when ν 1 = β 1 and ν 2 = 1.

We note that Adam has inspired many variants such as AMSGrad (Reddi et al., 2018) and AdamW (Loshchilov & Hutter, 2017) , which can be analogously modified.

We perform two categories of experiments: parameter sweeps and case studies.

For brevity, all experimental settings are summarized in TAB1 and comprehensively detailed in Appendix I.

With parameter sweeps, we aim to comprehensively study the various parameterizations of the QH algorithms using relatively small models.

We train for 90 epochs with size-64 minibatches.

For QHM, we initialize α = 1 and decay it 10-fold every 30 epochs.

The sweep grid for QHM (encapsulating various parameterizations of plain SGD, momentum, and NAG) is: 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1} β ∈ {0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995} For QHAdam, we fix α = 10 −3 , = 10 −8 , ν 2 = 1, and β 2 = 0.999, and sweep over ν 1 and β 1 .

DISPLAYFORM0 "Default" ν and β values Motivated by the popular momentum/NAG "default" of β = 0.9, we select a QH "default" of ν = 0.7 and β = 0.999 based on preliminary experimentation on the MNIST dataset (LeCun, 1998) along with the intuitions from Appendix A. In the following figures, we show these defaults along with the globally optimal parameterizations.

Results FIG0 presents selected results of these sweep experiments (full results in Appendix J).

Perhaps the most immediate observation is that the QH algorithms improve both training and validation metrics.

Even the hardcoded default ν = 0.7 and β = 0.999 handily outperforms the optimal parameterization of NAG or Adam in all settings.

In some settings, there remains a large gap between the QH and vanilla algorithms at the end of training.

In other settings, the gap shrinks to smaller levels.

However, even for these latter settings, the QH algorithm converges much faster, suggesting that a more aggressive learning rate schedule can significantly reduce training time.

What about plain SGD?

We note that in most of these experiments, there is little difference between the performance of plain SGD and NAG (particularly when compared to QHM).

Although not shown in the figures, there is also little difference between plain SGD and momentum.

This indicates that the benefit of momentum and NAG (in the common, unnormalized formulations) comes in large part from the increase in effective step size.

We thus suspect that much of the folk wisdom about momentum's benefits for SGD should instead be folk wisdom about using sensible learning rates.

In contrast, QHM provides significant benefits without changing the effective step size.

With case studies, we apply the QH algorithms to diverse settings, with (currently or recently) stateof-the-art models.

Our case studies cover image recognition, language modeling, reinforcement learning, and neural machine translation.

Each case study features a baseline setting and a QH setting, which are identical modulo the optimizer used.

Results are presented in FIG1 and TAB2 .Image recognition (RN152-ImageNet-QHM) We train a ResNet152 model (He et al., 2016a) on the ILSVRC2012 dataset (Russakovsky et al., 2015) .

The baseline setting is nearly identical to the size-256 minibatch baseline in Goyal et al. (2017) , using NAG with β = 0.9 and a decaying learning rate schedule.

The QH setting swaps out NAG for QHM, with ν = 0.7 and β = 0.999.

Running 3 seeds, QHM plainly trains much faster than NAG, and QHM converges to a marginally superior validation error as well.

Language modeling (FConvLM-WikiText103-QHM) Deep learning for NLP often features "spiky" gradient distributions (e.g. encountering rare words).

We train a FConv language model (Dauphin et al., 2016) on the WikiText-103 dataset (Merity et al., 2016) .

The baseline setting precisely follows the original paper, using NAG with β = 0.99.

The QH setting swaps out NAG for QHM, with ν = 0.98 and β = 0.998.

7 We suspect that high β improves stability in the presense of spiky gradients, and QHM's ν allows the use of high β.

Running 10 seeds, QHM outperforms the NAG baseline on validation perplexity by half a point.

Reinforcement learning (TD3-MuJoCo-QHAdam) Reinforcement learning presents a challenging task for gradient-based optimization, since the objective L is not stationary.

QH algorithms provide a natural way of upweighting the most recent gradient.

Here, we apply the TD3 algorithm to various MuJoCo environments BID7 .

The baseline precisely follows 's setup, which uses Adam with β 1 = 0.9 and β 2 = 0.999.

The QH setting swaps out Adam for QHAdam, with ν 1 = 0.9 and other parameters identical.

Running 10 seeds, QHAdam yields improvements in average reward on four environments out of seven tested, and virtually ties on another.

Neural machine translation (TF-WMT16ENDE-QHAdam) Many state-of-the-art neural machine translation (NMT) models are fragile to train.

As in language modeling, the gradient distribution is often "spiky"; thus, Adam training often fails to converge due to a very small number of large parameter updates.

10 Here, we empirically demonstrate that QHAdam improves both performance and robustness by using ν 2 to control the maximum per-step update.

We train a large transformer model BID12 on the WMT16 English-German dataset.

The baseline setting precisely follows the state-of-the-art setup of BID15 , using β 1 = 0.9 and β 2 = 0.98 for Adam.

The QH setting swaps out Adam for QHAdam, with ν 1 = 0.8, β 1 = 0.95, ν 2 = 0.7, and β 2 = 0.98.

Running 10 seeds, the Adam baseline explodes on 4 seeds.

QHAdam is more robust, converging for all seeds.

Ultimately, QHAdam yields a new state-of-the-art-result of 29.45 BLEU.

Thus, we improve both the stability and performance of the state-of-the-art with a simple optimizer swap.

8 We also train the model using plain SGD, again finding that plain SGD performs nearly as well as NAG throughout training.

Although not shown, plain SGD in fact performs better than momentum.

The validation loss curves for plain SGD, momentum, and NAG are indistinguishable throughout training, suggesting that momentum/NAG is not needed in Goyal et al. (2017) .

9 Here, we tried higher values of β1.

Significantly increasing β1 was not fruitful for either algorithm.

10 Refer to Appendix F for a more detailed theoretical treatment.

11 Here, we tried two other parameterizations (higher β1) with marginal success.

We offer some practical suggestions for deep learning practitioners, particularly those who default to momentum, NAG, or Adam with β = 0.9 as a rule of thumb:• Consider using QHM or QHAdam, instead of momentum, NAG, or Adam.• While QHM parameters should be tuned when feasible, a decent rule of thumb is to set ν = 0.7 and β = 0.999.

QHAdam parameter selection is somewhat more situational, although as discussed in Section 5, ν 2 = 1 and β 2 unchanged is usually reasonable when replacing a stable Adam optimizer with QHAdam.• Be mindful of learning rate differences between (unnormalized) momentum/NAG and QHM.

Convert learning rates from the former to the latter via multiplication by (1 − β) −1 .

For example, momentum/NAG with α = 0.1 and β = 0.9 should be replaced by QHM with α = 1.

This conversion is unnecessary for Adam, as it already normalizes all buffers.

This paper has only scratched the surface when it comes to empirical evaluation of QHM and QHAdam.

Future work could apply the algorithms to other well-studied tasks and architectures, both to assess the extent of their performance gains in diverse domains, and to further develop insights into hyperparameter choice.

Effective hyperparameter autotuning methods can improve the practicality of any optimization algorithm.

Thus, a useful direction for future work is to create an effective ν, β adapter, possibly based on techniques such as YellowFin (Zhang et al., 2017) or via continuous-time optimal control analysis, as in Li et al. (2017) .

Moreover, learning rate adaptation techniques such as Hypergradient Descent (Baydin et al., 2018) can be applied to both QHM and QHAdam.

Future work could develop convergence results for QHAdam.

Convergence results for QHM in a reasonably general stochastic setting would also be appealing, although we are not aware of compelling analogous results for momentum or NAG.Finally, momentum has been studied in the distributed, asynchronous setting, with some noting that the delays in asynchronous SGD are, in some sense, akin to adding momentum (Mitliagkas et al., 2016) .

As a result, the optimal momentum constant β shrinks as more asynchronous workers are added to optimization.

It would be interesting to extend these results to QHM, especially to disentagle the implicit effects of asynchrony on ν and β.

QHM and QHAdam are computationally cheap, intuitive to interpret, and simple to implement.

They can serve as excellent replacements for momentum/NAG and Adam in a variety of settings.

In particular, they enable the use of high exponential discount factors (i.e. β) through the use of immediate discounting (i.e. ν).

QHM recovers numerous other algorithms in an efficient and accessible manner.

Parameter sweep experiments and case studies demonstrate that the QH algorithms can handily outpace their vanilla counterparts.

We hope that practitioners and researchers will find these algorithms both practically useful and interesting as a subject of further study.

Organization This paper's appendices are ordered as follows:• Appendix A presents a view of momentum and QHM as discounted sums, and provides the original motivation for the development of QHM.• Appendix B regurgitates Recht (2018)'s excellent exposition of gradient-based optimization as PID control, with minor modifications.• Appendix C presents analyses of various other algorithms, towards connecting them to QHM.• Appendix D describes the set of all two-state optimization algorithms recovered by QHM.• Appendix E briefly discusses a PID control optimization setting by .•

Appendix F derives a tight upper bound on the updates of Adam and QHAdam (consequently disproving the bound in Kingma & Ba FORMULA18 ), then discusses the implications on training stability.• Appendix G provides miscellaneous derivations that do not cleanly fit in other sections.• Appendix H provides discussion and an empirical comparison of QHM and AggMo (Lucas et al., 2018 ).•

Appendix I comprehensively describes the setup of this paper's parameter sweep and case study experiments.• Appendix J comprehensively presents the results of this paper's parameter sweep experiments.

We now provide an interpretation of the momentum buffer as a discounted sum estimator, seeking to motivate the QHM algorithm from a variance reduction perspective.

For a discount function δ : N ≥0 → R and a sequence of vectors x 0...t ∈ R p , we define a discounted sum DS δ (x 0...t ) as: DISPLAYFORM0 δ(i) = 1 for all t ≥ 0, we call this a discounted sum average.

When ∞ i=0 δ(i) = 1, we call this a discounted sum average (modulo initialization bias).

For β ∈ (−1, 1), we define the exponential discount function δ EXP,β as: DISPLAYFORM0 and the exponentially weighted moving average EWMA β (x 0...t ) as: DISPLAYFORM1 The EWMA is a discounted sum average (modulo initialization bias), so it can be viewed as an estimator of the expectation of a random variable x if x 0...t ∼ x. Note that the momentum buffer g t from (1) is precisely an EWMA -specifically, g t = EWMA β (∇L 0...t (θ 0...t )).It is well known that the exponential discount function is the only time-consistent (commonly "memoryless"), discount function -i.e.

for any i, τ ≥ 0, the ratio d(i + τ )/d(i) depends only on τ .

This is precisely why the EWMA can be tracked with no auxiliary memory -for example, as in momentum's update rule.

We now provide the following fact about the covariance of the EWMA when x 0...t are random variables.

Fact A.1 (Limit covariance of EWMA).

Assume that x 0...t are independent random vectors, each with the covariance matrix Σ. Then: DISPLAYFORM0 This means that arbitrary variance reduction of the EWMA is possible by increasing β.

For example, β = 0.9 implies that the covariance is reduced to This provides an intuitive explanation of momentum as a variance reduction technique.

Assuming that the momentum buffer is normalized (and thus interpretable as an estimator of the gradient), applying momentum will reduce the variance of the update steps, with higher β leading to more variance reduction.

However, the flip side is that higher β induces more bias (informally, "staleness") in the momentum buffer with respect to the true gradient, as the momentum buffer becomes extremely slow to update.

Thus, the question arises: can we achieve variance reduction while guaranteeing that recent gradients contribute significantly to the update step?

For this, we must introduce time-inconsistency.

Hyperbolic discounting, first proposed by Chung & Hernstein (1961) , is the classical timeinconsistent discount function in consumer choice.

It is commonly used to model individual behaviors such as impatience.

We consider its use in the setting of stochastic optimization (in place of the EWMA buffer of momentum).For constants c, k > 0, we define the hyperbolic discount function as: 12 δ H,c,k (i) = c 1 + ki and the hyperbolic weighted moving average HWMA c,k (x 0...t ) as: DISPLAYFORM0 Note that the hyperbolic discount function is time-inconsistent, since: DISPLAYFORM1 depends on both i and τ .Unlike the EWMA, the HWMA is not a discounted sum average -in fact, ∞ i=0 δ H,c,k (i) = ∞ holds regardless of choice of c or k. Thus, to use an HWMA of gradients in an optimization algorithm, c (or the learning rate α) must be decayed at a logarithmic rate.

More concerning, however, is the computational inefficiency of the HWMA; specifically, the sum must be recomputed from scratch at each iteration from all past gradients.

This is unacceptable for use in most practical applications.

However, in preliminary stochastic optimization experiments, we did observe a marked benefit of HWMA over EWMA (i.e. momentum), limiting the number of past gradients used for tractability.

This indicates that time-inconsistency might be a useful property to have in a stochastic optimizer.

A.5 QUASI-HYPERBOLIC DISCOUNTING AND QHWMA Quasi-hyperbolic discounting, proposed by Phelps & Pollak (1968) and popularized in consumer choice by Laibson (1997) , seeks to qualitatively approximate the time-inconsistency of hyperbolic discounting by applying a discontinuous "upweighting" of the current step.

Its tractability has resulted in much wider adoption in consumer choice vs. pure hyperbolic discounting, and we find that it is also more suited for use in practical optimization.

For constants ν ∈ R and β ∈ (−1, 1), we define the quasi-hyperbolic discount function as: DISPLAYFORM2 and the quasi-hyperbolic weighted moving average QHWMA ν,β (x 0...t ) as: DISPLAYFORM3 The QHWMA, like the EWMA, is a discounted sum average (modulo initialization bias), so it can also be viewed as an estimator under the same assumptions.

When ν = 1, the QHWMA is precisely the EWMA (with identical β), and the quasi-hyperbolic discount function is precisely the exponential discount function (and thus time-consistent).

When ν = 1, the quasi-hyperbolic discount function, like the hyperbolic discount function, is timeinconsistent since: DISPLAYFORM4 depends on both i and τ ; specifically, i = 0 yields a different ratio than i > 0.Note from (5) that the QHWMA is a ν-weighted average of the EWMA (with identical β) and x 0 .

This means that the QHWMA can be easily computed online by simply keeping track of the EWMA, thus requiring no additional memory.

We now characterize the variance of a QHWMA using this fact:Fact A.2 (Limit covariance of QHWMA).

Assume that x 0...t are independent random vectors, each with the covariance matrix Σ. Then: DISPLAYFORM0 where ρ is defined as: DISPLAYFORM1 Proof.

Provided in Appendix G.ρ is essentially a scaling factor for the covariance of the QHWMA.

It can be verified that ρ decreases (thus inducing variance reduction) with both increasing β and increasing ν.

This leads to our motivation for QHM, which simply replaces the EWMA momentum buffer with a QHWMA.

Starting with any momentum parameterization (ν = 1 and β ∈ (0, 1)), β can be increased towards variance reduction (i.e. lowering ρ).

Then, ν can be decreased to make the QH-WMA less biased as a gradient estimator, thus mitigating the aforementioned "staleness" problem.

Note, however, that since decreasing ν will also increase ρ, we cannot simply decrease ν to zero.

Specifically, any ν < 1 imposes a tight lower bound of (1 − ν) 2 on ρ, regardless of choice of β.

For completeness, we explicitly write the update rules for the momentum and QHM algorithms.

Momentum The momentum update rule is: DISPLAYFORM0 which can be efficiently written using an auxiliary buffer g t as: DISPLAYFORM1 (2, revisited from Section 2) QHM The QHM update rule is: DISPLAYFORM2 which can be efficiently written using an auxiliary buffer g t as: DISPLAYFORM3 QHM vs. momentum Comparing (2) and FORMULA4 , QHM may seem at first glance identical to momentum with discount factor νβ.

However, replacing the β in (6) with νβ yields: DISPLAYFORM4 which plainly differs from (7) -most notably, in the exponential discount factor (νβ) for past gradients.

Thus, momentum with discount factor νβ does not recover QHM.

Section 4 presents numerous connections to other optimization algorithms that shed light on both deterministic and stochastic convergence properties of QHM.

However, we do not formally analyze the convergence properties of QHM from a variance reduction standpoint; this remains future work.

Here, we briefly discuss other work in variance reduction.

Finite sums Recently, much effort has been devoted towards reducing the variance of the stochastic gradients used in optimization algorithms.

Perhaps the most widely-studied setting is the "finite sum", or offline, stochastic optimization setting.

Methods analyzed in the finite-sum setting include SAG ( , and others.

We do not comment in detail on the finite sum setting due to its limited practical applicability to large-scale deep learning; for a fuller discussion of such methods, see Kidambi et al. (2018) .Momentum as variance reduction Some work in variance reduction has drawn an explicit connection to momentum.

For example, Roux et al. (2018) propose a method involving Bayesian updates of gradient estimates, which induces adaptive gradient averaging.

The authors note that this method boils down to momentum with an adaptive β.

We follow Recht (2018) in describing the connection between PID control and gradient-based optimization.

Continuous PID We slightly adapt the setting from BID0 .

t denotes time.

There is a setpoint (i.e. target state), r(t), and a process variable (i.e. current state), y(t).

The error of the system is defined as e(t) def = r(t) − y(t).

A "controller" outputs a control signal u(t), usually towards the goal of making the error zero.

The controller's choice of u(t) affects y(t) in some unspecified manner.

A PID controller, parameterized by k P , k I , and k D , uses the control function: DISPLAYFORM0 Here, the terms in the square brackets are typically referred to as the P, I, and D terms, respectively.

Discrete approximation In discrete time, the setpoint, process variable, and error are trivially discretized as r t , y t , and e t def = r t − y t , respectively.

The I term, which we label w t , is discretized as: DISPLAYFORM1 The D term, which we label v t , could be discretized as v t = e t − e t−1 (first differences).

However, a low-pass filter is often applied to mitigate noise, thus resulting in: DISPLAYFORM2 We simplify exposition by considering e −1 , w −1 , and v −1 to be 0.Finally, the PID control function FORMULA27 is trivially discretized as: DISPLAYFORM3 Optimization Recht (2018) relates optimization to PID control as follows: DISPLAYFORM4 That is, the process variable is the stochastic gradient, the controller's goal is to make this gradient zero, and the controller achieves this by choosing the next step's model parameters according to the update rule θ t+1 ← u t + θ 0 .

The update rule for a PID control optimizer is thus: DISPLAYFORM5 Recht demonstrates that PID in this setting encapsulates gradient descent, momentum, and NAG; for example, gradient descent is recovered when k P = k D = 0 and k I = α.

Intuition Finally, to provide some additional intuition, we can state the following fact about the D term (v t ): Fact B.1 (D term is gradient and momentum).

v t can be written as: DISPLAYFORM6 Proof.

Provided in Appendix G.Thus, the D term is simply a weighted sum of an EWMA of gradients (i.e. momentum buffer) and the current gradient, and a PID control optimizer's output is simply a weighted sum of the momentum buffer, the current gradient, and the sum of all past gradients.

This appendix presents a deeper theoretical treatment of Section 4.2 through Section 4.4, deriving and discussing connections between QHM and various other optimization algorithms.

Along the lines of Lessard et al. FORMULA22 , we consider optimizers as linear operators, interrupted by a nonlinear step (the gradient evaluation).

In this setting, optimizers have b internal state buffers, which we write as a stacked vector S t ∈ R b·p .

Optimizers accept the current optimizer state (S t ) and gradient (∇L t (θ t )), and they produce the new optimizer state (S t+1 ) and parameters (θ t ) using a square matrix T ∈ R (b+2)p×(b+2)p .14 Update rule For convenience, we impose the restriction that the output θ t can only depend on the state S t .

Then, for analytical purposes, the optimizer can be written as the following update rule: Coordinate-wise decomposition Since we only consider optimizers that act coordinate-wise (except for the gradient evaluation), we can write T as the Kronecker product of a coordinate-wise transition matrix A ∈ R (b+2)×(b+2) and the identity matrix I p .

That is, T = A ⊗ I p .

DISPLAYFORM0 Then, for t > 0, we can write θ t in terms of the initial state S 0,{1...b} and all past gradients, using the last row of various matrix powers of A: DISPLAYFORM1 C.2 QHMThe internal state of QHM includes two buffers: g t (momentum buffer) and θ t (model parameters).The transition matrix T QHM , mapping from g t θ t ∇L t (θ t ) 0 p to [g t+1 θ t+1 0 p θ t ] , is: DISPLAYFORM2 For n > 0, routine computation yields the last row of the (n + 1)-th matrix power: DISPLAYFORM3 Applying (10), the optimizer state θ t can be written as: DISPLAYFORM4 In the typical case of g 0 = 0, we have: DISPLAYFORM5 C.3 PID Recht (2018) draws a strong connection between gradient-based optimization and PID control.

We regurgitate the excellent exposition (with minor modifications) in Appendix B.Update rule A PID control optimizer, parameterized by k P , k I , k D ∈ R, uses the update rule: DISPLAYFORM6 Coordinate-wise decomposition The internal state of a PID control optimizer includes four buffers: e t−1 (P term), w t−1 (I term), v t−1 (D term), and θ 0 (initial parameters).

The transition matrix T PID , mapping from e t−1 w t−1 v t−1 θ 0 ∇L t (θ t ) 0 p to [e t w t v t θ 0 0 p θ t ] , is: DISPLAYFORM0 For n > 0, routine computation yields the last row of the (n + 1)-th matrix power: DISPLAYFORM1 0 where: DISPLAYFORM2 Applying (10), the optimizer state θ t can be written as: DISPLAYFORM3 Relationship with QHM In the typical case of e −1 = w −1 = v −1 = 0 p , we have: DISPLAYFORM4 15 The offset of −1 in the P, I, and D term subscripts is purely for convenience.

Then, equating with (11), we have that QHM is PID with: DISPLAYFORM5 or that PID is QHM with: DISPLAYFORM6 Viewing β as a constant, the following restriction holds on the PID coefficients that QHM can recover: DISPLAYFORM7 This restriction is looser than those for plain SGD (which has the additional restriction k P = k D = 0), momentum (which has the additional restriction k P /k I = k D /k P ), and NAG (which has the additional restriction k P /k I = βk D /k P ).Viewing β as a hyperparameter, QHM can recover all PID coefficients except when k I = 0 (i.e. P, D, or PD controller), or k P = 0 = k D (i.e. PI controller).To summarize, PID is a superfamily of QHM.

Viewing β as a constant, QHM imposes a restriction on the ratio between k P and k D .

Viewing β as a free variable, however, QHM can recover nearly all PID coefficients.

C.4 SNV Section 6 of Lessard et al. FORMULA22 describes a "synthesized Nesterov variant" algorithm, which we call "SNV" for convenience.

This algorithm is used to analyze and improve optimizer robustness under "relative deterministic noise" (i.e. multiplicative noise of the gradient).

Update rule SNV, parameterized by γ, β 1 , β 2 ∈ R, uses the update rule: DISPLAYFORM0 Coordinate-wise decomposition The internal state of a SNV optimizer includes two buffers: ξ t and ξ t−1 .The transition matrix T SNV , mapping from ξ t ξ t−1 ∇L t (θ t ) 0 p to [ξ t+1 ξ t 0 p θ t ] , is: DISPLAYFORM1 For n > 0, routine computation gives us the last row of the (n + 1)-th matrix power: DISPLAYFORM2 where: DISPLAYFORM3 Applying (10), the optimizer state θ t can be written as: DISPLAYFORM4 Relationship with QHM Initialize ξ 0 = ξ −1 = θ 0 .

The optimizer state θ t is: DISPLAYFORM5 Then, equating with (11), we have that QHM is SNV with: DISPLAYFORM6 or that SNV is QHM with: DISPLAYFORM7 To summarize, QHM and SNV recover each other.

By extension, QHM recovers the Robust Momentum method, which is a specific parameterization of SNV (Cyrus et al., 2018) .

Moreover, since Robust Momentum recovers the Triple Momentum of Scoy et al. (2018) , QHM also recovers Triple Momentum.

C.5 ACCSGD Jain et al. FORMULA24 and Kidambi et al. (2018) point out various failures of momentum and NAG in the setting of stochastic least squares optimization.

This motivates their proposal of the AccSGD algorithm, which yields faster convergence over momentum and NAG in certain least-squares regression settings.

Here, we discuss the formulation of Kidambi et al. (2018) .Update rule AccSGD, parameterized by δ > 0, κ > 1, ξ ≤ √ κ, and < 1, uses the update rule: DISPLAYFORM8 Coordinate-wise decomposition The internal state of an AccSGD optimizer includes two buffers: w t (a buffer) and w t (the iterate, identical to θ t ).The transition matrix T AccSGD , mapping from w t w t ∇L t (θ t ) 0 p to [w t+1 w t+1 0 p θ t ] , is: DISPLAYFORM9 For n > 0, routine computation gives us the last row of the (n + 1)-th matrix power: DISPLAYFORM10 where: DISPLAYFORM11 Applying (10), the optimizer state θ t can be written as: DISPLAYFORM12 Relationship with QHM Fix ∈ (0, 1), and initializew 0 = w 0 = θ 0 .

The optimizer state θ t is: DISPLAYFORM13 Then, equating with (11), we have that QHM is AccSGD with: DISPLAYFORM14 or that AccSGD is QHM with: DISPLAYFORM15 AccSGD cannot recover NAG Based on the above analysis, NAG (i.e. ν = β) is recovered 19 This disproves the claim in Kidambi et al. (2018) that AccSGD recovers NAG when ξ = √ κ.

DISPLAYFORM16 In fact, we demonstrate that AccSGD cannot recover NAG at all.

For ∈ (0, 1) and the aforementioned value of ξ, we have that ξ > √ κ: DISPLAYFORM17 Since AccSGD requires that ξ ≤ √ κ and that ∈ (0, 1) 20 , AccSGD cannot recover NAG.To summarize, QHM recovers AccSGD.

In the reverse direction, AccSGD does not recover QHM; specifically, we disprove the claim in Kidambi et al. (2018) that AccSGD recovers NAG.

Since QHM recovers NAG, AccSGD cannot fully recover QHM.

This appendix describes a generic two-state optimizer ("TSO") where one of the states is the iterate (θ t ) and the other is an auxiliary buffer (a t ).

The optimizer is parameterized by h, k, l, m, q, z ∈ R, and the update rule is: DISPLAYFORM0 We can write this as a transition matrix T TSO ∈ R 3×3 : DISPLAYFORM1 To simplify further derivations we diagonalize T TSO as: DISPLAYFORM2 , then QHM implements the TSO optimizer with: DISPLAYFORM3 Proof.

We can write down the unrolled TSO update rule for θ t , as follows: DISPLAYFORM4 Similarly, for QHM we can define a transition matrix T QHM ∈ R 3×3 that advances state g t θ t ∇L t (θ t ) as: DISPLAYFORM5 Thus, the unrolled update rule for QHM takes the following form: DISPLAYFORM6 Now we match the corresponding coefficients in both of the update rules to establish dependencies: DISPLAYFORM7 By solving the first equation we can establish values for α, β, and ν: DISPLAYFORM8 Per our assumption (Λ TSO ) 3,3 = 1 2 (h + q + φ) = 1 and h − q + φ = 0, we can recover the following relationships: DISPLAYFORM9 We can solve the second equation to find g 0 : DISPLAYFORM10 Given that (Λ TSO ) 3,3 = 1 and h − q + φ = 0 =⇒ h − q − φ = −2φ, we can simplify: DISPLAYFORM11 DISPLAYFORM12 Discussion This setting departs somewhat from typical PID control, in that the signal u t controls the derivative of the controller's output (i.e. θ t+1 − θ t ) rather than the output itself (i.e. θ t+1 − θ 0 ).

To avoid parameter blowup, this formulation necessitates the addition of exponential decay to the I term, with discount factor β.

The I term thus becomes the momentum buffer.

However, recall from Fact B.1 that the D term is a weighted sum of the momentum buffer and the P term.

It follows that the D term is a weighted sum of the P and I terms, and that this setting is degenerate (either "PI" or "PD").As a consequence, the proposed PID algorithm of An et al. FORMULA27 is less expressive than that of Recht (2018) .

Specifically, applying Fact B.1 demonstrates a mapping into QHM: DISPLAYFORM0 Efficiency This PID control optimizer is costlier than QHM.

It requires 2 auxiliary buffers of memory.

Computationally, it requires 2 in-place scalar-vector multiplications and 5 scaled vector additions per update step.

This appendix elaborates on Adam and QHAdam's stability properties through the lens of a step size upper bound.

It is well known that the training process for deep learning models can often "explode" due to a very small number of large parameter updates.

With Adam, these large updates can occur if there exist parameters whose stochastic gradients are almost always near zero but incur rare "spikes".

23 .

This is because the square root of the second moment estimate, used in normalizing the gradient for the update step, will be far below the magnitude of these spikes.

There are three main ways to address this instability:• Firstly, one can simply decrease the learning rate α.

However, this may be undesirable due to slower training.• Secondly, one can increase the hyperparameter.

However, the appropriate setting of depends on the exact magnitudes of these gradient spikes, which is often unknown.

Setting too high effectively turns Adam into SGD.

Thus, setting often reduces to guesswork.• Thirdly, one can clip gradients.

However, the appropriate magnitude of the gradient clipping also depends on the stochastic gradient distribution.

Thus, this solution also involves a fair amount of guesswork.

However, Adam does provide a useful guarantee -unlike SGD, Adam has an upper bound on the per-step update (Kingma & Ba, 2015) .

This upper bound is independent of the gradient distribution (or even temporal correlation), depending only on the hyperparameters α, β 1 , and β 2 .

Thus, no matter the gradient distribution, Adam will restrict the magnitude of the per-step updates to some known constant.

Kingma & Ba (2015) intuitively describe this bound as "establishing a trust region around the current parameter value".We show that the step size upper bound claimed in Section 2.1 of Kingma & Ba (2015) is incorrect, by providing the correct tight bound for both Adam and QHAdam.

We then demonstrate that with QHAdam, one can lower the maximum per-step update (and thus improve stability) simply by lowering ν 2 to be below 1.

We make two simplifications.

Firstly, we fix = 0.

24 Secondly, we remove the bias correction of the moment estimators (i.e. we use g t+1 ← g t+1 and s t+1 ← s t+1 ).In this setting, QHAdam applies the following update rule: DISPLAYFORM0 where:g DISPLAYFORM1 DISPLAYFORM2

We now bound QHAdam's update (before scaling by α) by a constant dependent only on β 1 , β 2 , ν 1 , and ν 2 :Fact F.1 (QHAdam tight upper bound on update).

Assume thats t+1 is nonzero at each coordinate and that 0 < β 1 < √ β 2 < 1.

Then, the following per-coordinate tight upper bound holds: DISPLAYFORM0 Proof.

Firstly and without loss of generality, we can treat the gradients as single coordinates DISPLAYFORM1 We perform the following simplification of FORMULA82 and FORMULA3 : DISPLAYFORM2 We now wish to find the values of x i that maximizeg 2 t+1 st+1 .

Applying FORMULA4 and FORMULA18 , these values are characterized by the following first-order conditions: DISPLAYFORM3 st+1 is invariant to scalar multiplication of all x i , we can simplify FORMULA22 to: DISPLAYFORM4 Plugging the values of x i from FORMULA24 into FORMULA4 and (15) yields: DISPLAYFORM5 The desired result follows immediately.

Limit case Consider the limit case of t → ∞. Then, the bound in Fact F.1 simplifies to: DISPLAYFORM6 For vanilla Adam (i.e. ν 1 = ν 2 = 1), (18) simplifies further to: DISPLAYFORM7 Note that since the bound in (19) is tight, this result contradicts the claim in Section 2.1 of Kingma & Ba (2015) that Adam's per-coordinate step size is bounded above by α · max{1, (1 − β 1 )/ √ 1 − β 2 }.

25 In the following discussion, we use the correct bounds from (18) and (19).

The recommended vanilla Adam setting of β 2 = 0.999 in Kingma & Ba (2015) makes the right-hand side of (19) to be large, and various work has employed Adam with a significantly lower β 2 ; e.g. 0.98 BID12 BID15 .

26 Decreasing β 2 is undesirable, often slowing down training.

27 Moving from Adam to QHAdam, an alternative solution is to decrease ν 2 to be below 1.

This decreases the right-hand side of (18), up to a point, and thus imposes a tighter constraint on the magnitudes of updates than the vanilla Adam setting of ν 2 = 1.

Fig. 3 shows an example of this phenomenon using a fixed ν 1 , β 1 , and β 2 .Figure 3: Bound from (18), fixing ν 1 = 0.8, β 1 = 0.95, and β 2 = 0.98, and varying ν 2 .

26 We performed experiments on these models indicating that increasing β2 far beyond 0.98 led to training explosion.

We suspect that these instability issues are especially prevalent in settings with rare inputs or labels, such as machine translation.

27 In proposing the AdamNC algorithm, Reddi et al. (2018) suggests that β2 should be high to capture a sufficiently long history of past gradients.

This appendix provides miscellaneous derivations that do not cleanly fit elsewhere.

where ρ is defined as: DISPLAYFORM0 Proof.

Due to the independence assumption, the covariance matrix of the QHWMA for t > 0 is simply: DISPLAYFORM1 The desired result follows immediately.

Fact B.1 (D term is gradient and momentum).

v t can be written as: DISPLAYFORM2 Proof.

We expand v t as follows, recalling that v −1 = 0: DISPLAYFORM3 We then proceed by separating out the sum in (20), recalling that e −1 = 0: DISPLAYFORM4 The desired result follows by substituting e t = −∇L t (θ t ) into (21).

We perform a brief empirical comparison of QHM and Aggregated Momentum (AggMo), proposed by Lucas et al. (2018) .

In short, we find that for an autoencoder task, we can take the optimal parameterization of AggMo from an extensive parameter sweep, and from that we can construct a QHM parameterization by hand which outperforms the optimal AggMo parameterization.

AggMo is a many-state optimizer that aggregates multiple momentum buffers in its update rule.

AggMo update rule The AggMo algorithm, parameterized by discount factors β ∈ R K and learning rate γ > 0, uses the update rule: DISPLAYFORM0 DISPLAYFORM1 Intuitively, AggMo maintains K unnormalized momentum buffers with different discount factors and uses the average of these buffers in the update rule.

Experimental setup: EMNIST autoencoders We perform the autoencoder experiments of Lucas et al. FORMULA27 using the authors' implementation, 28 with two changes:1.

We replace the MNIST dataset (LeCun, 1998) with the richer digits subset of the EMNIST dataset (Cohen et al., 2017) .

We hold out 10% of the training dataset for validation.

Performing the same sweep, we find that the best parameterization of AggMo uses discount factors β = [0, 0.9, 0.99, 0.999] and learning rate γ = 0.1.

We name this parameterization "AggMo-Best".Parameterizing QHM We now apply intuition to convert AggMo-Best into a QHM parameterization, which we name "QHM-Converted".

We calculate the effective step size α of AggMo-Best: DISPLAYFORM0 We round up and use α = 28 as the learning rate for QHM-Converted.

From Section 7.1, our rule of thumb for QHM is ν = 0.7 and β = 0.999.

However, noting that this rule of thumb is toward replacing momentum/NAG with discount factor 0.9, and observing that the best NAG parameterization reported by Lucas et al. (2018) uses discount factor 0.99, we instead use ν = 0.97 and β = 0.999 for QHM-Converted.

In summary, the parameterization of QHM-Converted is α = 28, ν = 0.97, and β = 0.999, and no optimization or parameter sweeps on this task were performed to construct this parameterization.

Results FIG7 and TAB6 present the performance of AggMo-Best and QHM-Converted on the autoencoder task.

QHM-Converted outperforms AggMo-Best on the mean squared error (MSE) metric over the training, validation, and testing datasets.

To recap, we take the optimal AggMo parameterization from an extensive sweep, we convert that parameterization by hand to one for QHM, and we find that the latter outperforms the former on this autoencoder task.

These results indicate that using multiple momentum buffers with an arbitrary weighting scheme (i.e. AggMo with K > 2) provides negligible benefit over using a single slow-decaying momentum buffer with an appropriate weight (i.e. QHM with high β and appropriate ν).

Lucas et al. (2018) offer an interpretation of AggMo as passive damping for physical systems.

In this interpretation, fast-decaying momentum buffers "dampen" the oscillations of slow-decaying momentum buffers by providing velocity in an opposite direction.

In this context and considering these results, we conjecture that the current gradient already provides adequate damping for a slow-decaying momentum buffer, and that the damping provided by additional momentum buffers is of marginal value.

Lucas et al. (2018) propose an extension of AggMo which allows for alternate weighting schemes via separate per-buffer learning rates.

The learning rate becomes a vector γ ∈ R K and (23) becomes the following:

Lucas et al. FORMULA27 motivate this extension by the recovery of NAG.

In fact, we observe that this extension, with K = 2 and discount factors [0, β], recovers QHM as well.

In independent preliminary experiments on different tasks, we found that various alternate weighting schemes of multiple momentum buffers (i.e. various parameterizations of extended AggMo with K > 2) did not result in material improvements over the single momentum buffer.

However, this preliminary investigation was neither rigorous nor conclusive.

Lucas et al. (2018) do not empirically explore these alternate weighting schemes, and it is unclear how to do so both comprehensively and efficiently, since the number of hyperparameters scales linearly with the number of momentum buffers K.Toward improving the usability of extended AggMo, we suggest as future work to investigate theoretically grounded or empirically tractable methods to determine good weighting schemes for extended AggMo.

However, given the added costs and complexity of AggMo (both standard and extended), we surmise in the meantime that QHM may be preferable for most practical applications.

Environment All experiments use Python 3.7 and PyTorch 0.4.1 (Paszke et al., 2017) .

Experiments are run on a mix of NVIDIA P100 and V100 GPUs, along with a mix of CUDA 9.0 and 9.2.

Common settings (all experiments) Training occurs over 90 epochs (minibatch size 64).

The first epoch uses linear warmup of the learning rate α (i.e. α starts from zero and grows to its "regular" value by the end of the epoch).

Each training run uses a single GPU.Each parameterization is run 3 times with different seeds, and we report training loss, training top-1 error, and validation top-1 error.

We use a step decay schedule for the learning rate: α ∈ {1, 0.1, 0.01}. That is, the first 30 epochs use α = 1.0, the next 30 epochs use α = 0.1, and the final 30 epochs use α = 0.01.

We sweep over ν and β using the following two-dimensional grid: 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1} β ∈ {0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995} Note that this grid encapsulates numerous parameterizations of plain SGD, momentum, and NAG (specifically, all parameterizations with the β values enumerated above).

DISPLAYFORM0

We fix α = 0.001, β 2 = 0.999, and = 10 −8 , as suggested in Kingma & Ba (2015) .

We also fix ν 2 = 1.We sweep over ν 1 and β 1 using the same grid as for QHM's ν and β.

Model The model is multinomial logistic regression with pixel vector input.

Task The task is digit recognition over the EMNIST dataset -specifically, the digits subset (Cohen et al., 2017 ).Optimizer The model is optimized with QHM.

The optimization objective is cross-entropy loss, plus L2 regularization with coefficient 1 2 · 10 −4 .

Model Same as in Logistic-EMNIST-QHM.Task Same as in Logistic-EMNIST-QHM.Optimizer The model is optimized with QHAdam.

The optimization objective is the same as in Logistic-EMNIST-QHM.

Model The model is a multilayer perceptron (specifically, 3 layer feed forward network) with pixel vector input.

The hidden layer sizes are 200, 100, and 50 units, and all hidden units are tanh nonlinearities.

The final layer is followed by softmax.

Task Same as in Logistic-EMNIST-QHM.Optimizer Same as in Logistic-EMNIST-QHM.

29 These learning rates may seem high, but recall that the effective step size is identical to that of "typical", unnormalized momentum/NAG with α ∈ {0.1, 0.01, 0.001} and β = 0.9.

Model Same as in MLP-EMNIST-QHM.Task Same as in MLP-EMNIST-QHM.Optimizer Same as in Logistic-EMNIST-QHAdam.

Model The model is a 18-layer convolutional residual network with preactivations (He et al., 2016b) .Task The task is image recognition on the CIFAR-10 dataset (Krizhevsky, 2009 ).Optimizer The model is optimized with QHM.

The optimization objective is cross-entropy loss, plus L2 regularization with coefficient Optimizer (QHM) The non-baseline optimizer is QHM with ν = 0.7 and β = 0.999.

Following Section 7.1, we increase the learning rate (α) 10-fold.

All other details are identical to the baseline.

Evaluation For each optimizer, we run 3 seeds and report validation top-1 error.

Other details See RN50-ImageNet-QHM for implementation details.

Model The model is the GCNN-14 variant of the gated convolutional language model described in Dauphin et al. (2016) .Dataset The task is language modeling on the WikiText-103 language dataset (Merity et al., 2016) .

@highlight

Mix plain SGD and momentum (or do something similar with Adam) for great profit.

@highlight

The paper proposes simple modifications to SGD and Adam, called QH-variants, that can recover the “parent” method and a host of other optimization tricks.

@highlight

A variant of classical momentum which takes a weighted average of momentum and gradient update, and an evaluation of its relationships between other momentum based optimization schemes.