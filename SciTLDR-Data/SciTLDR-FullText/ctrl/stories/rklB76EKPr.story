Gradient clipping is a widely-used technique in the training of deep networks, and is generally motivated from an optimisation lens: informally, it controls the dynamics of iterates, thus enhancing the rate of convergence to a local minimum.

This intuition has been made precise in a line of recent works, which show that suitable clipping  can yield significantly faster convergence than vanilla gradient descent.

In this paper, we propose a new lens for studying gradient clipping, namely, robustness: informally, one expects clipping to provide robustness to noise, since one does not overly trust any single sample.

Surprisingly, we prove that  for the common problem of label noise in classification, standard gradient clipping does not in general provide robustness.

On the other hand, we show that  a simple variant of gradient clipping is provably robust, and corresponds to suitably modifying the underlying loss function.

This yields a simple, noise-robust alternative to the standard cross-entropy loss which performs well empirically.

In this paper, we propose a new lens with which to study gradient clipping, namely, robustness: intuitively, clipping the gradient prevents over-confident descent steps, which is plausibly beneficial in the presence of noise.

Given this intuition, our interest is whether gradient clipping can mitigate the problem of label noise in classification, which has received significant recent interest (Scott et al., 2013; Natarajan et al., 2013; Liu & Tao, 2016; Patrini et al., 2017; Ghosh et al., 2017; Han et al., 2018; Zhang & Sabuncu, 2018; Song et al., 2019; Thulasidasan et al., 2019; Charoenphakdee et al., 2019) .

We study this question, and provide three main contributions:

(a) we show that gradient clipping alone does not endow label noise robustness to even simple models.

Specifically, we show that under stochastic gradient descent with linear models, gradient clipping is related to optimising a "Huberised" loss (Lemma 1, 2).

While such Huberised losses preserve classification calibration (Lemma 3), they are not robust to label noise (Proposition 4).

(b) we propose composite loss-based gradient clipping, a variant that does have label noise robustness.

Specifically, for losses comprising a base loss composed with a link function, (e.g., softmax crossentropy), we only clip the contribution of the base loss.

The resulting partially Huberised loss preserves classification calibration (Lemma 6), while being robust to label noise (Proposition 7).

(c) we empirically verify that on both synthetic and real-world datasets, partially Huberised versions of standard losses (e.g., softmax cross-entropy) perform well in the presence of label noise ( ??5).

To illustrate the difference between standard and composite loss-based gradient clipping in (a) and (b), consider the pervasive softmax cross-entropy loss, viz.

the logistic loss for binary classification.

Recall that the logistic loss comprises the log-loss (or cross-entropy) composed with a sigmoid link.

The Huberised loss arises from loss-based gradient clipping, and linearises the entire logistic loss beyond a threshold.

The partially Huberised loss arises from composite loss-based gradient clipping, and linearises only the base loss (i.e., log-loss) beyond a threshold (right panel).

Once combined with the sigmoid link, the overall partially Huberised loss asymptotically saturates.

Our analysis in ??3, ??4 establishes that Huberised losses are not robust to label noise, but partially Huberised losses are (Proposition 4, 7).

In (a), we relate gradient clipping to loss "Huberisation", which linearises the logistic loss when it exceeds a fixed threshold.

In (b), we introduce composite loss-based gradient clipping, which is equivalent to a "partial loss Huberisation" that linearises only the cross-entropy loss but leaves the sigmoid link untouched.

Figure 1 illustrates the Huberised and partially Huberised logistic loss.

We provide background on gradient clipping, loss functions for classification, and label noise.

Gradient clipping.

Consider a supervised learning task over instances X and labels Y, where we have a family of models indexed by ?? ??? ??, and the quality of a particular model is measured by a loss function ?? : X ?? Y ??? R. The gradient for a mini-batch {(x n , y n )} N n=1 is g(??)

One may instead compute the clipped gradient, which for user-specified threshold ?? > 0 is

if ||w|| 2 ??? ?? w else.

Employing??? ?? (??) for optimisation corresponds to clipped gradient descent.

This is closely related to normalised gradient descent (NGD) (Hazan et al., 2015) , wherein one employsg(??)

||g(??)||2 .

Hazan et al. (2015) showed that NGD can lead to convergence for a wider class of functions than standard gradient descent.

Levy (2016) showed that NGD can escape saddle points for non-convex problems.

Zhang et al. (2019) showed that gradient clipping can lead to accelerated convergence over gradient descent.

Gradient clipping has also been explored for privacy (Abadi et al., 2016; Pichapati et al., 2019) , motivated by it preventing any single instance from dominating parameter updates.

Loss functions.

In binary classification, one observes samples from a distribution D over X ?? {??1}, and seeks a predictor f : X ??? R with low risk R(f ) .

= E (y, f (x)) according to a loss which, in an overload of notation, we denote : {??1} ?? R ??? R + .

For the zero-one loss 01 (y, f ) = y ?? f < 0 , R(f ) is known as the misclassification risk.

Rather than directly using 01 , for computational convenience one often employs a margin loss (y, f ) = ??(y ?? f ), for convex ?? : R ??? R + .

We say that ?? is classification calibrated (Zhang, 2004; Bartlett et al., 2006) if driving the excess risk over the Bayes-optimal predictor for ?? to zero also drives the excess risk for 01 to zero; that is, minimising the ??-risk is statistically consistent for classification.

A canonical example is the hinge loss ??(z) = [1 ??? z] + .

We call ?? admissible if it is "well-behaved" in the sense of being bounded from below, strictly convex, continuously differentiable, non-increasing, and classification calibrated.

We say that ?? is proper composite (Reid & Williamson, 2010; Nock & Nielsen, 2009 ) (or for brevity composite) if its minimising scores can be interpreted as probabilities.

"Composite" here refers to such losses comprising a base loss ?? composed with an invertible link function F : R ??? [0, 1].

While ?? accepts as input real-valued scores (e.g., the final layer logits of a neural network), these are internally converted to probabilities via F .

A canonical example is the logistic loss ??(z) = ??? log F (z) with F : z ??? ??(z) for sigmoid ??(z) .

= (1 + e ???z ) ???1 .

The multiclass analogue is the softmax cross-entropy loss, wherein the sigmoid becomes a softmax.

See Appendix B for a more technical discussion.

Learning under label noise.

In classification under label noise, one has samples from distribution D where PD(y | x) is a noisy version of P D (y | x), e.g., all labels are corrupted with a fixed constant probability.

The goal remains to ensure low risk with respect to the clean D. This problem has a long history in statistics (Ekholm & Palmgren, 1982; Copas, 1988) , and has emerged as a topic of recent interest in machine learning.

There has been particular interest in the problem of learning under symmetric label noise, wherein all instances have a constant probability of their labels being flipped uniformly to any of the other classes.

Scott et al. (2013) ; Katz-Samuels et al. (2019) proposed a framework for analysing the more general setting of class-conditional label noise.

Takenouchi (Malach & Shalev-Shwartz, 2017; Han et al., 2018; Song et al., 2019) to abstention (Thulasidasan et al., 2019) .

We first show that gradient clipping in general does not endow robustness to label noise, even in simple settings.

Specifically, we establish that stochastic gradient clipping with linear models is equivalent to modifying the underlying loss (Lemma 1).

This modified loss is closely related to a Huberised loss (Lemma 2), which is equivalent to loss-based gradient clipping (L-gradient clipping).

Unfortunately, Huberised losses, and thus L-gradient clipping, are not robust to label noise (Proposition 4).

Consider a binary classification problem, with Y = {??1}. Suppose we use a scoring model s ?? (x) for ?? ??? ?? with margin m ?? (x, y) .

= y ?? s ?? (x), and margin loss ?? (x, y)

Now suppose we perform stochastic gradient descent (i.e., we use N = 1 in (1)) and use a linear scorer s ?? (x) = ?? T x. Here, gradient clipping of (3) is equivalent to modifying the loss as follows.

Lemma 1.

Pick any admissible margin loss ??, and ?? > 0.

Then, for loss ?? (x, y) .

= ??(m ?? (x, y)) with linear scorer, the clipped loss gradient (2) is equivalent to the gradient of a modified loss?? :

The loss?? in (4) is intuitive: if the derivative of the original loss ?? exceeds a certain effective clipping threshold, we replace ?? with a linear loss function.

This effective threshold ?? ?? ( ???m ?? (x, y) 2 ) ???1 takes into account the instance-dependent margin gradient, viz.

x 2 for linear s ?? (x).

We will comment in ??4.4 as to the properties of gradient clipping in more general scenarios.

For the moment, we note that Lemma 1 is closely related to the "Huber loss" (Huber, 1964) ,

This replaces the extremes of the square loss with the absolute loss.

Both the absolute and Huber losses are widely employed in robust regression (Huber, 1981 , Chapter 3), (Hampel et al., 1986, pg.

100) .

Note that (4) slightly differs from (5) as the effective clipping threshold is instance-dependent.

One may nonetheless arrive at Huber-style losses via a variant of gradient clipping; this connection will prove useful for our subsequent analysis.

Per (3), gradient clipping involves using

wherein the clipping considers the margin gradient.

Consider now the following loss-based gradient clipping (L-gradient clipping), wherein we clip only the contribution arising from the loss:

Compared to (6), we effectively treat the margin gradient norm ???m ?? (x, y) 2 as constant across instances, and focus on bounding the loss derivative.

The latter may be a significant component in the gradient norm; e.g., for linear models, ???m ?? (x, y) 2 = x 2 is often bounded.

Observe further that for linear models with x 2 ??? R across instances, (7) is a rescaled version of the clipped gradient:

Interestingly, L-gradient clipping equivalently uses the following "Huberised" version of the loss.

Lemma 2.

Pick any admissible margin loss ?? : R ??? R + with Fenchel conjugate ?? * , and ?? > 0.

Then, the clipped gradient in (7) is equivalent to employing a Huberised loss function?? ?? such that

Evidently,?? ?? linearises ?? once its derivative is sufficiently large, akin to the Huber loss in (5).

One may verify that for ??(z) = (1 ??? z) 2 , one arrives exactly at (5).

Example 3.1: For the logistic loss ??(z) = log(1 + e ???z ), the Huberised loss for ?? ??? (0, 1) is

Per Figure 1a , this linearises ?? beyond a fixed threshold.

See Appendix C for further illustrations.

The use of Huberised losses in classification is not new (Zhang et al., 2002; Zhang & Johnson, 2003; Rosset & Zhu, 2007) .

However, we now provide a novel study of their label noise robustness.

Before studying the effect of label noise on Huberised losses, it is apposite to consider whether they are suitable for use even in the absence of noise.

One way to formalise this is to ask whether the losses maintain classification calibration, in the sense of ??2.

This is a minimal requirement on a loss to be useful for classification (Zhang, 2004; Bartlett et al., 2006) .

One may further ask whether the losses preserve class-probabilities, i.e., are proper composite in the sense of ??2.

It is desirable to preserve this key trait of losses such as the logistic loss.

The following clarifies both points.

Lemma 3.

Pick any admissible margin loss ?? and ?? > 0.

Then, the Huberised loss?? ?? in (8) is classification calibrated.

If ?? is proper composite and ?? ??? ????? (0), then?? ?? is also proper composite.

L-gradient clipping is thus benign for classification, generalising Rosset & Zhu (2007, Section 3.4) , which was for the square-hinge loss.

Interestingly, for composite ?? and small ?? , the proof reveals that ?? ?? has a non-invertible link function.

Intuitively, such?? ?? are effectively linear, and linear losses are unsuited for estimating probabilities Charoenphakdee et al., 2019) .

We now turn to our central object of inquiry: does gradient clipping endow robustness to label noise?

To study this, we consider L-gradient clipping, which as noted above is a special case of gradient clipping under linear models with constant ???m ?? (x, y) 2 .

Since L-gradient clipping is in turn equivalent to using a Huberised loss, we may study the robustness properties of this loss.

Surprisingly, when our loss is convex (e.g., softmax cross-entropy), Huberised losses are not robust to even very simple forms of label noise.

Essentially, since these losses are still convex, they can still be affected by errant outlier observations.

Formally, using a result of Long & Servedio (2010) (see Appendix F.1), we arrive at the following.

Proposition 4.

Pick any admissible margin loss ?? and ?? > 0.

Then, ??? a separable distribution for which the optimal linear classifier under?? ?? is equivalent to random guessing under symmetric noise.

To situate Proposition 4 in a broader context, we note that for regression problems, it is well known that the Huber loss is susceptible to "high leverage" outliers (Hampel et al., 1986, pg.

313) , (Rousseeuw & Leroy, 1987, pg.

13) , (Maronna et al., 2019, pg.

104) , i.e., extremal instances which dominate the optimal solution.

Proposition 4 complements these for the case of label noise in classification.

Given that gradient clipping does not endow label noise robustness, how else might we proceed?

In a regression context, the outlier vulnerability of the Huber loss can be addressed by using a trimmed average of the loss (Rousseeuw & Leroy, 1987; Bhatia et al., 2015; Yang et al., 2018) .

Such ideas have been successfully explored for label noise problems (Shen & Sanghavi, 2019) .

We will however demonstrate that a simple variant of clipping yields a loss that does possess label noise robustness.

We now show that noise robustness can be achieved with CL-gradient clipping, a variant wherein for composite losses (e.g., softmax cross-entropy), we perform partial Huberisation of the base loss only.

Consider a composite margin loss ?? (x, y)

, where ?? = ?? ??? F for some base loss ?? and invertible link F :R ??? [0, 1]; e.g., the logistic loss has ??(u) = ??? log u and F (z) = ??(z).

We can interpret p ?? (x, y) .

= F (m ?? (x, y)) as a probability estimate; e.g., p ?? (x, 1) = ??(m ?? (x, 1)) is the probability of x being positive.

Now, rewriting ?? (x, y) = ??(p ?? (x, y)), we may express (3) as:

L-gradient clipping in (7) was defined as only clipping ?? = (?? ??? F ) above, which ensures that the resulting Huberised loss is Lipschitz.

Typically, however, F is already Lipschitz; e.g., this is the case for the commonly used sigmoid or softmax link.

This suggests the following composite loss-based gradient clipping (CL-gradient clipping), wherein we only clip the derivative for the base loss ??:

As before, CL-gradient clipping corresponds to optimising a new, "partially Huberised" loss.

Lemma 5.

Pick any admissible, composite margin loss ?? = ?? ??? F , and ?? > 0.

Then, the clipped gradient in (9) is equivalent to employing a partially Huberised loss?? ?? =?? ?? ??? F , wher???

Compared to the Huberised loss (7), the partially Huberised loss only linearises the base loss ??, while retaining the link F .

Consequently, the composite loss?? ?? behaves like the link beyond a certain threshold, and will thus be bounded:

Example 4.1: For the logistic loss, ??(z) = ??(F (z)) with ??(u) = ??? log u and F (z) = ??(z),

Note that partial Huberisation readily generalises to a multi-class setting.

Indeed, suppose we have softmax probability estimates p ?? (x, y) ??? exp(m ?? (x, y)).

Then, whereas the softmax cross-entropy employs ?? (x, y) = ??? log p ?? (x, y), our partially Huberised softmax cross-entropy for ?? > 1 is

Following ??3.2, we establish that CL-gradient clipping is always benign from a classification perspective, and provided ?? is sufficiently large, from a probability estimation perspective as well.

As before, we do this by exploiting the equivalence of CL-gradient clipping to a partially Huberised loss.

Lemma 6.

Pick any admissible composite margin loss ?? = ?? ??? F and ?? > 0.

Then, the loss?? ?? in (10) is classification calibrated.

If further ?? ??? ????? ( 1 /2), then?? ?? is also proper composite.

We now show that partially Huberised losses have an important advantage over Huberised losses: under symmetric label noise, the optimal solution on the clean distribution cannot be too far away from the optimal solution on the noisy distribution.

This implies that label noise (such as that considered in Proposition 4) cannot have an excessively deleterious influence on the loss.

Proposition 7.

Pick any proper loss ?? and ?? > 0.

Let f * be the risk minimiser of?? ?? on the clean distribution.

For any non-trivial level of symmetric label noise, let reg ?? (f * ) denote the excess risk of f * with respect to?? ?? on the noisy distribution.

Then, there exists C > 0 such that

Note that by van Rooyen et al. (2015, Proposition 4) , it is impossible for the above bound to hold with C = 0 without using a linear loss.

Nonetheless, by virtue of partially Huberised losses being partially linear, we are able to bound the degradation under label corruption.

The saturating behaviour of the partially Huberised loss also implies robustness to outliers in feature space; see Appendix E.

The partially Huberised log loss in (10), (11) can be related to a family of losses studied in several works (Ding & Vishwanathan, 2010; Hung et al., 2018; Zhang & Sabuncu, 2018; Amid et al., 2019b; a) :

3 for an illustration of ?? ?? .

There are two similarities between (11) and ?? ?? .

First, both proposals interpolate between the log and linear losses: when ?? ??? 0 + , ?? ?? approaches the log loss, and when ?? = 1, ?? ?? equals the linear loss.

Second, both proposals modify the base loss, allowing the link F to be chosen independently.

In particular, one may use the heavy-tailed F of Ding & Vishwanathan (2010) ; Amid et al. (2019b) in conjunction with our partially Huberised loss.

One difference between (11) and ?? ?? is that the partially Huberised loss is exactly linear for a suitable region; consequently, it is guaranteed to be Lipschitz, unlike ?? ?? .

This can be understood in terms of the loss gradients: for a class-probability estimate p ?? (x, y), let ?? (x, y)

.

Both gradients thus take into account whether a sample is "informative", in the sense of being poorly-predicted (p ?? (x, y) ??? 0).

Further, to guard against such samples being the result of label noise, both ensure this influence is not overwhelming, but in different ways: the generalised cross-entropy softens the influence, while still allowing it to be unbounded as p ?? (x, y) ??? 0.

On the other hand, the partially Huberised loss enforces a hard cap of ?? ???1 on the influence.

This is to be contrast with a truncated loss also considered in Zhang & Sabuncu (2018) , which enforces a hard cap on the loss, thus completely discarding poorly-predicted instances.

Table 1 summarises our results, highlighting the perspective of gradient clipping as equivalently modifying the loss.

Before proceeding, we make some qualifying comments.

First, our analysis has assumed symmetric label noise.

Often, one encounters asymmetric or instance-dependent noise .

While corresponding guarantees for the linear loss may be ported over to the partially Huberised loss, they require stronger distributional assumptions (Ghosh et al., 2015) .

Second, Proposition 4 exhibits a specific distribution which defeats the Huberised loss under linear models.

In practice, distributions may be more benign (Patrini et al., 2016) , and models are often nonlinear, meaning that Huberised losses (and thus gradient clipping) are thus unlikely to succumb as extremely to label noise as Proposition 4 suggests.

The aim of ??4 is however to establish that a simple modification of clipping avoids worst-case degradation, without adding significant complexity.

Equivalent loss Reference Label noise robust?

(Proposition 7) Table 1 : Summary of types of gradient clipping considered in this paper.

We consider binary classification problems involving a labelled example (x, y), parametrised scoring function s ?? (x) with margin m(??) .

= y ?? s ?? (x), and differentiable composite margin loss ??(z).

This loss internally converts scores to probabilities p(??) .

= F (m(??)) for link function F (??), which is evaluated with some base loss ??; i.e., ?? = ?? ??? F .

Gradient clipping applies to the full loss, i.e., ??(m(??)).

L-gradient clipping applies only to the composite loss, leaving the score untouched; this is equivalent to using a Huberised loss.

CL-gradient clipping applies only to the base loss, leaving the link untouched; this is equivalent to using a partially Huberised loss.

Only the latter has robustness guarantee under symmetric label noise.

Third, for minibatch size N > 1, the effect of clipping is not a simple loss modification, since the loss gradients for each sample will be modified by the entire minibatch loss gradient norm g(??) 2 .

Since this minibatch is randomly drawn, one cannot mimic gradient clipping by a simple modification of the loss function.

However, the results for N = 1 suffice to establish that gradient clipping cannot in general endow robustness.

One may use our partially Huberised loss in conjunction with minibatch gradient clipping, to potentially obtain both robustness and optimisation benefits.

Finally, partially Huberised losses such as (12) require setting a hyperparameter ?? (e.g., by crossvalidation), similar to ?? in the generalised cross-entropy per ??4.3.

Intuitively, the optimal ?? trades off the noise-robustness of the linear loss, and the gradient informativeness of the base loss (per the discussion in ??4.3).

Setting ?? to be large tacitly assumes that one's samples are largely noise-free.

We now present experiments illustrating that: (a) we may exhibit label noise scenarios that defeat a Huberised but not partially Huberised loss, confirming Propositions 4, 7, and (b) partially Huberised versions of existing losses perform well on real-world datasets subject to label noise.

Synthetic datasets.

Our first experiments involve two synthetic datasets, which control for confounding factors.

We begin with a setting from Long & Servedio (2010), comprising a 2D linearly separable distribution.

(See Appendix F.1 for an illustration.)

We draw N = 1, 000 random samples from this distribution, and flip each label with ?? = 45% probability.

We train a linear classifier to minimise one of several losses, and evaluate the classifier's accuracy on 500 clean test samples.

We compare the logistic loss with its Huberised and partially Huberised versions, using ?? = 1 and ?? = 2 respectively.

Figure 2a presents the results over 500 independent trials.

The logistic loss and its Huberised counterpart suffer significantly under noise, while the partially Huberised loss often achieves perfect near-discrimination.

This confirms that in the worst case, L-gradient clipping may succumb to noise, while CL-gradient clipping performs well in the same scenario.

We next consider a 1D setting based on Ding (2013, Section 3.2.3), comprising 10, 000 linearly separable "inliers" and 50 "outliers".

Assuming the use of a linear model parameterised by scalar 91.6 ?? 0.1 88.6 ?? 0.0 83.6 ?? 0.1 72.2 ?? 0.0 PHuber-GCE ?? = 10 92.0 ?? 0.1 88.5 ?? 0.1 80.8 ?? 0.1 62.6 ?? 0.2 CIFAR-100 CE 66.6 ?? 1.4 49.7 ?? 0.3 29.9 ?? 0.9 11.4 ?? 0.2 CE + clipping 28.8 ?? 0.1 20.6 ?? 0.4 14.7 ?? 0.6 9.0 ?? 0.4 Linear 12.1 ?? 1.6 6.6 ?? 1.2 5.7 ?? 0.9 3.6 ?? 0.1 GCE 70.1 ?? 0.1 63.9 ?? 0.1 52.0 ?? 0.2 29.9 ?? 0.5 PHuber-CE ?? = 10 66.2 ?? 1.5 56.2 ?? 2.2 44.4 ?? 0.7 18.5 ?? 0.4 PHuber-GCE ?? = 10 69.8 ?? 0.2 64.4 ?? 0.4 52.4 ?? 0.2 31.5 ?? 0.8 Table 2 : Test set accuracy where the training labels are corrupted with probability ??.

We report the mean and standard error over 3 trials.

The highlighted cells are the best performing loss at a given ??.

"PHuber" here refers to our partial Huberisation from ??4, which is equivalent to a variant of gradient clipping.

?? ??? R, we plot the empirical risk of the samples with and without the outlier observations as ?? is varied.

Figure 2b shows that the logistic loss and its Huberised variant are strongly affected by the outliers: their optimal solution goes from ?? * = +??? to ?? * = 0.

However, the partially Huberised loss is largely immune to the outliers.

Appendix F contains additional synthetic experiments.

Real-world datasets.

We now demonstrate that partially Huberised losses perform well with deep neural networks trained on MNIST, CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton, 2009 ).

For MNIST, we train a LeNet (Lecun et al., 1998) using Adam with batch size N = 32, and weight decay of 10 ???3 .

For CIFAR-10 and CIFAR-100, we train a ResNet-50 (He et al., 2016) using SGD with momentum 0.1, weight decay of 5 ?? 10 ???3 , batch normalisation, and N = 64, 128 respectively.

For each dataset, we corrupt the training labels with symmetric noise at flip probability ?? ??? {0.0, 0.2, 0.4, 0.6}. We compare the test set accuracy of various losses combined with a softmax link.

Our baseline is the cross-entropy loss (CE).

As representative noise-robust losses, we consider the linear or unhinged loss Ghosh et al., 2017) , and the generalised cross-entropy (GCE) with ?? = 0.7, following Zhang & Sabuncu (2018) .

We additionally assess global gradient clipping (with ?? = 0.1) of the CE, which per ??3 is akin to a Huberised loss.

We apply our partial Huberisation of (10) to the CE (12) ("PHuber-CE"), and the GCE ("PHuber-GCE").

The latter highlights that partial Huberisation is not tied to the cross-entropy, and is applicable even on top of existing noise-robust losses.

Recall that partial Huberisation offers a choice of tuning parameter ?? , similar to the ?? parameter in GCE, and noise-rate estimates in loss-correction techniques more generally.

For each dataset, we pick ?? ??? {2, 10} (equivalently corresponding to probability thresholds 0.5 and 0.1 respectively) so as to maximise accuracy on a validation set of noisy samples with the maximal noise rate ?? = 0.6; the chosen value of ?? was then used for each noise level.

Tuning ?? separately for each setting of the noise rate ?? can be expected to help performance, at the expense of increased computational cost.

Recall also that as ?? ??? 1, partial Huberisation mimics using the base loss, while as ?? ??? +???, partial Huberisation mimics using the linear loss; our hypothesis is that an intermediate ?? can attain a suitable balance between noise robustness, and gradient informativeness.

Table 2 shows that in the noise-free case (?? = 0.0), all methods perform comparably.

However, when injecting noise, accuracy for the CE degrades dramatically.

Further, gradient clipping sometimes offers improvements under high noise; however, the performance is far inferior to other losses, which is in keeping with their robustness guarantees.

Indeed, the linear loss, which is provably robust to symmetric noise, generally performs well even when ?? = 0.6.

However, optimisation under this loss is more challenging, since the gradient does not account for instances' importance (per ??4.3).

This is particularly reflected on the CIFAR-100 dataset, where this loss suffers to learn even under no noise.

The GCE and partially Huberised losses do not suffer from this issue, even at high noise levels.

Generally, the partially Huberised losses are competitive with or improve upon the counterpart losses they build upon.

In particular, the partially Huberised CE performs much better than the CE under high noise, while the partially Huberised GCE slightly bumps up the GCE numbers on CIFAR-100.

This indicates that partially Huberised may be useful in combining with generic base losses to cope with noise.

We reiterate here that our partially Huberised loss may be used in conjunction with other ideas, e.g., pruning (Zhang & Sabuncu, 2018) , consensus (Malach & Shalev-Shwartz, 2017; Han et al., 2018) , or abstention (Thulasidasan et al., 2019) .

We leave such exploration for future work.

We established that gradient clipping by itself does not suffice to endow even simple models with label noise robustness; however, a simple variant resolves this issue.

Experiments confirm that our composite loss-based gradient clipping performs well on datasets corrupted with label noise.

One interesting direction for future work is to analyse the behaviour of gradient-clipping inspired losses for the more general problem of distributionally robust learning (Shafieezadeh-Abadeh et al., 2015; Namkoong & Duchi, 2016; Sinha et al., 2018; Duchi & Namkoong, 2019) .

Similarly, for the linear loss function lin (x, y; ??) .

= ???m ?? (x, y), we have ??? lin (x, y; ??) = ??????m ?? (x, y).

To compute the normalised gradient, we need

since ?? (z) < 0 by assumption that ?? is admissible, and thus decreasing.

Consequently, if N = 1,

else.

Assuming a linear scorer s(x; ??) = ?? T x, the score gradient ???s(x; ??) = x, which is independent of ??.

The clipped gradient thus corresponds to the gradient under a "Huberised" loss function

else.

Proof of Lemma 2.

Since ?? is strictly convex and decreasing, it must be strictly decreasing.

We have

where (?? ) ???1 exists since ?? is strictly inceasing by definition of strict convexity.

Now, by definition, the Huberised loss is

and so??

which exactly equals clip ?? (?? (z)).

We remark here that?? ?? is continuous, since for any convex conjugate,

Plugging in u = ?? , at the intersection point z 0 .

???1 (????? ) of the two pieces of the function,

Proof of Lemma 3.

For admissible ??, the Huberised loss?? ?? of (8) is trivially convex, differentiable everywhere, and decreasing.

In particular, we must have?? ?? (0) < 0, and so?? ?? must be classification calibrated by Bartlett et al. (2006, Theorem 2) .

As an illustration, Figure 3 shows the minimiser of the conditional risk,

Note that this quantity must be non-negative if and only if ?? > 1 /2 for a loss to be classification calibrated.

This is easily verified to be true for the Huberised logistic loss, regardless of ?? .

is strictly monotone and continuous.

Continuity is immediate; for monotonicity, observe that by definition,

For brevity, let z 0 .

???1 (????? ), where we take

Thus, the quantity of interest is??

Since ?? is strictly convex, ?? is strictly decreasing and thus invertible.

Thus, the ratio??

is invertible, provided z 0 ??? 0, i.e., ?? ??? ????? (0).

Consequently,?? ?? is proper composite when ?? ??? ????? (0).

To intuit the need for the restriction on ?? , observe that by (Reid & Williamson, 2010, Corollary 12 ) the tentative link function for the loss is

When ?? < ????? (0), the above is seen to be non-invertible.

See also Appendix C for an illustration of this link function for the logistic loss.

Proof of Proposition 4.

In order to apply Long & Servedio (2010, Theorem 2), we simply need to check that the loss?? ?? is a convex potential in the sense of Long & Servedio (2010, Definition 1).

This requires that?? ?? is convex, non-increasing, continuously differentiable, and asymptotes to zero (or equally, is bounded from below).

Each of these is satisfied by assumption of ?? being admissible.

Proof of Lemma 5.

By Lemma 2, we may write clip ?? (?? (u)) as the derivative of a partially Huberised base loss?? ?? given b???

This induces a composite margin loss?? ??

, and defin??? ?? (x, y; ??)

.

=?? ?? (m ?? (x, y)).

The gradient under this loss is

Thus, CL-gradient clipping is equivalent to using the loss?? ?? .

Proof of Lemma 6.

We proceed in a similar manner to Lemma 3: to show that the loss is proper, we must establish that??

is invertible.

By Reid & Williamson (2010, Corollary 14) , for the margin loss ?? to be proper composite with link F , it must be true that F satisfies the symmetry condition F (???z) = 1 ??? F (z).

We thus have??

Since F is invertible by assumption, the above is invertible if and only if??

This quantity is invertible provided u 0 ??? 1 2 , i.e., ?? ??? ????? ( 1 /2).

A subtlety, however, is that the above does not necessarily span the entire range of [0, +???]; consequently,?? ?? itself is proper composite, with a link function of its own.

Even when ?? is small, one may verify that the loss?? ?? is nonetheless classification calibrated: this is because for any ?? ??? [0, 1], the minimiser z * (??) of the conditional risk must satisfy the stationarity condition

We thus need to find a suitable z * such that the left hand side equates to a given constant.

Now, for any C = 1 there is a unique u such that??

= C. One may verify that this z * > 0 ?????? ?? > 1 2 ; for example, see Figure 4 , which visualises the risk minimiser for various values of ?? .

Thus, the sign of the minimising score conveys whether or not the positive class-probability is dominant; thus, the loss is classification calibrated.

Proof of Proposition 7.

Let R ?? (f ) denote the risk on the clean distribution of a predictor f with respect to the partially Huberised loss?? ?? with parameter ?? .

Similarly, letR ?? (f ) denote the risk on the noisy distribution.

Following Ghosh et al. (2015) ; van Rooyen et al. (2015), we hav??

That is, the risk on the noisy distribution equals a scaled version of the risk on the clean distribution, plus an additional term.

This term is a constant independent of f if and only if?? ?? satisfies the symmetry condition of Ghosh et al. (2015) , namely,?? ?? (u) +?? ?? (1 ??? u) = C for some constant C.

Even when the symmetry condition does not hold, one may nonetheless aim to bound this additional term as follows.

For simplicity, we restrict attention here to ?? being the log or cross-entropy loss ??(u) = ??? log u. By definition, for any f ??? (0, 1),

?? .

Evidently, all piecewise functions involved are bounded on the respective intervals.

For example,

By taking the maximum of each of the quantities on the right hand side -which are constants depending on ?? -we may thus find constants C 1 , C 2 such that

Now let f * denote the minimiser of the clean risk R ?? (f ), andf * the minimiser of the noisy risk R ?? (f ).

Then, using each of the above inequalities,

where the last inequality is because R ?? (f * ) ??? R ?? (f * ) by definition of f * .

The claim follows.

Beyond requiring classification-calibration, it is often desirable to use classifier outputs as valid probabilities.

Proper losses (Savage, 1971; Schervish, 1989; Buja et al., 2005) ?? : {??1}??[0, 1] ???R + are the core losses of such class-probability estimation tasks, for which

Equation 13 stipulates that when using ?? to distinguish positive and negative labels, it is optimal to predict the positive class-probability.

Typically, it is more useful to work with losses that accept real-valued scores, e.g. as output by the pre-activation of the final layer of a neural network.

To this end, proper composite losses (Reid & Williamson, 2010; Nock & Nielsen, 2009)

Given a proper loss ?? and "symmetric" link F with F (???v) = 1???F (v), the loss (y, v) = ??(y, F (v)) defines a margin loss (Reid & Williamson, 2010, Corollary 14) .

Proper composite losses may also be extended to multiclass settings in the natural way (Gneiting & Raftery, 2007; Williamson et al., 2016) : one now defines a proper loss ?? :

where K is the number of classes and ??? K denotes the K-simplex.

A proper composite loss may be defined using a link F :

.

Combined with the log-loss ??(y, p) = ??? log p y , this yields the standard softmax cross-entropy loss.

We illustrate the Huberised, partially Huberised, and generalised cross-entropy losses as their underlying tuning parameters are varied.

Additionally, we illustrate the link functions that are implicit in each of the losses, which illustrates that they may be non-invertible if ?? is too large.

C.1 HUBERISED LOSS Figure 5 illustrates the Huberised version of the logistic loss, and its derivative.

Following the proof of Lemma 3, for ?? ??? (0, 1) and z 0 = ????? ???1 (?? ), the Huberised loss?? ?? has an implicit link function (see Figure 6 )

Compared to the standard sigmoid, the Huberised link saturates more slowly as ?? is decreased.

Note that when ?? ??? 1 2 , the link function is not invertible everywhere: this results in the loss not being proper composite per our definition.

Figure 7 illustrates the partially Huberised version of the logistic loss, as well as the base log-loss.

Following the proof of Lemma 6, the partially Huberised loss?? ?? has implicit link functio?? Figure 9 illustrates the link function.

The logistic loss has ??(u) = ??? log u, and so ?? ( 1 /2) = ???2.

For ?? = 1, the link function will be non-invertible everywhere, which is expected since the loss here is the linear loss, which is not suitable for class-probability estimation.

For ?? ??? (1, 2), the link function will be invertible for p / ??? 1 ??? 1 ?? , 1 ?? .

Intuitively, the case ?? ??? (1, 2) corresponds to the linear regions of the losses on positive and negative instances crossing over.

For ?? ??? 2, the link function will always be invertible.

It may be observed that partial Huberisation causes the link function to saturate at values [1/(1 + ?? ), ?? /(1 + ?? )]: this does not affect classification calibration, but does imply that rescaling is necessary in order to intepret the output probabilities.

Figure 8 illustrates the base ?? ?? loss, and its composition with a sigmoid link function. , the link function is not invertible everywhere.

= ??(y ?? ?? T x) with empirical risk minimiser?? N on a sample {(x n , y n )} N n=1 , suppose that the sample is corrupted with an outlier (x , y ).

One would like to ensure that?? N is not swayed by making x 2 arbitrarily large: that is,

??? (x n , y n ;?? N ) + ??? (x , y ;?? N ) = 0, or equivalently, lim x 2 ???+??? ??? (x , y ;?? N ) = 0.

Fortunately, the saturating behaviour of the partially Huberised loss affords this, as we now show.

Proposition 8.

Pick any convex, differentiable, proper composite margin loss ??, whose link F satisfies lim z????????? z ?? F (z) = 0.

For any ?? > 0, let?? ?? (x, y; ??)

.

=?? ?? (y ?? ?? T x) be the ?? -partially Huberised loss under a linear model.

Then, for any (x, y) and ?? such that ?? 2 < +??? and ?? T x = 0,

Proof of Proposition 8.

By definition,

Thus,

where ?? x,?? denotes the angle between x and ??.

If ?? T x = 0, then cos ?? x,?? = 0, and so the third term is finite.

Thus, ????? ?? (x, y; ??) 2 ??? lim z???????? |z|???? ?? (z), depending on the sign of y ???? T x. By definition of?? ?? , the derivative of the loss asymptotes to either F (v), or ?? (v).

Now, lim z????????? z ?? F (v) = 0 by assumption, and lim z???+??? z ?? ?? (v) = 0 since ?? is convex, the claim is shown.

We provide some more details regarding the synthetic data used in the body, as well as results on an additional two-dimensional synthetic dataset.

The problem considered in Long & Servedio (2010) (see Figure 10 ) comprises a distribution concentrated on six atoms {??(1, 0), ??(??, 5??), ??(??, ?????)} ??? R 2 for some ?? > 0; we chose ?? = 1 24 .

An instance (x 1 , x 2 ) is labelled as y = x 1 ??? 0 .

The instances are weighted so that the first four atoms have probability mass 1 8 , and the last two atoms mass 1 4 .

We modify this distribution slightly by treating the atoms as means of isotropic Gaussians, and treating the marginal distribution over instances to be a mixture of these Gaussians with mixing weights given by the corresponding probability masses of the atoms.

For the experiment involving outliers in feature space, the data comprises points on the line.

Positively labelled samples are drawn from a unit variance Gaussian centered at (1, 1), with positively labeled outliers drawn from (???200, 1).

Negatively labelled samples comprise the negation of all points.

We learn an unregularised linear classifier from this data, which corresponds to a single scalar ??.

We further illustrate the differences amongst methods on a 2D dataset inspired by Amid et al. (2019a) .

The data comprises 500 points, falling into two bands (Figure 11 ).

The decision boundaries for various losses, when trained with a linear model using explicit quadratic features, are shown in Figure 12 .

We subject the data to 45% symmetric label noise.

We see that the logistic and generalised cross-entropy losses see marked changes in their decision boundaries.

By contrast, the partially Huberised loss maintains the correct classification boundary. (2010), which defeats any member of a broad family of convex losses.

The data comprises six points, with the blue points labelled positive, and the red points labelled negative.

The two "fat" points have twice as much probability mass as their "thin" counterparts.

While the dataset is trivially linearly separable, minimising a broad range of convex losses with a linear model under any non-zero amount of symmetric label noise results in a predictor that is tantamount to random guessing.

On the clean version of the data, all losses yield roughly equitable decision boundaries.

However, when adding 45% symmetric label noise, the logistic loss sees marked changes to its boundary.

The partially Huberised loss maintains the correct classification boundary.

<|TLDR|>

@highlight

Gradient clipping doesn't endow robustness to label noise, but a simple loss-based variant does.