The importance-weighted autoencoder (IWAE) approach of Burda et al. defines a sequence of increasingly tighter bounds on the marginal likelihood of latent variable models.

Recently, Cremer et al. reinterpreted the IWAE bounds as ordinary variational evidence lower bounds (ELBO) applied to increasingly accurate variational distributions.

In this work, we provide yet another perspective on the IWAE bounds.

We interpret each IWAE bound as a biased estimator of the true marginal likelihood where for the bound defined on $K$ samples we show the bias to be of order O(1/K).

In our theoretical analysis of the IWAE objective we derive asymptotic bias and variance expressions.

Based on this analysis we develop jackknife variational inference (JVI), a family of bias-reduced estimators reducing the bias to $O(K^{-(m+1)})$ for any given m < K while retaining computational efficiency.

Finally, we demonstrate that JVI leads to improved evidence estimates in variational autoencoders.

We also report first results on applying JVI to learning variational autoencoders.



Our implementation is available at https://github.com/Microsoft/jackknife-variational-inference

Variational autoencoders (VAE) are a class of expressive probabilistic deep learning models useful for generative modeling, representation learning, and probabilistic regression.

Originally proposed in BID8 and BID22 , VAEs consist of a probabilistic model as well as an approximate method for maximum likelihood estimation.

In the generative case, the model is defined as DISPLAYFORM0 where z is a latent variable, typically a high dimensional vector; the corresponding prior distribution p(z) is fixed and typically defined as a standard multivariate Normal distribution N (0, I).

To achieve an expressive marginal distribution p(x), we define p θ (x|z) through a neural network, making the model (1) a deep probabilistic model.

Maximum likelihood estimation of the parameters θ in (1) is intractable, but BID8 and BID22 propose to instead maximize the evidence lower-bound (ELBO), log p(x) ≥ E z∼qω(z|x) log p θ (x|z) p(z) q ω (z|x)=: L E .Here, q ω (z|x) is an auxiliary inference network, parametrized by ω.

Simultaneous optimization of (2) over both θ and ω performs approximate maximum likelihood estimation in the model p(x) of FORMULA0 and forms the standard VAE estimation method.

1 The implementation is available at https://github.com/Microsoft/ jackknife-variational-inferenceIn practice L E is estimated using Monte Carlo: we draw K samples z i ∼ q ω (z|x), then use the unbiased estimatorL E of L E ,L DISPLAYFORM1 The VAE approach is empirically very successful but are there fundamental limitations?

One limitation is the quality of the model p θ (x|z): this model needs to be expressive enough to model the true distribution over x. Another limitation is that L E is only a lower-bound to the true likelihood.

Is this bound tight?

It can be shown, BID8 , that when q(z|x) = p(z|x) we have L E = log p(x), hence (2) becomes exact.

Therefore, we should attempt to choose an expressive class of distributions q(z|x) and indeed recent work has extensively investigated richer variational families.

We discuss these methods in Section 7 but now review the importance weighted autoencoder (IWAE) method we build upon.

The importance weighted autoencoder (IWAE) method BID2 seemingly deviates from (2) in that they propose the IWAE objective, defined for an integer K ≥ 1, log p(x) ≥ E z1,...,z K ∼qω(z|x) log 1 K DISPLAYFORM0 We denote withL K the empirical version which takes one sample z 1 , . . .

, z K ∼ q ω (z|x) and evaluates the inner expression in (6).

We can see that L 1 = L E , and indeed BID2 further show that DISPLAYFORM1 and lim K→∞ L K = log p(x).

These results are a strong motivation for the use of L K to estimate θ and the IWAE method can often significantly improve over L E .

The bounds L K seem quite different from L E , but recently BID3 and BID15 showed that an exact correspondence exists: any L K can be converted into the standard form L E by defining a modified distribution q IW (z|x) through an importance sampling construction.

We now analyze the IWAE boundL K in more detail.

Independently of our work BID19 has analysed nested Monte Carlo objectives, including the IWAE bound as special case.

Their analysis includes results equivalent to our Proposition 1 and 2.

We now analyze the statistical properties of the IWAE estimator of the log-marginal likelihood.

Basic consistency results have been shown in BID2 ; here we provide more precise results and add novel asymptotic results regarding the bias and variance of the IWAE method.

Our results are given as expansions in the order K of the IWAE estimator but do involve moments µ i which are unknown to us.

The jackknife method in the following sections will effectively circumvent the problem of not knowing these moments.

Proposition 1 (Expectation ofL K ).

Let P be a distribution supported on the positive real line and let P have finite moments of all order.

Let K ≥ 1 be an integer.

Let w 1 , w 2 , . . .

, w K ∼ P independently.

Then we have asymptotically, for K → ∞, DISPLAYFORM0 where DISPLAYFORM1 is the i'th central moment of P and µ := E P [w] is the mean.

Proof.

See Appendix A, page 12.The above result directly gives the bias of the IWAE method as follows.

Corollary 1 (Bias ofL K ).

If we seeL K as an estimator of log p(x), then for DISPLAYFORM2 Proof.

The bias (10) follows directly by subtracting the true value log p(x) = log E[w] from the right hand side of (8).The above result shows that the bias is reduced at a rate of O(1/K).

This is not surprising because the IWAE estimator is a smooth function applied to a sample mean.

The coefficient of the leading O(1/K) bias term uses the ratio µ 2 /µ 2 , the variance divided by the squared mean of the P distribution.

The quantity µ 2 /µ 2 is known as the coefficient of variation and is a common measure of dispersion of a distribution.

Hence, for large K the bias ofL K is small when the coefficient of variation is small; this makes sense because in case the dispersion is small the logarithm function behaves like a linear function and few bias results.

The second-order and higher-order terms takes into account higher order properties of P .The bias is the key quantity we aim to reduce, but every estimator is also measured on its variance.

We now quantify the variance of the IWAE estimator.

Proposition 2 (Variance ofL K ).

For K → ∞, the variance ofL K is given as follows.

DISPLAYFORM3 Proof.

See Appendix A, page 13.Both the bias B[L K ] and the variance V[L K ] vanish for K → ∞ at a rate of O(1/K) with similar coefficients.

This leads to the following result which was already proven in BID2 .

DISPLAYFORM4 Proof.

See Appendix A, page 13.How good are the asymptotic results?

This is hard to say in general because it depends on the particular distribution P (w) of the weights.

In FIG5 we show both a simple and challenging case to demonstrate the accuracy of the asymptotics.

The above results are reassuring evidence for the IWAE method, however, they cannot be directly applied in practice because we do not know the moments µ i .

One approach is to estimate the moments from data, and this is in fact what the delta method variational inference (DVI) method does, Teh et al. (2007) , (see Appendix B, page 14); however, estimating moments accurately is difficult.

We avoid the difficulty of estimating moments by use of the jackknife, a classic debiasing method.

We now review this method.

Bias of log p(x) evidence approximations, P =Gamma(1;1) Variance of log p(x) evidence approximations, P =Gamma(1;1) Variance of log p(x) evidence approximations, P =Gamma(0:1;1) DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8

We now provide a brief review of the jackknife and generalized jackknife methodology.

Our presentation deviates from standard textbook introductions, BID13 , in that we also review higher-order variants.

The jackknife methodology is a classic resampling technique originating with BID17 BID18 The basic intuition is as follows: in many cases it is possible to write the expectation of a consistent estimatorT n evaluated on n samples as an asymptotic expansion in the sample size n, that is, for large n → ∞ we have DISPLAYFORM0 In particular, this is possible in case the estimator is consistent and a smooth function of linear statistics.

If an expansion (13) is possible, then we can take a linear combination of two estimatorŝ T n andT n−1 to cancel the first order term, DISPLAYFORM1 Therefore, the jackknife bias-corrected estimatorT J := nT n − (n − 1)T n−1 achieves a reduced bias of O(n −2 ).

ForT n−1 any estimator which preserves the expectation (13) can be used.

In practice we use the original sample of size n to create n subsets of size n − 1 by removing each individual sample once.

Then, the empirical average of n estimatesT \i n−1 , i = 1, . . . , n is used in place ofT n−1 .

In Sharot (1976) this construction was proved optimal in terms of maximally reducing the variance of T J for any given sample size n.

In principle, the above bias reduction (16) can be repeated to further reduce the bias to O(n −3 ) and beyond.

The possibility of this was already hinted at in BID18 by means of an example.

A fully general and satisfactory solution to higher-order bias removal was only achieved by the generalized jackknife of BID26 , considering estimatorsT G of order m, each having the form,T DISPLAYFORM0 The form of the coefficients c(n, m, j) in FORMULA0 are defined by the ratio of determinants of certain Vandermonde matrices, see BID26 .

In a little known result, an analytic solution for c(n, m, j) is given by Sharot (1976) .

We call this form the Sharot coefficients, (Sharot, 1976, Equation (2.5) with r = 1), defined for m < n and 0 ≤ j ≤ m, DISPLAYFORM1 The generalized jackknife estimatorT (1971) .

For example, the classic jackknife is recovered because c(n, 1, 0) = n and c(n, 1, 1) = −(n − 1).

As an example of the second-order generalized jackknife we have DISPLAYFORM2 DISPLAYFORM3 The variance of generalized jackknife estimators is more difficult to characterize and may in general decrease or increase compared toT n .

Typically we have DISPLAYFORM4 G ] with asymptotic rates being the same.

The generalized jackknife is not the only method for debiasing estimators systematically.

One classic method is the delta method for bias correction Small (2010).

Two general methods for debiasing are the iterated bootstrap for bias correction (Hall, 2016, page 29) and the debiasing lemma McLeish (2010); Strathmann et al. (2015) ; BID23 .

Remarkably, the debiasing lemma exactly debiases a large class of estimators.

The delta method bias correction has been applied to variational inference by Teh et al. FORMULA1 ; we provide novel theoretical results for the method in Appendix B, page 14.

We now propose to apply the generalized jackknife for bias correction to variational inference by debiasing the IWAE estimator.

The resulting estimator of the log-marginal likelihood will have significantly reduced bias, however, in contrast to the ELBO and IWAE, it is no longer a lower bound on the true log-marginal likelihood.

Moreover, it can have increased variance compared to both IWAE and ELBO estimators.

We will empirically demonstrate that the variance is comparable to the IWAE estimate and that the bias reduction is very effective in improving our estimates.

Definition 1 (Jackknife Variational Inference (JVI)).

Let K ≥ 1 and m < K. The jackknife variational inference estimator of the evidence of order m with K samples iŝ DISPLAYFORM0 whereL K−j is the empirical average of one or more IWAE estimates obtained from a subsample of size K − j, and c(K, m, j) are the Sharot coefficients defined in (18).

In this paper we use all DISPLAYFORM1 where DISPLAYFORM2 is the i'th subset of size K − j among all K K−j subsets from the original samples DISPLAYFORM3 From the above definition we can see that JVI strictly generalizes the IWAE bound and therefore also includes the standard ELBO objective: we have the IWAE case forL J,0 K =L K , and the ELBO case forL DISPLAYFORM4 The proposed family of JVI estimators has less bias than the IWAE estimator.

The following result is a consequence of the existing theory on the generalized jackknife bias correction.

J,m K ).

For any K ≥ 1 and m < K we have that the bias of the JVI estimate satisfies DISPLAYFORM0 Proof.

The JVI estimatorL DISPLAYFORM1 K is the application of the higher-order jackknife to the IWAE estimator which has an asymptotic expansion of the bias (10) in terms of orders of 1/K. The stated result is then a special case of (Schucany et al., 1971, Theorem 4

We show an illustration of higher-order bias removal in Appendix C, page 15.

It is more difficult to characterize the variance ofL DISPLAYFORM0 we have been unable to derive a formal result to this end.

Note that the variance is over the sampling distribution of q(z|x), so we can always reduce the variance by averaging multiple estimatesL J,m K , whereas we cannot reduce bias this way.

Therefore, reducing bias while increasing variance is a sensible tradeoff in our application.

We now discuss how to efficiently compute (20).

For typical applications, for example in variational autoencoders, we will use small values of K, say K < 100.

However, even with K = 50 and m = 2 there are already 1276 IWAE estimates to compute in (20) (21) .

Therefore efficient computation is important to consider.

One property that helps us is that all these IWAE estimates are related because they are based on subsets of the same weights.

The other property that is helpful is that computation of the K weights is typically orders of magnitude more expensive than elementary summation operations required for computation of (21).We now give a general algorithm for computing the JVI estimatorL J,m K , then give details for efficient implementation on modern GPUs and state complexity results.

Algorithm 1 computes log-weights and implements equations (20-21) in a numerically robust manner.

Proof.

See Appendix C, page 15.

J,m K , the jackknife variational inference estimator 1: function COMPUTEJVI(m, K, p, q, x) 2: DISPLAYFORM0 end for 6: L ← 0 7: DISPLAYFORM1 for S ∈ EnumerateSubsets({1, . . . , K}, K − j) do list all subsets of size K − j 10:L ←L + log s∈S exp vs − log(K − j) IWAE estimate for subset S 11:end for 12: DISPLAYFORM2 Using equation FORMULA0 13: end for 14:return L JVI estimateL Runtime for the full MNIST test set (GPU)L DISPLAYFORM3 Figure 2: Runtime evaluation of theL DISPLAYFORM4 The above algorithm is suitable for CPU implementation; to utilize modern GPU hardware efficiently we can instead represent the second part of the algorithm using matrix operations.

We provide further details in Appendix C, page 16.

Figure 2 demonstrates experimental runtime evaluation on the MNIST test set for different JVI estimators.

We show all JVI estimators with less than 5,000 total summation terms.

The result demonstrates that runtime is largely independent of the order of the JVI correction and only depends linearly on K.

Variations of the JVI estimator with improved runtime exist.

Such reduction in runtime are possible if we consider evaluating only a fraction of all possible subsets in (21).

When tractable, our choice of evaluating all subsets is generally preferable in terms of variance of the resulting estimator.

However, to show that we can even reduce bias to order DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 The sum (25-27) can be computed in time O(K) by keeping a running partial sum k i=1 exp(v i ) for k ≤ K and by incrementally updating this sum 4 , meaning that (24) can be computed in O(K)4 To do this in a numerically stable manner, we need to use streaming log-sumexp computations, see for example http://www.nowozin.net/sebastian/blog/ streaming-log-sum-exp-computation.html overall.

As a generalized jackknife estimateL X K has bias O(K −K ).

We do not recommend its use in practice because its variance is large, however, developing estimators between the two extremes of taking one set and taking all sets of subsets of a certain size seems a good way to achieve high-order bias reduction while controlling variance.

We now empirically validate our key claims regarding the JVI method: 1.

JVI produces better estimates of the marginal likelihood by reducing bias, even for small K; and 2.

Higher-order bias reduction is more effective than lower-order bias reduction;To this end we will use variational autoencoders trained on MNIST.

Our setup is purposely identical to the setup of Tomczak & Welling FORMULA0 , where we use the dynamically binarized MNIST data set of BID24 .

Our numbers are therefore directly comparable to the numbers reported in the above works.

Our implementation is available at https://github.

com/Microsoft/jackknife-variational-inference.

We first evaluate the accuracy of evidence estimates given a fixed model.

This setting is useful for assessing model performance and for model comparison.

We train a regular VAE on the dynamically binarized MNIST dataset using either the ELBO, IWAE, or JVI-1 objective functions.

We use the same two-layer neural network architecture with 300 hidden units per layer as in (Tomczak & Welling, 2016) .

We train on the first 50,000 training images, using 10,000 images for validation.

We train with SGD for 5,000 epochs and take as the final model the model with the maximum validation objective, evaluated after every training epoch.

Hyperparameters are the batch size in {1024, 4096} and the SGD step size in {0.1, 0.05, 0.01, 0.005, 0.001}. The final model achieving the best validation score is evaluated once on the MNIST test set.

All our models are implemented using Chainer (Tokui et al., 2015) and run on a NVidia Titan X.For three separate models, trained using the ordinary ELBO, IWAE, and JVI-1 objectives, we then estimate the marginal log-likelihood (evidence) on the MNIST test set.

For evaluation we use JVI estimators up to order five in order to demonstrate higher-order bias reduction.

Among all possible JVI estimators up to order five we evaluate only those JVI estimators whose total sum of IWAE estimates has less than 5,000 terms.

For example, we do not evaluateL FIG12 shows the evidence estimates for three models.

We make the following observations, applying to all plots: 1.

Noting the logarithmic x-axis we can see that higher-order JVI estimates are more than one order of magnitude more accurate than IWAE estimates.

2.

The quality of the evidence estimates empirically improves monotonically with the order of the JVI estimator; 3.

In absolute terms the improvements in evidence estimates is larges for small values of K, which is what is typically used in practice; 4.

The higher-order JVI estimators remove low-order bias but significant higher-order bias remains even for K = 64, showing that on real VAE log-weights the contribution of higher-order bias to the evidence error is large; 5.

The standard error of each test set marginal likelihood (shown as error bars, best visible in a zoomed version of the plot) is comparable across all JVI estimates; this empirically shows that higher-order bias reduction does not lead to high variance.

We now report preliminary results on learning models using the JVI objectives.

The setting is the same as in Section 6.1 and we report the average performance of five independent runs.

Table 1 reports the results.

We make the following observations: 1.

When training on the IWAE and JVI-1 objectives, the respective score by the ELBO objective is impoverished and this effect makes Evidence Estimates on JVI-1-trained VAE (MNIST test set) Table 1 : Evaluating models trained using ELBO, IWAE, and JVI-1 learning objectives.

DISPLAYFORM0 sense in light of the work of BID3 .

Interestingly the effect is stronger for JVI-1.

2.

The model trained using the JVI-1 objective falls slightly behind the IWAE model, which is surprising because the evidence is clearly better approximated as demonstrated in Section 6.1.

We are not sure what causes this issue, but have two hypotheses: First, in line with recent findings BID20 ) a tighter log-evidence approximation could lead to poor encoder models.

In such case it is worth exploring two separate learning objectives for the encoder and decoder; for example, using an ELBO for training the encoder, and an IWAE or JVI-1 objective for training the decoder.

Second, because JVI estimators are no longer bounds it could be the case that during optimization of the learning objective a decoder is systematically learned in order to amplify positive bias in the log-evidence.

The IWAE bound and other Monte Carlo objectives have been analyzed by independently by BID19 .

Their analysis is more general than our IWAE analysis, but does not propose a method to reduce bias.

Delta-method variational inference (DVI) proposed by Teh et al. (2007) is the closest method we are aware of and we discuss it in detail as well as provide novel results in Appendix B, page 14.

Another exciting recent work is perturbative variational inference BID1 which considers different objective functions for variational inference; we are not sure whether there exists a deeper relationship to debiasing schemes.

There also exists a large body of work that uses the ELBO objective but considers ways to enlarge the variational family.

This is useful because the larger the variational family, the smaller the bias.

Non-linear but invertible transformations of reference densities have been used initially for density estimation in NICE BID4 and for variational inference in Hamiltonian variational inference BID25 .

Another way to improve the flexibility of the variational family has been to use implicit models (Mohamed & Lakshminarayanan, 2016) for variational inference; this line of work includes adversarial variational Bayes BID12 , wild variational inference BID10 , deep implicit models (Tran et al., 2017) , implicit variational models BID6 , and adversarial message passing approximations BID7 .

In summary we proposed to leverage classic higher-order bias removal schemes for evidence estimation.

Our approach is simple to implement, computationally efficient, and clearly improves over existing evidence approximations based on variational inference.

More generally our jackknife variational inference debiasing formula can also be used to debias log-evidence estimates coming from annealed importance sampling.

However, one surprising finding from our work is that using our debiased estimates for training VAE models did not improve over the IWAE training objective and this is surprising because apriori a better evidence estimate should allow for improved model learning.

One possible extension to our work is to study the use of other resampling methods for bias reduction; promising candidates are the iterated bootstrap, the Bayesian bootstrap, and the debiasing lemma.

These methods could offer further improvements on bias reduction or reduced variance, however, the key challenge is to overcome computational requirements of these methods or, alternatively, to derive key quantities analytically.

6 Application of the debiasing lemma in particular requires the careful construction of a truncation distribution and often produces estimators of high variance.

While variance reduction plays a key role in certain areas of machine learning, our hope is that our work shows that bias reduction techniques are also widely applicable.

DISPLAYFORM0 Note that only Y K is random in (28), all other quantities are constant.

Therefore, by taking the expectation on the left and right side of FORMULA1 we obtain DISPLAYFORM1 The right hand side of FORMULA1 is expressed in terms of the central moments for i ≥ 2, DISPLAYFORM2 of Y K , whereas we are interested in an expression using the central moments i ≥ 2, DISPLAYFORM3 we denote the shared first non-central moment.

Because Y K is a sample mean we can use existing results that relate γ i to µ i .

In particular (Angelova, 2012, Theorem 1) gives the relations DISPLAYFORM4 DISPLAYFORM5 Expanding FORMULA1 to order five and using the relations FORMULA2 to FORMULA2 gives DISPLAYFORM6 Regrouping the terms by order of K produces the result (8).

Proof. (Of Proposition 2, page 3) We use the definition of the variance and the series expansion of the logarithm function, obtaining DISPLAYFORM0 By expanding (37) to third order and expanding all products we obtain a moment expansion of Y K as follows.

DISPLAYFORM1 DISPLAYFORM2 By substituting the sample moments γ i of Y K with the central moments µ i of the original distribution P and simplifying we obtain DISPLAYFORM3 CONSISTENCY OFL K Proof.

We have DISPLAYFORM4 DISPLAYFORM5 The second term in (43) does not involve a random variable therefore is either zero or one.

For large enough K it will always be zero due to (10).For the first term in (43) we apply Chebyshev's inequality.

DISPLAYFORM6 and have DISPLAYFORM7 Thus, for K → ∞ and any > 0 we have that (43) has a limit of zero.

This establishes convergence in probability and hence consistency.

Definition 2 (Delta method Variational Inference (DVI), (Teh et al., 2007) ).

DISPLAYFORM0 where DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 so thatŵ 2 corresponds to the sample variance andŵ corresponds to the sample mean.

The practical Monte Carlo estimator of FORMULA3 is defined as follows.

DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 Proof.

Consider the function f (x, y) = x y 2 and its second order Taylor expansion around (x, y) = (µ 2 , µ), DISPLAYFORM7 Taking expectations on both sides cancels all linear terms and yields DISPLAYFORM8 By classic results we have that the expected variance of the sample mean around the true mean is related to the variance by Zhang (2007) showed a beautiful result about the covariance of sample mean and sample variance for arbitrary random variables, namely that DISPLAYFORM9 DISPLAYFORM10 Using both results in (57) produces DISPLAYFORM11 We can now decompose the expectation ofL Bias of log p(x) evidence approximations, P =Gamma(1;1) Variance of log p(x) evidence approximations, P =Gamma(1;1) Notably, in (62) the 1/K term is cancelled exactly by the delta method correction, even though we used an empirical ratio estimatorμ 2 /μ 2 .

Subtracting the true mean log p(x) = log E[w] from (62) yields the bias (54) and completes the proof.

DISPLAYFORM12 DISPLAYFORM13 DISPLAYFORM14 DISPLAYFORM15

We perform the experiment shown in FIG5 including the DVI estimator.

The result is shown in FIG19 and confirms that DVI reduces bias but that for the challenging case JVI is superior in terms of bias reduction.

For each of the S(K, m) sets we have to perform at most K operations to compute the log-sum-exp operation, which yields the stated complexity bound.

We illustrate the behaviour of the higher-order JVI estimators on the same P = Gamma(0.1, 1) example we used previously.

FIG21 demonstrates the increasing order of bias removal, O(K −(m+1) ) for theL J,m K estimators.

To this end let K ≥ 1 and m < K be fixed and assume the log-weights v i are concatenated in one column vector of K elements.

We then construct a matrix B of size (|S|, K), where S is the set of all subsets that will be considered, DISPLAYFORM0 EnumerateSubsets({1, . . .

, K}, K − j).There are |S| rows in B and each row in B corresponds to a subset S ∈ S of samples so that we can use S to index the rows in B. We set DISPLAYFORM1 where I pred is one if the predicate is true and zero otherwise.

We furthermore construct a vector A with |S| elements.

We set DISPLAYFORM2 Using these definitions we can express the estimator as A log(B exp(v)), with the log and exp operations being elementwise.

However, this is not numerically robust.

Instead we can compute the estimator in the log domain as logsumexp 2 (I S×1 v + log B) A, where logsumexp 2 denotes a log-sum-exp operation along the second axis.

This can be easily implemented in modern neural network frameworks and we plan to make our implementation available.

<|TLDR|>

@highlight

Variational inference is biased, let's debias it.

@highlight

Introduces jackknife variational inference, a method for debiasing Monte Carlo objectives such as the importance weighted auto-encoder.

@highlight

The authors analyze the bias and variance of the IWAE bound and derive a jacknife approach to estimate moments as a way to debias IWAE for finite importance weighted samples.