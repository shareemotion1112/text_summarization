We propose Regularized Learning under Label shifts (RLLS), a principled and a practical domain-adaptation algorithm to correct for shifts in the label distribution between a source and a target domain.

We first estimate importance weights using labeled source data and unlabeled target data, and then train a classifier on the weighted source samples.

We derive a generalization bound for the classifier on the target domain which is independent of the (ambient) data dimensions, and instead only depends on the complexity of the function class.

To the best of our knowledge, this is the first generalization bound for the label-shift problem where the labels in the target domain are not available.

Based on this bound, we propose a regularized estimator for the small-sample regime which accounts for the uncertainty in the estimated weights.

Experiments on the CIFAR-10 and MNIST datasets show that RLLS improves classification accuracy, especially in the low sample and large-shift regimes, compared to previous methods.

When machine learning models are employed "in the wild", the distribution of the data of interest(target distribution) can be significantly shifted compared to the distribution of the data on which the model was trained (source distribution).

In many cases, the publicly available large-scale datasets with which the models are trained do not represent and reflect the statistics of a particular dataset of interest.

This is for example relevant in managed services on cloud providers used by clients in different domains and regions, or medical diagnostic tools trained on data collected in a small number of hospitals and deployed on previously unobserved populations and time frames.

Label Shift p(x) = q(x) p(y|x) = q(y|x) p(y) = q(y) p(x|y) = q (x|y) There are various ways to approach distribution shifts between a source data distribution P and a target data distribution Q. If we denote input variables as x and output variables as y, we consider the two following settings: (i) Covariate shift, which assumes that the conditional output distribution is invariant: p(y|x) = q(y|x) between source and target distributions, but the source distribution p(x) changes. (ii) Label shift, where the conditional input distribution is invariant: p(x|y) = q(x|y) and p(y) changes from source to target.

In the following, we assume that both input and output variables are observed in the source distribution whereas only input variables are available from the target distribution.

While covariate shift has been the focus of the literature on distribution shifts to date, label-shift scenarios appear in a variety of practical machine learning problems and warrant a separate discussion as well.

In one setting, suppliers of machine-learning models such as cloud providers have large resources of diverse data sets (source set) to train the models, while during deployment, they have no control over the proportion of label categories.

In another setting of e.g. medical diagnostics, the disease distribution changes over locations and time.

Consider the task of diagnosing a disease in a country with bad infrastructure and little data, based on reported symptoms.

Can we use data from a different location with data abundance to diagnose the disease in the new target location in an efficient way?

How many labeled source and unlabeled target data samples do we need to obtain good performance on the target data?Apart from being relevant in practice, label shift is a computationally more tractable scenario than covariate shift which can be mitigated.

The reason is that the outputs y typically have a much lower dimension than the inputs x. Labels are usually either categorical variables with a finite number of categories or have simple well-defined structures.

Despite being an intuitively natural scenario in many real-world application, even this simplified model has only been scarcely studied in the literature.

Zhang et al. (2013) proposed a kernel mean matching method for label shift which is not computationally feasible for large-scale data.

The approach in Lipton et al. (2018) is based on importance weights that are estimated using the confusion matrix (also used in the procedures of Saerens et al. (2002); McLachlan (2004) ) and demonstrate promising performance on large-scale data.

Using a black-box classifier which can be biased, uncalibrated and inaccurate, they first estimate importance weights q(y)/p(y) for the source samples and train a classifier on the weighted data.

In the following we refer to the procedure as black box shift learning (BBSL) which the authors proved to be effective for large enough sample sizes.

However, there are three relevant questions which remain unanswered by their work: How to estimate the importance weights in low sample setting, What are the generalization guarantees for the final predictor which uses the weighted samples?

How do we deal with the uncertainty of the weight estimation when only few samples are available?

This paper aims to fill the gap in terms of both theoretical understanding and practical methods for the label shift setting and thereby move a step closer towards having a more complete understanding on the general topic of supervised learning for distributionally shifted data.

In particular, our goal is to find an efficient method which is applicable to large-scale data and to establish generalization guarantees.

Our contribution in this work is trifold.

Firstly, we propose an efficient weight estimator for which we can obtain good statistical guarantees without a requirement on the problem-dependent minimum sample complexity as necessary for BBSL.

In the BBSL case, the estimation error can become arbitrarily large for small sample sizes.

Secondly, we propose a novel regularization method to compensate for the high estimation error of the importance weights in low target sample settings.

It explicitly controls the influence of our weight estimates when the target sample size is low (in the following referred to as the low sample regime).

Finally, we derive a dimension-independent generalization bound for the final Regularized Learning under Label Shift (RLLS) classifier based on our weight estimator.

In particular, our method improves the weight estimation error and excess risk of the classifier on reweighted samples by a factor of k log(k), where k is the number of classes, i.e. the cardinality of Y.In order to demonstrate the benefit of the proposed method for practical situations, we empirically study the performance of RLLS and show weight estimation as well as prediction accuracy comparison for a variety of shifts, sample sizes and regularization parameters on the CIFAR-10 and MNIST datasets.

For large target sample sizes and large shifts, when applying the regularized weights fully, we achieve an order of magnitude smaller weight estimation error than baseline methods and enjoy at most 20% higher accuracy and F-1 score in corresponding predictive tasks.

For low target sample sizes, applying regularized weights partially also yields an accuracy improvement of at least 10% over fully weighted and unweighted methods.

Formally let us the short hand for the marginal probability mass functions of Y on finite Y with respect to P, Q as p, q : DISPLAYFORM0 representable by vectors in R k + which sum to one.

In the label shift setting, we define the importance weight vector w ∈ R k between these two domains as w(i) = q(i) p(i) .

We quantify the shift using the exponent of the infinite and second order Renyi divergence as follows DISPLAYFORM1 , DISPLAYFORM2 Published as a conference paper at ICLR 2019 Given a hypothesis class H and a loss function : Y × Y → [0, 1], our aim is to find the hypothesis h ∈ H which minimizes DISPLAYFORM3 In the usual finite sample setting however, L unknown and we observe samples {(x j , y j )} n j=1 from P instead.

If we are given the vector of importance weights w we could then minimize the empirical loss with importance weighted samples defined as DISPLAYFORM4 where n is the number of available observations drawn from P used to learn the classifier h. As w is unknown in practice, we have to find the minimizer of the empirical loss with estimated importance weights DISPLAYFORM5 where w are estimates of w. Given a set D p of n p samples from the source distribution P, we first divide it into two sets where we use (1 − β)n p samples in set D weight p to compute the estimate w and the remaining n = βn p in the set D class p to find the classifier which minimizes the loss (1), i.e. h w = arg min h∈H L n (h; w).

In the following, we describe how to estimate the weights w and provide guarantees for the resulting estimator h w .

The following simple correlation between the label distributions p, q was noted in Lipton et al. (2018) : for a fixed hypothesis h, if for all y ∈ Y it holds that q(y) ≥ 0 =⇒ p(y) ≥ 0, we have DISPLAYFORM0 for all i, j ∈ Y. This can equivalently be written in matrix vector notation as DISPLAYFORM1 where C h is the confusion matrix with [C h ] i,j = P(h(X) = i, Y = j) and q h is the vector which represents the probability mass function of h(X) under distribution Q. The requirement q(y) ≥ 0 =⇒ p(y) ≥ 0 is a reasonable condition since without any prior knowledge, there is no way to properly reason about a class in the target domain that is not represented in the source domain.

In reality, both q h and C h can only be estimated by the corresponding finite sample averages q h , C h .

Lipton et al. (2018) simply compute the inverse of the estimated confusion matrix C h in order to estimate the importance weight, i.e. DISPLAYFORM2 h q h is a statistically efficient estimator, w with estimated C −1 h can be arbitrarily bad since C −1 h can be arbitrary close to a singular matrix especially for small sample sizes and small minimum singular value of the confusion matrix.

Intuitively, when there are very few samples, the weight estimation will have high variance in which case it might be better to avoid importance weighting altogether.

Furthermore, even when the sample complexity in Lipton et al. (2018) , unknown in practice, is met, the resulting error of this estimator is linear in k which is problematic for large k.

We therefore aim to address these shortcomings by proposing the following two-step procedure to compute importance weights.

In the case of no shift we have w = 1 so that we define the amount of weight shift as θ = w − 1.

Given a "decent" black box estimator which we denote by h 0 , we make the final classifier less sensitive to the estimation performance of C (i.e. regularize the weight estimate) by 1.

calculating the measurement error adjusted θ (described in Section 2.1 for h 0 ) and 2.

computing the regularized weight w = 1+λ θ where λ depends on the sample size (1−β)n p .By "decent" we refer to a classifier h 0 which yields a full rank confusion matrix C h0 .

A trivial example for a non-"decent" classifier h 0 is one that always outputs a fixed class.

As it does not capture any characteristics of the data, there is no hope to gain any statistical information without any prior information.

Both the confusion matrix C h0 and the label distribution q h0 on the target for the black box hypothesis h 0 are unknown and we are instead only given access to finite sample estimates C h0 , q h0 .

In what follows all empirical and population confusion matrices, as well as label distributions, are defined with respect to the hypothesis h = h 0 .

For notation simplicity, we thus drop the subscript h 0 in what follows.

The reparameterized linear model (2) with respect to θ then reads b := q − C1 = Cθ with corresponding finite sample quantity b = q − C1.

When C is near singular, the estimation of θ becomes unstable.

Furthermore, large values in the true shift θ result in large variances.

We address this problem by adding a regularizing 2 penalty term to the usual loss and thus push the amount of shift towards 0, a method that has been proposed in (Pires & Szepesvári, 2012) .

In particular, we compute θ = arg min DISPLAYFORM0 Here, ∆ C is a parameter which will eventually be high probability upper bounds for C − C 2 .

Let ∆ b also denote the high probability upper bounds for b − b 2 .Lemma 1 For θ as defined in equation FORMULA9 , we have with probability at least 1 − δ that DISPLAYFORM1 where DISPLAYFORM2 The proof of this lemma can be found in Appendix B.1.

A couple of remarks are in order at this point.

First of all, notice that the weight estimation procedure (3) does not require a minimum sample complexity which is in the order of σ −2 min to obtain the guarantees for BBSL.

This is due to the fact that errors in the covariates are accounted for.

In order to directly see the improvements in the upper bound of Lemma 1 compared to Theorem 3 in Lipton et al. (2018) , first observe that in order to obtain their upper bound with a probability of at least 1 − δ, it is necessary that 3kn DISPLAYFORM3 .

Thus Lemma 1 improves upon the previous upper bound by a factor of k.

Furthermore, as in Lipton et al. (2018) , this result holds for any black box estimator h 0 which enters the bound via σ min (C h0 ).

We can directly see how a good choice of h 0 helps to decrease the upper bound in Lemma 1.

In particular, if h 0 is an ideal estimator, and the source set is balanced, C is the unit matrix with σ min = 1/k.

In contrast, when the model h 0 is uncertain, the singular value σ min is close to zero.

Moreover, for least square problems with Gaussian measurement errors in both input and target variables, it is standard to use regularized total least squares approaches which requires a singular value decomposition.

Finally, our choice for the alternative estimator in Eq. 3 with norm instead of norm squared regularization is motivated by the cases with large shifts θ, where using the squared norm may shrink the estimate θ too much and away from the true θ.

DISPLAYFORM4 where DISPLAYFORM5 6: Deploy h w if the risk is acceptable

When a few samples from the target set are available or the label shift is mild, the estimated weights might be too uncertain to be applied.

We therefore propose a regularized estimator defined as follows DISPLAYFORM0 Note that w implicitly depends on λ, and β.

By rewriting w = (1 − λ)1 + λ(1 + θ), we see that intuitively λ closer to 1 the more reason there is to believe that 1 + θ is in fact the true weight.

Define the set G( , H) = {g h (x, y) = w(y) (h(x), y) : h ∈ H} and its Rademacher complexity measure DISPLAYFORM1 with ξ i , ∀i as the Rademacher random variables (see e.g. BID2 ).

We can now state a generalization bound for the classifier h w in a general hypothesis class H, which is trained on source data with the estimated weights defined in equation (4).Theorem 1 (Generalization bound for h w ) Given n p samples from the source data set and n q samples from the target set, a hypothesis class H and loss function , the following generalization bound holds with probability at least 1 − 2δ DISPLAYFORM2 where DISPLAYFORM3 The proof can be found in Appendix B.4.

Additionally, we derive the analysis also for finite hypothesis classes in Appendix B.6 to provide more insight into the proof of general hypothesis classes.

The size of R n (G) is determined by the structure of the function class H and the loss .

For example for the 0/1 loss, the VC dimension of H can be deployed to upper bound the Rademacher complexity.

The bound (5) in Theorem 1 holds for all choices of λ.

In order to exploit the possibility of choosing λ and β to have an improved accuracy depending on the sample sizes, we first let the user define a set of shifts θ against which we want to be robust against, i.e. all shifts with θ 2 ≤ θ max .

For these shifts, we obtain the following upper bound DISPLAYFORM4 The bound in equation FORMULA19 suggests using Algorithm 1 as our ultimate label shift correction procedure.

where for step 2 of the algorithm, we choose λ = 1 whenever n q ≥ 1 θ 2 max (σmin− 1 √ np ) 2 (hereby neglecting the log factors and thus dependencies on k) and 0 else.

When using this rule, we DISPLAYFORM5 } which is smaller than the unregularized bound for small n q , n p .

Notice that in practice, we do not know σ min in advance so that in Algorithm 1 we need to use an estimate of σ min , which could e.g. be the minimum eigenvalue of the empirical confusion matrix C with an additional computational complexity of at most O(k 3 ).

Figure 1: Given a σ min and θ max , λ switches from 0 to 1 at a particular n q .

n p and k are fixed.

Figure 1 shows how the oracle thresholds vary with n q and σ min when n p is kept fix.

When the parameters are above the curves for fixed n p , λ should be chosen as 1 otherwise the samples should be unweighted, i.e. λ = 0.

This figure illustrates that when the confusion matrix has small singular values, the estimated weights should only be trusted for rather high n q and high believed shifts θ max .

Although the overall statistical rate of the excess risk of the classifier does not change as a function of the sample sizes, θ max could be significantly smaller than θ when σ min is very small and thus the accuracy in this regime could improve.

Indeed we observe this to be the case empirically in Section 3.3.In the case of slight deviation from the label shift setting, we expect the Alg.

1 to perform reasonably.

DISPLAYFORM6 as the deviation form label shift constraint, i.e., zero under label shift assumption, we have Theorem 2 (Drift in Label shift assumption) In the presence of d e (q||p) deviation from label shift assumption, the true importance weights ω(x, y) := q(x,y) p(x,y) , the RLLS generalizes as; DISPLAYFORM7 with high probability.

Proof in Appendix B.7.

In this section we illustrate the theoretical analysis by running RLLS on a variety of artificially generated shifts on the MNIST (LeCun & Cortes, 2010) and CIFAR10 (Krizhevsky & Hinton, 2009) datasets.

We first randomly separate the entire dataset into two sets (source and target pool) of the same size.

Then we sample, unless specified otherwise, the same number of data points from each pool to form the source and target set respectively.

We chose to have equal sample sizes to allow for fair comparisons across shifts.

There are various kinds of shifts which we consider in our experiments.

In general we assume one of the source or target datasets to have uniform distribution over the labels.

Within the non-uniform set, we consider three types of sampling strategies in the main text: the Tweak-One shift refers to the case where we set a class to have probability p > 0.1, while the distribution over the rest of the classes is uniform.

The Minority-Class Shift is a more general version of Tweak-One shift, where a fixed number of classes m to have probability p < 0.1, while the distribution over the rest of the classes is uniform.

For the Dirichlet shift, we draw a probability vector p from the Dirichlet distribution with concentration parameter set to α for all classes, before including sample points which correspond to the multinomial label variable according to p. Results for the tweak-one shift strategy as in Lipton et al. (2018) can be found in Section A.0.1.After artificially shifting the label distribution in one of the source and target sets, we then follow algorithm 1, where we choose the black box predictor h 0 to be a two-layer fully connected neural network trained on (shifted) source dataset.

Note that any black box predictor could be employed here, though the higher the accuracy, the more likely weight estimation will be precise.

Therefore, we use different shifted source data to get (corrupted) black box predictor across experiments.

If not noted, h 0 is trained using uniform data.

In order to compute ω = 1+ θ in Eq. (3), we call a built-in solver to directly solve the low dimensional problem min θ Cθ − b 2 + ∆ C θ 2 where we empirically observer that 0.01 times of the true ∆ C yields in a better estimator on various levels of label shift pre-computed beforehand.

It is worth noting that 0.001 makes the theoretical bound in Lemma.

1 O(1/0.01) times bigger.

We thus treat it as a hyperparameter that can be chosen using standard cross validation methods.

Finally, we train a classifier on the source samples weighted by ω, where we use a two-layer fully connected neural network for MNIST and a ResNet-18 (He et al., 2016) for CIFAR10.We sample 20 datasets with the label distributions for each shift parameter.

to evaluate the empirical mean square estimation error (MSE) and variance of the estimated weights E w − w 2 2 and the predictive accuracy on the target set.

We use these measures to compare our procedure with the black box shift learning method (BBSL) in Lipton et al. (2018) .

Notice that although KMM methods (Zhang et al., 2013) would be another standard baseline to compare with, it is not scalable to large sample size regimes for n p , n q above n = 8000 as mentioned by Lipton et al. (2018) .

In this set of experiments on the CIFAR10 dataset, we illustrate our weight estimation and prediction performance for Tweak-One source shifts and compare it with BBSL.

For this set of experiments, we set the number of data points in both source and target set to 10000 and sample from the two pools without replacement.

Figure 2 illustrates the weight estimation alongside final classification performance for Minority-Class source shift of CIFAR10.

We created shifts with ρ > 0.5.

We use a fixed black-box classifier that is trained on biased source data, with tweak-one ρ = 0.5.

Observe that the MSE in weight estimation is relatively large and RLLS outperforms BBSL as the number of minority classes increases.

As the shift increases the performance for all methods deteriorates.

Furthermore, Figure 2 (b) illustrates how the advantage of RLLS over the unweighted classifier increases as the shift increases.

Across all shifts, the RLLS based classifier yields higher accuracy than the one based on BBSL.

Results for MNIST can be found in Section A.1.

Figure 2: (a) Mean squared error in estimated weights and (b) accuracy on CIFAR10 for tweak-one shifted source and uniform target with h 0 trained using tweak-one shifted source data.

In this section, we compare the predictive performances between a classifier trained on unweighted source data and the classifiers trained on weighted loss obtained by the RLLS and BBSL procedure on CIFAR10.

The target set is shifted using the Dirichlet shift with parameters α = [0.01, 0.1, 1, 10].The number of data points in both source and target set is 10000.In the case of target shifts, larger shifts actually make the predictive task easier, such that even a constant majority class vote would give high accuracy.

However it would have zero accuracy on all but one class.

Therefore, in order to allow for a more comprehensive performance between the methods, we also compute the macro-averaged F-1 score by averaging the per-class quantity 2(precision · recall)/(precision + recall) over all classes.

For a class i, precision is the percentage of correct predictions among all samples predicted to have label i, while recall is the proportion of correctly predicted labels over the number of samples with true label i.

This measure gives higher weight to the accuracies of minority classes which have no effect on the total accuracy.

FIG4 depicts the MSE of the weight estimation (a), the corresponding performance comparison on accuracy (b) and F-1 score (c).

Recall that the accuracy performance for low shifts is not comparable with standard CIFAR10 benchmark results because of the overall lower sample size chosen for the comparability between shifts.

We can see that in the large target shift case for α = 0.01, the F-1 score for BBSL and the unweighted classifier is rather low compared to RLLS while the accuracy is high.

As mentioned before, the reason for this observation and why in FIG4 (b) the accuracy is higher when the shift is larger, is that the predictive task actually becomes easier with higher shift.

In the following, we present the average accuracy of RLLS in Figure 4 as a function of the number of target samples n q for different values of λ for small n q .

Here we fix the sample size in the source set to n p = 1000 and investigate a Minority-Class source shift with fixed p = 0.01 and five minority classes.

A motivation to use intermediate λ is discussed in Section 2.2, as λ in equation (4) may be chosen according to θ max , σ min .

In practice, since θ max is just an upper bound on the true amount of shift θ 2 , in some cases λ should in fact ideally be 0 when DISPLAYFORM0 Thus for target sample sizes n q that are a little bit above the threshold (depending on the certainty of the belief how close to θ max the norm of the shift is believed to be), it could be sensible to use an intermediate value λ ∈ (0, 1).

DISPLAYFORM1 Figure 4: Performance on MNIST for Minority-Class shifted source and uniform target with various target sample size and λ using (a) better predictor h 0 trained on tweak-one shifted source with ρ = 0.2, (b) neutral predictor h 0 with ρ = 0.5 and (c) corrupted predictor h 0 with ρ = 0.8.

Figure 4 suggests that unweighted samples (red) yield the best classifier for very few samples n q , while for 10 ≤ n q ≤ 500 an intermediate λ ∈ (0, 1) (purple) has the highest accuracy and for n q > 1000, the weight estimation is certain enough for the fully weighted classifier (yellow) to have the best performance (see also the corresponding data points in Figure 2 ).

The unweighted BBSL classifier is also shown for completeness.

We can conclude that regularizing the influence of the estimated weights allows us to adjust to the uncertainty on importance weights and generalize well for a wide range of target sample sizes.

Furthermore, the different plots in Figure 4 correspond to black-box predictors h 0 for weight estimation which are trained on more or less corrupted data, i.e. have a better or worse conditioned confusion matrix.

The fully weighted methods with λ = 1 achieve the best performance faster with a better trained black-box classifier (a), while it takes longer for it to improve with a corrupted one (c).

Furthermore, this reflects the relation between eigenvalue of confusion matrix σ min and target sample size n q in Theorem 1.

In other words, we need more samples from the target data to compensate a bad predictor in weight estimation.

So the generalization error decreases faster with an increasing number of samples for good predictors.

In summary, our RLLS method outperforms BBSL in all settings for the common image datasets MNIST and CIFAR10 to varying degrees.

In general, significant improvements compared to BBSL can be observed for large shifts and the low sample regime.

A note of caution is in order: comparison between the two methods alone might not always be meaningful.

In particular, there are cases when the estimator trained on unweighted samples outperforms both RLLS and BBSL.

Our extensive experiments for many different shifts, black box classifiers and sample sizes do not allow for a final conclusive statement about how weighting samples using our estimator affects predictive results for real-world data in general, as it usually does not fulfill the label-shift assumptions.

The covariate and label shift assumptions follow naturally when viewing the data generating process as a causal or anti-causal model (Schölkopf et al., 2012) : With label shift, the label Y causes the input X (that is, X is not a causal parent of Y , hence "anti-causal") and the causal mechanism that generates X from Y is independent of the distribution of Y .

A long line of work has addressed the reverse causal setting where X causes Y and the conditional distribution of Y given X is assumed to be constant.

This assumption is sensible when there is reason to believe that there is a true optimal mapping from X to Y which does not change if the distribution of X changes.

Mathematically this scenario corresponds to the covariate shift assumption.

Among the various methods to correct for covariate shift, the majority uses the concept of importance weights q(x)/p(x) (Zadrozny, 2004; BID11 BID10 Shimodaira, 2000) , which are unknown but can be estimated for example via kernel embeddings (Huang et al., 2007; Gretton et al., 2009; BID0 Zhang et al., 2013; Zaremba et al., 2013) or by learning a binary discriminative classifier between source and target (Lopez-Paz & Oquab, 2016; Liu et al., 2017) .

A minimax approach that aims to be robust to the worst-case shared conditional label distribution between source and target has also been investigated (Liu & Ziebart, 2014; BID9 .

Sanderson & Scott (2014); Ramaswamy et al. (2016) formulate the label shift problem as a mixture of the class conditional covariate distributions with unknown mixture weights.

Under the pairwise mutual irreducibility (Scott et al., 2013) assumption on the class conditional covariate distributions, they deploy the Neyman-Pearson criterion BID6 to estimate the class distribution q(y) which also investigated in the maximum mean discrepancy framework (Iyer et al., 2014) .Common issues shared by these methods is that they either result in a massive computational burden for large sample size problems or cannot be deployed for neural networks.

Furthermore, importance weighting methods such as (Shimodaira, 2000) estimate the density (ratio) beforehand, which is a difficult task on its own when the data is high-dimensional.

The resulting generalization bounds based on importance weighting methods require the second order moments of the density ratio (q(x)/p(x)) 2 to be bounded, which means the bounds are extremely loose in most cases BID11 .Despite the wide applicability of label shift, approaches with global guarantees in high dimensional data regimes remain under-explored.

The correction of label shift mainly requires to estimate the importance weights q(y)/p(y) over the labels which typically live in a very low-dimensional space.

Bayesian and probabilistic approaches are studied when a prior over the marginal label distribution is assumed (Storkey, 2009; BID8 .

These methods often need to explicitly compute the posterior distribution of y and suffer from the curse of dimensionality.

Recent advances as in Lipton et al. (2018) have proposed solutions applicable large scale data.

This approach is related to BID7 FORMULA5 provides theoretical analysis and generalization guarantees for distribution shifts when the H-divergence between joint distributions is considered, whereas BID12 proves generalization bounds for learning from multiple sources.

For the covariate shift setting, BID11 provides a generalization bound when q(x)/p(x) is known which however does not apply in practice.

To the best of our knowledge our work is the first to give generalization bounds for the label shift scenario.

In this work, we establish the first generalization guarantee for the label shift setting and propose an importance weighting procedure for which no prior knowledge of q(y)/p(y) is required.

Although RLLS is inspired by BBSL, it leads to a more robust importance weight estimator as well as generalization guarantees in particular for the small sample regime, which BBSL does not allow for.

RLLS is also equipped with a sample-size-dependent regularization technique and further improves the classifier in both regimes.

We consider this work a necessary step in the direction of solving shifts of this type, although the label shift assumption itself might be too simplified in the real world.

In future work, we plan to also study the setting when it is slightly violated.

For instance, x in practice cannot be solely explained by the wanted label y, but may also depend on attributes z which might not be observable.

In the disease prediction task for example, the symptoms might not only depend on the disease but also on the city and living conditions of its population.

In such a case, the label shift assumption only holds in a slightly modified sense, i.e. P(X|Y = y, Z = z) = Q(X|Y = y, Z = z).

If the attributes Z are observed, then our framework can readily be used to perform importance weighting.

Furthermore, it is not clear whether the final predictor is in fact "better" or more robust to shifts just because it achieves a better target accuracy than a vanilla unweighted estimator.

In fact, there is a reason to believe that under certain shift scenarios, the predictor might learn to use spurious correlations to boost accuracy.

Finding a procedure which can both learn a robust model and achieve high accuracies on new target sets remains to be an ongoing challenge.

Moreover, the current choice of regularization depends on the number of samples rather than data-driven regularization which is more desirable.

An important direction towards active learning for the same disease-symptoms scenario is when we also have an expert for diagnosing a limited number of patients in the target location.

Now the question is which patients would be most "useful" to diagnose to obtain a high accuracy on the entire target set?

Furthermore, in the case of high risk, we might be able to choose some of the patients for further medical diagnosis or treatment, up to some varying cost.

We plan to extend the current framework to the active learning setting where we actively query the label of certain x's BID5 as well as the cost-sensitive setting where we also consider the cost of querying labels (Krishnamurthy et al., 2017) .Consider a realizable and over-parameterized setting, where there exists a deterministic mapping from x to y, and also suppose a perfect interpolation of the source data with a minimum proper norm is desired.

In this case, weighting the samples in the empirical loss might not alter the trained classifier BID3 .

Therefore, our results might not directly help the design of better classifiers in this particular regime.

However, for the general overparameterized settings, it remains an open problem of how the importance weighting can improve the generalization.

We leave this study for future work.

This section contains more experiments that provide more insights about in which settings the advantage of using RLLS vs. BBSL are more or less pronounced.

Here we compare weight estimation performance between RLLS and BBSL for different types of shifts including the Tweak-one Shift, for which we randomly choose one class, e.g. i and set p(i) = ρ while all other classes are distributed evenly.

Figure 5 depicts the the weight estimation performance of RLLS compared to BBSL for a variety of values of ρ and α.

Note that larger shifts correspond to smaller α and larger ρ.

In general, one observes that our RLLS estimator has smaller MSE and that as the shift increases, the error of both methods increases.

For tweak-one shift we can additionally see that as the shift increases, RLLS outperforms BBSL more and more as both in terms of bias and variance.(a) (b) Figure 5 : Comparing MSE of estimated weights using BBSL and RLLS on CIFAR10 with (a) tweak-one shift on source and uniform target, and (b) Dirichlet shift on source and uniform target.

h 0 is trained using the same source shifted data respectively.

In order to show weight estimation and classification performance under different level of label shifts, we include several additional sets of experiments here in the appendix.

FIG7 shows the weight estimation error and accuracy comparison under a minority-class shift with p = 0.001.

The training and testing sample size is 10000 examples in this case.

We can see that whenever the weight estimation of RLLS is better, the accuracy is also better, except in the four classes case when both methods are bad in weight estimation.

Figure 7 demonstrates another case in minority-class shift when p = 0.01.

The black-box classifier is the same two-layers neural network trained on a biased source data set with tweak-one ρ = 0.5.

We observe that when the number of minority class is small like 1 or 2, the weight estimation is similar between two methods, as well as in the classification accuracy.

But when the shift get larger, the weights are worse and the performance in accuracy decreases, getting even worse than the unweighted classifier.(a) (b) Figure 7 : (a) Mean squared error in estimated weights and (b) accuracy on MNIST for minority-class shifted source and uniform target with p = 0.01, with h 0 trained on tweak-one shifted source data.

FIG8 illustrates the weight estimation alongside final classification performance for Minority-Class source shift of MNIST.

We use 1000 training and testing data.

We created large shifts of three or more minority classes with p = 0.005.

We use a fixed black-box classifier that is trained on biased source data, with tweak-one ρ = 0.5.

Observe that the MSE in weight estimation is relatively large and RLLS outperforms BBSL as the number of minority classes increases.

As the shift increases the performance for all methods deteriorates.

Furthermore, FIG8 (b) illustrates how the advantage of RLLS over the unweighted classifier increases as the shift increases.

Across all shifts, the RLLS based classifier yields higher accuracy than the one based on BBSL.

A.2 CIFAR10 EXPERIMENT UNDER DIRICHLET SOURCE SHIFTS Figure 9 illustrates the weight estimation alongside final classification performance for Dirichlet source shift of CIFAR10 dataset.

We use 10000 training and testing data in this experiment, following the way we generate shift on source data.

We train h 0 with tweak-one shifted source data with ρ = 0.5.

The results show that importance weighting in general is not helping the classification in this relatively large shift case, because the weighted methods, including true weights and estimated weights, are similar in accuracy with the unweighted method.

We show the performance of classifier with different regularization λ under a Dirichlet shift with α = 0.5 in Figure 10 .

The training has 5000 examples in this case.

We can see that in this low target sample case, λ = 1 only take over after several hundreds example, while some λ value between 0 and 1 outperforms it at the beginning.

Similar as in the paper, we use different black-box classifier that is corrupted in different levels to show the relation between the quality of black-box predictor and the necessary target sample size.

We use biased source data with tweak-one ρ = 0, 0.2, 0.6 to train the black-box classifier.

We see that we need more target samples for the fully weighted version λ = 1 to take over for a more corrupted black-box classifier.

DISPLAYFORM0 where we use the shorthand Υ(θ ) = Cθ − b 2 .We can get an upper bound on the right hand side of FORMULA25 is the infimum by simply choosing a feasible θ = θ.

We then have Cθ − b 2 = 0 and hence DISPLAYFORM1 as a consequence, DISPLAYFORM2 by definition of the minimum singular value, we thus have DISPLAYFORM3 Let us first notice that DISPLAYFORM4 The mathematical definition of the finite sample estimates C h , b h (in matrix and vector representation) with respect to some hypothesis h are as follows DISPLAYFORM5 where m = |D q | and I is the indicator function.

C h , b h can equivalently be expressed with the population over P for C h and over Q for b h respectively.

We now use the following concentration Lemmas to bound the estimation errors of C, b where we drop the subscript h for ease of notation.

Lemma 2 (Concentration of measurement matrix C) For finite sample estimate C we have DISPLAYFORM6 with probability at least (1 − δ).Lemma 3 (Concentration of label measurements) For the finite sample estimate b with respect to any hypothesis h it holds that DISPLAYFORM7 log FORMULA7 log(1/δ)(1 − β)n p + log(1/δ)

√ n q with probability at least 1 − 2δ.

By Lemma.

2 for concentration of C and Lemma.

3 for concentration of b we now have with probability at least 1 − δ DISPLAYFORM8 which, considering that O( DISPLAYFORM9 , yields the statement of the Lemma 1.

We prove this lemma using the theorem 1.4[Matrix Bernstein] and Dilations technique from Tropp (2012).

We can rewrite C h = E (x,y)∼P e h(x) e y where e(i) is the one-hot-encoding of index i. Consider a finite sequence {Ψ(i)} of independent random matrices with dimension k. By dilations, lets construct another sequence of self-adjoint random matrices of { Ψ(i)} of dimension 2k, such that for all i DISPLAYFORM0 which results in Ψ(i) 2 = Ψ(i) 2 .

The dilation technique translates the initial sequence of random matrices to the sequence of random self-adjoint matrices where we can apply the Matrix Bernstein theorem which states that, for a finite sequence of i.i.d.

self-adjoint matrices Ψ(i), such that, almost surely ∀i, E Ψ(i) = 0 and Ψ(i) ≤ R, then for all t ≥ 0, DISPLAYFORM1 with probability at least 1 − δ where DISPLAYFORM2 due to Eq. 8.

Therefore, thanks to the dilation trick and theorem 1.4[Matrix Bernstein] in Tropp (2012), DISPLAYFORM3 Ψ(i) ≤ R log (2k/δ) 3t + 2 2 log (2k/δ) t with probability at least 1 − δ.

Now, by plugging in Ψ(i) = e h(x(i)) e y(i) − C, we have E Ψ(i) = 0.

Together with Ψ(i) ≤ 2 as well as 2 = E Ψ 2 (i) 2 = 1 and setting t = n, we have DISPLAYFORM4 The proof of this lemma is mainly based on a special case of and appreared at proposition 6 in BID1 , Lemma F.1 in BID0 and Proposition 19 of BID0 .Analogous to the previous section we can rewrite x) ] where e(i) is the one-hot-encoding of index i. Note that (dropping the subscript h) we have DISPLAYFORM5 DISPLAYFORM6 We now bound both estimates of probability vectors separately.

Consider a fixed multinomial distribution characterized with probability vector of ς ∈ ∆ k−1 where DISPLAYFORM7 where ς(i) is the one-hot-encoding of the i'th sample.

Consider the empirical estimate mean of this distribution through empirical average of the samples; ς = DISPLAYFORM8 (1/δ) t with probability at least 1 − δ.

By plugging in ς = q h , ς = q h with t = n q and finally {ς(i)} nq i=1 = {e h(x(i)) }(i) nq and equivalently for p h we obtain; DISPLAYFORM9 with probability at least 1 − 2δ, therefore; DISPLAYFORM10 log FORMULA7 log(1/δ) DISPLAYFORM11 resulting in the statement in the Lemma 3.

We want to ultimately bound |L( h w ) − L(h )|.

By addition and subtraction we have DISPLAYFORM0 where n = βn p and we used optimality of h w .

Here (a) is the weight estimation error and (b) is the finite sample error.

Uniform law for bounding (b) We bound (b) using standard results for uniform laws for uniformly bounded functions which holds since w ∞ ≤ d ∞ (q||p) and ∞ ≤ 1.

Since |w(y) (h(x), y)| ≤ d ∞ (q||p), ∀x, y ∈ X × Y, by deploying the McDiarmid's inequality we then obtain that DISPLAYFORM1 where G( , H) = {g h (x, y) = w(y) (h(x), y) : h ∈ H} and the Rademacher complexity is defined as DISPLAYFORM2 DISPLAYFORM3 .

Notice that by definition 1 ≤ n and ∞ ≤ n from which it follows by Hoelder's inequality that 2 ≤ n. Furthermore, we slightly abuse notation and use w to denote the k-dimensional vector with w i = w(i).

Therefore, for all h we have via the Cauchy Schwarz inequality that DISPLAYFORM4 It then follows by Lemma 1 that DISPLAYFORM5 Lemma 4 (McDiarmid-Doob-Freedman-Rademacher) For a given A hypothesis class H, a set G( , H) = {g h (x, y) = w(y) (h(x), y) : h ∈ H}, under n data points and loss function we have DISPLAYFORM6 with probability at least 1 − δ Plugging both bounds back into equation FORMULA47 concludes the proof of the theorem.

With a bit abuse of notation let's restate the empirical loss with known importance weights instead on the random variables DISPLAYFORM0 We further define a ghost data set {(X i , Y i )} n 1 and the corresponding ghost loss; DISPLAYFORM1 This random variable is the key to derive the tight generalization bound in Lemma 4.This random variable has the following properties; DISPLAYFORM2 Which we can rewrite as DISPLAYFORM3 and swapping the sup with the expectation DISPLAYFORM4 We can remove the condition with law of iterated conditional expectation and have expectation on both of the data sets; DISPLAYFORM5 we further open the expression up; DISPLAYFORM6 In the following we use the usual symmetrizing technique through Rademacher variables {ξ i } n 1 .

Each ξ i is a uniform random variable either 1 or −1.

Therefore since (h(X ), Y ) − (h(X ), Y ) is a symmetric random variable we have DISPLAYFORM7 where the expectation is also over the Rademacher variables.

After propagation sup DISPLAYFORM8 By propagating the expectation and again symmetry in the Rademacher variable we have DISPLAYFORM9 where the right hand side is two times the Rademacher complexity of class G( , H).

Consider a sequence of Doob Martingale and filtration (U j , F j ) defined on some probability space (Ω, F, P r); DISPLAYFORM10 and the corresponding Martingale difference; DISPLAYFORM11 In the following we show that each |D j | is bounded above.

DISPLAYFORM12 DISPLAYFORM13 The same way we can bound −D j .

Therefore the absolute value each D j is bounded by DISPLAYFORM14 .

In the following we bound the conditional second moment, E D DISPLAYFORM15 Let's construct an event C j the event that a is bigger than b , and also C j its compliment.

Therefore, for the E D 2 j |F j−1 we have DISPLAYFORM16 For the firs term in Eq. 11 after again introducing ghost variables X , Y we have the following upper bound DISPLAYFORM17 DISPLAYFORM18 Moreover, if we multiply each loss with −1, it results in hypothesis class of −H which has the same Rademacher complexity as H, due the symmetric Rademacher random variable.

Let G n denote the same quantity as G n but on −H. We use this slack variable in order to bound the absolute value of G n .

Therefore DISPLAYFORM19 and the same bound for G n .

By solving it for and δ we have |G n | ≤ 2R(G( , H)) + 2d ∞ (q||p) log(2/δ) n + 2 d(q||p) log(2/δ) n Note: A few days prior to the camera ready submission, we realized that a quite similar analysis and statement to Theorem 4 is also studied in Ying (2004).

For finite hypothesis classes, one may bound (b) in (9) using Bernstein's inequality.

We bound (b) by first noting that w(Y ) (Y, h(X)) satisfies the Bernstein conditions so that Bernstein's inequality holds DISPLAYFORM0 by definition.

Because we assume ≤ 1, we directly have DISPLAYFORM1 Since we have a bound on the second moment of weighted loss while its first moment is L(h)

we can apply Bernstein's inequality to obtain for any fixed h that DISPLAYFORM2 For the uniform law for finite hypothesis classes make the union bound on all the hypotheses; as the deviation form label shift constraint which is zero in label shift setting.

Remark 1 (Drift in Label shift assumption) If the label shift assumption slightly violates, for the true importance weights ω(x, y) := q(x,y) p(x,y) , the RLLS, with high probability generalizes as; DISPLAYFORM3 Consider the case where the Label shift assumption is slightly violated, i.e., for each covariate and label, we have p(x|y) q(x|y), resulting importance weight ω(x, y) := q(x,y) p(x,y) for each covariate and label.

Similar to decomposing in Eq. 9, we have DISPLAYFORM4 where the desired excess risk is defined with respect to ω.

The differences between Eq. 15 and Eq. 9 are in a new term (c) as well as term (a).

The term (b) remains untouched.

Where in the last inequality we deploy Cauchy Schwarz inequality as well as loss is in [0, 1] and hold for h ∈ H.

It is worth noting that the expectation in d e (q||p) is with respect to Q and does not blow up if the supports of P and Q do not matchBound on term (a) For any h ∈ H, similar to the derivation in Eq. 10 we have DISPLAYFORM5 The previous weight estimation analysis does not directly hold for this case where the label shift is slightly violated, but with a few modification we provide an upper-bound on the error.

Given a classifier h 0 q h0 (Y = i) = where p(h 0 (X) = i) = q(h 0 (X) = i), resulting; DISPLAYFORM6 where we drop the h 0 in both b e and C. Both the confusion matrix C and the label distribution q h0 on the target for the black box hypothesis h 0 are unknown and we are instead only given access to finite sample estimates C h0 , q h0 .

Similar to previous analysis we have b := q − C1 = Cθ with corresponding finite sample quantity b = q − C1.

Similarly to the analysis when there was no violation in label shift assumption, we have Υ(θ ) = Cθ − b − b e 2 and the solution to Eq. 3 satisfies; DISPLAYFORM7 We can simplify the upper bound by setting θ = θ.

We then have Υ( θ) = C θ − b − b e 2 = C θ − θ 2 ≤ 2∆ C θ 2 + 2∆ b + 2 b e 2 ≤ 2∆ C θ 2 + 2∆ b + 2d e (q||p) resulting in θ − θ 2 ≤ 1 σ min (2∆ C θ 2 + 2∆ b + 2d e (q||p))

@highlight

A practical and provably guaranteed approach for training efficiently classifiers in the presence of label shifts between Source and Target data sets

@highlight

The authors propose a new algorithm for improving the stability of class importance weighting estimation procedure with a two-step procedure.

@highlight

The authors consider the problem of learning under label shifts, where label proportions differ while conditionals are equal, and propose an improved estimator with regularization.