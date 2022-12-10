Compressed representations generalize better (Shamir et al., 2010), which may be crucial when learning from limited or noisy labeled data.

The Information Bottleneck (IB) method (Tishby et al. (2000)) provides an insightful and principled approach for balancing compression and prediction in representation learning.

The IB objective I(X; Z) − βI(Y ; Z) employs a Lagrange multiplier β to tune this trade-off.

However, there is little theoretical guidance for how to select β.

There is also a lack of theoretical understanding about the relationship between β, the dataset, model capacity, and learnability.

In this work, we show that if β is improperly chosen, learning cannot happen: the trivial representation P(Z|X) = P(Z) becomes the global minimum of the IB objective.

We show how this can be avoided, by identifying a sharp phase transition between the unlearnable and the learnable which arises as β varies.

This phase transition defines the concept of IB-Learnability.

We prove several sufficient conditions for IB-Learnability, providing theoretical guidance for selecting β.

We further show that IB-learnability is determined by the largest confident, typical, and imbalanced subset of the training examples.

We give a practical algorithm to estimate the minimum β for a given dataset.

We test our theoretical results on synthetic datasets, MNIST, and CIFAR10 with noisy labels, and make the surprising observation that accuracy may be non-monotonic in β.

Compressed representations generalize better (Shamir et al., 2010) , which is likely to be particularly important when learning from limited or noisy labels, as otherwise we should expect our models to overfit to the noise.

Tishby et al. (2000) introduced the Information Bottleneck (IB) objective function which learns a representation Z of observed variables (X, Y ) that retains as little information about X as possible, but simultaneously captures as much information about Y as possible:min IB β (X, Y ; Z) = min I(X; Z) − βI(Y ; Z)I(X; Y ) = dx dy p(x, y)log p(x,y) p(x)p(y) is the mutual information.

The hyperparameter β controls the trade-off between compression and prediction, in the same spirit as Rate-Distortion Theory (Shannon, 1948) , but with a learned representation function P (Z|X) that automatically captures some part of the "semantically meaningful" information, where the semantics are determined by the observed relationship between X and Y .The IB framework has been extended to and extensively studied in a variety of scenarios, including Gaussian variables BID6 ), meta-Gaussians (Rey & Roth (2012) ), continuous variables via variational methods BID3 ; BID5 BID8 ), deterministic scenarios (Strouse & Schwab (2017a) ; BID12 ), geometric clustering (Strouse & Schwab (2017b) ), and is used for learning invariant and disentangled representations in deep neural nets BID0 b) ).

However, a core issue remains: how should we select β?

In the original work, the authors recommend sweeping β > 1, which can be prohibitively expensive in practice, but also leaves open interesting theoretical questions around the relationship between β, P (Z|X), and the observed data, P (X, Y ).

For example, under how much label noise will IB at a given β still be able to learn a useful representation?This work begins to answer some of those questions by characterizing the onset of learning.

Specifically:• We show that improperly chosen β may result in a failure to learn: the trivial solution P (Z|X) = P (Z) becomes the global minimum of the IB objective, even for β 1.• We introduce the concept of IB-Learnability, and show that when we vary β, the IB objective will undergo a phase transition from the inability to learn to the ability to learn.• Using the second-order variation, we derive sufficient conditions for IB-Learnability, which provide theoretical guidance for choosing a good β.• We show that IB-learnability is determined by the largest confident, typical, and imbalanced subset of the training examples, reveal its relationship with the slope of the Pareto frontier at the origin on the information plane I(Y ; Z) vs. I(X; Z), and discuss its relation with model capacity.

We use our main results to demonstrate on synthetic datasets, MNIST (LeCun et al., 1998) , and CIFAR10 BID13 ) under noisy labels that the theoretical prediction for IB-Learnability closely matches experiment.

We present an algorithm for estimating the onset of IB-Learnability, and demonstrate that it does a good job of estimating the theoretical predictions and the empirical results.

Finally, we observe discontinuities in the Pareto frontier of the information plane as β increases, and those dicontinuities correspond to accuracy decreasing as β increases.

We are given instances of (x, y) ∈ X × Y drawn from a distribution with probability (density) P (X, Y ), where unless otherwise stated, both X and Y can be discrete or continuous variables. (X, Y ) is our training data, and may be characterized by different types of noise.

We can learn a representation Z of X with conditional probability 1 p(z|x), such that X, Y, Z obey the Markov chain Z ← X ↔ Y .

Eq. (1) above gives the IB objective with Lagrange multiplier β, IB β (X, Y ; Z), which is a functional of p(z|x): IB β (X, Y ; Z) = IB β [p(z|x)].

The IB learning task is to find a conditional probability p(z|x) that minimizes IB β (X, Y ; Z).

The larger β, the more the objective favors making a good prediction for Y .

Conversely, the smaller β, the more the objective favors learning a concise representation.

How can we select β such that the IB objective learns a useful representation?

In practice, the selection of β is done empirically.

Indeed, Tishby et al. (2000) recommends "sweeping β".

In this section, we provide theoretical guidance for choosing β by introducing the concept of IB-Learnability and providing a series of IB-learnable conditions.

Definition 1 (IB β -Learnability). (X, Y ) is IB β -learnable if there exists a Z given by some p 1 (z|x), such that DISPLAYFORM0 , where p(z|x) = p(z) characterizes the trivial representation such that Z = Z trivial is independent of X.If (X; Y ) is IB β -learnable, then when IB β (X, Y ; Z) is globally minimized, it will not learn a trivial representation.

If (X; Y ) is not IB β -learnable, then when IB β (X, Y ; Z) is globally minimized, it may learn a trivial representation.

Necessary condition for IB-Learnability.

From Definition 1, we can see that IB β -Learnability for any dataset (X; Y ) requires β > 1.

In fact, from the Markov chain Z ← X ↔ Y , we have I(Y ; Z) ≤ I(X; Z) via the dataprocessing inequality.

If β ≤ 1, then since I(X; Z) ≥ 0 and I(Y ; Z) ≥ 0, we have that min(I(X; Z) − βI(Y ; Z)) = 0 = IB β (X, Y ; Z trivial ).

Hence (X, Y ) is not IB β -learnable for β ≤ 1.Theorem 1 characterizes the IB β -Learnability range for β (see Appendix B for the proof): Theorem 1.

If (X, Y ) is IB β1 -learnable, then for any β 2 > β 1 , it is IB β2 -learnable.

Based on Theorem 1, the range of β such that (X, Y ) is IB β -learnable has the form β ∈ (β 0 , +∞).

Thus, β 0 is the threshold of IB-Learnability.

Furthermore, the trivial representation is a stationary solution for the IB objective: Lemma 1.1.

p(z|x) = p(z) is a stationary solution for IB β (X, Y ; Z).The proof in Appendix E shows that the first-order variation δIB β [p(z|x)] = 0 vanishes at the trivial representation.

Lemma 1.1 yields our strategy for finding sufficient conditions for learnability: find conditions such that p(z|x) = p(z) is not a local minimum for the functional IB β [p(z|x)].

By requiring that the second order variation δ 2 IB β [p(z|x)]

< 0 at the trivial representation (Suff.

Cond.

1, Appendix C), and constructing a special form of perturbation at the trivial representation (Suff.

Cond.

2, Appendix F), we arrive at the key result of this paper (see Appendix G for the proof) 2 :Theorem 2 (Confident Subset Suff.

Cond.).

A sufficient condition for (X, Y ) to be IB β -learnable is X and Y are not independent, and DISPLAYFORM1 where Ω x denotes the event that x ∈ Ω x , with probability p(Ω x ).

Moreover, (inf Ωx⊂X β 0 (Ω x )) −1 gives a lower bound on the slope of the Pareto frontier at the origin of the information plane I(Y ; Z) vs. I(X; Z).Characteristics of dataset leading to low β 0 .

From Eq. (2), we see that three characteristics of the subset Ω x ⊂ X lead to low β 0 : (1) confidence: p(y|Ω x ) is large; (2) typicality and size: the number of elements in Ω x is large, or the elements in Ω x are typical, leading to a large probability of p(Ω x ); (3) imbalance: p(y) is small for the subset Ω x , but large for its complement.

In summary, β 0 will be determined by the largest confident, typical and imbalanced subset of examples, or an equilibrium of those characteristics.

Theorem 2 immediately leads to two important corollaries under special problem structures: classification with classconditional noisy labels BID4 ) and deterministic mappings.

Corollary 2.1.

Suppose that the true class labels are y * , and the input space belonging to each y * has no overlap.

We only observe the corrupted labels y with class-conditional noise p(y|x, y * ) = p(y|y * ) = p(y).

Then a sufficient condition for IB β -Learnability is: DISPLAYFORM2 Corollary 2.2.

For classification problems, if Y is a deterministic function of X and not independent of X, then a necessary and sufficient condition for IB β -Learnability is β > β 0 = 1.Therefore, if we find that β 0 > 1 for a classification task, we may infer that Y is not a deterministic function of X, i.e. either some classes have overlap, or the labels are noisy.

However, finite models may add effective class overlap if they have insufficient capacity for the learning task.

This may translate into a higher observed β 0 , even when learning deterministic functions.

Proofs are provided in Appendix H.

Based on Theorem 2, for general classification tasks we suggest Algorithm 1 in Appendix J to empirically estimate an upper-boundβ 0 ≥ β 0 .

Here, we give the intuition behind the algorithm.

First, we train a single maximum likelihood model on the dataset.

That model provides estimates for all p(y|x) in the training set.

Since learnability is defined with respect to the training data, it is correct to directly use the empirical probability of p(x) and p(y) in the training data.

Given p(x), p(y), and p(y|x), and the understanding that we are seaching for a confident subset Ω x , we can then perform an efficient targeted search of the exponential space of subsets of the training data.

The algorithm returns the lowest estimate ofβ 0 found during that process.

After estimatingβ 0 , we can then use it for learning with IB, either directly, or as an anchor for a region where we can perform a much smaller sweep than we otherwise would have.

This may be particularly important for very noisy datasets, where β 0 can be very large.

To test our theoretical results and Alg.

1, we perform experiments on synthetic datasets, MNIST, and CIFAR10.

Additional experiment details are provided in Appendix K.Synthetic datasets.

We generate a set of synthetic datasets with varying class-conditional noise rates.

FIG0 shows the results of sweeping β to find the empirical onset of learning, and compares that onset to the predicted onset using Eq. (3).

Clearly the estimate provides a tight upper bound in this simple setting.

Also note that β 0 grows exponentially as the label noise increases, underscoring that improperly-chosen β may result in an inability to learn useful representations, and the importance of theoretically-guided β selection as opposed to sweeping β in general.

MNIST.

We perform binary classification with digits 0 and 1, but again add class-conditional noise to the labels with varying noise rates ρ.

To explore how the model capacity influences the onset of learning, for each dataset we train two sets of Variational Information Bottleneck (Alemi et al., 2016) (VIB) models differing only by the number of neurons in their hidden layers of the encoder: one with n = 128 neurons, the other with n = 512 neurons.

Insufficient capacity will result in more uncertainty of Y given X from the point of view of the model, so we expect β 0,observed for the n = 128 model to be larger.

FIG0 confirms this prediction.

It also shows the β 0,estimated and β 0,predicted given by Algorithm 1 and Eq. (3), respectively.

We see that Algorithm 1 does a good job estimating the onset of learning for the large-capacity model, and that the estimated results line up well with the theoretical predictions.

CIFAR10 forgetting.

For CIFAR10 BID13 ), we study how forgetting varies with β.

In other words, given a VIB model trained at some high β 2 , if we anneal it down to some much lower β 1 , what accuracy does the model converge to?

We estimated β 0 = 1.0483 on a version of CIFAR10 with 20% label noise using Alg.

1.

The lowest β with performance above chance was β = 1.048.

See Appendix K.1 for experiment details.

As can be seen in FIG0 , there are large discontinuities in the Pareto frontier, even though we vary β in very small increments.

Those discontinuities start at points on the Pareto frontier where many values of β yield essentially the same I(X; Z) and I(Y ; Z), and end when β crosses apparent phase transitions that give large increases in both I(X; Z) and I(Y ; Z) (marked with red arrows).

FIG0 shows that the lowest value of β in each such region tends to have the highest accuracy.

A primary empirical result of our work is the following: some datasets have non-monotonic performance in regions where multiple values of β cluster together.

This surprising behavior is important to check for when training IB models.

More thorough study is needed, but based on our initial results, we may expect that reducing β to the minimal value that achieves a particular point on the information plane yields better representations.

The phenomenon of discontinuities is also observed in prediction error vs. information in the model parameter BID0 ; BID2 ), I(c; X) vs. H(c) (c denotes clusters) in geometric clustering (Strouse & Schwab (2017b) ).

Although these discontinuities (including ours) are observed via different axes, we conjecture that they may all have a shared root cause, which is an interesting topic for future research.

In this paper, we have presented theoretical results for predicting the onset of learning, and have shown that it is determined by the largest confident, typical and imbalanced subset of the examples.

We gave a practical algorithm for predicting the transition, and showed that those predictions are accurate, even in cases of extreme label noise.

We have also observed a surprising non-monotonic relationship between β and accuracy, and shown its relationship to discontinuities in the Pareto frontier of the information plane.

We believe these results will provide theoretical and practical guidance for choosing β in the IB framework for balancing prediction and compression.

Our work also raises other questions, such as whether there are other phase transitions in learnability that might be identified.

We hope to address some of those questions in future work.

Mélanie Rey and Volker Roth.

Meta-gaussian information bottleneck.

In Advances in Neural Information Processing Systems, pp.

1916 Systems, pp.

-1924 Systems, pp. , 2012 .Ohad Shamir, Sivan Sabato, and Naftali Tishby.

Learning and generalization with the information bottleneck.

The structure of the Appendix is as follows.

In Appendix A, we provide preliminaries for the first-order and secondorder variations on functionals.

Then we prove Theorem 1 in Appendix B. In Appendix C, we state and prove Sufficient Condition 1 for IB β -learnability.

In Appendix D, we calculate the first and second variations of IB β [p(z|x)] at the trivial representation p(z|x) = p(z), which is used in proving the Sufficient Condition 2 IB β -learnability (Appendix F).

After these preparations, we prove the key result of this paper, Theorem 2, in Appendix G. Then two important corollaries 2.1, 2.2 are proved in Appendix H. We provide additional discussions and insights for Theorem 2 in Appendix I, and Algorithm 1 for estimation of an upper boundβ 0 ≥ β 0 in Appendix J. Finally in Appendix K, we provide details for the experiments.

Let functional F [f (x)] be defined on some normed linear space R. Let us add a perturbative function h(x) to f (x), and now the functional F [f (x) + h(x)] can be expanded as DISPLAYFORM0 where ||h|| denotes the norm of h, DISPLAYFORM1 is a linear functional of h(x), and is called the first-order DISPLAYFORM2 is a quadratic functional of h(x), and is called the second- DISPLAYFORM3 B PROOF OF THEOREM 1Proof.

At the trivial representation p(z|x) = p(z), we have I(X; Z) = 0, and I(Y ; Z) = 0 due to the Markov chain, so DISPLAYFORM4 C SUFFICIENT CONDITION 1 AND PROOFIn this section, we prove the Sufficient Condition 1 for IB β -learnability, which will lay the foundation for the Sufficient condition 2 (Appendix F) and the Confident Subset Sufficient condition (key result of this paper, Theorem 2) that follow.

Theorem 3 (Suff.

Cond.

1).

A sufficient condition for (X, Y ) to be IB β -learnable is that there exists a perturbation function h(z|x) with 3 h(z|x)dz = 0, such that the second-order variation DISPLAYFORM5 Proof.

To prove Theorem 3, we use the Theorem 1 of Chapter 5 of BID9 which gives a necessary condition for F [f (x)] to have a minimum at f 0 (x).

Adapting to our notation, we have:Theorem 4 (Gelfand et al. FORMULA2 ).

A necessary condition for the functional F [f (x)] to have a minimum at f (x) = f 0 (x) is that for f (x) = f 0 (x) and all admissible h(x), DISPLAYFORM6 DISPLAYFORM7 , let us calculate the first and second-order variation of I(X; Z) and I(Y ; Z) w.r.t.

p(z|x), respectively.

Through this derivation, we use h(z|x) as a perturbative function, for ease of deciding different orders of variations.

We will finally absorb into h(z|x).

DISPLAYFORM8 We have DISPLAYFORM9 Expanding F 1 [p(z|x) + h(z|x)] to the second order of , we have DISPLAYFORM10 Collecting the first order terms of , we have DISPLAYFORM11 Collecting the second order terms of 2 , we have DISPLAYFORM12 Now let us calculate the first and second-order variation of F 2 [p(z|x)] = I(Z; Y ).

We have DISPLAYFORM13 Using the Markov chain Z ← X ↔ Y , we have DISPLAYFORM14 Then expanding F 2 [p(z|x) + h(z|x)] to the second order of , we have DISPLAYFORM15 Collecting the first order terms of , we have DISPLAYFORM16 Collecting the second order terms of , we have DISPLAYFORM17 Finally, we have DISPLAYFORM18 DISPLAYFORM19 Absorb into h(z|x), we get rid of the factor and obtain the final expression in Lemma 4.1.E PROOF OF LEMMA 1.1Proof.

Using Lemma 4.1, we have DISPLAYFORM20 Let p(z|x) = p(z) (the trivial representation), we have that log p(z|x) p(z) ≡ 0.

Therefore, the two integrals are both 0.

Hence, DISPLAYFORM21 F SUFFICIENT CONDITION 2 AND PROOF

Theorem 5 (Suff.

Cond.

2).

A sufficient condition for (X, Y ) to be IB β -learnable is X and Y are not independent, and β > inf DISPLAYFORM0 where the functional β 0 [h(x)] is given by DISPLAYFORM1 Moreover, we have that inf h(x) β[h(x)]

−1 is a lower bound of the slope of the Pareto frontier in the information plane I(Y ; Z) vs. I(X; Z) at the origin.

The proof is given in Appendix F, which also gives a construction for h(z|x) for Theorem 3 for any h(x) satisfying Theorem 5, and shows that the converse is also true: if there exists h(z|x) suth that the condition in Theorem 3 is true, then we can find h(x) satisfying the the condition in Theorem 5.The geometric meaning of (β 0 [h(x)]) −1 is as follows.

It equals DISPLAYFORM2 under a perturbation function of the form h 1 (z|x) = h(x)h 2 (z) (satisfying h 2 (z)dz = 0 and δ 2 I(X;Z) , which turns out to be equal to DISPLAYFORM3 under the class of perturbation functions h 1 (z|x) = h 2 (z)h(x), and provides a lower bound of sup h(z|x) DISPLAYFORM4 , which is the slope of the Pareto frontier in the information plane I(Y ; Z) vs. I(X; Z) at the origin.

Theorem 5 in essence states that as long as β −1 is lower than this lower bound of the slope of the Pareto frontier, (X; Y ) is IB β -learnable.

From Theorem 5, we see that it still has an infimum over an arbitrary function h(x), which is not easy to estimate.

To get rid of h(x), we can use a specific functional form for h(x) in Eq. FORMULA26 , and obtain a stronger sufficient condition for IB β -Learnability.

But we want to choose h(x) as near to the infimum as possible.

To do this, we note the following characteristics for the R.H.S of Eq. (5):• We can set h(x) to be nonzero if x ∈ Ω x for some region Ω x ⊂ X and 0 otherwise.

Then we obtain the following sufficient condition: DISPLAYFORM5 • The numerator of the R.H.S. of Eq. (6) attains its minimum when h(x) is a constant within Ω x .

This can be proved using the Cauchy-Schwarz inequality: DISPLAYFORM6 , and defining the inner product as u, v = u(x)v(x)dx.

Therefore, the numerator of the R.H.S. of Eq. (6) ≥ − 1, and attains equality when DISPLAYFORM7 Based on these observations, we can let h(x) be a nonzero constant inside some region Ω x ⊂ X and 0 otherwise, and the infimum over an arbitrary function h(x) is simplified to infimum over Ω x ⊂ X , and we obtain the confident subset sufficient condition (Theorem 2) for IB β -Learnability, which is a key result of this paper.

Proof.

Firstly, from the necessary condition of β > 1 in Section 2, we have that any sufficient condition for IB β -learnability should be able to deduce β > 1.

Now using Theorem 3, a sufficient condition for (X, Y ) to be IB β -learnable is that there exists h(z|x) with h(z|x)dx = 0 such that DISPLAYFORM0 At the trivial representation, p(z|x) = p(z) and hence p(x, z) = p(x)p(z).

Due to the Markov chain Z ← X ↔ Y , we have p(y, z) = p(y)p(z).

Substituting them into the δ 2 IB β [p(z|x)] in Lemma 4.1, the condition becomes: there exists h(z|x) with h(z|x)dz = 0, such that DISPLAYFORM1 Rearranging terms and simplifying, we have DISPLAYFORM2 where DISPLAYFORM3 Now we prove that the condition that ∃h(z|x) s.t.

p(z) dz > 0, and let h 1 (z|x) = h(x)h 2 (z).

Now we have DISPLAYFORM4 DISPLAYFORM5 In other words, the condition Eq. FORMULA35 is equivalent to requiring that there exists an h(x) such that G[h(x)] < 0 .

Hence, a sufficient condition for IB β -learnability is that there exists an h(x) such that DISPLAYFORM6 When h(x) = C = const in the entire input space X , Eq. (8) becomes: DISPLAYFORM7 which cannot be true.

Therefore, h(x) = const cannot satisfy Eq. (8).Rearranging terms and simplifying, and note that dxh(x)p(x) 2 > 0 due to h(x) ≡ 0 = const, we have DISPLAYFORM8 For the R.H.S. of Eq. (9), let us show that it is greater than 0.

Using Cauchy-Schwarz inequality: u, u v, v ≥ u, v 2 , and setting u(x) = h(x) p(x), v(x) = p(x), and defining the inner product as DISPLAYFORM9 It attains equality when DISPLAYFORM10 v(x) = h(x) is constant.

Since h(x) cannot be constant, we have that the R.H.S. of Eq. (9) is greater than 0.For the L.H.S. of Eq. (9), due to the necessary condition that β > 0, if FORMULA42 cannot hold.

Then the h(x) such that Eq. (9) holds is for those that satisfies DISPLAYFORM11 DISPLAYFORM12 We see this constraint contains the requirement that h(x) ≡ const.

Written in the form of expectations, we have DISPLAYFORM13 Since the square function is convex, using Jensen's inequality on the outer expectation on the L.H.S. of Eq. (10), we have DISPLAYFORM14 The equality holds iff E x∼p(x|y) [h(x)] is constant w.r.t.

y, i.e. Y is independent of X. Therefore, in order for Eq. FORMULA0 to hold, we require that Y is not independent of X.Using Jensen's inequality on the innter expectation on the L.H.S. of Eq. (10), we have DISPLAYFORM15 The equality holds when h(x) is a constant.

Since we require that h(x) is not a constant, we have that the equality cannot be reached.

Under the constraint that Y is not independent of X, we can divide both sides of Eq. 8, and obtain the condition: there exists an h(x) such that DISPLAYFORM16 Written in the form of expectations, we have DISPLAYFORM17 We can absorb the constraint Eq. (10) into the above formula, and get DISPLAYFORM18 where DISPLAYFORM19 which proves the condition of Theorem 5.Furthermore, from Eq. FORMULA0 we have DISPLAYFORM20 for h(x) ≡ const, which satisfies the necessary condition of β > 1 in Section 2.Proof of lower bound of slope of the Pareto frontier at the origin: Inflection point for general Z: If we do not assume that Z is at the origin of the information plane, but at some general stationary solution Z * with p(z|x), we define DISPLAYFORM21 DISPLAYFORM22 DISPLAYFORM23 It becomes a non-stable solution (non-minimum), and we will have other Z that achieves a better IB β (X, Y ; Z) than the current Z * .Multiple phase transitions To discuss multiple phase transitions, let us first obtain the β (1) for stationary solution for the IB objective.

At a stationary solution for IB β [p(z|x)], for valid perturbation h(z|x) satisfying dzh(z|x) = 0 for any x, we have δ IB β [p(z|x)] − dzdxλ(x)p(z|x) = 0 as a constraint optimization with λ(x) as Lagrangian multipliers.

Using Eq. (4), we have DISPLAYFORM24 Therefore, we have DISPLAYFORM25 The last equality is due to that the first equality is always true for any function h(z|x).

So we can take out the dxdzh(z|x) factor.

λ(x) is used for normalization of p(z|x).

Eq. FORMULA0 is equivalent to the result of the self-consistent equation in Tishby et al. (2000) .Eq.

FORMULA0 and Eq. (12) provide us with an ideal tool to study multiple phase transitions.

For each β, at the minimization of the IB objective, Eq. FORMULA0 is satisfied by some Z * that is at the Pareto frontier on the I(Y ; Z) vs. I(X; Z) plane.

As we increase β, the inf h(x) β (2) [h(x)] may remain stable for a wide range of β, until β is greater than inf h(x) β (2) [h(x)], at which point we will have a phase transition where suddenly there is a better Z = Z * * that achieves much lower IB β (X, Y ; Z) value.

For example, we can rewrite Eq. FORMULA0 as DISPLAYFORM26 whereλ DISPLAYFORM27 p(x) .

By substituting into Eq. FORMULA0 , we may proceed and get useful results.

Proof.

According to Theorem 5, a sufficient condition for (X, Y ) to be IB β -learnable is that X and Y are not independent, and DISPLAYFORM0 We can assume a specific form of h(x), and obtain a (potentially stronger) sufficient condition.

Specifically, we let DISPLAYFORM1 for certain Ω x ⊂ X .

Substituting into Eq. (16), we have that a sufficient condition for (X, Y ) to be IB β -learnable is DISPLAYFORM2 where DISPLAYFORM3 The denominator of Eq. FORMULA0 is DISPLAYFORM4 Using the inequality x − 1 ≥ logx, we have DISPLAYFORM5 Both equalities hold iff p(y|Ω x ) ≡ p(y), at which the denominator of Eq. FORMULA0 is equal to 0 and the expression inside the infimum diverge, which will not contribute to the infimum.

Except this scenario, the denominator is greater than 0.

Substituting into Eq. FORMULA0 , we have that a sufficient condition for (X, Y ) to be IB β -learnable is DISPLAYFORM6 Since Ω x is a subset of X , by the definition of h(x) in Eq. FORMULA0 , h(x) is not a constant in the entire X .

Hence the numerator of Eq. FORMULA0 is positive.

Since its denominator is also positive, we can then neglect the "> 0", and obtain the condition in Theorem 2.Since the h(x) used in this theorem is a subset of the h(x) used in Theorem 5, the infimum for Eq. FORMULA2 is greater than or equal to the infimum in Eq. (5).

Therefore, according to the second statement of Theorem 5, we have that the (inf Ωx β 0 (Ω x )) −1 is also a lower bound of the slope for the Pareto frontier of I(Y ; Z) vs. I(X; Z) curve.

Now we prove that the condition Eq. FORMULA2 is invariant to invertible mappings of X. In fact, if X = g(X) is a uniquely invertible map (if X is continuous, g is additionally required to be continuous), let X = {g(x)|x ∈ Ω x }, and denote DISPLAYFORM7 Additionally we have X = g(X ).

Then DISPLAYFORM8 For dataset (X , Y ) = (g(X), Y ), applying Theorem 2 we have that a sufficient condition for it to be IB β -learnable is DISPLAYFORM9 where the equality is due to Eq. (20) .

Comparing with the condition for IB β -learnability for (X, Y ) (Eq. FORMULA2 ), we see that they are the same.

Therefore, the condition given by Theorem 2 is invariant to invertible mapping of X. Proof.

We use Theorem 2.Let Ω x contain all elements x whose true class is y * for some certain y * , and 0 otherwise.

Then we obtain a (potentially stronger) sufficient condition.

Since the probability p(y|y * , x) = p(y|y * ) is classconditional, we have DISPLAYFORM10 , we obtain a sufficient condition for IB β learnability.

Proof.

We again use Theorem 2.

Since Y is a deterministic function of X, let Y = f (X).

Since it is classification problem, Y contains at least one value y such that its probability p(y) > 0, we let Ω x contain only x such that f (x) = y. Substituting into Eq. (2), we have DISPLAYFORM0 Therefore, the sufficient condition becomes β > 1.Furthermore, since a necessary condition for IB β -learnability is β > 1 (Section 2), we have that β > β 0 = 1 is a necessary and sufficient condition.

Similarity to information measures.

The denominator of Eq. (2) is closely related to mutual information.

Using the inequality x − 1 ≥ log(x) for x > 0, it becomes: DISPLAYFORM0 whereĨ(Ω x ; Y ) is the mutual information "density" at Ω x ⊂ X .

Of course, this quantity is also D KL [p(y|Ω x )||p(y)], so we know that the denominator of Eq. FORMULA2 is non-negative.

Incidentally, E y∼p (y|Ωx) p (y|Ωx) p(y) − 1 is the density of "rational mutual information" BID15 DISPLAYFORM1 Similarly, the numerator is related to the self-information of Ω x : DISPLAYFORM2 so we can estimate the phase transition as: DISPLAYFORM3 Since Eq. (22) uses upper bounds on both the numerator and the denominator, it does not give us a bound on β 0 .Multiple phase transitions.

Based on this characterization of Ω x , we can hypothesize datasets with multiple learnability phase transitions.

Specifically, consider a region Ω x0 that is small but "typical", consists of all elements confidently predicted as y 0 by p(y|x), and where y 0 is the least common class.

By construction, this Ω x0 will dominate the infimum in Eq. (2), resulting in a small value of β 0 .

However, the remaining X − Ω x0 effectively form a new dataset, X 1 .

At exactly β 0 , we may have that the current encoder, p 0 (z|x), has no mutual information with the remaining classes in X 1 ; i.e., I(Y 1 ; Z 0 ) = 0.

In this case, Definition 1 applies to p 0 (z|x) with respect to I(X 1 ; Z 1 ).

We might expect to see that, at β 0 , learning will plateau until we get to some β 1 > β 0 that defines the phase transition for X 1 .

Clearly this process could repeat many times, with each new dataset X i being distinctly more difficult to learn than X i−1 .

The end of Appendix F gives a more detailed analysis on multiple phase transitions.

Estimating model capacity.

The observation that a model can't distinguish between cluster overlap in the data and its own lack of capacity gives an interesting way to use IB-Learnability to measure the capacity of a set of models relative to the task they are being used to solve.

Learnability and the Information Plane.

Many of our results can be interpreted in terms of the geometry of the Pareto frontier illustrated in FIG1 , which describes the trade-off between increasing I(Y ; Z) and decreasing I(X; Z).

At any point on this frontier that minimizes IB min β ≡ min I(X; Z) − βI(Y ; Z), the frontier will have slope β −1 if it is differentiable.

If the frontier is also concave (has negative second derivative), then this slope β −1 will take its maximum β −1 0 at the origin, which implies IB β -Learnability for β > β 0 , so that the threshold for IB β -Learnability is simply the inverse slope of the frontier at the origin.

More generally, as long as the Pareto frontier is differentiable, the threshold for IB β -learnability is the inverse of its maximum slope.

Indeed, Theorem 2 gives lower bounds of the slope of the Pareto frontier at the origin.

This means that we lack IB β -learnability for β < β 0 , which makes the origin the optimal point.

If the frontier is convex, then we achieve optimality at the upper right endpoint if β > β 1 , otherwise on the frontier at the location between the two endpoints where the frontier slope is β −1 .

Learnability and contraction coefficient If we regard the true mapping from X to Y as a channel with transition kernel P Y |X , we can define contraction coefficient η KL (P Y |X ) = sup Q;P :0<DKL(P ||Q)<∞ BID16 ) as a measure of how much it keeps the two distributions P and Q intact (as opposed to being drawn nearer measured by KL-divergence) after pushing forward through the channel.

By BID16 DISPLAYFORM0 DISPLAYFORM1 .

Theorem 5 hence also provides a lower bound for the contraction coefficient DISPLAYFORM2 Similarly for Theorem 2.

In Alg.

1 we present a detailed algorithm for estimating β 0 .Algorithm 1 Estimating the upper bound for β 0 for IB β -Learnability Require: Dataset D = {(x i , y i )}, i = 1, 2, ...N .

The number of classes is C. Require ε: tolerance for estimating β 0 1: Learn a maximum likelihood model p θ (y|x) using the dataset D. DISPLAYFORM0 .

5: j * = arg max j (P y|x ) ij 6: Sort the rows of P y|x in decreasing values of (P y|x ) ij * .

7: Search i upper untilβ 0 = Getβ(P y|x , p y , Ω) is minimal with tolerance ε, where Ω = {1, 2, ...i upper }.

8: returnβ 0 Subroutine Getβ(P y|x , p y , Ω) s1: (N, C, n) ← (number of rows of P y|x , number of columns of P y|x , number of elements of Ω).

DISPLAYFORM1

We use the Variational Information Bottleneck (VIB) objective by BID3 .

For the synthetic experiment, the latent Z has dimension of 2.

The encoder is a neural net with 2 hidden layers, each of which has 128 neurons with ReLU activation.

The last layer has linear activation and 4 output neurons, with the first two parameterizes the mean of a Gaussian and the last two parameterizes the log variance of the Gaussian.

The decoder is a neural net with 1 hidden layers with 128 neurons and ReLU activation.

Its last layer has linear activation and outputs the logit for the class labels.

It uses a mixture of Gaussian prior with 500 components (for the experiment with class overlap, 256 components), each of which is a 2D Gaussian with learnable mean and log variance, and the weights for the components are also learnable.

For the MNIST experiment, the architecture is mostly the same, except the following: (1) for Z, we let it have dimension of 256.

For the prior, we use standard Gaussian with diagonal covariance matrix.

For all experiments, we use Adam BID11 ) optimizer with default parameters.

We do not add any regularization.

We use learning rate of 10 −4 and have a learning rate decay of 1 1+0.01×epoch .

We train in total 2000 epochs with batch size of 500.

All experiments has train-test split of 5:1, and we report the accuracy on the test set, w.r.t.

the true labels.

For estimation of β 0,exp in FIG0 , in the accuracy vs. β i curve, we take the mean and standard deviation of the accuracy for the lowest 5 β i values, denoting as µ β , σ β .

When β i is greater than µ β + 3σ β , we regard it as learning a non-trivial representation, and take the average of β i and β i−1 as the experimentally estimated onset of learning.

We also inspect manually and confirm that it is consistent with human intuition.

For the estimating β 0,estimated using Alg.

1, at step 7 we use the following discrete search algorithm.

We gradually narrow down the range

We trained a deterministic 28x10 wide resnet BID10 Zagoruyko & Komodakis, 2016) , using the open source implementation from BID7 .

However, we extended the final 10 dimensional logits of that model through another 3 layer MLP classifier, in order to keep the inference network architecture identical between this model and the VIB models we describe below.

During training, we dynamically added label noise according to the class confusion matrix in Tab.

K.1.

The mean label noise averaged across the 10 classes is 20%.

After that model had converged, we used it to estimate β 0 with Alg.

1.

Even with 20% label noise, β 0 was estimated to be 1.0483.We then trained 73 different VIB models using the same 28x10 wide resnet architecture for the encoder, parameterizing the mean of a 10-dimensional unit variance Gaussian.

Samples from the encoder distribution were fed to the same 3 layer MLP classifier architecture used in the deterministic model.

The marginal distributions were mixtures of 500 fully covariate 10-dimensional Gaussians, all parameters of which are trained.

The VIB models had β ranging from 1.02 to 2.0 by steps of 0.02, plus an extra set ranging from 1.04 to 1.06 by steps of 0.001 to ensure we captured the empirical β 0 with high precision.

However, this particular VIB architecture does not start learning until β > 2.5, so none of these models would train as described.

4 Instead, we started them all at β = 100, and annealed β down to the corresponding target over 10,000 training gradient steps.

The models continued to train for another 200,000 gradient steps after that.

In all cases, the models converged to essentially their final accuracy within 20,000 additional gradient steps after annealing was completed.

They were stable over the remaining ∼ 180, 000 gradient steps.

@highlight

Theory predicts the phase transition between unlearnable and learnable values of beta for the Information Bottleneck objective