Given samples from a group of related regression tasks, a data-enriched model describes observations by a common and per-group individual parameters.

In high-dimensional regime, each parameter has its own structure such as sparsity or group sparsity.

In this paper, we consider the general form of data enrichment where data comes in a fixed but arbitrary number of tasks $G$ and any convex function, e.g., norm, can characterize the structure of both common and individual parameters.

We propose an estimator for the high-dimensional data enriched model and investigate its statistical properties.

We delineate the sample complexity of our estimator and provide high probability non-asymptotic bound for estimation error of all parameters under a condition weaker than the state-of-the-art.

We propose an iterative estimation algorithm with a geometric convergence rate.

Overall, we present a first through statistical and computational analysis of inference in the data enriched model.

Over the past two decades, major advances have been made in estimating structured parameters, e.g., sparse, low-rank, etc., in high-dimensional small sample problems BID13 BID6 BID14 .

Such estimators consider a suitable (semi) parametric model of the response: y = φ(x, β * )+ω based on n samples {(x i , y i )} n i=1and β * ∈ R p is the true parameter of interest.

The unique aspect of such high-dimensional setup is that the number of samples n < p, and the structure in β * , e.g., sparsity, low-rank, makes the estimation possible (Tibshirani, 1996; BID7 BID5 ).

In several real world problems, natural grouping among samples arises and learning a single common model β 0 for all samples or many per group individual models β g s are unrealistic.

The middle ground model for such a scenario is the superposition of common and individual parameters β 0 + β g which has been of recent interest in the statistical machine learning community BID16 and is known by multiple names.

It is a form of multi-task learning (Zhang & Yang, 2017; BID17 when we consider regression in each group as a task.

It is also called data sharing BID15 since information contained in different group is shared through the common parameter β 0 .

And finally, it has been called data enrichment BID10 BID0 because we enrich our data set with pooling multiple samples from different but related sources.

In this paper, we consider the following data enrichment (DE) model where there is a common parameter β * 0 shared between all groups plus individual per-group parameters β * g which characterize the deviation of group g: y gi = φ(x gi , (β * 0 + β * g )) + ω gi , g ∈ {1, . . .

, G}, (1) where g and i index the group and samples respectively.

Note that the DE model is a system of coupled superposition models.

We specifically focus on the high-dimensional small sample regime for (1) where the number of samples n g for each group is much smaller than the ambient dimensionality, i.e., ∀g : n g p. Similar to all other highdimensional models, we assume that the parameters β g are structured, i.e., for suitable convex functions f g 's, f g (β g ) is small.

Further, for the technical analysis and proofs, we focus on the case of linear models, i.e., φ(x, β) = x T β.

The results seamlessly extend to more general non-linear models, e.g., generalized linear models, broad families of semi-parametric and single-index models, non-convex models, etc., using existing results, i.e., how models like LASSO have been extended (e.g. employing ideas such as restricted strong convexity (Negahban & Wainwright, 2012) ).In the context of Multi-task learning (MTL), similar models have been proposed which has the general form of y gi = x T gi (β * 1g + β * 2g ) + ω gi where B 1 = [β 11 , . . .

, β 1G ] and B 2 = [β 21 , . . . , β 2G ] are two parameter matrices (Zhang & Yang, 2017) .

To capture relation of tasks, different types of constraints are assumed for parameter matrices.

For example, BID11 assumes B 1 and B 2 are sparse and low rank respectively.

In this parameter matrix decomposition framework for MLT, the most related work to ours is the one proposed by BID17 where authors regularize the regression with B 1 1,∞ and B 2 1,1 where norms are p, q-norms on rows of matrices.

Parameters of B 1 are more general than DE's common parameter when we use f 0 (β 0 ) = β 0 1 .

This is because B 1 1,∞ regularizer enforces shared support of β * 1g s, i.e., supp(β * 1i ) = supp(β * 1j ) but allows β * 1i = β * 1j .

Further sparse variation between parameters of different tasks is induced by B 2 1,1 which has an equivalent effect to DE's individual parameters where f g (·)s are l 1 -norm.

Our analysis of DE framework suggests that it is more data efficient than this setup of BID17 ) because they require every task i to have large enough samples to learn its own common parameters β i while DE shares the common parameter and only requires the total dataset over all tasks to be sufficiently large.

The DE model where β g 's are sparse has recently gained attention because of its application in wide range of domains such as personalized medicine BID12 , sentiment analysis, banking strategy BID15 , single cell data analysis (Ollier & Viallon, 2015) , road safety (Ollier & Viallon, 2014) , and disease subtype analysis BID12 .

In spite of the recent surge in applying data enrichment framework to different domains, limited advances have been made in understanding the statistical and computational properties of suitable estimators for the data enriched model.

In fact, non-asymptotic statistical properties, including sample complexity and statistical rates of convergence, of regularized estimators for the data enriched model is still an open question BID15 Ollier & Viallon, 2014) .

To the best of our knowledge, the only theoretical guarantee for data enrichment is provided in (Ollier & Viallon, 2015) where authors prove sparsistency of their proposed method under the stringent irrepresentability condition of the design matrix for recovering supports of common and individual parameters.

Existing support recovery guarantees (Ollier & Viallon, 2015) , sample complexity and l 2 consistency results BID17 of related models are restricted to sparsity and l 1 -norm, while our estimator and norm consistency analysis work for any structure induced by arbitrary convex functions f g .

Moreover, no computational results, such as rates of convergence of the optimization algorithms associated with proposed estimators, exist in the literature.

We denote sets by curly V, matrices by bold capital V, random variables by capital V , and vectors by small bold v letters.

We take DISPLAYFORM0 Given G groups and n g samples in each as DISPLAYFORM1 , we can form the per group design matrix X g ∈ R ng×p and output vector y g ∈ R ng .The total number of samples is n = G g=1 n g .

The data enriched model takes the following vector form: DISPLAYFORM2 where each row of X g is x T gi and ω T g = (ω g1 , . . . , ω gng ) is the noise vector.

A random variable V is sub-Gaussian if its moments satisfies ∀p ≥ 1 : DISPLAYFORM3 is sub-Gaussian if the one-dimensional marginals v, u are sub-Gaussian random variables for all u ∈ R p .

The sub-Gaussian norm of v is defined (Vershynin, 2012) as , 2018) , where the expectation is over g ∼ N (0, I p×p ), a vector of independent zeromean unit-variance Gaussian.

DISPLAYFORM4

We propose the following Data Enrichment (DE) estimatorβ for recovering the structured parameters where the structure is induced by convex functions f g (·): DISPLAYFORM0 We present several statistical and computational results for the DE estimator (3) of the data enriched model:• The DE estimator (3) succeeds if a geometric condition that we call Data EnRichment Incoherence Condition (DERIC) is satisfied, FIG1 .

Compared to other known geometric conditions in the literature such as structural coherence BID16 and stable recovery conditions BID18 , DERIC is a weaker condition, FIG1 .•

Assuming DERIC holds, we establish a high probability non-asymptotic bound on the weighted sum of parameterwise estimation error, δ g =β g − β * g as: DISPLAYFORM1 where n 0 n is the total number of samples, γ max g∈ [G] n ng is the sample condition number, and C g is the error cone corresponding to β * g exactly defined in Section 2.

To the best of our knowledge, this is the first statistical estimation guarantee for the data enrichment.

• We also establish the sample complexity of the DE estimator for all parameters as ∀g ∈ [G] : DISPLAYFORM2 We emphasize that our result proofs that the recovery of the common parameter β 0 by DE estimator benefits from all of the n pooled samples.• We present an efficient projected block gradient descent algorithm DICER, to solve DE's objective (3) which converges geometrically to the statistical error bound of (4).

To the best of our knowledge, this is the first rigorous computational result for the high-dimensional data-enriched regression.

A compact form of our proposed DE estimator (3) is: DISPLAYFORM0 where y = (y DISPLAYFORM1 Example 1. (L 1 -norm) When all parameters β g s are s gsparse, i.e.,|supp(β * g )| = s g by using l 1 -norm as the sparsity inducing function, DE (5) instantiates to the spare DE: DISPLAYFORM2 Consider the group-wise estimation error δ g =β g − β * g .

Sinceβ g = β * g + δ g is a feasible point of (5), the error vector δ g will belong to the following restricted error set: DISPLAYFORM3 We denote the cone of the error set as C g Cone(E g ) and the spherical cap corresponding to it as DISPLAYFORM4 following two subsets of C play key roles in our analysis: DISPLAYFORM5 DISPLAYFORM6 Using optimality ofβ, we can establish the following deterministic error bound.

DISPLAYFORM7

The main assumptions of Theorem 1 is known as Restricted Eigenvalue (RE) condition in the literature of high dimensional statistics Negahban et al., 2012; Raskutti et al., 2010) : DISPLAYFORM0 Here, we show that for the design matrix X defined in (6), the RE condition holds with high probability under a suitable geometric condition we call Data EnRichment Incoherence Condition (DERIC) and for enough number of samples.

For the analysis, similar to existing work (Tropp, 2015; BID19 BID16 , we assume the design matrix to be isotropic sub-Gaussian.

1 Definition 1.

We assume x gi are i.i.d.

random vectors from a non-degenerate zero-mean, isotropic sub-Gaussian distribution.

In other words, DISPLAYFORM1 Further, we assume noise ω gi are i.i.d.

zero-mean, unit-variance sub-Gaussian with |||ω gi ||| ψ2 ≤ K.

).

There exists a non-empty set I ⊆

[G] \ of groups where for some scalars 0 <ρ ≤ 1 and λ min > 0 the following holds:

i∈I n i ≥ ρn .

2.

∀i ∈ I, ∀δ i ∈ C i , and δ 0 ∈ C 0 : DISPLAYFORM0 Using DERIC and the small ball method BID19 , a recent tool from empirical process theory in the following theorem, we elaborate the sample complexity required for satisfying the RE condition: Theorem 2.

Let x gi s be random vectors defined in Definition 1.

Assume DERIC condition of Definition 2 holds for error cones C g s and ψ I = λ minρ /3.

Then, for all δ ∈ H, when we have enough number of samples as ∀g DISPLAYFORM1 , with probability at least 1 − e −nκmin/4 we have inf δ∈H DISPLAYFORM2 4 is the lower bound of the RE condition.

Example 2. (L 1 -norm) The Gaussian width of the spherical cap of a p-dimensional s-sparse vector is ω(A) = Θ( √ s log p) Vershynin, 2018) .

Therefore, the number of samples per group and total required for satisfaction of the RE condition in the sparse DE estimator FORMULA10 is ∀g ∈ [G] : n g ≥ m g = Θ(s g log p).

Here, we provide a high probability upper bound for the deterministic upper bound of Theorem 1 and derive the final estimation error bound.

Theorem 3.

Assume x gi and ω gi distributed according to Definition 1 and τ > 0, then with probability at least 1 DISPLAYFORM0 we have: DISPLAYFORM1 The following corollary characterizes the general error bound and results from the direct combination of Theorem 1, Theorem 2, and Theorem 3.

Corollary 1.

For x gi and ω gi described in Definition 1 and τ > 0 when we have enough number of samples ∀g ∈ [G] : n g > m g which lead to κ > 0, the following general error bound holds with high probability for estimator (5): DISPLAYFORM2 Example 3. (L 1 -norm) For the sparse DE estimator of FORMULA10 , results of Theorem 2 and 3 translates to the following: For enough number of samples as ∀g DISPLAYFORM3 , the error bound of (11) simplifies to: DISPLAYFORM4 Therefore, individual errors are bounded as δ g 2 = O( (max g∈ [G] s g ) log p/n g ) which is slightly worse than DISPLAYFORM5 for g=1 to G do 5: DISPLAYFORM6 end for 7: DISPLAYFORM7 O( s g log p/n g ), the well-known error bound for recovering an s g -sparse vector from n g observations using LASSO or similar estimators BID8 BID4 BID9 BID2 .

Note that max g∈ [G] s g (instead of s g ) is the price we pay to recover the common parameter β 0 .

We propose Data enrIChER (DICER) a projected block gradient descent algorithm, Algorithm 1, where Π Ω fg is the Euclidean projection onto the set DISPLAYFORM0 To analysis convergence properties of DICER, we should upper bound the error of each iteration.

Let's δ (t) = β (t) − β * be the error of iteration t of DICER, i.e., the distance from the true parameter (not the optimization minimum,β).

We show that δ (t) 2 decreases exponentially fast in t to the statistical error δ 2 = β − β * 2 .

DISPLAYFORM1 2 , updates of the Algorithm 1 obey the following with high probability: DISPLAYFORM2 where r(τ ) < 1.

Corollary 2.

For enough number of samples, iterations of DE algorithm with step sizes µ 0 = Θ( 1 n ) and µ g = Θ( 1 √ nng ) geometrically converges to the following with high probability: DISPLAYFORM3 which is a scaled variant of statistical error bound determined in Corollary 1.

In this Section we present detail proof for each theorem and proposition.

To avoid cluttering, during our proofs, we state some needed results as lemmas and provide their proof in the next Section B.

Proof.

Starting from the optimality inequality, for the lower bound with the set H we get: DISPLAYFORM0 2 is known as Restricted Eigenvalue (RE) condition.

The upper bound will factorize as: DISPLAYFORM1 Putting together inequalities FORMULA6 and FORMULA8 completes the proof.

A.2.

Proof of Proposition 1 Proposition 1.

Assume observations distributed as defined in Definition 1 and pair-wise SC conditions are satisfied.

Consider each superposition model (2) in isolation; to recover the common parameter β * 0 requires at least one group i to have n i = O(ω 2 (A 0 )).

To recover the rest of individual parameters, we need ∀g = i : DISPLAYFORM2 Proof.

Consider only one group for regression in isolation.

Note that y g = X g (β * g + β * 0 ) + ω g is a superposition model and as shown in BID16 ) the sample complexity required for the RE condition and subsequently recovering DISPLAYFORM3

Let's simplify the LHS of the RE condition: DISPLAYFORM0 where the first inequality is due to Lyapunov's inequality.

To avoid cluttering we denote δ 0g = δ 0 + δ g where δ 0 ∈ C 0 and δ g ∈ C g .

Now we add and subtract the corresponding per-group marginal tail function, DISPLAYFORM1 where ξ g > 0.

Let ξ g = δ 0g 2 ξ then the LHS of the RE condition reduces to: DISPLAYFORM2 For the ease of exposition we have written the LHS of (16) as the difference of two terms, i.e., t 1 (X) − t 2 (X) and in the followings we lower bound the first term t 1 and upper bound the second term t 2 .A.3.1.

LOWER BOUNDING THE FIRST TERM Our main result is the following lemma which uses the DERIC condition of the Definition 2 and provides a lower bound for the first term t 1 (X):Lemma 1.

Suppose DERIC holds.

Let ψ I = λminρ 3 .

For any δ ∈ H, we have: DISPLAYFORM3 which implies that t 1 (X) = inf δ∈H G g=1 n G n ξ g Q 2ξg (δ 0g ) satisfies the same RHS bound of (17).

Let's focus on the second term, i.e., t 2 (X).

First we want to show that the second term satisfies the bounded difference property defined in Section 3.2. of BID3 .

In other words, by changing each of x gi the value of t 2 (X) at most change by one.

First, we rewrite t 2 as follows: DISPLAYFORM0 where g (x 11 , . . .

, x jk , . . .

, DISPLAYFORM1 To avoid cluttering let's X = {x 11 , . . .

, x jk , . . .

, x Gn G }.

We want to show that t 2 has the bounded difference property, meaning: DISPLAYFORM2 for some constant c i .

Note that for bounded functions f, g : X → R, we have | sup X f − sup X g| ≤ sup X |f − g|.

Therefore: DISPLAYFORM3 Note that for δ ∈ H we have δ 0 2 + ng n δ g 2 ≤ 1 which results in δ 0 2 ≤ 1 and δ g 2 ≤ n ng .

Now, we can invoke the bounded difference inequality from Theorem 6.2 of BID3 which says that with probability at least 1 − e −τ 2 /2 we have: DISPLAYFORM4 Having this concentration bound, it is enough to bound the expectation of the second term.

Following lemma provides us with the bound on the expectation.

Lemma 2.

For the random vector x of Definition 1, we have the following bound: DISPLAYFORM5

Set n 0 = n. Putting back bounds of t 1 (X) and t 2 (X) together from Lemma 1 and 2, with probability at least 1 − e − τ 2 2 we have: DISPLAYFORM0 Note that all κ g s should be bounded away from zero.

To this end we need the follow sample complexities: DISPLAYFORM1 Taking ξ = α 6 we can simplify the sample complexities to the followings: DISPLAYFORM2 Finally, to conclude, we take τ = √ nκ min /2.

Proof.

From now on, to avoid cluttering the notation assume ω = ω 0 .

We massage the equation as follows: DISPLAYFORM0 , δg δg 2 n ng ω g 2 and a g = ng n δ g 2 .

Then the above term is the inner product of two vectors a = (a 0 , . . .

, a G ) and b = (b 0 , . . .

, b G ) for which we have: DISPLAYFORM1 Now we can go back to the original form: DISPLAYFORM2 , u g and e g (τ ) = DISPLAYFORM3 Then from (20), we have: DISPLAYFORM4 To simplify the notation, we drop arguments of h g for now.

From the union bound we have: DISPLAYFORM5 where σ = max g∈ [G] σ g and the last inequality is a result of the following lemma:Lemma 3.

For x gi and ω gi defined in Definition 1 and τ > 0, with probability at least 1 − DISPLAYFORM6 we have: DISPLAYFORM7 where σ g , η g , ζ g and g are group dependent constants.

Proof.

To analysis convergence properties of DICER, we should upper bound the error of each iteration.

Let's δ (t) = β (t) −β * be the error of iteration t of DICER, i.e., the distance from the true parameter (not the optimization minimum,β).

We show that δ (t) 2 decreases exponentially fast in t to the statistical error δ 2 = β − β * 2 .

We first start with the required definitions for our analysis.

Definition 3.

We define the following positive constants as functions of step sizes µ g > 0: DISPLAYFORM0 where DISPLAYFORM1 is the intersection of the error cone and the unit ball.

In the following theorem, we establish a deterministic bound on iteration errors δ (t) g 2 which depends on constants defined in Definition 3.Theorem 5.

For Algorithm 1 initialized by β(1) = 0, we have the following deterministic bound for the error at iteration t + 1: DISPLAYFORM2 DISPLAYFORM3 The RHS of FORMULA2 consists of two terms.

If we keep ρ < 1, the first term approaches zero fast, and the second term determines the bound.

In the following, we show that for specific choices of step sizes µ g s, the second term can be upper bounded using the analysis of Section 4.

More specifically, the first term corresponds to the optimization error which shrinks in every iteration while the second term is constant times the upper bound of the statistical error characterized in Corollary 1.

Therefore, if we keep ρ below one, the estimation error of DE algorithm geometrically converges to the approximate statistical error bound.

One way for having ρ < 1 is to keep all arguments of max(· · · ) defining ρ strictly below 1.

To this end, we first establish high probability upper bound for ρ g , η g , and φ g (in the Appendix A.6) and then show that with enough number of samples and proper step sizes µ g , ρ can be kept strictly below one with high probability.

In the following lemma we establish a recursive relation between errors of consecutive iterations which leads to a bound for the tth iteration.

Lemma 4.

We have the following recursive dependency between the error of t + 1th iteration and tth iteration of DE: DISPLAYFORM4 By recursively applying the result of Lemma 4, we get the following deterministic bound which depends on constants defined in Definition 3: DISPLAYFORM5 where DISPLAYFORM6 µg φ g .

We have: DISPLAYFORM7 A.6.

Proof of Theorem 4Proof.

First we need following two lemmas which are proved separately in the following sections.

Lemma 5.

Consider a g ≥ 1, with probability at least 1 − 6 exp −γ g (ω(A g ) + τ ) 2 the following upper bound holds: DISPLAYFORM8 Lemma 6.

Consider a g ≥ 1, with probability at least 1 − 4 exp −γ g (ω(A g ) + τ ) 2 the following upper bound holds: DISPLAYFORM9 Note that Lemma 3 readily provides a high probability upper bound for η g (1/(a g n g )) as DISPLAYFORM10 where DISPLAYFORM11 Remember the following two results to upper bound ρ g s and φ g s from Lemmas 5 and 6:

DISPLAYFORM12 First we want to keep ρ 0 + G g=1 ng n φ g of FORMULA2 strictly below 1.

DISPLAYFORM13 Remember that a g ≥ 1 was arbitrary.

So we pick it as a g = 2 n ng 1 + c 0g DISPLAYFORM14 (because we need a g ≥ 1) and the condition becomes: DISPLAYFORM15 We want to upper bound the RHS by 1/θ f which will determine the sample complexity for the shared component: DISPLAYFORM16 Note that any lower bound on the RHS of (27) will lead to the correct sample complexity for which the coefficient of δ DISPLAYFORM17 (determined in (26)) will be below one.

Since a 0 ≥ 1 we can ignore the first term by assuming max g∈[G]

\ b g ≤ 1 and the condition becomes: DISPLAYFORM18 which can be simplified to: DISPLAYFORM19 DISPLAYFORM20 Secondly, we want to bound all of ρ g + µ 0 n ng φg µg terms of (26) for µ g = 1 agng by 1: DISPLAYFORM21 The condition becomes: DISPLAYFORM22 Remember that we chose a g = 2b DISPLAYFORM23 .

We substitute the value of a g by keeping in mind the constraints for the b g and the condition reduces to: DISPLAYFORM24 Note that any positive lower bound of the d g will satisfy the condition in (31) and the result is a valid sample complexity.

In the following we show that d g > 1.We have a 0 ≥ 1 condition from (28), so we take DISPLAYFORM25 and look for a lower bound for d g : DISPLAYFORM26 (a g from (28)) = 2b DISPLAYFORM27 The term inside of the last bracket (33) is always positive and therefore a lower bound is one, i.e., d g ≥ 1.

From the condition (31) we get the following sample complexity: DISPLAYFORM28 Now we need to determine b g from previous conditions (28), knowing that a 0 = 4 max g∈ DISPLAYFORM29 .

We have 0 < b g ≤ 1 in (28) and we take the largest step by setting b g = 1.Here we summarize the setting under which we have the linear convergence: DISPLAYFORM30 Now we rewrite the same analysis using the tail bounds for the coefficients to clarify the probabilities.

To simplify the notation, let r g1 = and r g (τ ) = r g1 + ng n ag a0 r g2 , ∀g ∈ [G] \ , and r(τ ) = max g∈[G]

r g .

All of which are computed using a g s specified in (35).

Basically r is an instantiation of an upper bound of the ρ defined in (26) using a g s in (35).We are interested to upper bound the following probability: DISPLAYFORM31 where the first inequality comes from the deterministic bound of (25), We first focus on bounding the first term P (ρ ≥ r(τ )): DISPLAYFORM32 Now we focus on bounding the second term: DISPLAYFORM33 , g ∈ [G] and a g ≥ 1: DISPLAYFORM34 where we used the intermediate form of Lemma 3 for τ > 0.

Putting all of the bounds (37), (38), and (39) back into the (36): DISPLAYFORM35 where υ = max(28, σ) and γ = min g∈ [G] γ g and τ = t + max( , γ −1/2 ) log(G + 1) where = k max g∈[G] η g .

Note that τ = t + C log(G + 1) increases the sample complexities to the followings: DISPLAYFORM36 and it also affects step sizes as follows: DISPLAYFORM37

Here, we present proofs of each lemma used during the proofs of theorems in Section A.

Proof.

LHS of FORMULA10 is the weighted summation of ξ g Q 2ξg (δ 0g ) = δ 0g 2 ξP(| x, , δ 0g / δ 0g 2 | > 2ξ) = δ 0g 2 ξQ 2ξ (u) where ξ > 0 and u = δ 0g / δ 0g 2 is a unit length vector.

So we can rewrite the LHS of (17) as: DISPLAYFORM0 With this observation, the lower bound of the Lemma 1 is a direct consequence of the following two results:Lemma 7.

Let u be any unit length vector and suppose x obeys Definiton 1.

Then for any u, we have DISPLAYFORM1 Lemma 8.

Suppose Definition 2 holds.

Then, we have: DISPLAYFORM2 Proof.

Consider the following soft indicator function which we use in our derivation: DISPLAYFORM3 0, |s| ≤ a (|s| − a)/a, a ≤ |s| ≤ 2a 1, 2a < |s|Now: DISPLAYFORM4 Eψ ξg ( x, δ 0g ) − ψ ξg ( x gi , δ 0g )≤ 2E sup DISPLAYFORM5 where gi are iid copies of Rademacher random variable which are independent of every other random variables and themselves.

Now we add back 1 n and expand δ 0g = δ 0 + δ g : DISPLAYFORM6 1 √ n g gi x gi , δ g (n 0 := n, 0i := 0 , x 0i := x i ) = 2 √ n E sup DISPLAYFORM7 Note that the h gi is a sub-Gaussian random vector which let us bound the E sup using the Gaussian width (Tropp, 2015) in the last step.

Proof.

To avoid cluttering let h g (ω g , X g ) = n ng ω g 2 sup ug∈Ag X T g ωg ωg 2, u g , e g = ζ g kω(A g ) + g √ log G + τ , where s g = n ng(2K 2 + 1)n g .P (h g (ω g , X g ) > e g s g ) = P h g (ω g , X g ) > e g s g n n g ω g 2 > s g P n n g ω g 2 > s g+ P h g (ω g , X g ) > e g s g n n g ω g 2 < s g P n n g ω g 2 < s g ≤ P n n g ω g 2 > s g + P h g (ω g , X g ) > e g s g n n g ω g 2 < s g ≤ P ω g 2 > (2K 2 + 1)n g + P sup DISPLAYFORM8 , u g > e g ≤ P ω g 2 > (2K 2 + 1)n g + sup DISPLAYFORM9 Let's focus on the first term.

Since ω g consists of i.i.d.

centered unit-variance sub-Gaussian elements with |||ω gi ||| ψ2 < K, ω 2 gi is sub-exponential with |||ω gi ||| ψ1 < 2K 2 .

Let's apply the Bernstein's inequality to ω g DISPLAYFORM10 We also know that E ω g 2 2 ≤ n g ) which gives us: DISPLAYFORM11 Finally, we set τ = 2K 2 n g : P ω g 2 > (2K 2 + 1)n g ≤ 2 exp (−ν g n g ) = 2 (G + 1) exp (−ν g n g + log(G + 1))Now we upper bound the second term of (42).

Given any fixed v ∈ S p−1 , X g v is a sub-Gaussian random vector with X T g v ψ2 ≤ C g k .

From Theorem 9 of for any v ∈ S p−1 we have: DISPLAYFORM12 where φ g = sup ug∈Ag u g 2 and in our problem φ g = 1.

We now substitute t = τ + g log(G + 1) where g = θ g C g k. DISPLAYFORM13

<|TLDR|>

@highlight

We provide an estimator and an estimation algorithm for a class of multi-task regression problem and provide statistical and computational analysis..