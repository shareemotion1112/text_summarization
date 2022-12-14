We consider the dictionary learning problem, where the aim is to model the given data as a linear combination of a few columns of a matrix known as a dictionary, where the sparse weights forming the linear combination are known as coefficients.

Since the dictionary and coefficients, parameterizing the linear model are unknown, the corresponding optimization is inherently non-convex.

This was a major challenge until recently, when provable algorithms for dictionary learning were proposed.

Yet, these provide guarantees only on the recovery of the dictionary, without explicit recovery guarantees on the coefficients.

Moreover, any estimation error in the dictionary adversely impacts the ability to successfully localize and estimate the coefficients.

This potentially limits the utility of existing provable dictionary learning methods in applications where coefficient recovery is of interest.

To this end, we develop NOODL: a simple Neurally plausible alternating Optimization-based Online Dictionary Learning algorithm, which recovers both the dictionary and coefficients exactly at a geometric rate, when initialized appropriately.

Our algorithm, NOODL, is also scalable and amenable for large scale distributed implementations in neural architectures, by which we mean that it only involves simple linear and non-linear operations.

Finally, we corroborate these theoretical results via experimental evaluation of the proposed algorithm with the current state-of-the-art techniques.

Sparse models avoid overfitting by favoring simple yet highly expressive representations.

Since signals of interest may not be inherently sparse, expressing them as a sparse linear combination of a few columns of a dictionary is used to exploit the sparsity properties.

Of specific interest are overcomplete dictionaries, since they provide a flexible way of capturing the richness of a dataset, while yielding sparse representations that are robust to noise; see BID13 ; Chen et al. (1998); Donoho et al. (2006) .

In practice however, these dictionaries may not be known, warranting a need to learn such representations -known as dictionary learning (DL) or sparse coding BID14 .

Formally, this entails learning an a priori unknown dictionary A ??? R n??m and sparse coefficients x * (j) ??? R m from data samples y (j) ??? R n generated as DISPLAYFORM0 This particular model can also be viewed as an extension of the low-rank model BID15 .

Here, instead of sharing a low-dimensional structure, each data vector can now reside in a separate low-dimensional subspace.

Therefore, together the data matrix admits a union-of-subspace model.

As a result of this additional flexibility, DL finds applications in a wide range of signal processing and machine learning tasks, such as denoising (Elad and Aharon, 2006) , image inpainting BID12 , clustering and classification (Ramirez et al., 2010; BID16 BID17 BID18 2019b; a) , and analysis of deep learning primitives (Ranzato et al., 2008; BID0 ; see also Elad (2010) , and references therein.

Notwithstanding the non-convexity of the associated optimization problems (since both factors are unknown), alternating minimization-based dictionary learning techniques have enjoyed significant success in practice.

Popular heuristics include regularized least squares-based BID14 BID8 BID12 BID9 BID7 , and greedy approaches such as the method of optimal directions (MOD) (Engan et al., 1999) and k-SVD (Aharon et al., 2006) .

However, dictionary learning, and matrix factorization models in general, are difficult to analyze in theory; see also BID10 .To this end, motivated from a string of recent theoretical works BID1 BID4 Geng and Wright, 2014) , provable algorithms for DL have been proposed recently to explain the success of aforementioned alternating minimization-based algorithms (Agarwal et al., 2014; Arora et al., 2014; BID20 .

However, these works exclusively focus on guarantees for dictionary recovery.

On the other hand, for applications of DL in tasks such as classification and clusteringwhich rely on coefficient recovery -it is crucial to have guarantees on coefficients recovery as well.

Contrary to conventional prescription, a sparse approximation step after recovery of the dictionary does not help; since any error in the dictionary -which leads to an error-in-variables (EIV) (Fuller, 2009 ) model for the dictionary -degrades our ability to even recover the support of the coefficients (Wainwright, 2009) .

Further, when this error is non-negligible, the existing results guarantee recovery of the sparse coefficients only in 2 -norm sense (Donoho et al., 2006) .

As a result, there is a need for scalable dictionary learning techniques with guaranteed recovery of both factors.

In this work, we present a simple online DL algorithm motivated from the following regularized least squares-based problem, where S(??) is a nonlinear function that promotes sparsity.

S(x (j) ).Although our algorithm does not optimize this objective, it leverages the fact that the problem (P1) is convex w.r.t A, given the sparse coefficients {x (j) }.

Following this, we recover the dictionary by choosing an appropriate gradient descent-based strategy (Arora et al., 2015; Engan et al., 1999) .

To recover the coefficients, we develop an iterative hard thresholding (IHT)-based update step BID3 Blumensath and Davies, 2009) , and show that -given an appropriate initial estimate of the dictionary and a mini-batch of p data samples at each iteration t of the online algorithmalternating between this IHT-based update for coefficients, and a gradient descent-based step for the dictionary leads to geometric convergence to the true factors, i.e., x (j) ???x * (j) and A

i ???A * i as t??????. In addition to achieving exact recovery of both factors, our algorithm -Neurally plausible alternating Optimization-based Online Dictionary Learning (NOODL) -has linear convergence properties.

Furthermore, it is scalable, and involves simple operations, making it an attractive choice for practical DL applications.

Our major contributions are summarized as follows:??? Provable coefficient recovery: To the best of our knowledge, this is the first result on exact recovery of the sparse coefficients {x * (j) }, including their support recovery, for the DL problem.

The proposed IHT-based strategy to update coefficient under the EIV model, is of independent interest for recovery of the sparse coefficients via IHT, which is challenging even when the dictionary is known; see also Yuan et al. (2016) and BID11 .???

Unbiased estimation of factors and linear convergence: The recovery guarantees on the coefficients also helps us to get rid of the bias incurred by the prior-art in dictionary estimation.

Furthermore, our technique geometrically converges to the true factors.??? Online nature and neural implementation: The online nature of algorithm, makes it suitable for machine learning applications with streaming data.

In addition, the separability of the coefficient update allows for distributed implementations in neural architectures (only involves simple linear and non-linear operations) to solve large-scale problems.

To showcase this, we also present a prototype neural implementation of NOODL.In addition, we also verify these theoretical properties of NOODL through experimental evaluations on synthetic data, and compare its performance with state-of-the-art provable DL techniques.

With the success of the alternating minimization-based techniques in practice, a push to study the DL problem began when BID1 showed that for m = n, the solution pair (A * , X * ) lies at a local minima of the following non-convex optimization program, where X = [x (1) , x (2) , . . .

, x (p) ] and Y = [y (1) , y (2) , . . .

, y (p) ], with high probability over the randomness of the coefficients, min O * ??? n ?? log(n) DISPLAYFORM0 The algorithms discussed above implicitly assume that the coefficients can be recovered, after dictionary recovery, via some sparse approximation technique.

However, as alluded to earlier, the guarantees for coefficient recovery -when the dictionary is known approximately -may be limited to some 2 norm bounds (Donoho et al., 2006) .

This means that, the resulting coefficient estimates may not even be sparse.

Therefore, for practical applications, there is a need for efficient online algorithms with guarantees, which serves as the primary motivation for our work.

We now detail the specifics of our algorithm -NOODL, outlined in Algorithm 1.

NOODL recovers both the dictionary and the coefficients exactly given an appropriate initial estimate A (0) of the dictionary.

Specifically, it requires A (0) to be ( 0 , 2)-close to A * for 0 = O * (1/ log(n)), where ( , ??)-closeness is defined as follows.

This implies that, the initial dictionary estimate needs to be column-wise, and in spectral norm sense, close to A * , which can be achieved via certain initialization algorithms, such as those presented in Arora et al. (2015) .

Given an integer n, we denote [n] = {1, 2, . . .

, n}. The bold upper-case and lower-case letters are used to denote matrices M and vectors v, respectively.

Mi, M (i,:) , Mij, and vi (and v(i) ) denote the i-th column, i-th row, (i, j) element of a matrix, and i-th element of a vector, respectively.

The superscript (??) (n) denotes the n-th iterate, while the subscript (??) (n) is reserved for the n-th data sample.

Given a matrix M, we use M and M F as the spectral norm and Frobenius norm.

Given a vector v, we use v , v 0, and v 1 to denote the 2 norm, 0 (number of non-zero entries), and 1 norm, respectively.

We also use standard notations O(??), ???(??) ( O(??), ???(??)) to indicate the asymptotic behavior (ignoring logarithmic factors).

Further, we use g(n) = O * (f (n)) to indicate that g(n) ??? Lf (n) for a small enough constant L, which is independent of n. We use c(??) for constants parameterized by the quantities in (??).

T?? (z) := z ?? 1 |z|????? denotes the hardthresholding operator, where "1" is the indicator function.

We use supp(??) for the support (the set of non-zero elements) and sign(??) for the element-wise sign.

DISPLAYFORM0 Form empirical gradient estimate: DISPLAYFORM1 Take a gradient descent step: DISPLAYFORM2 Normalize: DISPLAYFORM3 Due to the streaming nature of the incoming data, NOODL takes a mini-batch of p data samples at the t-th iteration of the algorithm, as shown in Algorithm 1.

It then proceeds by alternating between two update stages: coefficient estimation ("Predict") and dictionary update ("Learn") as follows.

Predict Stage:

For a general data sample y = A * x * , the algorithm begins by forming an initial coefficient estimate x (0) based on a hard thresholding (HT) step as shown in (3), where T ?? (z) := z ?? 1 |z|????? for a vector z. Given this initial estimate x (0) , the algorithm iterates over R = ???(log(1/?? R ))IHT-based steps (4) to achieve a target tolerance of ?? R , such that DISPLAYFORM4 x is the learning rate, and ?? (r) is the threshold at the r-th iterate of the IHT.

In practice, these can be fixed to some constants for all iterations; see A.6 for details.

Finally at the end of this stage, we have estimate DISPLAYFORM5 Learn Stage: Using this estimate of the coefficients, we update the dictionary at t-th iteration A (t) by an approximate gradient descent step (6), using the empirical gradient estimate (5) and the learning rate ?? A = ??(m/k); see also A.5.

Finally, we normalize the columns of the dictionary and continue to the next batch.

The running time of each step t of NOODL is therefore O(mnp log(1/?? R )).For a target tolerance of T and ?? T , such that A DISPLAYFORM6 NOODL uses an initial HT step and an approximate gradient descent-based strategy as in Arora et al. (2015) .

Following which, our IHT-based coefficient update step yields an estimate of the coefficients at each iteration of the online algorithm.

Coupled with the guaranteed progress made on the dictionary, this also removes the bias in dictionary estimation.

Further, the simultaneous recovery of both factors also avoids an often expensive post-processing step for recovery of the coefficients.

We start by introducing a few important definitions.

First, as discussed in the previous section we require that the initial estimate A (0) of the dictionary is ( 0 , 2)-close to A * .

In fact, we require this closeness property to hold at each subsequent iteration t, which is a key ingredient in our analysis.

This initialization achieves two goals.

First, the ??(i)A ??(i) ??? A Definition 3.

A matrix A ??? R n??m with unit-norm columns is ??-incoherent if for all i = j the inner-product between the columns of the matrix follow DISPLAYFORM0 The incoherence parameter measures the degree of closeness of the dictionary elements.

Smaller values (i.e., close to 0) of ?? are preferred, since they indicate that the dictionary elements do not resemble each other.

This helps us to effectively tell dictionary elements apart (Donoho and Huo, 2001; Cand??s and Romberg, 2007) .

We assume that ?? = O(log(n)) (Donoho and Huo, 2001).

Next, we assume that the coefficients are drawn from a distribution class D defined as follows.

Definition 4 (Distribution class D).

The coefficient vector x * belongs to an unknown distribution D, where the support S = supp(x * ) is at most of size k, DISPLAYFORM1 i |i ??? S] = 1, and when i ??? S, |x * i | ??? C for some constant C ??? 1.

In addition, the non-zero entries are sub-Gaussian and pairwise independent conditioned on the support.

The randomness of the coefficient is necessary for our finite sample analysis of the convergence.

Here, there are two sources of randomness.

The first is the randomness of the support, where the non-zero elements are assumed to pair-wise independent.

The second is the value an element in the support takes, which is assumed to be zero mean with variance one, and bounded in magnitude.

Similar conditions are also required for support recovery of sparse coefficients, even when the dictionary is known (Wainwright, 2009; Yuan et al., 2016) .

Note that, although we only consider the case |x * i | ??? C for ease of discussion, analogous results may hold more generally for x * i s drawn from a distribution with sufficiently (exponentially) small probability of taking values in [???C, C].Recall that, given the coefficients, we recover the dictionary by making progress on the least squares objective (P1) (ignoring the term penalizing S(??)).

Note that, our algorithm is based on finding an appropriate direction to ensure descent based on the geometry of the objective.

To this end, we adopt a gradient descent-based strategy for dictionary update.

However, since the coefficients are not exactly known, this results in an approximate gradient descent-based approach, where the empirical gradient estimate is formed as (5).

In our analysis, we establish the conditions under which both the empirical gradient vector (corresponding to each dictionary element) and the gradient matrix concentrate around their means.

To ensure progress at each iterate t, we show that the expected gradient vector is (???(k/m), ???(m/k), 0)-correlated with the descent direction, defined as follows.

DISPLAYFORM2 This can be viewed as a local descent condition which leads to the true dictionary columns; see also Cand??s et al. (2015) , Chen and Wainwright (2015) and Arora et al. (2015) .

In convex optimization literature, this condition is implied by the 2?? ??? -strong convexity, and 1/2?? + -smoothness of the objective.

We show that for NOODL, ?? t = 0, which facilitates linear convergence to A * without incurring any bias.

Overall our specific model assumptions for the analysis can be formalized as: DISPLAYFORM3 The coefficients are drawn from the distribution class D, as per Def.

4; A.3 The sparsity k satisfies k = O * ( ??? n/?? log(n)); A.4 A (0) is ( 0 , 2)-close to A * as per Def.

1, and 0 = O * (1/ log(n)); A.5 The step-size for dictionary update satisfies ?? A = ??(m/k);A.6 The step-size and threshold for coefficient estimation satisfies ?? (r) x < c 1 ( t , ??, n, k) = ???(k/ ??? n) < 1 and ?? (r) = c 2 ( t , ??, k, n) = ???(k 2 /n) for small constants c 1 and c 2 .We are now ready to state our main result.

A summary of the notation followed by a details of the analysis is provided in Appendix A and Appendix B, respectively.

Theorem 1 (Main Result).

Suppose that assumptions A.1-A.6 hold, and Algorithm 1 is provided with p = ???(mk 2 ) new samples generated according to model (1) at each iteration t. Then, with DISPLAYFORM4 alg , given R = ???(log(n)), the coefficient estimate x (t) i at t-th iteration has the correct signed-support and satisfies ( x DISPLAYFORM5 Furthermore, for some 0 < ?? < 1/2, the estimate A (t) at (t)-th iteration satisfies DISPLAYFORM6 2 , for all t = 1, 2, . . .

..

Our main result establishes that when the model satisfies A.1???A.3, the errors corresponding to the dictionary and coefficients geometrically decrease to the true model parameters, given appropriate dictionary initialization and learning parameters (step sizes and threshold); see A.4???A.6.

In other words, to attain a target tolerance of T and ?? T , where DISPLAYFORM7 R is the target decay tolerance for the IHT steps.

An appropriate number of IHT steps, R, remove the dependence of final coefficient error (per outer iteration) on the initial x (0) .

In Arora et al. (2015) , this dependence in fact results in an irreducible error, which is the source of bias in dictionary estimation.

As a result, since (for NOODL) the error in the coefficients only depends on the error in the dictionary, it can be made arbitrarily small, at a geometric rate, by the choice of T , ?? T , and ?? R .

Also, note that, NOODL can tolerate i.i.d.

noise, as long as the noise variance is controlled to enable the concentration results to hold; we consider the noiseless case here for ease of discussion, which is already highly involved.

Intuitively, Theorem 1 highlights the symbiotic relationship between the two factors.

It shows that, to make progress on one, it is imperative to make progress on the other.

The primary condition that allows us to make progress on both factors is the signed-support recovery (Def.

2).

However, the introduction of IHT step adds complexity in the analysis of both the dictionary and coefficients.

To analyze the coefficients, in addition to deriving conditions on the parameters to preserve the correct signed-support, we analyze the recursive IHT update step, and decompose the noise term into a component that depends on the error in the dictionary, and the other that depends on the initial coefficient estimate.

For the dictionary update, we analyze the interactions between elements of the coefficient vector (introduces by the IHT-based update step) and show that the gradient vector for the dictionary update is (???(k/m), ???(m/k), 0)-correlated with the descent direction.

In the end, this leads to exact recovery of the coefficients and removal of bias in the dictionary estimation.

Note that our analysis pipeline is standard for the convergence analysis for iterative algorithms.

However, the introduction of the IHT-based strategy for coefficient update makes the analysis highly involved as compared to existing results, e.g., the simple HT-based coefficient estimate in Arora et al. (2015) .NOODL has an overall running time of O(mnp log(1/?? R ) max(log(1/ T ), log( ??? k/?? T )) to achieve target tolerances T and ?? T , with a total sample complexity of p??T = ???(mk 2 ).

Thus to remove bias, the IHT-based coefficient update introduces a factor of log(1/?? R ) in the computational complexity as compared to Arora et al. (2015) (has a total sample complexity of p ?? T = ???(mk)), and also does not have the exponential running time and sample complexity as Barak et al. (2015) ; see TAB0 .

The neural plausibility of our algorithm implies that it can be implemented as a neural network.

This is because, NOODL employs simple linear and non-linear operations (such as inner-product and hard-thresholding) and the coefficient updates are separable across data samples, as shown in (4) of Algorithm 1.

To this end, we present a neural implementation of our algorithm in FIG2 , which showcases the applicability of NOODL in large-scale distributed learning tasks, motivated from the implementations described in BID14 and (Arora et al., 2015) .The neural architecture shown in FIG2 has three layers -input layer, weighted residual evaluation layer, and the output layer.

The input to the network is a data and step-size pair (y (j) , ?? x ) to each input node.

Given an input, the second layer evaluates the weighted residuals as shown in FIG2 .

Finally, the output layer neurons evaluate the IHT iterates x (r+1) (j) (4).

We illustrate the operation of this architecture using the timing diagram in FIG2 .

The main stages of operation are as follows.

Output: DISPLAYFORM0 The timing sequence of the neural implementation.

Initial Hard Thresholding Phase: The coefficients initialized to zero, and an input (y (j) , 1) is provided to the input layer at a time instant = 0, which communicates these to the second layer.

Therefore, the residual at the output of the weighted residual evaluation layer evaluates to y (j) at = 1.

Next, at = 2, this residual is communicated to the output layer, which results in evaluation of the initialization x (0) (j) as per (3).

This iterate is communicated to the second layer for the next residual evaluation.

Also, at this time, the input layer is injected with (y (j) , ?? x ) to set the step size parameter ?? x for the IHT phase, as shown in FIG2

Iterative Hard Thresholding (IHT) Phase: Beginning = 3, the timing sequence enters the IHT phase.

Here, the output layer neurons communicate the iterates x (r+1) (j) to the second layer for evaluation of subsequent iterates as shown in FIG2 .

The process then continues till the time instance = 2R + 1, for R = ???(log(1/?? R )) to generate the final coefficient estimate x DISPLAYFORM0 for the current batch of data.

At this time, the input layer is again injected with (y (j) , 1) to prepare the network for residual sharing and gradient evaluation for dictionary update.

The procedure now enters the dictionary update phase, denoted as "Hebbian Learning" in the timing sequence.

In this phase, each output layer neuron communicates the final coefficient estimate x (t) (j) = x (R) (j) to the second layer, which evaluates the residual for one last time (with ?? x = 1), and shares it across all second layer neurons ("Hebbian learning").

This allows each second layer neuron to evaluate the empirical gradient estimate (5), which is used to update the current dictionary estimate (stored as weights) via an approximate gradient descent step.

This completes one outer iteration of Algorithm 1, and the process continues for T iterations to achieve target tolerances T and ?? T , with each step receiving a new mini-batch of data.

We now analyze the convergence properties and sample complexity of NOODL via experimental evaluations 2 .

The experimental data generation set-up, additional results, including analysis of computational time, are shown in Appendix E.

We compare the performance of our algorithm NOODL with the current state-of-the-art alternating optimization-based online algorithms presented in Arora et al. (2015) , and the popular algorithm presented in BID12 (denoted as Mairal '09) .

First of these, Arora15(''biased''), is a simple neurally plausible method which incurs a bias and has a sample complexity of ???(mk).

The other, referred to as Arora15(''unbiased''), incurs no bias as per Arora et al. (2015) , but the sample complexity results were not established.

FIG4 , (b-i), (c-i), and (d-i) show the performance of the aforementioned methods for k = 10, 20, 50, and 100, respectively.

Here, for all experiments we set ?? x = 0.2 and ?? = 0.1.

We terminate NOODL when the error in dictionary is less than 10 ???10 .

Also, for coefficient update, we terminate when change in the iterates is below 10 ???12 .

For k = 10, 20 and k = 50, FIG4 : Comparative analysis of convergence properties.

Panels (a-i), (b-i), (c-i), and (d-i) show the convergence of NOODL, Arora15(''biased''), Arora15(''unbiased'') and Mairal '09, for different sparsity levels for n = 1000, m = 1500 and p = 5000.

Since NOODL also recovers the coefficients, we show the corresponding recovery of the dictionary, coefficients, and overall fit in panels (a-ii), (b-ii), (c-ii), and (d-ii), respectively.

Further, panels (e-i) and (e-ii) show the phase transition in samples p (per iteration) with the size of the dictionary m averaged across 10 Monte Carlo simulations for the two factors.

Here, n = 100, k = 3, ??x = 0.2, ?? = 0.1, 0 = 2/ log(n), ??A is chosen as per A.5.

A trial is considered successful if the relative Frobenius error incurred by A and X is below 5 ?? 10 ???7 after 50 iterations.

we note that Arora15(''biased'') and Arora15(''unbiased'') incur significant bias, while NOODL converges to A * linearly.

NOODL also converges for significantly higher choices of sparsity k, i.e., for k = 100 as shown in panel (d), beyond k = O( ??? n), indicating a potential for improving this bound.

Further, we observe that Mairal '09 exhibits significantly slow convergence as compared to NOODL.

Also, in panels (a-ii), (b-ii), (c-ii) and (d-ii) we show the corresponding performance of NOODL in terms of the error in the overall fit ( Y ??? AX F / Y F ), and the error in the coefficients and the dictionary, in terms of relative Frobenius error metric discussed above.

We observe that the error in dictionary and coefficients drops linearly as indicated by our main result.

We consider the online DL setting in this work.

We note that, empirically NOODL works for the batch setting also.

However, analysis for this case will require more sophisticated concentration results, which can address the resulting dependence between iterations of the algorithm.

In addition, our experiments indicate that NOODL works beyond the sparsity ranges prescribed by our theoretical results.

Arguably, the bounds on sparsity can potentially be improved by moving away from the incoherence-based analysis.

We also note that in our experiments, NOODL converges even when initialized outside the prescribed initialization region, albeit it achieves the linear rate once it satisfies the closeness condition A.4.

These potential directions may significantly impact the analysis and development of provable algorithms for other factorization problems as well.

We leave these research directions, and a precise analysis under the noisy setting, for future explorations.

We present NOODL, to the best of our knowledge, the first neurally plausible provable online algorithm for exact recovery of both factors of the dictionary learning (DL) model.

NOODL alternates between: (a) an iterative hard thresholding (IHT)-based step for coefficient recovery, and (b) a gradient descent-based update for the dictionary, resulting in a simple and scalable algorithm, suitable for large-scale distributed implementations.

We show that once initialized appropriately, the sequence of estimates produced by NOODL converge linearly to the true dictionary and coefficients without incurring any bias in the estimation.

Complementary to our theoretical and numerical results, we also design an implementation of NOODL in a neural architecture for use in practical applications.

In essence, the analysis of this inherently non-convex problem impacts other matrix and tensor factorization tasks arising in signal processing, collaborative filtering, and machine learning.

We summarizes the definitions of some frequently used symbols in our analysis in TAB1 .

In addition, we use D (v) as a diagonal matrix with elements of a vector v on the diagonal.

Given a matrix M, we use M ???i to denote a resulting matrix without i-th column.

Also note that, since we show that A (t)i ??? A * i ??? t contracts in every step, therefore we fix t , 0 = O * (1/ log(n)) in our analysis.

i-th column of the dictionary estimate at the t-th iterate.

DISPLAYFORM0 Upper-bound on column-wise error at the t-th iterate.

DISPLAYFORM1 Incoherence between the columns of A (t) ; See Claim 1.

DISPLAYFORM2 Inner-product between the error and the dictionary element.

DISPLAYFORM3 j on the diagonal for j ??? S.

i-th element the coefficient estimate at the r-th IHT iterate.

DISPLAYFORM0 Decay parameter for coefficients.

DISPLAYFORM1 Error in non-zero elements of the coefficient vector.

DISPLAYFORM2 j with probability at least (1 ??? ?? DISPLAYFORM3 1 F x * is the indicator function corresponding to the event that sign(x * ) = sign( x), denoted by F x * , and similarly for the complement F x * B PROOF OF THEOREM 1We now prove our main result.

The detailed proofs of intermediate lemmas and claims are organized in Appendix C and Appendix D, respectively.

Furthermore, the standard concentration results are stated in Appendix F for completeness.

Also, see TAB2 for a map of dependence between the results.

Given an ( 0 , 2)-close estimate of the dictionary, the main property that allows us to make progress on the dictionary is the recovery of the correct sign and support of the coefficients.

Therefore, we first show that the initial coefficient estimate (3) recovers the correct signed-support in Step I.A. Now, the IHT-based coefficient update step also needs to preserve the correct signed-support.

This is to ensure that the approximate gradient descent-based update for the dictionary makes progress.

Therefore, in Step I.B, we derive the conditions under which the signed-support recovery condition is preserved by the IHT update.

To get a handle on the coefficients, in Step II.A, we derive an upper-bound on the error incurred by each non-zero element of the estimated coefficient vector, i.e., | x i ??? x * i | for i ??? S for a general coefficient vector x * , and show that this error only depends on t (the column-wise error in the dictionary) given enough IHT iterations R as per the chosen decay parameter ?? R .

In addition, for analysis of the dictionary update, we develop an expression for the estimated coefficient vector inStep II.B.We then use the coefficient estimate to show that the gradient vector satisfies the local descent condition (Def.

5).

This ensures that the gradient makes progress after taking the gradient descent-based step (6).

To begin, we first develop an expression for the expected gradient vector (corresponding to each dictionary element) in Step III.A. Here, we use the closeness property Def 1 of the dictionary estimate.

Further, since we use an empirical estimate, we show that the empirical gradient vector concentrates around its mean in Step III.B. Now using Lemma 15, we have that descent along this direction makes progress.

Step IV.A and Step IV.B, we show that the updated dictionary estimate maintains the closeness property Def 1.

This sets the stage for the next dictionary update iteration.

As a result, our main result establishes the conditions under which any t-th iteration succeeds.

Our main result is as follows.

Theorem 1 (Main Result) Suppose that assumptions A.1-A.6 hold, and Algorithm 1 is provided with p = ???(mk 2 ) new samples generated according to model (1) at each iteration t. Then, with DISPLAYFORM0 i at t-th iteration has the correct signed-support and satisfies DISPLAYFORM1 Furthermore, for some 0 < ?? < 1/2, the estimate A (t) at (t)-th iteration satisfies DISPLAYFORM2 alg is some small constant, where ?? DISPLAYFORM3

As a first step, we ensure that our coefficient estimate has the correct signed-support (Def.

2).

To this end, we first show that the initialization has the correct signed-support, and then show that the iterative hard-thresholding (IHT)-based update step preserves the correct signed-support for a suitable choice of parameters.??? Step I.A: Showing that the initial coefficient estimate has the correct signed-supportGiven an ( 0 , 2)-close estimate A (0) of A * , we first show that for a general sample y the initialization step (3) recovers the correct signed-support with probability at least (1????? DISPLAYFORM0 ).

This is encapsulated by the following lemma.

Lemma 1 (Signed-support recovery by coefficient initialization step).

Suppose A (t) DISPLAYFORM1 , and t = O * (1/ log(m)), with probability at least (1 ??? ?? (t)T ) for each random sample y = A * x * : DISPLAYFORM2 ).Note that this result only requires the dictionary to be column-wise close to the true dictionary, and works for less stringent conditions on the initial dictionary estimate, i.e., requires DISPLAYFORM3 ; see also (Arora et al., 2015) .???

Step I.B: The iterative IHT-type updates preserve the correct signed support-Next, we show that the IHT-type coefficient update step (4) preserves the correct signed-support for an appropriate choice of step-size parameter ?? (r)x and threshold ?? (r) .

The choice of these parameters arises from the analysis of the IHT-based update step.

Specifically, we show that at each iterate r, the step-size ?? (r)x should be chosen to ensure that the component corresponding to the true coefficient value is greater than the "interference" introduced by other non-zero coefficient elements.

Then, if the threshold is chosen to reject this "noise", each iteration of the IHT-based update step preserves the correct signed-support.

Lemma 2 (IHT update step preserves the correct signed-support).

Suppose DISPLAYFORM4 T ), each iterate of the IHT-based coefficient update step shown in (4) has the correct signed-support, if for a constant c DISPLAYFORM5 the step size is chosen as ?? DISPLAYFORM6 1 , and the threshold ?? (r) is chosen as DISPLAYFORM7 for some constants c 1 and c 2 .

Here, DISPLAYFORM8 ) ,and ?? DISPLAYFORM9 ).Note that, although we have a dependence on the iterate r in choice of ?? (r)x and ?? (r) , these can be set to some constants independent of r. In practice, this dependence allows for greater flexibility in the choice of these parameters.

We now derive an upper-bound on the error incurred by each non-zero coefficient element.

Further, we derive an expression for the coefficient estimate at the t-th round of the online algorithm x (t) := x (R) ; we use x instead of x (t) for simplicity.??? Step II.A: Derive a bound on the error incurred by the coefficient estimate-Since Lemma 2 ensures that x has the correct signed-support, we now focus on the error incurred by each coefficient element on the support by analyzing x. To this end, we carefully analyze the effect of the recursive update (4), to decompose the error incurred by each element on the support into two components -one that depends on the initial coefficient estimate xand other that depends on the error in the dictionary.

We show that the effect of the component that depends on the initial coefficient estimate diminishes by a factor of (1 ??? ?? x + ?? x ??t ??? n ) at each iteration r. Therefore, for a decay parameter ?? R , we can choose the number of IHT iterations R, to make this component arbitrarily small.

Therefore, the error in the coefficients only depends on the per column error in the dictionary, formalized by the following result.

Lemma 3 (Upper-bound on the error in coefficient estimation).

With probability at least (1 ??? ?? DISPLAYFORM0 T ) the error incurred by each element i 1 ??? supp(x * ) of the coefficient estimate is upper-bounded as DISPLAYFORM1 ), and ?? t is the incoherence between the columns of A (t) ; see Claim 1.This result allows us to show that if the column-wise error in the dictionary decreases at each iteration t, then the corresponding estimates of the coefficients also improve.??? Step II.B: Developing an expression for the coefficient estimate-Next, we derive the expression for the coefficient estimate in the following lemma.

This expression is used to analyze the dictionary update.

Lemma 4 (Expression for the coefficient estimate at the end of R-th IHT iteration).

With probability at least (1 ??? ?? DISPLAYFORM2 ?? ) the i 1 -th element of the coefficient estimate, for each i 1 ??? supp(x * ), is given by DISPLAYFORM3 Here, ?? DISPLAYFORM4 ) and ?? DISPLAYFORM5 ).We again observe that the error in the coefficient estimate depends on the error in the dictionary via ??

Given the coefficient estimate we now show that the choice of the gradient as shown in (5) makes progress at each step.

To this end, we analyze the gradient vector corresponding to each dictionary element to see if it satisfies the local descent condition of Def.

5.

Our analysis of the gradient is motivated from Arora et al. (2015) .

However, as opposed to the simple HT-based coefficient update step used by Arora et al. (2015) , our IHT-based coefficient estimate adds to significant overhead in terms of analysis.

Notwithstanding the complexity of the analysis, we show that this allows us to remove the bias in the gradient estimate.

To this end, we first develop an expression for each expected gradient vector, show that the empirical gradient estimate concentrates around its mean, and finally show that the empirical gradient vector is (???(k/m), ???(m/k), 0)-correlated with the descent direction, i.e. has no bias.??? Step III.A: Develop an expression for the expected gradient vector corresponding to each dictionary element-The expression for the expected gradient vector g DISPLAYFORM0 j corresponding to j-th dictionary element is given by the following lemma.

Lemma 5 (Expression for the expected gradient vector).

Suppose that A (t) is ( t , 2)-near to A * .

Then, the dictionary update step in Algorithm 1 amounts to the following for the j-th dictionary element DISPLAYFORM1 j is given by g DISPLAYFORM2 ??? Step III.B: Show that the empirical gradient vector concentrates around its expectation-Since we only have access to the empirical gradient vectors, we show that these concentrate around their expected value via the following lemma.

Lemma 6 (Concentration of the empirical gradient vector).

Given p = ???(mk 2 ) samples, the empirical gradient vector estimate corresponding to the i-th dictionary element, g (t)i concentrates around its expectation, i.e., g DISPLAYFORM3 ??? Step III.C: Show that the empirical gradient vector is correlated with the descent directionNext, in the following lemma we show that the empirical gradient vector g DISPLAYFORM4 j is correlated with the descent direction.

This is the main result which enables the progress in the dictionary (and coefficients) at each iteration t. Lemma 7 (Empirical gradient vector is correlated with the descent direction).

Suppose DISPLAYFORM5 , and for any t ??? [T ], DISPLAYFORM6 This result ensures for at any t ??? [T ], the gradient descent-based updates made via FORMULA4 gets the columns of the dictionary estimate closer to the true dictionary, i.e., t+1 ??? t .

Moreover, this step requires closeness between the dictionary estimate A (t) and A * , in the spectral norm-sense, as per Def 1.

As discussed above, the closeness property (Def 1) is crucial to show that the gradient vector is correlated with the descent direction.

Therefore, we now ensure that the updated dictionary A (t+1) maintains this closeness property.

Lemma 7 already ensures that t+1 ??? t .

As a result, we show that A (t+1) maintains closeness in the spectral norm-sense as required by our algorithm, i.e., that it is still ( t+1 , 2)-close to the true dictionary.

Also, since we use the gradient matrix in this analysis, we show that the empirical gradient matrix concentrates around its mean.??? Step IV.A: The empirical gradient matrix concentrates around its expectation: We first show that the empirical gradient matrix concentrates as formalized by the following lemma.

Lemma 8 (Concentration of the empirical gradient matrix).

With probability at least DISPLAYFORM0 Step IV.B: The "closeness" property is maintained after the updates made using the empirical gradient estimate: Next, the following lemma shows that the updated dictionary Amaintains the closeness property.

Lemma 9 (A (t+1) maintains closeness).

Suppose A (t) is ( t , 2) near to A * with t = O * (1/ log(n)), and number of samples used in step t is p = ???(mk 2 ), then with probability DISPLAYFORM1

Proof of Theorem 1.

From Lemma 7 we have that with probability at least (1 ??? ?? DISPLAYFORM0 j is (???(k/m), ???(m/k), 0)-correlated with A * j .

Further, Lemma 9 ensures that each iterate maintains the closeness property.

Now, applying Lemma 15 we have that, for ?? A ??? ??(m/k), with probability at least (1 ??? ?? DISPLAYFORM1 0 .

where for 0 < ?? < 1/2 with ?? = ???(k/m)?? A .

That is, the updates converge geometrically to A * .

Further, from Lemma 3, we have that the result on the error incurred by the coefficients.

Here,

Published as a conference paper at ICLR 2019 DISPLAYFORM0 ).

That is, the updates converge geometrically to A * .

Further, from Lemma 3, we have that the error in the coefficients only depends on the error in the dictionary, which leads us to our result on the error incurred by the coefficients.

This completes the proof of our main result.

We present the proofs of the Lemmas used to establish our main result.

Also, see TAB2 for a map of dependence between the results, and Appendix D for proofs of intermediate results.

Claim 8Claim 9Lemma 7Lemma 8Claim 10Lemma 9Theorem 1Proof of Lemma 1.

Let y ??? R n be general sample generated as y = A * x * , where x * ??? R m is a sparse random vector with support S = supp(x * ) distributed according to D.4.The initial decoding step at the t-th iteration (shown in Algorithm 1) involves evaluating the innerproduct between the estimate of the dictionary A (t) , and y. The i-th element of the resulting vector can be written as DISPLAYFORM0 where DISPLAYFORM1 2 ??? t and DISPLAYFORM2 , otherwise.

Now, we focus on the w i and show that it is small.

By the definition of w i we have DISPLAYFORM3 Here, since var(x * ) = 1, w i is a zero-mean random variable with variance DISPLAYFORM4 Now, each term in this sum can be bounded as, DISPLAYFORM5 Therefore, we have the following as per our assumptions on ?? and k, DISPLAYFORM6 using Gershgorin Circle Theorem (Gershgorin, 1931) .

Therefore, we have DISPLAYFORM7 Finally, we have that DISPLAYFORM8 Now, we apply the Chernoff bound for sub-Gaussian random variables w i (shown in Lemma 12) to conclude that DISPLAYFORM9 ).Further, w i corresponding to each m should follow this bound, applying union bound we conclude that DISPLAYFORM10 T .Proof of Lemma 2.

Consider the (r + 1)-th iterate x (r+1) for the t-th dictionary iterate, where DISPLAYFORM11 ??? t for all i ??? [1, m] evaluated as the following by the update step described in Algorithm 1, DISPLAYFORM12 where ??(1)x < 1 is the learning rate or the step-size parameter.

Now, using Lemma 1 we know that x (0) (3) has the correct signed-support with probability at least (1 ??? ?? DISPLAYFORM13 can be written as DISPLAYFORM14 we can write the (r + 1)-th iterate of the coefficient update step using (7) as DISPLAYFORM15 Further, the j-th entry of this vector is given by DISPLAYFORM16 We now develop an expression for the j-th element of each of the term in FORMULA79 as follows.

First, we can write the first term as DISPLAYFORM17 Next, the second term in (8) can be expressed as DISPLAYFORM18 Finally, we have the following expression for the third term, DISPLAYFORM19 Now using our definition of ?? DISPLAYFORM20 2 , combining all the results for (8), and using the fact that since A (t) is close to A * , vectors A (t) j ??? A * j and A * j enclose an obtuse angle, we have the following for the j-th entry of the (r + 1)-th iterate, x (r+1) is given by DISPLAYFORM21 Here ?? (r+1) j is defined as DISPLAYFORM22 , we can write ?? DISPLAYFORM23 where ?? (t) j is defined as DISPLAYFORM24 Note that ?? (t) j does not change for each iteration r of the coefficient update step.

Further, by Claim 2 we show that |?? DISPLAYFORM25 where ?? DISPLAYFORM26 .

Further, using Claim 1, DISPLAYFORM27 since x (r???1) ??? x * 1 = O(k).

Therefore, for the (r + 1)-th iteration, we choose the threshold to be DISPLAYFORM28 and the step-size by setting the "noise" component of FORMULA84 to be smaller than the "signal" part, specifically, half the signal component, i.e., DISPLAYFORM29 Also, since we choose the threshold as ?? (r) := ?? DISPLAYFORM30 max , where x (0) min = C/2, we have the following for the (r + 1)-th iteration, DISPLAYFORM31 Therefore, for this step we choose ?? DISPLAYFORM32 Therefore, ?? DISPLAYFORM33 can be chosen as DISPLAYFORM34 .

In addition, if we set all ?? (r) DISPLAYFORM35 Further, since we initialize with the hard-thresholding step, the entries in |x (0) | ??? C/2.

Here, we define ?? FORMULA0 , we have DISPLAYFORM36 From Claim 2 we have that |?? (t) i1 | ??? t ?? with probability at least (1 ??? ?? (t) ?? ).

Further, using Claim 1 , and letting C DISPLAYFORM37 Rearranging the expression for (r + 1)-th update (9), and using (16) we have the following upperbound DISPLAYFORM38 .

i1 , where we define q= (1 ??? ?? DISPLAYFORM0 Here, ?? DISPLAYFORM1 is defined as DISPLAYFORM2 Our aim now will be to express C ( ) DISPLAYFORM3 . .

, i k , and all ?? ( ) x = ?? x .

Then, using Claim 3 we have the following expression for C DISPLAYFORM4 Here, DISPLAYFORM5 Next from Claim 4 we have that with probability at DISPLAYFORM6

, and using the result on sum of geometric series, we have DISPLAYFORM0

i1 is upper-bounded as DISPLAYFORM0 Further, since k = O( ??? n/?? log(n)), kc x < 1, therefore, we have DISPLAYFORM1 with probability at least (1 ??? ?? DISPLAYFORM2 i1 ?? R 0 for an appropriately large R. Therefore, the error in each non-zero coefficient is DISPLAYFORM3 with probability at least (1 ??? ?? (t) ?? ).

i1 as defined in (9), and recursively substituting for x (r) i1 we have DISPLAYFORM0 where we set all ?? r x to be ?? x .

Further, on defining DISPLAYFORM1 where ?? DISPLAYFORM2 Note that ?? (R) i1 can be made appropriately small by choice of R. Further, by Claim 5 we have DISPLAYFORM3 with probability at least (1 ??? ?? DISPLAYFORM4 Proof of Lemma 5.

From Lemma 4 we have that for each j ??? S, DISPLAYFORM5 with probability at least (1 ??? ?? DISPLAYFORM6 and let 1 F x * denote the indicator function corresponding to this event.

As we show in Lemma 2, this event occurs with probability at least (1 ??? ?? DISPLAYFORM7 T ).

Using this, we can write the expected gradient vector corresponding to the j-th sample as 1 DISPLAYFORM8 Here, ?? := E[(A (t) x ??? y)sign(x * j )1 F x * ] is small and depends on ?? (t)T and ??

?? , which in turn drops with t .

Therefore, ?? diminishes with t .

Further, since 1 F x * + 1 F x * = 1, and Pr[ DISPLAYFORM0 T ), is very large, DISPLAYFORM1 Therefore, we can write g DISPLAYFORM2 S ] can be made very small by choice of R, we absorb this term in ??.

Therefore, DISPLAYFORM3 Writing the expectation by sub-conditioning on the support, DISPLAYFORM4 where we have used the fact that E x * S [sign(x * j )] = 0 and introduced DISPLAYFORM5

Published as a conference paper at ICLR 2019 DISPLAYFORM0 Further, by Claim 7 we have that DISPLAYFORM1 This completes the proof.

Proof of Lemma 6.

Let W = {j : i ??? supp(x * (j) )} and then we have that DISPLAYFORM2 where x (j) (i) denotes the i-th element of the coefficient estimate corresponding to the (j)-th sample.

Here, for = |W | the summation DISPLAYFORM3 has the same distribution as ?? j=1 z j , where each z j belongs to a distribution as DISPLAYFORM4 Let w j = z j ??? E[z], we will now apply the vector Bernstein result shown in Lemma 11.

For this, we require bounds on two parameters for these -L := w j and ?? 2 := ?? j E[ w j 2 ] .

Note that, since the quantity of interest is a function of x * i , which are sub-Gaussian, they are only bounded almost surely.

To this end, we will employ Lemma 14 (Lemma 45 in (Arora et al., 2015) ) to get a handle on the concentration.

Bound on the norm w : This bound is evaluated in Claim 8, which states that with probability at least (1 ??? ?? DISPLAYFORM5 Bound on variance parameter E[ w 2 ]: Using Claim 9, we have DISPLAYFORM6 .

Therefore, the bound on the variance parameter ?? 2 is given by DISPLAYFORM7 From Claim 2 we have that with probability at least (1 ??? ?? DISPLAYFORM8 Applying vector Bernstein inequality shown in Lemma 11 and using Lemma 14 (Lemma 45 in (Arora et al., 2015) ), choosing = ???(k 3 ), we conclude DISPLAYFORM9 with probability at least (1 ??? ?? DISPLAYFORM10 Finally, substituting in (21) we have DISPLAYFORM11 with probability at least (1 ??? ?? DISPLAYFORM12 Proof of Lemma 7.

Since we only have access to the empirical estimate of the gradient g (t)i , we will show that this estimate is correlated with (A (t) j ??? A * j ).

To this end, first from Lemma 6 we have that the empirical gradient vector concentrates around its mean, specifically, DISPLAYFORM13 with probability at least (1 ??? ?? DISPLAYFORM14 HW ).

From Lemma 5, we have the following expression for the expected gradient vector DISPLAYFORM15 Then, g (t)i can be written as DISPLAYFORM16 where DISPLAYFORM17 Using the definition of v as shown in FORMULA2 we have DISPLAYFORM18

Further, using Claim 7 DISPLAYFORM0 Now, since A (t) ??? A * ??? 2 A * (the closeness property (Def.1) is maintained at every step using Lemma 9), and further since A * = O( m/n), we have that DISPLAYFORM1 Therefore, we have DISPLAYFORM2 Here, we use the fact that ?? drops with decreasing t as argued in Lemma 5.

Next, using (23), we have DISPLAYFORM3 we have that, DISPLAYFORM4 Substituting for v , this implies that g (t) DISPLAYFORM5 Further, we also have the following lower-bound DISPLAYFORM6

Published as a conference paper at ICLR 2019Here, we use the fact that R.H.S. can be minimized only if v is directed opposite to the direction of A (t) j ??? A * j .

Now, we show that this gradient is (?? , 1/100?? , 0) correlated, DISPLAYFORM0 Therefore, for this choice of k, i.e. k = O( ??? n), there is no bias in dictionary estimation in comparison to Arora et al. (2015) .

This gain can be attributed to estimating the coefficients simultaneously with the dictionary.

Further, since we choose 4?? = p j q j , we have that ?? = ??(k/m), as a result ?? + = 1/100?? = ???(m/k).

Applying Lemma 15 we have DISPLAYFORM1 Proof of Lemma 8.

Here, we will prove that g (t) defined as DISPLAYFORM2 concentrates around its mean.

Notice that each summand ( DISPLAYFORM3 ) is a random matrix of the form (y ??? A (t) x)sign( x) .

Also, we have g (t) defined as DISPLAYFORM4 To bound DISPLAYFORM5 , we are interested in p j=1 W j , where each matrix W j is given by DISPLAYFORM6 Noting that E[W j ] = 0, we will employ the matrix Bernstein result (Lemma 10) to bound g (t) ??? g (t) .

To this end, we will bound W j and the variance proxy DISPLAYFORM7 Now, since each x (j) has k non-zeros, sign( x (j) ) sign( x (j) ) = k, and using Claim 10, with proba- DISPLAYFORM8 Similarly, expanding E[W j W j ], and using the fact that DISPLAYFORM9 ] is positive semi-definite.

Now, using Claim 8 and the fact that entries of E [(sign( x (j) )sign( x (j) ) ] are q i on the diagonal and zero elsewhere, where DISPLAYFORM10 mp ).

Now, we are ready to apply the matrix Bernstein result.

Since, m = O(n) the variance statistic comes out to be O( DISPLAYFORM11 , then as long as we choose p = ???(mk 2 ) (using the bound on t ?? ), with probability at least (1 ??? ?? DISPLAYFORM12 Proof of Lemma 9.

This lemma ensures that the dictionary iterates maintain the closeness property (Def.1) and satisfies the prerequisites for Lemma 7.The update step for the i-th dictionary element at the s + 1 iteration can be written as DISPLAYFORM13 Here, g (t) i is given by the following as per Lemma 5 with probability at least (1 ??? ?? DISPLAYFORM14

i in the dictionary update step, DISPLAYFORM0 Therefore, the update step for the dictionary (matrix) can be written as DISPLAYFORM1 where, DISPLAYFORM2 i ) and V = A (t) Q, with the matrix Q given by, DISPLAYFORM3 , and using the following intermediate result shown in Claim 7, DISPLAYFORM4 Therefore, DISPLAYFORM5 We will now proceed to bound each term in (25).

Starting with (A (t) ??? A * )diag(1 ??? ?? A p i q i ), and using the fact that p i = O(1), q i = O(k/m), and DISPLAYFORM6 Using the results derived above, and the the result derived in Lemma 8 which states that with probability at least (1 ??? ?? DISPLAYFORM7

Proof of Claim 1.

We start by looking at the incoherence between the columns of A * , for j = i, DISPLAYFORM0 Claim 2 (Bound on ?? (t) j : the noise component in coefficient estimate that depends on t ).

With DISPLAYFORM1 ).Proof of Claim 2.

We have the following definition for ?? (t) j from (11), DISPLAYFORM2 Here, since x * i are independent sub-Gaussian random variables, ?? (t) j is a sub-Gaussian random variable with the variance parameter evaluated as shown below DISPLAYFORM3

Pr[|?? DISPLAYFORM0 Now, we need this for each ?? (t) j for j ??? supp(x * ), union bounding over k coefficients DISPLAYFORM1 ).Claim 3 (Error in coefficient estimation for a general iterate (r + 1)).

The error in a general iterate r of the coefficient estimation is upper-bounded as DISPLAYFORM2 Proof of Claim 3 .

From (17) we have the following expression for C (r+1) i1 DISPLAYFORM3 Our aim will be to recursively substitute for C DISPLAYFORM4 as a function of C 0 max .

To this end, we start by analyzing the iterates C DISPLAYFORM5 i1 , and so on to develop an expression for C (r+1) i1 as follows.

Published as a conference paper at ICLR 2019 DISPLAYFORM0 i1 is given by DISPLAYFORM1 Further, we know from (26) we have DISPLAYFORM2 Therefore, since DISPLAYFORM3 Expression for Ci1 -Next, we writing Ci1 , DISPLAYFORM4 i2 .Here, using (27) we have the following expression for C DISPLAYFORM5 Substituting for Ci2 in the expression for Ci1 , and rearranging the terms in the expression for Ci1 , we have DISPLAYFORM6 Expression for C DISPLAYFORM7 Substituting for Ci2 from (28), Ci2 from (27), Ci2 using (26), and rearranging, DISPLAYFORM8 i5 .Notice that the terms have a binomial series like form.

To reveal this structure, let each ?? DISPLAYFORM9 max for j = i 1 , i 2 , . . .

, i k .

Therefore, we have DISPLAYFORM10 Further upper-bounding the expression, we have DISPLAYFORM11 Therefore, DISPLAYFORM12 -With this, we are ready to write the general term, DISPLAYFORM13 Claim 4 (An intermediate result for bounding the error in coefficient calculations).

With prob- DISPLAYFORM14 Proof of Claim 4 .

Using (18), the quantity ?? DISPLAYFORM15

Published as a conference paper at ICLR 2019 Therefore, we are interested in DISPLAYFORM0 Consider the first term which depends on C (0) DISPLAYFORM1 , we have DISPLAYFORM2 where ?? R is a small constant, and a parameter which determines the number of iterations R required for the coefficient update step.

Now, coming back to the quantity of interest DISPLAYFORM3 Now, using sum of geometric series result, we have that DISPLAYFORM4 , and DISPLAYFORM5 .

Therefore, with probability at least (1 ??? ?? DISPLAYFORM6 2 and |?? (t)i | = t ?? with probability at least (1 ??? ?? (t) ?? ) using Claim 2.Claim 5 (Bound on the noise term in the estimation of a coefficient element in the support).

With probability (1 ??? ?? DISPLAYFORM7 i1 is defined as DISPLAYFORM8 i1 is as defined in FORMULA0 , DISPLAYFORM9 Therefore, we have the following expression for ?? DISPLAYFORM10

Published as a conference paper at ICLR 2019 DISPLAYFORM0 i1 can be upper-bounded as DISPLAYFORM1 Since from Claim 6 we have DISPLAYFORM2 Further, since 1 ??? (1 ??? ?? x ) r???1 ??? 1, we have that DISPLAYFORM3 Therefore, DISPLAYFORM4 i1 .

i | = t ?? with probability at least (1 ??? ?? DISPLAYFORM0 ?? ) for the t-th iterate, and k = O * ( ??? n ?? log(n) ), therefore kc x < 1, we have that DISPLAYFORM1 with probability at least (1 ??? ?? DISPLAYFORM2 , we have DISPLAYFORM3 Proof of Claim 6.

Here, from Claim 3 we have that for any i 1 , DISPLAYFORM4 is given by DISPLAYFORM5 Further, the term of interest C (r???1) i2(1 ??? ?? x ) R???r can be upper-bounded by DISPLAYFORM6 From the definition of ?? can be written as DISPLAYFORM7 Therefore, we have FORMULA0 , where ?? DISPLAYFORM8 DISPLAYFORM9 Therefore, DISPLAYFORM10 .Therefore, DISPLAYFORM11 Therefore, combining all the results we have that, for a constant DISPLAYFORM12 Claim 7 (Bound on the noise term in expected gradient vector estimate).

DISPLAYFORM13 Proof of Claim 7.

DISPLAYFORM14 From FORMULA43 we have the following definition for ?? DISPLAYFORM15 where ?? (t) j is defined as the following (11) DISPLAYFORM16 S is a vector with each element as defined in (30).

Therefore, the elements of the vector DISPLAYFORM17 Consider the general term of interest DISPLAYFORM18 Further, since DISPLAYFORM19 we have that DISPLAYFORM20 Further, for DISPLAYFORM21 In addition, for s =i ??? s we have that DISPLAYFORM22 Therefore, using the results for ??? and DISPLAYFORM23 and for i = j we have DISPLAYFORM24 Here, from Claim 6, for c x = DISPLAYFORM25 Further, due to our assumptions on sparsity, kc x ??? 1; in addition by Claim 2, and with probability at least (1 ??? ?? DISPLAYFORM26 with probability at least (1 ??? ?? (t) ?? ).

Combining results from (31), (32) and substituting for the terms in (33) using the analysis above, DISPLAYFORM27 Note that since ?? DISPLAYFORM28 i )) can be made small by choice of R. Also, since Pr[i, j ??? S] = q i,j , we have DISPLAYFORM29 Claim 8 (An intermediate result for concentration results).

With probability (1 ??? ?? DISPLAYFORM30 Proof of Claim 8.

First, using Lemma 4 we have DISPLAYFORM31 i1 .

Therefore, the vector x S , for S ??? supp(x * ) can be written as DISPLAYFORM32 where x has the correct signed-support with probability at least (1 ??? ?? T ) using Lemma 2.

Using this result, we can write y ??? A (t) x as DISPLAYFORM33 With x * S being independent and sub-Gaussian, using Lemma 13, which is a result based on the Hanson-Wright result BID2 ) for sub-Gaussian random variables, and since A (t) DISPLAYFORM34 we have that with probability at least (1 ??? ?? DISPLAYFORM35 ).

DISPLAYFORM36 Consider the ??? term.

Using Claim 5, each ?? (R) j is bounded by O(t ?? ).

with probability at least DISPLAYFORM37 Finally, combining all the results and using the fact that A * DISPLAYFORM38 Claim 9 (Bound on variance parameter for concentration of gradient vector).

DISPLAYFORM39 Proof of Claim 9.

For the variance E[ z 2 ], we focus on the following, DISPLAYFORM40 Here, x S is given by DISPLAYFORM41 We will now consider each term in (35) separately.

We start with ???.

Since x * S s are conditionally independent of S, E[x * S x * S ] = I. Therefore, we can simplify this expression as DISPLAYFORM42 .

Rearranging the terms we have the following for ???, DISPLAYFORM43 Therefore, ??? can be upper-bounded as DISPLAYFORM44 Next, since (1 ??? ?? (t) j ) ??? 1, we have the following bound for ??? 2 DISPLAYFORM45 Further, ??? 3 can be upper-bounded by using bounds for ??? 1 and ??? 2 .

Combining the results of upperbounding ??? 1 , ??? 2 , and ??? 3 we have the following for (36) DISPLAYFORM46 S .

Therefore we have DISPLAYFORM47 S .

where 1 m??m denotes an m ?? m matrix of ones.

Now, we turn to DISPLAYFORM0 ] in (38), which can be simplified as DISPLAYFORM1 ] in (38) which can also be bounded similarly as DISPLAYFORM2 Therefore, we have the following for ??? in (37) DISPLAYFORM3 Consider ??? in (37).

DISPLAYFORM4 S |S], and using the analysis similar to that shown in 7, we have that elements of M ??? R k??k are given by DISPLAYFORM5 We have the following, DISPLAYFORM6 m 2 ), and 1 m??m = m, DISPLAYFORM7 Therefore, DISPLAYFORM8 Similarly, ??? in (37) is also bounded as ???.

Next, we consider ??? in (37).

In this case, letting DISPLAYFORM9 where N ??? R k??k is a matrix whose each entry N i,j ??? |?? DISPLAYFORM10 .

with probability at least (1 ??? ?? DISPLAYFORM11 Again, using the result on |?? DISPLAYFORM12 Combining all the results for ???, ???, ??? and ???, we have, DISPLAYFORM13

We now present some additional results to highlight the features of NOODL.

Specifically, we compare the performance of NOODL (for both dictionary and coefficient recovery) with the state-of-theart provable techniques for DL presented in Arora et al. (2015) (when the coefficients are recovered via a sparse approximation step after DL) 3 .

We also compare the performance of NOODL with the popular online DL algorithm in BID12 , denoted by Mairal '09.

Here, the authors show that alternating between a 1 -based sparse approximation and dictionary update based on block co-ordinate descent converges to a stationary point, as compared to the true factors in case of NOODL.Data Generation: We generate a (n = 1000) ?? (m = 1500) matrix, with entries drawn from N (0, 1), and normalize its columns to form the ground-truth dictionary A * .

Next, we perturb A * with random Gaussian noise, such that the unit-norm columns of the resulting matrix, A (0) are 2/ log(n) away from A * , in 2 -norm sense, i.e., 0 = 2/ log(n); this satisfies the initialization assumptions in A.4.

At each iteration, we generate p = 5000 samples Y ??? R 1000??5000 as Y = A * X * , where X * ??? R m??p has at most k = 10, 20, 50, and 100, entries per column, drawn from the Radamacher distribution.

We report the results in terms of relative Frobenius error for all the experiments, i.e., for a recovered matrix M, we report M ??? M * F / M * F .

To form the coefficient estimate for Mairal '09 via Lasso (Tibshirani, 1996) we use the FISTA (Beck and Teboulle, 2009) algorithm by searching across 10 values of the regularization parameter at each iteration.

Note that, although our phase transition analysis for NOODL shows that p = m suffices, we use p = 5000 in our convergence analysis for a fair comparison with related techniques.

TAB3 summarizes the results of the convergence analysis shown in FIG4 .

Here, we compare the dictionary and coefficient recovery performance of NOODL with other techniques.

For Arora15(''biased'') and Arora15(''unbiased''), we report the error in recovered coefficients after the HT step (X HT ) and the best error via sparse approximation using Lasso 4 Tibshirani (1996) , denoted as X Lasso , by scanning over 50 values of regularization parameter.

For Mairal '09 at each iteration of the algorithm we scan across 10 values 5 of the regularization parameter, to recover the best coefficient estimate using Lasso ( via FISTA), denoted as X Lasso .

We observe that NOODL exhibits significantly superior performance across the board.

Also, we observe that using sparse approximation after dictionary recovery, when the dictionary suffers from a bias, leads to poor coefficient recovery 6 , as is the case with Arora15(''biased''), Arora15(''unbiased''), and Mairal '09.

This highlights the applicability of our approach in real-world machine learning tasks where coefficient recovery is of interest.

In fact, it is a testament to the fact that, even in cases where dictionary recovery is the primary goal, making progress on the coefficients is also important for dictionary recovery.

In addition, the coefficient estimation step is also online in case of NOODL, while for the stateof-the-art provable techniques (which only recover the dictionary and incur bias in estimation) need additional sparse approximation step for coefficient recovery.

Moreover, these sparse approximation techniques (such as Lasso) are expensive to use in practice, and need significant tuning.

In addition to these convergence results, we also report the computational time taken by each of these algorithms in TAB3 .

The results shown here were compiled using 5 cores and 200GB RAM of Intel Xeon E5 ??? 2670 Sandy Bridge and Haswell E5-2680v3 processors.

The primary takeaway is that although NOODL takes marginally more time per iteration as compared to other methods when accounting for just one Lasso update step for the coefficients, it (a) is in fact faster per iteration since it does not involve any computationally expensive tuning procedure to scan across regularization parameters; owing to its geometric convergence property (b) achieves orders of magnitude superior error at convergence, and as a result, (c) overall takes significantly less time to reach such a solution.

Further, NOODL's computation time can be further reduced via implementations using the neural architecture illustrated in Section 4.Note that since the coefficient estimates using just the HT step at every step may not yield a usable result for Arora15(''unbiased'') and Arora15(''biased'') as shown in TAB3 , in practice, one has to employ an additional 1 -based sparse recovery step.

Therefore, for a fair comparison, we account for running sparse recovery step(s) using Lasso (via the Fast Iterative ShrinkageThresholding Algorithm (FISTA) (Beck and Teboulle, 2009) ) at every iteration of the algorithms Arora15(''biased'') and Arora15(''unbiased'').For our technique, we report the average computation time taken per iteration.

However, for the rest of the techniques, the coefficient recovery using Lasso (via FISTA) involves a search over various values of the regularization parameters (10 values for this current exposition).

As a result, we analyze the computation time per iteration via two metrics.

First of these is the average computation time taken per iteration by accounting for the average time take per Lasso update (denoted as "Accounting for one Lasso update"), and the second is the average time taken per iteration to scan over all (10) values of the regularization parameter (denoted as "Overall Lasso search") .

4 We use the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) (Beck and Teboulle, 2009) , which is among the most efficient algorithms for solving the 1-regularized problems.

Note that, in our experiments we fix the step-size for FISTA as 1/L, where L is the estimate of the Lipschitz constant (since A is not known exactly).5 Note that, although scanning across 50 values of the regularization parameter for this case would have led to better coefficient estimates and dictionary recovery, we choose 10 values for this case since it is very expensive to scan across 50 of regularization parameter at each step.

This also highlights why Mairal '09 may be prohibitive for large scale applications.6 When the dictionary is not known exactly, the guarantees may exist on coefficient recovery only in terms of closeness in 2-norm sense, due to the error-in-variables (EIV) model for the dictionary (Fuller, 2009; Wainwright, 2009 ).

Arora et al. (2015) , we scan across 50 values of the regularization parameter for coefficient estimation using Lasso after learning the dictionary (A), and report the optimal estimation error for the coefficients (XLasso), while for Mairal '09, at each step the coefficients estimate is chosen by scanning across 10 values of the regularization parameters.

For k = 100, the algorithms of Arora et al. FORMULA0 Avg.

As shown in TAB3 , in comparison to NOODL the techniques described in Arora et al. (2015) still incur a large error at convergence, while the popular online DL algorithm of BID12 exhibits very slow convergence rate.

Combined with the convergence results shown in FIG4 , we observe that due to NOODL's superior convergence properties, it is overall faster and also geometrically converges to the true factors.

This again highlights the applicability of NOODL in practical applications, while guaranteeing convergence to the true factors.

Definition 6 (sub-Gaussian Random variable).

Let x ??? subGaussian(?? 2 ).

Then, for any t > 0, it holds that Pr[|x| > t] ??? 2 exp t 2 2?? 2 .

Lemma 10 (Matrix Bernstein (Tropp, 2015) ).

Consider a finite sequence W k ??? R n??m of independent, random, centered matrices with dimension n. Assume that each random matrix satisfies E[W k ] = 0 and W k ??? R almost surely.

Then, for all t ??? 0, DISPLAYFORM0 ?? 2 +Rt/3 , where ?? 2 := max{ DISPLAYFORM1 Furthermore, E[ k W k ] ??? 2?? 2 log(n + m) + 1 3 R log(n + m).

@highlight

We present a provable algorithm for exactly recovering both factors of the dictionary learning model. 