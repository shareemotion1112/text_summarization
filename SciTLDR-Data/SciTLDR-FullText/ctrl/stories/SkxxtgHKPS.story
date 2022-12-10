Generalization error (also known as the out-of-sample error) measures how well the hypothesis learned from training data generalizes to previously unseen data.

Proving tight generalization error bounds is a central question in statistical learning  theory.

In  this  paper,  we  obtain  generalization  error  bounds  for  learning general  non-convex  objectives,  which  has  attracted  significant  attention  in  recent years.

We develop a new framework,  termed Bayes-Stability,  for proving algorithm-dependent generalization error bounds.

The new framework combines ideas from both the PAC-Bayesian theory and the notion of algorithmic stability.

Applying the Bayes-Stability method, we obtain new data-dependent generalization bounds for stochastic gradient Langevin dynamics (SGLD) and several other noisy gradient methods (e.g., with momentum, mini-batch and acceleration, Entropy-SGD).

Our result recovers (and is typically tighter than) a recent result in Mou et al. (2018) and improves upon the results in Pensia et al. (2018).

Our experiments demonstrate that our data-dependent bounds can distinguish randomly labelled data from normal data, which provides an explanation to the intriguing phenomena observed in Zhang et al. (2017a).

We also study the setting where the total loss is the sum of a bounded loss and an additiona l`2 regularization term.

We obtain new generalization bounds for the continuous Langevin dynamic in this setting by developing a new Log-Sobolev inequality for the parameter distribution at any time.

Our new bounds are more desirable when the noise level of the processis not very small, and do not become vacuous even when T tends to infinity.

Non-convex stochastic optimization is the major workhorse of modern machine learning.

For instance, the standard supervised learning on a model class parametrized by R d can be formulated as the following optimization problem:

where w denotes the model parameter, D is an unknown data distribution over the instance space Z, and F : R d × Z → R is a given objective function which may be non-convex.

A learning algorithm takes as input a sequence S = (z 1 , z 2 , . . .

, z n ) of n data points sampled i.i.d.

from D, and outputs a (possibly randomized) parameter configurationŵ ∈ R d .

A fundamental problem in learning theory is to understand the generalization performance of learning algorithms-is the algorithm guaranteed to output a model that generalizes well to the data distribution D?

Specifically, we aim to prove upper bounds on the generalization error err gen (S) = L(ŵ, D) − L(ŵ, S), where L(ŵ, D) = Ez∼D[L(ŵ, z)] and L(ŵ, S) = 1 n n i=1 L(ŵ, z i ) are the population and empirical losses, respectively.

We note that the loss function L (e.g., the 0/1 loss) could be different from the objective function F (e.g., the cross-entropy loss) used in the training process (which serves as a surrogate for the loss L).

Classical learning theory relates the generalization error to various complexity measures (e.g., the VC-dimension and Rademacher complexity) of the model class.

Directly applying these classical complexity measures, however, often fails to explain the recent success of over-parametrized neural networks, where the model complexity significantly exceeds the amount of available training data (see e.g., Zhang et al. (2017a) ).

By incorporating certain data-dependent quantities such as margin and compressibility into the classical framework, some recent work (e.g., Bartlett et al. (2017) ; Arora et al. (2018) ; Wei & Ma (2019) ) obtains more meaningful generalization bounds in the deep learning context.

An alternative approach to generalization is to prove algorithm-dependent bounds.

One celebrated example along this line is the algorithmic stability framework initiated by Bousquet & Elisseeff (2002) .

Roughly speaking, the generalization error can be bounded by the stability of the algorithm (see Section 2 for the details).

Using this framework, Hardt et al. (2016) study the stability (hence the generalization) of stochastic gradient descent (SGD) for both convex and non-convex functions.

Their work motivates recent study of the generalization performance of several other gradient-based optimization methods: Kuzborskij & Lampert (2018) ; London (2016); Chaudhari et al. (2017) ; Raginsky et al. (2017) ; Mou et al. (2018) ; Pensia et al. (2018) ; Chen et al. (2018) .

In this paper, we study the algorithmic stability and generalization performance of various iterative gradient-based method, with certain continuous noise injected in each iteration, in a non-convex setting.

As a concrete example, we consider the stochastic gradient Langevin dynamics (SGLD) (see Raginsky et al. (2017) ; Mou et al. (2018) ; Pensia et al. (2018) ).

Viewed as a variant of SGD, SGLD adds an isotropic Gaussian noise at every update step:

where g t (W t−1 ) denotes either the full gradient or the gradient over a mini-batch sampled from training dataset.

We also study a continuous version of (1), which is the dynamic defined by the following stochastic differential equation (SDE):

where B t is the standard Brownian motion.

Most related to our work is the study of algorithm-dependent generalization bounds of stochastic gradient methods.

Hardt et al. (2016) first study the generalization performance of SGD via algorithmic stability.

They prove a generalization bound that scales linearly with T , the number of iterations, when the loss function is convex, but their results for general non-convex optimization are more restricted.

London (2017) presents a generalization bound that also combines PAC-Bayesian analysis with stability.

However, their prior and posterior are probability distributions on the hyperparameter space, while ours are distributions on the hypothesis space.

Our work is a follow-up of the recent work by Mou et al. (2018) , in which they provide generalization bounds for SGLD from both stability and PAC-Bayesian perspectives.

Another closely related work by Pensia et al. (2018) derives similar bounds for noisy stochastic gradient methods, based on the information theoretic framework of Xu & Raginsky (2017) .

However, their bounds scale as O( T /n) (n is the size of the training dataset) and are sub-optimal even for SGLD.

We acknowledge that besides the algorithm-dependent approach that we follow, recent advances in learning theory aim to explain the generalization performance of neural networks from many other perspectives.

Some of the most prominent ideas include bounding the network capacity by the norms of weight matrices Neyshabur et al. Chizat & Bach (2018) .

Most of these results are stated in the context of neural networks (some are tailored to networks with specific architecture), whereas our work addresses generalization in non-convex stochastic optimization in general.

We also note that some recent work provides explanations for the phenomenon reported in Zhang et al. (2017a) from a variety of different perspectives (e.g., Bartlett et al. (2017) ; Arora et al. (2018; 2019) ).

Welling & Teh (2011) first consider stochastic gradient Langevin dynamics (SGLD) as a sampling algorithm in the Bayesian inference context.

Raginsky et al. (2017) give a non-asymptotic analysis and establish the finite-time convergence guarantee of SGLD to an approximate global minimum.

Zhang et al. (2017b) analyze the hitting time of SGLD and prove that SGLD converges to an approximate local minimum.

These results are further improved and generalized to a family of Langevin dynamics based algorithms by the subsequent work of Xu et al. (2018) .

In this paper, we provide generalization guarantees for the noisy variants of several popular stochastic gradient methods.

The Bayes-Stability method and data-dependent generalization bounds.

We develop a new method for proving generalization bounds, termed as Bayes-Stability, by incorporating ideas from the PAC-Bayesian theory into the stability framework.

In particular, assuming the loss takes value in [0, C], our method shows that the generalization error is bounded by both 2C Ez[ 2KL(P, Q z )] and 2C Ez[ 2KL(Q z , P )], where P is a prior distribution independent of the training set S, and Q z is the expected posterior distribution conditioned on z n = z (i.e., the last training data is z).

The formal definition and the results can be found in Definition 5 and Theorem 7.

Inspired by Lever et al. (2013) , instead of using a fixed prior distribution, we bound the KLdivergence from the posterior to a distribution-dependent prior.

This enables us to derive the following generalization error bound that depends on the expected norm of the gradient along the optimization path:

Here S is the dataset and g e (t)

is the expected empirical squared gradient norm at step t; see Theorem 11 for the details. , where L is the global Lipschitz constant of the loss, our new bound (3) depends on the data distribution and is typically tighter (as the gradient norm is at most L).

In modern deep neural networks, the worstcase Lipschitz constant L can be quite large, and typically much larger than the expected empirical gradient norm along the optimization trajectory.

Specifically, in the later stage of the training, the expected empirical gradient is small (see Figure 1(d) for the details).

Hence, our generalization bound does not grow much even if we train longer at this stage.

Our new bound also offers an explanation to the difference between training on correct and random labels observed by Zhang et al. (2017a) .

In particular, we show empirically that the sum of expected squared gradient norm (along the optimization path) is significantly higher when the training labels are replaced with random labels (Section 3, Remark 13, Figure 1 , Appendix C.2).

We would also like to mention the PAC-Bayesian bound (for SGLD with 2 -regularization) proposed by Mou et al. (2018) .

(This bound is different from what we mentioned before; see Theorem 2 in their paper.)

Their bound scales as O(1/ √ n) and the numerator of their bound has a similar sum of gradient norms (with a decaying weight if the regularization coefficient λ > 0).

Their bound is based on the PAC-Bayesian approach and holds with high probability, while our bound only holds in expectation.

Extensions.

We remark that our technique allows for an arguably simpler proof of (Mou et al., 2018, Theorem 1) ; the original proof is based on SDE and Fokker-Planck equation.

More importantly, our technique can be easily extended to handle mini-batches and a variety of general settings as follows.

1.

Extension to other gradient-based methods.

Our results naturally extends to other noisy stochastic gradient methods including momentum due to Polyak (1964) (Theorem 26), Nesterov's accelerated gradient method in Nesterov (1983) (Theorem 26), and Entropy-SGD proposed by Chaudhari et al. (2017) (Theorem 27).

2.

Extension to general noises.

The proof of the generalization bound in Mou et al. (2018) relies heavily on that the noise is Gaussian 1 , which makes it difficult to generalize to other noise distributions such as the Laplace distribution.

In contrast, our analysis easily carries over to the class of log-Lipschitz noises (i.e., noises drawn from distributions with Lipschitz log densities).

3.

Pathwise stability.

In practice, it is also natural to output a certain function of the entire optimization path, e.g., the one with the smallest empirical risk or a weighted average.

We show that the same generalization bound holds for all such variants (Remark 12).

We note that the analysis in an independent work of Pensia et al. (2018) also satisfies this property,

(see Corollary 1 in their work), which scales at a slower O(1/ √ n) rate (instead of O(1/n)) when dealing with C-bounded loss.

Generalization bounds with 2 regularization via Log-Sobolev inequalities.

We also study the setting where the total objective function F is the sum of a C-bounded differentiable objective F 0 and an additional 2 regularization term λ 2 w 2 2 .

In this case, F can be treated as a perturbation of a quadratic function, and the continuous Langevin dynamics (CLD) is well understood for quadratic functions.

We obtain two generalization bounds for CLD, both via the technique of Log-Sobolev inequalities, a powerful tool for proving the convergence rate of CLD.

One of our bounds is as follows (Theorem 15):

The above bound has the following advantages:

1.

Applying e −x ≥ 1 − x, one can see that our bound is at most O( √ T /n), which matches the previous bound in ( Mou et al., 2018, Proposition 8) 3 .

2.

As time T grows, the bound is upper bounded by and approaches to 2e 4βC CLn −1 β/λ (unlike the previous O( √ T /n) bound that goes to infinity as T → +∞).

If the noise level is not so small (i.e., β is not very large), the generalization bound is quite desirable.

Our analysis is based on a Log-Sobolev inequality (LSI) for the parameter distribution at time t, whereas most known LSIs only hold for the stationary distribution of the Markov process.

We prove the new LSI by exploiting the variational formulation of the entropy formula.

Notations.

We use D to denote the data distribution.

The training dataset S = (z 1 , . . . , z n ) is a sequence of n independent samples drawn from D. S, S ∈ Z n are called neighboring datasets if and only if they differ at exactly one data point (we could assume without loss of generality that z n = z n ).

Let F (w, z) and L(w, z) be the objective and the loss functions, respectively, where w ∈ R d denotes a model parameter and z ∈ Z is a data point.

Define

and L(w, D) are defined similarly.

A learning algorithm A takes as input a dataset S, and outputs a parameter w ∈ R d randomly.

Let G be the set of all possible mini-batches.

G n = {B ∈ G : n ∈ B} denotes the collection of mini-batches that contain the n-th data point, while G n = G \ G n .

Let diam(A) = sup x,y∈A x − y 2 denote the diameter of a set A.

1 In particular, their proof leverages the Fokker-Planck equation, which describes the time evolution of the density function associated with the Langevin dynamics and can only handle Gaussian noise.

2 They assume the loss is sub-Gaussian.

By Hoeffding's lemma, C-bounded random variables are subGaussian with parameter C.

3 The proof of their O( √ T /n) bound can be easily extended to our setting with 2 regularization.

Definition 2 (Expected generalization error).

The expected generalization error of a learning algorithm A is defined as

Algorithmic Stability.

Intuitively, a learning algorithm that is stable (i.e., a small perturbation of the training data does not affect its output too much) can generalize well.

In the seminal work of Bousquet & Elisseeff (2002) (see also Hardt et al. (2016) ), the authors formally defined algorithmic stability and established a close connection between the stability of a learning algorithm and its generalization performance.

Definition 3 (Uniform stability). (Bousquet & Elisseeff (2002); Elisseeff et al. (2005) ) A randomized algorithm A is n -uniformly stable w.r.t.

loss L, if for all neighboring sets S, S ∈ Z n , it holds that

where w S and w S denote the outputs of A on S and S respectively.

Lemma 4 (Generalization in expectation). (Hardt et al. (2016) ) Suppose a randomized algorithm A is n -uniformly stable.

Then, |err gen | ≤ n .

In this section, we incorporate ideas from the PAC-Bayesian theory (see e.g., Lever et al. (2013) ) into the algorithmic stability framework.

Combined with the technical tools introduced in previous sections, the new framework enables us to prove tighter data-dependent generalization bounds.

First, we define the posterior of a dataset and the posterior of a single data point.

Definition 5 (Single-point posterior).

Let Q S be the posterior distribution of the parameter for a given training dataset S = (z 1 , . . .

, z n ).

In other words, it is the probability distribution of the output of the learning algorithm on dataset S (e.g., for T iterations of SGLD in (1), Q S is the pdf of W T ).

The single-point posterior Q (i,z) is defined as

For convenience, we make the following natural assumption on the learning algorithm: Assumption 6 (Order-independent).

For any fixed dataset S = (z 1 , . . . , z n ) and any permutation p, Q S is the same as Q S p , where S p = (z p1 , . . . , z pn ).

Assumption 6 implies Q (1,z) = · · · = Q (n,z) , so we use Q z as a shorthand for Q (i,z) in the following.

Note that this assumption can be easily satisfied by letting the learning algorithm randomly permute the training data at the beginning.

It is also easy to verify that both SGD and SGLD satisfy the order-independent assumption.

Now, we state our new Bayes-stability framework, which holds for any prior distribution P over the parameter space that is independent of the training dataset S. Theorem 7 (Bayes-Stability).

Suppose the loss function L(w, z) is C-bounded and the learning algorithm is order-independent (Assumption 6).

Then for any prior distribution P not depending on S, the generalization error is bounded by both 2C E z 2KL(P, Q z ) and 2C E z 2KL(Q z , P ) .

Remark 8.

Our Bayes-Stability framework originates from the algorithmic stability framework, and hence is similar to the notions of uniform stability and leave-one-out error (see Elisseeff et al. (2003) ).

However, there are important differences.

Uniform stability is a distribution-independent property, while Bayes-Stability can incorporate the information of the data distribution (through the prior P ).

Leave-one-out error measures the loss of a learned model on an unseen data point, yet Bayes-Stability focuses on the extent to which a single data point affects the outcome of the learning algorithm (compared to the prior).

To establish an intuition, we first apply this framework to obtain an expectation generalization bound for (full) gradient Langevin dynamics (GLD), which is a special case of SGLD in (1) (i.e., GLD uses the full gradient ∇ w F (W t−1 , S) as g t (W t−1 )).

Theorem 9.

Suppose that the loss function L is C-bounded.

Then we have the following expected generalization bound for T iterations of GLD:

where g e (t)

] is the empirical squared gradient norm, and W t is the parameter at step t of GLD.

Proof The proof builds upon the following technical lemma, which we prove in Appendix A.2.

Lemma 10.

Let (W 0 , . . .

, W T ) and (W 0 , . . .

, W T ) be two independent sequences of random variables such that for each t ∈ {0, . . .

, T }, W t and W t have the same support.

Suppose W 0 and W 0 follow the same distribution.

Then,

where W ≤t denotes (W 0 , . . .

, W t ) and W <t denotes W ≤t−1 . ,0) ], where 0 denotes the zero data point (i.e., f (w, 0) = 0 for any w).

Theorem 7 shows that

By the convexity of KL-divergence, for a fixed z ∈ Z, we have

Let (W t ) t≥0 and(W t ) t≥0 be the training process of GLD for S = (S, z) and S = (S, 0), respectively.

Note that for a fixed w <t , both W t |W <t = w <t and W t |W <t = w <t are Gaussian

Applying Lemma 10 and

Recall that W t−1 is the parameter at step t − 1 using S = (S, z) as dataset.

In this case, we can rewrite z as z n since it is the n-th data point of S. Note that SGLD satisfies the order-independent assumption, we can rewrite z as z i for all i ∈ [n].

Together with (5), (6), and using

x i , we can prove this theorem.

More generally, we give the following bound for SGLD.

The proof is similar to that of Theorem 9; the difference is that we need to bound the KL-divergence between two Gaussian mixtures instead of two Gaussians.

This proof is more technical and deferred to Appendix A.3.

Theorem 11.

Suppose that the loss function L is C-bounded and the objective function f is Llipschitz.

Assume that the following conditions hold:

Then, the following expected generalization error bound holds for T iterations of SGLD (1):

where g e (t)

] is the empirical squared gradient norm, and W t is the parameter at step t of SGLD.

Furthermore, based on essentially the same proof, we can obtain the following bound that depends on the population gradient norm:

The full proofs of the above results are postponed to Appendix A, and we provide some remarks about the new bounds.

Remark 12.

In fact, our proof establishes that the above upper bound holds for the two sequences W ≤T and W ≤T :

.

Hence, our bound holds for any sufficiently regular function over the parameter sequences:

.

In particular, our generalization error bound automatically extends to several variants of SGLD, such as outputting the average of the trajectory, the average of the suffix of certain length, or the exponential moving average.

Remark 13.

Inspired by Zhang et al. (2017a), we run both GLD ( Figure 1 ) and SGLD (Appendix C.2) to fit both normal data and randomly labelled data (see Appendix C for more experiment details).

As shown in Figure 1 and Figure 2 in Appendix C.2, larger random label portion p leads to both much higher generalization error and much larger generalization error bound.

Moreover, the shapes of the curves our bounds look quite similar to that of the generalization error curves.

In this section, we study the generalization error of Continuous Langevin Dynamics (CLD) with 2 regularization.

Throughout this section, we assume that the objective function over training set S is defined as F (w, S) = F 0 (w, S) + λ 2 w 2 2 , and moreover, the following assumption holds.

Assumption 14.

The loss function L and the original objective F 0 are C-bounded.

Moreover, F 0 is differentiable and L-lipschitz.

The Continuous Langevin Dynamics is defined by the following SDE:

where (B t ) t≥0 is the standard Brownian motion on R d and the initial distribution µ 0 is the centered Gaussian distribution in R d with covariance

We show that the generalization error of CLD is upper bounded by O e 4βC n −1 β/λ , which is independent of the training time T (Theorem 15).

Furthermore, as T goes to infinity, we have a tighter generalization error bound O βC 2 n −1 (Theorem 39 in Appendix B).

We also study the generalization of Gradient Langevin Dynamics (GLD), which is the discretization of CLD:

where ξ k is the standard Gaussian random vector in R d .

By leveraging a result developed in Raginsky et al. (2017), we show that, as Kη 2 tends to zero, GLD has the same generalization as CLD (see Theorems 15 and 39).

We first formally state our first main result in this section.

dw) has the following expected generalization error bound:

In addition, if L is M -smooth and non-negative, by setting λβ > 2, λ >

8M 2 ), GLD (running K iterations with the same µ 0 as CLD) has the expected generalization error bound:

where C 1 is a constant that only depends on M , λ, β, b, L and d.

The following lemma is crucial for establishing the above generalization bound for CLD.

In particular, we need to establish a Log-Sobolev inequality for µ t , the parameter distribution at time t, for every time step t > 0.

In contrast, most known LSIs only characterize the stationary distribution of the Markov process.

The proof of the lemma can be found in Appendix B. Lemma 16.

Under Assumption 14, let µ t be the probability measure of W t in CLD (with dµ 0 = 1 Z e −λβ w 2 2 dw).

Let ν be a probability measure that is absolutely continuous with respect to µ t .

Suppose dµ t = π t (w) dw and dν = γ(w) dw.

Then, it holds that

We sketch the proof of Theorem 15, and the complete proof is relegated to Appendix B.

Proof Sketch of Theorem 15 Suppose S and S are two neighboring datasets.

Let (W t ) t≥0 and (W t ) t≥0 be the process of CLD running on S and S , respectively.

Let γ t and π t be the pdf of W t and W t .

Let F S (w) denote F (w, S).

We have

The high level idea to prove this bound is very similar to that in Raginsky et al. (2017) .

We first observe that the (stationary) Gibbs distribution µ has a small generalization error.

Then, we bound the distance from µ t to µ. In our setting, we can use the Holley-Stroock perturbation lemma which allows us to bound the Logarithmic Sobolev constant, and we can thus bound the above distance easily.

In this paper, we prove new generalization bounds for a variety of noisy gradient-based methods.

Our current techniques can only handle continuous noises for which we can bound the KL-divergence.

One future direction is to study the discrete noise introduced in SGD (in this case the KL-divergence may not be well defined).

For either SGLD or CLD, if the noise level is small (i.e., β is large), it may take a long time for the diffusion process to reach the stable distribution.

Hence, another interesting future direction is to consider the local behavior and generalization of the diffusion process in finite time through the techniques developed in the studies of metastability (see e.g., Bovier et al. (2005) Lemma 17.

Under Assumption 6, for any prior distribution P not depending on the dataset S = (z 1 , . . .

, z n ), the generalization error is upper bounded by

where L(w) denotes the population loss L(w, D).

Let err train = ES Ew∼Q S L(w, S) and err test = ES Ew∼Q S L(w).

We can rewrite generalization error as err gen = err test − err train , where

and

Thus, we have

Now we are ready to prove Theorem 7, which we restate in the following.

Theorem 7 (Bayes-Stability).

Suppose the loss function L(w, z) is C-bounded and the learning algorithm is order-independent (Assumption 6), then for any prior distribution P not depending on S, the generalization error is bounded by both 2C E z 2KL(P, Q z ) and 2C E z 2KL(Q z , P ) .

Proof By Lemma 17,

The other bound follows from a similar argument.

Now we turn to the proof of Theorem 11.

The following lemma allows us to reduce the proof of algorithmic stability to the analysis of a single update step.

Proof By the chain rule of the KL-divergence,

The lemma follows from a summation over t = 1, . . .

, T .

The following lemma (see e.g., (Duchi, 2007, Section 9)) gives a closed-form formula for the KLdivergence between two Gaussian distributions.

The following lemma (Topsoe, 2000, Theorem 3) helps us to upper bound the KL-divergence.

Definition 19.

Let P and Q be two probability distributions on R d .

The directional triangular discrimination from P to Q is defined as

where

Lemma 20.

For any two probability distributions P and Q on R d ,

Recall that G is the set of all possible mini-batches.

G n = {B ∈ G : n ∈ B} denotes the collection of mini-batches that contain n, while G n = G \ G n .

diam(A) = sup x,y∈A x − y denotes the diameter of set A. The following technical lemma upper bounds the KL-divergence between two Gaussian mixtures induced by sampling a mini-batch from neighbouring datasets.

Lemma 21.

Suppose that batch size b ≤ n/2.

{µ B : B ∈ G} and {µ B : B ∈ G} are two collections of points in R d labeled by mini-batches of size b that satisfy the following conditions for some constant β ∈ [0, σ]:

B∈G p µ B ,σ and P = 1 |G| B∈G p µ B ,σ be two mixture distributions over all mini-batches.

Then,

Proof of Lemma 21 By Lemma 20, KL(P, P ) is bounded by

The numerator of the above integrand is upper bounded by

while the denominator can be lower bounded as follows:

which implies, by the convexity of 1/x, that

Inequalities (9) and (10) together imply

Now we bound the right-hand side of (11) for fixed A and B. By applying a translation and a rotation, we can assume without loss of generality that µ A = 0, and the last d − 2 coordinates of µ B and µ B are all zero.

Note that the integral is unchanged when we project the space to the twodimensional subspace corresponding to the first two coordinates.

Thus, it suffices to prove a bound for d = 2.

We rewrite (11) as

Let I be the integral in the right-hand side of (12).

Note that 2 , we make two claims which we will prove later:

, φ(y, δ) is non-increasing in δ.

The above claims imply that:

1.

For any r ∈ 0,

2.

For any r ∈ Plugging the above into (12) gives

We conclude that

n 2 σ 2 .

Finally, we prove the two claims used above:

, we have e (b)

2 e −δ(δ+2y) (δ + y) < 0 for y ≥ 0, we conclude that for any y ≥ 1/ √ 2:

Recall that SGLD on dataset S is defined as

Here γ t is the step size.

B t = {i 1 , . . . , i b } is a subset of {1, . . .

, n} of size b, and S Bt = (z i1 , . . . , z i b ) is the mini-batch indexed by B t .

Recall that F (w, S) denotes

.

We restate and prove Theorem 11 in the following.

Theorem 11.

Suppose that the loss function L is C-bounded and the objective function F is Llipschitz.

Assume that the following conditions hold:

Then, the following expected generalization error bound holds for T iterations of SGLD (1):

where g e (t) = Ew∼W t−1 [

] is the empirical squared gradient norm, and W t is the parameter at step t of SGLD.

Proof By Theorem 7, we have

for any prior distribution P .

In particular, we define the prior as P (w) = E S∼D n−1 [P S (w)], where P S (w) = Q (S,0) .

By the convexity of KL-divergence,

Fix a data point z ∈ Z. Let (W t ) t≥0 and (W t ) t≥0 be the training process of SGLD for S = (S, z) and S = (S, 0), respectively.

Fix a time step t and w <t = (w 0 , . . . , w t−1 ).

Let P t and P t denote the distribution of W t and W t conditioned on W <t = w <t and W <t = w <t , respectively.

By the definition of SGLD, we have

, and p µ denotes the Gaussian distribution

for B ∈ G n and µ B = µ B for B ∈ G n .

By applying Lemma 21 with β = γt ∇F (wt−1,z) 2 b and σ = σ t ,

By Lemma 10,

which implies that

Together with (13) and (14), we have

Since SGLD is order-independent, we can replace ∇F (w, z n ) with ∇F (w, z i ) for any i ∈ [n] in the right-hand side of the above bound.

Our theorem then follows from the concavity of √ x. Furthermore, if we bound KL(P, Q z ) instead of KL(Q z , P ) in the above proof, we obtain the following bound that depends on the population squared gradient norm:

We can extend the generalization bounds in previous sections, which require the noise to be Gaussian, to other general noises, namely the family of log-lipschitz noises.

Definition 22 (Log-Lipschitz Noises).

A probability distribution on R d with density p is L-loglipschitz if and only if ∇ ln p(w) ≤ L holds for any w ∈ R d .

A random variable ζ is called an L-log-lipschitz noise if and only if it is drawn from an L-log-lipschitz distribution.

The analog of SGLD, noisy momentum method (Definition 24), and noisy NAG (Definition 25) can be naturally defined by replacing the Gaussian noise ζ t at each iteration with an independent L-log-lipschitz noise in the definition.

The following lemma is an analog of Lemma 21 under L-log-lipschitz noises.

Recall that G denotes a collection of mini-batches of size b. Lemma 23 readily implies the analogs of Theorems 11, 26 and 27 under more general noise distributions.

Lemma 23.

Suppose that batch size b ≤ n/2 and N is an L noise -log-lipschitz distribution on R d .

{µ B : B ∈ G} and {µ B : B ∈ G} are two collections of points in R d that satisfy the following conditions for some constant β ∈ 0, 1 Lnoise :

For µ ∈ R d , let p µ denote the distribution of ζ +µ when ζ is drawn from N .

Let P = 1 |G| B∈G p µ B and P = 1 |G| B∈G p µ B be mixture distributions over all mini-batches.

Then,

for some constant C 0 that only depends on L noise .

Following the same argument as in the proof of Lemma 21, we have

where

Fixed A ∈ G n and B ∈ G n .

Let p noise denote the density of the noise distribution N .

Since µ A − µ B ≤ 1 and p noise is L noise -log-lipschitz, we have

Similarly, since µ B − µ B ≤ β, we have

Then, it follows from βL noise ≤ 1 that

Therefore, the integral on the righthand side of (16) can be upper bounded as follows:

Plugging the above inequality into (15) and (16) gives

and

We adopt the formulation of Classical Momentum and Nesterov's Accelerated Gradient (NAG) methods in Sutskever et al. (2013) and consider the noisy versions of them.

Definition 24 (Noisy Momentum Method).

Noisy Momentum Method on objective function F (w, z) and dataset S is defined as

Definition 25 (Noisy Nesterovs Accelerated Gradient).

Noisy Nesterovs Accelerated Gradient (NAG) on objective function F (w, z) and dataset S is defined as

In both definitions, γ t is the step size, mini-batch B t is drawn uniformly from G, ζ t is a Gaussian noise drawn from N (0,

, and η ∈ [0, 1] is the momentum coefficient.

Theorem 26.

Under the same assumptions on the loss function, objective function, batch size and learning rate as in Theorem 11, the generalization bounds in Theorem 11 still hold for noisy momentum method and noisy NAG.

For any time step t and w <t = (w 0 , w 1 , ..., w t−1 ), let P t and P t denote the distribution of W t and W t conditioned on W <t = w <t and W <t = w <t , respectively.

By definition, we have

If t = 1, for both noisy momentum method and noisy NAG, we have µ B = w t−1 − γ t ∇ w F (w t−1 , S B ), µ B = w t−1 − γ t ∇ w F (w t−1 , S B ).

For t > 1, if noisy momentum method is used, we have µ B = w t−1 + η(w t−1 − w t−2 ) − γ t ∇ w F (w t−1 , S B ),

.

Similarly, the following holds under noisy NAG:

In either case, it can be verified that the conditions of Lemma 21 hold for β = 2γtL b and σ = σ t .

The rest of the proof is the same as the proof of Theorem 11.

In the Entropy-SGD algorithm due to Chaudhari et al. (2017) , instead of directly optimizing the original objective F (w), we minimize the negative local entropy defined as follows:

Intuitively, a wider local minimum has a lower loss (i.e., −E(w, γ)) than sharper local minima.

See Chaudhari et al. (2017) for more details.

The Entropy-SGD algorithm invokes standard SGD to minimize the negative local entropy.

However, the gradient of negative local entropy

is hard to compute.

Thus, the algorithm uses exponential averaging to estimate the gradient in the SGLD loop; see Algorithm 1 for more details.

We have the following generalization bound for Entropy-SGD.

Algorithm 1: Entropy-SGD Input: Training set S = (z 1 , .., z n ) and loss function g(w, z).

Hyper-parameters: Scope γ, SGD learning rate η, SGLD step size η and batch size b.

Theorem 27.

Suppose that the loss function L is C-bounded and the objective function F is Llipschitz.

If batch size b ≤ n/2 and √ η ≤ ε/(20L), the following expected generalization error bound holds for Entropy-SGD:

where

] is the empirical squared gradient norm, and W t,k denotes the training process with respect to S.

Since g e (t, k) is at most L 2 , it further implies the generalization error of Entropy-SGD is bounded

Proof of Theorem 27 Define the history before time step (t, k) as follows:

Since µ is only determined by W , we only need to focus on W .

This proof is similar to the proof of Theorem 11.

By setting P = E S [Q (S,0) ].

Suppose S = (S, z) and S = (S, 0) are fixed, let W and W denote their training process, respectively.

Considering the following 3 cases:

1.

W t,0 ← W t−1,K+1 : In this case, for a fixed w ≤(t−1,K+1) , we have

In this case, fix a w ≤(t,k) , applying Lemma 21 gives

In this case, for a fixed w ≤(t,K) , we have

By applying Lemma 10, we have

and Where g e (t, k) is the empirical squared gradient norm of the k-th SGLD iteration in the t-th SGD iteration, respectively.

The rest of the proof is the same as the proof of Theorem 11.

The continuous version of the noisy gradient descent method is the Langevin dynamics, described by the following stochastic differential equation:

where B t is the standard Brownian motion.

To analyze the above Langevin dynamics, we need some preliminary knowledge about Log-Sobolev inequalities.

Let p t (w, y) denote the probability density function (i.e., probability kernel) describing the distribution of W t starting from w. For a given SDE such as (20), we can define the associated diffusion semigroup P: Definition 28 (Diffusion Semigroup). (see e.g., (Bakry et al., 2013, p. 39) ) Given a stochastic differential equation (SDE), the associated diffusion semigroup P = (P t ) t≥0 is a family of operators that satisfy for every t ≥ 0, P t is a linear operator sending any real-valued bounded measurable function f on R d to

The semigroup property P t+s = P t • P s holds for every t, s ≥ 0.

Another useful property of P t is that it maps a nonnegative function to a nonnegative function.

The carré du champ operator Γ of this diffusion semigroup (w.r.t (20)) is (Bakry et al., 2013, p. 42 )

We use the shorthand notation Γ(f ) = Γ(f, f ) = β −1 ∇f 2 2 , and define (with the convention that 0 log 0 = 0)

Definition 29 (Logarithmic Sobolev Inequality). (see e.g., (Bakry et al., 2013, p. 237) ) A probability measure µ is said to satisfy a logarithmic Sobolev inequality LS(α) (with respect to Γ), if for all functions f :

D(E) is the set of functions f ∈ L 2 (µ) for which the quantity

dµ has a finite (decreasing) limit as t decreases to 0.

A well-known Logarithmic Sobolev Inequality is the following result for Gaussian measures.

Lemma 30 (Logarithmic Sobolev Inequality for Gaussian measure). (Bakry et al., 2013, p. 258) Let µ be the centered Gaussian measure on R d with covariance matrix σ 2 I d .

Then µ satisfies the following LSI:

Lemma 30 states that the centered Gaussian measure with covariance matrix σ 2 I d satisfies LS(βσ 2 ) (with respect to Γ), where Γ = β −1 ∇f, ∇g is the carré du champ operator of the diffusion semigroup defined above.

Before proving our results, we need some known results from Markov diffusion process.

It is well known that the invariant measure (Bakry et al., 2013, p. 10) of the above CLD is the Gibbs measure dµ = 1 Zµ exp(−βF (w)) dw (Menz et al., 2014, (1.3) ).

In other words, µ satisfies R d P t f dµ = R d f dµ for every bounded positive measurable function f , where P t is the Markov semigroup in Definition 28.

The following lemma by Holley and Stroock Holley & Stroock (1987) (see also (Bakry et al., 2013, p. 240) ) allows us to determine the Logarithmic Sobolev constant of the invariant measure µ.

Lemma 31 (Bounded perturbation).

Assume that the probability measure ν satisfies LS(α) (with respect to Γ).

Let µ be a probability measure such that 1/b ≤ dµ/dν ≤ b for some constant b > 1.

Then µ satisfies LS(b 2 α) (with respect to Γ).

In fact, Lemma 31 is a simple consequence of the following variational formula in the special case that φ(x) = x log x, which we will also need in our proof:

Lemma 32 (Variational formula). (see .g., (Bakry et al., 2013, p. 240) ) Let φ : I → R on some open interval I ⊂ R be convex of class C 2 .

For every (bounded or suitably integrable) measurable function f : R d → R with values in I,

It is worth noting the integrand of the right-hand side is nonnegative due to the convexity of φ.

Recall that F S (w) = F (w, S) := F 0 (w, S) + λ w 2 2 /2 is the sum of the empirical original objective F 0 (w, S) and 2 regularization.

Let dµ = Lemma 33.

Under Assumption 14, let Γ(f, g) = β −1 ∇f, ∇g be the carré du champ operator of the diffusion semigroup associated to CLD, and µ be the invariant measure of the SDE.

Then, µ satisfies LS(e 4βC /λ) with respect to Γ.

Let µ t be the probability measure of W t .

By definition of P t , for any real-valued bounded measurable function f on R d and any s, t ≥ 0,

In particular, if the invariant measure µ = µ ∞ exists, we have

The following lemma is crucial for establishing the first generalization bound for CLD.

In fact, we establish a Log-Sobolev inequality for µ t , the parameter distribution at time t, for any time t > 0.

Note that our choice of the initial distribution µ 0 is important for the proof.

Lemma 34.

Under Assumption 14, let µ t be the probability measure of W t in (CLD) with initial

dw.

Let Γ be the carré du champ operator of diffusion semigroup associated to (CLD).

Then, for any f :

Proof Let µ be the invariant measure of CLD.

By Lemma 33 and Definition 29,

By applying Lemma 32 with φ(x) = x log x, we rewrite the left-hand side as

where the last equation holds by the definition of invariant measure P t f dµ = f dµ. Thus, we have

Let µ t be the probability measure of W t .

Lemma 32 and (22) together imply that

Since dµ dµ0 ≤ exp(2βC) and µ is the invariant measure, we conclude that

Lemma 16.

Under Assumption 14, let µ t be the probability measure of W t in CLD (with dµ 0 = 1 Z e −λβ w 2 2 dw).

Let ν be a probability measure that is absolutely continuous with respect to µ t .

Suppose dµ t = π t (w) dw and dν = γ(w) dw.

Then it holds that:

Proof Let f (w) = γ(w)/π t (w), by Lemma 34 and

We can see that the left-hand side is equal to KL(γ, π t ) 6 , and the right-hand side is equal to

This concludes the proof. .

We can rewrite F S (w) = 1 n n i=1 h(w, z i ).

Define µ S,k and ν S,t as the probability measure of W k (in GLD) and W t (in CLD), respectively.

Raginsky et al. (2017) provided a bound of KL(µ S,k , ν S,ηK ) under Assumption 35.

This bound enables us to derive a generalization error bound for the discrete GLD from the bound for the continuous CLD.

We use the assumption from Raginsky et al. (2017) .

Their work considers the following SGLD:

Where g S (w k ) is a conditionally unbiased estimate of the gradient ∇F S (w k ).

In our GLD setting,

1.

The function h takes non-negative real values, and there exist constants A, B ≥ 0, such that

2.

For each z ∈ Z, the function h(·, z) is M -smooth: for some M > 0,

3.

For each z ∈ Z, the function h(·, z) is (m, b)-dissipative: for some m > 0 and b ≥ 0,

4.

There exists a constant δ ∈ [0, 1), such that, for each S ∈ Z n ,

5.

The probability law µ 0 of the initial hypothesis W 0 has a bounded and strictly positive density p 0 with respect to the Lebesgue measure on R d , and dw) has the following expected generalization error bound:

6 Indeed,

8M 2 ), the GLD (running K iterations with the same µ 0 as CLD) has the expected generalization error bound:

where C 1 is a constant that only depends on M , λ, β, b, L and d.

We apply the uniform stability framework.

Suppose S and S are two neighboring datasets that differ on exactly one data point.

Let (W t ) t≥0 and (W t ) t≥0 be the process of CLD running on S and S , respectively.

Let γ t and π t be the pdf of W t and W t .

We have

According to Fokker-Planck equation (see Risken (1996) ) for CLD, we know that

It follows that

(integration by parts) and

(integration by parts)

Together with (33), we have

Solving this differential inequality gives

By Pinsker's inequality, we can finally see that

By Lemma 4, the generalization error of CLD is bounded by the right-hand side of the above inequality.

Now, we prove the second part of the theorem.

Let (W k ) k≥0 and (W k ) k≥0 be the (discrete) GLD processes training on S and S , respectively.

Then for any z ∈ Z:

Since λβ > 2 and λ >

and

From (35), we have

Combining (36), (37) and (38), we have

By Definition 3, GLD is n -uniformly stable.

Applying Lemma 4 gives the generalization bound of GLD.

Lemma 37 (Exponential decay in entropy). (Bakry et al., 2013, Theorem 5.2 .1) The logarithmic Sobolev inequality LS(α) for the probability measure µ is equivalent to saying that for every positive function ρ in L 1 (µ) (with finite entropy),

for every t ≥ 0.

The following Lemma shows that P t ( dµ0 dµ ) = µ t in our diffusion process.

Lemma 38.

Let P denote the diffusion semigroup of CLD.

Let µ denote the invariant measure of P and let µ t denote the probability measure of W t .

Then P t ( dµ0 dµ ) = µ t .

Proof Let dµ = µ(x) dx and dµ t = µ t (x) dx.

As shown in (Pavliotis, 2014, page 118), our diffusion process (Smoluchowski dynamics) is reversible, which means µ(x)p t (x, y) = µ(y)p t (y, x).

Thus for any g(x), we have

Since g is arbitrary, P t ( dµ0 dµ ) and µ t must be the same.

dw) has the following expected generalization error bound:

In addition, if F 0 is also M -smooth and non-negative, by setting λβ > 2, λ > 1 2 and η ∈ [0, 1 ∧ 2λ−1 8M 2 ), the GLD process (running K iterations with the same µ 0 as CLD) has the expected generalization error bound:

where C 1 is a constant that only depends on M , λ, β, b, L and d.

Proof of Theorem 39 Suppose S and S are two datasets that differ on exactly one data point.

Let (W t ) t≥0 and (W t ) t≥0 be their processes, respectively.

Let dµ t = π t (w) dw and dµ t = π t (w) dw be the probability measure of W t and W t , respectively.

The invariant measure of CLD for S and S are denoted as µ and µ , respectively.

Recall that dµ = 1 Z µ e −βF S (w) dw, dµ = 1 Z µ e −βF S (w) dw.

The total variation distance of µ and µ is

Since Zµ Z µ exp(−β(F S (w) − F S (w))) ∈ e

Since µ and µ satisfy LS(e 4βC/λ ) (Lemma 33), applying Lemma 37 with ρ = dµ0 dµ and ρ = dµ 0 dµ and Lemma 38 yields:

KL(µ t , µ) ≤ exp −2λt e 4βC KL(µ 0 , µ), KL(µ t , µ ) ≤ exp −2λt e 4βC KL(µ 0 , µ ).

Since KL(µ 0 , µ) and KL(µ 0 , µ ) are upper bounded by 2βC, Pinsker's inequality implies that TV(µ t , µ) and TV(µ t , µ ) are upper bounded by exp −2λt e 4βC βC. Combining with (40) and note that TV(µ t , µ t ) ≤ TV(µ t , µ) + TV(µ, µ ) + TV(µ t , µ ), we have

−2λt e 4βC βC + 8βC 2 n .

By Lemma 4, the generalization error of CLD is bounded by the right-hand side.

The proof for GLD proceeds in the same way as the second part of the proof of Theorem 15.

We first present the general setup of our experiments: • Small AlexNet: k is the kernel size, d is the depth of a convolution layer, fc(m) is the fullyconnected layer that has m neurons.

The ReLU activation are used in the first 6 layers.

• MLP: The MLP used in our experiment has 3 hidden layers, each having width 512.

We also use ReLU as the activation function in MLP.

Objective function: For a data point z = (x, y) in MNIST, the objective function is

Random labels: Suppose the dataset contains n datapoint, and the corruption portion is p. We randomly select n · p data points, and replace their labels with random labels, as in Zhang et al. (2017a) .

The result of this experiment (see Figure 1 ) is discussed in Section 3, Remark 13.

Here we present our implementation details.

We repeat our experiment 5 times.

At every individual run, we first randomly sample 10000 data points from the complete MNIST training data.

The initial learning rate γ 0 = 0.003.

It decays 0.995 after every 60 steps, and it stops decaying when it is lower than 0.0005.

During the training, we keep σ t = 0.2 √ 2γ t .

Recall that the empirical squared gradient norm g e (t) in our bound (Theorem 9)

Under review as a conference paper at ICLR 2020

Since it is time-consuming to compute the exact g e (t), in our experiment, we use an unbiased estimation instead.

At every step, we randomly sample a minibatch B with batch size 200 from the training data, and use 1 200 i∈B ∇f (W t−1 , z i ) 2 as g e (t) to compute our bound in Figure 1 .

The estimation of g e (t) at every step t is shown in Figure 1(d) .

Since g e (t) is not very stable, in our figure, we plot its moving average over a window of size 100 to make the curve smoother (i.e., g avg (t) = 1 100 t+100 τ =t g e (τ )).

In this subsection, we present some experiment results for running SGLD on both MNIST and CIFAR10 datasets, to demonstrate that our bound (see Theorem 11), in particular the sum of the empirical squared gradient norms along the training path, can distinguish normal dataset from dataset that contains random labels.

We note that in our experiments, the learn rate we choose is larger than that is required by the condition of Theorem 11.

Due to the (non-optimal) constant in our bound, the bound is currently greater than 1, and hence we ignore the numbers on the y-axis.

However, again, the curves of our bounds look quite similar to the generalization curves (see Figure 2 ).

This indicates that the sum of squared empirical gradient norms is highly related to the generalization performance, and we believe by further optimizing the constants in our bound, we can achieve a generalization bound that is close to the real generalization error.

<|TLDR|>

@highlight

We give some generalization error bounds of noisy gradient methods such as SGLD, Langevin dynamics, noisy momentum and so forth.