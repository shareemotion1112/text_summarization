Mini-batch stochastic gradient descent (SGD) is state of the art in large scale distributed training.

The scheme can reach a linear speed-up with respect to the number of workers, but this is rarely seen in practice as the scheme often suffers from large network delays and bandwidth limits.

To overcome this communication bottleneck recent works propose to reduce the communication frequency.

An algorithm of this type is local SGD that runs SGD independently in parallel on different workers and averages the sequences only once in a while.

This scheme shows promising results in practice, but eluded thorough theoretical analysis.

We prove concise convergence rates for local SGD on convex problems and show that it converges at the same rate as mini-batch SGD in terms of number of evaluated gradients, that is, the scheme achieves linear speed-up in the number of workers and mini-batch size.

The number of  communication rounds can be reduced up to a factor of T^{1/2}---where T denotes the number of total steps---compared to mini-batch SGD.

This also holds for asynchronous implementations.



Local SGD can also be used for large scale training of deep learning models.

The results shown here aim serving as a guideline to further explore the theoretical and practical aspects of local SGD in these applications.

Stochastic Gradient Descent (SGD) BID29 consists of iterations of the form DISPLAYFORM0 for iterates (weights) x t , x t+1 ∈ R d , stepsize (learning rate) η t > 0, and stochastic gradient g t ∈ R d with the property E g t = ∇f (x t ), for a loss function f : R d → R. This scheme can easily be parallelized by replacing g t in (1) by an average of stochastic gradients that are independently computed in parallel on separate workers (parallel SGD).

This simple scheme has a major drawback: in each iteration the results of the computations on the workers have to be shared with the other workers to compute the next iterate x t+1 .

Communication has been reported to be a major bottleneck for many large scale deep learning applications, see e.g. BID32 BID17 .

Mini-batch parallel SGD addresses this issue by increasing the compute to communication ratio.

Each worker computes a mini-batch of size b ≥ 1 before communication.

This scheme is implemented in state-of-the-art distributed deep learning frameworks BID0 BID26 BID31 .

Recent work in BID43 BID10 explores various limitations of this approach, as in general it is reported that performance degrades for too large mini-batch sizes BID13 BID18 BID42 .In this work we follow an orthogonal approach, still with the goal to increase the compute to communication ratio: Instead of increasing the mini-batch size, we reduce the communication frequency.

Rather than keeping the sequences on different machines in sync, we allow them to evolve locally on each machine, independent from each other, and only average the sequences once in a while (local SGD).

Such strategies have been explored widely in the literature, under various names.

An extreme instance of this concept is one-shot SGD (McDonald et al., 2009; BID53 where the local sequences are only exchanged once, after the local runs have converged.

Zhang et al. (2013) show statistical convergence (see also BID33 BID9 BID12 ), but the analysis restricts the algorithm to at most one pass over the data, which is in general not enough for the training error to converge.

More practical are schemes that perform more frequent averaging of the parallel sequences, as e.g. BID22 for perceptron training (iterative parameter mixing), see also BID6 , BID48 BID4 BID46 for the training of deep neural networks (model averaging) or in federated learning BID23 .The question of how often communication rounds need to be initiated has eluded a concise theoretical answer so far.

Whilst there is practical evidence, the theory does not even resolve the question whether averaging helps when optimizing convex functions.

Concretely, whether running local SGD on K workers is K times faster than running just a single instance of SGD on one worker.

We fill this gap in the literature and provide a concise convergence analysis of local SGD.

We show that averaging helps.

Frequent synchronization of K local sequences increases the convergence rate by a factor of K, i.e. a linear speedup can be attained.

Thus, local SGD is as efficient as parallel mini-batch SGD in terms of computation, but the communication cost can be drastically reduced.

We consider finite-sum convex optimization problems f : DISPLAYFORM0 where f is L-smooth 2 and µ-strongly convex 3 .

We consider K parallel mini-batch SGD sequences with mini-batch size b that are synchronized (by averaging) after at most every H iterations.

For appropriate chosen stepsizes and an averaged iteratex T after T steps (for T sufficiently large, see Section 3 below for the precise statement of the convergence result with bias and variance terms) and synchronization delay H = O( T /(Kb)) we show convergence DISPLAYFORM1 with second moment bound G 2 ≥ E ∇f i (x) 2 .

Thus, we see that compared to parallel minibatch SGD the communication rounds can be reduced by a factor H = O( T /(Kb)) without hampering the asymptotic convergence.

Equation (3) shows perfect linear speedup in terms of computation, but with much less communication that mini-batch SGD.

The resulting speedup when taking communication cost into account is illustrated in FIG0 (see also Section D below).

Under the assumption that (3) is tight, one has thus now two strategies to improve the compute to communication ratio (denoted by ρ): (i) either to increase the mini-batch size b or (ii) to increase the communication interval H. Both strategies give the same improvement when b and H are small (linear speedup).

Like mini-batch SGD that faces some limitations for b 1 (as discussed in e.g. BID7 BID18 BID42 ), the parameter H cannot be chosen too large in local SGD.

We give some pratical guidelines in Section 4.Our proof is simple and straightforward, and we imagine that-with slight modifications of the proof-the technique can also be used to analyze other variants of SGD that evolve sequences on 1 On convex functions, the average of the K local solutions can of course only decrease the objective value, but convexity does not imply that the averaged point is K times better.

DISPLAYFORM2 different worker that are not perfectly synchronized.

Although we do not yet provide convergence guarantees for the non-convex setting, we feel that the positive results presented here will spark further investigation of local SGD for this important application (see e.g. BID44 ).

A parallel line of work reduces the communication cost by compressing the stochastic gradients before communication.

For instance, by limiting the number of bits in the floating point representation BID11 BID24 BID30 , or random quantization BID40 .

The ZipML framework applies this technique also to the data .

Sparsification methods reduce the number of non-zero entries in the stochastic gradient BID39 .

A very aggressive-and promising-sparsification method is to keep only very few coordinates of the stochastic gradient by considering only the coordinates with the largest magnitudes BID32 BID36 BID8 BID2 BID37 BID17 BID35 .Allowing asynchronous updates provides an alternative solution to disguise the communication overhead to a certain amount BID25 BID30 BID15 , though alternative strategies might be better when high accuracy is desired .

The analysis of BID1 shows that asynchronous SGD on convex functions can tolerated delays up to O( T /K), which is identical to the maximal length of the local sequences in local SGD.

Asynchronous SGD converges also for larger delays (see also BID52 ) but without linear speedup, a similar statement holds for local SGD (see discussion in Section 3).

The current frameworks for the analysis of asynchronous SGD do not cover local SGD.

A fundamental difference is that asynchronous SGD maintains a (almost) synchronized sequence and gradients are computed with respect this unique sequence (but just applied with delays), whereas each worker in local SGD evolves a different sequence and computes gradient with respect those iterates.

is different to local SGD, as it uses the average of the iterates only to guide the local sequences but does not perform a hard reset after averaging.

Among the first theoretical studies of local SGD in the non-convex setting are BID6 Zhou & Cong, 2018) that did not establish a speedup, in contrast to two more recent analyses BID44 BID38 .

BID44 show linear speedup of local SGD on non-convex functions for H = O(T 1/4 K −3/4 ), which is more restrictive than the constraint on H in the convex setting.

BID16 study empirically hierarchical variants of local SGD.Local SGD with averaging in every step, i.e. H = 1, is identical to mini-batch SGD.

BID7 show that batch sizes b = T δ , for δ ∈ (0, 1 2 ) are asymptotically optimal for mini-batch SGD, however they also note that this asymptotic bound might be crude for practical purposes.

Similar considerations might also apply to the asymptotic upper bounds on the communication frequency H derived here.

Local SGD with averaging only at the end, i.e. H = T , is identical to one-shot SGD.

BID12 give concise speedup results in terms of bias and variance for one-shot SGD with constant stepsizes for the optimization of quadratic least squares problems.

In contrast, our upper bounds become loose when H → T and our results do not cover one-shot SGD.Recently, BID41 provided a lower bound for parallel stochastic optimization (in the convex setting, and not for strongly convex functions as considered here).

The bound is not known to be tight for local SGD.

We formally introduce local SGD in Section 2 and sketch the convergence proof in Section 3.

In Section 4 show numerical results to illustrate the result.

We analyze asynchronous local SGD in Section 5.

The proof of the technical results, further discussion about the experimental setup and implementation guidelines are deferred to the appendix.

DISPLAYFORM0 if t + 1 ∈ I T then 6: DISPLAYFORM1 else 8: DISPLAYFORM2 end if 10:end parallel for 11: end for

The algorithm local SGD (depicted in Algorithm 1) generates in parallel K sequences {x DISPLAYFORM0 Here K denotes the level of parallelization, i.e. the number of distinct parallel sequences and T the number of steps (i.e. the total number of stochastic gradient evaluations is T K).

DISPLAYFORM1 with T ∈ I T denote a set of synchronization indices.

Then local SGD evolves the sequences {x DISPLAYFORM2 in the following way: DISPLAYFORM3 where indices i DISPLAYFORM4 [n] and {η t } t≥0 denotes a sequence of stepsizes.

If I T = [T ] then the synchronization of the sequences is performed every iteration.

In this case, (4) amounts to parallel or mini-batch SGD with mini-batch size K.4 On the other extreme, if I T = {T }, the synchronization only happens at the end, which is known as one-shot averaging.

In order to measure the longest interval between subsequent synchronization steps, we introduce the gap of a set of integers.

Definition 2.1 (gap).

The gap of a set P := {p 0 , . . . , p t } of t + 1 integers, p i ≤ p i+1 for i = 0, . . . , t − 1, is defined as gap(P) := max i=1,...,t (p i − p i−1 ).

Before jumping to the convergence result, we first discuss an important observation.

Parallel (mini-batch) SGD.

For carefully chosen stepsizes η t , SGD converges at rate O σ 2 T 5 on strongly convex and smooth functions f , where DISPLAYFORM0 is an upper bound on the variance, see for instance BID50 .

By averaging K stochastic gradients-such as in parallel SGD-the variance decreases by a factor of K, and we conclude that parallel SGD converges at a rate O Towards local SGD.

For local SGD such a simple argument is elusive.

For instance, just capitalizing the convexity of the objective function f is not enough: this will show that the averaged iterate of K independent SGD sequences converges at rate O σ 2 T , i.e. no speedup can be shown in this way.

This indicates that one has to show that local SGD decreases the variance σ 2 instead, similar as in parallel SGD.

Suppose the different sequences x k t evolve close to each other.

Then it is reasonable to assume that averaging the stochastic gradients DISPLAYFORM1 can still yield a reduction in the variance by a factor of K-similar as in parallel SGD.

Indeed, we will make this statement precise in the proof below.

Theorem 2.2.

Let f be L-smooth and µ-strongly convex, DISPLAYFORM0 are generated according to (4) with gap(I T ) ≤ H and for stepsizes η t = 4 µ(a+t) with shift parameter a > max{16κ, H}, for DISPLAYFORM1 DISPLAYFORM2 We were not especially careful to optimize the constants (and the lower order terms) in (5), so we now state the asymptotic result.

Corollary 2.3.

Letx T be as defined as in Theorem 2.2, for parameter a = max{16κ, H}. Then DISPLAYFORM3 For the last estimate we used E µ x 0 − x ≤ 2G for µ-strongly convex f , as derived in (Rakhlin et al., 2012, Lemma 2) .

Remark 2.4 (Mini-batch local SGD).

So far, we assumed that each worker only computes a single stochastic gradient.

In mini-batch local SGD, each worker computes a mini-batch of size b in each iteration.

This reduces the variance by a factor of b, and thus Theorem (2.2) gives the convergence rate of mini-batch local SGD when σ 2 is replaced by DISPLAYFORM4 We now state some consequences of equation FORMULA17 .

For the ease of the exposition we omit the dependency on L, µ, σ 2 and G 2 below, but depict the dependency on the local mini-batch size b.

Convergence rate.

For T large enough and assuming σ > 0, the very first term is dominating in (6) and local SGD converges at rate O(1/(KT b)).

That is, local SGD achieves a linear speedup in both, the number of workers K and the mini-batch size b. Global synchronization steps.

It needs to hold H = O( T /(Kb)) to get the linear speedup.

This yields a reduction of the number of communication rounds by a factor O( T /(Kb)) compared to parallel mini-batch SGD without hurting the convergence rate.

Extreme Cases.

We have not optimized the result for extreme settings of H, K, L or σ.

For instance, we do not recover convergence for the one-shot averaging, i.e. the setting H = T (though convergence for H = o(T ), but at a lower rate).

Unknown Time Horizon/Adaptive Communication Frequency BID46 empirically observe that more frequent communication at the beginning of the optimization can help to get faster time-to-accuracy (see also BID16 ).

Indeed, when the number of total iterations T is not known beforehand (as it e.g. depends on the target accuracy, cf.

(6) and also Section 4 below), then increasing the communication frequency seems to be a good strategy to keep the communication low, why still respecting the constraint H = O( T /(Kb)) for all T .

DISPLAYFORM5 .

It will be useful to define DISPLAYFORM6 Observex t+1 =x t − η t g t and E g t =ḡ t .Now the proof proceeds as follows: we show (i) that the virtual sequence {x t } t≥0 almost behaves like mini-batch SGD with batch size K (Lemma 3.1 and 3.2), and (ii) the true iterates {x k t } t≥0,k∈[K] do not deviate much from the virtual sequence FIG6 .

These are the main ingredients in the proof.

To obtain the rate we exploit a technical lemma from BID35 .

Lemma 3.1.

Let {x t } t≥0 and {x t } t≥0 for k ∈ [K] be defined as in (4) and (7) and let f be L-smooth and µ-strongly convex and η t ≤ 1 4L .

Then DISPLAYFORM7 Bounding the variance.

From equation FORMULA21 it becomes clear that we should derive an upper bound on E g t −ḡ t 2 .

We will relate this to the variance σ 2 .

DISPLAYFORM8 Bounding the deviation.

Further, we need to bound DISPLAYFORM9 For this we impose a condition on I T and an additional condition on the stepsize η t .

Lemma 3.3.

If gap(I T ) ≤ H and sequence of decreasing positive stepsizes {η t } t≥0 satisfying η t ≤ 2η t+H for all t ≥ 0, then DISPLAYFORM10 where G 2 is a constant such that DISPLAYFORM11 Optimal Averaging.

Similar as in BID14 BID34 BID28 we define a suitable averaging scheme for the iterates {x t } t≥0 to get the optimal convergence rate.

In contrast to BID14 ) that use linearly increasing weights, we use quadratically increasing weights, as for instance BID34 BID35 .

Lemma 3.4 ((Stich et al., 2018)).

Let {a t } t≥0 , a t ≥ 0, {e t } t≥0 , e t ≥ 0 be sequences satisfying DISPLAYFORM12 DISPLAYFORM13 for w t = (a + t) 2 and S T := T −1 DISPLAYFORM14 Proof.

This is a reformulation of Lemma 3.3 in BID35 .

The proof of the theorem thus follows immediately from the four lemmas that we have presented, i.e. by Lemma 3.4 with e t := E(f (x t ) − f ) and constants A = (a) Theoretical speedup S(K) ( > 0, T small).

In this section we show some numerical experiments to illustrate the results of Theorem 2.2.Speedup.

When Algorithm 1 is implemented in a distributed setting, there are two components that determine the wall-clock time: (i) the total number of gradient computations, T K, and (ii) the total time spend for communication.

In each communication round 2(K − 1) vectors need to be exchanged, and there will be T /H communication rounds.

Typically, the communication is more expensive than a single gradient computation.

We will denote this ratio by a factor ρ ≥ 1 (in practice, ρ can be 10-100, or even larger on slow networks).

The parameter T depends on the desired accuracy > 0, and according to (6) we roughly have T ( , H, K) DISPLAYFORM0 Thus, the theoretical speedup S(K) of local SGD on K machines compared to SGD on one machine (H = 1, K = 1) is DISPLAYFORM1 Theoretical.

Examining (13), we see that (i) increasing H can reduce negative scaling effects due to parallelization (second bracket in the denominator of FORMULA0 ), and (ii) local SGD only shows linear scaling for 1 (i.e. T large enough, in agreement with the theory).

In FIG5 we depict S(K), once for = 0 in FIG5 , and for positive > 0 in FIG5 under the assumption ρ = 25.

We see that for = 0 the largest values of H give the best speedup, however, when only a few epochs need to be performed, then the optimal values of H change with the number of workers K. We also see that for a small number of workers H = 1 is never optimal.

If T is unknown, then these observations seem to indicate that the technique from BID46 , i.e. adaptively increasing H over time seems to be a good strategy to get the best choice of H when the time horizon is unknown.

Experimental.

We examine the practical speedup on a logistic regression problem, DISPLAYFORM2 2 , where a i ∈ R d and b i ∈ {−1, +1} are the data samples.

The regularization parameter is set to λ = 1/n.

We consider the w8a dataset BID27 (d = 300, n = 49749).

We initialize all runs with x 0 = 0 d and measure the number of iterations to reach the target accuracy .

We consider the target accuracy reached, when either the last iterate, the uniform average, the average with linear weights, or the average with quadratic weights (such as in Theorem 2.2) reaches the target accuracy.

By extensive grid search we determine for each configuration (H, K, B) the best stepsize from the set {min(32, cn t+1 ), 32c}, where c can take the values c = 2 i for i ∈ Z. For more details on the experimental setup refer Section D in the appendix.

We depict the results in FIG6 , again under the assumption ρ = 25.

1: Initialize variables DISPLAYFORM0 atomic aggregation of the updates 8: end for 12: end parallel for Conclusion.

The restriction on H imposed by theory is not severe for T → ∞. Thus, for training that either requires many passes over the data or that is performed only on a small cluster, large values of H are advisable.

However, for smaller T (few passes over the data), the O(1/ √ K) dependency shows significantly in the experiment.

This has to be taken into account when deploying the algorithm on a massively parallel system, for instance through the technique mentioned in BID46 .

DISPLAYFORM1

In this section we present asynchronous local SGD that does not require that the local sequences are synchronized.

This does not only reduce communication bottlenecks, but by using load-balancing techniques the algorithm can optimally be tuned to heterogeneous settings (slower workers do less computation between synchronization, and faster workers do more).

We will discuss this in more detail in Section C.

.

Similar as in Section 2 we introduce sets of synchronization indices, DISPLAYFORM0 Note that the sets do not have to be equal for different workers.

Each worker k evolves locally a sequence x k t in the following way: DISPLAYFORM1 wherex k t+1 denotes the state of the aggregated variable at the time when worker k reads the aggregated variable.

To be precise, we use the notation DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 Published as a conference paper at ICLR 2019Hence, for T large enough and (H + τ ) = O( T /K), asynchronous local SGD converges with rate O G 2 KT , the same rate as synchronous local SGD.

We prove convergence of synchronous and asynchronous local SGD and are the first to show that local SGD (for nontrivial values of H) attains theoretically linear speedup on strongly convex functions when parallelized among K workers.

We show that local SGD saves up to a factor of O(T 1/2 ) in global communication rounds compared to mini-batch SGD, while still converging at the same rate in terms of total stochastic gradient computations.

Deriving more concise convergence rates for local SGD could be an interesting future direction that could deepen our understanding of the scheme.

For instance one could aim for a more fine grained analysis in terms of bias and variance terms (similar as e.g. in BID7 BID12 ), relaxing the assumptions (here we relied on the bounded gradient assumption), or investigating the data dependence (e.g. by considering data-depentent measures like e.g. gradient diversity BID42 ).

There are also no apparent reasons that would limit the extension of the theory to non-convex objective functions; Lemma 3.3 does neither use the smoothness nor the strong convexity assumption, so this can be applied in the non-convex setting as well.

We feel that the positive results shown here can motivate and spark further research on non-convex problems.

Indeed, very recent work (Zhou & Cong, 2018; BID44 analyzes local SGD for non-convex optimization problems and shows convergence of SGD to a stationary point, though the restrictions on H are stronger than here.

The author thanks Jean-Baptiste Cordonnier, Tao Lin and Kumar Kshitij Patel for spotting various typos in the first versions of this manuscript, as well as Martin Jaggi for his support.hence we can continue in (28) and obtain DISPLAYFORM0 Finally, we can plug (30) back into (18).

By taking expectation we get DISPLAYFORM1 Proof of Lemma 3.2.

By definition of g t andḡ t we have DISPLAYFORM2 where we used Var( DISPLAYFORM3 Var(X k ) for independent random variables.

Proof of Lemma 3.3.

As the gap( DISPLAYFORM4 where we used η t ≤ η t0 for t ≥ t 0 and the assumption DISPLAYFORM5 Finally, the claim follows by the assumption on the stepsizes, DISPLAYFORM6

In this Section we prove Theorem 5.1.

The proof follows closely the proof presented in Section 3.

We again introduce the virtual sequencē DISPLAYFORM0 as before.

By the property T ∈ I k T for k ∈ K we know that all workers will have written their updates when the algorithm terminates.

This assumption is not very critical and could be relaxed, but it facilitates the (already quite heavy) notation in the proof.

As for synchronous local SGD, the weighted averages of the iterates (if needed), can be tracked on each worker locally by a recursive formula as explained above.

A more important aspect that we do not have discussed yet, is that Algorithm 2 allows for an easy procedure to balance the load in heterogeneous settings.

In our notation, we have always associated the local sequences {x k t } with a specific worker k. However, the computation of the sequences does not need to be tied to a specific worker.

Thus, a fast worker k that has advanced his local sequence too much already, can start computing updates for another sequence k = k, if worker k is lagged behind.

This was not possible in the synchronous model, as there all communications had to happen in sync.

We demonstrate this principle in TAB4 below for two workers.

Note that also the running averages can still be maintained.

We here state the precise procedure that was used to generate the figures in this report.

As briefly stated in Section 4 we examine empirically the speedup on a logistic regression problem, f (x) = 1 n n i=1 log(1 + exp(−b i a i x)) + λ 2 x 2 , where a i ∈ R d and b i ∈ {−1, +1} are the data samples.

The regularization parameter is set to λ = 1/n.

We consider the small scale w8a dataset BID27 (d = 300, n = 49749).For each run, we initialize x 0 = 0 d and measure the number of iterations 6 (and number of stochastic gradient evaluations) to reach the target accuracy ∈ {0.005, 0.0001}. As we prove convergence only for a special weighted sum of the iterates in Theorem 2.2 and not for standard criteria (last iterate or uniform average), we evaluate the function value for different weighted averages y t = 1 t i=0 wi t i=0 w i x t , and consider the accuracy reached when one of the averages satisfies f (y t ) − f ≤ , with f := 0.126433176216545 (numerically determined).

The precise formulas for the averages that we used are given in TAB3 For each configuration (K, H, b, ), we report the best result found with any of the following two stepsizes: η t := min(32, cn t+1 ) and η t = 32c.

Here c is a parameter that can take the values c = 2 i for i ∈ Z. For each stepsize we determine the best parameter c by a grid search, and consider parameter c optimal, if parameters {2 −2 c, 2 −1 c, 2c, 2 2 c} yield worse results (i.e. more iterations to reach the target accuracy).

<|TLDR|>

@highlight

We prove that parallel local SGD achieves linear speedup with much lesser communication than parallel mini-batch SGD.

@highlight

Provides a convergence proof for local SGD, and proves that local SGD can provide the same speedup gains as minibatch, but may be able to communicate significantly less.

@highlight

This paper presents an analysis of local SGD and bounds on how frequent the estimators obtained by running SGD required to be averaged in order to yield linear parallelization speedups.

@highlight

The authors analyze the local SGD algorithm, where $K$ parallel chains of SGD are run, and the iterates are occasionally synchronized across machines by averaging