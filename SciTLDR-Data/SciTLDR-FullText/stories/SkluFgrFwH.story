Learning Mahalanobis metric spaces is an important problem that has found numerous applications.

Several algorithms have been designed for this problem, including Information Theoretic Metric Learning (ITML) [Davis et al. 2007] and Large Margin Nearest Neighbor (LMNN) classification [Weinberger and Saul 2009].

We consider a formulation of Mahalanobis metric learning as an optimization problem,where the objective is to minimize the number of violated similarity/dissimilarity constraints.

We show that for any fixed ambient dimension, there exists a fully polynomial time approximation scheme (FPTAS) with nearly-linear running time.

This result is obtained using tools from the theory of linear programming in low dimensions.

We also discuss improvements of the algorithm in practice, and present experimental results on synthetic and real-world data sets.

Our algorithm is fully parallelizable and performs favorably in the presence of adversarial noise.

Learning metric spaces is a fundamental computational primitive that has found numerous applications and has received significant attention in the literature.

We refer the reader to Kulis et al. (2013) ; Li and Tian (2018) for detailed exposition and discussion of previous work.

At the high level, the input to a metric learning problem consists of some universe of objects X, together with some similarity information on subsets of these objects.

Here, we focus on pairwise similarity and dissimilarity constraints.

Specifically, we are given S, D Ă`X 2˘, which are sets of pairs of objects that are labeled as similar and dissimilar respectively.

We are also given some u, ą 0, and we seek to find a mapping f : X Ñ Y , into some target metric space pY, ρq, such that for all x, y P S, ρpf pxq, f pyqq ď u, and for all x, y P D, ρpf pxq, f pyqq ě .

In the case of Mahalanobis metric learning, we have X Ă R d , with |X| " n, for some d P N, and the mapping f : R d Ñ R d is linear.

Specifically, we seek to find a matrix G P R dˆd , such that for all tp, qu P S, we have

and for all tp, qu P D, we have

1.1 OUR CONTRIBUTION

In general, there might not exist any G that satisfies all constraints of type 1 and 2.

We are thus interested in finding a solution that minimizes the fraction of violated constraints, which corresponds to maximizing the accuracy of the mapping.

We develop a p1`εq-approximation algorithm for optimization problem of computing a Mahalanobis metric space of maximum accuracy, that runs in near-linear time for any fixed ambient dimension d P N. This algorithm is obtained using tools from geometric approximation algorithms and the theory of linear programming in small dimension.

The following summarizes our result.

Theorem 1.1.

For any d P N, ε ą 0, there exists a randomized algorithm for learning d-dimensional Mahalanobis metric spaces, which given an instance that admits a mapping with accuracy r˚, computes a mapping with accuracy at least r˚´ε, in time d Op1q nplog n{εq Opdq , with high probability.

The above algorithm can be extended to handle various forms of regularization.

We also propose several modifications of our algorithm that lead to significant performance improvements in practice.

The final algorithm is evaluated experimentally on both synthetic and real-world data sets, and is compared against the currently best-known algorithms for the problem.

Several algorithms for learning Mahalanobis metric spaces have been proposed.

Notable examples include the SDP based algorithm of Xing et al. Xing et al. (2003) , the algorithm of Globerson and Roweis for the fully supervised setting Globerson and Roweis (2006) , Information Theoretic Metric Learning (ITML) by Davis et al. Davis et al. (2007) , which casts the problem as a particular optimization minimizing LogDet divergence, as well as Large Margin Nearest Neighbor (LMNN) by Weinberger et al. Weinberger et al. (2006) , which attempts to learn a metric geared towards optimizing k-NN classification.

We refer the reader to the surveys Kulis et al. (2013) ; Li and Tian (2018) for a detailed discussion of previous work.

Our algorithm differs from previous approaches in that it seeks to directly minimize the number of violated pairwise distance constraints, which is a highly non-convex objective, without resorting to a convex relaxation of the corresponding optimization problem.

The rest of the paper is organized as follows.

Section 2 describes the main algorithm and the proof of Theorem 1.1.

Section 3 discusses practical improvements used in the implementation of the algorithm.

Section 4 presents the experimental evaluation.

In this Section we present an approximation scheme for Mahalanobis metric learning in d-dimensional Euclidean space, with nearly-linear running time.

We begin by recalling some prior results on the class of LP-type problems, which generalizes linear programming.

We then show that linear metric learning can be cast as an LP-type problem.

Let us recall the definition of an LP-type problem.

Let H be a set of constraints, and let w : 2 H Ñ R Y t´8,`8u, such that for any G Ă H, wpGq is the value of the optimal solution of the instance defined by G. We say that pH, wq defines an LP-type problem if the following axioms hold:

(A1) Monotonicity.

For any F Ď G Ď H, we have wpF q ď wpGq.

(A2) Locality.

For any F Ď G Ď H, with´8 ă wpF q " wpGq, and any h P H, if wpGq ă wpG Y thuq, then wpF q ă wpF Y thuq.

More generally, we say that pH, wq defines an LP-type problem on some H 1 Ď H, when conditions (A1) and (A2) hold for all F Ď G Ď H 1 .

A subset B Ď H is called a basis if wpBq ą´8 and wpB 1 q ă wpBq for any proper subset B 1 Ĺ B. A basic operation is defined to be one of the following:

(B0) Initial basis computation.

Given some G Ď H, compute any basis for G.

(B1) Violation test.

For some h P H and some basis B Ď H, test whether wpB Y thuq ą wpBq (in other words, whether B violates h).

(B2) Basis computation.

For some h P H and some basis B Ď H, compute a basis of B Y thu.

We now show that learning Mahalanobis metric spaces can be expressed as an LP-type problem.

We first note that we can rewrite (1) and (2) as

and

where A " G T G is positive semidefinite.

We define H " t0, 1uˆ`R d 2˘, where for each p0, tp, quq P H, we have a constraint of type (3), and for every p1, tp, quq P H, we have a constraint of type (4).

Therefore, for any set of constraints F Ď H, we may associate the set of feasible solutions for F with the set A F of all positive semidefinite matrices A P R nˆn , satisfying (3) and (4) for all constraints in F .

Let w : 2 H Ñ R, such that for all F P H, we have

where r P R d is a vector chosen uniformly at random from the unit sphere from some rotationallyinvariant probability measure.

Such a vector can be chosen, for example, by first choosing some r 1 P R d , where each coordinate is sampled from the normal distribution N p0, 1q, and setting r " r 1 {}r 1 } 2 .

Lemma 2.1.

When w is chosen as above, the pair pH, wq defines an LP-type problem of combinatorial dimension Opd 2 q, with probability 1.

Moreover, for any n ą 0, if each r i is chosen using Ωplog nq bits of precision, then for each F Ď H, with n " |F |, the assertion holds with high probability.

Proof.

Since adding constraints to a feasible instance can only make it infeasible, it follows that w satisfies the monotonicity axiom (A1).

We next argue that the locality axion (A2) also holds, with high probability.

Let F Ď G Ď H, with 8 ă wpF q " wpGq, and let h P H, with wpGq ă wpG Y thuq.

Let A F P A F and A G P A G be some (not necessarily unique) infimizers of wpAq, when A ranges in A F and A G respectively.

The set A F , viewed as a convex subset of R d 2 , is the intersection of the SDP cone with n half-spaces, and thus A F has at most n facets.

There are at least two distinct infimizers for wpA G q, when A G P A G , only when the randomly chosen vector r is orthogonal to a certain direction, which occurs with probability 0.

When each entry of r is chosen with c log n bits of precision, the probability that r is orthogonal to any single hyperplane is at most 2´c log n " n´c; the assertion follows by a union bound over n facets.

This establishes that axiom (A2) holds with high probability.

It remains to bound the combinatorial dimension, κ.

Let F Ď H be a set of constraints.

For each A P A F , define the ellipsoid

For any A, A 1 P A F , with E A " E A 1 , and

Therefore in order to specify a linear transformation G, up to an isometry, it suffices to specify the ellipsoid E A .

Each tp, qu P S corresponds to the constraint that the point pp´qq{u must lie in E A .

Similarly each tp, qu P D corresponds to the constraint that the point pp´qq{ must lie either on the boundary or the exterior of E A .

Any ellipsoid in R d is uniquely determined by specifying at most pd`3qd{2 " Opd 2 q distinct points on its boundary (see Welzl (1991); Chazelle (2000) ).

Therefore, each optimal solution can be uniquely specified as the intersection of at most Opd 2 q constraints, and thus the combinatorial dimension is Opd 2 q. The basis computation step (B2) can be performed starting with the set of constraints B Y thu, and iteratively remove every constraint whose removal does not decrease the optimum cost, until we arrive at a minimal set, which is a basis.

In total, we need to solve at most d SDPs, each of size Opd 2 q, which can be done in total time d Op1q .

Finally, by the choice of w, any set containing a single constraint in S is a valid initial basis.

Using the above formulation of Mahalanobis metric learning as an LP-type problem, we can obtain our approximation scheme.

Our algorithm uses as a subroutine an exact algorithm for the problem (that is, for the special case where we seek to find a mapping that satisfies all constraints).

We first present the exact algorithm and then show how it can be used to derive the approximation scheme.

An exact algorithm.

Welzl (1991) obtained a simple randomized linear-time algorithm for the minimum enclosing ball and minimum enclosing ellipsoid problems.

This algorithm naturally extends to general LP-type problems (we refer the reader to Har-Peled (2011); Chazelle (2000) for further details).

With the interpretation of Mahalanobis metric learning as an LP-type problem given above, we thus obtain a linear time algorithm for in R d , for any constant d P N. The resulting algorithm on a set of constraints F Ď H is implemented by the procedure Exact-LPTMLpF ; Hq, which is presented in Algorithm 1.

The procedure LPTMLpF ; Bq takes as input sets of constraints F, B Ď H. It outputs a solution A P R dˆd to the problem induced by the set of constraints F Y B, such that all constraints in B are tight (that is, they hold with equality); if no such solution solution exists, then it returns nil.

The procedure Basic-LPTMLpBq computes LPTMLpH; Bq.

The analysis of Welzl (1991) implies that when Basic-LPTMLpBq is called, the cardinality of B is at most the combinatorial dimension, which by Lemma 2.1 is Opd 2 q. Thus the procedure Basic-LPTML can be implemented using one initial basis computation (B0) and Opd 2 q basis computations (B2), which by Lemma 2.2 takes total time d

Op1q .

Algorithm 1 An exact algorithm for Mahalanobis metric learning.

An p1`εq-approximation algorithm.

It is known that the above exact linear-time algorithm leads to an nearly-linear-time approximation scheme for LP-type problems.

This is summarized in the following.

We refer the reader to Har-Peled (2011) for a more detailed treatment.

Lemma 2.3 (Har-Peled (2011), Ch.

15).

Let A be some LP-type problem of combinatorial dimension κ ą 0, defined by some pair pH, wq, and let ε ą 0.

There exists a randomized algorithm which given some instance F Ď H, with |F | " n, outputs some basis B Ď F , that violates at most p1`εqk constraints in F , such that wpBq ď wpB 1 q, for any basis B 1 violating at most k constraints in F , in

, log κ`2 n kε 2κ`2 )¯p t 1`t2 q¯, where t 0 is the time needed to compute an arbitrary initial basis of A, and t 1 , t 2 , and t 3 are upper bounds on the time needed to perform the basic operations (B0), (B1) and (B2) respectively.

The algorithm succeeds with high probability.

For the special case of Mahalanobis metric learning, the corresponding algorithm is given in Algorithm 2.

The approximation guarantee for this algorithm is summarized in 1.1.

We can now give the proof of our main result.

Proof of Theorem 1.1.

Follows immediately by Lemmas 2.2 and 2.3.

Algorithm 2 An approximation algorithm for Mahalanobis metric learning.

procedure LPTML(F ) for i " 0 to log 1`ε n do p Ð p1`εq´i for j " 1 to log Opd 2 q n do subsample F j Ď F , where each element is chosen independently with probability p A i,j Ð Exact-LPTMLpF j q end for end for return a solution out of tA i,j u i,j , violating the minimum number of constraints in F end procedure Regularization.

We now argue that the LP-type algorithm described above can be extended to handle certain types of regularization on the matrix A. In methods based on convex optimization, introducing regularizers that are convex functions can often be done easily.

In our case, we cannot directly introduce a regularizing term in the objective function that is implicit in Algorithm 2.

More specifically, let costpAq denote the total number of constraints of type (3) and (4) that A violates.

Algorithm 2 approximately minimizes the objective function costpAq.

A natural regularized version of Mahalanobis metric learning is to instead minimize the objective function cost 1 pAq :" costpAq`η¨regpAq, for some η ą 0, and regularizer regpAq.

One typical choice is regpAq " trpACq, for some matrix C P R dˆd ; the case C " I corresponds to the trace norm (see Kulis et al. (2013) ).

We can extend the Algorithm 2 to handle any regularizer that can be expressed as a linear function on the entries of A, such as trpAq.

The following summarizes the result.

Theorem 2.4.

Let regpAq be a linear function on the entries of A, with polynomially bounded coefficients.

For any d P N, ε ą 0, there exists a randomized algorithm for learning d-dimensional Mahalanobis metric spaces, which given an instance that admits a solution A 0 with cost 1 pA 0 q " c˚, computes a solution A with cost 1 pAq ď p1`εqc˚, in time d Op1q nplog n{εq Opdq , with high probability.

Proof.

If η ă ε t , for sufficiently large constant t ą 0, since the coefficients in regpAq are polynomially bounded, it follows that the largest possible value of η¨regpAq is Opεq, and can thus be omitted without affecting the result.

Similarly, if η ą p1{εqn t 1 , for sufficiently large constant t 1 ą 0, since there are at most`n 2˘c onstraints, it follows that the term costpAq can be omitted form the objective.

Therefore, we may assume w.l.o.g.

that regpA 0 q P rε Op1q , p1{εqn Op1q s. We can guess some i " Oplog n`logp1{εqq, such that regpA 0 q P pp1`εq i´1 , p1`εq i s. We modify the SDP used in the proof of Lemma 2.2 by introducing the constraint regpAq ď p1`εq i .

Guessing the correct value of i requires Oplog n`logp1{εqq executions of Algorithm 2, which implies the running time bound.

We now discuss some modifications of the algorithm described in the previous section that significantly improve its performance in practical scenarios, and have been integrated in our implementation.

Move-to-front and pivoting heuristics.

We use heuristics that have been previously used in algorithms for linear programming Seidel (1990) ; Clarkson (1995) , minimum enclosing ball in R 3 Megiddo (1983) , minimum enclosing ball and ellipsoid is R d , for any fixed d P N Welzl (1991), as well as in fast implementations of minimum enclosing ball algorithms Gärtner (1999) .

The move-to-front heuristic keeps an ordered list of constraints which gets reorganized as the algorithm runs; when the algorithm finds a violation, it moves the violating constraint to the beginning of the list of the current sub-problem.

The pivoting heuristic further improves performance by choosing to add to the basis the constraint that is "violated the most".

For instance, for similarity constraints, we pick the one that is mapped to the largest distance greater than u; for dissimilarity constraints, we pick the one that is mapped to the smallest distance less than .

Approximate counting.

The main loop of Algorithm 2 involves counting the number of violated constraints in each iteration.

In problems involving a large number of constraints, we use approximate counting by only counting the number of violations within a sample of Oplog 1{εq constraints.

We denote by LPTML t for the version of the algorithm that performs a total of t iterations of the inner loop.

Early termination.

A bottleneck of Algorithm 2 stems from the fact that the inner loop needs to be executed for log Opd 2 q n iterations.

In practice, we have observed that a significantly smaller number of iterations is needed to achieve high accuracy.

Parallelization.

Algorithm 2 consists of several executions of the algorithm Exact-LPTML on independently sampled sub-problems.

Therefore, Algorithm 2 can trivially be parallelized by distributing a different set of sub-problems to each machine, and returning the best solution found overall.

We have implemented Algorithm 2, incorporating the practical improvements described in Section 3, and performed experiments on synthetic and real-world data sets.

Our LPTML implementation and documentation can be found at the supplementary material 1 .

We now describe the experimental setting and discuss the main findings.

Classification task.

Each data set used in the experiments consists of a set of labeled points in R d .

The label of each point indicates its class, and there is a constant number of classes.

The set of similarity constraints S (respt.

dissimilarity constraints D) is formed by uniformly sampling pairs of points in the same class (resp.

from different classes).

We use various algorithms to learn a Mahalanobis metric for a labeled input point set in R d , given these constraints.

The values u and are chosen as the 90th and 10th percentiles of all pairwise distances.

We used 2-fold cross-validation: At the training phase we learn a Mahalanobis metric, and in the testing phase we use k-NN classification, with k " 4, to evaluate the performance.

Data sets.

We have tested our algorithm on the following synthetic and real-world data sets:

1. Real-world: We have tested the performance of our implementation on the Iris, Wine, Ionosphere and Soybean data sets from the UCI Machine Learning Repository 2 .

2.

Synthetic: Next, we consider a synthetic data set that is constructed by first sampling a set of 100 points from a mixture of two Gaussians in R 2 , with identity covariance matrices, and with means p´3, 0q and p3, 0q respectively; we then apply a linear transformation that stretches the y axis by a factor of 40.

This linear transformation reduces the accuracy of k-NN on the underlying Euclidean metric with k " 4 from 1 to 0.68.

3.

Synthetic + Adversarial Noise: We modify the above synthetic data set by introducing a small fraction of points in an adversarial manner, before applying the linear transformation.

Figure  3b depicts the noise added as five points labeled as one of the classes, and sampled from a Gaussian with identity covariance matrix and mean p´100, 0q (Figure 3a ).

Algorithms.

We compare the performance of our algorithm against ITML and LMNN.

We used the implementations provided by the authors of these works, with minor modifications.

Accuracy.

Algorithm 2 minimizes the number of violated pairwise distance constraints.

It is interesting to examine the effect of this objective function on the accuracy of k-NN classification.

Comparison to ITML and LMNN.

We compared the accuracy obtained by LPTML t , for t " 2000 iterations, against ITML and LMNN.

Table 1 summarizes the findings on the real-world and data sets and the synthetic data set without adversarial noise.

We observe that LPTML achieves accuracy that is comparable to ITML and LMNN.

We observe that LPTML outperforms ITML and LMNN on the Synthetic + Adversarial Noise data set.

This is due to the fact that the introduction of adversarial noise causes the relaxations used in ITML and LMNN to be biased towards contracting the x-axis.

In contrast, the noise does not "fool" LPTML because it only changes the optimal accuracy by a small amount.

The results are summarized in Figure 2 .

@highlight

Fully parallelizable and adversarial-noise resistant metric learning algorithm with theoretical guarantees.