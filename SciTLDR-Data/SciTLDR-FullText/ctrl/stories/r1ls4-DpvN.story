Influence diagrams provide a modeling and inference framework for sequential decision problems, representing the probabilistic knowledge by a Bayesian network and the preferences of an agent by utility functions over the random variables and decision variables.

MDPs and POMDPS, widely used for planning under uncertainty can also be represented by influence diagrams.

The time and space complexity of computing the maximum expected utility (MEU) and its maximizing policy is exponential in the induced width of the underlying graphical model, which is often prohibitively large due to the growth of the information set under the sequence of decisions.

In this paper, we develop a weighted mini-bucket approach for bounding the MEU.

These bounds can be used as a stand-alone approximation that can be improved as a function of a controlling i-bound parameter.

They can also be used as heuristic  functions to guide search, especially for planning  such as MDPs and POMDPs.

We evaluate the scheme empirically against state-of-the-art, thus illustrating its potential.

An influence diagram (ID) BID4 ) is a graphical model for sequential decision-making under uncertainty that compactly captures the local structure of the conditional independence of probability functions and the additivity of utility functions.

Its structure is captured by a directed acyclic graph (DAG) over nodes representing the variables (decision and chance variables).

The standard query on an ID is finding the maximum expected utility (MEU) and the corresponding optimal policy for each decision, subject to the history of observations and decisions.

Computing the MEU is recognized as one of the hardest tasks over graphical models, and hence recent work aims at developing anytime bounding schemes that tighten the bounds if more time and memory is available.

Often, the target is to incorporate such bounds as heuristic functions to guide search algorithms.

In this paper, we focus on computing the upper bound of MEU for a single agent sequential decision making problem with no-forgetting assumptions.

We build on the methodology of weighted mini-bucket with costshifting that was used in the past for bounding probabilistic queries such as the partition function, Maximum A Posteriori (MAP) and Marginal MAP (MMAP) BID0 BID12 BID5 BID14 BID13 ).

Upper bounding schemes for IDs are mostly based on either decomposing the underlying graphical model of an ID or on relaxing the information constraints that impose a partial elimination ordering for inference (e.g., by variable elimination).

Both approaches are orthogonal and both can contribute to tightening the bounds.

We elaborate below.

Decomposition-based bounds A preliminary extension of the mini-bucket scheme BID0 to the MEU tasks was presented in a workshop paper by BID2 .

This scheme decomposes the constrained join-tree BID6 to a mini-bucket tree by partitioning a cluster in the join-tree into mini-buckets whose number of variables is bounded by the i-bound parameter.

The mini-bucket scheme outputs bounds conditioned on partial assignment relative to a variable ordering and is therefore well poised to yield heuristics for search along the same ordering.

Also recently, dual decomposition schemes, which are not directional, were extended for mixed graphical models queries such as Marginal Map (MMAP) by BID22 .

The scheme was also extended to the MEU task by using the framework of valuation algebra BID23 BID16 BID17 .

The valuation algebra for IDs defines operators such as combination and marginalization over pairs of probability and utility functions called potentials BID6 .Information relaxation An alternative approach is to relax the information constraints in the ID thus allowing flexible variable orderings for processing BID19 .

In particular, the information relaxation scheme for IDs can be viewed as re-ordering the chance variables in the constrained variable ordering.

BID24 integrated the re-ordering upper bounds as heuristics to guide a branch and bound search algorithm for solving IDs.

We develop a weighted mini-bucket scheme for generating upper bounds on the MEU.

Given a consistent variable ordering, the scheme generates bounds for each variable conditioned on past histories, observations and decisions relative to a given variable ordering.

We show empirically that the scheme can offer effective bounds faster than current stateof-the-art schemes.

Thus, these bounds have high potential to be used as heuristics for search, in future work.

An ID is a tuple M :" xC, D, P, U, Oy consisting of a set of discrete random variables C " tC i |i P I C u, a set of discrete decision variables D " tD i |i P I D u, a set of conditional probability functions P " tP i |P i P I P u , and a set of real-valued additive utility functions U " tU i |U i P I U u.

We use I S " t0, 1,¨¨¨, |S|´1u to denote the set of indices of each element in a set S, where |S| is the cardinality of S. As illustrated in FIG0 , an ID can be associated with a DAG containing three types of nodes: the chance nodes C drawn as circles, the decision nodes D drawn as squares, and the value nodes U drawn as diamonds.

There are also three types of directed edges: edges directed into a chance node C i from its parents papC i q Ď C Y D representing the conditional probability function P i pC i |papC i qq, edges directed into a value node U i denoting the utility function U i pX i q from its scope X i Ď C Y D, and informational arcs (dashed arrows in FIG0 ) directed from chance nodes to a decision node.

The set of parent nodes associated with a decision node D i is called the information set I i , and denotes chance nodes that are assumed to be observed immediately before making decision D i .

The constrained variable ordering O obeys a partial ordering which alternates between information sets and decision variables tI 0 ă D 0 ă¨¨¨ă I |D|´1 ă D |D|´1 ă I |D| u. The partial elimination ordering should ensure the regularity of the ID (a decision can only be preceded by at most one decision), and dictates the available information at each decision D i so that the non-forgetting agent makes decisions in a multi-staged manner based on the history available at each stage i, DISPLAYFORM0 Solving an ID is computing the maximum expected utility Er ř UiPU U i |∆ ∆ ∆s and finding a set of optimal policies ∆ ∆ ∆ " t∆ i |∆ i : RpD i q Þ Ñ D i , @D i P Du, where ∆ i is a deterministic decision rule for D i and RpD i q Ď HpD i q is the subset of history called the requisite information to D i , namely, the only relevant history for making a decision BID18 .

The valuation algebra for IDs is an algebraic framework for computing the expected utility values, or values for short, based on the combination and marginalization on potentials BID6 BID16 BID10 .

Let a valuation ΨpXq be a pair of probability and value functions pP pXq, V pXqq over a set of variables X called its scope.

Occasionally, we will abuse the notation by dropping the scope from a function, e.g., writing P 1 pX 1 q as P 1 .

The combination and marginalization operators are defined as follows.

Definition 1. (combination of two valuations) Given two valuations Ψ 1 pX 1 q :" pP 1 pX 1 q, V 1 pX 1 qq and Ψ 2 pX 2 q :" pP 1 pX 2 q, V 1 pX 2 qq, the combination of the two valuations over X 1 Y X 2 is defined by DISPLAYFORM0 Following Definition 1, the identity is p1, 0q and the inverse of pP pXq, V pXqq is p1{P pXq,´V pXq{P 2 pXqq.

Definition 2. (marginalization of a valuation) Given a valuation ΨpXq:"pP pXq, V pXqq, marginalizing over Y Ď X by summation, maximization, or powered-summation with weights w are defined by DISPLAYFORM1 The powered-sum elimination operator ř w X is defined by ř w X f pXq " r ř X |f pXq| 1{w s w , which replaces maximization and summation with a weight 0 ď w ď 1 for a variable X, and it reduces to maximization and summation when wÑ0 and w"1, respectively.

Finally, we define the comparison operator for the valuation algebra as a partial order as follows.

Definition 3. (comparison of two valuations) Given two valuations Ψ 1 :" pP 1 , V 1 q and Ψ 2 : pP 2 , V 2 q on the same scopes, we define inequality Ψ 1 ď Ψ 2 iff.

P 1 ď P 2 and V 1 ď V 2 .In the following section, we state the decomposition bound using the valuation algebra notation, which were defined in .

An ID can be compactly represented by the valuation algebra as M :" xX, Ψ Ψ Ψ, Oy, where X " C Y D and Ψ Ψ Ψ " tpP i , 0q|P i P Pu Y tp1, U i q|U i P Uu.

The MEU can be written as ÿ DISPLAYFORM0 where X α denotes the scope of Ψ α .

The dependence relation between variables can be captured by a primal graph FIG0 .

The minibucket tree was generated along a constrained elimination ordering D1, S2, S3, D0, S0, S1 (from top to bottom) of the variables labelling the buckets.

The mini-buckets are created at bucket S2 by limiting the maximum cluster size from 4 to 3 (i-bound is 3).

The non-negative weights wS 2 and w S 1 2 for the variable S2 associated with mini-buckets S2 and S 1 2 sum to 1.G p " pV, Eq, where the set of nodes V are the variables, and an edge e P E connects two nodes if their corresponding variables appear in the scope of some function.

To obtain a primal graph of an ID, the information arcs and utility nodes should be removed after moralization.

The mini-bucket scheme BID0 relaxes an exact tree decomposition BID3 by duplicating variables whenever the maximum clique size exceeds an i-bound parameter.

By splitting a bucket into mini-buckets, the space and time complexity is exponential in the i-bound only.

The weighted mini-bucket elimination scheme BID11 tightens the mini-bucket relaxation by using Hölder's inequality, DISPLAYFORM0 where q is the number of mini-buckets generated from the bucket of variable X, Ψ α pX α q is the valuation at the α-th mini-bucket, w X is the weight of the variable X that is either 1 for the sum-marginal and 0 for the max-marginal, w α X is the set of non-negative weights distributed to q minibuckets such that w X " ř αPI F w α X .

The weighted minibuckets reduces back to the naive mini-buckets by assigning one of the weights to 1.0 (sum-marginal) and all others 0.0 (max-marginal).

FIG1 shows the schematic trace of the weighted mini-bucket elimination algorithm for the ID in FIG0 .

We can see that the bucket labelled by the variable S 2 decomposed into two mini-buckets by duplicating the variable S 2 to S 1 2 and distributing the non-negative weights to w S2 and w S 1 2 such that w S2`wS 1 2 " 1.

The message from each mini-bucket propagates to the closest bucket labelled by the variable that appears in the scope of the message after marginalizing the variable S 2 and S 1 2 , respectively.

Join-Graph Decomposition A join-tree decomposition of a primal graph G p is a tree of cliques generated by triangulating the G p along a constrained ordering O compatible with the information constraints of the ID.

The space and time complexity of exact inference algorithms for solving Eq. FORMULA3 is exponential in the graph parameter called treewidth BID1 .

Join-graph decomposition BID15 ) is an approximation scheme that decomposes the cliques in a tree decomposition, yielding clusters whose scope size is bounded by the i-bound, yielding a loopy graph of finer grained clusters.

A node in a join-graph is associated with a set of functions and a subset of variables containing their scopes.

An edge between two adjacent nodes is labelled by a separator set that is a subset of variables shared between the two nodes.

A valid join-graph must satisfy the running intersection property; for each variable X P X, the set of clusters containing variable X in their scopes induces a connected subgraph.

Such a join-graph can be constructed by connecting the mini-buckets in a mini-bucket tree by a chain whose separator is the single variable of the bucket.

For example, the mini-bucket tree in FIG1 can be turned into a join graph by connecting the two mini-buckets for the variable S 2 with a separator set tS 2 u.

The generalized dual decomposition scheme BID22 (GDD) provides upper bounds to the marginal MAP query by generalizing the Hölder's inequality in Eq. FORMULA4 to the fully decomposed setting expressed by DISPLAYFORM1 where I α is the index set of all nodes in a join graph, Ψ α pX α q is a valuation that combines all valuations in the α-th cluster, w " tw X1 ,¨¨¨, w X |X| u is the set of all weights corresponds to the set of all variables X, X α is the set of duplicated copies of all variables to the α-th cluster, and DISPLAYFORM2 u is the set of weights to the X α such that w Xi " ř αPIα w α Xi for all X i P X. Reparameterized Decomposition Bounds The upper bounds provided by the various decomposition schemes can be tightened by introducing auxiliary optimization parameters to the decomposition bounds, resulting in reparameterizing of the original functions.

Most recently, ) presented a GDD reparameterized bounds for MEU task by extending the fully decomposed bounds over a join-graph decomposition of IDs.

The auxiliary optimization parameters are the cost shifting valuations δ pCi,Cj q and δ pCj ,Ciq between two nodes C i and C j defined over the separator variables S pCi,Cj q such that both cancels to the identity δ pCi,Cj q b δ pCj ,Ciq " p1, 0q, and the weight parameters w C that are distributed to each cluster C. From Eq. (3), we can rewrite the reparameterized bound for IDs as, DISPLAYFORM3 where each valuation Ψ α at the α-th cluster is reparameterized by the costs from all adjacent edges in the join graph.

The local optimum that tightens the MEU can be obtained by minimizing the value component of the right-hand side of Eq. (4) relative to the optimization variables w α for all α P I α , and δ pCi,Cj q and δ pCj ,Ciq for all pC i , C j q P S, subject to the constraints w X " ř α w α X for all X P X, and δ pCi,Cj q b δ pCj ,Ciq " p1, 0q for all pC i , C j q P S.

The value component of the decomposition bounds for IDs as an Eq. (4) does not have a convex form because the global expected utility value combines the probability and value components from all the decomposed clusters.

This nonconvexity degrades the quality of the upper bounds computed by algorithms that optimize the bounds.

For example, the JGDID scheme often shows degradation of the quality of the upper bounds even with a higher i-bound; the number of optimization parameters is exponential in the i-bound, so the dimension of the parameter space rapidly increases.

An alternative approach we explore here is to interleave the variable elimination and decomposition/optimization of the clusters on-the-fly while performing a variable elimination scheme.

In this way, the intermediate reparameterization step optimizes a partial decomposition scheme applied to a single cluster of the join-tree only, resulting in a lower dimensional optimization space.

In the following subsections, we develop the weighted mini-bucket elimination bounds for IDs (WMBE-ID) based on this idea.

Given an ID M :" xX, Ψ Ψ Ψ, Oy, we apply the weighted mini-bucket decomposition by Eq. (2) for one variable at a time following the constrained elimination ordering O : tX |X| , X |X|´1 ,¨¨¨, X 1 u. The intermediate messages are sent to lower mini-buckets as illustrated in FIG1 .

To tighten the upper bound, the auxiliary valuations between mini-buckets can be introduced, yielding the following parameterized bound for each bucket independently, DISPLAYFORM0 where δ pα,α`1q pXq is the cost shifting valuation between mini-buckets from the α-th mini-bucket to the pα`1q-th minibucket.

Following the example in FIG1 , the parameterized upper bound to the weighted mini-bucket decomposition at Bucket S 2 can be written as, DISPLAYFORM1 However, the value component of the parameterized bound in Eq (5) cannot be served as an objective function for tightening the upper bound to the MEU because the scope size of the combined valuations from mini-buckets after eliminating variable X α at the right-hand side of the inequality is as large as the induced width.

Therefore, we propose the following surrogate objective function for minimizing the upper bound as follows.

Theorem 1. (weighted MBE Bounds for IDs) Given an ID M :" xX, Ψ Ψ Ψ, Oy and a constrained variable elimination ordering O :"tX |X| , X |X|´1 ,¨¨¨, X 1 u, assume that the variables tX |X| , X |X|´1 ,¨¨¨X n`1 u are already eliminated by weighted mini-bucket elimination algorithm.

Let Ψ Xi pX 1:i q be the combination of the valuations allocated to bucket X i of the join-tree, Q Xi :" t1,¨¨¨, q Xi u be the mini-bucket partitioning for bucket X i , and Ψ Xn α pX Xi α q be the combination of the valuations allocated at the α-th mini-bucket.

Then, the exact MEU of the subproblem defined over variables X 1:n :" tX 1 ,¨¨¨, X n u can be bounded by The weights w 1:n :" tw X1 ,¨¨¨w Xn u in Eq. (6) is the set of weights of the variables X 1:n , each of them is either 1 for X i being a chance variable or 0 for a decision variable, and the weights w Xi,α 1:n in Eq. (8) is the set of weights of the variables X 1:n in the α-th mini-bucket partition in bucket X n such that DISPLAYFORM2 Proof.

The upper bound of Eq. (7) can be obtained by applying the weighted mini-bucket scheme in Eq. (2) to the bucket X n , and the upper bound of Eq. (8) can be obtained by first partitioning all buckets to mini-buckets and applying the fully decomposed bound of Eq. (3) to each minibucket, α P Q Xi for all X i P X BID22 BID10 .

Optimization Objectives and Parameters The upper bound derived in Theorem 1 can be reparameterized by the cost functions on the chain of mini-buckets before processing and sending messages.

Given an ID M :" xX, Ψ Ψ Ψ, Oy and a constrained variable elimination ordering O :" tX |X| , X |X|´1 ,¨¨¨, X 1 u, the weighted mini-bucket bounds for IDs in Theorem 1 can be parameterized over the chain of mini-buckets Q Xn as follows, assuming that variable X n is removed after reparameterization.

Require: Influence diagram M " xX, Ψ Ψ Ψ, Oy, total constrained elimination order O :" tXN , XN´1,¨¨¨, X1u, i-bound, iteration limit L, Ensure: an upper bound of the MEU 1: Initialization: Generate a schematic mini-bucket tree BID0 and allocate valuations to proper mini-buckets.

2: U b Ð p1, 0q 3: for i Ð N to 1 do 4: Partition bucket Xi to mini-buckets QX i :" t1,¨¨¨, qX i u with i-bound 5: for α P QX i do 6: Ψ DISPLAYFORM0 α q Ð combine valuations at the mini-bucket α 7: end for 8: iter " 0 9: Initialize join-graph with the uniform weights for all remaining mini-buckets tQX 1 ,¨¨¨QX n u 10: Evaluate objective function Eq. (9) for all remaining minibuckets tQX 1 ,¨¨¨QX n u 11: while iter ă L or until bounds improved do 12:Update a set of cost functions tδ pα,α`1q|αPQ X i u subject to the constraints in Eq. FORMULA3 and FORMULA3 13:Update a set of weights tw DISPLAYFORM1 |α P QX i u 14: end while 15: for α P QX i do

pλ DISPLAYFORM0 if pλ DISPLAYFORM1 Send message pλ The optimization parameters are the set of cost functions between two mini-buckets and the weights over the minibuckets Q Xn , DISPLAYFORM2 where δ Xn pα,α`1q pX n q is defined by the probability component λ Note that, the evaluation of upper bounds while performing the weighted mini-bucket elimination requires reconfiguring the join-graph and weight parameters of the remaining minibuckets after eliminating a variable.

However, the evaluation of the optimization objective inside optimization procedure does not require re-evaluation of all mini-buckets because the fully decomposed bounds of all nodes except the minibuckets under optimization does not change subject to the changes in the cost functions and weights.

In the empirical evaluation, the cost functions and weights are updated separately by calling the constrained optimization routines to update the cost and the exponentiated gradient descent BID8 for updating the weights of mini-buckets.

Since the powered-sum elimination operator is defines over the absolute value of a function, the value components at all mini-buckets are constrained to remain positive after reparameterization.

Let Ψ Xn α pX α q be the valuation at the α-th mini-bucket of bucket X n with the probability component λ Xn α pX α q and the value component η Xn α pX α qq, and δ pα,α`1q pX n q :" pλ pα,α`1q pX n q, η pα,α`1q pX n qq be the cost shifting valuation between the α-th and pα`1q-th minibuckets.

Then, the non-negativity of the value components after the reparameterization is ensured by: DISPLAYFORM0 pα,α`1q pX n q λ pα,α`1q pX n q`η pα´1,αq pX n q λ pα´1,αq pX n q ě 0.

FORMULA3 In addition, the non-negativity of probability components is ensured by: λ pα,α`1q pX n q ě 0for all mini-buckets α P Q Xn .

Equipped with the optimization objective and constraints, any constrained optimization procedure can be applied to reparameterize the mini-buckets.

For the empirical evaluation, we integrated off-the-shelf optimization libraries such as sequential least square programming BID9 .Interleaving Elimination and Optimization Algorithm 1 outlines the overall procedure of the weighted mini-bucket elimination interleaved with reparameterization to compute the upper bound of MEU.

Given an input ID M and a total constrained elimination order O, the schematic bucket tree elimination algorithm BID1 ) is called to generate a join-tree and allocate valuations at the initialization step (line 1).

Variables are processed from first to last in the ordering, as follows.

Given the current variable X i , the algorithm partitions its bucket into mini-buckets Q Xi and combines the valuations placed in each mini-bucket (lines 4-7).

The fully decomposed join-graph decomposition based bound is pre-computed using the uniform weights at all mini-buckets remaining in the problem (lines 9-10).

Subsequently, the cost functions and weights that parameterize the mini-buckets corresponding to variable X i are updated in order to tighten the upper bound of the inequality Eq. (9).

After the optimization step, messages from mini-buckets are computed by marginalizing the reparameterized valuationsΨ Xi α using the powered-sum operator with weights tw Xi,α Xi |@α P Q Xi u. maximum values for each of the problem parameters: n -the number of chance and decision variables, f -the number of probability and utility functions, k -the domain size, s -the scope size, and w -the induced width, respectively.

We compare the performance of our proposed bounding scheme WMBE-ID with earlier approaches on 4 domains each containing 5 problem instances.

The benchmark statistics are summarized in TAB1 .

For our purpose, we generated 4 domains in the following way: (1) Factored FH-MDP instances are generated from two stage factored MDP templates by varying the number of state and action variables, the scope size of functions, and the length of time steps between 3 and 10.

(2) Factored FH-POMDP instances are generated similarly to MDP instances, but it incorporates observed variables.

(3) Random influence diagrams (RAND) are generated from a random topology of influence diagram by varying the number of chance, decision, and value nodes.

(4) BN instances are IDs converted from the Bayesian network released in the UAI-2006 probabilistic inference challenge by converting random nodes to decision nodes and adding utility nodes.

We evaluate the proposed WMBE-ID algorithm in 3 different configurations: (1) uniform weights without cost updates (WMBE-U) (2) uniform weights with cost updates (WMBE-UC), and (3) update both weights and costs (WMBE-WC).

For comparison, we consider the following earlier approaches: the mini-bucket elimination bound (MBE), MBE combined with the re-ordering relaxation (MBE-Re), and the state-of-the-art join graph decomposition bounds for IDs (JGDID).

We implemented all algorithms in Python using the NumPy BID20 and SciPy BID7 libraries.

WMBE-U, MBE, and MBE-Re are non-iterative algorithm that computes the upper bounds in a single pass.

On the other hand, WMBE-UC, WMBE-WC, and JGDID are iterative algorithms that reparameterize cost functions and weights until the iteration limit or convergence.

The number of iterations of WMBE-UC and WBME-WC is the maximum number of calls allowed to reparameterizing the cost functions of mini-buckets by the off-the-shelf optimization library, sequential least square programming in SciPy BID7 ) with the default parameters.

JGDID updates all parameters by the gradient descent method at each iteration.

Table 2 shows the quality of upper bounds of all 6 algorithms at each instance of four domains with the i-bounds 1, 5, 10, and 15, and the iteration limit 1, 5, 10, and 20.

We can see that the quality of the bound from MBE and MBE-Re is a magnitude worse than the other algorithms.

JGDID algorithm generates the most tight bound on many of the cases but it consistently produces worse bounds with higher i-bound and it takes more time.

On the other hand, WMBE-ID algorithms consistently improve the quality of the bounds with higher i-bounds.

Comparing the 3 variants of WMBE-ID algorithms, optimizing both weights and cost functions greatly improved the quality of bound with additional time overhead.

In case of ID from BN 78 w19d3 we can also observe that WMBE-WC produces better bound than the best bound produced by JGDID; 23.44 in 1078 seconds by WMBE-WC with i-bound 15, and 27.53 in 1281 seconds by JGDID with ibound 1.

Similarly, WBME-WC produced better bounds for mdp9-32-3-8-3 instance; 21.48 in 5905 seconds by WMBE-WC with i-bound 10, and 23.58 in 15340 seconds by JGDID with i-bound 1.Comparing WMBE-ID vs. JGDID Table 3 compares the quality of the upper bounds as well as the running time against JGDID(i=1).

Clearly, we can see that JGDID with i-bound 10 shows degradation of the quality of the bounds on all 20 instances.

On the other hand, both WMBE-UC and WMBE-WC improves the upper bounds with higher i-bounds, and WMBE-WC produces tighter bounds than JGDID on 7 instances with the i-bound 15 in shorter time bounds than JGDID(i=1).

The experiments shows that WMBE-ID produces high quality bounds in a shorter time bounds compared to JGDID, and it improves the tightness of the bounds with higher i-bounds as opposed to JGDID.

More importantly, WMBE-ID can be pre-compiled as a static heuristic function if all the intermediate messages are stored as a look up table before starting search.

This characteristic is especially desired for heuristic evaluation functions that require less overhead on computing the heuristic values.

The weighted mini-bucket heuristic functions for MAP and MMAP have shown state-of-the-art performance when used to guide AND/OR search strategies BID13 .

Therefore our plan is to integrate WMBE-ID as a heuristic generator for AND/OR search algorithms for solving IDs.

We presented a new bounding scheme for influence diagrams, called WMBE-ID, which computes upper bounds of the MEU by interleaving variable elimination with optimizing partial decomposition within each variable's bucket.

Compared with the previous approaches, our proposed upper bounding scheme produces high quality upper bounds in shorter time bounds.

This is instrumental for our plan to Table 2 : The performance of the bounding schemes on individual instances.

n is the number of variables, f is the number of functions, k is the maximum domain size, s is the maximum scope size, w is the constrained induced width.

We show the (time, upper bound) for various i-bounds and number of iterations for algorithms updating the costs or weights.

WMBE-U is the mini-bucket elimination with uniform weights, WMBE-UC preforms cost shifting without optimizing the weight, WMBE-WC optimizes both weights and costs, JGDID is the fully decomposed bound over a join graph that optimizes both weights and costs, MBE is the simple mini-bucket elimination, and MBE-Re is mini-bucket elimination with relaxed variable ordering.

MBE, MBE-RE, and WMBE-U do not optimize the bound.

The best upper bounds are highlighted.

Table 3 : Comparing the ratio of time and quality of upper bounds against JGDID(i=1).

WMBE-UC and WMBE-WC were provided with i-bound 10 and 15 with the number of iteration fixed to 5, and JGDID were provided i-bound 1 and 10 with the maximum number iteration limited by 100.

All the quantities are normalized by the statistics of JGDID(i=1).

DISPLAYFORM0 use such bounds as a heuristic evaluation function for search algorithms for solving influence diagrams.

<|TLDR|>

@highlight

This paper introduces an elimination based heuristic function for sequential decision making, suitable for guiding AND/OR search algorithms for solving influence diagrams.

@highlight

generalizes minibuckets inference heuristic to influence diagrams.