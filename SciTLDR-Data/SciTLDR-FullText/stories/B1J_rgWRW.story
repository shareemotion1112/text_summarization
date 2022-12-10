In this paper we investigate the family of functions representable by deep neural networks (DNN) with rectified linear units (ReLU).

We give an algorithm to train a ReLU DNN with one hidden layer to {\em global optimality} with runtime polynomial in the data size albeit exponential in the input dimension.

Further, we improve on the known lower bounds on size (from exponential to super exponential) for approximating a ReLU deep net function by a shallower ReLU net.

Our gap theorems hold for smoothly parametrized families of ``hard'' functions, contrary to countable, discrete families known in the literature.

An example consequence of our gap theorems is the following: for every natural number $k$ there exists a function representable by a ReLU DNN with $k^2$ hidden layers and total size $k^3$, such that any ReLU DNN with at most $k$ hidden layers will require at least $\frac12k^{k+1}-1$ total nodes.

Finally, for the family of $\R^n\to \R$ DNNs with ReLU activations, we show a new lowerbound on the number of affine pieces, which is larger than previous constructions in certain regimes of the network architecture and most distinctively our lowerbound is demonstrated by an explicit construction of a \emph{smoothly parameterized} family of functions attaining this scaling.

Our construction utilizes the theory of zonotopes from polyhedral theory.

Deep neural networks (DNNs) provide an excellent family of hypotheses for machine learning tasks such as classification.

Neural networks with a single hidden layer of finite size can represent any continuous function on a compact subset of R n arbitrary well.

The universal approximation result was first given by Cybenko in 1989 for sigmoidal activation function BID4 , and later generalized by Hornik to an arbitrary bounded and nonconstant activation function BID15 .

Furthermore, neural networks have finite VC dimension (depending polynomially on the number of edges in the network), and therefore, are PAC (probably approximately correct) learnable using a sample of size that is polynomial in the size of the networks BID1 .

However, neural networks based methods were shown to be computationally hard to learn BID1 and had mixed empirical success.

Consequently, DNNs fell out of favor by late 90s.to address the issue of efficiently training DNNs.

These include heuristics such as dropouts BID39 , but also considering alternate deep architectures such as convolutional neural networks BID33 , deep belief networks BID14 , and deep Boltzmann machines BID31 .

In addition, deep architectures based on new non-saturating activation functions have been suggested to be more effectively trainable -the most successful and widely popular of these is the rectified linear unit (ReLU) activation, i.e., σ(x) = max{0, x}, which is the focus of study in this paper.

In this paper, we formally study deep neural networks with rectified linear units; we refer to these deep architectures as ReLU DNNs.

Our work is inspired by these recent attempts to understand the reason behind the successes of deep learning, both in terms of the structure of the functions represented by DNNs, Telgarsky (2015; ; BID17 ; BID36 , as well as efforts which have tried to understand the non-convex nature of the training problem of DNNs better BID18 ; BID10 .

Our investigation of the function space represented by ReLU DNNs also takes inspiration from the classical theory of circuit complexity; we refer the reader to BID2 ; BID37 ; BID16 ; BID32 ; BID0 for various surveys of this deep and fascinating field.

In particular, our gap results are inspired by results like the ones by BID12 , BID27 and BID38 which show a strict separation of complexity classes.

We make progress towards similar statements with deep neural nets with ReLU activation.

We extend the ReLU activation function to vectors x ∈ R n through entry-wise operation: σ(x) = (max{0, x 1 }, max{0, x 2 }, . . .

, max{0, x n }).

For any (m, n) ∈ N, let A n m and L n m denote the class of affine and linear transformations from R m → R n , respectively.

[ReLU DNNs, depth, width, size]

For any number of hidden layers k ∈ N, input and output dimensions w 0 , w k+1 ∈ N, a R w0 → R w k+1 ReLU DNN is given by specifying a sequence of k natural numbers w 1 , w 2 , . . .

, w k representing widths of the hidden layers, a set of k affine transformations T i : R wi−1 → R wi for i = 1, . . .

, k and a linear transformation T k+1 : R w k → R w k+1 corresponding to weights of the hidden layers.

Such a ReLU DNN is called a (k + 1)-layer ReLU DNN, and is said to have k hidden layers.

The function f : R n1 → R n2 computed or represented by this ReLU DNN is DISPLAYFORM0 where • denotes function composition.

The depth of a ReLU DNN is defined as k + 1.

The width of a ReLU DNN is max{w 1 , . . . , w k }.

The size of the ReLU DNN is w 1 + w 2 + . . .

+ w k .

Definition 2.

We denote the class of R w0 → R w k+1 ReLU DNNs with k hidden layers of widths DISPLAYFORM1 Definition 3. [Piecewise linear functions]

We say a function f : R n → R is continuous piecewise linear (PWL) if there exists a finite set of polyhedra whose union is R n , and f is affine linear over each polyhedron (note that the definition automatically implies continuity of the function because the affine regions are closed and cover R n , and affine functions are continuous).

The number of pieces of f is the number of maximal connected subsets of R n over which f is affine linear (which is finite).Many of our important statements will be phrased in terms of the following simplex.

Definition 4.

Let M > 0 be any positive real number and p ≥ 1 be any natural number.

Define the following set: DISPLAYFORM2

One of the main advantages of DNNs is that they can represent a large family of functions with a relatively small number of parameters.

In this section, we give an exact characterization of the functions representable by ReLU DNNs.

Moreover, we show how structural properties of ReLU DNNs, specifically their depth and width, affects their expressive power.

It is clear from definition that any function from R n → R represented by a ReLU DNN is a continuous piecewise linear (PWL) function.

In what follows, we show that the converse is also true, that is any PWL function is representable by a ReLU DNN.

In particular, the following theorem establishes a one-to-one correspondence between the class of ReLU DNNs and PWL functions.

Theorem 2.1.

Every R n → R ReLU DNN represents a piecewise linear function, and every piecewise linear function R n → R can be represented by a ReLU DNN with at most log 2 (n + 1) + 1 depth.

Proof Sketch:

It is clear that any function represented by a ReLU DNN is a PWL function.

To see the converse, we first note that any PWL function can be represented as a linear combination of piecewise linear convex functions.

More formally, by Theorem 1 in (Wang & Sun, 2005) , for every piecewise linear function f : R n → R, there exists a finite set of affine linear functions 1 , . . .

, k and subsets S 1 , . . .

, S p ⊆ {1, . . . , k} (not necessarily disjoint) where each S i is of cardinality at most n + 1, such that DISPLAYFORM0 where s j ∈ {−1, +1} for all j = 1, . . .

, p. Since a function of the form max i∈Sj i is a piecewise linear convex function with at most n + 1 pieces (because |S j | ≤ n + 1), Equation ( is implementable by a two layer ReLU network and use this construction in an inductive manner to show that maximum of n + 1 numbers can be computed using a ReLU DNN with depth at most log 2 (n + 1) .While Theorem 2.1 gives an upper bound on the depth of the networks needed to represent all continuous piecewise linear functions on R n , it does not give any tight bounds on the size of the networks that are needed to represent a given piecewise linear function.

For n = 1, we give tight bounds on size as follows:

Theorem 2.2.

Given any piecewise linear function R → R with p pieces there exists a 2-layer DNN with at most p nodes that can represent f .

Moreover, any 2-layer DNN that represents f has size at least p − 1.Finally, the main result of this section follows from Theorem 2.1, and well-known facts that the piecewise linear functions are dense in the family of compactly supported continuous functions and the family of compactly supported continuous functions are dense in DISPLAYFORM1 is the space of Lebesgue integrable functions f such that |f | q dµ < ∞, where µ is the Lebesgue measure on R n (see BID29 ).

DISPLAYFORM2 can be arbitrarily well-approximated in the L q norm (which for a function f is given by ||f || q = ( |f | q ) 1/q ) by a ReLU DNN function with at most log 2 (n + 1) hidden layers.

Moreover, for n = 1, any such L q function can be arbitrarily well-approximated by a 2-layer DNN, with tight bounds on the size of such a DNN in terms of the approximation.

Proofs of Theorems 2.2 and 2.3 are provided in Appendix A. We would like to remark that a weaker version of Theorem 2.1 was observed in (Goodfellow et al., 2013, Proposition 4 .1) (with no bound on the depth), along with a universal approximation theorem (Goodfellow et al., 2013, Theorem 4.

3) similar to Theorem 2.3.

The authors of BID9 also used a previous result of Wang (Wang, 2004) for obtaining their result.

In a subsequent work Boris Hanin BID11 has, among other things, found a width and depth upper bound for ReLU net representation of positive PWL functions on [0, 1] n .

The width upperbound is n+3 for general positive PWL functions and n + 1 for convex positive PWL functions.

For convex positive PWL functions his depth upper bound is sharp if we disallow dead ReLUs.

Success of deep learning has been largely attributed to the depth of the networks, i.e. number of successive affine transformations followed by nonlinearities, which is shown to be extracting hierarchical features from the data.

In contrast, traditional machine learning frameworks including support vector machines, generalized linear models, and kernel machines can be seen as instances of shallow networks, where a linear transformation acts on a single layer of nonlinear feature extraction.

In this section, we explore the importance of depth in ReLU DNNs.

In particular, in Section 3.1, we provide a smoothly parametrized family of R → R "hard" functions representable by ReLU DNNs, which requires exponentially larger size for a shallower network.

Furthermore, in Section 3.2, we construct a continuum of R n → R "hard" functions representable by ReLU DNNs, which to the best of our knowledge is the first explicit construction of ReLU DNN functions whose number of affine pieces grows exponentially with input dimension.

The proofs of the theorems in this section are provided in Appendix B.

In this section, we are only concerned about R → R ReLU DNNs, i.e. both input and output dimensions are equal to one.

The following theorem shows the depth-size trade-off in this setting.

Theorem 3.1.

For every pair of natural numbers k ≥ 1, w ≥ 2, there exists a family of hard functions representable by a R → R (k + 1)-layer ReLU DNN of width w such that if it is also representable by a (k + 1)-layer ReLU DNN for any k ≤ k, then this (k + 1)-layer ReLU DNN has size at least DISPLAYFORM0 In fact our family of hard functions described above has a very intricate structure as stated below.

Theorem 3.2.

For every k ≥ 1, w ≥ 2, every member of the family of hard functions in Theorem 3.1 has w k pieces and this family can be parametrized by DISPLAYFORM1 i.e., for every point in the set above, there exists a distinct function with the stated properties.

The following is an immediate corollary of Theorem 3.1 by choosing the parameters carefully.

Corollary 3.3.

For every k ∈ N and > 0, there is a family of functions defined on the real line such that every function f from this family can be represented by a (k 1+ ) + 1-layer DNN with size k 2+ and if f is represented by a k +1-layer DNN, then this DNN must have size at least DISPLAYFORM2 Moreover, this family can be parametrized as, DISPLAYFORM3 A particularly illuminative special case is obtained by setting = 1 in Corollary 3.3: Corollary 3.4.

For every natural number k ∈ N, there is a family of functions parameterized by the set DISPLAYFORM4 such that any f from this family can be represented by a k 2 + 1-layer DNN with k 3 nodes, and every k + 1-layer DNN that represents f needs at least DISPLAYFORM5 We can also get hardness of approximation versions of Theorem 3.1 and Corollaries 3.3 and 3.4, with the same gaps (upto constant terms), using the following theorem.

Theorem 3.5.

For every k ≥ 1, w ≥ 2, there exists a function f k,w that can be represented by a (k + 1)-layer ReLU DNN with w nodes in each layer, such that for all δ > 0 and k ≤ k the following holds: DISPLAYFORM6 where G k ,δ is the family of functions representable by ReLU DNNs with depth at most k + 1, and size at most k DISPLAYFORM7 The depth-size trade-off results in Theorems 3.1, and 3.5 extend and improve Telgarsky's theorems from (Telgarsky, 2015; in the following three ways:(i) If we use our Theorem 3.5 to the pair of neural nets considered by Telgarsky in Theorem 1.1 in Telgarsky (2016) which are at depths k 3 (of size also scaling as k 3 ) and k then for this purpose of approximation in the 1 −norm we would get a size lower bound for the shallower net which scales as Ω(2 k 2 ) which is exponentially (in depth) larger than the lower bound of Ω(2 k ) that Telgarsky can get for this scenario.(ii) Telgarsky's family of hard functions is parameterized by a single natural number k. In contrast, we show that for every pair of natural numbers w and k, and a point from the set in equation 3.1, there exists a "hard" function which to be represented by a depth k network would need a size of at least w k k k .

With the extra flexibility of choosing the parameter w, for the purpose of showing gaps in representation ability of deep nets we can shows size lower bounds which are super-exponential in depth as explained in Corollaries 3.3 and 3.4.(iii) A characteristic feature of the "hard" functions in Boolean circuit complexity is that they are usually a countable family of functions and not a "smooth" family of hard functions.

In fact, in the last section of Telgarsky (2015) , Telgarsky states this as a "weakness" of the state-of-the-art results on "hard" functions for both Boolean circuit complexity and neural nets research.

In contrast, we provide a smoothly parameterized family of "hard" functions in Section 3.1 (parametrized by the set in equation 3.1).

Such a continuum of hard functions wasn't demonstrated before this work.

We point out that Telgarsky's results in (Telgarsky, 2016) apply to deep neural nets with a host of different activation functions, whereas, our results are specifically for neural nets with rectified linear units.

In this sense, Telgarsky's results from (Telgarsky, 2016) are more general than our results in this paper, but with weaker gap guarantees.

Eldan-Shamir BID36 BID7 show that there exists an R n → R function that can be represented by a 3-layer DNN, that takes exponential in n number of nodes to be approximated to within some constant by a 2-layer DNN.

While their results are not immediately comparable with Telgarsky's or our results, it is an interesting open question to extend their results to a constant depth hierarchy statement analogous to the recent result of Rossman et al BID28 .

We also note that in last few years, there has been much effort in the community to show size lowerbounds on ReLU DNNs trying to approximate various classes of functions which are themselves not necessarily exactly representable by ReLU DNNs (Yarotsky, 2016; BID22 BID30 .

One measure of complexity of a family of R n → R "hard" functions represented by ReLU DNNs is the asymptotics of the number of pieces as a function of dimension n, depth k + 1 and size s of the ReLU DNNs.

More precisely, suppose one has a family H of functions such that for every n, k, w ∈ N the family contains at least one R n → R function representable by a ReLU DNN with depth at most k + 1 and maximum width at most w.

The following definition formalizes a notion of complexity for such a H.Definition 5 (comp H (n, k, w)).

The measure comp H (n, k, w) is defined as the maximum number of pieces (see Definition 3) of a R n → R function from H that can be represented by a ReLU DNN with depth at most k + 1 and maximum width at most w.

Similar measures have been studied in previous works BID24 ; BID25 ; BID26 .

The best known families H are the ones from Theorem 4 of BID24 ) and a mild generalization of Theorem 1.1 of (Telgarsky, 2016) to k layers of ReLU activations with width w; these constructions achieve ( DISPLAYFORM0 At the end of this section we would explain the precise sense in which we improve on these numbers.

An analysis of this complexity measure is done using integer programming techniques in BID34 .

DISPLAYFORM1 Figure 1: We fix the a vectors for a two hidden layer R → R hard function as DISPLAYFORM2 Left: A specific hard function induced by 1 norm: 0) .

Note that in this case the function can be seen as a composition of H a 1 ,a 2 with 1 -norm The set of vertices of DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 The following results are well-known in the theory of zonotopes (Ziegler, 1995) .

Theorem 3.6.

The following are all true.

DISPLAYFORM6 .

The set of (b 1 , . . .

, b m ) ∈ R n × . . .

× R n such that this does not hold at equality is a 0 measure set.

DISPLAYFORM7 Definition 7 (extremal zonotope set).

The set S(n, m) will denote the set of DISPLAYFORM8 .

S(n, m) is the so-called "extremal zonotope set", which is a subset of R nm , whose complement has zero Lebesgue measure in R nm .Lemma 3.7.

Given any b 1 , . . .

, b m ∈ R n , there exists a 2-layer ReLU DNN with size 2m which represents the function γ Z(b 1 ,...,b m ) (r).Definition 8.

For p ∈ N and a ∈ ∆ p M , we define a function h a : R → R which is piecewise linear over the segments DISPLAYFORM9 DISPLAYFORM10 Proposition 3.8.

Given any tuple (b 1 , . . . , b m ) ∈ S(n, m) and any point DISPLAYFORM11 n−1 w k pieces and it can be represented by a k + 2 layer ReLU DNN with size 2m + wk.

Finally, we are ready to state the main result of this section.

Theorem 3.9.

For every tuple of natural numbers n, k, m ≥ 1 and w ≥ 2, there exists a family of R n → R functions, which we call ZONOTOPE n k,w,m with the following properties:(i) Every f ∈ ZONOTOPE n k,w,m is representable by a ReLU DNN of depth k + 2 and size 2m + wk, and has n−1 i=0 m−1 i w k pieces.(ii) Consider any f ∈ ZONOTOPE n k,w,m .

If f is represented by a (k + 1)-layer DNN for any k ≤ k, then this (k + 1)-layer DNN has size at least DISPLAYFORM12 The family ZONOTOPE n k,w,m is in one-to-one correspondence with DISPLAYFORM13 Comparison to the results in BID24 Firstly we note that the construction in BID24 requires all the hidden layers to have width at least as big as the input dimensionality n. In contrast, we do not impose such restrictions and the network size in our construction is independent of the input dimensionality.

Thus our result probes networks with bottleneck architectures whose complexity cant be seen from their result.

Secondly, in terms of our complexity measure, there seem to be regimes where our bound does better.

One such regime, for example, is when n ≤ w < 2n and k ∈ Ω( n log(n) ), by setting in our construction m < n.

Thirdly, it is not clear to us whether the construction in BID24 gives a smoothly parameterized family of functions other than by introducing small perturbations of the construction in their paper.

In contrast, we have a smoothly parameterized family which is in one-to-one correspondence with a well-understood manifold like the higher-dimensional torus.

In this section we consider the following empirical risk minimization problem.

Given D data points (x i , y i ) ∈ R n × R, i = 1, . . .

, D, find the function f represented by 2-layer R n → R ReLU DNNs of width w, that minimizes the following optimization problem DISPLAYFORM0 where : R × R → R is a convex loss function (common loss functions are the squared loss, (y, y ) = (y − y ) 2 , and the hinge loss function given by (y, y ) = max{0, 1 − yy }).

Our main result of this section gives an algorithm to solve the above empirical risk minimization problem to global optimality.

Proof Sketch: A full proof of Theorem 4.1 is included in Appendix C. Here we provide a sketch of the proof.

When the empirical risk minimization problem is viewed as an optimization problem in the space of weights of the ReLU DNN, it is a nonconvex, quadratic problem.

However, one can instead search over the space of functions representable by 2-layer DNNs by writing them in the form similar to (2.1).

This breaks the problem into two parts: a combinatorial search and then a convex problem that is essentially linear regression with linear inequality constraints.

This enables us to guarantee global optimality.

Where DISPLAYFORM0 All possible instantiations of top layer weights 3: DISPLAYFORM1 All possible partitions of data into two parts 4: DISPLAYFORM2 for s ∈ S do 7: DISPLAYFORM3 end for 11: OPT = argmin loss(count) 12: end for 13:return {ã}, {b}, s corresponding to OPT's iterate 14: end function Let T 1 (x) = Ax + b and T 2 (y) = a · y for A ∈ R w×n and b, a ∈ R w .

If we denote the i-th row of the matrix A by a i , and write b i , a i to denote the i-th coordinates of the vectors b, a respectively, due to homogeneity of ReLU gates, the network output can be represented as DISPLAYFORM4 whereã i ∈ R n ,b i ∈ R and s i ∈ {−1, +1} for all i = 1, . . .

, w. For any hidden node i ∈ {1 . . .

, w}, the pair (ã i ,b i ) induces a partition P i := (P DISPLAYFORM5 − and a i · x j +b i ≥ 0 ∀j ∈ P i + which are imposed for all i = 1, . . .

, w, which is a convex program.

Algorithm 1 implements the empirical risk minimization (ERM) rule for training ReLU DNN with one hidden layer.

To the best of our knowledge there is no other known algorithm that solves the ERM problem to global optimality.

We note that due to known hardness results exponential dependence on the input dimension is unavoidable Blum & Rivest (1992); Shalev-Shwartz & BenDavid (2014) ; Algorithm 1 runs in time polynomial in the number of data points.

To the best of our knowledge there is no hardness result known which rules out empirical risk minimization of deep nets in time polynomial in circuit size or data size.

Thus our training result is a step towards resolving this gap in the complexity literature.

A related result for improperly learning ReLUs has been recently obtained by Goel et al BID8 .

In contrast, our algorithm returns a ReLU DNN from the class being learned.

Another difference is that their result considers the notion of reliable learning as opposed to the empirical risk minimization objective considered in (4.1).

The running time of the algorithm that we give in this work to find the exact global minima of a two layer ReLU-DNN is exponential in the input dimension n and the number of hidden nodes w. The exponential dependence on n can not be removed unless P = N P ; see BID35 ; BID3 ; BID6 .

However, we are not aware of any complexity results which would rule out the possibility of an algorithm which trains to global optimality in time that is polynomial in the data size and/or the number of hidden nodes, assuming that the input dimension is a fixed constant.

Resolving this dependence on network size would be another step towards clarifying the theoretical complexity of training ReLU DNNs and is a good open question for future research, in our opinion.

Perhaps an even better breakthrough would be to get optimal training algorithms for DNNs with two or more hidden layers and this seems like a substantially harder nut to crack.

It would also be a significant breakthrough to get gap results between consecutive constant depths or between logarithmic and constant depths.

We would like to thank Christian Tjandraatmadja for pointing out a subtle error in a previous version of the paper, which affected the complexity results for the number of linear regions in our constructions in Section 3.2.

Anirbit would like to thank Ramprasad Saptharishi, Piyush Srivastava and Rohit Gurjar for extensive discussions on Boolean and arithmetic circuit complexity.

This paper has been immensely influenced by the perspectives gained during those extremely helpful discussions.

Amitabh Basu gratefully acknowledges support from the NSF grant CMMI1452820.

Raman Arora was supported in part by NSF BIGDATA grant IIS-1546482.Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.

Sequence to sequence learning with neural networks.

In Advances in neural information processing systems, pp.

Proof of Theorem 2.2.

Any continuous piecewise linear function R → R which has m pieces can be specified by three pieces of information, (1) s L the slope of the left most piece, (2) the coordinates of the non-differentiable points specified by a (m − 1)−tuple {(a i , b i )} One notes that for any a, r ∈ R, the function DISPLAYFORM0 is equal to sgn(r) max{|r|(x − a), 0}, which can be implemented by a 2-layer ReLU DNN with size 1.

Similarly, any function of the form, DISPLAYFORM1 is equal to − sgn(t) max{−|t|(x − a), 0}, which can be implemented by a 2-layer ReLU DNN with size 1.

The parameters r, t will be called the slopes of the function, and a will be called the breakpoint of the function.

If we can write the given piecewise linear function as a sum of m functions of the form (A.1) and (A.2), then by Lemma D.2 we would be done.

It turns out that such a decomposition of any p piece PWL function h : R → R as a sum of p flaps can always be arranged where the breakpoints of the p flaps all are all contained in the p − 1 breakpoints of h. First, observe that adding a constant to a function does not change the complexity of the ReLU DNN expressing it, since this corresponds to a bias on the output node.

Thus, we will assume that the value of h at the last break point a m−1 is b m−1 = 0.

We now use a single function f of the form (A.1) with slope r and breakpoint a = a m−1 , and m − 1 functions g 1 , . . .

, g m−1 of the form (A.2) with slopes t 1 , . . .

, t m−1 and breakpoints a 1 , . . .

, a m−1 , respectively.

Thus, we wish to express h = f + g 1 + . . .

+ g m−1 .

Such a decomposition of h would be valid if we can find values for r, t 1 , . . .

, t m−1 such that (1) the slope of the above sum is = s L for x < a 1 , (2) the slope of the above sum is = s R for x > a m−1 , and (3) for each i ∈ {1, 2, 3, .., m − 1} we have DISPLAYFORM2 The above corresponds to asking for the existence of a solution to the following set of simultaneous linear equations in r, t 1 , . . .

, t m−1 : DISPLAYFORM3 It is easy to verify that the above set of simultaneous linear equations has a unique solution.

Indeed, r must equal s R , and then one can solve for t 1 , . . .

, t m−1 starting from the last equation b m−2 = t m−1 (a m−2 − a m−1 ) and then back substitute to compute t m−2 , t m−3 , . . .

, t 1 .

The lower bound of p − 1 on the size for any 2-layer ReLU DNN that expresses a p piece function follows from Lemma D.6.One can do better in terms of size when the rightmost piece of the given function is flat, i.e., s R = 0.

In this case r = 0, which means that f = 0; thus, the decomposition of h above is of size p − 1.

A similar construction can be done when s L = 0.

This gives the following statement which will be useful for constructing our forthcoming hard functions.

Corollary A.1.

If the rightmost or leftmost piece of a R → R piecewise linear function has 0 slope, then we can compute such a p piece function using a 2-layer DNN with size p − 1.Proof of theorem 2.3.

Since any piecewise linear function R n → R is representable by a ReLU DNN by Corollary 2.1, the proof simply follows from the fact that the family of continuous piecewise linear functions is dense in any L p (R n ) space, for 1 ≤ p ≤ ∞.

Lemma B.1.

For any M > 0, p ∈ N, k ∈ N and a 1 , . . .

, a k ∈ ∆ p M , if we compose the functions h a 1 , h a 2 , . . .

, h a k the resulting function is a piecewise linear function with at most (p + 1) k + 2 pieces, i.e., DISPLAYFORM0 is piecewise linear with at most (p + 1) k + 2 pieces, with (p + 1) k of these pieces in the range [0, M ] (see Figure 2) .

Moreover, in each piece in the range [0, M ], the function is affine with minimum value 0 and maximum value M .Proof.

Simple induction on k.

Proof of Theorem 3.2.

Given k ≥ 1 and w ≥ 2, choose any point DISPLAYFORM1 By Definition 8, each h a i , i = 1, . . .

, k is a piecewise linear function with w + 1 pieces and the leftmost piece having slope 0.

Thus, by Corollary A.1, each h a i , i = 1, . . . , k can be represented by a 2-layer ReLU DNN with size w. Using Lemma D.1, H a 1 ,...,a k can be represented by a k + 1 layer DNN with size wk; in fact, each hidden layer has exactly w nodes.

inside any triangle of s q , any affine function will incur an 1 error of at least DISPLAYFORM2

Proof of Theorem 4.1.

Let : R → R be any convex loss function, and let (x 1 , y 1 ), . . . , (x D , y D ) ∈ R n × R be the given D data points.

As stated in (4.1), the problem requires us to find an affine transformation T 1 : R n → R w and a linear transformation T 2 : R w → R, so as to minimize the empirical loss as stated in (4.1).

Note that T 1 is given by a matrix A ∈ R w×n and a vector b ∈ R w so that T (x) = Ax + b for all x ∈ R n .

Similarly, T 2 can be represented by a vector a ∈ R w such that T 2 (y) = a · y for all y ∈ R w .

If we denote the i-th row of the matrix A by a i , and write b i , a i to denote the i-th coordinates of the vectors b, a respectively, we can write the function represented by this network as DISPLAYFORM0 In other words, the family of functions over which we are searching is of the form DISPLAYFORM1 whereã i ∈ R n , b i ∈ R and s i ∈ {−1, +1} for all i = 1, . . .

, w. We now make the following observation.

For a given data point (x j , y j ) ifã i · x j +b i ≤ 0, then the i-th term of (C.1) does not contribute to the loss function for this data point (x j , y j ).

Thus, for every data point (x j , y j ), there exists a set S j ⊆ {1, . . .

, w} such that f (x j ) = i∈Sj s i (ã i · x j +b i ).

In particular, if we are given the set S j for (x j , y j ), then the expression on the right hand side of (C.1) reduces to a linear function ofã i ,b i .

For any fixed i ∈ {1, . . . , w}, these sets S j induce a partition of the data set into two parts.

In particular, we define P i + := {j : i ∈ S j } and P i − := {1, . . .

, D} \ P i + .

Observe now that this partition is also induced by the hyperplane given byã i ,b i : DISPLAYFORM2 and DISPLAYFORM3 .

Our strategy will be to guess the partitions P For a fixed selection of partitions (P i + , P i − ), i = 1, . . .

, w and a vector s in {+1, −1} w , the algorithm solves the following convex optimization problem with decision variablesã i ∈ R n ,b i ∈ R for i = 1, . . .

, w (thus, we have a total of (n + 1) · w decision variables).

The feasible region of the optimization is given by the constraints DISPLAYFORM4 which are imposed for all i = 1, . . .

, w. Thus, we have a total of D · w constraints.

Subject to these constraints we minimize the objective DISPLAYFORM5 Assuming the loss function is a convex function in the first argument, the above objective is a convex function.

Thus, we have to minize a convex objective subject to the linear inequality constraints from (C.2).We finally have to count how many possible partitions (P

n which only holds for n ≥ 2.

For n = 1, a similar algorithm can be designed, but one which uses the characterization achieved in Theorem 2.2.

Let : R → R be any convex loss function, and let (x 1 , y 1 ), . . .

, (x D , y D ) ∈ R 2 be the given D data points.

Using Theorem 2.2, to solve problem (4.1) it suffices to find a R → R piecewise linear function f with w pieces that minimizes the total loss.

In other words, the optimization problem (4.1) is equivalent to the problem DISPLAYFORM0 f is piecewise linear with w pieces .(C.3)We now use the observation that fitting piecewise linear functions to minimize loss is just a step away from linear regression, which is a special case where the function is contrained to have exactly one affine linear piece.

Our algorithm will first guess the optimal partition of the data points such that all points in the same class of the partition correspond to the same affine piece of f , and then do linear regression in each class of the partition.

Altenatively, one can think of this as guessing the interval (x i , x i+1 ) of data points where the w − 1 breakpoints of the piecewise linear function will lie, and then doing linear regression between the breakpoints.

More formally, we parametrize piecewise linear functions with w pieces by the w slope-intercept values (a 1 , b 1 ), . . .

, (a 2 , b 2 ), . . .

, (a w , b w ) of the w different pieces.

This means that between breakpoints j and j + 1, 1 ≤ j ≤ w − 2, the function is given by f (x) = a j+1 x + b j+1 , and the first and last pieces are a 1 x + b 1 and a w x + b w , respectively.

Define I to be the set of all (w − 1)-tuples (i 1 , . . .

, i w−1 ) of natural numbers such that DISPLAYFORM1 Given a fixed tuple I = (i 1 , . . .

, i w−1 )

∈ I, we wish to search through all piecewise linear functions whose breakpoints, in order, appear in the intervals (x i1 , x i1+1 ), (x i2 , x i2+1 ), . . .

, (x iw−1 , x iw−1+1 ).

Define also S = {−1, 1} w−1 .

Any S ∈ S will have the following interpretation: if S j = 1 then a j ≤ a j+1 , and if S j = −1 then a j ≥ a j+1 .

Now for every I ∈ I and S ∈ S, requiring a piecewise linear function that respects the conditions imposed by I and S is easily seen to be equivalent to imposing the following linear inequalities on the parameters (a 1 , b 1 ) , . . .

, (a 2 , b 2 ), . . .

, (a w , b w ): DISPLAYFORM2 Let the set of piecewise linear functions whose breakpoints satisfy the above be denoted by PWL 1 I,S for I ∈

I, S ∈ S.Given a particular I ∈ I, we define DISPLAYFORM3 The right hand side of the above equation is the problem of minimizing a convex objective subject to linear constraints.

Now, to solve (C.3), we need to simply solve the problem (C.5) for all I ∈ I, S ∈ S and pick the minimum.

Since |I| =

Now we will collect some straightforward observations that will be used often.

The following operations preserve the property of being representable by a ReLU DNN.

Proof.

Follows from (1.1) and the fact that a composition of affine transformations is another affine transformation.

Proof.

We prove this by induction on k. The base case is k = 1, i.e, we have a 2-layer ReLU DNN.

Since every activation node can produce at most one breakpoint in the piecewise linear function, we can get at most w 1 breakpoints, i.e., w 1 + 1 pieces.

Now for the induction step, assume that for some k ≥ 1, any R → R ReLU DNN with depth k + 1 and widths w 1 , . . .

, w k of the k hidden layers produces at most 2 k−1 · (w 1 + 1) · w 2 · . . .

· w k pieces.

Consider any R → R ReLU DNN with depth k + 2 and widths w 1 , . . .

, w k+1 of the k + 1 hidden layers.

Observe that the input to any node in the last layer is the output of a R → R ReLU DNN with depth k + 1 and widths w 1 , . . .

, w k .

By the induction hypothesis, the input to this node in the last layer is a piecewise linear function f with at most 2 k−1 · (w 1 + 1) · w 2 · . . .

· w k pieces.

When we apply the activation, the new function g(x) = max{0, f (x)}, which is the output of this node, may have at most twice the number of pieces as f , because each original piece may be intersected by the x-axis; see Figure 4 .

Thus, after going through the layer, we take an affine combination of w k+1 functions, each with at most 2 · (2 k−1 · (w 1 + 1) · w 2 · . . .

· w k ) pieces.

In all, we can therefore get at most 2·(2 k−1 ·(w 1 +1)·w 2 ·. .

.·w k )·w k+1 pieces, which is equal to 2 k ·(w 1 +1)·w 2 ·. .

.·w k ·w k+1 , and the induction step is completed.

Lemma D.5 has the following consequence about the depth and size tradeoffs for expressing functions with agiven number of pieces.

Lemma D.6.

Let f : R → R be a piecewise linear function with p pieces.

If f is represented by a ReLU DNN with depth k + 1, then it must have size at least 1 2 kp 1/k − 1.

Conversely, any piecewise linear function f that represented by a ReLU DNN of depth k + 1 and size at most s, can have at most ( 2s k ) k pieces.

Proof.

Let widths of the k hidden layers be w 1 , . . .

, w k .

By Lemma D.5, we must have DISPLAYFORM0 By the AM-GM inequality, minimizing the size w 1 + w 2 + . . .

+ w k subject to (D.1), means setting w 1 + 1 = w 2 = . . .

= w k .

This implies that w 1 + 1 = w 2 = . . .

= w k ≥ 1 2 p 1/k .

The first statement follows.

The second statement follows using the AM-GM inequality again, this time with a restriction on w 1 + w 2 + . . .

+ w k .

@highlight

This paper 1) characterizes functions representable by ReLU DNNs, 2) formally studies the benefit of depth in such architectures,  3) gives an algorithm to implement empirical risk minimization to global optimality for two layer ReLU nets.