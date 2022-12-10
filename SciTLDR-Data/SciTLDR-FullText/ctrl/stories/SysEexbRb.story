Due to the success of deep learning to solving a variety of challenging machine learning tasks, there is a rising interest in understanding loss functions for training neural networks from a theoretical aspect.

Particularly, the properties of critical points and the landscape around them are of importance to determine the convergence performance of optimization algorithms.

In this paper, we provide a necessary and sufficient characterization of the analytical forms for the critical points (as well as global minimizers) of the square loss functions for linear neural networks.

We show that the analytical forms of the critical points characterize the values of the corresponding loss functions as well as the necessary and sufficient conditions to achieve global minimum.

Furthermore, we exploit the analytical forms of the critical points to characterize the landscape properties for the loss functions of linear neural networks and shallow ReLU networks.

One particular conclusion is that: While the loss function of linear networks has no spurious local minimum, the loss function of one-hidden-layer nonlinear networks with ReLU activation function does have local minimum that is not global minimum.

In the past decade, deep neural networks BID8 have become a popular tool that has successfully solved many challenging tasks in a variety of areas such as machine learning, artificial intelligence, computer vision, and natural language processing, etc.

As the understandings of deep neural networks from different aspects are mostly based on empirical studies, there is a rising need and interest to develop understandings of neural networks from theoretical aspects such as generalization error, representation power, and landscape (also referred to as geometry) properties, etc.

In particular, the landscape properties of loss functions (that are typically nonconex for neural networks) play a central role to determine the iteration path and convergence performance of optimization algorithms.

One major landscape property is the nature of critical points, which can possibly be global minima, local minima, saddle points.

There have been intensive efforts in the past into understanding such an issue for various neural networks.

For example, it has been shown that every local minimum of the loss function is also a global minimum for shallow linear networks under the autoencoder setting and invertibility assumptions BID1 and for deep linear networks BID11 ; BID14 ; Yun et al. (2017) respectively under different assumptions.

The conditions on the equivalence between local minimum or critical point and global minimum has also been established for various nonlinear neural networks Yu & Chen (1995) ; BID9 ; BID15 ; BID17 ; BID6 under respective assumptions.

However, most previous studies did not provide characterization of analytical forms for critical points of loss functions for neural networks with only very few exceptions.

In BID1 , the authors provided an analytical form for the critical points of the square loss function of shallow linear networks under certain conditions.

Such an analytical form further helps to establish the landscape properties around the critical points.

Further in BID13 , the authors characterized certain sufficient form of critical points for the square loss function of matrix factorization problems and deep linear networks.

The focus of this paper is on characterizing the sufficient and necessary forms of critical points for broader scenarios, i.e., shallow and deep linear networks with no assumptions on data matrices and network dimensions, and shallow ReLU networks over certain parameter space.

In particular, such analytical forms of critical points capture the corresponding loss function values and the necessary and sufficient conditions to achieve global minimum.

This further enables us to establish new landscape properties around these critical points for the loss function of these networks under general settings, and provides alternative (yet simpler and more intuitive) proofs for existing understanding of the landscape properties.

OUR CONTRIBUTION 1) For the square loss function of linear networks with one hidden layer, we provide a full (necessary and sufficient) characterization of the analytical forms for its critical points and global minimizers.

These results generalize the characterization in BID1 to arbitrary network parameter dimensions and any data matrices.

Such a generalization further enables us to establish the landscape property, i.e., every local minimum is also a global minimum and all other critical points are saddle points, under no assumptions on parameter dimensions and data matrices.

From a technical standpoint, we exploit the analytical forms of critical points to provide a new proof for characterizing the landscape around the critical points under full relaxation of assumptions, where the corresponding approaches in BID1 are not applicable.

As a special case of linear networks, the matrix factorization problem satisfies all these landscape properties.2) For the square loss function of deep linear networks, we establish a full (necessary and sufficient) characterization of the analytical forms for its critical points and global minimizers.

Such characterizations are new and have not been established in the existing art.

Furthermore, such analytical form divides the set of non-global-minimum critical points into different categories.

We identify the directions along which the loss function value decreases for two categories of the critical points, for which our result directly implies the equivalence between the local minimum and the global minimum.

For these cases, our proof generalizes the result in BID11 under no assumptions on the network parameter dimensions and data matrices.3) For the square loss function of one-hidden-layer nonlinear neural networks with ReLU activation function, we provide a full characterization of both the existence and the analytical forms of the critical points in certain types of regions in the parameter space.

Particularly, in the case where there is one hidden unit, our results fully characterize the existence and the analytical forms of the critical points in the entire parameter space.

Such characterization were not provided in previous work on nonlinear neural networks.

Moreover, we apply our results to a concrete example to demonstrate that both local minimum that is not a global minimum and local maximum do exist in such a case.

Analytical forms of critical points: Characterizing the analytical form of critical points for loss functions of neural networks dates back to BID1 , where the authors provided an analytical form of the critical points for the square loss function of linear networks with one hidden layer.

In BID13 , the authors provided a sufficient condition of critical points of a generic function, i.e., the fixed point of invariant groups.

They then characterized certain sufficient forms of critical points for the square loss function of matrix factorization problems and deep linear networks, whereas our results provide sufficient and necessary forms of critical points for deep linear networks via a different approach.

Properties of critical points: BID1 ; BID0 studied the linear autoencoder with one hidden layer and showed the equivalence between the local minimum and the global minimum.

Moreover, BID2 generalized these results to the complex-valued autoencoder setting.

The deep linear networks were studied by some recent work BID11 ; BID14 Yun et al. (2018) , in which the equivalence between the local minimum and the global minimum was established respectively under different assumptions.

Particularly, Yun et al. (2017) established a necessary and sufficient condition for a critical point of the deep linear network to be a global minimum.

A similar result was established in BID7 for deep linear networks under the setting that the widths of intermediate layers are larger than those of the input and output layers.

The effect of regularization on the critical points for a two-layer linear network was studied in Taghvaei et al. (2017) .For nonlinear neural networks, Yu & Chen (1995) studied a nonlinear neural network with one hidden layer and sigmoid activation function, and showed that every local minimum is also a global minimum provided that the number of input units equals the number of data samples.

BID9 considered a class of multi-layer nonlinear networks with a pyramidal structure, and showed that all critical points of full column rank achieve the zero loss when the sample size is less than the input dimension.

These results were further generalized to a larger class of nonlinear networks in BID15 , in which they also showed that critical points with non-degenerate Hessian are global minimum.

BID3 b) connected the loss surface of deep nonlinear networks with the Hamiltonian of the spin-glass model under certain assumptions and characterized the distribution of the local minimum.

BID11 further eliminated some of the assumptions in BID3 , and established the equivalence between the local minimum and the global minimum by reducing the loss function of the deep nonlinear network to that of the deep linear network.

BID17 showed that a two-layer nonlinear network has no bad differentiable local minimum.

BID6 studied a one-hidden-layer nonlinear neural network with the parameters restricted in a set of directions of lines, and showed that most local minima are global minima.

Tian (2017) considered a two-layer ReLU network with Gaussian input data, and showed that critical points in certain region are non-isolated and characterized the critical-point-free regions.

Geometric curvature BID10 established the gradient dominance condition of deep linear residual networks, and Zhou & Liang (2017) further established the gradient dominance condition and regularity condition around the global minimizers for deep linear, deep linear residual and shallow nonlinear networks.

BID12 studied the property of the Hessian matrix for deep linear residual networks.

The local strong convexity property was established in BID16 for overparameterized nonlinear networks with one hidden layer and quadratic activation functions, and was established in Zhong et al. (2017) for a class of nonlinear networks with one hidden layer and Gaussian input data.

Zhong et al. (2017) further established the local linear convergence of gradient descent method with tensor initialization.

BID18 studied a one-hidden-layer nonlinear network with a single output, and showed that the volume of sub-optimal differentiable local minima is exponentially vanishing in comparison with the volume of global minima.

BID5 investigated the saddle points in deep neural networks using the results from statistical physics and random matrix theory.

Notation: The pseudoinverse, column space and null space of a matrix M are denoted by M † , col(M ) and ker(M ), respectively.

For any index sets I, J ⊂ N, M I,J denotes the submatrix of M formed by the entries with the row indices in I and the column indices in J. For positive integers i ≤ j, we define i : j = {i, i + 1, . . . , j − 1, j}. The projection operator onto a linear subspace V is denoted by P V .

In this section, we study linear neural networks with one hidden layer.

Suppose we have an input data matrix X ∈ R d0×m and a corresponding output data matrix Y ∈ R d2×m , where there are in total m data samples.

We are interested in learning a model that maps from X to Y via a linear network with one hidden layer.

Specifically, we denote the weight parameters between the output layer and the hidden layer of the network as A 2 ∈ R d2×d1 , and denote the weight parameters between the hidden layer and the input layer of the network as A 1 ∈ R d1×d0 .

We are interested in the square loss function of this linear network, which is given by DISPLAYFORM0 Note that in a special case where X = I, L reduces to a loss function for the matrix factorization problem, to which all our results apply.

The loss function L has been studied in BID1 under the assumptions that d 2 = d 0 ≥ d 1 and the matrices XX , Y X (XX ) −1 XY are invertible.

In our study, no assumption is made on either the parameter dimensions or the invertibility of the data matrices.

Such full generalization of the results in BID1 turns out to be critical for our study of nonlinear shallow neural networks in Section 4.We further define Σ := Y X † XY and denote its full singular value decomposition as U ΛU .

Suppose that Σ has r distinct positive singular values σ 1 > · · · > σ r > 0 with multiplicities m 1 , . . .

, m r , respectively, and hasm zero singular values.

Recall that DISPLAYFORM1 Our first result provides a full characterization of all critical points of L. Theorem 1 (Characterization of critical points).

All critical points of L are necessarily and sufficiently characterized by a matrix L 1 ∈ R d1×d0 , a block matrix V ∈ R d2×d1 and an invertible matrix C ∈ R d1×d1 via DISPLAYFORM2 (2) DISPLAYFORM3 , where both V i ∈ R mi×pi and V ∈ Rm ×p consist of orthonormal columns with the number of columns DISPLAYFORM4 Theorem 1 characterizes the necessary and sufficient forms for all critical points of L. Intuitively, the matrix C captures the invariance of the product A 2 A 1 under an invertible transform, and L 1 captures the degree of freedom of the solution set for linear systems.

In general, the set of critical points is uncountable and cannot be fully listed out.

However, the analytical forms in eqs. (1) and (2) do allow one to construct some critical points of L by specifying choices of L 1 , V , C that fulfill the condition in eq. (3).

For example, choosing L 1 = 0 guarantees eq. (3), in which case eqs. (1) and (2) yield a critical point (C −1 V U Y X † , U V C) for any invertible matrix C and any block matrix V that takes the form specified in Theorem 1.

For nonzero L 1 , one can fix a proper V and solve the linear equation on C in eq. (3).

If a solution exists, we then obtain the form of a corresponding critical point.

We further note that the analytical structures of the critical points are more important, which have direct implications on the global optimality conditions and landscape properties as we show in the remaining part of the section.

Remark 1.

We note that the block pattern parameters {p i } r i=1 andp denote the number of columns of {V i } r i=1 and V , respectively, and their sum equals the rank of A 2 , i.e., DISPLAYFORM5 The parameters p i , i = 1, . . .

, r,p of V contain all useful information of the critical points that determine the function value of L as presented in the following proposition.

DISPLAYFORM6 Proposition 1 evaluates the function value L at a critical point using the parameters {p i } r i=1 .

To explain further, recall that the data matrix Σ has each singular value σ i with multiplicity m i .

For each i, the critical point captures p i out of m i singular values σ i .

Hence, for a σ i with larger value (i.e., a smaller index i), it is desirable that a critical point captures a larger number p i of them.

In this way, the critical point captures more important principle components of the data so that the value of the loss function is further reduced as suggested by Proposition 1.

In summary, the parameters {p i } r i=1 characterize how well the learned model fits the data in terms of the value of the loss function.

Moreover, the parameters {p i } r i=1 also determine a full characterization of the global minimizers as given below.

Proposition 2 (Characterization of global minimizers).

A critical point (A 1 , A 2 ) of L is a global minimizer if and only if it falls into the following two cases.

DISPLAYFORM7 The analytical form of any global minimizer can be obtained from Theorem 1 with further specification to the above two cases.

Proposition 2 establishes the neccessary and sufficient conditions for any critical point to be a global minimizer.

If the data matrix Σ has a large number of nonzero singular values, i.e., the first case, one needs to exhaust the representation budget (i.e., rank) of A 2 and capture as many large singular values as the rank allows to achieve the global minimum; Otherwise, A 2 of a global minimizer can be non-full rank and still captures all nonzero singular values.

Note that A 2 must be full rank in the case 1, and so is A 1 if we further adopt the assumptions on the network size and data matrices in BID1 .

Furthermore, the parameters {p i } r i=1 naturally divide all non-global-minimum critical points (A 1 , A 2 ) of L into the following two categories.• (Non-optimal order): The matrix V specified in Theorem 1 satisfies that there exists 1 ≤ i < j ≤ r such that p i < m i and p j > 0.• (Optimal order): rank(A 2 ) < min{d 2 , d 1 } and the matrix V specified in Theorem 1 satisfies that DISPLAYFORM8 To understand the above two categories, note that a critical point of L with non-optimal order captures a smaller singular value σ j (since p j > 0) while skipping a larger singular value σ i with a lower index i < j (since p i < m i ), and hence cannot be a global minimizer.

On the other hand, although a critical point of L with optimal order captures the singular values in the optimal (i.e., decreasing) order, it does not fully utilize the representation budget of A 2 (because A 2 is non-full rank) to further capture nonzero singular values and reduce the function value, and hence cannot be a global minimizer either.

Next, we show that these two types of non-global-minimum critical points have different landscape properties around them.

Throughout, a matrix M is called the perturbation of M if it lies in an arbitrarily small neighborhood of M .Proposition 3 (Landscape around critical points).

The critical points of L have the following landscape properties.1.

A non-optimal-order critical point (A 1 , A 2 ) has a perturbation ( A 1 , A 2 ) with rank( A 2 ) = rank(A 2 ), which achieves a lower function value; 2.

An optimal-order critical point (A 1 , A 2 ) has a perturbation ( A 1 , A 2 ) with rank( A 2 ) = rank(A 2 ) + 1, which achieves a lower function value; 3.

Any point in X := {(A 1 , A 2 ) : A 2 A 1 X = 0} has a perturbation (A 1 , A 2 ), which achieves a higher function value;As a consequence, items 1 and 2 imply that any non-global-minimum critical point has a descent direction, and hence cannot be a local minimizer.

Thus, any local minimizer must be a global minimizer.

Item 3 implies that any point has an ascent direction whenever the output is nonzero.

Hence, there does not exist any local/global maximizer in X .

Furthermore, item 3 together with items 1 and 2 implies that any non-global-minimum critical point in X has both descent and ascent directions, and hence must be a saddle point.

We summarize these facts in the following theorem.

Theorem 2 (Landscape of L).

The loss function L satisfies: 1) every local minimum is also a global minimum; 2) every non-global-minimum critical point in X is a saddle point.

We note that the saddle points in Theorem 2 can be non-strict when the data matrices are singular.

As an illustrative example, consider the following loss function of a shallow linear network L(a 2 , a 1 ) = 1 2 (a 2 a 1 x − y) 2 , where a 1 , a 2 , x and y are all scalars.

Consider the case y = 0.

Then, the Hessian at the saddle point a 1 = 0, a 2 = 1 is [x 2 , 0; 0, 0], which does not have any negative eigenvalue.

From a technical point of view, the proof of item 1 of Proposition 3 applies that in BID0 and generalizes it to the setting where Σ can have repeated singular values and may not be invertible.

To further understand the perturbation scheme from a high level perspective, note that non-optimalorder critical points capture a smaller singular value σ j instead of a larger one σ i with i < j. Thus, one naturally perturbs the singular vector corresponding to σ j along the direction of the singular vector corresponding to σ i .

Such a perturbation scheme preserves the rank of A 2 and reduces the value of the loss function.

More importantly, the proof of item 2 of Proposition 3 introduces a new technique.

As a comparison, BID1 proves a similar result as item 2 using the strict convexity of the function, which requires the parameter dimensions to satisfy d 2 = d 0 ≥ d 1 and the data matrices to be invertible.

In contrast, our proof completely removes these restrictions by introducing a new perturbation direction and exploiting the analytical forms of critical points in eqs. (1) and (2) and the condition in eq. (3).

The accomplishment of the proof further requires careful choices of perturbation parameters as well as judicious manipulations of matrices.

We refer the reader to the supplemental materials for more details.

As a high level understanding, since optimal-order critical points capture the singular values in an optimal (i.e., decreasing) order, the previous perturbation scheme for non-optimal-order critical points does not apply.

Instead, we increase the rank of A 2 by one in a way that the perturbed matrix captures the next singular value beyond the ones that have already been captured so that the value of the loss function can be further reduced.

In this section, we study deep linear networks with ≥ 2 layers.

We denote the weight parameters between the layers as A k ∈ R d k ×d k−1 for k = 1, . . .

, , respectively.

The input and output data are denoted by X ∈ R d0×m , Y ∈ R d ×m , respectively.

We are interested in the square loss function of deep linear networks, which is given by DISPLAYFORM0 , respectively, andm(k) zero singular values.

Our first result provides a full characterization of all critical points of L D , where we denote DISPLAYFORM1 Theorem 3 (Characterization of critical points).

All critical points of L D are necessarily and sufficiently characterized by matrices DISPLAYFORM2 . .

, A can be individually expressed out recursively via the following two equations: DISPLAYFORM3 DISPLAYFORM4 Note that the forms of the individual parameters A 1 , . . .

, A can be obtained as follows by recursively applying eqs. (4) and (5).

First, eq. (5) with k = 0 yields the form of A ( ,2) .

Then, eq. (4) with k = 0 and the form of A ( ,2) yield the form of A 1 .

Next, eq. (5) with k = 1 yields the form of A ( ,3) , and then, eq. (4) with k = 1 and the forms of A ( ,3) , A 1 further yield the form of A 2 .

Inductively, one obtains the expressions of all individual parameter matrices.

Furthermore, the first condition in eq. FORMULA13 is a consistency condition that guarantees that the analytical form for the entire product of parameter matrices factorizes into the forms of individual parameter matrices.

Similarly to shallow linear networks, while the set of critical points here is also uncountable, Theorem 3 suggests ways to obtain some critical points.

For example, if we set L k = 0 for all k (i.e., eq. (6) is satisfied), we can obtain the form of critical points for any invertible C k and proper V k with the structure specified in Theorem 3.

For nonzero L k , eq. (6) needs to be verified for given C k and V k to determine a critical point.

Similarly to shallow linear networks, the parameters {p i (0)} r (0) i=1 ,p(0) determine the value of the loss function at the critical points and further specify the analytical form for the global minimizers, as we present in the following two propositions.

DISPLAYFORM5 DISPLAYFORM6 In particular, A ( ,2) can be non-full rank with rank(A ( ,2) ) = DISPLAYFORM7 The analytical form of any global minimizer can be obtained from Theorem 3 with further specification to the above two cases.

In particular for case 1, if we further adopt the invertibility assumptions on data matrices as in BID1 and assume that all parameter matrices are square, then all global minima must correspond to full rank parameter matrices.

We next exploit the analytical forms of the critical points to further understand the landscape of the loss function L D .

It has been shown in BID11 that every local minimum of L D is also a global minimum, under certain conditions on the parameter dimensions and the invertibility of the data matrices.

Here, our characterization of the analytical forms for the critical points allow us to understand such a result from an alternative viewpoint.

The proofs for certain cases (that we discuss below) are simpler and more intuitive, and no assumption is made on the data matrices and dimensions of the network.

Similarly to shallow linear networks, we want to understand the local landscape around the critical points.

However, due to the effect of depth, the critical points of L D are more complicated than those of L. Among them, we identify the following subsets of the non-global-minimum critical DISPLAYFORM8 • (Deep-non-optimal order): There exist 0 ≤ k ≤ − 2 such that the matrix V k specified in Theorem 3 satisfies that there exist 1 ≤ i < j ≤ r(k) such that p i (k) < m i (k) and p j (k) > 0.• (Deep-optimal order): (A , A −1 ) is not a global minimizer of L D with A ( −2,1) being fixed, rank(A ) < min{d , d −1 }, and the matrix V −2 specified in Theorem 3 satisfies that DISPLAYFORM9 The following result summarizes the landscape of L D around the above two types of critical points.

The loss function L D has the following landscape properties.

deep-non-optimal-order critical point (A 1 , . . . , A ) has a perturbation (A 1 , . . .

, A k+1 , . . .

, A ) with rank( A ) = rank(A ), which achieves a lower function value.

2.

A deep-optimal-order critical point (A 1 , . . . , A ) has a perturbation (A 1 , . . .

, A −1 , A ) with rank( A ) = rank(A ) + 1, which achieves a lower function value.

3.

Any point in X D := {(A 1 , . . .

, A ) : A ( ,1) X = 0} has a perturbation (A 1 , . . .

, A ) that achieves a higher function value.

Consequently, 1) every local minimum of L D is also a global minimum for the above two types of critical points; and 2) every critical point of these two types in X D is a saddle point.

Theorem 4 implies that the landscape of L D for deep linear networks is similar to that of L for shallow linear networks, i.e., the pattern of the parameters {p i (k)} r(k) i=1 implies different descent directions of the function value around the critical points.

Our approach does not handle the remaining set of non-global minimizers, i.e., there exists q ≤ −1 such that (A , . . .

, A q ) is a global minimum point of L D with A (q−1,1) being fixed, and A ( ,q) is of optimal order.

It is unclear how to perturb the intermediate weight parameters using their analytical forms for deep networks , and we leave this as an open problem for the future work.

In this section, we study nonlinear neural networks with one hidden layer.

In particular, we consider nonlinear networks with ReLU activation function σ : R → R that is defined as σ(x) := max{x, 0}. Our study focuses on the set of differentiable critical points.

The weight parameters between the layers are denoted by A 2 ∈ R d2×d1 , A 1 ∈ R d1×d0 , respectively, and the input and output data are denoted by X ∈ R d0×m , Y ∈ R d2×m , respectively.

We are interested in the square loss function which is given by DISPLAYFORM0 where σ acts on A 1 X entrywise.

Existing studies on nonlinear networks characterized the sufficient conditions for critical points being global minimum BID9 Since the activation function σ is piecewise linear, the entire parameter space can be partitioned into disjoint cones.

In particular, we consider the set of cones K I×J where I ⊂ {1, . . .

, d 1 }, J ⊂ {1, . . .

, m} that satisfy DISPLAYFORM1 where "≥" and "<" represent entrywise comparisons.

Within K I×J , the term σ(A 1 X) activates only the entries σ(A 1 X) I:J , and the corresponding loss function L N is equivalent to DISPLAYFORM2 Hence, within K I×J , L N reduces to the loss of a shallow linear network with parameters ((A 2 ) :,I , (A 1 ) I,: ) and input & output data pair (X :,J , Y :,J ).

Note that our results on shallow linear networks in Section 2 are applicable to all parameter dimensions and data matrices.

Thus, Theorem 1 fully characterizes the forms of critical points of L N in K I×J .

Moreover, the existence of such critical points can be analytically examined by substituting their forms into eq. (8).

In summary, we obtain the following result, where we denote Σ J := Y :,J X † :,J X :,J Y :,J with the full singular value decomposition U J Λ J U J , and suppose that Σ J has r(J) distinct positive singular values σ 1 (J) > · · · > σ r(J) (J) with multiplicities m 1 , . . .

, m r(J) , respectively, andm(J) zero singular values.

Proposition 6 (Characterization of critical points).

All critical points of L N in K I×J for any I ⊂ {1, . . .

, d 1 }, J ⊂ {1, . . .

, m} are necessarily and sufficiently characterized by an L 1 ∈ R |I|×d0 , a block matrix V ∈ R d2×|I| and an invertible matrix C ∈ R |I|×|I| such that DISPLAYFORM3 DISPLAYFORM4 ×p consist of orthonormal columns with p i ≤ m i for i = 1, . . .

, r(J),p ≤m such that DISPLAYFORM5 Moreover, a critical point in K I×J exists if and only if there exists such C, V , L 1 that DISPLAYFORM6 Other entries of A 1 X < 0.To further illustrate, we consider a special case where the nonlinear network has one unit in the hidden layer, i.e., d 1 = 1, in which case A 1 and A 2 are row and column vectors, respectively.

Then, the entire parameter space can be partitioned into disjoint cones taking the form of K I×J , and I = {1} is the only nontrivial choice.

We obtain the following result from Proposition 6.Proposition 7 (Characterization of critical points).

Consider L N with d 1 = 1 and any J ⊂ {1, . . .

, m}. Then, any nonzero critical point of L N within K {1}×J can be necessarily and sufficiently characterized by an 1 ∈ R 1×d0 , a block unit vector v ∈ R d2×1 and a scalar c ∈ R such that DISPLAYFORM7 Specifically, v is a unit vector that is supported on the entries corresponding to the same singular value of Σ J .

Moreover, a nonzero critical point in K {1}×J exists if and only if there exist such c, v, 1 that satisfy DISPLAYFORM8 DISPLAYFORM9 We note that Proposition 7 characterizes both the existence and the forms of critical points of L N over the entire parameter space for nonlinear networks with a single hidden unit.

The condition in eq. FORMULA24 is guaranteed because P ker(v) = 0 for v = 0.To further understand Proposition 7, suppose that there exists a critical point in K {1}×J with v being supported on the entries that correspond to the i-th singular value of Σ J .

Then, Proposition 1 implies that DISPLAYFORM10 In particular, the critical point achieves the local minimum DISPLAYFORM11 .

This is because in this case the critical point is full rank with an optimal order, and hence corresponds to the global minimum of the linear network in eq. (9).

Since the singular values of Σ J may vary with the choice of J, L N may achieve different local minima in different cones.

Thus, local minimum that is not global minimum can exist for L N .

The following proposition concludes this fact by considering a concrete example.

Proposition 8.

For one-hidden-layer nonlinear neural networks with ReLU activation function, there exists local minimum that is not global minimum, and there also exists local maximum.

FORMULA13 and FORMULA19 hold if c −1 (v) 1,: ≥ 0, ( 1 ) 1,: < 0.

Similarly to the previous case, choosing c = 1, v = (1, 0) , 1 = (−1, 0) yields a local minimum that achieves the function value L n = 2.

Hence, local minimum that is not global minimum does exist.

Moreover, in the cone K I×J with I = {1}, J = ∅, the function L N remains to be the constant 5 2 , and all points in this cone are local minimum or local maximum.

Thus, the landscape of the loss function of nonlinear networks is very different from that of the loss function of linear networks.

In this paper, we provide full characterization of the analytical forms of the critical points for the square loss function of three types of neural networks, namely, shallow linear networks, deep linear networks, and shallow ReLU nonlinear networks.

We show that such analytical forms of the critical points have direct implications on the values of the corresponding loss functions, achievement of global minimum, and various landscape properties around these critical points.

As a consequence, the loss function for linear networks has no spurious local minimum, while such point does exist for nonlinear networks with ReLU activation.

In the future, it is interesting to further explore nonlinear neural networks.

In particular, we wish to characterize the analytical form of critical points for deep nonlinear networks and over the full parameter space.

Such results will further facilitate the understanding of the landscape properties around these critical points.

Notations: For any matrix M , denote vec(M ) as the column vector formed by stacking its columns.

Denote the Kronecker product as "⊗".

Then, the following useful relationships hold for any dimension compatible matrices M , U , V , W : DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Recall that a point DISPLAYFORM4 DISPLAYFORM5 We first prove eqs. (1) and (2) .

DISPLAYFORM6 Next, we derive the form of A 2 .

Recall the full singular value decomposition Σ = U ΛU , where Λ is a diagonal matrix with distinct singular values σ 1 > . . .

> σ r > 0 and multiplicities m 1 , . . .

, m r , respectively.

We also assume that there arem number of zero singular values in Λ. Using the fact that P col(A2) = U P col(U A2) U , the last equality in eq. (26) reduces to DISPLAYFORM7 By the multiplicity pattern of the singular values in Λ, P col(U A2) must be block diagonal.

Specifically, we can write P col(U A2) = diag( P 1 , . . .

, P r , P), where P i ∈ R mi×mi and P ∈ Rm ×m .Also, since P col(U A2) is a projection, P 1 , . . . , P r , P must all be projections.

Note that P col(U A2) has rank rank(A 2 ), and suppose that P 1 , . . .

, P r , P have ranks p 1 , . . .

, p r ,p, respectively.

Then, we must have p i ≤ m i for i = 1, . . .

, r,p ≤m and r i=1 p i +p = rank(A 2 ).

Also, note that each projection can be expressed as P i = V i V i with V i ∈ R mi×pi , V ∈ Rm ×p consisting of orthonormal columns.

Hence, we can write P col(U A2) = V V where V = diag(V 1 , . . .

, V r , V ).

We then conclude that P col(A2) = U P col(U A2) U = U V V U .

Thus, A 2 has the same column space as U V , and there must exist an invertible matrix DISPLAYFORM8 Then, plugging A † 2 = C −1 V U into eq. (25) yields the desired form of A 1 .We now prove eq. (3).

Note that the above proof is based on the equations DISPLAYFORM9 Hence, the forms of A 1 , A 2 in eqs. (1) and (2) need to further satisfy ∇ A2 L = 0.

By eq. FORMULA19 and the form of A 2 , we obtain that DISPLAYFORM10 This expression, together with the form of A 1 in eq. (1), implies that DISPLAYFORM11 where (i) uses the fact that X † XX = X , (ii) uses the fact that the block pattern of V is compatible with the multiplicity pattern of the singular values in Λ, and hence V V ΛV = ΛV .

On the other hand, we also obtain that DISPLAYFORM12 Thus, to satisfy ∇ A2 L = 0 in eq. FORMULA12 , we require that DISPLAYFORM13 which is equivalent to DISPLAYFORM14 Lastly, note that (I − U V (U V ) ) = P col(U V ) ⊥ , and (I − V V ) = P ker(V ) , which concludes the proof.

By expansion we obtain that L = DISPLAYFORM0 .

Consider any (A 1 , A 2 ) that satisfies eq. FORMULA4 , we have shown that such a point also satisfies eq. (27), which further yields that DISPLAYFORM1 where (i) follows from the fact that Tr( P col(A2) Σ P col(A2) ) = Tr( P col(A2) Σ), and (ii) uses the fact that P col(A2) = U P col(U A2) U .

In particular, a critical point (A 1 , A 2 ) satisfies eq. (28).

Moreover, using the form of the critical point A 2 = U V C, eq. FORMULA20 further becomes DISPLAYFORM2 where (i) is due to P col(V C) = P col(V ) = V V , and (ii) utilizes the block pattern of V and the multiplicity pattern of Λ that are specified in Theorem 1.

(1): Consider a critical point (A 1 , A 2 ) with the forms given by Theorem 1.

By choosing L 1 = 0, the condition in eq. FORMULA4 is guaranteed.

Then, we can specify a critical point with any V that satisfies the block pattern specified in Theorem 1, i.e., we can choose any p i , i = 1, . . .

, r,p such that p i ≤ m i for i = 1, . . .

, r,p ≤m and DISPLAYFORM0 m i , the global minimum value is achieved by a full rank A 2 with rank(A 2 ) = min{d 2 , d 1 } and DISPLAYFORM1 That is, the singular values are selected in a decreasing order to minimize the function value.(2): If (A 2 , A 1 ) is a global minimizer and min{d y , d} > r i=1 m i , the global minimum can be achieved by choosing p i = m i for all i = 1, . . .

, r andp ≥ 0.

In particular, we do not need a full rank A 2 to achieve the global minimum.

For example, we can choose rank(A 2 ) = r i=1 m i < min{d y , d} with p i = m i for all i = 1, . . .

, r andp = 0.

We first prove item 1.

Consider a non-optimal-order critical point (A 1 , A 2 ).

By Theorem 1, we can write A 2 = U V C where V = [diag(V 1 , . . .

, V r , V ), 0] and V i , i = 1, . . .

, r, V consist of orthonormal columns.

Define the orthonormal block diagonal matrix Since (A 1 , A 2 ) is a non-optimal-order critical point, there exists 1 ≤ i < j ≤ r such that p i < m i and p j > 0.

Then, consider the following perturbation of U S for some > 0.

DISPLAYFORM0 DISPLAYFORM1 with which we further define the perturbation matrix A 2 = M S V C. Also, let the perturbation matrix A 1 be generated by eq. (1) with U ← M and V ← S V .

Note that with this construction, ( A 1 , A 2 ) satisfies eq. (25), which further implies eq. (27) for ( A 1 , A 2 ), i.e., A 2 A 1 X = P col( A2) Y X † X. Thus, eq. (28) holds for the point ( A 1 , A 2 ), and we obtain that DISPLAYFORM2 where the last equality uses the fact that S ΛS = Λ, as can be observed from the block pattern of S and the multiplicity pattern of Λ. Also, by the construction of M and the form of S V , a careful calculation shows that only the i, j-th diagonal elements of P col(S U M S V ) have changed, i.e., DISPLAYFORM3 As the index i, j correspond to the singular values σ i , σ j , respectively, and σ i > σ j , one obtain that DISPLAYFORM4 Thus, the construction of the point ( A 2 , A 1 ) achieves a lower function value for any > 0.

Letting → 0 and noticing that M is a perturbation of U S, the point ( A 2 , A 1 ) can be in an arbitrary neighborhood of (A 2 , A 1 ).

Lastly, note that rank( A 2 ) = rank(A 2 ).

This completes the proof of item 1.Next, we prove item 2.

Consider an optimal-order critical point (A 1 , A 2 ).

Then, A 2 must be non-full rank, since otherwise a full rank A 2 with optimal order corresponds to a global minimizer by Proposition 2.

Since there exists some k ≤ r such that 0] .

Using this expression, eq. (1) yields that DISPLAYFORM5 DISPLAYFORM6 We now specify our perturbation scheme.

Recalling the orthonormal matrix S defined in eq. (29).

Then, we consider the following matrices for some 1 , 2 > 0 DISPLAYFORM7 For this purpose, we need to utilize the condition of critical points in eq. (3), which can be equivalently expressed as DISPLAYFORM8 (ii) ⇔ (CL 1 ) (rank(A2)+1):d1,: XY (I − U S :,1:(q−1) (U S :,1:(q−1) ) ) = 0where (i) follows by taking the transpose and then simplifying, and (ii) uses the fact that V = SS V = S :,1:(q−1) in the case of optimal-order critical point.

Calculating the function value at ( A 1 , A 2 ), we obtain that DISPLAYFORM9 .

We next simplify the above three trace terms using eq. (31).

For the first trace term, observe that DISPLAYFORM10 2 Tr(S :,q ΛS :,q ) where (i) follows from eq. (31) as S :,q is orthogonal to the columns of S :,1:(q−1) .

For the second trace term, we obtain that DISPLAYFORM11 = 2Tr( 2 U S :,q (CL 1 ) (rank(A2)+1),: XY U V diag (U V diag ) ) + 2Tr( 1 2 U S :,q S :,q ΛSS V diag (U V diag ) ) (i) = 2Tr( 2 U S :,q (CL 1 ) (rank(A2)+1),: XY U V diag (U V diag ) ) + 2Tr( 1 2 σ k U S :,q e q S V diag (U V diag ) )(ii) = 2Tr( 2 U S :,q (CL 1 ) (rank(A2)+1),: XY U V diag (U V diag ) ), where (i) follows from S :,q ΛS = σ k e q , and (ii) follows from e q S V diag = 0.

For the third trace term, we obtain that 2Tr(P Y ) = 2Tr( 2 U S :,q (CL 1 ) (rank(A2)+1),: XY ) + 2Tr( 1 2 U S :,q (U S :,q ) Σ) = 2Tr( 2 U S :,q (CL 1 ) (rank(A2)+1),: XY ) + 2Tr( 1 2 S :,q ΛS :,q ).Combining the expressions for the three trace terms above, we conclude that Consider a critical point (A 1 , . . .

, A ) so that eq. FORMULA4

Observe that the product matrix A ( ,2) is equivalent to the class of matrices B 2 ∈ R min{d ,...,d2}×d1 .Consider a critical point (B 2 , A 1 ) of the shallow linear network L :=

The proof is similar to that for shallow linear networks.

Consider a deep-non-optimal-order critical point (A 1 , . . .

, A ), and define the orthonormal block matrix S k using the blocks of V k in a similar way as eq. (29).

Then, A (l,k+2) takes the form A (l,k+2) = U k S k S k V k C k .

Since A (l,k+2) is of non-optimal order, there exists i < j < r(k) such that p i (k) < m i (k) and p j (k) > 0.

Thus, we perturb the j-th column of U k S k to be , and denote the resulting matrix as M k .Then, we perturb A to be A = M k (U k S k ) A so that A A ( −1,k+2) = M k S k V k C k .

Moreover, we generate A k+1 by eq. (4) with U k ← M k , V k ← S k V k .

Note that such construction satisfies eq. (32), and hence also satisfies eq. (34), which further yields that DISPLAYFORM0 With the above equation, the function value at this perturbed point is evaluated as DISPLAYFORM1 Then, a careful calculation shows that only the i, j-th diagonal elements of DISPLAYFORM2 have changed, and are Now consider a deep-optimal-order critical point (A 1 , . . .

, A ).

Note that with A ( −2,1) fixed to be a constant, the deep linear network reduces to a shallow linear network with parameters (A , A −1 ).

Since (A , A −1 ) is not a non-global minimum critical point of this shallow linear network and A is of optimal-order, we can apply the perturbation scheme in the proof of Proposition 3 to identify a perturbation ( A , A −1 ) with rank( A ) = rank(A ) + 1 that achieves a lower function value.

Consider any point in X D .

Since A ( ,1) X = 0, we can scale the nonzero row, say, the i-th row (A ) i,: A ( −1,1) X properly in the same way as that in the proof of Proposition 3 to increase the function value.

Lastly, item 1 and item 2 imply that every local minimum is a global minimum for these two types of critical points.

Moreover, combining items 1,2 and 3, we conclude that every critical point of these two types in X D is a saddle point.

<|TLDR|>

@highlight

We provide necessary and sufficient analytical forms for the critical points of the square loss functions for various neural networks, and exploit the analytical forms to characterize the landscape properties for the loss functions of these neural networks.