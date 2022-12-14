We study the error landscape of deep linear and nonlinear neural networks with the squared error loss.

Minimizing the loss of a deep linear neural network is a nonconvex problem, and despite recent progress, our understanding of this loss surface is still incomplete.

For deep linear networks, we present necessary and sufficient conditions for a critical point of the risk function to be a global minimum.

Surprisingly, our conditions provide an efficiently checkable test for global optimality, while such tests are typically intractable in nonconvex optimization.

We further extend these results to deep nonlinear neural networks and prove similar sufficient conditions for global optimality, albeit in a more limited function space setting.

Since the advent of AlexNet BID10 , deep neural networks have surged in popularity, and have redefined the state-of-the-art across many application areas of machine learning and artificial intelligence, such as computer vision, speech recognition, and natural language processing.

However, a concrete theoretical understanding of why deep neural networks work well in practice remains elusive.

From the perspective of optimization, a significant barrier is imposed by the nonconvexity of training neural networks.

Moreover, it was proved by BID2 that training even a 3-node neural network to global optimality is NP-Hard in the worst case, so there is little hope that neural networks have properties that make global optimization tractable.

Despite the difficulties of optimizing weights in neural networks, the empirical successes suggest that the local minima of their loss surfaces could be close to global minima; and several papers have recently appeared in the literature attempting to provide a theoretical justification for the success of these models.

For example, by relating neural networks to spherical spin-glass models from statistical physics, BID3 provided some empirical evidence that the increase of size of neural networks makes local minima close to global minima.

Another line of results BID16 BID14 BID15 BID13 provides conditions under which a critical point of the empirical risk is a global minimum.

Such results roughly involve proving that if full rank conditions of certain matrices (as well as some additional technical conditions) are satisfied, derivative of the risk being zero implies loss being zero.

However, these results are obtained under restrictive assumptions; for example, BID13 require the width of one of the hidden layers to be as large as the number of training examples.

BID14 and BID15 require the product of widths of two adjacent layers to be at least as large as the number of training examples, meaning that the number of parameters in the model must grow rapidly as we have more training data available.

Another recent paper BID4 provides a sufficient condition for global optimality when the neural network is composed of subnetworks with identical architectures connected in parallel and a regularizer is designed to control the number of parallel architectures.

Towards obtaining a more precise characterization of the loss-surfaces, a valuable conceptual simplification of deep nonlinear networks is deep linear neural networks, in which all activation functions are linear and the output of the entire network is a chained product of weight matrices with the input vector.

Although at first sight a deep linear model may appear overly simplistic, even its opti-mization is nonconvex, and only recently theoretical results on this problem have started emerging.

Interestingly, already in 1989, BID0 showed that some shallow linear neural networks have no local minima.

More recently, BID8 extended this result to deep linear networks and proved that any local minimum is also global while any other critical point is a saddle point.

Subsequently, BID11 provided a simpler proof that any local minimum is also global, with fewer assumptions than BID8 .

Motivated by the success of deep residual networks BID6 b) , BID5 investigated loss surfaces of deep linear residual networks and showed every critical point is a global minimum in a near-identity region; subsequently, Bartlett et al. (2017) extended this result to a nonlinear function space setting.

Inspired by this recent line of work, we study deep linear and nonlinear networks, in settings either similar to or more general than existing work.

We summarize our main contributions below.??? We provide both necessary and sufficient conditions for a critical point of the empirical risk to be a global minimum.

Specifically, Theorem 2.1 shows that if the hidden layers are wide enough, then a critical point of the risk function is a global minimum if and only if the product of all parameter matrices is full-rank.

In Theorem 2.2, we consider the case where some hidden layers have smaller width than both the input and output layers, and again provide necessary and sufficient conditions for global optimality.

In comparison, BID8 only proves that every critical point of the risk is either a global minimum or a saddle; it is an "existence" result without any computational implication.

In contrast, we present efficiently checkable conditions for distinguishing the two different types of critical points; we can even use these conditions while running optimization to test whether the critical points we encounter are saddle points or not, if desired.

It is also worth noting that such tests are intractable for general nonconvex optimization BID12 .???

Under the same assumption as BID5 on the data distribution, namely, a linear model with Gaussian noise, we can modify Theorem 2.1 to handle the population risk.

As a corollary, we not only recover Theorem 2.2 in BID5 , but also extend it to a strictly larger set, while removing their assumption that the true underlying linear model has a positive determinant.??? Motivated by (Bartlett et al., 2017) , we extend our results on deep linear networks to obtain sufficient conditions for global optimality in deep nonlinear networks, although only via a function space view; these are presented in Theorems 4.1 and 4.2.

In this section, we describe the problem formulation and notations for deep linear neural networks, state main results (Theorems 2.1 and 2.2), and explain their implication.

Suppose we have m input-output pairs, where the inputs are of dimension d x and outputs of dimension d y .

Let X ??? R dx??m be the data matrix and Y ??? R dy??m be the output matrix.

Suppose we have H hidden layers in the network, each having width d 1 , . . .

, d H .

For notational simplicity we DISPLAYFORM0 The weights between adjacent layers can be represented as matrices W k ??? R d k ??d k???1 , for k = 1, . . .

, H + 1, and the output of the network can be written as the product of weight matrices W H+1 , . . .

, W 1 and data matrix X: W H+1 W H ?? ?? ?? W 1 X. We consider minimizing the summation of squared error loss over all data points (i.e. empirical risk), DISPLAYFORM1 where W is a shorthand notation for the tuple (W 1 , . . .

, W H+1 ).Assumptions.

We assume that d x ??? m and d y ??? m, and that XX T and Y X T have full ranks.

These assumptions are common when we consider supervised learning problems with deep neural networks (e.g. BID8 ).

We also assume that the singular values of Y X T (XX T ) ???1 X are all distinct, which is made for notational simplicity and can be relaxed without too much difficulty.

Notation.

Given a matrix A, let ?? max (A) and ?? min (A) denote the largest and smallest singular values of A, respectively.

Let row(A), col(A), null(A), rank(A), and A F be respectively the row space, column space, null space, rank, and Frobenius norm of matrix A. Given a subspace V of R n , we denote V ??? as its orthogonal complement.

Given a set V, let V c denote the complement of V.Let us denote k := min i???{0,...,H+1} d i , and define p ??? argmin i???{0,...,H+1} d i .

That is, p is any layer with the smallest width, and k = d p is the width of that layer.

Here, p might not be unique, but our results hold for any layer p with smallest width.

Notice also that the product DISPLAYFORM2 Let?? ??? R dy??k be a matrix consisting of the first k columns of U .

We now present two main theorems for deep linear neural networks.

The theorems describe two sets, one for the case k = min{d x , d y } and the other for k < min{d x , d y }, inside which every critical point of L(W ) is a global minimum.

Moreover, the sets have another remarkable property that every critical point outside of these sets is a saddle point.

Previous works BID8 BID11 showed that any critical point is either a global minimum or a saddle point, without providing any condition to distinguish between the two; here, we take a step further and partition the domain of L(W ) into two sets clearly delineating one set which only contains global minima and the other set with only saddle points.

DISPLAYFORM0 2 is a saddle point.

Theorems 2.1 and 2.2 provide necessary and sufficient conditions for a critical point of L(W ) to be globally optimal.

From an algorithmic perspective, they provide easily checkable conditions, which we can use to determine if the critical point the algorithm encountered is a global optimum or not.

Given that L(W ) is nonconvex, it is interesting to have such efficient tests for global optimality, which is not possible in general BID12 .In BID5 , the authors consider minimizing population risk of linear residual networks: DISPLAYFORM1 They assume that x is drawn from a zero-mean distribution with a fixed covariance matrix, and y = Rx + ?? where ?? is iid standard Gaussian noise and R is the true underlying matrix with det(R) > 0.

With these assumptions they prove that whenever ?? max (W i ) < 1 for all i, any critical point is a global minimum (Hardt & Ma, 2017, Theorem 2.2) .Under the same assumptions on data distribution, we can slightly modify Theorem 2.1 to derive a population risk counterpart, and in fact notice that the result proved in BID5 is a corollary of this modification because having ?? max (W i ) < 1 for all i is a sufficient condition for (I + W H+1 ) ?? ?? ?? (I + W 1 ) having full rank.

Moreover, notice that we can remove the assumption det(R) > 0 which was required by BID5 .

We state this special case as a corollary: Corollary 2.3 (Theorem 2.2 of BID5 ).

Under assumptions on data distribution as described above, any critical point of DISPLAYFORM2 We also note in passing that the classical problem of matrix factorization DISPLAYFORM3 is a special case of deep linear neural networks, so our theorems can also be directly applied.

Remarks.

The previous result BID8 assumed d y ??? d x and showed that: 1) every local minimum is a global minimum, and 2) any other critical point is a saddle point.

A subsequent paper by BID11 proved 1) without the assumption d y ??? d x , but as far as we know there is no result showing 2) in the case of d y > d x .

We provide the proof for this case in Lemma B.1.

In fact, we propose an alternative proof technique for handling degenerate critical points, which is much simpler than the technique presented by BID8 .

In this section, we provide proofs for Theorems 2.1 and 2.2.

We first analyze the globally optimal solution of a "relaxation" of L(W ), which turns out to be very useful while proving Theorems 2.1 and 2.2.

Consider the relaxed risk function DISPLAYFORM0 F , where R ??? R dy??dx and rank(R) ??? k. For any W , the product W H+1 W H ?? ?? ?? W 1 has rank at most k and setting R to be this product gives the same loss values: DISPLAYFORM1

This observation is very important in proofs; we will show that inside certain sets, any critical point DISPLAYFORM0 By restating this observation as an optimization problem, the solution of problem in FORMULA1 is bounded below by the minimum value of the following: DISPLAYFORM1 In case where k = min{d x , d y }, (2) is actually an unconstrained optimization problem.

Note that L 0 is a convex function of R, so any critical point is a global minimum.

By differentiating and setting the derivative to zero, we can easily get the unique globally optimal solution DISPLAYFORM2 In case of k < min{d x , d y }, the problem becomes non-convex because of the rank constraint, but its exact solution can still be computed easily.

We present the solution of this case as a proposition and defer the proof to Appendix C due to its technicalities.

Proposition 3.1.

Suppose k < min{d x , d y }.

Then the optimal solution to (2) is DISPLAYFORM3 which is the orthogonal projection of Y X T (XX T ) ???1 onto the column space of?? .

By simple matrix calculus, we can calculate the derivatives of L(W ) with respect to W i , for i = 1, . . .

, H + 1.

We present the result as the following lemma, and defer the details to Appendix C. Lemma 3.2.

The partial derivative of L(W ) with respect to W i is given as DISPLAYFORM0 DISPLAYFORM1 This result will be used throughout the proof of Theorems 2.1 and 2.2.

For clarity in notation, note that when i = 1, W DISPLAYFORM2 is an identity matrix in R dy??dy .We also state an elementary lemma which proves useful in our proofs, whose proof we defer to Appendix C. Lemma 3.3.

1.

For any A ??? R m??n and B ??? R n??l where m ??? n, DISPLAYFORM3 We prove Theorem 2.1, which addresses the case k = min{d x , d y }.

First, recall that the set defined in Theorem 2.1 is DISPLAYFORM4 As seen in (3), the unique minimum point of L 0 has rank k. So, no point W ??? V c 1 can be a global minimum of L. Therefore, by Kawaguchi (2016, Theorem 2.3.(iii) ) and Lemma B.1, any critical point in V c 1 must be a saddle point.

For the rest of our proof, we need to consider two cases: DISPLAYFORM5 cases work.

The outline of the proof is as follows: we define a new set W , show that any critical point in the set W is a global minimum, and then show that every W ??? V 1 is also in W for some > 0.

This proves that any critical point of L(W ) in V 1 is also a critical point in W for some > 0, hence a global minimum.

The following proposition proves the first step: Proposition 3.4.

Assume that k = min{d x , d y }.

For any > 0, define the following set: DISPLAYFORM6

The product is the unique globally optimal solution (3) of the relaxed problem in (2), so W is a global minimum point of L. DISPLAYFORM0 , and the rest of the proof flows in a similar way as the previous case.

The next proposition proves the theorem: Proposition 3.5.

For any point W ??? V 1 , there exists an > 0 such that W ??? W .Proof.

Define a new set W, a "limit" version (as ??? 0) of W , as DISPLAYFORM1 on the cases.

Then, we can set DISPLAYFORM2 We always have > 0 because the matrices are full rank, and we can see that W ??? W .

In this section we prove Theorem 2.2, which tackles the case k < min{d x , d y }.

Note that this assumption also implies that 1 ??? p ??? H.As for the proof of Theorem 2.1, define DISPLAYFORM0 The globally optimal point of the relaxed problem (2) has rank k, as seen in (4).

Thus, any point outside of V 1 cannot be a global minimum.

Then, by Kawaguchi (2016, Theorem 2.3.(iii) ) and Lemma B.1, it follows that any critical point in V c 1 must be a saddle point.

The remaining proof considers points in V 1 .For this section, let us introduce some additional notations to ease presentation.

Define DISPLAYFORM1 Notice that A H+1 and B 1 are identity matrices.

Now consider any tuple W ??? V 1 .

Since the full product W H+1 ?? ?? ?? W 1 has rank k, any partial products A i and B i must have rank( DISPLAYFORM2 However, we have k ??? rank(A 1 ) and k ??? rank(B H+1 ), so the ranks are all identically k. Also, DISPLAYFORM3 but it was just shown that the these spaces have the same dimensions, which equals k, meaning row(A 1 ) = row(A 2 ) = ?? ?? ?? = row(A p ) and col(B H+1 ) = col(B H ) = ?? ?? ?? = col(B p+1 ).Using this observation, we can now state a proposition showing necessary and sufficient conditions for a tuple W ??? V 1 to be a critical point of L(W ).

Proposition 3.6.

A tuple W ??? V 1 is a critical point of L if and only if A p E = 0 and EB p+1 = 0.

Now recall that B 1 and A H+1 are identity matrices, so col(E) ??? row(A p ) ??? and row(E) ??? col(B p+1 ) ??? , which proves A p E = 0 and EB p+1 = 0.Now we present a proposition that specifies the necessary and sufficient condition in which a critical point of L(W ) in V 1 is a global minimum.

Recall that when we take the SVD Y X T (XX T ) ???1 X = U ??V T ,?? ??? R dy??k is defined to be a matrix consisting of the first k columns of U .

Proof.

Since W is a critical point, by Proposition 3.6 we have A p E = 0.

Also note from the definitions of A i 's and DISPLAYFORM0 Because rank(A p ) = k, and A p A T p ??? R k??k is invertible, so B p+1 is determined uniquely as DISPLAYFORM1 Comparing this with (4), W is a global minimum solution if and only if From Proposition 3.7, we can define the set V 2 that appeared in Theorem 2.2, and conclude that every critical point of L(W ) in V 2 is a global minimum, and any other critical points are saddle points.

DISPLAYFORM2

In this section, we present some sufficient conditions for global optimality for deep nonlinear neural networks via a function space view.

Given a smooth nonlinear function h * that maps input to output, Bartlett et al. (2017) described a method to decompose it into a number of smooth nonlinear functions h * = h H+1 ??? ?? ?? ?? ??? h 1 where h i 's are close to identity.

Using Fr??chet derivatives of the population risk with respect to each function h i , they showed that when all h i 's are close to identity, any critical point of the population risk is a global minimum.

One can see that these results are direct generalization of Theorems 2.1 and 2.2 of BID5 to nonlinear networks and utilize the classical "small gain" arguments often used in nonlinear analysis and control BID9 BID17 .

Motivated by this result, we extended Theorem 2.1 to deep nonlinear neural networks and obtained sufficient conditions for global optimality in function space.

Suppose the data X ??? R dx and its corresponding label Y ??? R dy are drawn from some distribution.

Notice that in this section, X and Y are random vectors instead of matrices.

We want to predict Y given X with a deep nonlinear neural network that has H hidden layers.

We express each layer of the network as functions h i : R di???1 ??? R di , so the entire network can be expressed as a composition of functions: h H+1 ??? h H ??? ?? ?? ?? ??? h 1 .

Our goal is to obtain functions h 1 , . . .

, h H+1 that minimize the population risk functional: DISPLAYFORM0 where h is a shorthand notation for (h 1 , . . .

, h H+1 ).

It is well-known that the minimizer of squared error risk is the conditional expectation of Y given X, which we will denote h DISPLAYFORM1 .

With this, we can separate the risk functional into two terms DISPLAYFORM2 where the constant C denotes the variance that is independent of h 1 , . . .

, h H+1 .

Note that if h H+1 ??? ?? ?? ?? ??? h 1 = h * almost surely, the first term in L(h) vanishes and the optimal value L * of L(h) is C.Assumptions.

Define the function spaces as the following: DISPLAYFORM3 DISPLAYFORM4 where F i are defined for all i = 1, . . .

, H + 1.

Assume that h * ??? F, and that we are optimizing L(h) with h 1 ??? F 1 , . . .

, h H+1 ??? F H+1 .

In other words, the functions in F, F 1 , . . .

, F H+1 are differentiable and show sublinear growth starting from 0.

Notice that h H+1 ??? ?? ?? ?? ??? h 1 ??? F, because a composition of differentiable functions is also differentiable, and a composition of sublinear functions is also sublinear.

We also assume that d i ??? min{d x , d y } for all i = 1, . . .

, H + 1, which is identical to the assumption k = min{d x , d y } in Theorem 2.1.

As in the matrix case, h 0:1 and h H+1:H+2 mean identity maps in R dx and R dy , respectively.

Given a function f , let J[f ](x) be the Jacobian matrix of function f evaluated at DISPLAYFORM0 is a linear functional that maps a function (direction) ?? ??? F i to a real number (directional derivative).

Here, we present two theorems which give sufficient conditions for a critical point (D hi [L(h)] = 0 for all i) in the function space to be a global optimum.

The proofs are deferred to Appendix A. Theorem 4.1.

Consider the case d x ??? d y .

If there exists > 0 such that DISPLAYFORM0 , is a global minimum.

DISPLAYFORM1 , is a global minimum.

Note that these theorems give sufficient conditions, whereas Theorems 2.1 and 2.2 provide necessary and sufficient conditions.

So, if the sets we are describing in Theorems 4.1 and 4.2 do not contain any critical point, the claims would be vacuous.

We ensure that there are critical points in the sets, by presenting the following proposition, whose proof is also deferred to Appendix A. Discussion and Future work.

Theorems 4.1 and 4.2 state that in certain sets of (h 1 , . . .

, h H+1 ), any critical point in function space a global minimum.

However, this does not imply that any critical point for a fixed sigmoid or arctan network is a global minimum.

As noted in (Bartlett et al., 2017) , there is a downhill direction in function space at any suboptimal point, but this direction might be orthogonal to the function space represented by a fixed network, and may hence result in local minima in the parameter space of the fixed architecture.

Understanding the connection between the function space and parameter space of commonly used architectures is an open direction for future research, and we believe that these results can be good initial steps from the theoretical point of view.

For example, we can see that one of the sufficient conditions for global optimality is the Jacobian matrix being full rank.

Given that a nonlinear function can locally be linearly approximated using Jacobians, this connection is already interesting.

An extension of the function space viewpoint to cover different architectures or design new architectures (that have "better" properties when viewed via the function space view) should also be possible and worth studying.

In this section, we introduce additional notation that is used in the proofs.

To emphasize that the Fr??chet derivative D hi [L(h)] is a linear functional that outputs a real number, we will write DISPLAYFORM0 This notation also helps avoiding confusion coming from multiple parentheses and square brackets.

There are many different kinds of norms that appear in the proofs.

Given a finite-dimensional real vector v, v 2 denotes its 2 norm.

For a matrix A, its operator norm is defined as DISPLAYFORM1 .

Let h ??? F. Then define a "generalized" induced norm for nonlinear functions with sublinear growth: DISPLAYFORM2 , where the subscript nl is used to emphasize that this norm is for nonlinear functions.

The norm ?? nl is defined in the same way for F i '

s. Now, given a linear functional G that maps a function f ??? F i to a real number G, f , define the operator norm DISPLAYFORM3

By definition of Fr??chet derivatives, we have DISPLAYFORM0 Therefore, DISPLAYFORM1 This equation (6) will be used in the proof of Theorems 4.1 and 4.2.A.3 PROOF OF THEOREM 4.1 DISPLAYFORM2 .

Since A(X) has full row rank by assumption, A(X)A(X) T is invertible.

Then define a particular directio?? DISPLAYFORM3 .

It remains to check if?? ??? F 1 .

It is easily checked that??(0) = 0 because h H+1:1 (0) ??? h * (0) = 0.

Since J[h H+1:2 ] is differentiable by assumption and h 1 ??? F 1 , A(X) is differentiable and A(X) T , (A(X)A(X) T ) ???1 are differentiable functions.

Also, h H+1:1 ??? h * ??? F, so we can conclude that?? is differentiable.

Moreover, if we decompose A(X) with SVD, A(X) = U ??V T , ?? is of the form ?? = [?? 1 0] and DISPLAYFORM4 from which we can see that DISPLAYFORM5 by our assumption.

Note that, for any X ??? R dx , DISPLAYFORM6 Since this holds for any X, we have DISPLAYFORM7 which ensures that?? ??? F 1 .

Finally, DISPLAYFORM8 From this we can see that if we have a critical point of DISPLAYFORM9

A.4 PROOF OF THEOREM 4.2Recall that by assumption we have j ??? {1, . . .

, H + 1} such that DISPLAYFORM0 As done in the previous theorem, for any w ??? R dj???1 , let A(w) = J[h H+1:j+1 ](h j (w)).

Since A(w) has full row rank by assumption, A(w)A(w)T is invertible.

Then defin??? DISPLAYFORM1 We need to check if?? ??? F j .

It is easily checked that??(0) = 0.

Since J[h H+1:j+1 ] is differentiable by assumption and h j ??? F j , A(w) is differentiable, and so are A(w) T and (A(w)A(w) T ) ???1 .

The inverse function of a differentiable and invertible function is also differentiable, so (h H+1:1 ??? h * ) ??? h ???1 j???1:1 is differentiable.

Hence, we can conclude that?? is differentiable.

As seen in the previous section, DISPLAYFORM2 By the assumption that h j???1:1 is invertible and h j???1:1 (u) 2 ??? 1 u 2 , DISPLAYFORM3 for all v ??? R dj???1 .

From this, we can see that h DISPLAYFORM4 From this, we have DISPLAYFORM5 A.5 PROOF OF PROPOSITION 4.3 (Theorem 4.1) By assumption, we have DISPLAYFORM6 where for every x ??? R dx , the first d y components of h 1 (x) are identical to h * (x), and all other components are zero.

For the rest of h i 's, define h i : DISPLAYFORM7 for all w ??? R di???1 .

Since d i ??? d y for all i, we can check that h H+1 ??? ?? ?? ?? ??? h 1 = h * , and h i ??? F i for all i. Moreover, for all z ??? R d1 , J[h H+1:2 ](z) is all 0 except 1's in diagonal entries, so ?? min (J[h H+1:2 ](z)) ??? 1 and h H+1:2 (z) is twice-differentiable.(Theorem 4.2) It is given that we have j ??? {1, . . .

, H + 1} such that DISPLAYFORM8 , where the first d y components are h * (x) and the rest are zero.

All the rest of h i are set as in (7).

Then, it can be easily checked that h i ??? F i for all i and all the conditions of the theorem are satisfied.

Lemma B.1.

Suppose we are given a data matrix X ??? R dx??m and an output matrix Y ??? R dy??m , where d x < d y .

Assume XX T and Y X T have full ranks.

Consider minimizing the empirical squared error risk: DISPLAYFORM0 where DISPLAYFORM1 . .

, H + 1 are weight matrices of the linear neural network, and d 0 = d x and d H+1 = d y for simplicity in notation.

Also let W denote the tuple (W 1 , . . .

, W H+1 ).

Then, any critical point of L(W ) that is not a local minimum is a saddle point.

Proof.

For this lemma, we separate the proof into two cases: W H ?? ?? ?? W 1 = 0 and W H ?? ?? ?? W 1 = 0.

The crux of the proof is to show that any critical point cannot be a local maximum.

Then, any critical point is either a local minimum or a saddle point, so the conclusion of this lemma follows.

In case of W H ?? ?? ?? W 1 = 0, we use some of the results in BID8 and examine the Hessian of L(W ) with respect to vec(W T H+1 ), where vec(A) denotes vectorization of matrix A. Kawaguchi (2016, Lemma 4.

3) that the Hessian matrix DISPLAYFORM2 DISPLAYFORM3 where ??? denotes the Kronecker product of two matrices.

Notice that H(W ) is positive semidefinite.

Since XX T is full rank, whenever W H ?? ?? ?? W 1 = 0 there exists a strictly positive eigenvalue in H(W ), which means that there exists an increasing direction.

So W cannot be a local maximum.

The case where W H ?? ?? ?? W 1 = 0 requires a bit more careful treatment.

Note that this case corresponds to where we have degenerate critical points, which are in many cases much harder to handle.

For any arbitrary > 0, we describe a procedure that perturbs the matrices W 1 , . . .

, W H+1 by perturbations sampled from Frobenius norm balls of radius centered at 0, which we will denote as B i ( ), i = 1, . . .

, H +1.

Let U(B i ( )) be the uniform distribution over the ball B i ( ).

The algorithm goes as the following:1.

For i ??? {1, . . .

, H + 1} DISPLAYFORM4 First, recall that the set of rank-deficient matrices have Lebesgue measure zero, so for any sample DISPLAYFORM5 has full rank with probability 1.

If we proceed the for loop until i = H + 1, we have a full-rank V H+1 ?? ?? ?? V 1 with probability 1, which means that the algorithm must return i * ??? {1, . . .

, H + 1} with probability 1.

Notice that before and after the i * -th iteration, we have DISPLAYFORM6 This means that if we define??? = W H+1 ?? ?? ?? W i * +1 ??? i * V i * ???1 ?? ?? ?? V 1 , then??? = 0.

Also, notice that DISPLAYFORM7

(1) = (V 1 , . . .

, V ) must hold.

This shows that for any > 0, there is a point U in -neighborhood of W with a strictly greater function value L(U ).

This proves that W cannot be a local maximum.

C DEFERRED PROOFS C.1 PROOF OF PROPOSITION 3.1In case of k < min{d x , d y }, we can decompose the loss function in the following way: DISPLAYFORM0 Let us take a close look into the last term in the RHS.

Note that Y X T (XX T ) ???1 X is the orthogonal projection of Y onto row(X), so each row of Y X T (XX T ) ???1 X ??? Y must be in null(X).

Also, DISPLAYFORM1 It is X T right-multiplied with some matrix, so its columns must lie in col(X T ) = row(X).

By the fact that null(X) ??? = row(X), DISPLAYFORM2 holds.

Now, (2) becomes a problem of minimizing RX ??? Y X T (XX T ) ???1 X 2 F subject to the rank constraint rank(R) ??? k. The optimal solution for this is obtained when RX is the k-rank approximation of Y X T (XX T ) ???1 X. Then, k-rank approximation of Y X T (XX T ) ???1 X can be expressed as???? T Y X T (XX T ) ???1 X, where?? is unique due to our assumption that all singular values are distinct.

Therefore, DISPLAYFORM3 is the unique global minimum solution of (2) when k < min{d x , d y }.

DISPLAYFORM4

@highlight

We provide efficiently checkable necessary and sufficient conditions for global optimality in deep linear neural networks, with some initial extensions to nonlinear settings.

@highlight

The paper gives conditions for the global optimality of the loss function of deep linear neural networks

@highlight

The paper gives theoretical results regarding the existence of local minima in the objective function of deep neural networks.

@highlight

Studies some theoretical properties of deep linear networks.