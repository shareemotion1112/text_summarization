The selection of initial parameter values for gradient-based optimization of deep neural networks is one of the most impactful hyperparameter choices in deep learning systems, affecting both convergence times and model performance.

Yet despite significant empirical and theoretical analysis, relatively little has been proved about the concrete effects of different initialization schemes.

In this work, we analyze the effect of initialization in deep linear networks, and provide for the first time a rigorous proof that drawing the initial weights from the orthogonal group speeds up convergence relative to the standard Gaussian initialization with iid weights.

We show that for deep networks, the width needed for efficient convergence to a global minimum with orthogonal initializations is independent of the depth, whereas the width needed for efficient convergence with Gaussian initializations scales linearly in the depth.

Our results demonstrate how the benefits of a good initialization can persist throughout learning, suggesting an explanation for the recent empirical successes found by initializing very deep non-linear networks according to the principle of dynamical isometry.

Through their myriad successful applications across a wide range of disciplines, it is now well established that deep neural networks possess an unprecedented ability to model complex real-world datasets, and in many cases they can do so with minimal overfitting.

Indeed, the list of practical achievements of deep learning has grown at an astonishing rate, and includes models capable of human-level performance in tasks such as image recognition (Krizhevsky et al., 2012) , speech recognition , and machine translation (Wu et al., 2016 ).

Yet to each of these deep learning triumphs corresponds a large engineering effort to produce such a high-performing model.

Part of the practical difficulty in designing good models stems from a proliferation of hyperparameters and a poor understanding of the general guidelines for their selection.

Given a candidate network architecture, some of the most impactful hyperparameters are those governing the choice of the model's initial weights.

Although considerable study has been devoted to the selection of initial weights, relatively little has been proved about how these choices affect important quantities such as rate of convergence of gradient descent.

In this work, we examine the effect of initialization on the rate of convergence of gradient descent in deep linear networks.

We provide for the first time a rigorous proof that drawing the initial weights from the orthogonal group speeds up convergence relative to the standard Gaussian initialization with iid weights.

In particular, we show that for deep networks, the width needed for efficient convergence for orthogonal initializations is independent of the depth, whereas the width needed for efficient convergence of Gaussian networks scales linearly in the depth.

Orthogonal weight initializations have been the subject of a significant amount of prior theoretical and empirical investigation.

For example, in a line of work focusing on dynamical isometry, it was found that orthogonal weights can speed up convergence for deep linear networks (Saxe et al., 2014; Advani & Saxe, 2017) and for deep non-linear networks Xiao et al., 2018; Gilboa et al., 2019; Chen et al., 2018; Pennington et al., 2017; Tarnowski et al., 2019; Ling & Qiu, 2019) when they operate in the linear regime.

In the context of recurrent neural networks, orthogonality can help improve the system's stability.

A main limitation of prior work is that it has focused almost exclusively on model's properties at initialization.

In contrast, our analysis focuses on the benefit of orthogonal initialization on the entire training process, thereby establishing a provable benefit for optimization.

The paper is organized as follows.

After reviewing related work in Section 2 and establishing some preliminaries in Section 3, we present our main positive result on efficient convergence from orthogonal initialization in Section 4.

In Section 5, we show that Gaussian initialization leads to exponentially long convergence time if the width is too small compared with the depth.

In Section 6, we perform experiments to support our theoretical results.

Deep linear networks.

Despite the simplicity of their input-output maps, deep linear networks define high-dimensional non-convex optimization landscapes whose properties closely reflect those of their non-linear counterparts.

For this reason, deep linear networks have been the subject of extensive theoretical analysis.

A line of work (Kawaguchi, 2016; Hardt & Ma, 2016; Lu & Kawaguchi, 2017; Yun et al., 2017; Zhou & Liang, 2018; Laurent & von Brecht, 2018 ) studied the landscape properties of deep linear networks.

Although it was established that all local minima are global under certain assumptions, these properties alone are still not sufficient to guarantee global convergence or to provide a concrete rate of convergence for gradient-based optimization algorithms.

Another line of work directly analyzed the trajectory taken by gradient descent and established conditions that guarantee convergence to global minimum (Bartlett et al., 2018; Arora et al., 2018; Du & Hu, 2019) .

Most relevant to our work is the result of Du & Hu (2019) , which shows that if the width of hidden layers is larger than the depth, gradient descent with Gaussian initialization can efficiently converge to a global minimum.

Our result establishes that for Gaussian initialization, this linear dependence between width and depth is necessary, while for orthogonal initialization, the width can be independent of depth.

Our negative result for Gaussian initialization also significantly generalizes the result of Shamir (2018) , who proved a similar negative result for 1-dimensional linear networks.

Orthogonal weight initializations.

Orthogonal weight initializations have also found significant success in non-linear networks.

In the context of feedforward models, the spectral properties of a network's input-output Jacobian have been empirically linked to convergence speed (Saxe et al., 2014; Pennington et al., 2017; 2018; Xiao et al., 2018) .

It was found that when this spectrum concentrates around 1 at initialization, a property dubbed dynamical isometry, convergence times improved by orders of magnitude.

The conditions for attaining dynamical isometry in the infinitewidth limit were established by Pennington et al. (2017; 2018) and basically require that input-output map to be approximately linear and for the weight matrices to be orthogonal.

Therefore the training time benefits of dynamical isometry are likely rooted in the benefits of orthogonality for deep linear networks, which we establish in this work.

Orthogonal matrices are also frequently used in the context of recurrent neural networks, for which the stability of the state-to-state transition operator is determined by the spectrum of its Jacobian (Haber & Ruthotto, 2017; Laurent & von Brecht, 2016) .

Orthogonal matrices can improve the conditioning, leading to an ability to learn over long time horizons (Le et al., 2015; Henaff et al., 2016; Chen et al., 2018; Gilboa et al., 2019) .

While the benefits of orthogonality can be quite large at initialization, little is known about whether or in what contexts these benefits persist during training, a scenario that has lead to the development of efficient methods of constraining the optimization to the orthogonal group (Wisdom et al., 2016; Vorontsov et al., 2017; Mhammedi et al., 2017) .

Although we do not study the recurrent setting in this work, an extension of our analysis might help determine when orthogonality is beneficial in that setting.

Denote by · the 2 norm of a vector or the spectral norm of a matrix.

Denote by · F the Frobenius norm of a matrix.

For a symmetric matrix A, let λ max (A) and λ min (A) be its maximum and minimum eigenvalues, and let λ i (A) be its i-th largest eigenvalue.

For a matrix B ∈ R m×n , let σ i (B) be its i-th largest singular value (i = 1, 2, . . .

, min{m, n}), and let σ max (B) = σ 1 (B), σ min (B) = σ min{m,n} (B).

Denote by vec (A) be the vectorization of a matrix A in column-first order.

The Kronecker product between two matrices A ∈ R m1×n1 and B ∈ R m2×n2 is defined as

where a i,j is the element in the (i, j)-th entry of A.

We use the standard O(·), Ω(·) and Θ(·) notation to hide universal constant factors.

We also use C to represent a sufficiently large universal constant whose specific value can differ from line to line.

Suppose that there are n training examples

dx×n the input data matrix and by Y = (y 1 , . . .

, y n ) ∈ R dy×n the target matrix.

Consider an L-layer linear neural network with weight matrices W 1 , . . .

, W L , which given an input x ∈ R dx computes

where

and α is a normalization constant which will be specified later according to the initialization scheme.

We study the problem of training the deep linear network by minimizing the 2 loss over training data:

The algorithm we consider to minimize the objective (2) is gradient descent with random initialization, which first randomly samples the initial weight matrices

from a certain distribution, and then updates the weights using gradient descent: for time t = 0, 1, 2, . . .

,

where η > 0 is the learning rate.

For convenience, we denote

The time index t is used on any variable that depends on W 1 , . . .

, W L to represent its value at time t, e.g., W j:

In this section we present our main positive result for orthogonal initialization.

We show that orthogonal initialization enables efficient convergence of gradient descent to a global minimum provided that the hidden width is not too small.

In order to properly define orthogonal weights, we let the widths of all hidden layers be equal:

. .

, W L−1 are m × m square matrices, and

We sample each initial weight matrix W i (0) independently from a uniform distribution over scaled orthogonal matrices satisfying

The same scaling factor was adopted in Du & Hu (2019) , which preserves the expectation of the squared 2 norm of any input.

F .

Then * is the minimum value for the objective (2).

Denote r = rank(X), κ = λmax(X X) λr(X X)

2 Our main theorem in this section is the following:

for some δ ∈ (0, 1) and a sufficiently large universal constant C > 0.

Set the learning rate η ≤ dy 2L X 2 .

Then with probability at least 1 − δ over the random initialization, we have

where (t) is the objective value at iteration t.

Notably, in Theorem 4.1, the width m need not depend on the depth L.

This is in sharp contrast with the result of Du & Hu (2019) for Gaussian initialization, which requires m ≥Ω(Lrκ 3 d y ).

It turns out that a near-linear dependence between m and L is necessary for Gaussian initialization to have efficient convergence, as we will show in Section 5.

Therefore the requirement in Du & Hu (2019) is nearly tight in terms of the dependence on L. These results together rigorously establish the benefit of orthogonal initialization in optimizing very deep linear networks.

If we set the learning rate optimally according to Theorem 4.1 to η = Θ( dy L X 2 ), we obtain that (t) − * decreases by a ratio of 1 − Θ(κ −1 ) after every iteration.

This matches the convergence rate of gradient descent on the (1-layer) linear regression problem min

The proof uses the high-level framework from Du & Hu (2019) , which tracks the evolution of the network's output during optimization.

This evolution is closely related to a time-varying positive semidefinite (PSD) matrix (defined in (7)), and the proof relies on carefully upper and lower bounding the eigenvalues of this matrix throughout training, which in turn implies the desired convergence result.

First, we can make the following simplifying assumption without loss of generality.

See Appendix B in Du & Hu (2019) for justification.

Assumption 4.1. (Without loss of generality) X ∈ R dx×r , rank(X) = r, Y = W * X, and * = 0. (2019)'s framework.

The key idea is to look at the network's output, defined as

We also write U (t) = αW L:1 (t)X as the output at time t. Note that

F .

According to the gradient descent update rule, we write

.

2r is known as the stable rank of X, which is always no more than the rank.

where E(t) contains all the high-order terms (i.e., those with η 2 or higher).

With this definition, the evolution of U (t) can be written as the following equation:

where

Notice that P (t) is always PSD since it is the sum of L PSD matrices.

Therefore, in order to establish convergence, we only need to (i) show that the higher-order term E(t) is small and (ii) prove upper and lower bounds on P (t)'s eigenvalues.

For the second task, it suffices to control the singular values of

3 Under orthogonal initialization, these matrices are perfectly isometric at initialization, and we will show that they stay close to isometry during training, thus enabling efficient convergence.

The following lemma summarizes some properties at initialization.

Lemma 4.2.

At initialization, we have

Furthermore, with probability at least 1 − δ, the loss at initialization satisfies

Proof sketch.

The spectral property (8) follows directly from (4).

To prove (9), we essentially need to upper bound the magnitude of the network's initial output.

This turns out to be equivalent to studying the magnitude of the projection of a vector onto a random lowdimensional subspace, which we can bound using standard concentration inequalities.

The details are given in Appendix A.1.

Now we proceed to prove Theorem 4.1.

We define

F which is the upper bound on (0) from (9).

Conditioned on (9) being satisfied, we will use induction on t to prove the following three properties A(t), B(t) and C(t) for all t = 0, 1, . . .

:

• B(t):

• C(t):

A(0) and B(0) are true according to Lemma 4.2, and C(0) is trivially true.

In order to prove A(t), B(t) and C(t) for all t, we will prove the following claims for all t ≥ 0:

Claim 4.4.

C(t) =⇒ B(t).

The proofs of these claims are given in Appendix A. Notice that we finish the proof of Theorem 4.1 once we prove A(t) for all t ≥ 0.

In this section, we show that gradient descent with Gaussian random initialization necessarily suffers from a running time that scales exponentially with the depth of the network, unless the width becomes nearly linear in the depth.

Since we mostly focus on the dependence of width and running time on depth, we will assume the depth L to be very large.

Recall that we want to minimize the objective

F by gradient descent.

We assume Y = W * X for some W * ∈ R dy×dx , so that the optimal objective value is 0.

For convenience, we assume X F = Θ(1) and Y F = Θ(1).

, and all weights in the network are independent.

We set the scaling factor α such that the initial output of the network does not blow up exponentially (in expectation):

Note that E f (x;

We also assume that the magnitude of initialization at each layer cannot vanish with depth:

Note that the assumptions (10) and (11) are just sanity checks to rule out the obvious pathological cases -they are easily satisfied by all the commonly used initialization schemes in practice.

Now we formally state our main theorem in this section.

Theorem 5.1.

for some universal constant 0 < γ ≤ 1.

Then there exists a universal constant c > 0 such that, if gradient descent is run with learning rate η ≤ e cL γ , then with probability at least 0.9 over the random initialization, for the first e Theorem 5.1 establishes that efficient convergence from Gaussian initialization is impossible for large depth unless the width becomes nearly linear in depth.

This nearly linear dependence is the best we can hope for, since Du & Hu (2019) proved a positive result when the width is larger than linear in depth.

Therefore, a phase transition from untrainable to trainable happens at the point when the width and depth has a nearly linear relation.

Furthermore, Theorem 5.1 generalizes the result of Shamir (2018) , which only treats the special case of

For convenience, we define a scaled version of

We first give a simple upper bound on A j:

Lemma 5.2.

With probability at least 1 − δ, we have A j:

The proof of Lemma 5.2 is given in Appendix B.1.

It simply uses Markov inequality and union bound.

Furthermore, a key property at initialization is that if j − i is large enough, A j:i (0) will become exponentially small.

Lemma 5.3.

With probability at least

Proof.

We first consider a fixed pair (i, j) such that j − i ≥ L 10 .

In order to bound A j:i (0) , we first take an arbitrary unit vector v ∈ R di−1 and bound A j:i (0)v .

We can write A j:

. .

, j).

Recall the expression for the moments of chi-squared random variables:

(∀λ > 0).

Taking λ = 1 2 and using the bound

Choose a sufficiently small constant c > 0.

By Markov inequality we have

.

Therefore we have shown that for any fixed unit vector v ∈ R di−1 , with probability at least 1 − e

−Ω(L γ ) we have

Next, we use this to bound A j:i (0) via an -net argument.

We partition the index set

Now, for any u ∈ R di−1 , we write it as u = q l=1 a l u l where a l is a scalar and u l is a unit vector supported on S l .

By the definition of

The above inequality is valid for any u ∈ R di−1 .

Thus we can take the unit vector u that maximizes A j:i (0)u .

This gives us A j:

Finally, we take a union bound over all possible (i, j).

The failure probaility is at most

The following lemma shows that the properties in Lemmas 5.2 and 5.3 are still to some extent preserved after applying small perturbations on all the weight matrices.

Lemma 5.4.

Suppose that the initial weights satisfy A j:

, where c 1 > 0 is a universal constant.

Then for another set of matrices

, we must have

Proof.

It suffices to show that the difference

Expanding this product, except for the one term corresponding to A j:i (0), every other term has the form A j:

By assumption, each ∆ k has spectral norm e −0.6c1L γ , and each A j :i (0) has spectral norm O(L 3 ), so we have

Therefore we have

The proof of the second part of the lemma is postponed to Appendix B.2.

As a consequence of Lemma 5.4, we can control the objective value and the gradient at any point sufficiently close to the random initialization.

Lemma 5.5.

For a set of weight matrices (12), the objective and the gradient satisfy

The proof of Lemma 5.5 is given in Appendix B.3.

Finally, we can finish the proof of Theorem 5.1 using the above lemmas.

Proof of Theorem 5.1.

From Lemmas 5.2 and 5.3, we know that with probability at least 0.9, we have (i) A j:

Here c 1 > 0 is a universal constant.

From now on we are conditioned on these properties being satisfied.

We suppose that the learning rate η is at most e 0.2c1L

We say that a set of weight matrices W 1 , . . .

, W L are in the "initial neighborhood" if

.

From Lemmas 5.4 and 5.5 we know that in the "initial neighborhood" the objective value is always between 0.4 Y 2 F and 0.6 Y 2 F .

Therefore we have to escape the "initial neighborhood" in order to get the objective value out of this interval.

Now we calculate how many iterations are necessary to escape the "initial neighborhood." According to Lemma 5.5, inside the "initial neighborhood" each W i can move at most η(

in one iteration by definition of the gradient descent algorithm.

In order to leave the "initial neighborhood," some W i must satisfy

In order to move this amount, the number of iterations has to be at least

This finishes the proof.

In this section, we provide empirical evidence to support the results in Sections 4 and 5.

To study how depth and width affect convergence speed of gradient descent under orthogonal and Gaussian initialization schemes, we train a family of linear networks with their widths ranging from 10 to (0) at t = 1258 and t = 10000, for different depth-width configurations and different initialization schemes.

Darker color means smaller loss.

1000 and depths from 1 to 700, on a fixed synthetic dataset (X, Y ).

4 Each network is trained using gradient descent staring from both Gaussian and orthogonal initializations.

In Figure 1 , We lay out the logarithm of the relative training loss (t) (0) , using heap-maps, at steps t = 1258 and t = 10000.

In each heat-map, each point represents the relative training loss of one experiment; the darker the color, the smaller the loss.

Figure 1 clearly demonstrates a sharp transition from untrainable to trainable (i.e., from red to black) when we increase the width of the network:

• for Gaussian initialization, this transition occurs across a contour characterized by a linear relation between width and depth; • for orthogonal initialization, the transition occurs at a width that is approximately independent of the depth.

These observations excellently verify our theory developed in Sections 4 and 5.

To have a closer look into the training dynamics, we also plot "relative loss v.s. training time" for a variety of depth-width configurations.

See Figure 2 .

There again we can clearly see that orthogonal initialization enables fast training at small width (independent of depth), and that the required width for Gaussian initialization depends on depth.

In this work, we studied the effect of the initialization parameter values of deep linear neural networks on the convergence time of gradient descent.

We found that when the initial weights are iid Gaussian, the convergence time grows exponentially in the depth unless the width is at least as large 4 We choose X ∈ R 1024×16 and W * ∈ R 10×1024 , and set Y = W * X. Entries in X and W * are drawn i.i.d.

from N (0, 1).

as the depth.

In contrast, when the initial weight matrices are drawn from the orthogonal group, the width needed to guarantee efficient convergence is in fact independent of the depth.

These results establish for the first time a concrete proof that orthogonal initialization is superior to Gaussian initialization in terms of convergence time.

A.1 PROOF OF LEMMA 4.2

Proof of Lemma 4.2.

We only need to prove (9).

We first upper bound the magnitude of the network's initial output on any given input

/ z 2 has the same distribution as

.

Note that m > C · log(r/δ).

We know that with probability at least 1 − δ r we have

which implies

Finally, taking a union bound, we know that with probability at least 1 − δ, the inequality (13) holds for every x ∈ {x 1 , . . .

, x r }, which implies

A.2 PROOF OF CLAIM 4.3

Proof of Claim 4.3.

.

Thus we can bound the gradient norm as follows for all 0 ≤ s ≤ t and all i ∈ [L]:

where we have used B(s).

Then for all i ∈ [L] we have:

.

A.3 PROOF OF CLAIM 4.4

Proof of Claim 4.4.

Expanding this product, each term except W j:i (0) has the form:

where i ≤ k 1 < · · · < k s ≤ j are locations where terms like ∆ k l are taken out.

Note that every factor in (15) of the form

according to (8).

Thus, we can bound the sum of all terms of the form (15) as

Here the last step uses m > C(LR) 2 which is implied by (5).

Combined with (8), this proves B(t).

Proof of Claim 4.5.

Recall that we have the dynamics (6) for U (t).

In order to establish convergence from (6) we need to prove upper and lower bounds on the eigenvalues of P (t), as well as show that the high-order term E(t) is small.

We will prove these using B(t).

Using the definition (7) and property B(t), we have

In the lower bound above, we make use of the following relation on dimensions: m ≥ d x ≥ r, which enables the inequality λ min (W i−1:1 (t)X) (

Next, we will prove the following bound on the high-order term E(t):

Recall that E(t) is the sum of all high-order terms in the product

Same as (14), we have

It suffices to show that the above bound is at most 1 6 ηλ min (P t ) U (t) − Y F = 1 6 ηλ min (P t ) 2 (t).

Since λ min (P t ) ≥ Using (12), and noting that either L − i − 1 or i − 1 is greater than L 4 , we have

@highlight

We provide for the first time a rigorous proof that orthogonal initialization speeds up convergence relative to Gaussian initialization, for deep linear networks.