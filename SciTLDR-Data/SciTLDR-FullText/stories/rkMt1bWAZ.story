We achieve bias-variance decomposition for Boltzmann machines using an information geometric formulation.

Our decomposition leads to an interesting phenomenon that the variance does not necessarily increase when more parameters are included in Boltzmann machines, while the bias always decreases.

Our result gives a theoretical evidence of the generalization ability of deep learning architectures because it provides the possibility of increasing the representation power with avoiding the variance inflation.

Understanding why the deep learning architectures can generalize well despite their high representation power with a large number of parameters is one of crucial problems in theoretical deep learning analysis, and there are a number of attempts to solve the problem with focusing on several aspects such as sharpness and robustness BID4 BID25 BID11 BID17 BID10 .

However, the complete understanding of this phenomenon is not achieved yet due to the complex structure of deep learning architectures.

To theoretically analyze the generalizability of the architectures, in this paper, we focus on Boltzmann machines BID0 and its generalization including higher-order Boltzmann machines BID20 BID14 , the fundamental probabilistic model of deep learning (see the book by Goodfellow et al. (2016, Chapter 20) for an excellent overview), and we firstly present bias-variance decomposition for Boltzmann machines.

The key to achieve this analysis is to employ an information geometric formulation of a hierarchical probabilistic model, which was firstly explored by BID1 ; BID15 ; BID16 .

In particular, the recent advances of the formulation by BID22 enables us to analytically obtain the Fisher information of parameters in Boltzmann machines, which is essential to give the lower bound of variances in bias-variance decomposition.

We show an interesting phenomenon revealed by our bias-variance decomposition: The variance does not necessarily increase while the bias always monotonically decreases when we include more parameters in Boltzmann machines, which is caused by its hierarchical structure.

Our result indicates the possibility of designing a deep learning architecture that can reduce both of bias and variance, leading to better generalization ability with keeping the representation power.

The remainder of this paper is organized as follows: First we formulate the log-linear model of hierarchical probability distributions using an information geometric formulation in Section 2, which includes the traditional Boltzmann machines (Section 2.2) and arbitrary-order Boltzmann machines (Section 2.3).

Then we present the main result of this paper, bias-variance decomposition for Boltzmann machines, in Section 3 and discuss its property.

We empirically evaluate the tightness of our theoretical lower bound of the variance in Section 4.

Finally, we conclude with summarizing the contribution of this paper in Section 5.

To theoretically analyze learning of Boltzmann machines BID0 , we introduce an information geometric formulation of the log-linear model of hierarchical probability distributions, which can be viewed as a generalization of Boltzmann machines.

First we prepare a log-linear probabilistic model on a partial order structure, which has been introduced by BID22 .

Let (S, ≤) be a partially ordered set, or a poset BID6 , where a partial order ≤ is a relation between elements in S satisfying the following three properties for all x, y, z ∈ S: (1) x ≤ x (reflexivity), (2) x ≤ y, y ≤ x ⇒ x = y (antisymmetry), and (3) x ≤ y, y ≤ z ⇒ x ≤ z (transitivity).

We assume that S is always finite and includes the least element (bottom) ⊥ ∈ S; that is, ⊥ ≤ x for all x ∈ S. We denote S \ {⊥} by S + .We use two functions, the zeta function ζ : S × S → {0, 1} and the Möbius function µ : S × S → Z. The zeta function ζ is defined as ζ(s, x) = 1 if s ≤ x and ζ(s, x) = 0 otherwise, while the Möbius function µ is its convolutional inverse, that is, DISPLAYFORM0 which is inductively defined as DISPLAYFORM1 For any functions f , g, and h with the domain S such that DISPLAYFORM2 f is uniquely recovered using the Möbius function: DISPLAYFORM3 This is the Möbius inversion formula and is fundamental in enumerative combinatorics BID9 .

BID23 introduced a log-linear model on S, which gives a discrete probability distribution with the structured outcome space (S, ≤).

Let P denote a probability distribution that assigns a probability p(x) for each x ∈ S satisfying ∑ x∈S p(x) = 1.

Each probability p(x) for x ∈ S is defined as DISPLAYFORM4 From the Möbius inversion formula, θ is obtained as DISPLAYFORM5 In addition, we introduce η : S → R as DISPLAYFORM6 The second equation is from the Möbius inversion formula.

BID23 showed that the set of distributions S = {P | 0 < p(x) < 1 and ∑ p(x) = 1} always becomes the dually flat Riemannian manifold.

This is why two functions θ and η are dual coordinate systems of S connected with the Legendre transformation, that is, DISPLAYFORM7 with two convex functions DISPLAYFORM8 Moreover, the Riemannian metric g(ξ) (ξ = θ or η) such that DISPLAYFORM9 which corresponds to the gradient of θ or η, is given as DISPLAYFORM10 DISPLAYFORM11 for all x, y ∈ S + .

Furthermore, S is in the exponential family BID22 , where θ coincides with the natural parameter and η with the expectation parameter.

Let us consider two types of submanifolds: DISPLAYFORM12 for all x ∈ dom(β) } specified by two functions α, β with dom(α), dom(β) ⊆ S + , where the former submanifold S α has constraints on θ while the latter S β has those on η.

It is known in information geometry that S α is e-flat and S β is m-flat, respectively (Amari, 2016, Chapter 2.4) .

Suppose that dom(α) ∪ dom(β) = S + and dom(α)

∩ dom(β) = ∅. Then the intersection S α ∩ S β is always the singleton, that is, the distribution Q satisfying Q ∈ S α and Q ∈ S β always uniquely exists, and the following Pythagorean theorem holds: DISPLAYFORM13 for any P ∈ S α and R ∈ S β .

A Boltzmann machine is represented as an undirected graph G = (V, E) with a vertex set V = {1, 2, . . .

, n} and an edge set E ⊆ {{i, j} | i, j ∈ V }.

The energy function Φ:{0, 1} n → R of the Boltzmann machine G is defined as DISPLAYFORM0 . .

, b n ) and w = (w 12 , w 13 , . . . , w n−1n ) are parameter vectors for vertices (bias) and edges (weight), respectively, such that w ij = 0 if {i, j} ̸ ∈ E. The probability p(x; b, w) of the Boltzmann machine G is obtained for each x ∈ {0, 1} n as DISPLAYFORM1 with the partition function Z such that DISPLAYFORM2 to ensure the condition ∑ x∈{0,1} n p(x; b, w) = 1.

It is clear that a Boltzmann machine is a special case of the log-linear model in Equation FORMULA4 with S = 2 V , the power set of V , and ⊥ = ∅. Let each x ∈ S be the set of indices of "1" of x ∈ {0, 1} n and ≤ be the inclusion relation, that is, x ≤ y if and only if x ⊆

y. Suppose that DISPLAYFORM3 where |x| is the cardinality of x, which we call a parameter set.

The Boltzmann distribution in Equations FORMULA15 and FORMULA16 directly corresponds to the log-linear model in Equation FORMULA4 : DISPLAYFORM4 where θ(x) = b x if |x| = 1 and θ(x) = w x if |w| = 2.

This means that the set of Boltzmann distributions S(B) that can be represented by a parameter set B is a submanifold of S given as DISPLAYFORM5 Given an empirical distributionP .

Learning of a Boltzmann machine is to find the best approximation ofP from the Boltzmann distributions S(B), which is formulated as a minimization problem of the KL (Kullback-Leibler) divergence:

DISPLAYFORM6 This is equivalent to maximize the log-likelihood DISPLAYFORM7 which is well known as the learning equation of Boltzmann machines asη(x) and η B (x) coincides with the expectation for the outcome x with respect to the empirical distributionP obtained from data and the model distribution P B represented by a Boltzmann Machine, respectively.

Thus the minimizer DISPLAYFORM8 This distribution P B is known as m-projection ofP onto S(B) BID23 , which is unique and always exists as S has the dually flat structure with respect to (θ, η).

The parameter set B is fixed in Equation (10) in the traditional Boltzmann machines, but our loglinear formulation allows us to include or remove any element in S + = 2 V \ {⊥} as a parameter.

This attempt was partially studied by BID20 ; BID14 that include higher order interactions of variables to increase the representation power of Boltzmann machines.

For S = 2 V with V = {1, 2, . . .

, n} and a parameter set B ⊆ S + , which is an arbitrary subset of S + = S \ {∅}, the Boltzmann distribution given by an arbitrary-order Boltzmann machine is defined by DISPLAYFORM0 and the submanifold of Boltzmann distributions is given by Equation (12).

Hence Equation FORMULA4 gives the MLE (maximum likelihood estimation) of the empirical distributionP .Let B 1 , B 2 , . . .

, B m be a sequence of parameter sets such that DISPLAYFORM1 Since we have a hierarchy of submanifolds DISPLAYFORM2 we obtain the decreasing sequence of KL divergences: DISPLAYFORM3 where each P Bi = argmin P ∈S(Bi) D KL (P, P ), the best approximation ofP using B i .There are two extreme cases as a choice of the parameter set B. On the one hand, if B = ∅, the Boltzmann distribution is always the uniform distribution, that is, p(x) = 1/2 |V | for all x ∈ S. Thus there is no variance but nothing will be learned from data.

On the other hand, if B = S + , the Boltzmann distribution can always exactly represent the empirical distributionP , that is, D KL (P, P B ) = D KL (P,P ) = 0.

Thus there is no bias in each training but the variance across different samples will be large.

To analyze the tradeoff between the bias and the variance, we perform bias-variance decomposition in the next section.

Another strategy to increase the representation power is to use hidden variables BID12 such as restricted Boltzmann machines (RBMs) BID21 BID8 and deep Boltzmann machines (DBMs) BID18 BID19 .

A Boltzmann machine with hidden variables is represented as G = (V ∪H, E), where V and H correspond to visible and hidden variables, respectively, and the resulting domain S = 2 V ∪H (see Appendix for the formulation of Boltzmann machines with hidden variables as the log-linear model).

It is known that the resulting model can be singular BID26 BID24 and its statistical analysis cannot be directly performed.

Studying bias-variance decomposition for such Boltzmann machines with hidden variables is the interesting future work.

Here we present the main result of this paper, bias-variance decomposition for Boltzmann machines.

We focus on the expectation of the squared KL divergence E[D KL (P * ,P B ) 2 ] from the true (unknown) distribution P * to the MLEP B of an empirical distributionP by a Boltzmann machine with a parameter set B, and decompose it using information geometric properties.

In the following, we use the MLE P * B of the true distribution P * , which is the closest distribution in the set of distributions that can be modeled by Boltzmann machines in terms of the KL divergence and is mathematically obtained with replacingP with P * in Equation (13).Theorem 1 (Bias-variance decomposition of the KL divergence).

Given a Boltzmann machine with a parameter set B. Let P * ∈ S be the true (unknown) distribution, P *

* and an empirical distributionP , respectively.

We have Proof.

From the Pythagorean theorem illustrated in Figure 1 , DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 Hence we have DISPLAYFORM3 The second term is DISPLAYFORM4 ) ,where cov(θ B (s),θ B (u) ) denotes the error covariance betweenθ B (s) andθ B (u) and var( ψ(θ B ) ) denotes the variance of ψ(θ B ), and the last equality comes from the condition in Equation FORMULA4 .Here the term of the (co)variance of the normalizing constant (partition function) ψ(θ): DISPLAYFORM5 ) is the irreducible error since ψ(θ) = −θ(⊥) is orthogonal for every parameter θ(s), s ∈ S and the Fisher information vanishes from Equation (4), i.e., DISPLAYFORM6 For the variance term DISPLAYFORM7 we have from the Cramér-Rao bound (Amari, 2016, Theorem 7 DISPLAYFORM8 with the equality holding when N → ∞, where I ∈ R |B|×|B| is the Fisher information matrix with respect to the parameter set B such that DISPLAYFORM9 for all s, u ∈ B, which is given in Equation FORMULA11 , and I −1 is its inverse.

Finally, from Equation FORMULA4 we obtain DISPLAYFORM10 with the equality holding when N → ∞. DISPLAYFORM11 ′ has more parameters than B. Then it is clear that the bias always reduces, that is, DISPLAYFORM12 However, this monotonicity does not always hold for the variance.

We illustrate this non-monotonicity in the following example.

Let S = 2 V with V = {1, 2, 3} and assume that the true distribution P * is given by 0.2144, 0.0411, 0.2037, 0.145, 0.1423, 0.0337, 0.0535, 0.1663) , which was randomly generated from the uniform distribution.

Suppose that we have three types of parameter sets B 1 = {{1}}, B 2 = {{1}, {2}}, and DISPLAYFORM13 DISPLAYFORM14

We empirically evaluate the tightness of our lower bound.

In each experiment, we randomly generated a true distribution P * , followed by repeating 1, 000 times generating a sample (training data) with the size N from P * to empirically estimate the variance var(P * B , B).

We consistently used the Boltzmann machine represented as the fully connected graph G = (V, E) such that V = {1, 2, . . .

, n} and E = {{i, j} | i, j ∈ V }.

Thus the parameter set B is given as DISPLAYFORM0 We report in FIG0 the mean ± SD (standard deviation) of the empirically obtained variance and its theoretical lower bound var(P * B , B) obtained by repeating the above estimation 100 times.

In FIG0 (a) the sample size N is varied from 10 to 10, 000 with fixing the number of variables n = 5 while in FIG0 (b) n is varied from 3 to 7 with fixing N = 100.

These results overall show that our theoretical lower bound is tight enough if N is large and is reasonable across each n.

In this paper, we have firstly achieved bias-variance decomposition of the KL divergence for Boltzmann machines using the information geometric formulation of hierarchical probability distributions.

Our model is a generalization of the traditional Boltzmann machines, which can incorporate arbitrary order interactions of variables.

Our bias-variance decomposition reveals the nonmonotonicity of the variance with respect to growth of parameter sets, which has been also reported elsewhere for non-linear models BID5 .

This result indicates that it is possible to reduce both bias and variance when we include more higher-order parameters in the hierarchical deep learning architectures.

To solve the open problem of the generalizability of the deep learning architectures, our finding can be fundamental for further theoretical development.

Hidden layer 1 Hidden layer 2 1 2 3 4 {1} {2} {3} {4} {1,2} {1,2,3} {1,2,4} {1,3,4} {2,3,4} {1,3} {1,4} {2,3} {2,4} {1,2,3,4} {3,4} Ø {1} {2} {3} {4} {1,2} {1,2,3} {1,2,4} {1,3,4} {2,3,4} {1,3} {1,4} {2,3} {2,4} {1,2,3,4} {3,4} Ø Figure 3 : An example of a deep Boltzmann machine (left) with an input (visible) layer V = {1, 2} with two hidden layers H 1 = {3} and H 2 = {4}, and the corresponding domain set S V ∪H (right).

In the right-hand side, the colored objects {1}, {2}, {3}, {4}, {1, 3}, {2, 3}, and {3, 4} denote the parameter set B, which correspond to nodes and edges of the DBM in the left-hand side.

A Boltzmann machine with hidden variables is represented as G = (V ∪ H, E), where V and H correspond to visible and hidden variables, respectively, and the resulting domain S = 2 V ∪H .

In particular, restricted Boltzmann machines (RBMs) BID21 BID8 are often used in applications, where the edge set is given as BID18 BID19 , which is the beginning of the recent trend of deep learning BID13 BID7 , the hidden variables H are divided into k disjoint subsets (layers) H 1 , H 2 , . . .

, H k and E = { {i, j} | i ∈ H l−1 , j ∈ H l , l ∈ {1, . . .

, k} } , where V = H 0 for simplicity.

DISPLAYFORM0

V and S ′ = 2 V ∪H and S and S ′ be the set of distributions with the domains S and S ′ , respectively.

In both cases of RBMs and DBMs, we have B = { x ∈ S ′ | |x| = 1 or x ∈ E } , (see Figure 3 ) and the set of Boltzmann distributions is obtained as S ′ (B) = { P ∈ S ′ θ(x) = 0 for all x ̸ ∈ B } .

Since the objective of learning Boltzmann machines with hidden variables is MLE (maximum likelihood estimation) with respect to the marginal probabilities of the visible part, the target empirical distributionP ∈ S is extended to the submanifold S ′ (P ) such that S ′ (P ) = { P ∈ S ′ η(x) =η(x) for all x ∈ S } , and the process of learning Boltzmann machines with hidden variables is formulated as double minimization of the KL divergence such that min DISPLAYFORM0 D KL (P, P B ).Since two submanifolds S ′ (B) and S ′ (P ) are e-flat and m-flat, respectively, it is known that the EM-algorithm can obtain a local optimum of Equation FORMULA4 (Amari, 2016, Section 8.1.3) , which was first analyzed by BID3 .

Since this computation is infeasible due to combinatorial explosion of the domain S ′ = 2 V ∪H , a number of approximation methods such as Gibbs sampling have been proposed BID19 DISPLAYFORM1 as B ⊆ B ′ implies S ′ (B) ⊆ S ′ (B ′ ).

This result corresponds to Theorem 1 in BID12 , the representation power of RBMs.

@highlight

We achieve bias-variance decomposition for Boltzmann machines using an information geometric formulation.

@highlight

The goal of this paper is to analyze the effectiveness and generalizability of deep learning by presenting a theoretical analysis of bias-variance decomposition for hierarchical models, specifically Boltzmann Machines  

@highlight

The paper arrives at the main conclusion that it is possible to reduce both the bias and the variance in a hierarchical model.