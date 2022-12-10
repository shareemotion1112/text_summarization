We prove the precise scaling, at finite depth and width, for the mean and variance of the neural tangent kernel (NTK) in a randomly initialized ReLU network.

The standard deviation is exponential in the ratio of network depth to width.

Thus, even in the limit of infinite overparameterization, the NTK is not deterministic if depth and width simultaneously tend to infinity.

Moreover, we prove that for such deep and wide networks, the NTK has a non-trivial evolution during training by showing that the mean of its first SGD update is also exponential in the ratio of network depth to width.

This is sharp contrast to the regime where depth is fixed and network width is very large.

Our results suggest that, unlike relatively shallow and wide networks, deep and wide ReLU networks are capable of learning data-dependent features even in the so-called lazy training regime.

Modern neural networks are typically overparameterized: they have many more parameters than the size of the datasets on which they are trained.

That some setting of parameters in such networks can interpolate the data is therefore not surprising.

But it is a priori unexpected that not only can such interpolating parameter values can be found by stochastic gradient descent (SGD) on the highly non-convex empirical risk but also that the resulting network function generalizes to unseen data.

In an overparameterized neural network N (x) the individual parameters can be difficult to interpret, and one way to understand training is to rewrite the SGD updates ∆θ p = − λ ∂L ∂θ p , p = 1, . . .

, P of trainable parameters θ = {θ p } P p=1 with a loss L and learning rate λ as kernel gradient descent updates for the values N (x) of the function computed by the network:

Here B = {(x 1 , y 1 ), . . . , (x |B| , y |B| )} is the current batch, the inner product is the empirical 2 inner product over B, and K N is the neural tangent kernel (NTK):

Relation (1) is valid to first order in λ.

It translates between two ways of thinking about the difficulty of neural network optimization:

(i) The parameter space view where the loss L, a complicated function of θ ∈ R #parameters , is minimized using gradient descent with respect to a simple (Euclidean) metric; (ii) The function space view where the loss L, which is a simple function of the network mapping x → N (x), is minimized over the manifold M N of all functions representable by the architecture of N using gradient descent with respect to a potentially complicated Riemannian metric K N on M N .

A remarkable observation of Jacot et al. (2018) is that K N simplifies dramatically when the network depth d is fixed and its width n tends to infinity.

In this setting, by the universal approximation theorem (Cybenko, 1989; Hornik et al., 1989) , the manifold M N fills out any (reasonable) ambient linear space of functions.

The results in Jacot et al. (2018) then show that the kernel K N in this limit is frozen throughout training to the infinite width limit of its average E[K N ] at initialization, which depends on the depth and non-linearity of N but not on the dataset.

This mapping between parameter space SGD and kernel gradient descent for a fixed kernel can be viewed as two separate statements.

First, at initialization, the distribution of K N converges in the infinite width limit to the delta function on the infinite width limit of its mean E[K N ].

Second, the infinite width limit of SGD dynamics in function space is kernel gradient descent for this limiting mean kernel for any fixed number of SGD iterations.

As long as the loss L is well-behaved with respect to the network outputs N (x) and E[K N ] is non-degenerate in the subspace of function space given by values on inputs from the dataset, SGD for infinitely wide networks will converge with probability 1 to a minimum of the loss.

Further, kernel method-based theorems show that even in this infinitely overparameterized regime neural networks will have non-vacuous guarantees on generalization (Wei et al., 2018) .

But replacing neural network training by gradient descent for a fixed kernel in function space is also not completely satisfactory for several reasons.

First, it suggests that no feature learning occurs during training for infinitely wide networks in the sense that the kernel E[K N ] (and hence its associated feature map) is data-independent.

In fact, empirically, networks with finite but large width trained with initially large learning rates often outperform NTK predictions at infinite width.

One interpretation is that, at finite width, K N evolves through training, learning data-dependent features not captured by the infinite width limit of its mean at initialization.

In part for such reasons, it is important to study both empirically and theoretically finite width corrections to K N .

Another interpretation is that the specific NTK scaling of weights at initialization (Chizat & Bach, 2018b; a; Mei et al., 2019; 2018; Rotskoff & Vanden-Eijnden, 2018a; b) and the implicit small learning rate limit (Li et al., 2019) obscure important aspects of SGD dynamics.

Second, even in the infinite width limit, although K N is deterministic, it has no simple analytical formula for deep networks, since it is defined via a layer by layer recursion.

In particular, the exact dependence, even in the infinite width limit, of K N on network depth is not well understood.

Moreover, the joint statistical effects of depth and width on K N in finite size networks remain unclear, and the purpose of this article is to shed light on the simultaneous effects of depth and width on K N for finite but large widths n and any depth d. Our results apply to fully connected ReLU networks at initialization for which our main contributions are:

1.

In contrast to the regime in which the depth d is fixed but the width n is large, K N is not approximately deterministic at initialization so long as d/n is bounded away from 0.

Specifically, for a fixed input x the normalized on-diagonal second moment of K N satisfies

Thus, when d/n is bounded away from 0, even when both n, d are large, the standard deviation of K N (x, x) is at least as large as its mean, showing that its distribution at initialization is not close to a delta function.

See Theorem 1.

2.

Moreover, when L is the square loss, the average of the SGD update ∆K N (x, x) to K N (x, x) from a batch of size one containing x satisfies

where n 0 is the input dimension.

Therefore, if d 2 /nn 0 > 0, the NTK will have the potential to evolve in a data-dependent way.

Moreover, if n 0 is comparable to n and d/n > 0 then it is possible that this evolution will have a well-defined expansion in d/n.

See Theorem 2.

In both statements above, means is bounded above and below by universal constants.

We emphasize that our results hold at finite d, n and the implicit constants in both and in the error terms Under review as a conference paper at ICLR 2020

2 ) are independent of d, n. Moreover, our precise results, stated in §2 below, hold for networks with variable layer widths.

We have denoted network width by n only for the sake of exposition.

The appropriate generalization of d/n to networks with varying layer widths is the parameter

which in light of the estimates in (1) and (2) plays the role of an inverse temperature.

A number of articles (Bietti & Mairal, 2019; Dyer & Gur-Ari, 2018; Lee et al., 2019; Yang, 2019) have followed up on the original NTK work Jacot et al. (2018 (2018)) can be promoted to d/n expansions.

Also, the sum-over-path approach to studying correlation functions in randomly initialized ReLU nets was previously taken up for the forward pass by Hanin & Rolnick (2018) and for the backward pass by Hanin (2018) and Hanin & Nica (2018) .

Consider a ReLU network N with input dimension n 0 , hidden layer widths n 1 , . . .

, n d−1 , and output dimension n d = 1.

We will assume that the output layer of N is linear and initialize the biases in N to zero.

Therefore, for any input x ∈ R n0 , the network N computes

given by

where for i = 1, . . . , d − 1

and µ is a fixed probability measure on R that we assume has a density with respect to Lebesgue measure and satisfies:

µ is symmetric around 0,

The three assumptions in (4) hold for virtually all standard network initialization schemes with the exception of orthogonal weight initialization.

But believe our results extend hold also for this case but not do take up this issue.

The on-diagonal NTK is

and we emphasize that although we have initialized the biases to zero, they are not removed them from the list of trainable parameters.

Our first result is the following: Theorem 1 (Mean and Variance of NKT on Diagonal at Init).

We have

Moreover, we have that E[K N (x, x) 2 ] is bounded above and below by universal constants times

In particular, if all the hidden layer widths are equal (i.e. n i = n, for i = 1, . . . , d − 1), we have

where f g means f is bounded above and below by universal constants times g.

This result shows that in the deep and wide double scaling limit

the NTK does not converge to a constant in probability.

This is contrast to the wide and shallow regime where n i → ∞ and d < ∞ is fixed.

Our next result shows that when L is the square loss K N (x, x) is not frozen during training.

To state it, fix an input x ∈ R n0 to N and define ∆K N (x, x) to be the update from one step of SGD with a batch of size 1 containing x (and learning rate λ).

Theorem 2 (Mean of Time Derivative of NTK on Diagonal at Init).

We have that E λ −1 ∆K N (x, x) is bounded above and below by universal constants times

times a multiplicative error of size

1/n i , as in Theorem 1.

In particular, if all the hidden layer widths are equal (i.e. n i = n, for i = 1, . . . , d − 1), we find

Observe that when d is fixed and n i = n → ∞, the pre-factor in front of exp (5β) scales like 1/n.

This is in keeping with the results from Dyer & Gur-Ari (2018) and Jacot et al. (2018) .

Moreover, it shows that if d, n, n 0 grow in any way so that dβ/n 0 = d 2 /nn 0 → 0, the update ∆K N (x, x) to K N (x, x) from the batch {x} at initialization will have mean 0.

It is unclear whether this will be true also for larger batches and when the arguments of K N are not equal.

In contrast, if n i n and β = d/n is bounded away from 0, ∞, and the n 0 is proportional to d, the average update E[∆K N (x, x)] has the same order of magnitude as E[K N (x)].

The remainder of this article is structured as follows.

First, we give an outline of the proofs of Theorems 1 and 2 in §3 and particularly in §3.1, which gives an in-depth but informal explanation of our strategy for computing moments of K N and its time derivative.

Next, in the Appendix Section §A, we introduce some notation about paths and edges in the computation graph of N .

This notation will be used in the proofs of Theorems 1 and 2 presented in the Appendix Section §B- §D. The computations in §B explain how to handle the contribution to K N and ∆K N coming only from the weights of the network.

They are the most technical and we give them in full detail.

Then, the discussion in §C and §D show how to adapt the method developed in §B to treat the contribution of biases and mixed bias-weight terms in K N , K 2 N and ∆K N .

Since the arguments are simpler in these cases, we omit some details and focus only on highlighting the salient differences.

3 OVERVIEW OF PROOF OF THEOREMS 1 AND 2

The proofs of Theorems 1 and 2 are so similar that we will prove them at the same time.

In this section and in §3.1 we present an overview of our argument.

Then, we carry out the details in Appendix Sections §B- §D below.

Fix an input x ∈ R n0 to N .

Recall from (5) that

where we've set

and have suppressed the dependence on x, N .

Similarly, we have

where we have introduced

and have used that the loss on the batch {x} is given by

2 for some target value N * (x).

To prove Theorem 1 we must estimate the following quantities:

To prove Theorem 2, we must control in addition

The most technically involved computations will turn out to be those involving only weights: namely, the terms

.

These terms are controlled by writing each as a sum over certain paths γ that traverse the network from the input to the output layers.

The corresponding results for terms involving the bias will then turn out to be very similar but with paths that start somewhere in the middle of network (corresponding to which bias term was used to differentiate the network output).

The main result about the pure weight contributions to K N is the following Proposition 3 (Pure weight moments for K N , ∆K N ).

We have

Finally,

We prove Proposition 3 in §B below.

The proof already contains all the ideas necessary to treat the remaining moments.

In §C and §D we explain how to modify the proof of Proposition 3 to prove the following two Propositions: Proposition 4 (Pure bias moments for K N , ∆K N ).

We have

Moreover,

Finally, with probability 1, we have ∆ bb = 0.

Proposition 5 (Mixed bias-weight moments for K N , ∆K N ).

We have

Further, E[∆ wb ] is bounded above and below by universal constants times

The statements in Theorems 1 and 2 that hold for general n i now follow directly from Propositions 3-5.

The asymptotics when n i n follow from some routine algebra.

Before turning to the details of the proof of Propositions 3-5 below, we give an intuitive explanation of the key steps in our sum-over-path analysis of the moments of

Since the proofs of all three Propositions follow a similar structure and Proposition 3 is the most complicated, we will focus on explaining how to obtain the first 2 moments of K w .

Since the biases are initialized to zero and K w involves only derivatives with respect to the weights, for the purposes of analyzing K w the biases play no role.

Without the biases, the output of the neural network, N (x) can be express as a weighted sum over paths in the computational graph of the network:

where Γ 1 a is the collection of paths in N starting at neuron a and the weight of a path wt(γ) is defined in (13) in the Appendix and includes both the product of the weights along γ and the condition that every neuron in γ is open at x. The path γ begins at some neuron in the input layer of N and passes through a neuron in every subsequent layer until ending up at the unique neuron in the output layer (see (10)).

Being a product over edge weights in a given path, the derivative of wt(γ) with respect to a weight W e on an edge e of the computational graph of N is:

There is a subtle point here that wt(γ) also involves indicator functions of the events that neurons along γ are open at x. However, with probability 1, the derivative with respect to W e of these indicator functions is identically 0 at x.

The details are in Lemma 11.

Because K w is a sum of derivatives squared (see (6)), ignoring the dependence on the network input x, the kernel K w roughly takes the form

where the sum is over collections (γ 1 , γ 2 ) of two paths in the computation graph of N and edges e in the computational graph of N that lie on both (see Lemma 6 for the precise statement).

When computing the mean, E[K w ], by the mean zero assumption of the weights W e (see (4)), the only contribution is when every edge in the computational graph of N is traversed by an even number of paths.

Since there are exactly two paths, the only contribution is when the two paths are identical, dramatically simplifying the problem.

This gives rise to the simple formula for E[K w ] (see (23)).

The expression

w is more complex.

It involves sums over four paths in the computational graph of N as in the second statement of Lemma 6.

Again recalling that the moments of the weights have mean 0, the only collections of paths that contribute to E[K 2 w ] are those in which every edge in the computational graph of N is covered an even number of times:

However, there are now several ways the four paths can interact to give such a configuration.

It is the combinatorics of these interactions, together with the stipulation that the marked edges e 1 , e 2 belong to particular pairs of paths, which complicates the analysis of E[K 2 w ].

We estimate this expectation in several steps:

1.

Obtain an exact formula for the expectation in (8):

where F (Γ, e 1 , e 2 ) is the product over the layers = 1, . . . , d in N of the "cost" of the interactions of γ 1 , . . .

, γ 4 between layers − 1 and .

The precise formula is in Lemma 7.

2.

Observe the dependence of F (Γ, e 1 , e 2 ) on e 1 , e 2 is only up to a multiplicative constant:

F (Γ, e 1 , e 2 ) F * (Γ).

The precise relation is (24).

This shows that, up to universal constants,

γ1,γ2 togethe at layer 1 γ3,γ4 togethe at layer 2 .

This is captured precisely by the terms I j , II j defined in (27),(28).

3.

Notice that F * (Γ) depends only on the un-ordered multiset of edges E = E Γ ∈ Σ 4 even determined by Γ (see (17) for a precise definition).

We therefore change variables in the sum from the previous step to find

where Jacobian(E, e 1 , e 2 ) counts how many collections of four paths Γ ∈ Γ 4 even that have the same E Γ also have paths γ 1 , γ 2 pass through e 1 and paths γ 3 , γ 4 pass through e 2 .

Lemma 8 gives a precise expression for this Jacobian.

It turns outs, as explained just below Lemma 8, that

Jacobian(E, e 1 , e 2 ) 6 #loops(E) , where a loop in E occurs when the four paths interact.

More precisely, a loop occurs whenever all four paths pass through the same neuron in some layer (see Figures 1 and 2 ).

4.

Change variables from unordered multisets of edges E ∈ Σ 4 even in which every edge is covered an even number of times to pairs of paths V ∈ Γ 2 .

The Jacobian turns out to be 2 −#loops(E) (Lemma 9), giving

5.

Just like F * (V ), the term 3 #loops(V ) is again a product over layers in the computational graph of N of the "cost" of interactions between our four paths.

Aggregating these two terms into a single functional F * (E) and factoring out the 1/n terms in F * (V ) we find that:

where the 1/n terms cause the sum to become an average over collections V of two independent paths in the computational graph of N , with each path sampling neurons uniformly at random in every layer.

The precise result, including the dependence on the input x, is in (42).

6.

Finally, we use Proposition 10 to obtain for this expectation estimates above and below that match up multiplicative constants.

Figure 1: Cartoon of the four paths γ 1 , γ 2 , γ 3 , γ 4 between layers 1 and 2 in the case where there is no interaction.

Paths stay with there original partners γ 1 with γ 2 and γ 3 with γ 4 at all intermediate layers.

Figure 2: Cartoon of the four paths γ 1 , γ 2 , γ 3 , γ 4 between layers 1 and 2 in the case where there is exactly one "loop" interaction between the marked layers.

Paths swap away from their original partners exactly once at some intermediate layer after 1 , and then swap back to their original partners before 2 .

Taken together Theorems 1 and 2 show that in fully connected ReLU nets that are both deep and wide the neural tangent kernel K N is genuinely stochastic and enjoys a non-trivial evolution during training.

This suggests that in the overparameterized limit n, d → ∞ with d/n ∈ (0, ∞), the kernel K N may learn data-dependent features.

Moreover, our results show that the fluctuations of both K N and its time derivative are exponential in the inverse temperature β = d/n.

It would be interesting to obtain an exact description of its statistics at initialization and to describe the law of its trajectory during training.

Assuming this trajectory turns out to be data-dependent, our results suggest that the double descent curve Belkin et al. (2018; 2019); Spigler et al. (2018) that trades off complexity vs. generalization error may display significantly different behaviors depending on the mode of network overparameterization.

However, it is also important to point out that the results in Hanin (2018); Hanin & Nica (2018); Hanin & Rolnick (2018) show that, at least for fully connected ReLU nets, gradient-based training is not numerically stable unless d/n is relatively small (but not necessarily zero).

Thus, we conjecture that there may exist a "weak feature learning" NTK regime in which network depth and width are both large but 0 < d/n 1.

In such a regime, the network will be stable enough to train but flexible enough to learn data-dependent features.

In the language of Chizat & Bach (2018b) one might say this regime displays weak lazy training in which the model can still be described by a stochastic positive definite kernel whose fluctuations can interact with data.

Finally, it is an interesting question to what extent our results hold for non-linearities other than ReLU and for network architectures other than fully connected (e.g. convolutional and residual).

Typical ConvNets, for instance, are significantly wider than they are deep, and we leave it to future work to adapt the techniques from the present article to these more general settings.

In this section, we introduce some notation, adapted in large part from Hanin & Nica (2018) , that will be used in the proofs of Theorems 1 and 2.

For n ∈ N, we will write

[n] := {1, . . .

, n}.

It will also be convenient to denote

k | every entry in a appears an even number of times}.

Given a ReLU network N with input dimension n 0 , hidden layer widths n 1 , . . .

, n d−1 , and output dimension n d = 1, its computational graph is a directed multipartite graph whose vertex set is the disjoint union

Definition 1 (Path in the computational graph of N ).

, a path γ in the computational graph of N from neuron z( 1 , α 1 ) to neuron z( 2 , α 2 ) is a collection of neurons in layers 1 , . . .

, 2 :

Further, we will write

Given a collection of neurons

γj is a path starting at neuron z( j ,αj ) ending at the output neuron z(d,1)

Note that with this notation, we have

Correspondingly, we will write

If each edge e in the computational graph of N is assigned a weight W e , then associated to a path γ is a collection of weights:

)

Definition 2 (Weight of a path in the computational graph of N ).

Fix 0 ≤ ≤ d, and let γ be a path in the computation graph of N starting at layer and ending at the output.

The weight of a this path at a given input x to N is

where

, is the event that all neurons along γ are open for the input x.

Here y ( ) is as in (2).

Next, for an edge e ∈ [n i−1 ] × [n i ] in the computational graph of N we will write

for the layer of e. In the course of proving Theorems 1 and 2, it will be useful to associate to every Γ ∈ Γ k ( n) an unordered multi-set of edges E Γ .

Definition 3 (Unordered multisets of edges and their endpoints).

For n, n , ∈ N set

to be the unordered multiset of edges in the complete directed bi-paritite graph K n,n oriented from [n] to [n ].

For every E ∈ Σ k (n, n ) define its left and right endpoints to be

where L(E), R(E) are unordered multi-sets.

Using this notation, for any collection

of edges between layers − 1 and that are present in Γ. Similarly, we will write

Z,even }.

We will moreover say that for a path γ an edge e = (α, β) ∈ [n i−1 ] × [n i ] in the computational graph of N belongs to γ (written e ∈ γ) if

Finally, for an edge e = (α, β) ∈ [n i−1 ] × [n i ] in the computational graph of N , we set

for the normalized and unnormalized weights on the edge corresponding to e (see (3)).

We begin with the well-known formula for the output of a ReLU net N with biases set to 0 and a linear final layer with one neuron:

The weight of a path wt(γ) was defined in (13) and includes both the product of the weights along γ and the condition that every neuron in γ is open at x. As explained in §A, the inner sum in (19) is over paths γ in the computational graph of N that start at neuron a in the input layer and end at the output neuron and the random variables W

γ are the normalized weights on the edge of γ between layer i − 1 and layer i (see (12)).

Differentiating this formula gives sum-over-path expressions for the derivatives of N with respect to both x and its trainable parameters.

For the NTK and its first SGD update, the result is the following:

Lemma 6 (weight contribution to K N and ∆K N as a sum-over-paths).

With probability 1,

where the sum is over collections Γ of two paths in the computation graph of N and edges e that lie on both paths.

Similarly, almost surely,

, and

plus a term that has mean 0.

a , e ∈ γ, etc is defined in §A. We prove Lemma 6 in §B.1 below.

Let us emphasize that the expressions for K 2 w and ∆ ww are almost identical.

The main difference is that in the expression for ∆ ww , the second path γ 2 must contain both e 1 and e 2 while γ 4 has no restrictions.

Hence, while for K 2 w the contribution from a collection of four paths Γ = (γ 1 , γ 2 , γ 3 , γ 4 ) is the same as from the collection Γ = (γ 2 , γ 1 , γ 4 , γ 3 ) , for ∆ ww the contributions are different.

This seemingly small discrepancy, as we shall see, causes the normalized expectation E[∆ ww ]/E[K w ] to converge to zero when d < ∞ is fixed and n i → ∞ (see the 1/n factors in the statement of Theorem 2).

In contrast, in the same regime, the normalized second moment E[K Lemma 7 (Expectation of K w , K 2 w , ∆ ww as sums over 2, 4 paths).

We have,

where

Similarly,

where

Finally,

Lemma 7 is proved in §B.2.

The expression (20) is simple to evaluate due to the delta function in H(Γ, e).

We obtain:

where in the second-to-last equality we used that the number of paths in the comutational graph of N from a given neuron in the input to the output neuron equals i=1,...,d n i and in the last equality we used that n d = 1.

This proves the first equality in Theorem 1.

It therefore remains to evaluate (21) and (22).

Since they are so similar, we will continue to discuss them in parallel.

To start, notice that the expression F (Γ, e 1 , e 2 ) appearing in (21) and (22) satisfies

where

For the remainder of the proof we will write

Thus, in particular,

The advantage of F * (Γ) is that it does not depend on e 1 , e 2 .

Observe that for every a = (α 1 , α 2 , α 3 , α 4 ) ∈ [n 0 ] 4 even , we have that either α 1 = α 2 , α 1 = α 3 , or α 1 = α 4 .

Thus, by symmetry, the sum over Γ 4 even ( n) in (21) and (22) we find

and similarly,

where

F * (Γ)# {edges e 1 , e 2 | e 1 ∈ γ 1 ∩ γ 2 , e 2 ∈ γ 3 ∩ γ 4 } (27)

F * (Γ)# {edges e 1 , e 2 | e 1 ∈ γ 1 ∩ γ 2 , e 2 ∈ γ 2 , γ 3 , e 1 = e 2 } .

To evaluate I j , II j let us write

for the indicator function of the event that paths γ α , γ β pass through the same edge between layers i − 1, i in the computational graph of N .

Observe that

and # {edges e 1 , e 2 | e 1 ∈ γ 1 ∩ γ 2 , e 2 ∈ γ 2 , γ 3 , e 1 = e 2 } =

i2 .

Thus, we have

where

i2 .

To simplify I j,i1,i2 and II j,i1,i2 observe that F * (Γ) depends only on Γ only via the unordered edge multi-set (i.e. only which edges are covered matters; not their labelling)

defined in Definition 3.

Hence, we find that for j = 1, 2, 3, 4, i 1 , i 2 = 1, . . .

, d,

The counts in I j, * ,i1,i2 and II j, * ,i1,i2 have a convenient representation in terms of

(32)

Informally, the event C(E, i 1 , i 2 ) indicates the presence of a "collision" of the four paths in Γ before the earlier of the layers i 1 , i 2 , while C(E, i 1 , i 2 ) gives a "collision" between layers i 1 , i 2 ; see Section 3.1 for the intuition behind calling these collisions.

We also write

Finally, for E ∈ Σ 4 a,even ( n), we will define

That is, a loop is created at layer i if the four edges in E all begin at occupy the same vertex in layer i − 1 but occupy two different vertices in layer i.

We have the following Lemma.

Lemma 8 (Evaluation of Counting Terms in (30) and (31)).

Suppose E ∈ Σ 4 aj ,even for some j = 1, 2, 3, 4.

For each i 1 , i 2 ∈ {1, . . .

, d},

Similarly,

We prove Lemma 8 in §B.3 below.

Assuming it for now, observe that

and that the conditions L(E(1)) = a j are the same for j = 2, 3, 4 since the equality it is in the sense of unordered multi-sets.

Thus, we find that E[K 2 w ] is bounded above/below by a constant times

Similarly, E[∆ ww ] is bounded above/below by a constant times

Observe that every unordered multi-set four edge multiset E ∈ Σ 4 even can be obtained by starting from some V ∈ Γ 2 , considering its unordered edge multi-set E V and doubling all its edges.

This map from Γ 2 to Σ 4 even is surjective but not injective.

The sizes of the fibers is computed by the following Lemma.

, where as in (35),

Lemma 9 is proved in §B.4.

Using it and that 0 ≤ C(E, i 1 , i 2 ) ≤ 1, the relation (38) shows that E[K 2 w ] is bounded above/below by a constant times

Similarly, E[∆ ww ] is bounded above/below by a constant times

where, in analogy to (32), we have

.

Since the number of V in Γ 2 ( n) with specified V (0) equals

, we find that so that for each

and similarly,

Here, E x is the expectation with respect to the probability measure on V = (v 1 , v 2 ) ∈ Γ 2 obtained by taking v 1 , v 2 independent, each drawn from the products of the measure

We are now in a position to complete the proof of Theorems 1 and 2.

To do this, we will evaluate the expectations E x above to leading order in i 1/n i with the help of the following elementary result which is proven as Lemma 18 in Hanin & Nica (2018).

Proposition 10.

Let A 0 , A 1 , . . .

, A d be independent events with probabilities p 0 , . . .

, p d and B 0 , . . .

, B d be independent events with probabilities q 0 , . . .

, q d such that

Denote by X i the indicator that the event A i happens, X i := 1 {Ai} , and by Y i the indicator that B i happens,

Then, if γ i ≥ 1 for every i, we have:

where by convention α 0 = γ 0 = 1.

In contrast, if γ i ≤ 1 for every i, we have:

We first apply Proposition 10 to the estimates above for

we find that

Since the contribution for each layer in the product is bounded above and below by constants, we have that

2 is bounded below by a constant times

and above by a constant times

Here, note that the initial condition given by x and the terminal condition that all paths end at one neuron in the final layer are irrelevant.

The expression (45) is there precisely

3 ≤ 1, and K i = 1.

Thus, since for i = 1, . . . , d − 1, the probability of X i is 1/n i + O(1/n 2 i ), we find that

where in the last inequality we used that 1 + x ≥ e

When combined with (23) this gives the lower bound in Proposition 3.

The matching upper bound is obtained from (46) in the same way using the opposite inequality from Proposition 10.

To complete the proof of Proposition 3, we prove the analogous bounds for E[∆ ww ] in a similar fashion.

Namely, we fix 1 ≤ i 1 < i 2 ≤ d and write

The set A is the event that the first collision between layers i 1 , i 2 occurs at layer .

We then have

On the event A , notice that F * (V ) only depends on the layers 1 ≤ i ≤ i 1 and layers < i ≤ d because the event A fixes what happens in layers i 1 < i ≤ .

Mimicking the estimates (45), (46) and the application of Proposition 10 and using independence, we get that:

Finally, we compute:

Under review as a conference paper at ICLR 2020

Combining this we obtain that E[∆ ww ]/ x 4 2 is bounded above and below by constants times

This completes the proof of Proposition 3, modulo the proofs of Lemmas 6-9, which we supply below.

Fix an input x ∈ R n0 to N .

We will continue to write as in (2) y (i) for the vector of pre-activations as layer i corresponding to x. We need the following simple Lemma.

Lemma 11.

With probability 1, either there exists i so that

we have y For each fixed Γ this event defines a co-dimension 1 set in the space of all the weights.

Hence, since the joint distribution of the weights has a density with respect to Lebesgue measure (see just before (4)), the union of this (finite number) of events has measure 0.

This shows that on the even that y ( ) = 0 for every , y (i) j = 0 with probability 1.

Taking the union over i, j completes the proof.

Lemma 11 shows that for our fixed x, with probability 1, the derivative of each ξ (i) j in (19) vanishes.

Hence, almost surely, for any edge e in the computational graph of N :

This proves the formulas for K N , K 2 N .

To derive the result for ∆K N , we write

where the loss L on a single batch containing only x is 1 2 (N (x) − N * (x)) 2 .

We therefore find

Using (47) and again applying Lemma 11, we find that with probability 1

Under review as a conference paper at ICLR 2020

Thus, almost surely

To complete the proof of Lemma 6 it therefore remains to check that this last term has mean 0.

To do this, recall that the output layer of N is assumed to be linear and that the distribution of each weight is symmetric around 0 (and hence has vanishing odd moments).

Thus, the expectation over the weights in layer d has either 1 or 3 weights in it and so vanishes.

Lemma 7 is almost a corollary of of Theorem 3 in Hanin (2018) and Proposition 2 in Hanin & Nica (2018) .

The difference is that, in Hanin (2018); Hanin & Nica (2018) , the biases in N were assumed to have a non-degenerate distribution, whereas here we've set them to zero.

The nondegeneracy assumption is not really necessary, so we repeat here the proof from Hanin (2018)

To compute the inner expectation, write F j for the sigma algebra generated by the weight in layers up to and including j. Let us also define the events:

where we recall from (2) that x (j) are the post-activations in layer j. Supposing first that e is not in layer d, the expectation becomes

We have

Thus, the expectation in (48) becomes

Note that given F d−2 , the pre-activations y

.

Recall that by assumption, the weight matrix

.

This replacement leaves the product

.

On the event S d−1 (which occurs whenever y

= 0 with probability 1 since we assumed that the distribution of each weight has a density relative to Lebesgue measure.

Hence, symmetrizing over ± W (d) , we find that

.

Similarly, if e is in layer i, then we automatically find that γ 1 (i − 1) = γ 2 (i − 1) and γ 1 (i) = γ 2 (i), giving an expectation of 1/n i−1 1

.

Proceeding in this way yields

which is precisely (20).

The proofs of (21) and (22) are similar.

We have

As before let us first assume that edges e 1 , e 2 are not in layer d. Then,

The the inner expectation is

1 each weight appears an even number of times Again symmetrizing with respect to ± W (d) and using that the pre-activation of different neurons are independent given the activations in the previous layer we find that, on the event {y

where L is the event that |Γ(d − 1)| = |Γ(d)| = 1 and e 1 , e 2 are not in layer d − 1.

Proceeding in this way one layer at a time completes the proofs of (21) and (22).

Fix j = 1, . . . , 4, edges e 1 , e 2 with (e 1 ) ≤ (e 2 ) in the computational graph of N and E ∈ Σ 4 aj ,even .

The key idea is to decompose E into loops.

To do this, define

For each i = 1, . . .

, d there exists unique k = 1, . . .

, #loops(E)

so that

We will say that two layers i, j = 1, . . .

, d belong to the same loop of E if exists k = 1, . . .

, #loops(E)

so that

We proceed layer by layer to count the number of Γ ∈ Γ 4 aj ,even satisfying Γ(0) = a j and E Γ = E.

To do this, suppose we are given Γ(i − 1) ∈ [n i−1 ] 4 and we have L(E(i)) = 2.

Then Γ(i − 1) is some permutation of (α 1 , α 1 , α 2 , α 2 ) with α 1 = α 2 .

Moreover, for j = 1, 2 there is a unique edge (with multiplicity 2) in E(i) whose left endpoint is α j .

Therefore, Γ(i − 1) determines Γ(i) when L(E(i)) = 2.

In contrast, suppose L(E(i)) = 1.

If R(E(i)) = 1, then E(i) consists of a single edge with multiplicity 4, which again determines Γ(i − 1), Γ(i).

In short, Γ(i) determines Γ(j) for all j belonging to the same loop of E as i. Therefore, the initial condition Γ(0) = a j determines Γ(i) for all i ≤ i 1 and the conditions e 1 ∈ γ 1 , e 2 ∈ γ 2 determine Γ in the loops of E containing the layers of e 1 , e 2 .

Finally, suppose L(E(i)) = 1 and R(E(i)) = 2 (i.e. i = i k (E) for some k = 1, . . .

, d) and that e 1 , e 2 are not contained in the same loop of E layer i.

Then all 4 2 = 6 choices of Γ(i) satisfy Γ(i) = R(E(i)), accounting for the factor of 6 #loops(E) .

The concludes the proof in the case j = 1.

the only difference in the cases j = 2, 3, 4 is that if γ 1 (0) = γ 2 (0) (and hence γ 3 (0) = γ 4 (0)), then since (e 1 ) ≤ (e 2 ) in order to satisfy e 1 ∈ γ 1 , γ 2 we must have that i 1 (E) < (e 1 ).

The proof of Lemma 9 is essentially identical to the proof of Lemma 8.

In fact it is slightly simpler since there are no distinguished edges e 1 , e 2 to consider.

We omit the details.

In this section, we seek to estimate

The approach is essentially identical to but somewhat simpler than our proof of Proposition 3 in §B. We will therefore focus here on explaining the salient differences.

Our starting point is the following analog of Lemma 6, which gives a sum-over-paths expression for the bias contribution K b to the neural tangent kernel.

To state it, let us define, for any collection Z = (z 1 , . . .

, z k ) ∈ Z k of k neurons in N 1 {y Z >0} := k j=1 1 {yz j >0} , to be the event that the pre-activations of the neurons z k are positive.

Lemma 12 (K b as a sum over paths).

With probability 1,

where Z 1 , Γ 2 (Z,Z) , wt(γ) are defined in §A. Further, almost surely,

The proof of this result is a small modification of the proof of Lemma 6 and hence is omitted.

Taking expectations, we therefore obtain the following analog to Lemma 7.

Moreover, E[K

The proof is identical to the argument used in §B.2 to establish Lemma 7, so we omit the details.

The relation (51) is easy to simplify:

where we used that the number paths from a neuron in layer to the output of N equals

Proof.

The proof of Lemma 14 is a simplified version of the computation of E[K 2 w ] (starting around (24) and ending at the end of the proof of Proposition 3).

Specifically, note that for Γ = (γ 1 , . . .

, γ 4 ) ∈ Γ 4 Z,even with (z 1 ) ≤ (z 2 ), the delta functions 1 {γ 1(i )=γ2(i )} 1 {γ 1(i −1)=γ2(i −1)} in the definition (52) of H(Γ) ensures that γ 1 , γ 2 go through the same neuron in layer (z 2 ).

To condition on the index of this neuron, we recall that we denote by z(j, β) neuron number β in layer j.

We have

where Z = (z( (z 2 ), β), z( (z 2 ), β), z 2 , z 2 ) and .

Since the inner sum in (54) is independent of β by symmetry, we find

where Z = (1, 1, z 2 , z 2 ).

The inner sum in (55) is now precisely one of the terms I j from (27) without counting terms involving edges e 1 , e 2 , except that the paths start at neuron 1 in layer (z 2 ).

The changes of variables from Γ ∈ Γ 4 even to E ∈ Σ 4 even to V ∈ Γ 2 that we used to estimate the I j 's are no far simpler.

In particular, Lemma 8 still holds but without any of the A(E, i 1 , i 2 ), C(E, i 1 , i 2 ), C(E, i 1 , i 2 ) terms.

Thus, we find that where for the second estimate we applied Lemma 9 and have written Z = (1, z 2 ).

Thus, as in the derivation of (42), we find that

@highlight

The neural tangent kernel in a randomly initialized ReLU net is non-trivial fluctuations as long as the depth and width are comparable. 