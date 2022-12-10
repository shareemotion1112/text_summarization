We address the following question: How redundant is the parameterisation of ReLU networks?

Specifically, we consider transformations of the weight space which leave the function implemented by the network intact.

Two such transformations are known for feed-forward architectures: permutation of neurons within a layer, and positive scaling of all incoming weights of a neuron coupled with inverse scaling of its outgoing weights.

In this work, we show for architectures with non-increasing widths that permutation and scaling are in fact the only function-preserving weight transformations.

For any eligible architecture we give an explicit construction of a neural network such that any other network that implements the same function can be obtained from the original one by the application of permutations and rescaling.

The proof relies on a geometric understanding of boundaries between linear regions of ReLU networks, and we hope the developed mathematical tools are of independent interest.

Ever since its early successes, deep learning has been a puzzle for machine learning theorists.

Multiple aspects of deep learning seem at first sight to contradict common sense: single-hidden-layer networks suffice to approximate any continuous function (Cybenko, 1989; Hornik et al., 1989 ), yet in practice deeper is better; the loss surface is highly non-convex, yet it can be minimised by first-order methods; the capacity of the model class is immense, yet deep networks tend not to overfit (Zhang et al., 2017) .

Recent investigations into these and other questions have emphasised the role of overparameterisation, or highly redundant function representation.

It is now known that overparameterised networks enjoy both easier training (Allen-Zhu et al., 2019; Du et al., 2019; Frankle & Carbin, 2019) , and better generalisation (Belkin et al., 2019; Neyshabur et al., 2019; Novak et al., 2018) .

However, the specific mechanism by which over-parameterisation operates is still largely a mystery.

In this work, we study one particular aspect of over-parameterisation, namely the ability of neural networks to represent a target function in many different ways.

In other words, we ask whether many different parameter configurations can give rise to the same function.

Such a notion of parameterisation redundancy has so far remained unexplored, despite its potential connections to the structure of the loss landscape, as well as to the literature on neural network capacity in general.

Specifically, we consider feed-forward ReLU networks, with weight matrices W 1 , . . .

, W L , and biases b 1 , . . .

, b L ,.

We study parameter transformations which preserve the output behaviour of the network h(z) = W L σ(W L−1 σ(. . .

W 1 z + b 1 . . . ) + b L−1 ) + b L for all inputs z in some domain Z. Two such transformations are known for feed-forward ReLU architectures:

1. Permutation of units (neurons) within a layer, i.e. for some permutation matrix P,

2.

Positive scaling of all incoming weights of a unit coupled with inverse scaling of its outgoing weights.

Applied to a whole layer, with potentially different scaling factors arranged into a diagonal matrix M, this can be written as

Our main theorem applies to architectures with non-increasing widths, and shows that there are no other function-preserving parameter transformations besides permutation and scaling.

Stated formally:

Theorem 1.

Consider a bounded open nonempty domain Z ⊆ R d0 and any architecture

For this architecture, there exists a ReLU network h θ : Z → R, or equivalently a setting of the weights θ (W 1 , b 1 , . . .

, W L , b L ), such that for any 'general' ReLU network h η : Z → R (with the same architecture) satisfying h θ (z) = h η (z) for all z ∈ Z, there exist permutation matrices P 1 , . . .

P L−1 , and positive diagonal matrices M 1 , . . .

, M L−1 , such that

where η (W 1 , b 1 , . . . , W L , b L ) are the parameters of h η .

In the above, 'general' networks is a class of networks meant to exclude degenerate cases.

We give a more precise definition in Section 3; for now it suffices to note that almost all networks are general.

The proof of the result relies on a geometric understanding of prediction surfaces of ReLU networks.

These surfaces are piece-wise linear functions, with non-differentiabilities or 'folds' between linear regions.

It turns out that folds carry a lot of information about the parameters of a network, so much in fact, that some networks are uniquely identified (up to permutation and scaling) by the function they implement.

This is the main insight of the theorem.

In the following sections, we introduce in more detail the concept of a fold-set, and describe its geometric structure for a subclass of ReLU networks.

The paper culminates in a proof sketch of the main result.

The full proof, including proofs of intermediate results, is included in the Appendix.

The functional equivalence of neural networks is a well-researched topic in classical connectionist literature.

The problem was first posed by Hecht-Nielsen (1990) , and soon resolved for feed-forward networks with the tanh activation function by Chen et al. (1993) , who showed that any smooth transformation of the weight space that preserves the function of all neural networks is necessarily a composition of permutations and sign flips.

For the same class of networks, Fefferman & Markel (1994) showed a somewhat stronger result: knowledge of the input-output mapping of a neural network determines both its architecture and its weights, up to permutations and sign flips.

Similar results have been proven for single-layer networks with a saturating activation function such as sigmoid or RBF (Kůrková & Kainen, 1994) , as well as single-layer recurrent networks with a smooth activation function (Albertini & Sontag, 1993a; b) .

To the best of our knowledge, no such theoretical results exist for networks with the ReLU activation, which is non-saturating, asymmetric and non-smooth.

Broadly related is the recent work by Petersen et al. (2018) and Berner et al. (2019) who study whether two neural networks (ReLU or otherwise) that are close in the functional space have parameterisations that are close in the weight space.

This is called inverse stability.

In contrast, we are interested in ReLU networks that are functionally identical, and ask about all their possible parameterisations.

In terms of proof technique, our approach is based on the geometry of piece-wise linear functions, specifically the boundaries between linear regions.

The intuition for this kind of analysis has previously been presented by Raghu et al. (2017) and Serra et al. (2018) , and somewhat similar proof techniques to ours have been used by Hanin & Rolnick (2019) in the context of counting the number of linear decision regions.

Finally, the sets of equivalent parametrisations can be viewed as symmetries in the weight space, with implications for optimisation.

Multiple authors, including e.g. Neyshabur et al. (2015) ; Badrinarayanan et al. (2016); Stock et al. (2019) , have observed that the naive loss gradient is sensitive to reparametrisation by scaling, and proposed alternative, scaling-invariant optimisation procedures.

We will omit the subscript θ when it is clear from the context.

In this work, we restrict our attention to so-called general ReLU networks.

Intuitively, a general network is one that satisfies a number of non-degeneracy properties, such as all weight matrices having non-zero entries and full rank, no two network units exactly cancelling each other out, etc.

It can be shown 1 that almost all ReLU networks are general.

In other words, a sufficient condition for a ReLU network to be general with probability one is that its weights are sampled from a distribution with a density.

More formally, a general ReLU network is one that satisfies the following three conditions.

1.

For any unit (l, i), the local optima of h 1:l i do not have value exactly zero.

2.

For all k ≤ l and all diagonal matrices (I k , . . .

, I l ) with entries in {0, 1}, General networks are convenient to study, as they exclude many degenerate special cases.

The second important class of ReLU networks are so-called transparent networks.

Their significance as well as their name will become clear in the next section.

For now, we state the definition.

In words, we require that for any input, at least one unit on each layer is active.

In this section we introduce the concept of fold-sets, which is key to our understanding of ReLU networks and their prediction surfaces.

Since ReLU networks are piece-wise linear functions, a great deal about them is revealed by the boundaries between individual linear regions.

A network's fold-set is simply the union of all these boundaries.

More formally, if Z is an open set, and f : Z → R is any continuous, piece-wise linear function, we define the fold-set of f , denoted by F(f ), as the set of all points at which f is non-differentiable.

It turns out there is a class of networks whose fold-sets are especially easy to understand; these are the ones we have termed transparent.

For transparent networks, we have the following characterisation of the fold-set (which also motivates the name 'transparent').

To appreciate the significance of the lemma, suppose we are given some transparent ReLU network function h and we want to infer its parameters.

This lemma shows that the knowledge of the endto-end mapping h h 1:L in fact gives us information about the network's hidden units h 1:l i (hence 'transparent').

Moreover, this information is very explicit: we observe the units' zero-level sets, which in the case of a linear unit on a full-dimensional space already determines the unit's parameters up to scaling 2 .

Of course, dealing with piece-wise linearity and disambiguating the union into its constituent zero-level sets remains a challenge for upcoming sections.

In this section, we provide a geometric description of fold-sets of transparent networks.

Intuitively, the fold-sets look like the sets shown in Figure 1 .

The first-layer units of a network are linear, so the component i z | h 1:1 i (z) = 0 of the fold-set (9) is a union of hyperplanes, illustrated by the blue lines in Figure 1 .

These hyperplanes partition the input space into a number of regions that each correspond to a different activation pattern.

For a fixed activation pattern, or equivalently on each region, the second-layer units are linear, so their zero-level sets i z | h 1:2 i (z) = 0 are composed of piece-wise hyperplanes on the partition induced by the first-layer units.

This is shown by the orange lines in Figure 1 .

More generally, the l th -layer zero-level sets i z | h 1:l i (z) = 0 consist of piece-wise hyperplanes on the partition induced by all lower-layer units.

This yields a fold-set that looks like the set in the right pane of Figure 1 , but potentially much more complicated.

We now define these concepts more precisely.

Piece-wise hyperplane.

Let P be a partition of Z. We say H ⊆ Z is a piece-wise hyperplane with respect to partition P, if H is nonempty and there exist (w, b) = (0, 0) and P ∈ P such that H = {z ∈ P | w z + b = 0}. The final ingredient we will need to be able to reason about the parameterisation of ReLU networks is a more precise characterisation of the fold-set, in particular, the dependence structure between individual piece-wise hyperplanes.

For example, consider the piece-wise linear surface in Figure 1 and compare it to the one in Figure 2 .

Suppose as before that the blue hyperplanes come from first-layer units, the orange hyperplanes come from second-layer units, and the black hyperplanes come from third-layer units.

The difference between Figure 1 and Figure 2 is that if we observe only the fold-set, i.e. only the union of the zero-level sets over all layers (as shown in the right pane of Figure 2 ), then in the case of Figure 2 , it is impossible to know which folds come from which layers.

For instance, the blue folds and the orange folds could be assigned to the first and second layer almost arbitrarily; there is not enough information (i.e. intersection) in the fold-set to tell which is which.

In contrast, the piece-wise linear surface in the right pane of Figure 1 could in principle be disambiguated into first-, second-and third-layer folds by the following procedure: 1.

Take the largest possible union of hyperplanes that is a subset of the fold-set, and assign the hyperplanes to layer one.

2.

Take all piece-wise hyperplanes with respect to the partition induced by the first-layer folds, and assign them to layer two.

3.

Take all piece-wise hyperplanes with respect to the partition induced by the first-and second-layer folds, and assign them to layer three.

This procedure is not guaranteed to assign all folds to their original layers because it ignores how piece-wise hyperplanes are connected; for example for the piece-wise linear surface in Figure 1 , the procedure yields the layer assignment shown in Figure 3 .

However, it is sufficient for our purposes, and it is easier to work with mathematically.

Formally, for a piece-wise linear surface S, we denote k S := {S ⊆ S | S is a piece-wise linear surface of order at most k}.

One can show 4 that k S is itself a piece-wise linear surface of order at most k, so one can think of k S as the 'largest possible' subset of S that is a piece-wise linear surface of order at most k. For the piece-wise linear surface in Figure 3 , the set 1 S consists of the blue hyperplanes, 2 S consists of the blue and the orange (piece-wise) hyperplanes, and 3 S = S.

This definition allows us to uniquely decompose S into its piece-wise hyperplanes.

Let S = l∈[κ],i∈[n l ] H l i be any representation of S in terms of its piece-wise hyperplanes.

We say the rep-

.

One can show 5 that such a representation exists and is unique up to subscript indexing.

Importantly, it assigns a unique 'layer' to each piece-wise hyperplane, its superscript.

In other words, for architectures with non-increasing widths, there exists a ReLU network h such that knowledge of the input-output mapping h determines the network's parameters uniquely up to permutation and scaling.

The idea behind the proof is as follows.

Suppose we are given the function h. Then we also know its fold-set F(h), and if h is general and transparent, the fold-set is a piece-wise linear surface (by Lemma 2) of the form

As we have mentioned earlier, this union of zero-level sets contains a lot of information about the network's parameters, provided we can disambiguate the union to obtain the zero-level sets of individual units.

This disambiguation of the union is crucial, but is impossible in general.

To see why, consider the first-layer units: given F(h), we want to identify i z | h 1:1

is a union of d 1 hyperplanes, we are done.

In general however, F(h) may contain more than d 1 hyperplanes, such as for example in Figure 2 .

In such a setting it is impossible to tell which hyperplanes come from the first layer.

The key insight here is the following: even though, say, a last-layer unit can create a fold that looks like a hyperplane, this hyperplane cannot have any dependencies, or descendants in the dependency graph.

This follows from the fact that the layer is the last.

More generally, if a (piece-wise) hyperplane has a chain of descendants of length m, it must come from a layer that is at least m layers below the last one.

Formally, we have the following lemma.

Lemma 3.

Let h : Z → R be a general ReLU network.

Denote S : Main proof idea.

This lemma motivates the main idea of the proof.

We explicitly construct a network h such that the dependency graph of its fold-set is well connected.

More precisely, we ensure that each of the hyperplanes corresponding to first-layer units has a chain of descendants of length L − 2.

This implies by Lemma 3 that the first-layer hyperplanes can be identified as such, using only the information contained in the fold-set.

One can show that this is sufficient to recover the parameters W 1 , b 1 , up to permutation and scaling.

To extend the argument to higher-layers, we then consider the truncated network h l:L .

In h l:L , layer l becomes the first layer, and we apply the same reasoning as above to recover W l , b l .

The next lemma shows that a network with a 'well connected' dependency graph exists.

In what follows, f | A denotes the restriction of a function f to a domain A, and Z for all i. One can show 6 that this implies the existence of scalars m 1 , . . .

We know that

We have thus shown that there exists a permutation matrix P l ∈ R d l ×d l and a nonzero-entry diagonal matrix

One can also show that the scalars m i are positive.

For the inductive step, let l ∈ {2, . . .

, L − 1}, and assume that there exist permutation matrices P 1 , . . .

, P l−1 , and positive-entry diagonal matrices M 1 , . . . , M l−1 , such that (65) holds up to layer l − 1.

Then h

.

Since the end-to-end mappings are the same, h

whereη := (

We therefore apply the same argument to h

as we presented above for the case l = 1.

We obtain that there exists a permutation matrix P l ∈ R d l ×d l and a positive-entry diagonal matrix

Finally, consider the last layer.

We know that h

Discussion of assumptions.

Most of the theorem's assumptions have their origin in Lemma 4.

The reason we restrict the domain of h l:L to the interior of Z l−1 is that we want h l:L to be defined on an open set (otherwise fold-sets become unwieldy).

For similar reasons, we study only architectures with non-increasing widths; otherwise int Z l−1 may be empty.

We conjecture that the theorem does not hold for more general architectures.

If it does, the proof will likely go beyond fold-sets.

To guarantee transparency, our construction is such that for each input z ∈ Z and layer l ∈ [L − 1], either h 1:l 1 (z) > 0 or h 1:l 2 (z) > 0.

Transparency could in principle be achieved with just a single unit, but it would have to be positive everywhere.

This is why we impose d l ≥ 2.

Guaranteeing transparency for the first layer (whose inputs are not constrained to the positive quadrant) also necessitates boundedness of Z. Boundedness can be lifted if we consider a slightly modified definition of transparency; proofs become more complicated though and we do not consider this crucial.

Almost all of the proof carries over to the case of leaky ReLU activations (where σ is defined as σ(u) i = max {αu i , u i } for some small α > 0).

The part that does not carry over is our proof that M l has only positive entries on the diagonal: In this part, we compare the slope of h l:L θ for inputs on the positive and negative side of a given ReLU unit, and notice that the negative-side slope is 'singular' in the sense that some basis directions have zero magnitude.

This particular argument does not work for the leaky ReLU, though we cannot rule out that a simple workaround exists.

In this work, we have shown that for architectures with non-increasing widths, certain ReLU networks are almost uniquely identified by the function they implement.

The result suggests that the function-equivalence classes of ReLU networks are surprisingly small, i.e. there may be only little redundancy in the way ReLU networks are parameterised, contrary to what is commonly believed.

This apparent contradiction could be explained in a number of ways:

• It could be the case that even though exact equivalence classes are small, approximate equivalence is much easier to achieve.

That is, it could be that h θ − h η ≤ is satisfied by a disproportionately larger class of parameters η than h θ − h η = 0.

This issue is related to the so-called inverse stability of the realisation map of neural nets, which is not yet well understood.

• Another possibility is that the kind of networks we consider in this paper is not representative of networks typically encountered in practice, i.e. it could be that 'typical networks' do not have well connected dependency graphs, and are therefore not easily identifiable.

• Finally, we have considered only architectures with non-increasing widths, whereas some previous theoretical work has assumed much wider intermediate layers compared to the input dimension.

It is possible that parameterisation redundancy is much larger in such a regime compared to ours.

However, gains from over-parameterisation have also been observed in practical settings with architectures not unlike those considered here.

We consider these questions important directions for further research.

We also hypothesise that our analysis could be extended to convolutional and recurrent networks, and to other piece-wise linear activation functions such as leaky ReLU.

Definition A.1 (Partition).

Let S ⊆ Z. We define the partition of Z induced by S, denoted P Z (S), as the set of connected components of Z \ S. Definition A.2 (Piece-wise hyperplane).

Let P be a partition of Z. We say H ⊆ Z is a piece-wise hyperplane with respect to partition P, if H = ∅ and there exist (w, b) = (0, 0) and P ∈ P such that H = {z ∈ P | w z + b = 0}. Definition A.3 (Piece-wise linear surface / pwl.

surface).

A set S ⊆ Z is called a piece-wise linear surface on Z of order κ if it can be written as

, and no number smaller than κ admits such a representation.

Lemma A.1.

If S 1 , S 2 are piece-wise linear surfaces on Z of order k 1 and k 2 , then S 1 ∪ S 2 is a piece-wise linear surface on Z of order at most max {k 1 , k 2 }.

We can write H

Given sets Z and S ⊆ Z, we introduce the notation

(The dependence on Z is suppressed.)

By Lemma A.1, i S is itself a pwl.

surface on Z of order at most i. Lemma A.2.

For i ≤ j and any set S, we have i j S = j i S = i S.

Proof.

We will need these definitions:

j S = {S ⊆ S | S is a pwl.

surface of order at most j},

i j S = {S ⊆ j S | S is a pwl.

surface of order at most i},

surface of order at most j}.

Consider first the equality j i S = i S.

We know that j i S ⊆ i S because the square operator always yields a subset.

At the same time, i S ⊆ j i S, because i S satisfies the condition for membership in (6).

To prove the equality i j S = i S, we use the inclusion j S ⊆ S to deduce i j S ⊆ i S. Now let S ⊆ S be one of the sets under the union in (3), i.e. it is a pwl.

surface of order at most i.

Then it is also a pwl.

surface of order at most j, implying S ⊆

j S.

This means S is also one of the sets under the union in (5), proving that i S ⊆ i j S.

Lemma A.3.

Let Z and S ⊆ Z be sets.

Then one can write k+1 S = k S ∪ i H i where H i are piece-wise hyperplanes wrt.

P Z ( k S).

Proof.

At the same time,

is a pwl.

surface of order at most k + 1 because k S is a pwl.

surface of order at most k and H k+1 i

can be decomposed into piece-wise hyperplanes wrt.

Definition A.4 (Canonical representation of a pwl.

surface).

Let S be a pwl.

surface on Z. The pwl.

is a pwl.

surface in canonical form, then κ is the order of S.

Proof.

Denote the order of S by λ.

By the definition of order, λ ≤ κ, and S = λ S. Then, since

It follows that κ = λ.

Lemma A.5.

Every pwl.

surface has a canonical representation.

Proof.

The inclusion l∈[k],i∈[n l ]

H l i ⊆ k S holds for any representation.

We will show the other inclusion by induction in the order of S. If S is order one, 1 S ⊆ S = i∈[n1] H 1 i holds for any representation and we are done.

Now assume the lemma holds up to order κ − 1, and let S be order κ.

Then by Lemma A.3, S = κ S = κ−1 S ∪ i H κ i , where H κ i are piece-wise hyperplanes wrt.

P Z ( κ−1 S).

By the inductive assumption, κ−1 S has a canonical representation, Proof.

Let k ∈ [κ].

Because both representations are canonical, we have

where H k i and G k j are piece-wise hyperplanes wrt.

where on both sides above we have a union of hyperplanes on an open set.

The claim follows.

Definition A.5 (Dependency graph of a pwl.

surface).

Let S be a piece-wise linear surface on Z, and let S = l∈[κ],i∈[n l ] H l i be its canonical representation.

We define the dependency graph of S as the directed graph that has the piece-wise hyperplanes H l i l,i as vertices, and has an edge

We denote by σ the ReLU function: σ(u) i = max {0, u i } for i ∈ [dim(u)].

Definition A.6 (ReLU network).

Let Z ⊆ R d0 with d 0 ≥ 2 be a nonempty open set, and let

where

For a ReLU network h θ :

(We will omit the subscript θ when it is clear from the context.)

We write f | A to denote the restriction of the function f to the domain A.

(I 1 , . . .

, I L−1 ) is called an activation indicator if I l = diag(i l ) ∈ R d l ×d l and i l ∈ {0, 1} d l for l ∈ [L − 1].

It is called non-trivial if i l = 0 for all l ∈ [L − 1] and non-trivial up to k if i l = 0 for all l ∈ [k].

and an activation indicator I, we introduce the notation

(We will omit the argument θ when it is clear from the context.)

These quantities characterise the different linear pieces implemented by the network's units.

Also define I θ (z) (I

Proof.

Left as exercise.

Definition A.8 (Fold-set).

Let Z be an open set, and f : Z → R a continuous, piece-wise linear function.

We define the fold-set of f , denoted by F(f ), as the set of all points at which f is nondifferentiable.

Definition A.9 (Positive / negative in a neighbourhood).

Let Z be an open set.

The function f : Z → R is positive (negative) in the neighbourhood of z ∈ Z if for any > 0 there exists z ∈ B (z) such that f (z ) > 0 (f (z ) < 0).

Definition A.10 (Unit fold-set).

Let h θ : Z → R be a ReLU network.

We define the unit (l, i) fold-set of h θ , denoted F Proof.

We will prove that if z satisfies any of the two conditions, then z ∈ F(σ •f ), and if it violates both, then z ∈ F(σ • f ) c .

We begin with the latter implication.

Let z be such that f (z) > 0 and z / ∈ F(f ), i.e. f is differentiable at z. Since f is piece-wise linear, there exists > 0 such that all of B (z) lies inside a single linear region of f and f (B (z)) ⊆ (0, ∞].

Then, on B (z), the ReLU behaves like an identity, implying σ • f is differentiable at z, proving that z ∈ F(σ • f ) c .

Next, consider z such that f (z) = 0.

For it to violate the second condition, there must exist a ball B (z) around z such that f (B (z)) ⊆ (−∞, 0].

(This is also true if f (z) < 0.)

Then, on B (z), the ReLU behaves like a constant zero, implying that σ • f is differentiable at z.

We now prove the other implication.

If f (z) > 0 and z ∈ F(f ), then there exists > 0 such that f (B (z)) ⊆ (0, ∞], which guarantees that the ReLU behaves like an identity on B (z).

In this ball, we have σ

If f (z) = 0 and f is positive in the neighbourhood of z, we distinguish several cases.

If z / ∈ F(f ), then there exists a ball B δ (z) on which f behaves linearly, i.e. σ(f (z)) = σ(w z + b), implying z ∈ F(σ • f ).

If z ∈ F(f ) and, in addition, there exists a ball B δ (z) such that f (B δ (z)) ⊆ [0, ∞), then the ReLU behaves like an identity on B δ (z) and z ∈ F(σ • f ).

The final case is z ∈ F(f ) such that f attains both positive and negative values in its neighbourhood.

Since f is piece-wise linear, there exist p, n such that f (z + n) < 0 < f (z + p), and

Lemma A.9.

Let Z be an open set, and let f 1 , . . .

, f n : Z → R be continuous, piece-wise linear functions.

For any w 1 , . . .

,

Proof.

Left as exercise.

Lemma A.10.

For all θ except a closed zero-measure set,

for all activation indicators I and all k ≤ l.

Proof.

First, notice that (16) is just a special case of (17) with I l equal to the identity matrix.

It therefore suffices to prove (17).

To further simplify, we will prove the statement for a single fixed activation indicator I. Then if Θ(I) is the set of networks for which (17) holds given I, and Θ(I) contains all networks except a closed zero-measure set, then also I Θ(I) contains all networks except a closed zero-measure set, proving the lemma.

Let us hence fix I, and let k ∈ [L].

We proceed by induction.

For the initial step, notice that the matrix I k W k is just W k with some rows replaced by zeroes.

The rank of such a matrix is the same as the matrix obtained by removing the zero rows, which has size (rank(I k ), d k−1 ).

For all W k except a closed zero-measure set, this matrix has rank min {d k−1 , rank(I k )}.

For the inductive step, denoteW i := I i W i · · ·

I k W k and

We assume that rank(W i−1 ) = r i−1 and want to prove the same for i. Notice that for all W i except a closed zero-measure set, any r i rows of W i are linearly independent and their span intersects with ker(W i−1 ) only at 0.

To see this, recall that by the inductive assumption, rank(W i−1 ) = r i−1 , so ker(W i−1 ) has dimension d i−1 −r i−1 .

We can concatenate any r i -subset of rows of W i to the basis of ker(W i−1 ) to obtain a matrix of size (r i + d i−1 − r i−1 , d i−1 ), which is a wide matrix, because r i ≤ r i−1 .

Hence, its rows are linearly independent for all W i except a closed zero-measure set.

We now prove that rank(I i W iWi−1 ) = min rank(W i−1 ), rank(I i )

r i .

The "≤" direction is immediate.

For the "≥" direction, we distinguish between two cases.

If rank(I i ) ≤ rank(W i−1 ), let v 1 , . . .

v ri be the (linearly independent) nonzero rows of I i W i .

We want to show that v jW i−1 j are linearly independent, i.e. that I i W iWi−1 has at least r i linearly independent rows.

If ri j=1 λ j v jW i−1 = 0, then ri j=1 λ j v j ∈ ker(W i−1 ), which by assumption implies λ j v j = 0.

By the independence of {v j }, we obtain λ j = 0, i.e. v jW i−1 j are linearly independent, and rank(I i W iWi−1 ) = r i .

If rank(I i ) > rank(W i−1 ), we can reduce the problem to the case rank(I i ) ≤ rank(W i−1 ) by observing that rank(I i W iWi−1 ) ≥ rank(J i W iWi−1 ) if J i equals I i only with some 1's replaced by 0's.

We can thus take any such J i and apply the argument from the previous paragraph to obtain rank(

Lemma A.11.

For all θ except a closed zero-measure set, the following holds.

Let (l, i), (k, j) be any units, let I be an activation indicator non-trivial up to l − 1, and let J be an activation indicator non-trivial up to k − 1, such that (l, i, I 1:l−1 ) = (k, j, J 1:k−1 ).

Then, for all scalars c ∈ R, it holds that [w

Proof.

First, we exclude from consideration all θ = (

for some l, k, i, j, and some I non-trivial up to l − 1.

Since for any fixed (l, k, i, j, I), the set of θ satisfying the above is the set of roots of a non-trivial polynomial in θ, it is zero-measure and closed.

Because there are only finitely many configurations of (l, k, i, j, I), we have thus excluded a closed zero-measure set of parameters.

We will denote its complement Θ * .

From now on, we assume θ ∈ Θ * .

Notice that the case c = 0 of the lemma is thus automatically satisfied, since w

In the following, we can therefore assume c = 0 and treat (l, i, I) and (k, j, J) symmetrically.

Denote by Θ ¬ ⊆ Θ * the set of parameters θ for which the lemma does not hold; we need to show that Θ ¬ is closed and zero-measure.

We start by showing the latter property by contradiction.

Suppose Θ ¬ is positive-measure.

We know that for all θ ∈ Θ ¬ , there exist triples (l, i, I), (k, j, J) as stated in the lemma, and a scalar c ∈ R such that [w

Let C denote the set of all triplet-pairs ((l, i, I), (k, j, J)) satisfying the conditions of the lemma; then the previous statement can be written as

Since C is finite, there exist ((l, i, I), (k, j, J)) ∈ C for which the set under the union (call it Θ ) is positive-measure.

We now consider two cases.

If (l, i) = (k, j), then observe that Θ must contain some θ, θ such

[w

Notice that w

Putting everything together, we have that

which implies (c − c)v = [0, δ], and in particular w k j (θ, J) = 0.

This contradicts the assumption that θ ∈ Θ * and completes the proof for the case (l, i) = (k, j).

Definition A.11 (General ReLU network).

A ReLU network is general if it satisfies Lemmas A.10, A.11 and A.12.

All ReLU networks except a closed zero-measure set are general.

Lemma A.13.

If h is a general ReLU network, then F(h

) follows from Lemma A.9.

For the other inclusion, let z ∈ F(ȟ

)

such that I(z 1 ( )) =: I and I(z 2 ( )) =: J are independent of , and ∇ȟ

.

We consider three cases based on the (non-)triviality of I and J.

First, suppose both I and J are trivial up to l − 1.

Then by Lemma A.7,

and similarly ∇ȟ 1:l−1 k (z 2 ( )) = 0, which contradicts ∇ȟ

Hence, at least one of I, J, must be non-trivial up to l − 1.

Second, say both I and J are non-trivial up to l − 1.

From ∇ȟ

follows that I 1:l−1 = J 1:l−1 , we can therefore apply Lemma A.11 to (l, i, I) and (l, i, J).

We obtain w Lemma A.14.

Let h : Z → R be a ReLU network, and let

Proof.

We will abbreviate h λ:L | int Z λ−1 as h λ:L .

Assume h is general.

Then h λ:L clearly satisfies Lemma A.10, and for all (l, i), W l [i, :] = 0 .

Next, we prove that h λ:L satisfies Lemma A.11.

Suppose this was not the case; then there exist units (λ − 1 + l, i), (λ − 1 + k, j), and non-trivial activation indicators I = (I λ , . . .

, I λ−1+l ), J = (J λ , . . .

, I λ−1+k ), with (l, i, I) = (k, j, J), and a scalar C ∈ R such that

and

Then for any non-trivial indicator (I 1 , . . . , I λ−1 ) (J 1 , . . . , J λ−1 ), we obtain by post-multiplying (29),

and for all ι ∈ [λ − 1],

The first equality means that w The last condition of generality is Lemma A.12.

Suppose h λ:L does not satisfy the lemma.

Then there exists a unit (l, i) such that

i is positive and negative in the neighbourhood of z , i.e. there exists z ∈ int Z l−1 such that h λ:l i (z) = 0, and for some > 0 either h

However, then there exists z ∈ Z such thatȟ 1:l−1 (z ) = z, and for z we obtain h 1:l i (z ) = 0, and by continuity, there is δ > 0 such that either h

This contradicts the fact that h satisfies Lemma A.12.

We have thus shown that if h is general, then h λ:L | int Z λ−1 is general.

Finally, assume h is transparent, i.e. for all z ∈ Z and l

Lemma A.15.

a) For all ReLU networks h :

In particular, for all general transparent ReLU networks,

Proof.

We give a proof of b) only.

A proof of a) can be obtained by replacing some equalities by inclusions.

We will prove by induction that F(h

holds; we will prove the same statement for l + 1.

By Lemma A.8 and Lemma A.13, we have

Since

It remains to show the reverse inclusion; we do so by contradiction.

) is the partition of the input space into the linear regions of h 1:l+1 i , and Lemma A.15 , the function h 1:l+1 i is also linear on the regions of

, denote the slope and bias of h 1:l+1 i on P by w(P ), b(P ).

Then

The positivity condition guarantees that (w(P ),

is either an empty set or a piece-wise hyperplane.

• the set F(h l:L | intZ l−1 ) is a pwl.

surface whose dependency graph contains d l directed paths of length (L − 1 − l) with distinct starting vertices.

We will show that this construction satisfies the lemma.

The networks are transparent because of how we define W l 2 : for all x ∈ X and l ∈ [L − 1], either h

be its canonical representation, and let G denote its dependency graph.

To find the required paths in G, we first identify some important vertices.

For λ ∈ [L − l], denote

This set is nonempty and open because P l+λ is nonempty and open.

Next, for any unit (λ, ι),

By the definition of W l+λ−1 1 and the fact that h l:l+λ−1 (Z We now show that G contains the edge H

Then because of how {P l } l are defined, there existsz ∈ P l such that h l:l+λ−1 (z ) =z, so it satisfies

It follows thatz ∈ F λ,i) .

At the same time, the preimage (h l:l+λ−1 ) −1 (B (z)) is open by continuity, and containsz .

So there exists a ball B (z ) ⊆ P l such that all z ∈ B (z ) satisfy (z ) = 0 intersects the center of the half-ball, z .

Therefore there exists a sequence of points {z n } ⊆ P l such that z n →z and

We obtain thatz ∈ cl(F Proof.

Because the representation is canonical, we have

which implies H

where P runs over the linear regions of h jι−1 , but included in the same hyperplane.

However, by Lemma A.11, no two piece-wise hyperplanes in S are included in a single hyperplane, so we get a contradiction.

Hence, we obtain l 0 < l 1 < · · · < l m ≤ λ, which yields l 0 ≤ λ − m. Let h θ : X → R be a general ReLU network satisfying Lemma A.17, and let h η : X → R be any general ReLU network such that h θ (x) = h η (x) for all x ∈ X. Denote η (W 1 , b 1 , . . .

, W L , b L ).

Then there exist permutation matrices P 1 , . . .

P L−1 , and positiveentry diagonal matrices M 1 , . . .

, M L−1 , such that

Published as a conference paper at ICLR 2020

Finally, consider the last layer.

We know that h

@highlight

We prove that there exist ReLU networks whose parameters are almost uniquely determined by the function they implement.