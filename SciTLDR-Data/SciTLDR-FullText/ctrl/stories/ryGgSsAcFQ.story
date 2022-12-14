In order to choose a neural network architecture that will be effective for a particular modeling problem, one must understand the limitations imposed by each of the potential options.

These limitations are typically described in terms of information theoretic bounds, or by comparing the relative complexity needed to approximate example functions between different architectures.

In this paper, we examine the topological constraints that the architecture of a neural network imposes on the level sets of all the functions that it is able to approximate.

This approach is novel for both the nature of the limitations and the fact that they are independent of network depth for a broad family of activation functions.

Neural networks have become the model of choice in a variety of machine learning applications, due to their flexibility and generality.

However, selecting network architectures and other hyperparameters is typically a matter of trial and error.

To make the choice of neural network architecture more straightforward, we need to understand the limits of each architecture, both in terms of what kinds of functions any given network architecture can approximate and how those limitations impact its ability to learn functions within those limits.

A number of papers (3; 6; 11; 13) have shown that neural networks with a single hidden layer are a universal approximator, i.e. that they can approximate any continuous function on a compact domain to arbitrary accuracy if the hidden layer is allowed to have an arbitrarily high dimension.

In practice, however, the neural networks that have proved most effective tend to have a large number of relatively low-dimensional hidden layers.

This raises the question of whether neural networks with an arbitrary number of hidden layers of bounded dimension are also a universal approximator.

In this paper we demonstrate a fairly general limitation on functions that can be approximated with the L ∞ norm on compact subsets of a Euclidean input space by layered, fully-connected feedforward neural networks of arbitrary depth and activation functions from a broad family including sigmoids and ReLus, but with layer widths bounded by the dimension of the input space.

By a layered network, we mean that hidden nodes are grouped into successive layers and each node is only connected to nodes in the previous layer and the next layer.

The constraints on the functions are defined in terms of topological properties of the level sets in the input space.

This analysis is not meant to suggest that deep networks are worse than shallow networks, but rather to better understand how and why they will perform differently on different data sets.

In fact, these limitations may be part of the reason deep nets have proven more effective on datasets whose structures are compatible with these limitations.

By a level set, we mean the set of all points in the input space that the model maps to a given value in the output space.

For classification models, a level set is just a decision boundary for a particular cutoff.

For regression problems, level sets don't have a common interpretation.

The main result of the paper, Theorem 1, states that the deep, skinny neural network architectures described above cannot approximate any function with a level set that is bounded in the input space.

This can be rephrased as saying that for every function that can be approximated, every level set must be unbounded, extending off to infinity.

While a number of recent papers have made impressive progress in understanding the limitations of different neural network architectures, this result is notable because it is independent of the number of layers in the network, and because the limitations are defined in terms of a very simple topological property.

Topological tools have recently been employed to study the properties of data sets within the field known as Topological Data Analysis (9), but this paper exploits topological ideas to examine the topology of the models themselves.

By demonstrating topological constraints on a widely used family of models, we suggest that there is further potential to apply topological ideas to understand the strengths and weaknesses of algorithms and methodologies across machine learning.

After discussing the context and related work in Section 2, we introduce the basic definitions and notation in Section 3, then state the main Theorem and outline the proof in Section 4.

The detailed proof is presented in Sections 5 and 6.

We present experimental results that demonstrate the constraints in Section 7, then in Section 8 we present conclusions from this work.

A number of papers have demonstrated limitations on the functions that can be approximated by neural networks with particular architectures (2; 12; 14; 15; 18; 19; 21; 22; 23; 24; 25; 27; 29; 32) .

These are typically presented as asymptotic bounds on the size of network needed to approximate any function in a given family to a given ε.

Lu et al BID15 gave the first non-approximation result that is independent of complexity, showing that there are functions that no ReLu-based deep network of width equal to the dimension of the input space can approximate, no matter how deep.

However, they consider convergence in terms of the L 1 norm on the entire space R n rather than L ∞ on a compact subset.

This is a much stricter definition than the one used in this paper so even for ReLu networks, Theorem 1 is a stronger result.

The closest existing result to Theorem 1 is a recent paper by Nguyen, Mukkamala and Hein BID24 which shows that for multi-label classification problems defined by an argmax condition on a higherdimensional output function, if all the hidden layers of a neural network have dimension less than or equal to the input dimension then the region defining each class must be connected.

The result applies to one-to-one activation functions, but could probably be extended to the family of activation functions in this paper by a similar limiting argument.

Universality results have been proved for a number of variants of the networks described in Theorem 1.

Rojas BID26 showed that any two discrete classes of points can be separated by a decision boundary of a function defined by a deep, skinny network in which each layer has a single perceptron that is connected both to the previous layer and to the input layer.

Because of the connections back to the input space, such a network is not layered as defined above, so Theorem 1 doesn't contradict this result.

In fact, to carry out Rojas' construction with a layered feed-forward network, you would need to put all the perceptrons in a single hidden layer.

Sutskever and Hinton BID28 showed that deep belief networks whose hidden layers have the same dimension as the input space can approximate any function over binary vectors.

This binary input space can be interpreted as a discrete subset of Euclidean space.

So while Theorem 1 does not apply to belief networks, it's worth noting that any function on a discrete set can be extended to the full space in such a way that the resulting function satisfies the constraints in Theorem 1.This unexpected constraint on skinny deep nets raises the question of whether such networks are so practically effective despite being more restrictive than wide networks, or because of it.

Lin, Tegmark and Rolnick BID14 showed that for data sets with information-theoretic properties that are common in physics and elsewhere, deep networks are more efficient than shallow networks.

This may be because such networks are restricted to a smaller search space concentrated around functions that model shapes of data that are more likely to appear in practice.

Such a conclusion would be consistent with a number of papers showing that there are functions defined by deep networks that can only by approximated by shallow networks with asymptotically much larger number of nodes (4; 7; 10; 20; 31).A slightly different phenomenon has been observed for recurrent neural networks, which are universal approximators of dynamic systems BID6 .

In this setting, Collins, Sohl-Dickstein and Sussillo BID3 showed that many differences that have been reported on the performance of RNNs are due to their training effectiveness, rather than the expressiveness of the networks.

In other words, the effectiveness of a given family of models appears to have less to do with whether it includes an accurate model, and more to do with whether a model search algorithm like gradient descent is likely to find an accurate model within the search space of possible models.

A model family is a subset M of the space C(R n , R m ) of continuous functions from input space R n to output space R m .

For parametric models, this subset is typically defined as the image of a map DISPLAYFORM0 , where R k is the parameter space.

A non-parametric model family is typically the union of a countably infinite collection of parametric model families.

We will not distinguish between parametric and non-parametric families in this section.

Given a function g : R n → R m , a compact subset A ⊂ R n and a value > 0, we will say that a second function f ( , A)-approximates g if for every x ∈ A, we have |f (x) − g(x)| < .

Similarly, we will say that a model family M ( , A)-approximates g if there is a function f in M that ( , A)-approximates g.

More generally, we will say that M approximates f if for every compact A ⊂ R n and value > 0 there is a function f in M that ( , A)-approximates g. This is equivalent to the statement that there is a sequence of functions f i ∈ M that converges pointwise (though not necessarily uniformly) to g on all of R n .

However, we will use the ( , A) definition throughout this paper.

We'll describe families of layered neural networks with the following notation: Given an activation function ϕ : R → R and a finite sequence of positive integers n 0 , n 1 , . . .

, n κ , let N ϕ,n0,n1,...,nκ be the family of functions defined by a layered feed-forward neural network with n 0 inputs, n κ outputs and fully connected hidden layers of width n 1 , . . .

, n κ−1 .With this terminology, Hornik et al's results can be restated as saying that the (non-parametric) model family defined as the union of all families N ϕ,n0,n1,1 approximates any continuous function.(Here, κ = 2 and n 2 = 1.)We're interested in deep networks with bounded dimensional layers, so we'll let N * ϕ,n be the union of all the model families N ϕ,n0,n1,...

,nκ−1,1 such that n i ≤ n for all i < κ.

For the main result, we will restrict our attention to a fairly large family of activation functions.

We will say that an activation function ϕ is uniformly approximated by one-to-one functions if there is a sequence of continuous, one-to-one functions that converge to ϕ uniformly (not just pointwise).Note that if the activation function is itself one-to-one (such as a sigmoid) then we can let every function in the sequence be ϕ and it will converge uniformly.

For the ReLu function, we need to replace the the large horizontal portion with a function such as 1 n arctan(x).

Since this function is one-to-one and negative for x < 0, each function in this sequence will be one-to-one.

Since it's bounded between − 1 n and 0, the sequence will converge uniformly to the ReLu function.

The main result of the paper is a topological constraint on the level sets of any function in the family of models N * ϕ,n .

To understand this constraint, recall that in topology, a set C is path connected if any two points in C are connected by a continuous path within C. A path component of a set A is a subset C ⊂ A that is connected, but is not a proper subset of a larger connected subset of A. Definition 1.

We will say that a function f : R n → R has unbounded level components if for every y ∈ R, every path component of f −1 (y) is unbounded.

The main result of this paper states that deep, skinny neural networks can only approximate functions with unbounded level components.

Note that this definition is stricter than just requiring that every level set be bounded.

The stricter definition in terms of path components guarantees that the property is preserved by limits, a fact that we will prove, then use in the proof of Theorem 1.

Just having bounded level sets is not preserved under limits.

Theorem 1.

For any integer n ≥ 2 and uniformly continuous activation function ϕ : R → R that can be approximated by one-to-one functions, the family of layered feed-forward neural networks with input dimension n in which each hidden layer has dimension at most n cannot approximate any function with a level set containing a bounded path component.

The proof of Theorem 1 consists of two steps.

In the first step, described in Section 5, we examine the family of functions defined by deep, skinny neural networks in which the activation is one-to-one and the transition matrices are all non-singular.

We prove two results about this smaller family of functions: First, Lemma 2 states that any function that can be approximated by N * ϕ,n can be approximated by functions in this smaller family.

This is fairly immediate from the assumptions on ϕ and the fact that singular transition matrices can be approximated by non-singular ones.

Second, Lemma 4 states that the level sets of these functions have unbounded level components.

The proof of this Lemma is, in many ways, the core argument of the paper and is illustrated in FIG0 .

The idea is that any function in this smaller family can be written as a composition of a one-to-one function and a linear projection, as in the top row of the Figure.

As suggested in the bottom row, this implies that each level set/decision boundary in the full function is defined by the intersection of the image of the one-to-one function (the gray patch in the middle) with a hyperplane that maps to a single point in the second function.

Intuitively, this intersection extends out to the edges of the gray blob, so its preimage in the original space must extend out to infinity in Euclidean space, i.e. it must be unbounded.

The second part of the proof of Theorem 1, described in Section 5, is Lemma 5 which states that the limit of functions with unbounded level components also has unbounded level components.

This is a subtle technical argument, though it should be intuitively unsurprising that unbounded sets cannot converge to bounded sets.

The proof of Theorem 1 is the concatenation of these three Lemmas: If a function can be approximated by N * ϕ,n then it can be approximated by the smaller model family (Lemma 2), so it can be approximated by functions with unbounded level components (Lemma 4), so it must also have unbounded level components (Lemma 5).

We will say that a function in N * ϕ,n is non-singular if ϕ is continuous and one-to-one, n i = n for all i < k and the matrix defined by the weights between each pair of layers is nonsingular.

Note that if ϕ is not one-to-one, then N * ϕ,n will not contain any non-singular functions.

If it is one-to-one then N * ϕ,n will contain a mix of singular and non-singular functions.

Define the model family of non-singular functionsN n to be the union of all non-singular functions in families N * ϕ,n for all activation functions ϕ and a fixed n. Lemma 2.

If g is approximated by N * ϕ,n for some continuous activation function ϕ that can be uniformly approximated by one-to-one functions then it is approximated byN n .To prove this Lemma, we will employ a technical result from point-set topology, relying on the fact that a function in N * ϕ,n can be written as a composition of linear functions defined by the weights between successive layers, and non-linear functions defined by the activation function ϕ. DISPLAYFORM0 is a continuous function.

Let A ⊂ R n be a compact subset and choose ε > 0.

One can prove Lemma 3 by induction on the number of functions in the composition, choosing each A i ⊂ R ni to be a closed ε-neighborhood of the image of A in the composition up to i. For each new function, the δ on the compact set tells you what δ you need to choose for the composition of the preceding functions.

We will not include the details here.

Proof of Lemma 2.

We'll prove this Lemma by showing thatN n approximates any given function in N * ϕ,n .

Then, given ε > 0, a compact set A ⊂ R n and a function g that is approximated by N * ϕ,n , we can choose a function f ∈ N * ϕ,n that (ε/2, A)-approximates g and a function inN n that (ε/2, A)-approximates f .

So we will reset the notation, let g be a function in N * ϕ,n , let A ⊂ R n be a compact subset and choose ε > 0.

As noted above, g is a composition g = ν κ • κ • · · · • ν 0 • 0 where each i is a linear function defined by the weights between consecutive layers and each ν i is a nonlinear function defined by a direct product of the activation function ϕ.If any of the hidden layers in the network defining g have dimension strictly less than n then we can define the same function with a network in which that layer has dimension exactly n, but the weights in and out of the added neurons are all zero.

Therefore, we can assume without loss of generality that all the hidden layers in g have dimension exactly n, though the linear functions may be singular.

Let {A i } and δ > 0 be as defined by Lemma 3.

We want to find functionsν i andˆ i that (δ, A i )-approximate each ν i and i and whose composition is inN n .For the composition to be inN n , we need eachˆ i to be non-singular.

If i is already non-singular, then we chooseˆ i = i .

Otherwise, we can perturb the weights that define the linear map i by an arbitrarily small amount to make it non-singular.

In particular, we can choose this arbitrarily small amount to be small enough that the function values change by less than δ on A i .

Similarly, we want eachν i to be a direct product of a continuous, one-to-one activation functions.

By assumption, ϕ can be approximated by such functions and we can choose the tolerance for this approximation to be small enough thatν i (δ, A i )-approximates ν i .

In fact, we can choose a single activation function for all the nonlinear layers, on each corresponding compact set.

Thus we can choose eachˆ i and an activation functionφ that defines all the functions ν i , so that the composition is inN n and, by Lemma 3, the composition (ε, A)-approximates g.

Lemma 2 implies that if N * ϕ,n is universal then so isN n .

So to prove Theorem 1, we will show that every function inN n has level sets with only unbounded components, then show that this property extends to any function that it approximates.

Lemma 4.

If f is a function inN n then every level set f −1 (y) is homeomorphic to an open (possibly empty) subset of R n−1 .

This implies that f has unbounded level components.

Proof.

Assume f is a non-singular function inN n , where ϕ is continuous and one-to-one.

Let f : R n → R n be the function defined by all but the last layer of the network.

Letf : R n → R be the function defined by the map from the last hidden layer to the final output layer so that f =f •f .The functionf is a composition of the linear functions defined by the network weights and the nonlinear function at each step defined by applying the activation function to each dimension.

Because f is nonsingular, the linear functions are all one-to one.

Because ϕ is continuous and one-to-one, so are all the non-linear functions.

Thus the compositionf is also one-to-one, and therefore a homeomorphism from R n onto its image If .

Since R n is homeomorphic to an open n-dimensional ball, If is an open subset of R n , as indicated in the top row of FIG0 .The functionf is the composition of a linear function to R with ϕ, which is one-to-one by assumption.

So the preimagef −1 (y) for any y ∈ R is an (n − 1)-dimensional plane in R n .

The preimage f −1 (y) is the preimage inf of this (n − 1)-dimensional plane, or rather the preimage of the intersection If ∩f −1 (y), as indicated in the bottom right/center of the Figure.

Since If is open as a subset of R n , the intersection is open as a subset off −1 (y).Sincef is one-to-one, its restriction to this preimage (shown on the bottom left of the Figure) is a homeomorphism from f −1 (y) to this open subset of the (n − 1)-dimensional planef −1 (y).

Thus f −1 (y) is homeomorphic to an open subset of R n−1 .Finally, recall that the preimage in a continuous function of a closed set is closed, so f −1 (y) is closed as a subset of R n .

If it were also bounded, then it would be compact.

However, the only compact, open subset of R n−1 is the empty set, so f −1 (y) is either unbounded or empty.

Since each path component of a subset of R n−1 is by definition non-empty, this proves that any component of f is unbounded.

All that remains is to show that this property extends to the functions thatN n approximates.

If M is a model family in which every function has unbounded level components then any function approximated by M has unbounded level components.

Proof.

Let g : R n → R be a function with a level set g −1 (y) containing a bounded path component C. Note that level sets are closed as subsets of R n and bounded, closed sets are compact so C is compact.

We can therefore choose a value µ such that any point of g −1 (y) outside of C is distance greater than µ from every point in C.Let η C be the set of all points that are distance strictly less than µ/2 from C. This is an open subset of R n , shown as the shaded region in the center of FIG1 , and we will let F be the frontier of η C -the set of all points that are limit points of both η C and limit points of its complement.

By construction, every point in F is distance µ/2 from C so F is disjoint from C. Moreover, since every point in g −1 (y) \ C is distance at least µ from C, F is disjoint from the rest of g −1 (y) as well, so y is in the complement of g(F )

.The frontier is the intersection of two closed sets, so F is closed.

It's also bounded, since all points are a bounded distance from C, so F is compact.

This implies that g(F ) is a compact subset of R, so its complement is open.

Since y is in the complement of g(F ), this means that there is an open interval U = (y − ε, y + ε) that is disjoint from g(F ).LetĈ be the component of g −1 (U ) that contains C, as indicated on the right of the Figure.

Note that this set intersects η C but is disjoint from its frontier.

SoĈ must be contained in η C , and is therefore bounded as well.

In particular, each level set that intersectsĈ has a compact component inĈ.Let x be a point in C ⊂Ĉ. SinceĈ is bounded, there is a value r such that every point inĈ is distance at most r from x.

Assume for contradiction that g is approximated by a model family M in which each function has unbounded level components.

Choose R > r and let B R (x) be a closed ball of radius R, centered at x. Because this is a compact set and g is approximated by M , we can choose a function f ∈ M that (ε/2, B R (x))-approximates g.

Then |f (x) − g(x)| < ε/2 so f (x) ∈ [y − ε/2, y + ε/2] ⊂ U and we will define y = f (x).Since f ∈ M , every path component of f −1 (y ) is unbounded, so there is a path ⊂ f −1 (y ) from x to a point that is distance R from x. If passes outside of B R (x)), we can replace with the component of ∩ B R (x)) containing x to ensure that stays inside of B R (x)), but still reaches a point that is distance R from x.

Since every point x ∈ is contained in B R (x), we have |f (x ) − g(x )| < ε/2.

This implies g(x ) ∈ [y − ε, y + ε] = U so the path is contained in g −1 (U ), and thus in the path componentĈ of g −1 (U ).However, by construction the path ends at a point whose distance from x is R > r, contradicting the assumption that every point inĈ is distance at most r from x. This contradiction proves that g cannot be approximated by a model family M in which each function has unbounded level components.

Proof of Theorem 1.

Let g be a function that is approximated by N * ϕ,n , where ϕ is a continuous activation function that can be uniformly approximated by one-to-one functions.

By Lemma 2, since g is approximated by N (a) The decision boundary learned with six, two-dimensional hidden layers is an unbounded curve that extends outside the region visible in the image.(b) A network with a single threedimensional hidden layer learns a bounded decision boundary relatively easily.

To demonstrate the effect of Theorem 1, we used the TensorFlow Neural Network Playground (1) to train two different networks on a standard synthetic dataset with one class centered at the origin of the two-dimensional plane, and the other class forming a ring around it.

We trained two neural networks and examined the plot of the resulting functions to characterize the level sets/decision boundaries.

In these plots, the decision boundary is visible as the white region between the blue and orange regions defining the two labels.

The first network has six two-dimensional hidden layers, the maximum number of layers allowed in the webapp.

As shown in Figure 3a , the decision boundary is an unbounded curve that extends beyond the region containing all the data points.

The ideal decision boundary between the two classes of points would be a (bounded) loop around the blue points in the middle, but Theorem 1 proves that such a network cannot approximate a function with such a level set.

A decision boundary such as the one shown in the Figure is as close as it can get.

The extra hidden layers allow the decision boundary to curve around and minimize the neck of the blue region, but they do not allow it to pinch off completely.

The second network has a single hidden layer of dimension three -one more than that of the input space.

As shown in Figure 3b , the decision boundary for the learned function is a loop that approximates the ideal decision boundary closely.

It comes from the three lines defined by the hidden nodes, which make a triangle that gets rounded off by the activation function.

Increasing the dimension of the hidden layer would make the decision boundary rounder, though in this case the model doesn't need the extra flexibility.

Note that this example generalizes to any dimension n, though without the ability to directly graph the results.

In other words, for any Euclidean input space of dimension n, a sigmoid neural network with one hidden layer of dimension n + 1 can define a function that cannot be approximated by any deep network with an arbitrary number of hidden layers of dimension at most n. In fact, this will be the case for any activation function that is bounded above or below, though we will not include the details of the argument here.

In this paper, we describe topological limitations on the types of functions that can be approximated by deep, skinny neural networks, independent of the number of hidden layers.

We prove the result using standard set theoretic topology, then present examples that visually demonstrate the result.

<|TLDR|>

@highlight

This paper proves that skinny neural networks cannot approximate certain functions, no matter how deep they are.