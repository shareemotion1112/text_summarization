We apply canonical forms of gradient complexes (barcodes) to explore neural networks loss surfaces.

We present an algorithm for calculations of the objective function's barcodes of minima.

Our experiments confirm two principal observations: (1) the barcodes of minima are located in a small lower part of the range of values of objective function and (2) increase of the neural network's depth brings down the minima's barcodes.

This has natural implications for the neural network learning and the ability to generalize.

The learning via finding minima of objective functions is the principal strategy underlying majority of learning algorithms.

For example, in Neural Network training, the objective function's input is model parameters (weights) and the objective function's output is the loss on training dataset.

The graph of the loss function, often called loss surface, typically has complex structure (e.g. see loss surface visualisations by Li et al. (2018) ): non-convexity, many local minima, flat regions, steep slopes.

These obstacles harm exploration of the loss surface and complicate searching for optimal network weights.

The optimization of modern neural networks is based on the gradient descent algorithm.

The global topological characteristics of the gradient vector field trajectories are captured by the Morse complex via decomposing the parameter space into cells of uniform flow, see Barannikov (1994) ; Le Roux et al. (2018) and references therein.

The invariants of Morse complex called "canonical forms"(or barcodes) constitute the fundamental summary of the topology of the gradient vector field flow.

The "canonical forms", or barcodes, in this context are decompositions of the change of topology of the sublevel sets of objective function into simple "birth-death" phenomena of topological feautures of different dimensions.

The calculation of the barcodes for different functions constitutes the essence of the topological data analysis.

The currently available software packages for the calculation of barcodes of functions, also called "sublevel persistence", are GUDHI, Dionysus, PHAT, and TDA package which incorporates all three previous packages B.T. Fasy et al. (2014) .

They are based on the algorithm, described in Barannikov (1994) , see also appendix and e.g. Bauer et al. (2014) and references therein.

This algorithm which has complexity of O(n 3 ).

These packages can currently handle calculations of barcodes for functions defined on a grid of up to 10 6 points, and in dimensions two and three.

Thus all current packages have the scalability issues.

We describe a new algorithm for computations of the barcodes of functions in lowest degree.

Our algorithm works with functions defined on randomly sampled or specifically chosen point clouds.

Point cloud based methods are known to work better than grid based methods in optimization related problems (Bergstra and Bengio (2012) ).

We also use the fact that the definition of the barcode of lowest degree can be reformulated in geometrical terms (see definition 1 in section 2).

The previously known algorithms were based on the more algebraic approach as in definition 3.

Our algorithm has complexity of O(n log(n)).

It was tested in dimensions up to 16 and with number of points of up to 10 8 .

In this work, we develop a methodology to describe the properties of the loss surface of the neural network via topological features of local minima.

We emphasize that the value of the objective function at the minimum can be viewed as only a part of its topological characteristic from the "canonical form" (barcode).

The second half can be described as the value of objective function at the index-one saddle, which can be naturally associated with each local minimum.

The difference between the values of objective function at the associated index-one saddle and at the local minimum is a topological invariant of the minimum.

For optimization algorithms this quantity measures, in particular, the obligatory penalty for moving from the given local minimum to a lower minimum.

The main contributions of the paper are as follows:

Applying the one-to-one correspondence between local minima and 1-saddles to exploration of loss surfaces.

For each local minimum p there is canonically defined 1-saddle q (see Section 2).

The 1-saddle associated with p can be described as follows.

The 1-saddle q is precisely the point where the connected component of the sublevel set Θ f ≤c = {θ ∈ Θ | f (θ) ≤ c} containing the minimum p merges with another connected component of the sublevel set whose minimum is lower.

This correspondence between the local minima and the 1-saddles, killing a connected component of Θ f ≤c , is one-to-one.

The segment [f (p), f (q)] is then the "canonical form" invariant attached to the minimum p.

The set of all such segments is the barcode ("canonical form") of minima invariant of f .

It is a robust topological invariant of objective function.

It is invariant in particular under the action of homeomorphisms of Θ. Full "canonical form" invariants give a concise summary of the topology of objective function and of the global structure of its gradient flow.

Algorithm for calculations of the barcodes (canonical invariants) of minima.

We describe an algorithm for calculation of the canonical invariants of minima.

The algorithm works with function's values on a a randomly sampled or specifically chosen set of points.

The local minima give birth to clusters of points in sublevel sets.

The algorithm works by looking at neighbors of each point with lower value of the function and deciding if this point belongs to the existing clusters, gives birth to a new cluster (minimum), or merges two or more clusters (index one saddle).

A variant of the algorithm has complexity of O(n log(n)), where n is the cardinality of the set of points.

Calculations confirming observations on behaviour of neural networks loss functions barcodes.

We calculate the canonical invariants (barcodes) of minima for small fully-connected neural networks of up to three hidden layers and verify that all segments of minima's barcode belong to a small lower part of the total range of loss function's values and that with the increase in the neural network depth the minima's barcodes descend lower.

The usefulness of our approach and algorithms is clearly not limited to the optimization problems.

Our algorithm permits really fast computation of the canonical form invariants (persistence barcodes) of many functions which were not accessible until now.

These sublevel persistence barcodes have been successfully applied in different discipline, to mention just a few: cognitive science (M. K. Chung and Kim (2009) ), cosmology (Sousbie et al. (2011) ), see e.g. Pun et al. (2018) and references therein.

Our viewpoint should also have applications in chemistry and material science where 1-saddle points on potential energy landscapes correspond to transition states and minima are stable states corresponding to different materials or protein foldings (see e.g. Dellago et al. (2003) , Oganov and Valle (2009) ).

The article is structured as follows.

First we describe three definitions of barcodes of minima.

After that our algorithm for their calculation is described.

In the last part we give examples of calculations, including the loss functions of simple neural nets.

The "canonical form" invariants (barcodes) give a concise summary of topological features of functions (see Barannikov (1994) , Le Roux et al. (2018) and references therein).

These invariants describe a decomposition of the change of topology of the function into the finite sum of "birth"-"death" of elementary features.

We propose to apply these invariants as a tool for exploring topology of loss surfaces.

In this work we concentrate on the part of these canonical form invariants, describing the "birth"-"death" phenomena of connected components of sublevel sets of the function.

However it should be stressed that this approach works similarly also for "almost minima", i.e. for the critical points (manifolds) of small indexes, which are often the terminal points of the optimization algorithms in very high dimensions.

We give three definitions of the "canonical form" invariants of minima.

The values of parameter c at which the topology of sublevel set

Let p be one of minima of f .

When c increases from f (p)− to f (p)+ , a new connected component of the set Θ f ≤c is born (see fig 1a, the connected components S 1 , S 2 , S 3 of sublevel set are born at the blue, green and red minima correspondingly.

If p is a minimum, which is not global, then, when c is increased, the connected component of Θ f ≤c born at p merges with a connected component born at a lower minimum.

Let q is the merging point where this happens.

The intersection of the set Θ f <f (q) with any small neighborhood of q has two connected components.

This is the index-one saddle q associated with p.

(a) "Death" of the connected component S3.

The connected component S3 of sublevel set merges with connected component S2 at red saddle, red saddle is associated with the red minimum.

(b) "Death" of the connected component S4.

The connected component S4 of sublevel set merges with connected component S1 at violet saddle, violet saddle is associated with the violet minimum (c) "Death" of the connected component S2.

The connected component S2 of sublevel set merges with connected component S1 at green saddle, green saddle is associated with the green minimum.

Figure 1: Merging of connected components of sublevel sets at saddles.

Note that the green saddle is associated with the green minimum which is separated by another minimum from the green saddle.

Also these two subsets of small neighborhood of q belong to two different connected components of the whole set Θ f <f (q) .

The 1-saddles of this type are called "+" ("plus") or "death" type.

The described correspondence between local minima and 1-saddles of this type is one-to-one.

In a similar way, the 1-saddle q associated with p can be described also as follows.

Proposition 2.1.

Consider various paths γ starting from the local minimum p and going to a lower minimum.

Let m γ ∈ Θ is the maximum of the restriction of f to such path γ.

Then 1-saddle q which is paired with the local minimum p is the minimum over the set of all such paths γ of the maxima m γ :

The correspondence in the opposite direction can be described analogously.

Let q is a 1-saddle point of such type that the two branches of the set Θ f ≤f (q)− near q belong to two different connected components of Θ f ≤f (q)− .

A new connected component of the set Θ f ≤c is formed when c decreases from f (q) + to f (q) − .

The restriction of f to each of the two connected components has its global minimum.

Proposition 2.2.

Given a 1-saddle q, the minimum p which is paired with q is the new minimum of f on the connected component of the set Θ f ≤c which is formed when c decreases from f (q) + to f (q) − .

The two branches of the set Θ f ≤f (q)− near q can also belong to the same connected components of this set.

Then such saddle is of "birth" type and it is naturally paired with index-two saddle of "death" type (see theorem 2.3).

Chain complex is the algebraic counterpart of intuitive idea representing complicated geometric objects as a decomposition into simple pieces.

It converts such a decomposition into a collection of vector spaces and linear maps.

A chain complex (C * , ∂ * ) is a sequence of finite-dimensional k-vector spaces and linear operators

The j−th homology of the chain complex (C * , ∂ * ) is the quotient

A chain complex C * is called R−filtered if C * is equipped with an increasing sequence of sub-

by a finite set of real numbers s 1 < s 2 < . . .

< s max .

Theorem 2.3. (Barannikov (1994) ) Any R−filtered chain complex C * can be brought by a linear transformation preserving the filtration to "canonical form", a canonically defined direct sum of R−filtered complexes of two types: one-dimensional complexes with trivial differential ∂ j (e i ) = 0 and two-dimensional complexes with trivial homology ∂ j (e i2 ) = e i1 .

The resulting canonical form is uniquely determined.

The full barcode is a visualization of the decomposition of an R−filtered complexes according to the theorem 2.3.

Each filtered 2-dimensional complex with trivial homology ∂ j (e i2 ) = e i1 , e i1 = F ≤s1 , e i1 , e i2 = F ≤s2 describes a topological feature in dimension j which is "born" at s 1 and which "dies" at s 2 .

It is represented by segment [s 1 , s 2 ] in the degree-j barcode.

And each filtered 1-dimensional complex with trivial differential, ∂ j e i = 0 , e i = F ≤r describes a topological feature in dimension j which is "born" at r and never "dies".

It is represented by the half-line [r, +∞[ in the degree-j barcode.

The proof of the theorem is given in Appendix.

Essentially, one can bring an R−filtered complex to the required canonical form by induction, starting from the lowest basis elements of degree one, in such a way that the manipulation of degree j basis elements does not destroy the canonical form in degree j − 1 and in lower filtration pieces in degree j.

Let f : Θ → R is smooth, or more generally, piece-wise smooth continuous function such that the sublevel sets Θ f ≤c = {θ ∈ Θ | f (θ) ≤ c} are compact.

One filtered complex naturally associated with function f and such that the subcomplexes F s C * compute the homology of sublevel sets Θ f ≤s is the gradient (Morse) complex, see e.g. Barannikov (1994) ; Le Peutrec et al. (2013) and references therein.

Without loss of generality the function f can be assumed smooth here, otherwise one can always replace f by its smoothing.

By adding a small perturbation such as a regularization term we can also assume that critical points of f are non-degenerate.

The generators of the gradient (Morse) complex correspond to the critical points of f .

The differential is defined by counting gradient trajectories between critical points when their number is finite.

The canonical form of the gradient (Morse) complex describes a decomposition of the gradient flow associated with f into standard simple pieces.

Let p be a minimum, which is not a global minimum.

Then the generator corresponding to p represents trivial homology class in the canonical form, since the homology class of its connected component is already represented by the global minimum.

Then p is the lower generator of a two-dimensional complex with trivial homology in the canonical form.

I.e. p is paired with an index-one saddle q in the canonical form.

The segment [f (p), f (q)] is then the canonical invariant (barcode) corresponding to the minimum p.

The full canonical form of the gradient (Morse) complex of all indexes is a summary of global structure of the objective function's gradient flow.

The total number of different topological features in sublevel sets Θ f ≤c of the objective function can be read immediately from the barcode.

Namely the number of intersections of horizontal line at level c with segments in the index j barcode gives the number of independent topological features of dimension j in Θ f ≤c .

The description of the barcode of minima on manifold Θ with nonempty boundary ∂Θ is modified in the following way.

A connected component can be also born at a local minimum of restriction of f to the boundary f | ∂Θ , if gradf is pointed inside manifold Θ. The merging of two connected components can also happen at an index-one critical point of f | ∂Θ , if gradf is pointed inside Θ.

In this section we describe the developed algorithm for calculation of the canonical form invariants of local minima.

The computation exploits the first definition of barcodes (see Section 2), which is based on the evolution on the connected components of the sublevel sets.

To analyse the surface of the given function f : Θ → R, we first build its approximation by finite graph-based construction.

To do this, we consider a random subset of points {θ 1 , . . .

, θ N } ∈ Θ and build a graph with these points as vertices.

The edges connect close points.

Thus, for every vertex θ n , by comparing f (θ n ) with f (θ n ) for neighbors θ n of θ n , we are able to understand the local topology near the point θ n .

At the same time, connected componenets of sublevel sets Θ f ≤c = {θ ∈ Θ | f (θ) ≤ c} will naturally correspond to connected components of the subgraph on point θ n , such that f (θ n ) ≤ c.

Two technical details here are the choice of points θ n and the definition of closeness, i.e. when to connect points by an edge.

In our experiments, we sample points uniformly from some rectangular box of interest.

To add edges, we compute the oriented k-Nearest Neighbor Graph on the given points and then drop the orientation of edges.

Thus, every node in the obtained graph has a degree in [k, 2k] .

In all our experiments we use k = 2D, where D is the dimension of f 's input.

Next we describe our algorithm, which computes barcodes of a function from its graph-based approximation described above.

The key idea is to monitor the evolution of the connected components of the sublevel sets of the graph, i.e. sets Θ c = {θ n | f (θ n ) ≤ c} for increasing c.

For simplicity we assume that points θ are ordered w.r.t.

the value of function f , i.e. for n < n we have f (θ n ) < f (θ n ).

In this case we are interested in the evolution of connected components throughout the process sequential adding of vertices θ 1 , θ 2 , . . .

, θ N to graph, starting from an empty graph.

We denote the subgraph on vertices θ 1 , . . .

, θ n by Θ n .

When we add new vertex θ n+1 to θ n , there are three possibilities for connected componenets to evolve:

1.

Vertex θ n+1 has zero degree in Θ n+1 .

This means that θ n+1 is a local minimum of f and it forms a new connected component in the sublevel set.

2.

All the neighbors of θ n+1 in Θ n+1 belong to one connected component in Θ n .

3.

All the neighbors of θ n+1 in Θ n+1 belong to ≥ 2 connected components s 1 , s 2 , . . .

, s K ⊂ Θ n .

Thus, all these components will form a single connected component in Θ n+1 .

Algorithm 1: Barcodes of minima computation for function on a graph.

Input : Connected undirected graph G = (V, E); function f on graph vertices.

Output : Barcodes: a list of "birth"-"death" pairs.

In the third case, according to definition 1 of Section 2 the point θ n+1 is a 1-saddle point.

Thus, one of the components s k swallows the rest.

This is the component which has the lowest minimal value.

For other components, 2 this gives their barcodes: for s k the birth-death pair is min

We summarize the procedure in the following algorithm 1.

Note that we assume that the input graph is connected (otherwise the algorithm can be run on separate connected components).

In the practical implementation of the algorithm, we precompute the values of function f at all the vertices of G. Besides that, we use the disjoint set data structure to store and merge connected components during the process.

We also keep and update the global minima in each component.

We did not include these tricks into the algorithm's pseuso-code in order to keep it simple.

The resulting complexity of the algorithm is O(N log N ) in the number of points.

Here it is important to note that the procedure of graph creation may be itself time-consuming.

In our case, the most time consuming operation is nearest neighbor search.

In our code, we used efficient HNSW Algorithm for aproximate NN search by Malkov and Yashunin (2018) .

In this section we apply our algorithm to describing the surfaces of functions.

In Subsection 4.1 we apply the algorithm to toy visual examples.

In Subsection 4.2 we apply the algorithm to analyse the loss surfaces of small neural networks.

In this subsection we demonstrate the application of the algorithm to simple toy functions f : R D → R. For D ∈ {1, 2} we consider three following functions: 1.

Polynomial of a single variable of degree 4 with 2 local minima (see Fig. 2a ):

2.

Camel function with 3 humps, i.e. 3 local minima (see Fig. 2b ):

3.

Camel function with 6 humps, i.e. 6 local minima (see Fig. 2c ):

Function plots with their corresponding barcodes of minima are given in Figure 2 .

The barcode of the global minimum is represented by the dashed half-line which goes to infinity.

In this section we compute and analyse barcodes of small fully connected neural networks with up to three hidden layers.

For several architectures of the neural networks many results on the loss surface and its local minima are known (see e.g. Kawaguchi (2016) Gori and Tesi (1992) and references therein).

Different geometrical and topological properties of loss surfaces were studied in Cao et al. (2017); Yi et al. (2019); Chaudhari et al. (2017) ; Dinh et al. (2017) .

There is no ground truth on how should the best loss surface of a neural network looks like.

Nevertheless, there exists many common opinions on this topic.

First of all, from practical optimization point of view, the desired local (or global) minima should be easily reached via basic training methods such as Stochastic Gradient Descent, see Ruder (2016) .

Usually this requires more-or-less stable slopes of the surface to prevent instabilities such as gradient explosions or vanishing gradients.

Secondly, the value of obtained minimum is typically desired to be close to global, i.e. attain smallest training error.

Thirdly, from the generalization point of view, such minima are required to provide small loss on the testing set.

Although in general it is assumed that the good local optimum is the one that is flat, some recent development provide completely contrary arguments and examples, e.g. sharp minima that generalize well.

Besides the optimization of the weights for a given architecture, neural network training implies also a choice of the architecture of the network, as well as the loss function to be used for training.

In fact, it is the choice of the architecture and the loss function that determines the shape of the loss surface.

Thus, proper selection of the network's architecture may simplify the loss surface and lead to potential improvements in the weight optimization procedure.

We have analyzed very tiny neural networks.

However our method permits full exploration of the loss surface as opposed to stochastical exploration of higher-dimensional loss surfaces.

Let us emphasize that even from practical point of view it is important to understand first the behavior of barcodes in simplest examples where all hyper-parameters optimization schemes can be easily turned off.

For every analysed neural network the objective function is its mean squared error for predicting (randomly selected) function g : [−π, π] → R given by

plus l 2 −regularization.

The error is computed for prediction on uniformly distributed inputs x ∈ {−π + 2π 100 k | k = 0, 1, . . .

, 100}. The neural networks considered were fully connected one-hidden layer with 2, 3 and 4 neurons, two-hidden layers with 2x2, 3x2 and 3x3 neurons, and three hidden layers with 2x2x2 and 3x2x2 neurons.

We have calculated the barcodes of the loss functions on the hyper-cubical sets Θ which were chosen based on the typical range of parameters of minima.

The results are as shown in Figure  3 .

We summarize our findings into two main observations:

In this work we have introduced a methodology for analysing the plots of functions, in particular, loss surfaces of neural networks.

The methodology is based on computing topological invariants called canonical forms or barcodes.

To compute barcodes we used a graph-based construction which approximates the function plot.

Then we apply the algorithm we developed to compute the barcodes of minima on the graph.

Our experimental results of computing barcodes for small neural networks lead to two principal observations.

First all barcodes sit in a tiny lower part of the total function's range.

Secondly, with increase of the depth of neural network the barcodes descend lower.

From the practical point of view, this means that gradient descent optimization cannot stuck in high local minima, and it is also not difficult to get from one local minimum to another (with smaller value) during learning.

The method we developed has several further research directions.

Although we tested the method on small neural networks, it is possible to apply it to large-scale modern neural networks such as convolutional networks (i.e. ResNet, VGG, AlexNet, U-Net, see Alom et al. (2018) ) for imageprocessing based tasks.

However, in this case the graph-based approximation we use requires wise choice of representative graph vertices, which is a hardcore in high-dimensional spaces (dense filling of area by points is computationally intractable).

Another direction is to study the connections between the barcode of local minima and the generalization properties of given minimum and of neural network.

There are clearly also connections, deserving further investigation, between the barcodes of minima and results concerning the rate of convergency during learning of neural networks.

<|TLDR|>

@highlight

We apply canonical forms of gradient complexes (barcodes) to explore neural networks loss surfaces.