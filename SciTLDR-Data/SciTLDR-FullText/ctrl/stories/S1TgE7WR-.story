Most existing neural networks for learning graphs deal with the issue of permutation invariance by conceiving of the network as a message passing scheme, where each node sums the feature vectors coming from its neighbors.

We argue that this imposes a limitation on their representation power, and instead propose a new general architecture for representing objects consisting of a hierarchy of parts, which we call Covariant Compositional Networks (CCNs).

Here covariance means that the activation of each neuron must transform in a specific way under permutations, similarly to steerability in CNNs.

We achieve covariance by making each activation transform according to a tensor representation of the permutation group, and derive the corresponding tensor aggregation rules that each neuron must implement.

Experiments show that CCNs can outperform competing methods on some standard graph learning benchmarks.

Learning on graphs has a long history in the kernels literature, including approaches based on random walks BID14 BID1 BID11 , counting subgraphs BID35 , spectral ideas BID41 , label propagation schemes with hashing BID36 Neumann et al., 2016) , and even algebraic ideas BID21 .

Many of these papers address moderate size problems in chemo-and bioinformatics, and the way they represent graphs is essentially fixed.

Recently, with the advent of deep learning and much larger datasets, a sequence of neural network based approaches have appeared to address the same problem, starting with BID33 .

In contrast to the kernels framework, neural networks effectively integrate the classification or regression problem at hand with learning the graph representation itself, in a single, end-to-end system.

In the last few years, there has been a veritable explosion in research activity in this area.

Some of the proposed graph learning architectures BID8 BID18 BID29 directly seek inspiration from the type of classical CNNs that are used for image recognition BID25 Krizhevsky et al., 2012) .

These methods involve first fixing a vertex ordering, then moving a filter across vertices while doing some computation as a function of the local neighborhood to generate a representation.

This process is then repeated multiple times like in classical CNNs to build a deep graph representation.

Other notable works on graph neural networks include BID26 BID34 BID0 BID20 .

Very recently, BID15 showed that many of these approaches can be seen to be specific instances of a general message passing formalism, and coined the term message passing neural networks (MPNNs) to refer to them collectively.

While MPNNs have been very successful in applications and are an active field of research, they differ from classical CNNs in a fundamental way: the internal feature representations in CNNs are equivariant to such transformations of the inputs as translation and rotations BID4 , the internal representations in MPNNs are fully invariant.

This is a direct result of the fact that MPNNs deal with the permutation invariance issue in graphs simply by summing the messages coming from each neighbor.

In this paper we argue that this is a serious limitation that restricts the representation power of MPNNs.

MPNNs are ultimately compositional (part-based) models, that build up the representation of the graph from the representations of a hierarchy of subgraphs.

To address the covariance issue, we study the covariance behavior of such networks in general, introducing a new general class of neural network architectures, which we call compositional networks (comp-nets).

One advantage of this generalization is that instead of focusing attention on the mechanics of how information propagates from node to node, it emphasizes the connection to convolutional networks, in particular, it shows that what is missing from MPNNs is essentially the analog of steerability.

Steerability implies that the activations (feature vectors) at a given neuron must transform according to a specific representation (in the algebraic sense) of the symmetry group of its receptive field, in our case, the group of permutations, S m .

In this paper we only consider the defining representation and its tensor products, leading to first, second, third etc.

order tensor activations.

We derive the general form of covariant tensor propagation in comp-nets, and find that each "channel" in the network corresponds to a specific way of contracting a higher order tensor to a lower order one.

Note that here by tensor activations we mean not just that each activation is expressed as a multidimensional array of numbers (as the word is usually used in the neural networks literature), but also that it transforms in a specific way under permutations, which is a more stringent criterion.

The parameters of our covariant comp-nets are the entries of the mixing matrix that prescribe how these channels communicate with each other at each node.

Our experiments show that this new architecture can beat scalar message passing neural networks on several standard datasets.

Graph learning encompasses a broad range of problems where the inputs are graphs and the outputs are class labels (classification), real valued quantities (regression) or more general, possibly combinatorial, objects.

In the standard supervised learning setting this means that the training set consists of m input/output pairs {(G 1 , y 1 ), (G 2 , y 2 ), . . .

, (G m , y m )}, where each G i is a graph and y i is the corresponding label, and the goal is to learn a function h : G → y that will successfully predict the labels of further graphs that were not in the training set.

By way of fixing our notation, in the following we assume the each graph G is a pair (V, E), where V is the vertex set of G and E ⊆

V × V is its edge set.

For simplicity, we assume that V = {1, 2, . . .

, n}. We also assume that G has no self-loops ((i, i) ∈ E for any i ∈ V ) and that G is symmetric, i.e., (i, j) ∈ E ⇒ (j, i) ∈ E 1 .

We will, however, allow each edge (i, j) to have a corresponding weight w i,j , and each vertex i to have a corresponding feature vector (vertex label) l i ∈ R d .

The latter, in particular, is important in many scientific applications, where l i might encode, for example, what type of atom occupies a particular site in a molecule, or the identity of a protein in a biochemical interaction network.

All the topological information about G can be summarized in an adjacency matrix A ∈ R n×n , where A i,j = w i,j if i and j are connected by an edge, and otherwise A i,j = 0.

When dealing with labeled graphs, we also have to provide (l 1 , . . .

, l n ) to fully specify G.One of the most fascinating aspects of graphs, but also what makes graph learning challenging, is that they involve structure at multiple different scales.

In the case when G is the graph of a protein, for example, an ideal graph learning algorithm would represent G in a manner that simultaneously captures structure at the level of individual atoms, functional groups, interactions between functional groups, subunits of the protein, and the protein's overall shape.

The other major requirement for graph learning algorithms relates to the fact that the usual ways to store and present graphs to learning algorithms have a critical spurious symmetry: If we were to 1 2 3 4 5 6 3 1 2 6 4 5 Figure 1 : (a) A small graph G with 6 vertices and its adjacency matrix.

(b) An alternative form G of the same graph, derived from G by renumbering the vertices by a permutation σ : {1, 2, . . .

, 6} → {1, 2, . . .

, 6}. The adjacency matrices of G and G are different, but topologically they represent the same graph.

Therefore, we expect the feature map φ to satisfy φ(G) = φ(G ).permute the vertices of G by any permutation σ : {1, 2, . . .

, n} → {1, 2, . . .

, n} (in other words, rename vertex 1 as σ(1), vertex 2 as σ(2), etc.), then the adjacency matrix would change to DISPLAYFORM0 and simultaneously the vertex labels would change to (l 1 , . . .

, l n ), where l i = l σ −1 (i) .

However, G = (A , l 1 , . . . , l n ) would still represent exactly the same graph as G = (A, l 1 , . . .

, l n ).

In particular, (a) in training, whether G or G is presented to the algorithm must not make a difference to the final hypothesis h that it returns, (b) h itself must satisfy h(G) = h(G ) for any labeled graph and its permuted variant.

Most learning algorithms for combinatorial objects hinge on some sort of fixed or learned internal representation of data, called the feature map, which, in our case we denote φ(G).

The set of all n!

possible permutations of {1, 2, . . .

, n} forms a group called the symmetric group of order n, denoted S n .

The permutation invariance criterion can then be formulated as follows (Figure 1 ).

Definition 1.

Let A be a graph learning algorithm that uses a feature map G → φ(G).

We say that the feature map φ (and consequently the algorithm A) is permutation invariant if, given any n ∈ N, any n vertex labeled graph G = (A, l 1 , . . .

, l n ), and any permutation σ ∈ S n , letting G = (A , l 1 , . . .

, l n ), where A i,j = A σ −1 (i),σ −1 (j) and l i = l σ −1 (i) , we have that φ(G) = φ(G ).Capturing multiscale structure and respecting permutation invariance are the two the key constraints around which most of the graph learning literature revolves.

In kernel based learning, for example, invariant kernels have been constructed by counting random walks BID14 , matching eigenvalues of the graph Laplacian BID41 and using algebraic ideas BID21 .

Many recent graph learning papers, whether or not they make this explicit, employ a compositional approach to modeling graphs, building up the representation of G from representations of subgraphs.

At a conceptual level, this is similar to part-based modeling, which has a long history in machine learning BID12 BID30 BID40 BID9 BID45 BID10 .

In this section we introduce a general, abstract architecture called compositional networks (comp-nets) for representing complex objects as a combination of their parts, and show that several exisiting graph neural networks can be seen as special cases of this framework.

Definition 2.

Let G be a compound object with n elementary parts (atoms) E = {e 1 , . . .

, e n }.

A composition scheme for G is a directed acyclic graph (DAG) M in which each node n i is associated with some subset P i of E (these subsets are called the parts of G) in such a way that 1.

If n i is a leaf node, then P i contains a single atom e ξ(i) 2 .

2.

M has a unique root node n r , which corresponds to the entire set {e 1 , . . .

, e n }.

3.

For any two nodes n i and n j , if n i is a descendant of n j , then P i ⊂ P j .We define a compositional network as a composition scheme in which each node n i also carries a feature vector f i that provides a representation of the corresponding part (Figure 2 ).

When we want DISPLAYFORM0 n 10 {e1, e2, e4} n r {e1, e2, e3, e4} DISPLAYFORM1 Figure 2: (a) A composition scheme for an object G is a DAG in which the leaves correspond to atoms, the internal nodes correspond to sets of atoms, and the root corresponds to the entire object.(b) A compositional network is a composition scheme in which each node n i also carries a feature vector f i .

The feature vector at n i is computed from the feature vectors of the children of n i .

Figure 3 : A minimal requirement for composition schemes is that they be invariant to permutation, i.e. that if the numbering of the atoms is changed by a permutation σ, then we must get an isomorphic DAG.

Any node in the new DAG that corresponds to {e i1 , . . .

, e i k } must have a corrresponding node in the old DAG corresponding to {e σ −1 (i1) , . . .

, e σ −1 (i k ) }.to emphasize the connection to more classical neural architectures, we will refer to n i as the i'th neuron, P i as its receptive field 3 , and f i as its activation.

Definition 3.

Let G be a compound object in which each atom e i carries a label l i , and M a composition scheme for G. The corresponding compositional network N is a DAG with the same structure as M in which each node n i also has an associated feature vector f i such that 1.

If n i is a leaf node, then f i = l ξ(i) .

2.

If n i is a non-leaf node, and its children are n c1 , . . .

, n c k , then f i = Φ(f c1 , f c2 , . . .

, f c k ) for some aggregation function Φ. (Note: in general, Φ can also depend on the relationships between the subparts, but for now, to keep the discussion as simple as possible, we ignore this possibility.)

The representation φ(G) afforded by the comp-net is given by the feature vector f r of the root.

Note that while, for the sake of concreteness, we call the f i 's "feature vectors", there is no reason a priori why they need to be vectors rather than some other type of mathematical object.

In fact, in the second half of the paper we make a point of treating the f i 's as tensors, because that is what will make it the easiest to describe the specific way that they transform with respect to permutations.

In compositional networks for graphs, the atoms will usually be the vertices, and the P i parts will correspond to clusters of nodes or neighborhoods of given radii.

Comp-nets are particularly attractive in this domain because they can combine information from the graph at different scales.

The comp-net formalism also suggests a natural way to satisfy the permutation invariance criterion of Definition 1.Definition 4.

Let M be the composition scheme of an object G with n atoms and M the composition scheme of another object that is equivalent in structure to G, except that its atoms have been permuted by some permutation σ ∈ S n (e i = e σ −1 (i) and i = σ −1 (i) ).

We say that M (more precisely, the algorithm generating M) is permutation invariant if there is a bijection ψ : M → M taking each n a ∈ M to some n b ∈ M such that if P a = {e i1 , . . .

, e i k }, then P b = {e σ(i1) , . . . , e σ(i k ) }.

Proposition 1.

Let φ(G) be the output of a comp-net based on a composition scheme M. Assume 1.

M is permutation invariant in the sense of Definition 4.

2.

The aggregation function Φ(f c1 , f c2 , . . . , f c k ) used to compute the feature vector of each node from the feature vectors of its children is invariant to the permutations of its arguments.

Then the overall representation φ(G) is invariant to permutations of the atoms.

In particular, if G is a graph and the atoms are its vertices, then φ is a permutation invariant graph representation.

Graph learning is not the only domain where invariance and multiscale structure are important: the most commonly cited reasons for the success of convolutional neural networks (CNNs) in image tasks is their ability to address exactly these two criteria in the vision context.

Furthermore, each neuron n i in a CNN aggregates information from a small set of neurons from the previous layer, therefore its receptive field, corresponding to P i , is the union of the receptive fields of its "children", so we have a hierarchical structure very similar to that described in the previous section.

In this sense, CNNs are a specific kind of compositional network, where the atoms are pixels.

This connection has inspired several authors to frame graph learning as a generalization of convolutional nets to the graph domain BID2 Henaff et al., 2015; BID8 BID7 BID20 .

While in mathematics convolution has a fairly specific meaning that is side-stepped by this analogy, the CNN analogy does suggest that a natural way to define the Φ aggregation functions is to let Φ(f c1 , f c2 , . . .

, f c k ) be a linear function of f c1 , f c2 , . . .

, f c k followed by a pointwise nonlinearity, such as a ReLU operation.

To define a comp-net for graphs we also need to specify the composition scheme M. Many algorithms define M in layers, where each layer (except the last) has one node for each vertex of G: M1.

In layer = 0 each node n 0 i represents the single vertex P 0 i = {i}. M2.

In layers = 1, 2, . . .

, L, node n i is connected to all nodes from the previous level that are neighbors of i in G, i.e., the children of n i are ch(n i ) = {n DISPLAYFORM0 where N (i) denotes the set of neighbors of i in G. Therefore, DISPLAYFORM1

In layer L+1 we have a single node n r that represents the entire graph and collects information from all nodes at level L. Since this construction only depends on topological information about G, the resulting composition scheme is guaranteed to be permutation invariant in the sense of Definition 4.A further important consequence of this way of defining M is that the resulting comp-net can be equivalently interpreted as label propagation algorithm, where in each round = 1, 2, . . .

, L, each vertex aggregates information from its neighbors and then updates its own label.

for each vertex i f DISPLAYFORM0 Many authors choose to describe graph neural networks exclusively in terms of label propagation, without mentioning the compositional aspect of the model.

BID15 call this general approach message passing neural networks, and point out that a range of different graph learning architectures are special cases of it.

More broadly, the classic Weisfeiler-Lehman test of isomorphism also follows the same logic BID43 BID32 BID3 , and so does the related Weisfeiler-Lehman kernel, arguably the most successful kernel-based approach to graph learning BID36 .

Note also that in label propagation or message passing algorithms there is a clear notion of the source domain of vertex i at round , as the set of vertices that can influence f i , and this corresponds exactly to the receptive field P i of "neuron" n i in the comp-net picture.

The following proposition is immediate from the form of Algorithm 1 and reassures us that message passing neural networks, as special cases of comp-nets, do indeed produce permutation invariant representations of graphs.

Proposition 2.

Any label propagation scheme in which the aggregation function Φ is invariant to the permutations of its arguments is invariant to permutations in the sense of Definition 1.In the next section we argue that invariant message passing networks are limited in their representation power, however, and describe a generalization via comp-nets that overcomes some of these limitations.

One of the messages of the present paper is that invariant message passing algorithms, of the form described in the previous section, are not the most general possible compositional models for producing permutation invariant representations of graphs (or of compound objects, in general).Once again, an analogy with image recognition is helpful.

Classical CNNs face two types of basic image transformations: translations and rotations.

With respect to translations (barring pooling, edge effects and other complications), CNNs behave in a quasi-invariant way, in the sense that if the input image is translated by any integer amount (t x , t y ), the activations in each layer = 1, 2, . . .

L translate the same way: the activation of any neuron n i,j is simply transferred to neuron n i+t1,j+t2 , i.e., f i+t1,j+t2 = f i,j .

This is the simplest manifestation of a well studied property of CNNs called equivariance BID4 Worrall et al., 2017) .With respect to rotations, however, the situation is more complicated: if we rotate the input image by, e.g., 90 degrees, not only will the part of the image that fell in the receptive field of a particular neuron n i,j move to the receptive field of a different neuron n j,−i , but the orientation of the receptive field will also change ( FIG0 .

Consequently, features which were, for example, previously picked up by horizontal filters will now be picked up by vertical filters.

Therefore, in general, f j,−i = f i,j .

It can be shown that one cannot construct a CNN for images that behaves in a quasi-invariant way with respect to both translations and rotations unless every filter is directionless.

It is, however, possible to construct a CNN in which the activations transform in a predictable and reversible way, in particular, f j,−i = R(f i,j ) for some fixed invertible function R. This phenomenon is called steerability, and has a significant literature in both classical signal processing BID13 BID37 BID31 BID38 BID27 and the neural networks field BID5 .

t 2 ) , what used to fall in the receptive field of neuron n i,j is moved to the receptive field of n i+t1,j+t2 .

Therefore, the activations transform in the very simple way f i+t1,j+t2 = f i,j .

In contrast, rotations not only move the receptive fields around, but also permute the neurons in the receptive field internally, therefore, in general, f j,−i = f i,j .

The right hand figure shows that if the CNN has a horizontal filter (blue) and a vertical one (red) then their activations are exchanged by a 90 degree rotation.

In steerable CNNs, if (i, j) → (i , j ), then f i ,j = R(f i,j ) for some fixed linear function of the rotation.

The situation in compositional networks is similar.

The comp-net and message passing architectures that we have examined so far, by virtue of the aggregation function being symmetric in its arguments, are all quasi-invariant (with respect to permutations) in the following sense.

Definition 5.

Let G be a compound object of n parts and G an equivalent object in which the atoms have been permuted by some permutation σ.

Let N be a comp-net for G based on an invariant composition scheme, and N be the corresponding network for G .

We say that N is quasi-invariant if for any n i ∈ N , letting n j be the corresponding node in N , f i = f j for any σ ∈ S n Quasi-invariance in comp-nets is equivalent to the assertion that the activation f i at any given node must only depend on P i = {e j1 , . . .

, e j k } as a set, and not on the internal ordering of the atoms e j1 , . . . , e j k making up the receptive field.

At first sight this seems desirable, since it is exactly what we expect from the overall representation φ(G).

On closer examination, however, we realize that this property is potentially problematic, since it means that n i has lost all information about which vertex in its receptive field has contributed what to the aggregate information f i .

In the CNN analogy, we can say that we have lost information about the orientation of the receptive field.

In particular, if, further upstream, f i is combined with some other feature vector f j from a node with an overlapping receptive field, the aggregation process has no way of taking into account which parts of the information in f i and f j come from shared vertices and which parts do not ( Figure 5 ).The solution is to upgrade the P i receptive fields to be ordered sets, and explicitly establish how f i co-varies with the internal ordering of the receptive fields.

To emphasize that henceforth the P i sets are ordered, we will use parentheses rather than braces to denote their content.

Definition 6.

Let G, G , N and N be as in Definition 5.

Let n i be any node of N and n j the corresponding node of N .

Assume that P i = (e p1 , . . . , e pm ) while P j = (e q1 , . . . , e qm ), and let π ∈ S m be the permutation that aligns the orderings of the two receptive fields, i.e., for which e q π(a) = e pa .

We say that N is covariant to permutations if for any π, there is a corresponding function R π such that f j = R π (f i ).

The form of covariance prescribed by Definition 6 is very general.

To make it more specific, in line with the classical literature on steerable representations, we make the assumption that the {f → R π (f )} π∈Sm maps are linear, and by abuse of notation, from now on simply treat them as matrices (with R π (f ) = R π f ).

The linearity assumption automatically implies that {R π } π∈Sm is a representation of S m in the group theoretic sense of the word (for the definition of group representations, see the Appendix) 4 .Proposition 3.

If for any π ∈ S m , the f → R π (f ) map appearing in Definition 6 is linear, then the corresponding {R π } π∈Sm matrices form a representation of S m .

Figure 5: Top left: At level = 1 n 3 aggregates information from {n 4 , n 5 } and n 2 aggregates information {n 5 , n 6 }.

At = 2, n 1 collects this summary information from n 3 and n 2 .

Bottom left: This graph is not isomorphic to the top one, but the activations of n 3 and n 2 at = 1 will be identical.

Therefore, at = 2, n 1 will get the same inputs from its neighbors, irrespective of whether or not n 5 and n 7 are the same node or not.

Right: Aggregation at different levels.

For keeping the figure legible only the neighborhood around one node in higher levels is marked.

The representation theory of symmetric groups is a rich subject that goes beyond the scope of the present paper (Sagan, 2001) .

However, there is one particular representation of S m that is likely familiar even to non-algebraists, the so-called defining representation, given by the P π ∈ R n×n permutation matrices DISPLAYFORM0 It is easy to verify that P π2π1 = P π2 P π1 for any π 1 ,π 2 ∈ S m , so {P π }

π∈Sm is indeed a representation of S m .

If the transformation rules of the f i activations in a given comp-net are dictated by this representation, then each f i must necessarily be a |P i | dimensional vector, and intuitively each component of f i carries information related to one specific atom in the receptive field, or the interaction of that specific atom with all the others.

We call this case first order permutation covariance.

Definition 7.

We say that n i is a first order covariant node in a comp-net if under the permutation of its receptive field P i by any π ∈ S |Pi| , its activation trasforms as f i → P π f i .

It is easy to verify that given any representation (R g ) g∈G of a group G, the matrices (R g ⊗ R g ) g∈G also furnish a representation of G. Thus, one step up in the hierarchy from P π -covariant comp-nets are P π ⊗ P π -covariant comp-nets, where the f i feature vectors are now |P i | 2 dimensional vectors that transform under permutations of the internal ordering by π as DISPLAYFORM0 If we reshape f i into a matrix F i ∈ R |Pi|×|Pi| , then the action DISPLAYFORM1 is equivalent to P π ⊗P π acting on f i .

In the following, we will prefer this more intuitive matrix view, since it clearly expresses that feature vectors that transform this way express relationships between the different constituents of the receptive field.

Note, in particular, that if we define A↓ Pi as the restriction of the adjacency matrix to P i (i.e., if P i = (e p1 , . . . , e pm ) then [A↓ Pi ] a,b = A pa,p b ), then A↓ Pi transforms exactly as F i does in the equation above.

Definition 8.

We say that n i is a second order covariant node in a comp-net if under the permutation of its receptive field P i by any π ∈ S |Pi| , its activation transforms as F i → P π F i P π .

Taking the pattern further lets us consider third, fourth, and general, k'th order nodes in our compnet, in which the activations are k'th order tensors, transforming under permutations as DISPLAYFORM0 In the more compact, so called Einstein notation 5 , DISPLAYFORM1 In general, we will call any quantity which transforms according to this equation a k'th order Ptensor.

Note that this notion of tensors is distinct from the common usage of the term in neural networks, and more similar to how the word is used in Physics, because it not only implies that F i is a quanity representable by an m × m × . . .

× m array of numbers, but also that F i transforms in a specific way.

Since scalars, vectors and matrices can be considered as 0 th , 1 st and 2 nd order tensors, respectively, the following definition covers Definitions 5, 7 and 8 as special cases (with quasi-invariance being equivalent to zeroth order equivariance).

To unify notation and terminology, regardless of the dimensionality, in the following we will always talk about feature tensors rather than feature vectors, and denote the activations with F i rather than f i , as we did in the first half of the paper.

Definition 9.

We say that n i is a k'th order covariant node in a comp-net if the corresponding activation F i is a k'th order P -tensor, i.e., it transforms under permutations of P i according to (1), or the activation is a sequence of c separate P -tensors F

The previous sections prescribed how activations must transform in comp-nets of different orders, but did not explain how this can be assured, and what it entails for the Φ aggregation functions.

Fortunately, tensor arithmetic provides a compact framework for deriving the general form of these operations.

Recall the four basic operations that can be applied to tensors 6 : 1.

The tensor product of A ∈ T k with B ∈ T p yields a tensor C = A ⊗ B ∈ T p+k where DISPLAYFORM0 2.

The elementwise product of A ∈ T k with B ∈ T p along dimensions (a 1 , a 2 , . . . , a p ) yields a tensor C = A (a1,...,ap) B ∈ T k where DISPLAYFORM1 3.

The projection (summation) of A ∈ T k along dimensions {a 1 , a 2 , . . . , a p } yields a tensor C = A↓ a1,...,ap ∈ T k−p with DISPLAYFORM2 where we assume that i a1 , . . .

, i ap have been removed from amongst the indices of C. 4.

The contraction of A ∈ T k along the pair of dimensions {a, b} (assuming a < b) yields a k − 2 order tensor DISPLAYFORM3 The Einstein convention is that if, in a given tensor expression the same index appears twice, once "upstairs" and once "downstairs", then it is summed over.

For example, the matrix/vector product y = Ax would be written yi = A j i xj 6 Here and in the following T k will denote the class of k'th order tensors (k dimensional tensors), regardless of their transformation properties.

where again we assume that i a and i b have been removed from amongst the indices of C. Using Einstein notation this can be written much more compactly as DISPLAYFORM4 where δ ia,i b is the diagonal tensor with δ i,j = 1 if i = j and 0 otherwise.

In a somewhat unorthodox fashion, we also generalize contractions to (combinations of) larger sets of indices {{a Note that this subsumes projections, since it allows us to write A↓ a1,...,ap in the slightly unusual looking form DISPLAYFORM5 The following proposition shows that, remarkably, all of the above operations (as well as taking linear conbinations) preserve the way that P -tensors behave under permutations and thus they can be freely "mixed and matched" within Φ.Proposition 4.

Assume that A and B are k'th and p'th order P -tensors, respectively.

Then 1.

A ⊗ B is a k + p'th order P -tensor.

2.

A (a1,...,ap) B is a k'th order P -tensor.

3.

A↓ a1,...,ap is a k − p'th order P -tensor.

DISPLAYFORM6 pq is a k − j p j 'th order P -tensor.

In addition, if A 1 , . . .

, A u are k'th order P -tensors and α 1 , . . .

, α u are scalars, then j α j A j is a k'th order P -tensor.

The more challenging part of constructing the aggregation scheme for comp-nets is establishing how to relate P -tensors at different nodes.

The following two propositions answer this question.

Proposition 5.

Assume that node n a is a descendant of node n b in a comp-net N , P a = (e p1 , . . . , e pm ) and P b = (e q1 , . . .

, e q m ) are the corresponding ordered receptive fields (note that this implies that, as sets, P a ⊆ P b ), and χ a→b ∈ R m×m is an indicator matrix defined DISPLAYFORM7 Assume that F is a k'th order P -tensor with respect to permutations of (e p1 , . . .

, e pm ).

Then, dropping the a→b superscript for clarity, DISPLAYFORM8 is a k'th order P -tensor with respect to permutations of (e q1 , . . .

, e q m ).Equation 2 tells us that when node n b aggregates P -tensors from its children, it first has to "promote" them to being P -tensors with respect to the contents of its own receptive field by contracting along each of their dimensions with the appropriate χ a→b matrix.

This is a critical element in comp-nets to guarantee covariance.

Proposition 6.

Let n c1 , . . .

, n cs be the children of n t in a message passing type comp-net with corresponding k'th order tensor activations F c1 , . . .

, F cs .

Let DISPLAYFORM9 be the promotions of these activations to P -tensors of n t .

Assume that P t = (e p1 , . . . , e pm ).

Now let F be a k + 1'th order object in which the j'th slice is F pj if n pj is one of the children of n t , i.e., DISPLAYFORM10 and zero otherwise.

Then F is a k + 1'th order P -tensor of n t .Finally, as already mentioned, the restriction of the adjacency matrix to P i is a second order Ptensor, which gives an easy way of explicitly adding topological information to the activation.

Proposition 7.

If F i is a k'th order P -tensor at node n i , and A↓ Pi is the restriction of the adjacency matrix to P i as defined in Section 4.2, then F ⊗ A↓ Pi is a k + 2'th order P -tensor.

Combining all the above results, assuming that node n t has children n c1 , . . .

, n cs , we arrive at the following general algorithm for the aggregation rule Φ t :1.

Collect all the k'th order activations F c1 , . . .

, F cs of the children.

2.

Promote each activation to F c1 , . . .

, F cs (Proposition 5).

3.

Stack F c1 , . . .

, F cs together into a k + 1 order tensor T (Proposition 6).

4.

Optionally form the tensor product of T with A↓ Pt to get a k+3 order tensor H (otherwise just set H = T ) (Proposition 7).

5.

Contract H along some number of combinations of dimensions to get s separate lower order tensors Q 1 , . . .

, Q s (Proposition 4).

6.

Mix Q 1 , . . .

, Q s with a matrix W ∈ R s ×s and apply a nonlinearity Υ to get the final activation of the neuron, which consists of the s output tensors DISPLAYFORM0 where the b i scalars are bias terms, and 1 is the |P t | × . . .

× |P t | dimensional all ones tensor.

A few remarks are in order about this general scheme: 1.

Since F c1 , . . .

, F cs are stacked into a larger tensor and then possibly also multiplied by A↓ Pt , the general tendency would be for the tensor order to increase at every node, and the corresponding storage requirements to increase exponentially.

The purpose of the contractions inStep 5 is to counteract this tendency, and pull the order of the tensors back to some small number, typically 1, 2 or 3.

2.

However, since contractions can be done in many different ways, the number of channels will increase.

When the number of input channels is small, this is reasonable, since otherwise the number of learnable weights in the algorithm would be too small.

However, if unchecked, this can also become problematic.

Fortunately, mixing the channels by W on Step 6 gives an opportunity to stabilize the number of channels at some value s .

3.

In the pseudocode above, for simplicity, the number of input channels is one and the number of output channels is s .

More realistically, the inputs would also have multiple channels (say, s 0 ) which would be propagated through the algorithm independently up to the mixing stage, making W an s × s × s 0 dimension tensor (not in the P -tensor sense!).

4.

The conventional part of the entire algorithm is Step 6, and the only learnable parameters are the entries of the W matrix (tensor) and the b i bias terms.

These parameters are shared by all nodes in the network and learned in the usual way, by stochastic gradient descent.

5.

Our scheme could be elaborated further while maintaining permutation covariance by, for example taking the tensor product of T with itself, or by introducing A↓ Pt in a different way.

However, the way that F c1 , . . .

, F cs and A↓ Pt are combined by tensor products is already much more general and expressive than conventional message passing networks.

6.

Our framework admits many design choices, including the choice of the order odf the activations, the choice of contractions, and c .

However, the overall structure of Steps 1-5 is fully dictated by the covariance constraint on the network.

7.

The final output of the network φ(G) = F r must be permutation invariant.

That means that the root node n r must produce a tuple of zeroth order tensors (scalars) (Fr , . . .

, F (c) r ).

This is similar to how many other graph representation algorithms compute φ(G) by summing the activations at level L or creating histogram features.

We consider a few special cases to explain how tensor aggregation relates to more conventional message passing rules.

Constraining both the input tensors F c1 , . . .

, F cs and the outputs to be zeroth order tensors, i.e., scalars, and foregoing multiplication by A↓ Pt greatly simplifies the form of Φ. In this case there is no need for promotions, and T is just the vector (F c1 , . . .

, F cs ).

There is only one way to contract a vector into a scalar, and that is to sum its elements.

Therefore, in this case, the entire aggregation algorithm reduces to the simple formula DISPLAYFORM0 For a neural network this is too simplistic.

However, it's interesting to note that the WeisfeilerLehmann isomorphism test essentially builds on just this formula, with a specific choice of Υ BID32 .

If we allow more channels in the inputs and the outputs, W becomes a matrix, and we recover the simplest form of neural message passing algorithms BID8 .

In first order tensor aggregation, assuming that |P i | = m, F c1 , . . .

, F cs are m dimensional column vectors, and T is an m × m matrix consisting of F c1 , . . .

, F cs stacked columnwise.

There are two ways of contracting (in our generalized sense) a matrix into a vector: by summing over its rows, or summing over its columns.

The second of these choices leads us back to summing over all contributions from the children, while the first is more interesting because it corresponds to summing F c1 , . . .

, F cs as vectors individually.

In summary, we get an aggregation function that transforms a single input channel to two output channels of the form DISPLAYFORM0 where 1 denotes the m dimensional all ones vector.

Thus, in this layer W ∈ R 2×2 .

Unless constrained by c , in each subsequent layer the number of channels doubles further and these channels can all mix with each other, so W (2) ∈ R 4×4 , W (3) ∈ R 8×8 , and so on.

In second order tensor aggregation, T is a third order P -tensor, which can be contracted back to second order in three different ways, by projecting it along each of its dimensions.

Therefore the outputs will be the three matrices DISPLAYFORM0 and the weight matrix is W ∈ R 3×3 .

The first nontrivial tensor contraction case occurs when F c1 , . . .

, F cs are second order tensors, and we multiply with A↓ Pt , since in that case T is 5th order, and can be contracted down to second order in a total of 50 different ways: 1.

The "1+1+1" case contracts T in the form T i1,i2,i3,i4,i5 δ ia 1 δ ia 2 δ ia 3 , i.e., it projects T down along 3 of its 5 dimensions.

This alone can be done in 5 3 = 10 different ways 7 2.

The "1+2" case contracts T in the form T i1,i2,i3,i4,i5 δ ia 1 δ ia 2 ,ia 3 , i.e., it projects T along one dimension, and contracts it along two others.

This can be done in 3 5 3 = 30 ways.

3.

The "3" case is a single 3-fold contraction T i1,i2,i3,i4,i5 δ ia 1 ,ia 2 ,ia 3 , which again can be done in 5 3 = 10 different ways.

The tensor T i1,i2,i3,i4,i5 will be symmetric with respect to two sets of indices, following the structure of the promotion tensors and the adjacency matrix.

Including these symmetries, the number of contractions is 18 including: five "1+1+1", ten "1+2", and three "3".

DISPLAYFORM0 Tensor to be contracted Figure 6 : The activations of vertices in the receptive field P v = {w 1 , w 2 , w 3 } of vertex v at level -th are stacked into a 3rd order tensor and undergo a tensor product operation with the restricted adjacency matrix, and then contracted in different ways.

In this figure, we only consider single channel, each channel is represented by a 5th order tensor.

In the general case of multi channels, the resulting tensor would have 6th order, but we contract on each channel separately.

We compared the second order variant (CCN 2D) of our CCNs framework (Section 4.2) to several standard graph learning algorithms on three types of datasets that involve learning the properties of molecules from their structure:1.

The Harvard Clean Energy Project BID16 , consisting of 2.3 million organic compounds that are candidates for use in solar cells.

The regression target in this case is Power Conversion Efficiency (PCE).

Due to time constraints, instead of using the entire dataset, the experiments were ran on a random subset of 50,000 molecules.2.

QM9, which is a dataset of all 133k organic molecules with up to nine heavy atoms (C,O,N and F) out of the GDB-17 universe of molecules.

Each molecule has 13 target properties to predict.

The dataset does contain spatial information relating to the atomic configurations, but we only used the chemical graph and atom node labels.

For our experiments we normalized each target variable to have mean 0 and standard deviation 1.

We report both MAE and RMSE for all normalized learning targets.3.

Graph kernels datasets, specifically (a) MUTAG, which is a dataset of 188 mutagenic aromatic and heteroaromatic compounds BID6 ; (b) PTC, which consists of 344 chemical compounds that have been tested for positive or negative toxicity in lab rats BID39 ; (c) NCI1 and NCI109, which have 4110 and 4127 compounds respectively, each screened for activity against small cell lung cancer and ovarian cancer lines BID42 .In the case of HCEP, we compared CCN to lasso, ridge regression, random forests, gradient boosted trees, optimal assignment Wesifeiler-Lehman graph kernel BID23 ) (WL), neural graph fingerprints BID8 , and the "patchy-SAN" convolutional type algorithm from BID29 ) (referred to as PSCN).

For the first four of these baseline methods, we created simple feature vectors from each molecule: the number of bonds of each type (i.e. number of H-H bonds, number of C-O bonds, etc) and the number of atoms of each type.

Molecular graph fingerprints uses atom labels of each vertex as base features.

For ridge regression and lasso, we cross validated over λ.

For random forests and gradient boosted trees, we used 400 trees, and cross validated over max depth, minimum samples for a leaf, minimum samples to split a node, and learning rate (for GBT).

For neural graph fingerprints, we used 2 layers and a hidden layer size of 10.

In PSCN, we used a patch size of 10 with two convolutional layers and a dense layer on top as described in their paper.

For the graph kernels datasets, we compare against graph kernel results as reported in BID22 ) (which computed kernel matrices using the Weisfeiler-Lehman, Weisfeiler-edge, shortest paths, graphlets and multiscale Laplacian graph kernels and used a C-SVM on top), Neural graph fingerprints (with 2 levels and a hidden size of 10) and PSCN.

For QM9, we compared against the Weisfeiler-Lehman graph kernel (with C-SVM on top), neural graph fingerprints, and PSCN.

The settings for NGF and PSCN are as described for HCEP.For our own method, second order CCN, we initialized the base features of each vertex with computed histogram alignment features, inspired by BID23 , of depth up to 10.

Each vertex receives a base label l i = concat 10 j=1 H j (i) where H j (i) ∈ R d (with d being the total number of distinct discrete node labels) is the vector of relative frequencies of each label for the set of vertices at distance equal to j from vertex i. We use exactly 18 unique contractions defined in 5.1.4 that result in additional channels.

We used up to three levels and the intermediate number of channels increases 18 time at each level.

To avoid exponentially growing channels, we applied learnable weight matrices to compress the channels into a fixed number of channels.

In each experiment we used 80% of the dataset for training, 10% for validation, and evaluated on the remaining 10% test set.

For the kernel datasets we performed the experiments on 10 separate training/validation/test stratified splits and averaged the resulting classification accuracies.

We used Adam optimization method BID19 .

Our initial learning rate was set to 0.001 after experimenting on a held out set.

The learning rate decayed linearly after each step towards a minimum of 10 −6 .

We developed our custom Deep Learning framework in C++/CUDA named GraphFlow that supports symbolic/automatic differentiation, dynamic computation graphs, specialized tensor operations, and computational acceleration with GPU.

Our method, Covariant Compositional Networks, and other graph neural networks such as Neural Graph Fingerprints BID8 , PSCN BID29 and Gated Graph Neural Networks BID26 are implemented based on the GraphFlow framework.

Our source code can be found at https://github.com/HyTruongSon/ GraphFlow.

One challenge of the implementation of Covariant Compositional Networks is that the high-order tensors (for example, in figure 6, we have a 5th order tensor after the tensor product operation) cannot be stored explicitly in the memory.

Our solution is to propose a virtual indexing system in such a way that we never compute the whole sparse high-order tensor at once, but only compute its elements when given the indices.

Basically, we always work with a virtual tensor, and that allows us to implement our tensor reduction/contraction operations efficiently with GPU.

On the subsampled HCEP dataset, CCN outperforms all other methods by a very large margin.

For the graph kernels datasets, SVM with the Weisfeiler-Lehman kernels achieve the highest accuracy on NCI1 and NCI109, while CCN wins on MUTAG and PTC.

Perhaps this poor performance is to be expected, since the datasets are small and neural network approaches usually require tens of thousands of training examples at minimum to be effective.

Indeed, neural graph fingerprints and PSCN also perform poorly compared to the Weisfeiler-Lehman kernels.

In the QM9 experiments, CCN beats the three other algorithms in both mean absolute error and root mean squared error.

It should be noted that BID15 obtained stronger results on QM9, but we cannot properly compare our results with theirs because our experiments only use the adjacency matrices and atom labels of each node, while theirs includes comprehensive chemical features that better inform the target quantum properties.

We have presented a general framework called covariant compositional networks (CCNs) for constructing covariant graph neural networks, which encompasses other message passing approaches as special cases, but takes a more general and principled approach to ensuring covariance with respect to permutations.

Experimental results on several benchmark datasets show that CCNs can outperform other state-of-the-art algorithms.

clearly true, since f a = f a = ξ(a) .

Now assume that it is true for all nodes with height up to h * .

For any node n a with h(a) = h * + 1, f a = Φ(f c1 , f c2 , . . . , f c k ), where each of the children c 1 , . . . , c k are of height at most h * , therefore f a = Φ(f c1 , f c2 , . . . , f c k ) = Φ(f c1 , f c2 , . . . , f c k ) = f a .Thus, f a = f a for every node in G. The proposition follows by φ(G) = f r = f r = φ(G ).Proof of Proposition 3.

Let G, G , N and N be as in Definition 5.

As in Definition 6, for each node (neuron) n i in N there is a node n j in N such that their receptive fields are equivalent up to permutation.

That is, if |P i | = m, then |P j | = m, and there is a permutation π ∈ S m , such that if P i = (e p1 , . . . , e pm ) and P j = (e q1 , . . .

, e qm ), then e q π(a) = e pa .

By covariance, then f j = R π (f i ).Now let G be a third equivalent object, and N the corresponding comp-net.

N must also have a node, n k , that corresponds to n i and n j .

In particular, letting its receptive field be P k = (e r1 , . . .

, e rm ), there is a permutation σ ∈ S m for which e r σ(b) = e q b .

Therefore, f k = R σ (f j ).At the same time, n k is also in correspondence with n i .

In particular, letting τ = σπ (which corresponds to first applying the permutation π, then applying σ), e r τ (a) = e pa , and therefore f k = R τ (f i ).

Hence, the {R π } maps must satisfy Case 4.

Follows directly from 3.Case 5.

Finally, if A 1 , ..., A u are k'th order P -tensors and C = j α j A j then DISPLAYFORM0 so C is a k'th order P -tensor.

Proof of Proposition 5.

Under the action of a permutation π ∈ S m on P b , χ (dropping the a→b superscipt) transforms to χ , where χ i,j = χ π −1 (i),j .

However, this can also be written as DISPLAYFORM1 Therefore, F i1,...,i k transforms to DISPLAYFORM2 so F is a P -tensor.

Proof of Proposition 6.

By Proposition 5, under the action of any permutation π, each of the F pj slices of F transforms as DISPLAYFORM3 At the same time, π also permutes the slices amongst each other according to DISPLAYFORM4 so F is a k + 1'th order P -tensor.

Proof of Proposition 7.

Under any permutation π ∈ S m of P i , A↓ P i transforms to A↓ P i , where [A↓ P i ] π(a),π(b) = [A↓ Pi ] a,b .

Therefore, A↓ Pi is a second order P -tensor.

By the first case of Proposition 4, F ⊗ A↓ Pi is then a k + 2'th order P -tensor.

<|TLDR|>

@highlight

A general framework for creating covariant graph neural networks