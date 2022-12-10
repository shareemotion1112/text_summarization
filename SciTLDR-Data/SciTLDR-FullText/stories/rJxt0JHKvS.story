In this paper, we show that a simple coloring scheme can improve, both theoretically and empirically, the expressive power of Message Passing Neural Networks (MPNNs).

More specifically, we introduce a graph neural network called Colored Local Iterative Procedure (CLIP) that uses colors to disambiguate identical node attributes, and show that this representation is a universal approximator of continuous functions on graphs with node attributes.

Our method relies on separability, a key topological characteristic that allows to extend well-chosen neural networks into universal representations.

Finally, we show experimentally that CLIP is capable of capturing structural characteristics that traditional MPNNs fail to distinguish, while being state-of-the-art on benchmark graph classification datasets.

Learning good representations is seen by many machine learning researchers as the main reason behind the tremendous successes of the field in recent years (Bengio et al., 2013) .

In image analysis (Krizhevsky et al., 2012) , natural language processing (Vaswani et al., 2017) or reinforcement learning (Mnih et al., 2015) , groundbreaking results rely on efficient and flexible deep learning architectures that are capable of transforming a complex input into a simple vector while retaining most of its valuable features.

The universal approximation theorem (Cybenko, 1989; Hornik et al., 1989; Hornik, 1991; Pinkus, 1999) provides a theoretical framework to analyze the expressive power of such architectures by proving that, under mild hypotheses, multi-layer perceptrons (MLPs) can uniformly approximate any continuous function on a compact set.

This result provided a first theoretical justification of the strong approximation capabilities of neural networks, and was the starting point of more refined analyses providing valuable insights into the generalization capabilities of these architectures (Baum and Haussler, 1989; Geman et al., 1992; Saxe et al., 2014; Bartlett et al., 2018) .

Despite a large literature and state-of-the-art performance on benchmark graph classification datasets, graph neural networks yet lack a similar theoretical foundation (Xu et al., 2019) .

Universality for these architectures is either hinted at via equivalence with approximate graph isomorphism tests (k-WL tests in Xu et al. 2019; Maron et al. 2019a ), or proved under restrictive assumptions (finite node attribute space in Murphy et al. 2019) .

In this paper, we introduce Colored Local Iterative Procedure 1 (CLIP), which tackles the limitations of current Message Passing Neural Networks (MPNNs) by showing, both theoretically and experimentally, that adding a simple coloring scheme can improve the flexibility and power of these graph representations.

More specifically, our contributions are: 1) we provide a precise mathematical definition for universal graph representations, 2) we present a general mechanism to design universal neural networks using separability, 3) we propose a novel node coloring scheme leading to CLIP, the first provably universal extension of MPNNs, 4) we show that CLIP achieves state of the art results on benchmark datasets while significantly outperforming traditional MPNNs as well as recent methods on graph property testing.

The rest of the paper is organized as follows:

Section 2 gives an overview of the graph representation literature and related works.

Section 3 provides a precise definition for universal representations, as well as a generic method to design them using separable neural networks.

In Section 4, we show that most state-of-the-art representations are not sufficiently expressive to be universal.

Then, using the analysis of Section 3, Section 5 provides CLIP, a provably universal extension of MPNNs.

Finally, Section 6 shows that CLIP achieves state-of-the-art accuracies on benchmark graph classification taks, as well as outperforming its competitors on graph property testing problems.

The first works investigating the use of neural networks for graphs used recurrent neural networks to represent directed acyclic graphs (Sperduti and Starita, 1997; Frasconi et al., 1998) .

More generic graph neural networks were later introduced by Gori et al. (2005) ; Scarselli et al. (2009) , and may be divided into two categories.

1) Spectral methods (Bruna et al., 2014; Henaff et al., 2015; Defferrard et al., 2016; Kipf and Welling, 2017 ) that perform convolution on the Fourier domain of the graph through the spectral decomposition of the graph Laplacian.

2) Message passing neural networks (Gilmer et al., 2017) , sometimes simply referred to as graph neural networks, that are based on the aggregation of neighborhood information through a local iterative process.

This category contains most state-of-the-art graph representation methods such as (Duvenaud et al., 2015; Grover and Leskovec, 2016; Lei et al., 2017; Ying et al., 2018; Verma and Zhang, 2019) , DeepWalk (Perozzi et al., 2014) , graph attention networks (Velickovic et al., 2018) , graphSAGE (Hamilton et al., 2017) or GIN (Xu et al., 2019) .

Recently, (Xu et al., 2019) showed that MPNNs were, at most, as expressive as the WeisfeilerLehman (WL) test for graph isomorphism (Weisfeiler and Lehman, 1968) .

This suprising result led to several works proposing MPNN extensions to improve their expressivity, and ultimately tend towards universality (Maron et al., 2019a; b; Murphy et al., 2019; Chen et al., 2019) .

However, these graph representations are either as powerful as the k-WL test (Maron et al., 2019a) , or provide universal graph representations under the restrictive assumption of finite node attribute space (Murphy et al., 2019) .

Other recent approaches (Maron et al., 2019c) implies quadratic order of tensors in the size of the considered graphs.

Some more powerfull GNNs are studied and benchmarked on real classical datasets and on graph property testing (Kriege et al., 2018; Murphy et al., 2019; Chen et al., 2019) : a set of problems that classical MPNNs cannot handle.

Our work thus provides a more general and powerful result of universality, matching the original definition of (Cybenko, 1989) for MLPs.

In this section we present the theoretical tools used to design our universal graph representation.

More specifically, we show that separable representations are sufficiently flexible to capture all relevant information about a given object, and may be extended into universal representations.

Let X , Y be two topological spaces, then F(X , Y) (resp.

C(X , Y)) denotes the space of all functions (resp.

continuous functions) from X to Y. Moreover, for any group G acting on a set X , X /G denotes the set of orbits of X under the action of G (see Appendix B for more details).

Finally, · is a norm on R d , and P n is the set of all permutation matrices of size n. In what follows, we assume that all the considered topological spaces are Hausdorff (see e.g. (Bourbaki, 1998) for an in-depth review): each pair of distinct points can be separated by two disjoint open sets.

This assumption is rather weak (e.g. all metric spaces are Hausdorff) and is verified by most topological spaces commonly encountered in the field of machine learning.

Let X be a set of objects (e.g. vectors, images, graphs, or temporal data) to be used as input information for a machine learning task (e.g. classification, regression or clustering).

In what follows, we denote as vector representation of X a function f : X → R d that maps each element x ∈ X to a d-dimensional vector f (x) ∈ R d .

A standard setting for supervised representation learning is to define a class of vector representations F d ⊂ F(X , R d ) (e.g. convolutional neural networks for images) and use the target values (e.g. image classes) to learn a good vector representation in light of the supervised learning task (i.e. one vector representation f ∈ F d that leads to a good accuracy on the learning task).

In order to present more general results, we will consider neural network architectures that can output vectors of any size, i.e. F ⊂ ∪ d∈N * F(X , R d ), and will denote the set of d-dimensional vector representations of F. A natural characteristic to ask from the class F is to be generic enough to approximate any vector representation, a notion that we will denote as universal representation (Hornik et al., 1989) .

In other words, F is a universal representation of a normed space X if and only if, for any continuous function φ : X → R d , any compact K ⊂ X and any ε > 0, there exists f ∈ F such that

One of the most fundamental theorems of neural network theory states that one hidden layer MLPs are universal representations of the m-dimensional vector space R m .

Theorem 1 (Pinkus, 1999, Theorem 3.1) .

Let ϕ : R → R be a continuous non polynomial activation function.

For any compact K ⊂ R m and d ∈ N * , two layers neural networks with activation ϕ are uniformly dense in the set C(K, R d ).

However, for graphs and structured objects, universal representations are hard to obtain due to their complex structure and invariance to a group of transformations (e.g. permutations of the node labels).

We show in this paper that a key topological property, separability, may lead to universal representations of those structures.

Loosely speaking, universal representations can approximate any vector-valued function.

It is thus natural to require that these representations are expressive enough to separate each pair of dissimilar elements of X .

Definition 2 (Separability).

A set of functions F ⊂ F(X , Y) is said to separate points of X if for every pair of distinct points x and y, there exists f ∈ F such that f (x) = f (y).

we will say that F is separable if its 1-dimensional representations F 1 separates points of X .

Separability is rather weak, as we only require the existence of different outputs for every pair of inputs.

Unsurprisingly, we now show that it is a necessary condition for universality (see Appendix A for all the detailed proofs).

Proposition 1.

Let F be a universal representation of X , then F 1 separates points of X .

While separability is necessary for universal representations, it is also key to designing neural network architectures that can be extended into universal representations.

More specifically, under technical assumptions, separable representations can be composed with a universal representation of R d (such as MLPs) to become universal.

Theorem 2.

For all d ≥ 0, let M d be a universal approximation of R d .

Let F be a class of vector representations of X such that:

Stability by concatenation is verified by most neural networks architectures, as illustrated for MLPs in Figure 1 .

The proof of Theorem 2 relies on the Stone-Weierstrass theorem (see e.g. Rudin, 1987, Theorem 7.32 ) whose assumptions are continuity, separability, and the fact that the class of functions is an algebra.

Fortunately, composing a separable and concatenable representation with a universal representation automatically leads to an algebra, and thus the applicability of the StoneWeierstrass theorem and the desired result.

A complete derivation is available in Appendix A. Since MLPs are universal representations of R d , Theorem 2 implies a convenient way to design universal representations of more complex object spaces: create a separable representation and compose it with a simple MLP (see Figure 2) .

Corollary 1.

A continuous, concatenable and separable representation of X composed with an MLP is universal.

Note that many neural networks of the deep learning literature have this two steps structure, including classical image CNNs such as AlexNet (Krizhevsky et al., 2012) or Inception (Szegedy et al., 2016) .

In this paper, we use Corollary 1 to design universal graph and neighborhood representations, although the method is much more generic and may be applied to other objects.

In this section, we first provide a proper definition for graphs with node attributes, and then show that message passing neural networks are not sufficiently expressive to be universal.

Consider a dataset of n interacting objects (e.g. users of a social network) in which each object i ∈ 1, n has a vector attribute v i ∈ R m and is a node in an undirected graph G with adjacency matrix A ∈ R n×n .

Definition 3.

The space of graphs of size n with m-dimensional node attributes is the quotient space

where A is the adjacency matrix of the graph, v contains the m-dimensional representation of each node in the graph and the set of permutations matrices P n is acting on (v, A) by

Moreover, we limit ourselves to graphs of maximum size n max , where n max is a large integer.

This allows us to consider functions on graphs of different sizes without obtaining infinite dimensional spaces and infinitely complex functions that would be impossible to learn via a finite number of samples.

We thus define Graph m = n≤nmax Graph m,n .

More details on the technical topological aspects of the definition are available in Appendix B, as well as a proof that Graph m is Hausdorff.

A common method for designing graph representations is to rely on local iterative procedures.

Following the notations of Xu et al. (2019) , a message passing neural network (MPNN) (Gilmer et al., 2017 ) is made of three consecutive phases that will create intermediate node representations x i,t for each node i ∈ 1, n and a final graph representation x G as described by the following procedure: 1) Initialization: All node representations are initialized with their node attributes

2) Aggregation and combination: T local iterative steps are performed in order to capture larger and larger structural characteristics of the graph.

3) Readout: This step combines all final node representations into a single graph representation:

where READOUT is permutation invariant.

Unfortunately, while MPNNs are very efficient in practice and proven to be as expressive as the Weisfeiler-Lehman algorithm (Weisfeiler and Lehman, 1968; Xu et al., 2019) , they are not sufficiently expressive to construct isomorphism tests or separate all graphs (for example, consider k-regular graphs without node attributes, for which a small calculation shows that any MPNN representation will only depend on the number of nodes and degree k (Xu et al., 2019) ).

As a direct application of Proposition 1, MPNNs are thus not expressive enough to create universal representations.

In this section, we present Colored Local Iterative Procedure (CLIP), an extension of MPNNs using colors to differentiate identical node attributes, that is able to capture more complex structural graph characteristics than traditional MPNNs.

This is proved theoretically through a universal approximation theorem in Section 5.3 and experimentally in Section 6.

CLIP is based on three consecutive steps: 1) graphs are colored with several different colorings, 2) a neighborhood aggregation scheme provides a vector representation for each colored graph, 3) all vector representations are combined to provide a final output vector.

We now provide more information on the coloring scheme.

In order to distinguish non-isomorphic graphs, our approach consists in coloring nodes of the graph with identical attributes.

This idea is inspired by classical graph isomorphism algorithms that use colors to distinguish nodes (McKay, 1981) , and may be viewed as an extension of one-hot encodings used for graphs without node attributes (Xu et al., 2019) .

For any k ∈ N, let C k be a finite set of k colors.

These colors may be represented as one-hot encodings (C k is the natural basis of R k ) or more generally any finite set of k elements.

At initialization, we first partition the nodes into groups of identical attributes V 1 , ..., V K ⊂ 1, n .

Then, for a subset V k of size |V k |, we give to each of its nodes a distinct color from C k (hence a subset of size |V k |).

For example, Figure 3 shows two colorings of the same graph, which is decomposed in three groups V 1 , V 2 and V 3 containing nodes with attributes a, b and c respectively.

Since V 1 contains only two nodes, a coloring of the graph will attribute two colors ((1, 0) and (0, 1), depicted as blue and red) to these nodes.

More precisely, the set of colorings C(v, A) of a graph G = (v, A) are defined as

In the CLIP algorithm, we add a coloring scheme to an MPNN in order to distinguish identical node attributes.

This is achieved by modifying the initialization and readout phases of MPNNs as follows.

We first select a set C k ⊆ C(v, A) of k distinct colorings uniformly at random (see Eq. (4)).

Then, for each coloring c ∈ C k , node representations are initialized with their node attributes concatenated with their color:

This step is performed for all colorings c ∈ C k using a universal set representation as the aggregation function:

where ψ and ϕ are MLPs with continuous non-polynomial activation functions and ψ(x, y) denotes the result of ψ applied to the concatenation of x and y. The aggregation scheme we propose is closely related to DeepSet (Zaheer et al., 2017) , and a direct application of Corollary 1 proves the universality of our architecture.

More details, as well as the proof of universality, are available in Appendix C.

3.

Colored readout: This step performs a maximum over all possible colorings in order to obtain a final coloring-independent graph representation.

In order to keep the stability by concatenation, the maximum is taken coefficient-wise

where ψ is an MLP with continuous non polynomial activation functions.

We treat k as a hyper-parameter of the algorithm and call k-CLIP (resp.

∞-CLIP) the algorithm using k colorings (resp.

all colorings, i.e. k = |C(v, A)|).

Note that, while our focus is graphs with node attributes, the approach used for CLIP is easily extendable to similar data structures such as directed or weighted graphs with node attributes, graphs with node labels, graphs with edge attributes or graphs with additional attributes at the graph level.

As the colorings are chosen at random, the CLIP representation is itself random as soon as k < |C(v, A)|, and the number of colorings k will impact the variance of the representation.

However, ∞-CLIP is deterministic and permutation invariant, as MPNNs are permutation invariant.

The separability is less trivial and is ensured by the coloring scheme.

Theorem 3.

The ∞-CLIP algorithm with one local iteration (T = 1) is a universal representation of the space Graph m of graphs with node attributes.

The proof of Theorem 3 relies on showing that ∞-CLIP is separable and applying Corollary 1.

This is achieved by fixing a coloring on one graph and identifying all nodes and edges of the second graph using the fact that all pairs (v i , c i ) are dissimilar (see Appendix D).

Similarly to the case of MLPs, only one local iteration is necessary to ensure universality of the representation.

This rather counter-intuitive result is due to the fact that all nodes can be identified by their color, and the readout function can aggregate all the structural information in a complex and non-trivial way.

However, as for MLPs, one may expect poor generalization capabilities for CLIP with only one local iteration, and deeper networks may allow for more complex representations and better generalization.

This point is addressed in the experiments of Section 6.

Moreover, ∞-CLIP may be slow in practice due to a large number of colorings, and reducing k will speed-up the computation.

Fortunately, while k-CLIP is random, a similar universality theorem still holds even for k = 1.

Theorem 4.

The 1-CLIP algorithm with one local iteration (T = 1) is a random representation whose expectation is a universal representation of the space Graph m of graphs with node attributes.

The proof of Theorem 4 relies on using ∞-CLIP on the augmented node attributes v i = (v i , c i ).

As all node attributes are, by design, different, the max over all colorings in Eq. (5) disappears and, for any coloring, 1-CLIP returns an ε-approximation of the target function (see Appendix D).

Remark 1.

Note that the variance of the representation may be reduced by averaging over multiple samples.

Moreover, the proof of Theorem 4 shows that the variance can be reduced to an arbitrary precision given enough training epochs, although this may lead to very large training times in practice.

As the local iterative steps are performed T times on each node and the complexity of the aggregation depends on the number of neighbors of the considered node, the complexity is proportional to the number of edges of the graph E and the number of steps T .

Moreover, CLIP performs this iterative aggregation for each coloring, and its complexity is also proportional to the number of chosen colorings k = |C k |.

Hence the complexity of the algorithm is in O(kET ).

Note that the number of all possible colorings for a given graph depends exponentially in the size of the groups V 1 , ..., V K ,

and thus ∞-CLIP is practical only when most node attributes are dissimilar.

This worst case exponential dependency in the number of nodes can hardly be avoided for universal representations.

Indeed, a universal graph representation should also be able to solve the graph isomorphism problem.

Despite the existence of polynomial time algorithms for a broad class of graphs (Luks, 1982; Bodlaender, 1990) , graph isomorphism is still quasi-polynomial in general (Babai, 2016) .

As a result, creating a universal graph representation with polynomial complexity for all possible graphs and functions to approximate is highly unlikely, as it would also induce a graph isomorphism test of polynomial complexity and thus solve a very hard and long standing open problem of theoretical computer science.

In this section we show empirically the practical efficiency of CLIP and its relaxation.

We run two sets of experiments to compare CLIP w.r.t.

state-of-the-art methods in supervised learning settings: i) on 5 real-world graph classification datasets and ii) on 4 synthetic datasets to distinguish structural graph properties and isomorphism.

Both experiments follow the same experimental protocol as described in Xu et al. (2019) : 10-fold cross validation with grid search hyper-parameter optimization.

More details on the experimental setup are provided in Appendix E.

We performed experiments on five benchmark datasets extracted from standard social networks (IMDBb and IMDBm) and bio-informatics databases (MUTAG, PROTEINS and PTC).

All dataset characteristics (e.g. size, classes), as well as the experimental setup, are available in Appendix E. Following standard practices for graph classification on these datasets, we use one-hot encodings of node degrees as node attributes for IMDBb and IMDBm (Xu et al., 2019) , and perform singlelabel multi-class classification on all datasets.

We compared CLIP with six state-of-the-art baseline algorithms: 1) WL: Weisfeiler-Lehman subtree kernel (Shervashidze et al., 2011) , 2) AWL: Anonymous Walk Embeddings (Ivanov and Burnaev, 2018) , 3) DCNN: Diffusion-convolutional neural networks (Atwood and Towsley, 2016) , 4) PS: PATCHY-SAN (Niepert et al., 2016) , 5) DGCNN: Deep Graph CNN (Zhang et al., 2018) and 6) GIN: Graph Isomorphism Network (Xu et al., 2019) .

WL and AWL are representative of unsupervised methods coupled with an SVM classifier, while DCNN, PS, DGCNN and GIN are four deep learning architectures.

As the same experimental protocol as that of Xu et al. (2019) was used, we present their reported results on Table 1 .

Table 1 shows, CLIP can achieve state-of-the-art performance on the five benchmark datasets.

Moreover, CLIP is consistent across all datasets, while all other competitors have at least one weak performance.

This is a good indicator of the robustness of the method to multiple classification tasks and dataset types.

Finally, the addition of colors does not improve the accuracy for these graph classification tasks, except on the MUTAG dataset.

This may come from the small dataset sizes (leading to high variances) or an inherent difficulty of these classification tasks, and contrasts with the clear improvements of the method for property testing (see Section 6.2).

More details on the performance of CLIP w.r.t.

the number of colors k are available in Appendix E. Remark 2.

In three out of five datasets, none of the recent state-of-the-art algorithms have statistically significantly better results than older methods (e.g. WL).

We argue that, considering the high variances of all classification algorithms on classical graph datasets, graph property testing may be better suited to measure the expressiveness of graph representation learning algorithms in practice.

We now investigate the ability of CLIP to identify structural graph properties, a task which was previously used to evaluate the expressivity of graph kernels and on which the Weisfeiler-Lehman subtree kernel has been shown to fail for bounded-degree graphs (Kriege et al., 2018) .

The performance of our algorithm is evaluated for the binary classification of four different structural properties: 1) connectivity, 2) bipartiteness, 3) triangle-freeness, 4) circular skip links (Murphy et al., 2019 ) (see Appendix E for precise definitions of these properties) against three competitors: a) GIN, arguably the most efficient MPNN variant yet published (Xu et al., 2019) , b) Ring-GNN, a permutation invariant network that uses the ring of matrix addition and multiplication (Chen et al., 2019) , c) RP-GIN, the Graph Isomorphism Network combined with Relational Pooling, as described by Murphy et al. (2019) , which is able to distinguish certain cases of non-isomorphic regular graphs.

We provide all experimental details in Appendix E. Table 2 : Classification accuracies of the synthetic datasets.

k-RP-GIN refers to a relational pooling averaged over k random permutations.

We report Ring-GNN results from Chen et al. (2019) .

Connectivity Bipartiteness Triangle-freeness Circular skip links mean ± std mean ± std mean ± std mean ± std max min Table 2 shows that CLIP is able to capture the structural information of connectivity, bipartiteness, triangle-freeness and circular skip links, while MPNN variants fail to identify these graph properties.

Furthermore, we observe that CLIP outperforms RP-GIN, that was shown to provide very expressive representations for regular graphs (Murphy et al., 2019) , even with a high number of permutations (the equivalent of colors in their method is set to k = 16).

Moreover, both for k-RP-GIN and k-CLIP, the increase of permutations and colorings respectively lead to higher accuracies.

In particular, CLIP can capture almost perfectly the different graph properties with as little as k = 16 colorings.

In this paper, we showed that a simple coloring scheme can improve the expressive power of MPNNs.

Using such a coloring scheme, we extended MPNNs to create CLIP, the first universal graph representation.

Universality was proven using the novel concept of separable neural networks, and our experiments showed that CLIP is state-of-the-art on both graph classification datasets and property testing tasks.

The coloring scheme is especially well suited to hard classification tasks that require complex structural information to learn.

The framework is general and simple enough to extend to other data structures such as directed, weighted or labeled graphs.

Future work includes more detailed and quantitative approximation results depending on the parameters of the architecture such as the number of colors k, or number of hops of the iterative neighborhood aggregation.

Proof of Theorem 2.

The proof relies on the Stone-Weierstrass theorem we recall below.

We refer to (Rudin, 1987, Theorem 7.32 ) for a detailed proof of the following classical theorem.

Theorem 5 (Stone-Weierstrass).

Let A be an algebra of real functions on a compact Hausdorff set K. If A separates points of K and contains a non-zero constant function, then A is uniformly dense in C(K, R).

We verify that under the assumptions of Theorem 2 the Stone-Weierstrass theorem applies.

In this setting, we first prove the theorem for m = 1 and use induction for the general case.

Let K ⊂ X be a compact subset of X .

We will denote

and will proceed in two steps: we first show that A 0 is uniformly dense in C(K, R), then that A is dense in A 0 , hence proving Theorem 2.

Proof.

The subset A 0 contains zero and all constants.

Let f, g ∈ A 0 so that

,

and by assumption ϕ ∈ F.

We have

so that f + g ∈ A 0 and we conclude that A 0 is a vectorial subspace of C(K, R).

We proceed similarly for the product in order to finish the proof of the lemma.

Because F 1 separates the points of X by assumption, A 0 also separates the points of X .

Indeed, let x = y two distinct points of X so that ∃f ∈ F such that f (x) = f (y).

There exists g ∈ C(R d , R) such that g(f (x)) = g(f (y)).

From Theorem 5 we deduce that A 0 is uniformly dense in C(K, R) for all compact subsets K ⊂ X .

Lemma 2.

For any compact subset K ⊂ X , A is uniformly dense in A 0 .

Proof.

Let > 0 and h = ψ 0 • f ∈ A 0 with f ∈ F and ψ 0 ∈ C(R d , R).

Thanks to the continuity of f , the imageK = f (K) is a compact of R d .

By Theorem 1 there exists an MLP ψ such that

This last lemma completes the proof in the case m = 1.

, f ∈ F} and proceed in a similar manner than Lemma 2 by decomposing

and applying Lemma 1 for each coefficient function

Proof of Proposition 1.

Assume that there exists x, y ∈ X s.t.

∀f ∈ F 1 , f (x) = f (y).

Then K = {x, y} is a compact subset of X and let φ ∈ C(K, R) be such that φ(x) = 1 and φ(y) = 0.

Thus, for all f ∈ F 1 , max z∈{x,y} φ(z) − f (z) ≥ 1/2 which contradicts universality (see Definition 1).

In what follows, X is always a topological set and G a group of transformations acting on X .

The orbits of X under the action of G are the sets Gx = {g · x : g ∈ G}. Moreover, we denote as X /G the quotient space of orbits, also defined by the equivalence relation: x ∼ y ⇐⇒ ∃g ∈ G s.t.

x = g ·

y.

As stated in Section 5, graphs with node attributes can be defined using invariance by permutation of the labels.

We prove here that the resulting spaces are Hausdorff.

Lemma 3 ( (Bourbaki, 1998, I, §8.

3)).

Let X be a Hausdorff space and R an equivalence relation of X .

Then X /R is Hausdorff if and only if any two distinct equivalence classes in X are contained in disjoints saturated open subsets of X .

Thanks to this lemma we prove the following proposition.

Proposition 2.

Let G a finite group acting on an Hausdorff space X , then the orbit space X /G is Hausdorff.

Proof.

Let Gx and Gy two distinct classes with disjoint open neighbourhood U and V .

By finiteness of G, the application π :

Suppose that there exists z ∈Ũ ∩Ṽ , then π(z) ∈ π(U )

∩ π(V ) and we finally get that Gz ⊂ U ∩ V = ∅. ThereforeŨ ∩Ṽ is empty and X /G is Hausdorff by Lemma 3.

Proposition 2 directly implies that the spaces Graph m and Neighborhood m are Hausdorff.

We now provide more details on the aggregation and combination scheme of CLIP, and show that a simple application of Corollary 1 is sufficient to prove its universality for node neighborhoods.

Each local aggregation step takes as input a couple (x i , {x j } j∈Ni ) where x i ∈ R m is the representation of node i, and {x j } j∈Ni is the set of vector representations of the neighbors of node i. In the following, we show how to use Corollary 1 to design universal representations for node neighborhoods.

Definition 5.

The set of node neighborhoods for m-dimensional node attributes is defined as

where the set of permutation matrices P n is acting on R n×m by P · v = P v.

The main difficulty to design universal neighborhood representations is that the node neighborhoods of Definition 5 are permutation invariant w.r.t.

neighboring node attributes, and hence require permutation invariant representations.

The graph neural network literature already contains several deep learning architectures for permutation invariant sets (Guttenberg et al., 2016; Qi et al., 2017; Zaheer et al., 2017; Xu et al., 2019) , among which PointNet and DeepSet have the notable advantage of being provably universal for sets.

Following Corollary 1, we compose a separable permutation invariant network with an MLP that will aggregate both information from the node itself and its neighborhood.

While our final architecture is similar to Deepset (Zaheer et al., 2017) , this section emphasizes that the general universality theorems of Section 3 are easily applicable in many settings including permutation invariant networks.

The permutation invariant set representation used for the aggregation step of CLIP is as follows:

where ψ and ϕ are MLPs with continuous non-polynomial activation functions and ψ(x, y) denotes the result of the MLP ψ applied to the concatenation of x and y.

Theorem 6.

The set representation described in Eq. (9)

Taking ψ(x, y) = y and ε = 1/3 max{|S 1 |, |S 2 |}, we have

which proves separability and, using Corollary 1, the universality of the representation.

Proof of Theorem 3.

First of all, as the activation functions of the MLPs are continuous, CLIP is made of continuous and concatenable functions, and is thus also continuous and concatenable.

Second, as the node aggregation step (denoted NODEAGGREGATION below) is a universal set representation (see Appendix C), it is capable of approximating any continuous function.

We will thus first replace this function by a continuous function φ, and then show that the result still holds for NODEAGGREGATION (1) by a simple density argument.

Let

2 ) be two distinct graphs of respective sizes n 1 and n 2 (up to a permutation).

If n 1 = n 2 , then ψ(x) = x and φ(x) = 1 returns the number of nodes, and hence x G 1 = n 1 = n 2 = x G 2 .

Otherwise, let V = {v k i } i∈ 1,n 1 ,k∈{1,2} be the set of node attributes of G 1 and G 2 , c 1 be a coloring of G 1 , ψ(x) = x and φ be a continuous function such that, ∀x ∈ V and S ⊂ V ,

The existence of φ ∈ C(R m , R) is assured by Urysohn's lemma (see e.g. (Rudin, 1987, lemma 2.12) ).

Then, x G counts the number of matching neighborhoods for the best coloring, and we have x G 1 = n 1 and x G 2 ≤ n 1 − 1.

Finally, taking ε < 1/2n 1 in the definition of universal representation leads to the desired result, as then, using an ε-approximation of φ as NODEAGGREGATION

(1) , we have

Proof of Theorem 4.

Consider a continuous function ψ :

nmax and we define φ :

Moreover, observe that for any coloring c ∈ C(v, A), ∞-CLIP and 1-CLIP applied to ((v, c) , A) returns the same result, as all node attributes are dissimilar (by definition of the colorings) and C((v, c), A) = ∅. Finally, 1-CLIP applied to (v, A) is equivalent to applying 1-CLIP to ((v, C), A) where C is a random coloring in C(v, A), and Eq. (12) thus implies that any random sample of 1-CLIP is within an ε error of the target function ψ.

As a result, its expectation is also within an ε error of the target function ψ, which proves the universality of the expectation of 1-CLIP.

E.1 REAL-WORLD DATASETS Table 3 summarizes the characteristics of all benchmark graph classification datasets used in Section 6.1.

We now provide complementary information on these datasets.

Social Network Datasets (IMDBb, IMDBm): These datasets refer to collaborations between actors/actresses, where each graph is an ego-graph of every actor and the edges occur when the connected nodes/actors are playing in the same movie.

The task is to classify the genre of the movie that the graph derives from.

IMDBb is a single-class classification dataset, while IMDBm is multi-class.

For both social network datasets, we used one-hot encodings of node degrees as node attribute vectors.

Bio-informatics Datasets (MUTAG, PROTEINS, PTC): MUTAG consists of mutagenic aromatic and heteroaromatic nitrocompounds with 7 discrete labels.

PROTEINS consists of nodes, which correspond to secondary structureelements and the edges occur when the connected nodes are neighbors in the amino-acidsequence or in 3D space.

It has 3 discrete labels.

PTC consists of chemical compounds that reports the carcinogenicity for male and female rats and it has 19 discrete labels.

For all bio-informatics datasets we used the node labels as node attribute vectors.

Experimentation protocol: We follow the same experimental protocol as described in Xu et al. (2019) , and thus report the results provided in this paper corresponding to the accuracy of our six baselines in Table 1 .

We optimized the CLIP hyperparameters by grid search according to 10-fold cross-validated accuracy means.

We use 2-layer MLPs, an initial learning rate of 0.001 and decreased the learning rate by 0.5 every 50 epochs for all possible settings.

For all datasets the hyperparameters we tested are: the number of hidden units within {32, 64}, the number of colorings c ∈ {1, 2, 4, 8}, the number of MPNN layers within {1, 3, 5}, the batch size within {32, 64}, and the number of epochs, that means, we select a single epoch with the best cross-validation accuracy averaged over the 10 folds.

Note that standard deviations are fairly high for all models due to the small size of these classic datasets.

Table 4 summarizes the performances of CLIP while increasing the number of colorings k. Overall we can see a small increase in performances and a reduction of the variances when k is increasing.

Nevertheless we should not jump to any conclusions since none of the models are statistically significantly better than the others.

In Section 6.2 we evaluate the expressive power of CLIP on benchmark synthetic datasets.

Our goal is to show that CLIP is able to distinguish basic graph properties, where classical MPNN cannot.

We considered a binary classification task and we constructed balanced synthetic datasets 2 for each of the examined graph properties.

The 20-node graphs are generated using Erdös-Rényi model (Erdös and Rényi, 1959 ) (and its bipartite version for the bipartiteness) with different probabilities p for edge creation.

All nodes share the same (scalar) attribute.

We thus have uninformative feature vectors.

In particular, we generated datasets for different classical tasks Kriege et al. (2018) : 1) connectivity, 2) bipartiteness, 3) triangle-freeness, and 4) circular skip links (Murphy et al., 2019) .

In the following, we present the generating protocol of the synthetic datasets and the experimentation setup we used for the experiments.

In every case of synthetic dataset we follow the same pattern: we generate a set of random graphs using Erdös-Rényi model, which contain a specific graph property and belong to the same class and by proper edge addition we remove this property, thus creating the second class of graphs.

By this way, we assure that we do not change different structural characteristics other than the examined graph property.

-Connectivity dataset: this dataset consists of 1000 (20-node) graphs with 500 positive samples and 500 negative ones.

The positive samples correspond to disconnected graphs with two 10-node connected components selected among randomly generated graphs with an Erdös-Rényi model probability of p = 0.5.

We constructed negative samples by adding to positive samples a random edge between the two connected components.

-Bipartiteness dataset: this dataset consists of 1000 (20-node) graphs with 500 positive samples and 500 negative ones.

The positive samples correspond to bipartite graphs generated with an Erdös-Rényi (bipartite) model probability of p = 0.5.

For the negative samples (non-bipartite graphs) we chose the positive samples and for each of them we added an edge between randomly selected nodes from the same partition, in order to form odd cycles 3 .

-Triangle-freeness dataset: this dataset consists of 1000 (20-node) graphs with 500 positive samples and 500 negative ones.

The positive samples correspond to triangle-free graphs selected among randomly generated graphs with an Erdös-Rényi model probability of p = 0.1.

We constructed negative samples by randomly adding new edges to positive samples until it creates at least one triangle.

-Circular skip links: this dataset consists of 150 graphs of 41 nodes as described in (Murphy et al., 2019; Chen et al., 2019) .

The Circular Skip Links graphs are undirected regular graphs with node degree 4.

We denote a Circular skip link graph by G n,k an undirected graph of n nodes, where (i, j) ∈ E holds if and only if |i − j| ≡ 1 or k( mod n) This is a 10-class multiclass classification task whose objective is to classify each graph according to its isomorphism class.

Experimentation protocol: We evaluate the different configurations of CLIP and its competitors GIN and RP-GIN based on their hyper-parameters.

For the architecture implementation of the GIN, we followed the best performing architecture, presented in Xu et al. (2019) .

In particular, we used the summation as the aggregation operator, MLPs as the combination level for the node embedding generation and the sum operator for the readout function along with its refined version of concatenated graph representations across all iterations/layers of GIN, as described in Xu et al. (2019) .

In all the tested configurations for CLIP and its competitors (GIN, RP-GIN) we fixed the number of layers of the MLPs and the learning rate: we chose 2-layer MLPs and we used the Adam optimizer with initial learning rate of 0.001 along with a scheduler decaying the learning rate by 0.5 every 50 epochs.

Concerning the other hyper-parameters, we optimized: the number of hidden units within {16, 32, 64} (except for the CSL task where we only use 16 hidden units to be fair w.r.t.

RP-GIN

@highlight

This paper introduces a coloring scheme for node disambiguation in graph neural networks based on separability, proven to be a universal MPNN extension.