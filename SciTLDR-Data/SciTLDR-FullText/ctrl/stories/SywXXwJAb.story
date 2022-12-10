Formal understanding of the inductive bias behind deep convolutional networks, i.e. the relation between the network's architectural features and the functions it is able to model, is limited.

In this work, we establish a fundamental connection between the fields of quantum physics and deep learning, and use it for obtaining novel theoretical observations regarding the inductive bias of convolutional networks.

Specifically, we show a structural equivalence between the function realized by a convolutional arithmetic circuit (ConvAC) and a quantum many-body wave function, which facilitates the use of quantum entanglement measures as quantifiers of a deep network's expressive ability to model correlations.

Furthermore, the construction of a deep ConvAC in terms of a quantum Tensor Network is enabled.

This allows us to perform a graph-theoretic analysis of a convolutional network, tying its expressiveness to a min-cut in its underlying graph.

We demonstrate a practical outcome in the form of a direct control over the inductive bias via the number of channels (width) of each layer.

We empirically validate our findings on standard convolutional networks which involve ReLU activations and max pooling.

The description of a deep convolutional network in well-defined graph-theoretic tools and the structural connection to quantum entanglement, are two interdisciplinary bridges that are brought forth by this work.

A central factor in the application of machine learning to a given task is the restriction of the hypothesis space of learned functions known as inductive bias.

In deep convolutional networks, inductive bias manifests itself in architectural features such as number of layers, number of channels per layer, and more BID17 .

Formal understanding of the inductive bias behind convolutional networks is limited -the assumptions encoded into these models, which seem to form an excellent prior knowledge for different types of data (e.g. BID16 ; BID15 ; van den Oord et al. (2016) ), are for the most part a mystery.

An important aspect of the influence that a certain architectural feature has on the inductive bias, is its effect on the network's ability to model correlations between regions of its input.

In this regard, one typically considers partitions that divide input regions into disjoint sets, and asks how far the function realized by the network is from being separable with respect to these partitions BID5 Levine et al., 2017) .

For example, BID5 show that when separability is measured through the algebraic notion of separation-rank, deep Convolutional Arithmetic Circuits (ConvACs) BID7 support exponential (in network size) separation-ranks for certain input partitions, while being limited to polynomial separation-ranks for others.

ConvACs are a special class of convolutional networks, characterized by linear activations and product pooling, which served a key role in theoretical analyses of convolutional networks, in virtue of their algebraic structure.

In this work, we draw upon formal similarities between how physicists describe a system of manyparticles as a quantum mechanical wave function, and how machine learning practitioners map a high-dimensional input (e.g. image) to a set of output labels through a deep network.

In particular, we show that there is a structural equivalence between a function modeled by a ConvAC and a many-body quantum wave function, which relies on their underlying tensorial structure.

This allows employment of the well-established physical notion of quantum entanglement measures (Plenio and Virmani, 2007) , which subsumes other algebraic notions of separability such as the separation-rank mentioned above, for the analysis of correlations modeled by deep convolutional networks.

Importantly, quantum entanglement is used by physicists as prior knowledge to form compact representations of many-body wave functions in what is known as Tensor Networks (TNs) (Östlund and Rommer, 1995; Verstraete and Cirac, 2004; Vidal, 2008; BID11 .

In the domain of machine learning, a network in the form of a ConvAC is effectively a compact representation of a multi-dimensional array related to the convolutional weights.

This has been analyzed to date via tensor decompositions -where the representations are based on linear combinations of outer-products between lower-order tensors BID7 .

A TN, on the other hand, is a way to compactly represent a higher-order tensor through inner-products among lower-order tensors, which allows a natural representation of TNs through an underlying graph.

Although the fundamental language is different, we show that a ConvAC can be mapped to a TN, and thus a graph-theoretic setting for studying functions modeled by deep convolutional networks is brought forth.

In particular, notions of max-flow/min-cut are shown to convey important meaning.

The results we present, connect the inductive bias of deep convolutional networks to the number of channels in each layer, and indicate how these should be set in order to satisfy prior knowledge on the task at hand.

Specifically, the ability of a ConvAC to represent correlations between input regions is shown to be related to a min-cut over all edge-cut sets that separate the corresponding input nodes in the associated TN.

Such results enable one to avoid bottle-necks and adequately tailor the network architecture through application of prior knowledge.

Our results are theoretically proven for a deep ConvAC architecture; their applicability to a conventional deep convolutional network architecture, which involves ReLU activations and max pooling, is demonstrated through experiments.

Some empirical reasoning regarding the influence of the channel numbers on the network's performance has been suggested (e.g. Szegedy et al. (2016) ), mainly regarding the issue of bottle-necks which is naturally explained via our theoretical analysis below.

Such insights on the architectural design of deep networks are new to machine learning literature, and rely on TN bounds recently derived in physics literature, referred to as 'quantum min-cut max-flow' BID8 .

The mapping we present between ConvACs and TNs indicates new possibilities for the use of graphtheory in deep networks, where min-cut analysis could be just the beginning.

Additionally, the connections we derive to quantum entanglement and quantum TNs may open the door to further well-established physical insights regarding correlation structures modeled by deep networks.

The use of TNs in machine learning has appeared in an empirical context where Stoudenmire and Schwab (2016) trained a matrix product state (MPS) TN architecture to perform supervised learning tasks on the MNIST data-set.

Additionally, there is a growing interest in the physics community in RBM based forms for variational many-body wave functions (e.g. BID1 ).

BID2 present a theoretical mapping between RBMs and TNs which allows them to connect the entanglement bounds of a TN state to the expressiveness of the corresponding RBM.

We provide below the minimal tensor analysis background required for following the analyses of ConvACs and TNs that are carried out in this paper.

The core concept in tensor analysis is a tensor, which may be thought of as a multi-dimensional array.

The order of a tensor is defined to be the number of indexing entries in the array, which are referred to as modes.

The dimension of a tensor in a particular mode is defined as the number of values that may be taken by the index in that mode.

If A is a tensor of order N and dimension M i in each mode i ∈ [N ], its entries are denoted A d1...d N , where the index in each mode takes values between 1 and the appropriate dimension, d i ∈ [M i ].

Suppose A is a tensor of order N , and let (A, B) be a partition of [N ] := {1, . . .

, N }, i.e. A and B are disjoint subsets of [N ] whose union covers the entire set.

The matricization of A w.r.t.

the partition (A, B), denoted A A,B , is essentially the arrangement of the tensor elements as a matrix whose rows correspond to A and columns to B (see appendix A for exact definition).

A TN (see overview in Orús (2014) is a weighted graph, where each node corresponds to a tensor whose order is equal to the degree of the node in the graph.

Accordingly, the edges emanating out of a node, also referred to as its legs, represent the different modes of the corresponding tensor.

The weight of each edge in the graph, is equal to the dimension of the appropriate tensor mode.

conv 0 (j 0 , )Figure 2: The Convolutional Arithmetic Circuit (ConvAC) network BID7 .Moving on to the connectivity properties of a TN, edges which connect two nodes in the TN represent an operation between the two corresponding tensors.

A index which represents such an edge is called a contracted index, and the operation of contracting that index is a summation over all of the values it can take.

An index representing an edge with one loose end is called an open index.

The tensor represented by the entire TN, whose order is equal to the number of open indices, can be calculated by summing over all of the contracted indices in the network.

In FIG0 , a TN corresponding to the operation of multiplying a vector v ∈ R r1 by a matrix M ∈ R r2×r1 is depicted.

The computation is performed by summing over the only contracted index, k. Since there is only one open index, d, the result of contracting the network is an order 1 tensor (a vector): u ∈ R r2 which upholds u = M v. Though we use below the contraction of indices in a more elaborate TN, this operation can be essentially viewed as a generalization of matrix multiplication.

When describing the quantum mechanical properties of a system composed of many interacting particles, referred to as a many-body quantum system, physicists are required to employ functions which are able to express an elaborate relation between the different particles.

Similarly, machine learning tasks require functions with the ability to express a complex relation between many input elements, e.g. many pixels in an image.

In this section, we formulate this analogy.

Our construction will be based on the ConvAC architecture introduced by BID7 , illustrated in FIG17 .

The ConvAC is a deep convolutional network that operates similarly to a regular convolutional network, only with linear activations and product pooling layers (which introduce the non-linearity) instead of the more common non-linear activations (e.g. ReLU) and average/max pooling.

ConvACs are closely related to SimNets BID3 BID6 , and their underlying operations lend themselves to mathematical analyses based on measure theory and tensor analysis.

From an empirical perspective, ConvACs work well in many practical settings, e.g. for optimal classification with missing data (Sharir et al.) , and for compressed networks BID6 .

Importantly, through the concept of generalized tensor decompositions, a ConvAC can be transformed to a standard convolutional network with ReLU activation and average/max pooling, which laid the foundation for extending its proof methodologies to such ConvNets BID4 .

This deep learning architecture was chosen for our analysis below due to its underlying tensorial structure which resembles the quantum many-body wave function, as will soon be shown.

The input space of the network, denoted by X = (x 1 , ..., x N ), can be thought of as an image, where each x j corresponds to a local patch from that image.

The Y network outputs, denoted by h y (x 1 , ..., x N ) for y ∈ [Y ], are shown in BID7 to have the following form: DISPLAYFORM0 where A y and A (rank-1) are tensors of order N and dimension M in each mode.

The entries of the conv-weights tensor A y are given by polynomials in the network's convolutional weights, a l,j,γ (see FIG17 and BID7 ).

The entries of A (rank-1) are given by the application of the M linearly independent representation functions {f θ d } M d=1 on the input patches, which are an initial mapping of the inputs to an M -dimensional feature space.

We now turn to a brief presentation of the methods with which physicists describe the quantum mechanical properties of a many-body system (see appendix B for a more detailed introduction).A state of a system, which is a complete description of a physical system, is given in quantum mechanics as a wave function, denoted by |ψ .

We limit our discussion to states which reside in finite dimensional Hilbert spaces, as these are at the heart of our analogy to convolutional networks.

We discuss the case of N particles, each corresponding to a local Hilbert space H j for j ∈ [N ] such that ∀j : dim(H j ) = M .

Denoting an orthonormal basis of the local Hilbert space by {|ψ d } M d=1 , the many-body wave function |ψ ∈ H = ⊗ N j=1 H j can be written as: DISPLAYFORM1 where |ψ d1 ⊗ · · · ⊗ |ψ d N is a basis vector of the M N dimensional Hilbert space H, and the coefficients tensor A d1...d N is the tensor holding the corresponding coefficients.

We will tie between the function realized by a ConvAC given in eq. 1, and the many-body quantum wave function given in eq. 2.

First, we consider a special case of N particles which exhibit no quantum correlations (to be formulated in section 4 below).

The state of such a system is called a product state, and can be written down as a single tensor product of local states |φ j ∈ H j : |ψ ps = |φ 1 ⊗ · · · ⊗ |φ N .

By expanding each local state in the respective basis, DISPLAYFORM2 dj ψ dj , the product state assumes a form similar to eq. 2: DISPLAYFORM3 with the entries of its coefficients tensor given by: DISPLAYFORM4 dj .

If we compose each local state |φ j such that its projection on the local basis vector equals v DISPLAYFORM5 , then the inner product between the many-body quantum state |ψ and the tailored product state |ψ ps is equal to: DISPLAYFORM6 reproducing eq. 1 for a single class y, as A DISPLAYFORM7 by construction.

This result ties between the function realized by a convolutional network to that which a many-body wave function models.

Specifically, the conv-weights tensor is analogous to the coefficients tensor of the manybody wave function, while the input to the convolutional network is analogous to the constructed product state.

In the following sections, we will use this analogy to acquire means of analyzing the expressiveness of a convolutional network via the properties of its underlying tensor.

The structural connection between the many-body wave function and the function realized by a ConvAC, presented in the previous section, creates an opportunity to employ well-established physical insights and tools for analyzing the inductive bias of convolutional networks.

We present in this section the concept of quantum entanglement measures, and use it to motivate and extend previously suggested means for quantifying correlations of a deep convolutional network.

In BID5 ; Levine et al. (2017) , the algebraic notion of separation-rank is used as a tool for measuring correlations modeled by a function between two disjoint parts of its input.

Let f (·) be a function over x 1 . . .

x N , and let (A, B) be a partition of [N ] .

The separation-rank of f (·) w.r.t. (A, B) measures the strength of correlation that f (·) models between input elements corresponding to A ({x i } i∈A ) and those corresponding to B ({x j } j∈B ).

If f (·) is separable w.r.t. (A, B) , meaning there exist functions g(·) and h(·) such that f (x 1 , ..., x N ) = g((x i ) i∈A ) · h((x j ) j∈B ), then under f (·) there is absolutely no correlation between the inputs of A and those of B.1 In this case, the separation-rank is equal to 1 by definition.

In general, the separation rank of f (·) w.r.t. (A, B) is defined to be the minimal number of summands that together give f (·), where each summand is separable w.r.t. (A, B) .

Higher separation rank indicates larger deviation from separability, i.e. stronger interaction (correlation) modeled between sides of the partition.

2 The analysis of separation ranks allows control over the inductive bias when designing a deep network architecture -the network can be designed such that characteristic correlations in the input are modeled, i.e. partitions that split correlated regions have high separation ranks.

In the physics domain, special attention is given to the inter-particle correlation structure characterizing a many-body wave function, as it has broad implications regarding the physical properties of the examined system.

We present below the concept of quantum entanglement measures that is widely used by physicists as a quantifier of correlations in a many-body quantum system.

Remarkably, this approach for quantifying correlations is very similar to the above presented tool of the separation-rank, which in fact corresponds to a particular quantum entanglement measure.

Consider a partition of N particles labeled by integers [N ], which splits it into two disjoint subsystems A and B. Let H A and H B be the Hilbert spaces corresponding to particles in subsystems A and B, respectively.

In what is referred to as a 'Schmidt decomposition', the many-body quantum wave function in eq. 2 can be written as (see appendix B.1 for derivation): DISPLAYFORM0 where DISPLAYFORM1 are the singular values of the matricization A A,B , and DISPLAYFORM2 are r vectors in the bases of H A and H B , respectively, obtained by a singular value decomposition.

Eq. 5 represents the N particle wave function in terms of a sum of tensor products between two disjoint parts of it.

Each summand in eq. 5 is a separable state w.r.t.

the partition (A, B), which is analogous to the separable function in the above discussion.

Intuitively, as above, the more correlated two sides of a partition are, the more 'complicated' the function describing their relation should be.

Essentially, a measure of entanglement w.r.t.

the partition (A, B) is a quantity that represents the difference between the state in question and a state that is separable w.r.t.

this partition.

There are several different measures, such as the entanglement entropy (Vedral and Plenio, 1998) This method for quantifying quantum correlations can now be readily transferred into the machine learning domain.

Utilizing the structural analogy that was established in section 3, the measures of entanglement constitute an instrument for quantifying the correlations that a convolutional network can model.

Specifically, we've shown the conv-weights tensor to be analogous to the coefficients tensor of the many-body wave function, thus the entanglement measures can be analogously defined using the singular values of a matricization of the conv-weights tensor.

Since it was shown by BID5 that the separation-rank is equal to the rank of the matricized convweights tensor, it is precisely equal to the Schmidt number.

The analogy to physics suggests that correlation measures more sensitive than the separation rank may be borrowed, providing a more sensitive algebraic perspective on the hypothesis space of convolutional networks, which takes into account the relative magnitudes of A A,B 's non-zero singular values and not merely their number.

Physicists have a rich tool-set for exploiting knowledge regarding quantum entanglement measures for the design of computational representations of quantum wave functions.

We are now in a position to borrow such tools, and use them for the design of convolutional networks.

In particular, we will establish a relation between the correlations modeled by a ConvAC and the widths of its hidden layers, and make use of these relations for controlling the inductive bias of the network.

In the previous section, we have seen that the coefficients or conv-weights tensor A d1...d N , which has M N entries, encapsulates the information regarding the correlations of the many-body quantum wave function or of the function realized by a ConvAC.

The curse of dimensionality manifests itself in the exponential dependence on the number of particles or image patches.

In a quantum many-body setting, this renders impractical the ability to investigate or even store a wave function of more than a few dozens of interacting particles.

A common tool used to tackle this problem in the physics community is a Tensor Network, which allows utilizing prior knowledge regarding correlations when attempting to represent an exponentially complicated wave function with a polynomial amount of resources.

In appendix C we provide a thorough introduction to TNs, which were briefly introduced in section 2.

In this section, we draw inspiration from the physics approach and present a construction of a ConvAC as a TN.

This construction will allow us to demonstrate how adequately tailoring the number of channels in each layer of the deep network can enhance its expressivity by fitting the form of the function realized by it to given correlations of the input.

In this we show how the parameters of the ConvAC can be most efficiently distributed given prior knowledge on the nature of the input, which is in fact matching the inductive bias to the task at hand.

FIG17 , with pooling windows of size 2 and N = 8 (see appendix D for full details of construction).

The round (order 2) nodes in each layer represent matrices holding the convolutional weights of that layer.

The triangle nodes correspond to a special tensor that hosts 1's on its super-diagonal and 0's elsewhere, effectively enforcing the same channel pooling attribute of the network.

The tensor A y d1...d N is obtained upon summation over all the indices which correspond to internal edges, leaving the external edges which correspond to d 1 , ..., d N , y open.

As mentioned above, a TN is a weighted graph, and the weights marked next to each edge in this TN are equal by construction to the number of channels in the corresponding ConvAC layer l, denoted r l .

This last equivalence will allow us to draw a direct relation between the number of channels in each layer of a deep ConvAC and the functions it is able to model.

Accordingly, it will allow us to provide prescriptions regarding the layer widths for the design of a network that is meant to support known input correlations.

Our main result, presented in theorem 1, relies on one of the most recent advances in the study of the quantitative connection between quantum entanglement and TNs, namely 'quantum min-cut max-flow' BID8 .

The key accomplishment that the TNs tool brings forth, is the ability to apply graph-theoretic tools to a deep convolutional network.

Specifically, we tie the network's ability to model correlations between two disjoint input regions A and B, as measured by the Schmidt entanglement measure, to the minimal value, over all cuts separating A from B, of the multiplication of the cut edges' weights (the multiplicative minimal cut).

FIG17 with pooling windows of size 2.

Assume that the channel numbers across the layers are all powers of the same integer, 5 and suppose we randomize the network weights by some continuous distribution.

Then, with probability 1, the rank of the matricization A y A,B (the Schmidt measure w.r.t. (A, B) ) is equal to the multiplicative minimal cut separating A from B in the respective TN.Theorem 1 leads to practical implications regarding the construction of a deep network architecture when there is prior knowledge on the task at hand.

If one wishes to construct a deep ConvAC that is expressive enough to model an intricate correlation structure according to some partition, it is advisable to choose the channel numbers such that the network is able to support such correlations, by ensuring that all the cuts separating these two parts in the corresponding TN have high weights.

For example, consider the left-right partition in which A and B hold the left and right input patches, respectively.

The multiplicative minimal cut weight is in this case equals min(r L−1 , r L−2 , ..., r DISPLAYFORM0 , where L := log 2 N (in the example given in FIG2 , L = 3).

We see that choosing a small number of channels for the deeper layers can create an undesired 'shortcut' which harms the expressiveness of the network in a way that prevents it from modeling the long ranged correlations which correspond to this partition, present for example in symmetric face images.

Alternatively, considering the interleaved partition where A and B hold the odd and even input patches, respectively, the multiplicative minimal cut weight will be equal to min(r N/4 0 , M N/2 ) -dependent only on the first layers' channel numbers, and exponential in N .

The partitions mentioned above represent two extreme cases that correspond to shortest and longest ranged correlations.

However, the min-cut result applies to any partition of the inputs, so that conclusions regarding the layer widths can be established for any intermediate length-scale of correlations.

For example, the relevant factors that contribute to the min-cut between (A, B) for which both A and B have contiguous segments of a certain length ξ are M, r 0 , ..., r log 2 ξ .

This is in fact a generalization of the treatment above with ξ = 1 for the interleaved partition and ξ = N/2 for the left-right partition, and can be understood by flow considerations in the graph underlying the TN: a cut that is located above a certain sub-branch can not assist in cutting the flow between A and B vertices that reside within that sub-branch.

Thus, the addition of more parameters to layers l such that l > log 2 ξ would result in an increase of the capacity of edges in the TN which will not belong to the min-cut.

The observation presented in the previous paragraph has practical implications.

For a data-set with features of a characteristic size D (e.g. in a two-dimensional digit classification task, D could be the size of digits that are to be classified), such partitions of length scales ξ < D are guaranteed to separate between different parts of a feature placed in any input location.

In order to classify a feature correctly, an elaborate function modeling a strong dependence between different parts of it must be realized by the network.

As discussed above, this means that a high measure of entanglement w.r.t.

partitions that separate the feature must be supported by the network, and theorem 1 allows us to describe this measure of entanglement in terms of a min-cut in the TN graph.

The following 'rule of thumb' is thus implied -the channel numbers up to layer l = log 2 D are more important than those of deeper layers, therefore it is advisable to concentrate more parameters (in the form of more channels) in these levels.

Additionally, an analysis of the min-cut in the ConvAC TN shows that among the more important layers l = 1, ..., log 2 D, deeper ones need to be wider, as is apparent for example in the above expression of the minimal cut weight for the high-low partition.

In a more general task it may be hard to point out a single most important length scale D, however the conclusions presented in this section can be viewed as an incentive to develop adequate means of characterizing the most relevant data correlations for different tasks.

The min-cut analysis on the TN representing a deep ConvAC translates prior knowledge on how correlations among input variables (e.g. image patches) are modeled, into the architectural design of number of channels per layer in a ConvAC.

In this section, we demonstrate empirically that the theoretical findings established above for the deep ConvAC, apply to a regular convolutional network architecture which involves the more common ReLU activations and average or max pooling.

Two tasks were designed, one with a short characteristic length to be referred to as the 'local task', and the other with a long characteristic length to be referred to as the 'global task'.

Both tasks are based on the MNIST data-set and consist of 64 × 64 black background images on top of which resized binary MNIST images were placed in random positions, to make sure we account for correlation distances without biasing towards a particular location in the image.

For the local task, the digits were shrunken to 8 × 8 images while for the global task they were enlarged to size 32 × 32.

Note that both tasks are more challenging than the standard MNIST task, and that the local task is even more challenging than the global one.

We designed two network architectures that tackle these two tasks, with a difference in the channel ordering scheme.

Each architecture was designed to better match the correlation structure of one of the above tasks, in accordance with the analysis presented in the previous section.

In both networks, the first layer is a representation layer -a 3 × 3 (stride 1) shared convolutional layer.

Following it are 6 hidden layers, each with 1 × 1 shared convolution kernels followed by ReLU activations and 2×2 max pooling (stride 2).

Classification in both networks was performed through Y = 10 outputs, with prediction following the strongest activation.

The difference between the two networks is in the channel ordering -in the wide-base (WB) network they are wider in the beginning and narrow down in the deeper layers, while in the wide-tip (WT) network they follow the opposite trend.

Specifically, we set a parameter r to determine each pair of such networks according to WB: [10; 4r; 4r; 2r; 2r; r; r; 10] and WT: [10; r; r; 2r; 2r; 4r; 4r; 10] (The channel numbers from left to right go from shallow to deep).

According to the above conclusions, this choice for increase of widths towards deeper layers in the WT network makes it fit the global task in which all layers are important.

similarly, the conclusions dictate that the choice WB network makes it fit the local task, in which shallower layers are more important.

The specific channel arrangement ensures that the amount of learned parameters for both configurations is equal.

Figure 4 : Applying ReLU networks with max pooling to the global and local classification tasks.

Fig. 4 shows the results of applying both the WB and WT networks to the local and global tasks.

Each task consisted of 60000 training images and 10000 test images, in correspondence with the MNIST database.

Indeed, the WB network significantly outperforms the WT network in the local task, whereas a clear opposite trend can be seen for the global task.

This complies with our theoretical analysis, according to which the WB network which holds more parameters in shallow layers should be able to support short correlation lengths of the input, whereas the WT network in which deeper layers are wider is predicted to put focus on longer correlation lengths.

The fact that the global task gets higher accuracies for all choices of r is unsurprising, as it is clearly an easier task.

Overall, these experiments constitute a demonstration of how prior knowledge regarding a task at hand may be used to tailor the inductive bias of a deep convolutional network by appropriately designing layer widths.

We have shown how phenomena that were indicated by the theoretical analysis that was presented in this paper in the context of ConvACs, manifest themselves on the most prevalent and successful convolutional network architectures (ReLU activation, max pooling).

The construction of a deep ConvAC in terms of a TN brought forth the main theoretical achievements of this paper.

This method enabled us to carry a graph-theoretic analysis of a convolutional network, and tie its expressiveness to a minimal cut in the graph characterizing it.

Our construction began with a structural equivalence between the function realized by a ConvAC and a quantum many-body wave function.

This facilitated the transfer of mathematical and conceptual tools employed by physicists, such as the tool of TNs and the concept of 'entanglement measures', providing well-defined quantifiers for a deep network's expressive ability to model correlations between regions of its input.

By employing these tools, we were able to present theoretical observations regarding the role that the number of channels in each layer fulfills in the overall expressiveness of a deep convolutional network, and how they affect its ability to model given input correlations.

Furthermore, practical implications were presented for the construction of a deep network architecture when there is prior knowledge regarding the input correlations.

Apart from the direct results discussed above, two important interdisciplinary bridges emerge from this work.

The results we drew between min-cut in the graph representation of a ConvAC to network expressivity measures, may constitute an initial example for employing the connection to TNs for the application of graph-theoretic measures and tools to the analysis of the function realized by a deep convolutional network.

The second bridge, is the mathematical connection between the two fields of quantum physics and deep learning.

The field of quantum TNs is a rapidly evolving one, and the established construction of a successful deep learning architecture in the language of TNs may allow applications and insights to be transferred between the two domains.

For example, the tree shaped TN that was shown in this work to be equivalent to a deep convolutional network, has been known in the physics community for nearly a decade to be inferior to another deep TN architecture by the name of MERA (Vidal, 2008) , in its expressiveness and in its ability to model correlations.

The MERA TN constitutes an exemplar case of how the TNs/deep-learning connection established in this work allows a bi-directional flow of tools and intuition.

MERA architecture introduces over-laps by adding 'disentangling' operations prior to the pooling operations, which, in translation to terms of deep learning, effectively mix activations that are intended to be pooled in different pooling windows.

Physicists have a good grasp of how these specific overlapping operations allow a most efficient representation of functions that exhibit high correlations at all length scales (Vidal, 2007) .

Accordingly, a new view of the role of overlaps in the high expressivity of deep networks as effectively 'disentangling' intricate correlations in the data can be established.

In the other direction, as deep convolutional networks are the most empirically successful machine learning architectures to date, physicists may benefit from trading their current 'overlaps by disentangling' scheme to the use of overlapping convolutional windows (proven to contribute exponentially to the expressive capacity of neural networks by Sharir and Shashua FORMULA1 ), in their search for expressive representations of quantum wave functions.

Overall, We view this work as an exciting bridge for transfer of tools and ideas between fields, and hope it will reinforce a fruitful interdisciplinary discourse.

DISPLAYFORM0

We provide below a short introduction to the notation used by physicists when describing quantum mechanical properties of a many-body system.

We follow relevant derivations in Preskill (1998) and Hall FORMULA1 , referring the interested reader to these sources for a more comprehensive mathematical introduction to quantum mechanics.

A state of a system, which is a complete description of a physical system, is given in quantum mechanics as a ray in a Hilbert space (to be defined below).

Relevant Hilbert spaces in quantum mechanics are vector spaces over the complex numbers.

We restrict our discussion to vector spaces over R, as the properties related to complex numbers are not required for our analysis and do not affect it.

Physicists denote such vectors in the 'ket' notation, in which a vector ψ is denoted by: |ψ ∈ H. The Hilbert space H has an inner product denoted by φ|ψ , that maps a pair of two vectors in H to a scalar.

This inner product operation is also referred to as 'projecting |ψ onto |φ '.

A ray is an equivalence class of vectors that differ by multiplication by a nonzero scalar.

For any nonzero ray, a representative of the class, |ψ , is conventionally chosen to have a unit norm: ψ|ψ = 1.

A 'bra' notation φ|, is used for the 'dual vector' which formally is a linear mapping between vectors to scalars defined as |ψ → φ|ψ .

We can intuitively think of a 'ket' as a column vector and 'bra' as a row vector.

Relevant Hilbert spaces can be infinite dimensional or finite dimensional.

We limit our discussion to quantum states which reside in finite dimensional Hilbert spaces, as these lie at the heart of our analogy to convolutional networks.

Besides being of interest to us, these spaces are extensively investigated in the physics community as well.

For example, the spin component of a spinful particle's wave function resides in a finite dimensional Hilbert space.

One can represent a general single particle state |ψ ∈ H1, where dim(H1) = M , as a linear combination of some orthonormal basis vectors: DISPLAYFORM0 where v ∈ R M is the vector of coefficients compatible with the basis {|ψ d } M d=1 of H1, each entry of which can be calculated by the projection: DISPLAYFORM1 The extension to the case of N particles, each with a wave function residing in a local finite dimensional Hilbert space Hj for j ∈ [N ] (e.g. N spinful particles), is readily available through the tool of a tensor product.

In order to define a Hilbert space which is the tensor product of the local Hilbert spaces: H := ⊗ N j=1 Hj, we will specify its scalar product.

Denote the scalar product in each Hj by ·|· j , then the scalar product in the tensor product finite dimensional Hilbert space DISPLAYFORM2 For simplicity, we set the dimensions of the local Hilbert spaces Hj to be equal for all j, i.e. ∀j : dim(Hj) = M .

In the spin example, this means that the particles have the same spin, e.g. for N electrons (spin 1/2), M = 2.

Denoting as above the orthonormal basis of the local Hilbert space by DISPLAYFORM3 , the many-body quantum wave function |ψ ∈ H = ⊗ N j=1 Hj can be written as: DISPLAYFORM4 Reproducing eq. 2.

Consider a partition of the above described system of N particles labeled by integers [N ] := {1, . . .

, N }, which splits it into two disjoint subsystems DISPLAYFORM0 DISPLAYFORM1 where DISPLAYFORM2 and {|ψ DISPLAYFORM3 are bases for H A and H B , respectively, 9 and A A,B is the matricization of A w.r.t.

the partition (A, B).

Let us denote the maximal rank of A A,B by r := min(dim(H A ), dim(H B )).

A singular value decomposition on A A,B results in the following form (also referred to as the Schmidt decomposition): DISPLAYFORM4 where λ1 ≥ · · · ≥ λr are the singular values of A A,B , and DISPLAYFORM5 are r vectors in new bases for H A and H B , respectively, obtained by the decomposition.

It is noteworthy that since |ψ is conventionally chosen to be normalized, the singular values uphold α |λα| 2 = 1, however this constraint can be relaxed for our needs.

A Tensor Network (TN) is formally represented by an underlying undirected graph that has some special attributes, we elaborate on this formal definition in appendix E.1.

In the following, we give a more intuitive description of a TN, which is nonetheless exact and required for our construction of the ConvAC TN.

The basic building blocks of a TN are tensors, which are represented by nodes in the network.

The order of a tensor represented by a node, is equal to its degree -the number of edges incident to it, also referred to as its legs.

FIG6 (a) shows three examples: 1) A vector, which is a tensor of order 1, is represented by a node with one leg.

2) A matrix, which is a tensor of order 2, is represented by a node with two legs.

3) Accordingly, a tensor of order N is represented in the TN as a node with N legs.

In a TN, each edge is associated with a number called the bond dimension.

The bond dimension assigned to a specific leg of a node, is simply the dimension of the corresponding mode of the tensor represented by this node (see definitions for a mode and its dimension in section 2).A TN is a collection of such tensors represented by nodes, with edges that can either be connected to a node on one end and loose on the other end or connect between two nodes.

Each edge in a TN is represented by an index that runs between 1 and its bond dimension.

An index representing an edge which connects between two tensors is called a contracted index, while an index representing an edge with one loose end is called an open index.

The set of contracted indices will be denoted by K = {k1, ..., kP } and the set of open indices will be denoted by D = {d1, ..., dN }.

The operation of contracting the network is defined by summation over all of the P contracted indices An example for a contraction of a simple TN is depicted in FIG6 .

There, a TN corresponding to the operation of multiplying a vector v ∈ R r 1 by a matrix M ∈ R r 2 ×r 1 is performed by summing over the only contracted index, k. As there is only one open index, d, the result of contracting the network is an order 1 tensor (a vector): u ∈ R r 2 which upholds u = M v. In FIG6 (c) a somewhat more elaborate example is illustrated, where a TN composed of order 2 and 3 tensors represents a tensor of order 5.

This network represents a decomposition known as a tensor train (Oseledets (2011)) in the tensor analysis community or a matrix product state (MPS) (see overview in e.g. Orús (2014)) in the condensed matter physics community, which arranges order 3 tensors in such a 'train' architecture and allows the representation of an order N tensor with a linear (in N ) amount of parameters.

The MPS exemplifies a typical desired quality of TNs.

The decomposition of a higher order tensor into a set of sparsely interconnected lower order tensors, was shown (Oseledets and Tyrtyshnikov (2009); BID0 ) to greatly diminish effects related to the curse of dimensionality discussed above.

8

A ⊗ H B with equality obtained upon a permutation of the local spaces that is compliant with the partition (A, B) .9 It is possible to write ψ hidden layer 0 DISPLAYFORM0 Figure 6: The original Convolutional Arithmetic Circuits network as presented by BID7 .

We begin by reviewing tensor decompositions of the conv-weights tensor shown in BID7 to be equivalent to the shallow and deep versions of the ConvAC network given in the main text and reproduced for convenience in fig. 6 .The CP decomposition of the conv-weights tensor corresponds to a ConvAC depicted in fig. 6 with one hidden layer, which collapses the entire spatial structure through global pooling -a shallow ConvAC.

Explicitly, the CP decomposition of the order N conv-weights tensor of a specific class y is a sum of rank-1 tensors, each of which is attained by a tensor product of N weights vectors: DISPLAYFORM0 where DISPLAYFORM1 The deep version of fig. 6 , where the pooling windows between convolutional layers are of minimal size, corresponds to a specific tensor decomposition of A y , which is a restricted version of a hierarchical Tucker decomposition, referred to in short as the HT decomposition.

The restriction is related to the fact that the pooling scheme of the ConvAC architecture presented in fig. 6 involves only entries from the same channel, while in the general HT decomposition pooling operations would involve entries from different channels.

For brevity of notation, we will present the expressions for a scenario where the input patches are aligned along a one-dimensional line (can also correspond to a one-dimensional signal, e.g. sound or text), and the pooling widows are of size 2.

The extension to the two-dimensional case follows quite trivially, and was presented in BID5 .

Under the above conditions, the decomposition corresponding to a deep ConvAC can be defined recursively by BID7 ): DISPLAYFORM2 The decomposition in eq. 11 recursively constructs the conv-weights tensor {A y } y∈ [Y ] by assembling vectors DISPLAYFORM3 in an incremental fashion.

This is done in the form of tensor products, which are the natural form for tensor decompositions.

The index l stands for the level in the decomposition, corresponding to the l th layer of the ConvAC network given in fig. 6 .

j represents the 'location' in the feature map of level l, and γ corresponds to the individual tensor in level l and location j.

The index r l is referred to as level-l rank, and is defined to be the number of tensors in each location of level l (we denote for completeness rL := Y ).

In the ConvAC network given in fig. 6 , r l is equal to the number of fig. 6 in its shallow form, i.e. with one hidden layer followed by a global pooling operation which collapses the feature maps into Y different class scores.

The matrices A (j) hold the convolutional weights of the hidden layer and the matrix G holds the weights of the final dense layer.

The central δ tensor effectively enforces the same channel pooling, as can be seen by its form in eq. 12 and its role in the calculation of this TN given in eq. 13.

channels in the l th layer -this is important in our analysis of the role played by the channel numbers.

The tensor φ l,j,γ is of order 2 l , and we assume for simplicity that N -the order of A y , is a power of 2 (this is merely a technical assumption also made in Hackbusch (2012), it does not limit the generality of the analysis).

The parameters of the decomposition are the final level weights {a DISPLAYFORM4 , the intermediate levels' DISPLAYFORM5 , and the first level weights {a DISPLAYFORM6

In order to construct the TN equivalent of the shallow ConvAC, we define the order N +1 tensor δ ∈ R K×···×K , referred to as the δ tensor, as follows: DISPLAYFORM0 with kj ∈ [K] ∀j ∈ [N + 1], i.e. its entries are equal to 1 only on the super-diagonal and are zero otherwise.

We shall draw nodes which correspond to such δ tensors as triangles in the TN, to remind the reader of the restriction given in eq. 12.

Let G ∈ R Y ×K be a matrix holding the convolutional weight vector of the final layer v y ∈ R K in its y th row and let A (j) ∈ R K×M be a matrix holding the convolutional weights vector a k,j ∈ R M in its k th row.

One can observe that per class y, the k th summand in eq. 10 is equal to the tensor product of the N vectors residing in the k th rows of all the matrices A (j) , j ∈ [N ], multiplied by a final weight associated with class y. Tensors represented by nodes in the TN will have parenthesis in the superscript, which denote labels such as the position j in the above, to differentiate them from 'real' indices that must be taken into consideration when contracting the TN.

Per convention, such 'real' indices will be written in the subscript.

Having defined the above, the TN equivalent of the CP decomposition is illustrated in FIG7 .

Indeed, though they represent the same exact quantity, the form of eq. 10 isn't apparent at a first glance of the network portrayed in FIG7 .

Essentially, the TN equivalent of the CP decomposition involves contractions between the matrices A (j) , G, and the δ tensor, as can be seen in the expression representing it: DISPLAYFORM1 The role of the δ tensor in eq. 13 can be observed as 'forcing' elements of the k th row of any matrix A (j) to be multiplied only by elements of k th rows of the other matrices which in effect enforces same channel pooling.

10 If one were to switch the δ k 1 ...k N in eq. 13 by a general tensor G k 1 ...k N ∈ R K×···×K , a TN equivalent of an additional acclaimed decomposition would be attained, namely the Tucker decomposition.

Similar to other tensor decompositions, the Tucker decomposition is more commonly given in an outer product form: DISPLAYFORM0

We describe below a TN corresponding to the deep ConvAC calculation, given by eq. 1.

The ConvAC calculation is constructed as an inner-product between two tensors: the conv-weights tensor A DISPLAYFORM0 which is given in eq. 11 in terms of a tensor decomposition, and A Considering the upper block, it is worth noting that it is not a sketch of a TN but the actual full description compliant with the graph notations described in appendix C. Accordingly, the two legged nodes represent matrices, where each matrix A (l,j) ∈ R r l ×r l−1 (with r−1 := M ) is constructed such that it holds the convolutional weight vector a l,j,γ ∈ R r l−1 , γ ∈ [r l ] in its γ th row.

The triangle node appearing between levels l − 1 and l represents an order 3 tensor δ ∈ R r l−1 ×r l−1 ×r l−1 , obeying eq. 12.

The δ tensor is the element which dictates the same channel pooling in this TN construction.

As mentioned above, the lower block in FIG8 is the TN representing A assumes is exactly the form that the coefficients tensor of a product state assumes when represented as a TN.

As can be seen in FIG8 , a final contraction of the indices d1, ..., d8 results in the class scores vector calculated by the ConvAC, hy(x1, ..., x8).The calculation performed by a one-dimensional ConvAC for a general N (s.t.

log 2 N ∈ N), is given by the recursively defined TN representation shown in FIG12 .

v (l,j) ∈ R r l−1 is a vector of actual activations generated during a computation across in the l th level of the network shown in fig. 6 .

Recall that r−1 := M , and that DISPLAYFORM1 is a vector in the representation layer (see FIG8 ).

To demonstrate that this TN indeed defines the calculations performed by a ConvAC, we conjecture that the equality in FIG12 holds, namely that for DISPLAYFORM2 where d ∈ [r l−1 ].

In the first line of eq. 14 we simply followed the TN prescription given in appendix C and wrote a summation over all of the contracted indices in the left hand side of FIG12 , and in the second line we used the definition of matrix multiplication.

According to the construction of A (l,j) given in appendix D.2, DISPLAYFORM3 where the weights vector a l,j,γ ∈ R r l−1 was introduced in eq. 11 .

Thus, eq. 14 is reduced to: DISPLAYFORM4 Finally, by definition of the δ tensor, the sum vanishes and we obtain the required expression for the operation of the ConvAC: DISPLAYFORM5 where an activation in the d th feature map of the l th level holds the multiplicative pooling of the results of two activation vectors from the previous layer convolved with the d th convolutional weight vector for that layer.

Applying this procedure recursively is exactly the conv→pool→ ... →conv→pool scheme that lies at the heart of the ConvAC operation ( fig. 6 ).

Recalling that rL := Y , the output of the network is given by: DISPLAYFORM6 To conclude this section, we have presented a translation of the computation performed by a ConvAC to a TN.

The convolutional weights are arranged as matrices (two legged nodes) placed along the network, and the same channel pooling characteristic is made available due to three legged δ tensors in a deep network, and an N + 1 legged δ tensor in a shallow network.

Finally, and most importantly for our analysis, the bond dimension of each level in the TN representing the ConvAC is equal to r l , which is the number of feature maps (i.e. the number of channels) comprising that level in the corresponding ConvAC architecture.

Below we provide upper and lower bounds on the ability of a deep ConvAC to model correlations of its inputs, as measured by the Schmidt entanglement measure (see section 4 for definition).

We address a general setting FIG0 : The components comprising a 'ConvAC-weights TN' φ that describes the weights tensor A y of a ConvAC, are an undirected graph G(V, E) and a bond dimensions function c. The bond dimension is specified next to each edge e ∈ E, and is given by the function c(e).

As shown in appendix D.2, the bond dimension of the edges in each layer of this TN is equal to the number of channels in the corresponding layer in the ConvAC.

The node set in the graph G(V, E) presented above decomposes to V = V tn · ∪ V inputs , where V tn (grey) are vertices which correspond to tensors in the ConvAC TN and V inputs (blue) are degree 1 vertices which correspond to the N open edges in the ConvAC TN.

The vertices in V inputs are 'virtual' -were added for completeness, so G can be viewed as a legal graph.

The open edge emanating from the top-most tensor (marked by a dashed line) is omitted from the graph, as it does not effect our analysis below -no flow between any two input groups can pass through it.of the number of channels in a deep ConvAC.

The result stated in theorem 1, which applies when all of the channel numbers in a deep ConvAC architecture are powers of some integer, is implied (specifically by the equality of the upper bound in claim 1 and the lower bound in lemma 2 below).

We begin by presenting a description of the TN as a 'legal' graph in section E.1 and move on to prove the bounds in sec E.2.

The ability to represent a deep convolutional network (ConvAC) as a 'legal' graph, is a key accomplishment that the Tensor Networks tool brings forth.

Our main results rely on this graph-theoretic description and tie the expressiveness of a ConvAC to a minimal cut in the graph characterizing it, via the connection to quantum entanglement measures.

This is in fact a utilization of the 'Quantum min-cut max-flow' concept presented by BID8 .

Essentially, the quantum max-flow between A and B is a measure of the ability of the TN to model correlations between A and B, and the quantum min-cut is a quantity that bounds this ability and can be directly inferred from the graph defining it -that of the corresponding TN.We focus on the TN that describes A To turn φ into a graph we do the following.

First, we remove the open edge associated with the output.

As our analysis is going to be based on flow between groups of input vertices, no flow can pass through that open edge therefore removing it does not influence our analysis.

Second, we add N virtual vertices incident to the open edges associated with the input.

Those virtual vertices are the only vertices whose degree is equal to 1 (see FIG0 ).

The TN φ is now described below using graph terminology:• An undirected graph G(V, E), with a set of vertices V and a set of edges E. The set of nodes is divided into two subsets V = V tn · ∪ V inputs , where V inputs are the N degree-1 virtual vertices and V tn corresponds to tensors of the TN.

• A function c : E → N, associating a number r ∈ N with each edge in the graph, that equals to the bond dimension of the corresponding edge in the TN.Having described the object representing the ConvAC-weights TN φ, let us define an edge-cut set with respect to a partition of the N nodes of V inputs , and then introduce a cut weight associated with such a set.

An edge-cut set with respect to the partition V A · ∪V B = V inputs is a set of edges C s.t.

there exists a partitionṼ DISPLAYFORM0 We note that this is a regular definition of an edge-cut set in a graph G with respect to the partition of vertices (V A , V B ).

Let C = {e1, ..., e |C| } be such a set, we define its multiplicative cut weight as: DISPLAYFORM1 The weight definition given in eq. 18 is simply a multiplication of the bond dimensions of all the edges in a cut.

FIG0 shows a pictorial demonstration of this weight definition, which is at the center of our results to come.

In the following section, we use a max-flow / min-cut analysis on φ to obtain new results on the expressivity of the corresponding deep ConvAC via measures of entanglement w.r.t.

a bi-partition of its input patches, which are related to the number of channels in each layer of the ConvAC.

In claim 1 below, we provide an upper bound on the ability of a deep ConvAC to model correlations of its inputs, as measured by the Schmidt entanglement measure (see section 4).

This claim is closely related to attributes of TNs that are known in different forms in the literature.

is no greater than: minC WC , where C is a cut w.r.t (V A , V B ) and WC is the multiplicative weight defined by eq. 18.Proof.

We will use the example shown in FIG0 (a) of a general TN with arbitrary connectivity.

The edges of the TN φ are marked by the index associated with them.

Any index p ∈ {d, k} runs between 1 and its bond dimension marked by cp, which upholds cp := c(ep) where ep ∈ E is the edge associated with the index p.

For the given partition (A, B), denote A = {a1, ..., a |A| } , B = {b1, ..., b |B| } and let IA · ∪ IB = {d1, . . .

, dN } be the corresponding partition of external indices, where IA = {da 1 , ..., da |A| } and IB = {d b 1 , ..., d b |B| }.

A and H B be the spaces corresponding to the different configurations of the indices in IA and IB, respectively, their dimensions given by: DISPLAYFORM0 In the example shown in FIG0 , the graph is arranged s.t.

A is on the left and B is on the right.

The marked cut C that separates between A and B is arbitrarily chosen as a representative cut, and we denote the indices of the cut edges by IC = {k1, ..., k |C| }.

It is noteworthy, that any index ki in the cut is allowed to be an external index, i.e. the cut is allowed to contain any amount of external edges.

Now, two contractions can be performed, separately contracting all the tensors to the left of the cut and to the right of it.

We are left with two higher order tensors, FIG0 ).

DISPLAYFORM1 According to the TN in FIG0 , the matricization A A,B can be written as a multiplication of two matrices.

Component wise, this can be written as: DISPLAYFORM2 where any amount of cut indices that are also external indices translate as blocks of the identity matrix on the diagonal.

Finally, since this construction is true for any cut C w.r.t (A, B) , the rank of A A,B upholds: rank( A A,B ) ≤ minC WC , satisfying the claim for any general TN, and specifically for the ConvAC TN.The upper bound provided above, alerts us when a deep ConvAC is too weak to model a desired correlation structure, according to the number of channels in each layer.

Below, we provide a lower bound similar in spirit to a bound shown in BID8 .

Their claim is applicable for a TN with general tensors (no δ tensors), and we adapt it to the ConvAC-weights TN (that has δ tensors) which in effect ensures us that the entanglement measure cannot fall below a certain value for any specific arrangement of channels per layer.

Theorem 2 above implies that the upper bound given in Claim 1 is saturated when all of the channel numbers in a deep ConvAC architecture are powers of some integer p.

For a general arrangement of channel numbers, the upper bound is not tight and theorem 2 guarantees that the rank will not be lower than that of any ConvAC architecture with channel numbers which are powers of some integer p yet are not higher than the original ConvAC channel numbers.

Even though this is the lower bound we prove, we have a reason to believe the actual lower bound is much tighter.

In section G, we show simulations which indicate that deviations from the upper bound are actually quite rare and unsubstantial in value.

In the following we prove theorem 2.

Our proof strategy is similar to the one taken in Cui et al. FORMULA1 , however we must deal with the restricted δ tensors present in the network corresponding to a ConvAC (the triangle nodes in FIG8 ).

We first quote and show a few results that will be of use to us.

We begin by quoting a claim regarding the prevalence of the maximal matrix rank for matrices whose entries are polynomial functions -claim 2.

Next, we quote a famous graph theory result known as the Undirected Menger's Theorem (Menger (1927), BID10 , BID12 ) which relates the number of edge disjoint paths in an undirected graph to the cardinality of the minimal cut -theorem 3.

After this, we show that the rank of matricization of the tensor represented by φ p that is defined in theorem 2, is a lower bound on the rank of matricization of the tensor represented by φ -lemma 1.

Then, we prove that the upper bound in claim 1 is tight when all of the channel numbers are any powers of the same integer p ∈ N -lemma 2 (effectively showing theorem 1).

Finally, when all the preliminaries are in place, we show how the result in theorem 2 is implied.

DISPLAYFORM3 If there exists a point x ∈ R K s.t.

rank(A(x))

≥ r, then the set {x ∈ R K : rank(A(x)) < r} has zero measure (w.r.t.

the Lebesgue measure over R K ).Proof.

See Sharir et al..

Claim 2 implies that it suffices to show an assignment of the ConvAC network weights achieving a certain rank of matricization of the conv-weights tensor, in order to show this is the rank for all configurations of the network weights but a set of Lebesgue measure zero.

Essentially, this means that it is enough to provide a specific assignment that achieves the required bound in theorem 2 in order to prove the theorem.

Next, we present the following well-known graph theory result:Theorem 3. (Menger (1927), BID10 , BID12 ) [Undirected Menger's Theorem] Let G = (V, E) be an undirected graph with a specified partition (A, B) of the set of degree 1 vertices.

Let M F (G) be the maximum number of edge disjoint paths (paths which are allowed to share vertices but not edges) in G connecting a vertex in A to a vertex in B. Let M C(G) be the minimum cardinality of all edge-cut sets between A and B. Then, M F (G) = M C(G).Proof.

See e.g. BID8 .Theorem 3 will assist us in the proof of lemma 2.

We will use it in order to assert the existence of edge disjoint paths in an auxiliary graph ( FIG0 ), which we eventually utilize in order to provide the required weights assignment in lemma 2.

Next, we show lemma 1, which roughly states that a tensor which 'contains' another tensor in some sense will not have a lower matricization rank than that of the 'contained' tensor.

fig. 6 .

Let φ be the TN corresponding to this ConvAC network, and let φ p be a TN with the same connectivity as φ, where all of the bond dimensions are modified to be equal the closest power of p to their value in φ from below.

Let (A p ) y be the tensor represented by φ p and let there exist an assignment of all of the tensors in the network φ p for which rank( (A p ) y A,B ) = R. Then, rank( A y A,B ) is at least R almost always, i.e. for all configurations of the weights of φ but a set of Lebesgue measure zero.

Proof.

Consider the specific assignment of all of the tensors in the network φ p which achieves rank( (A p ) y A,B ) = R, and leads to the resultant tensor (A p ) y upon contraction of the network.

Observing the form of the deep ConvAC TN presented in appendix D.2, we see that it is composed of δ tensors and of weight matrices A (l,j) ∈ R r l ×r l−1 .

Recalling that the entries of the former are dictated by construction and obey eq. 12, the assignment of all of the tensors in the network φ p is an assignment of all entries of the weight matrices in φ DISPLAYFORM4 We denote the bond dimension at level l ∈ [L] ∨ {−1, 0} of φ p by r p l (recall that we defined r−1 = M ).

By the definition of φ p , this bond dimension cannot be higher than the bond dimension in the corresponding level in φ : ∀l r p l ≤ r l .

Accordingly, the matrices in φ do not have lower dimensions (rows or columns) than the corresponding matrices in φ p .

Thus, one can choose an assignment of the weights of all the matrices in φ p l when observing only its first r p l entries in each dimension, as is clear from the δ tensor definition in eq. 12.

Finally, we observe that this construction leads to A y containing the tensor (A p ) y as a hypercube in its entirety and holding zeros elsewhere, leading to rank( A Lemma 1 basically implies that showing that the upper bound on the rank of the matricization of the deep ConvAC conv-weights tensor that is presented in claim 1 is tight when all of the channel numbers are powers of some integer p (which we show below in lemma 2), is enough in order to prove the lower bound stated in theorem 2.

It is noteworthy, that lemma 2 is stated similarly to claim 1, with two differences: 1) minC WC appears as a lower bound on the rank of matricization of the conv-weights tensor rather than an upper bound, and 2) all of the channel numbers are restricted to powers of the same integer p. That is to say, by proving this lemma we in fact show that the upper bound proven in claim 1 is tight for this quite general setting of channel numbers.

In the following, we provide an assignment of indices for the tensors in φ for which the rank of the matricization A y A,B is at least: minC WC .

In accordance with claim 2, this will satisfy the lemma as it implies this rank is achieved for all configurations of the ConvAC network weights but a set of Lebesgue measure zero.

The proof of lemma 2 is organized as follows.

We begin with the construction of the TN φ * presented in FIG0 from the original network φ, and the show that it suffices to analyze φ * for our purposes.

Next, we elaborate on the form that the δ tensors in φ assume when constructed in φ * .

We then use this form to define the concept of δ restricted edge disjoint paths, which morally are paths from A to B that are guaranteed to be compliant with the form of a δ tensor when passing through it.

Finally, we use such paths in order to provide an assignment of the indices for the tensors in φ * which upholds the required δ condition.

A and H B with dimensions obeying eq. 19 be the spaces corresponding to the different configurations of the indices in IA and IB, respectively.

We construct a TN φ * with a graph G * (V * , E * ) and a bond dimensions function c * : E * →

N for which there is a one-to-one correspondence between the tensor assignments in φ and tensor assignments in φ * , such that the resulting linear maps between H A and H B have the same rank.

For each edge e ∈ E, denote ne := log p c(e).

By the conditions of the lemma, ∀e : ne ∈ N as c(e) is an integer power of p for all edges in E. The graph G * of the network φ * is constructed as follows.

Starting with G * = (V, ∅), for each edge e = (u, v) ∈ E we insert ne parallel edges connecting u to v in G * , to form the edge set E * .

Additionally, we define the bond dimensions function of the network φ * to assign the value of p to all of the added edges, i.e. ∀e * ∈ E * : c * (e * ) = p. In FIG0 an example for such a construction of φ * is shown for some N = 8 ConvAC TN.In the paragraphs below, we derive eq. 25 which shows that an analysis of φ * suffices for our purposes.

This result is intuitive in some sense, as the construction of φ * keeps intact the architecture of the network and the distribution of the degrees of freedom to some extent.

As it is the key to our proof, we formulate this argument hereinafter.

As each edge e ∈ E was translated into ne edges in E * , there are .

This means that an index number in A * corresponding to an edge e * ∈ E * would be in A * (resp.

B * ) if the edge e ∈ E from which it originated corresponded to an index number in A that was in A (resp.

B).

This is easily understood pictorially, see FIG0 .

Accordingly denote the corresponding partition of the degree 1 vertices in G * by (V A * , V B * ).

We will now show that the rank of the matricization of A w.r.t.

the partition (A, B) is equal to the rank of the matricization of A * w.r.t.

the partition (A * , B * ).

DISPLAYFORM0 We denote by τv the tensors corresponding to a vertex v ∈ V in the network φ, and by τ * v the tensors corresponding to the same vertex v in the network φ * .

Let z be the order of τv, and denote the set of edges in E incident to v by {e k 1 , ..., e kz } where k1, ..., kz are the corresponding indices.

For every index kj, j ∈ [z], let K * j = {k * j 1 , ..., k * j ne k j } be the indices corresponding to the edges which were added to φ * in the location of e k j in φ.

According to the construction above, there is a one-to-one correspondence between the elements in K * j and kj, that can be written as: DISPLAYFORM1 where DISPLAYFORM2 Thus, if one has the entries of the tensors in φ * , the following assignment to the entries of the tensors in φ: DISPLAYFORM3 would ensure : DISPLAYFORM4 Effectively, we have shown that the claim to be proved regarding rank( A A,B ) can be equivalently proved for rank( A * A * ,B * ).The Form of the δ Tensor in φ * :It is worthwhile to elaborate on the form of a tensor in φ * which corresponds to an order 3 δ tensor in φ.

We denote by τ v δ = δ a δ tensor in φ, and by τ * v δ the corresponding tensor in φ * .

FIG0 shows an example for a transformed tensor in φ * that originated in an order 3 δ tensor in φ, all edges of which uphold ne = 2.

From eqs. 23 and 24, and from the form of the δ tensor given in eq. 12, it is evident that in this case an entry is non-zero in τ * v δ only when k * 1 DISPLAYFORM5 2 .

In the general case, the condition for an entry of 1 in τ * v δ is: DISPLAYFORM6 where ne = log p c(e) for any edge e incident to v in G. Hence, a tensor τ * v δ in φ * which corresponds to a δ tensor in φ can be written as: DISPLAYFORM7 δ Restricted Edge Disjoint PathsConsider an edge-cut set in G that achieves the minimal multiplicative weight over all cuts w.r.t the partition (V A , V B ) in the graph G: Cmin ∈ argmin C WC , and consider the corresponding edge-cut set C * min in G * s.t.

for each edge e ∈ Cmin, the ne edges constructed from it are in C * min .

By the construction of G * , there are exactly L := log p (minC WC ) edges in C * min and their multiplicative weight upholds DISPLAYFORM8 A search for a minimal multiplicative cut, can be generally viewed as a classical min-cut problem when defining a maximum capacity for each edge that is a logarithm of its bond dimension.

Then, a min-cut/max-flow value can be obtained classically in a graph with additive capacities and a final exponentiation of the result provides the minimal multiplicative value of the min-cut.

Since all of the bond dimensions in φ * are equal to p, such a process results in a network with all of its edges assigned capacity 1.

From the application of theorem 3 on such a graph, it follows that the maximal number of edge disjoint paths between V A * and V B * in the graph G * , which are paths between V A * and V B * that are allowed to share vertices but are not allowed to share edges, is equal to the cardinality of the minimum edge-cut set C * min .

In our case, this number is L, as argued above.

Denote these edge disjoint paths by q1, ..., qL.In accordance with the form of τ * v δ , the tensors in φ * corresponding to δ tensors in φ given in eq. 27, we introduce the concept of δ restricted edge disjoint paths between V A * and V B * in the graph G * , which besides being allowed to share vertices but not to share edges, uphold the following restriction.

For every δ tensor τ v δ of order 3 in the graph G, with e ∈ E a representative edge incident to v in G, a maximum of ne such paths can pass through v in G * , each assigned with a different number t ∈ [ne].

The paths uphold that when passing through v in G * each path enters through an edge with index k * j in t in and leaves through an edge with index k * j out t out only if jin = jout : jin, jout ∈ [3] and tin = tout = t, where no two paths can have the same t.

This restriction imposed on the indices of τ * v δ in φ * , to be called hereinafter the δ restriction, is easily understood pictorially, e.g. in FIG0 the paths crossing the τ * v δ tensor must only contain edges of the same color in order to uphold the δ restriction.

We set out to show, that for the network in question one can choose the L edge disjoint paths to uphold the δ restriction.

Then, a weight assignment compliant with the δ tensors in the network can be guaranteed to uphold the requirements of the lemma, despite the fact that most of the entries in the δ tensors are equal to zero.

Denote the set of ne edges in G * that originated from a certain edge e in G, by X * e ⊂ E * .

We first show that one can choose the L edge disjoint paths s.t.

in a flow directed from V A * to V B * w.l.o.g, there is no set of edges X * e that corresponds to any e ∈ E for which two edges e * i , e * j ∈ X * e ⊂ E * belong to paths qi, qj which flow in opposite directions.

FIG0 clarifies this claim.

We observe the classical max-flow in the graph G, i.e. when assigning a maximum capacity for each edge e that is equal to ne := log p c(e), a maximum flow of L is possible between V A and V B in G. Observing the paths in G that flow w.l.o.g.

from V A to V B , together they can transfer a maximum capacity of L. Note that in G, these paths most certainly do not need to be edge disjoint paths.

We argue that one can choose such paths from V A to V B in G such that on each edge e there is an integer capacity transferred.

The existence of such paths in G follows directly from the integral flow theorem BID9 ), which states that if each edge has integral capacity, then there exists such an integral maximal flow.

Note, that these paths must also uphold the basic rule that the sum of capacities transferred on a certain edge e ∈ E, even if this is performed via several paths, is less than the edge maximum capacity ne.

Published as a conference paper at ICLR 2018 FIG0 , each edge is split into n e = 2 edges of bond dimension p.

The δ tensor structure in φ translates into this τ * v δ tensor holding a non-zero entry only when the indices corresponding to all of the edges that are marked by the same color are equal to each other (eq. 27).

Additionally, paths crossing this τ * v δ tensor must only contain edges of the same color in order to be called δ restricted edge disjoint paths.

(b) There are L guaranteed edge disjoint paths between V A * and V B * .

In a flow directed from V A * to V B * (w.l.o.g), we argue that one can choose these paths such that they have the same flow direction in all edges in φ * that originate from a certain edge in φ.

One can now construct L paths in G * in a recursive manner, if a maximum additive capacity for each edge e * ∈ E * is similarly defined to be log p c * (e * ) = log p p := 1.

Starting with a single quanta of flow along some path in G, construct a single path in the corresponding position in G * .

Each edge that is part of this path in G * will transfer exactly one quanta of flow, as that is their maximum capacity that is chosen to be saturated in order to transfer the same amount of capacity that is transferred in G. Now, remove the full edges in G * and reduce the capacities of all edges along the original path in G by one.

Repeating this process until a capacity of L is transferred in both graphs, since ne is the number of new edges added to G * in the place of each edge e, and it is also an upper bound on the integer capacity this path transfers in G, it follows that in G * one finds L paths between V A * and V B * that correspond exactly to the paths transferring integer capacity in G guaranteed by integral flow theorem.

These paths in G * are edge disjoint since the edges of each path were removed from the graph when constructed.

Choosing precisely these edge disjoint paths in G * , one is guaranteed that the flow from V A * to V B * in all of the edges in X * e that belong to these paths would be in the same direction, as they originated in the same edge e in G that had a flow in that single specific direction from A to B. Pictorially, since the different edges in X * e all originate from one single edge that obviously cannot have two opposite directions of net flow, they can all be chosen to transfer flow in the same direction.

Observing an order 3 δ tensor τ v δ in φ, denote the three edges incident to v in G by e1, e2, e3 ∈ E, and denote ne := ne 1 = ne 2 = ne 3 .

Now that we have asserted that all of the L edge disjoint paths in G * may uphold the above condition, we choose the paths as such, i.e. under this choice all of the edges in each respective set (namely X * e 1 , X * e 2 or X * e 3 ) pass flow from V A * to V B * in the same direction.

In this case, a maximum of ne paths can pass through the delta tensor.

This can be easily understood by the following argument.

Denote a set X * e i by 'I' if the paths passing through its edges are incoming to the δ tensor in a flow from V A * to V B * , and by 'O' if they are outgoing from the δ tensor in such a flow.

W.l.o.g.

we assume that X * e 1 , X * e 2 are denoted by 'I' and X * e 3 is denoted by 'O'.

in this case, only ne such edge disjoint paths can flow out of the δ tensor.

In the opposite case, where two groups of edges out of the three are denoted by 'O' and only one group is denoted by 'I', only ne such edge disjoint paths can flow into the δ tensor.

The contrary, i.e. if more than ne such paths were to cross the δ tensor, would imply a cross flow of edge disjoint paths in at least one of the sets X * e 1 , X * e 2 , X * e 3 , in contradiction to this choice of paths.

This provides us with the ability to distribute the paths in the following manner, that upholds the δ restriction described above.

Assume w.l.o.g that X * e 1 is the set for which the most edges are in the chosen edge disjoint paths.

Denote by q1, ..., qN 2 the paths that include edges in X * e 1 and X * e 2 , and by qN 2 +1, ..., qN 2 +N 3 the paths that include edges in X * e 1 and X * e 3 .

Finally, assign the index t to the path qt.

From the statement above, it is guaranteed that N2 + N3 ≤ ne.

Therefore, this choice of paths is guaranteed to uphold the delta restriction defined above, which states that each path must receive a different value t ∈ [ne].

Specifically, this implies that the maximal number of δ restricted edge disjoint paths between V A * and V B * in the graph G * is L.With all the preliminaries in place, the proof of theorem 2 readily follows: DISPLAYFORM9 For a specific p, consider the network φ p such as defined in theorem 2, i.e. a TN with the same connectivity as φ, where all of the bond dimensions are modified to be equal the closest power of p to their value in φ from below.

Let (

The exponential depth efficiency result shown in BID7 , can be straightforwardly reproduced by similar graph-theoretic considerations.

We show below an upper bound on the rank of matricization of the conv-weights tensor for a case of a general pooling window.

The bound implies that any amount of edges in a cut that are connected to the same δ tensor will contribute their bond dimension only once to the multiplicative weight of this cut, thus effectively reducing the upper bound when many cut edges belong to the same δ tensor.

This does not affect our analysis of the deep network above, as the δ tensors in that network are only three legged (see FIG8 ).

Therefore, in the above analyzed deep ConvAC, a cut containing more than one δ tensor leg can be replaced by an equivalent cut containing only one leg of that δ tensor, and the value of minC WC is unchanged.

Formally, in order to apply similar considerations to the ConvAC with general sized pooling windows, such as the one presented in fig. 6 , one must consider more closely the restrictions imposed by the δ tensors.

To this end, we define the object underlying a ConvAC-weights TN with general sized pooling windows φ to be composed of the following three:• An undirected graph G(V, E), with a set of vertices V and a set of edges E. The set of nodes is divided into two subsets V = V tn · ∪ V inputs , where V inputs are the N degree-1 virtual vertices and V tn corresponds to tensors of the TN.

DISPLAYFORM0 , where b is the number of δ tensors in the network.

If we label each δ tensor in the network by a number i ∈ [b], this function upholds f (e) = i for e ∈ E that is incident to a vertex which represents the i th delta tensor in the ConvAC TN.

For each edge e ∈ E incident to a degree 1 vertex, this function assigns a different number f (e) = i for i ∈ b + [N ].

Such an edge is an external edge in the ConvAC TN, which according to the construction presented in appendix D is the only type of edge not incident to any δ tensor.

In words, the function f divides al the edges in E into b + N groups, where edges are in the same group if they are incident to the same vertex which represents a certain δ tensor in the network.• A function c : [b + N ] → N, associating a bond dimension r ∈ N with each different group of edges defined by the set: Ei = {e ∈ E : f (e) = i}.Observing an edge-cut set with respect to the partition (A, B) and the corresponding set G C = {f (e) : e ∈ C}. We denote the elements of G C by g DISPLAYFORM1 These elements represent the different groups that the edges in C belong to (by the definition of f , edges incident to the same delta tensor belong to the same group).

We define the modified weight of such an edge-cut set C as: DISPLAYFORM2 The weight definition given in eq. 32 can be intuitively viewed as a simple multiplication of the bond dimensions of all the edges in a cut, with a single restriction: the bond dimension of edges in the cut which are connected to a certain δ tensor, will only be multiplied once (such edges have equal bond dimensions by definition, see eq. 12).

An example of this modified weight can be seen in FIG0 , where the replacement of a general tensor by a δ tensor results in a reduction in the minimal cut, due to the rules defined above.

In the following claim, we provide an upper bound on the ability of a ConvAC with a general pooling window to model correlations of its inputs, as measured by the Schmidt entanglement measure (see section 4).

fig. 6 with a general pooling window.

Let G(V, E, f, c) represent the TN φ corresponding to the ConvAC network, and let (V A , V B ) be the vertices partition of V inputs in G corresponding to (A, B) .

Then, the rank of the matricization A y A,B is no greater than: minCWC , where C represents a cut w.r.t (V A , V B ) andWC is the modified multiplicative weight defined by eq. 32.

FIG0 : An example for the effect that a δ tensor has on the upper bound on the rank of the matricization of the overall tensor represented by a TN.

min CWC is defined in eq. 32 and shown in claim 3 to be the upper bound on the rank of the matricization of the conv-weights tensor of a ConvAC represented by a TN.

In this example, the upper bound is reduced upon changing a single general tensor in the TN shown in (a) (identical to FIG0 ), whose entries are free to be equal any value, with a δ tensor in the TN shown in (b) which obeys the constraint given in eq. 12.

The centrality of the δ tensor in the TN compliant with a shallow ConvAC (that is depicted in FIG7 , is in effect the element which limits the expressiveness of the shallow network, as is discussed in appendix F.Having seen the proof of the claim 1 above and its accompanying graphics, the proof of the upper bound presented in claim 3 can be readily attained.

The only difference between the two lies in the introduction of the δ tensors to the network, which allows us to derive the tighter lower bound shown in claim 3.

The modification to the above proof of claim 1 focuses on the coalescence of the cut indices IC into a single index m ∈ [ DISPLAYFORM0 Assume that any two indices of multiplicands in this product, denoted by ki and kj, are connected to the same δ tensor that has some bond dimension q := c k i = c k j .

Upon contraction of the TN in FIG0 , the cut indices are internal indices that are to be summed upon.

However, whenever ki ∈ [q] and kj ∈ [q] are different, by the constraint imposed in the δ tensor definition (eq. 12), the entire term vanishes and there is no contribution to the final value of A d 1 ...d N calculated by this contraction.

Thus, ki, kj and any other index connected to the same δ tensor can be replaced by a representative index k α ∈ [q] whenever they appear in the summation.

α ∈ G C upholding c(α) = q, is the group index of this δ tensor, given by α = f (e k i ) = f (e k j ) with e k i and e k j the edges corresponding to the indices ki and kj in the network.

Thus, the single index m achieved by coalescing all of the cut indices can be defined in the range m ∈ [WC ], with WC defined by eq. 32 upholdingWC ≤ |C| i=1 c k i , where the equality is satisfied when no two edges in the cut are incident to the same δ tensor.

Finally, the matricization A A,B can be written as a multiplication of two matrices as portrayed in FIG0 : DISPLAYFORM1 DISPLAYFORM2 Recalling that as in the proof of claim 1 the edge-cut set may include the external edges, we attain: rank( A A,B ) ≤ min CW C .Observing FIG7 which shows the TN corresponding to the shallow ConvAC architecture, the central positioning of a single δ tensor implies that under any partition of the inputs (A, B) s.t.

|A| = |B| = N/2, the minimal cut will obey W min C = min(M N/2 , k).

Thus, in order to reach the exponential in N measure of entanglement w.r.t.

the interleaved partition that was obtained in section 5 for the deep network, the number of channels in the single hidden layer of the shallow network k, must grow exponentially with N .

Therefore, one must exponentially enlarge the size of the shallow network in order to achieve the expressiveness that a polynomially sized deep network achieves, and an exponential depth efficiency is demonstrated.

In this section, we describe simulations performed on an N = 16 deep ConvAC TN (with pooling windows of size 2), which are aimed at quantifying the prevalence of deviations from the upper bound on the ranks of the matricization of conv-weights tensor presented in claim 1.

In section E.2 we proved theorem 2, showing in effect that this upper bound is tight when all of the channel numbers are powers of some integer p, and guaranteeing a positive result in all cases.

However, for the general setting of channel numbers there is no theoretical guarantee that the upper bound is tight.

Indeed, BID8 show a counter example where the matricization rank is effectively lower the minimal multiplicative cut for a general TN (that has no δ tensors such as in the ConvAC TN).

There is no reason to believe that the upper bound is tight for the TN representing a ConvAC for a general setting of channel numbers, and indeed our simulations below show deviations from it.

However, as is indicated below such deviations are negligible in prevalence and low in value.

A theoretical formulation of this is left for future work.

The experiments were performed in matlab, and tensor contractions were computed using a function introduced by Pfeifer et al. (2014) .

An N = 16 with M = 2 ConvAC TN was constructed (see figs. 8 and 9), with the entries of the weights matrices randomized according to a normal distribution.

The bond dimensions of layers 0 through 3 were drawn from the set of the first 6 prime numbers: [2, 3, 5, 7, 11, 13] , to a total of 360 different arrangements of bond dimensions.

This was done in order to resemble a situation as distinct as possible from the case where all of the bond dimensions are powers of the same integer p, for which the tightness of the upper bound is guaranteed by theorem 2.

Per bond dimension arrangement, all of the = 6435 different partitions were checked, for a total of 360 · 6435 = 2.3166 · 10 6 different configurations.

As argued in section E.2, the logarithm of the upper bound on the rank of the conv-weights tensor matricization that is shown in claim 1, is actually the max-flow in a network with the same connectivity that has edge capacities which are equal to the logarithm of the respective bond dimensions.

Therefore, a configuration for which the rank of matricization is equal to the exponentiation of the max-flow through such a corresponding network, effectively reaches the upper bound.

We calculated the max-flow independently for each configuration using the Ford-Fulkerson algorithm BID12 ), and set out to search for deviations from such an equivalence.

The results of the above described simulations are as follows.

Only 1300 configurations, which constitute a negligible fraction of the 2.3166 million configurations that were checked, failed to reach the upper bound and uphold the min-cut max-flow equivalence described above.

Moreover, in those rare occasions that a deviation occurred, the percentage of deviations from the upper bound did not exceed 10% of the value of the upper bound.

This check was performed on a bond setting that is furthest away from all channel numbers being powers of the same integer, yet the tightness of the upper bound emerges as quite robust, justifying experimentally our general view of the minimal weight over all cuts in the network, minC WC , as the effective indication for the matricization rank of the conv-weights tensor w.r.t.

the partition of interest.

A caveat to be stated with this conclusion is that we checked only up to N = 16, and the discrepancies that were revealed here might become more substantial for larger networks.

As mentioned above, this is left for future theoretical analysis, however the lower bound shown in theorem 2 guarantees a positive result regarding the rank of the matricization of the conv-weights tensor in all of the cases.

<|TLDR|>

@highlight

Employing quantum entanglement measures for quantifying correlations in deep learning, and using the connection to fit the deep network's architecture to correlations in the data.