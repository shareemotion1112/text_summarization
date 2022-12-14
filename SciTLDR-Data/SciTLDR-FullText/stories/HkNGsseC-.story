Expressive efficiency refers to the relation between two architectures A and B, whereby any function realized by B could be replicated by A, but there exists functions realized by A, which cannot be replicated by B unless its size grows significantly larger.

For example, it is known that deep networks are exponentially efficient with respect to shallow networks, in the sense that a shallow network must grow exponentially large in order to approximate the functions represented by a deep network of polynomial size.

In this work, we extend the study of expressive efficiency to the attribute of network connectivity and in particular to the effect of "overlaps" in the convolutional process, i.e., when the stride of the convolution is smaller than its filter size (receptive field).

To theoretically analyze this aspect of network's design, we focus on a well-established surrogate for ConvNets called Convolutional Arithmetic Circuits (ConvACs), and then demonstrate empirically that our results hold for standard ConvNets as well.

Specifically, our analysis shows that having overlapping local receptive fields, and more broadly denser connectivity, results in an exponential increase in the expressive capacity of neural networks.

Moreover, while denser connectivity can increase the expressive capacity, we show that the most common types of modern architectures already exhibit exponential increase in expressivity, without relying on fully-connected layers.

One of the most fundamental attributes of deep networks, and the reason for driving its empirical success, is the "Depth Efficiency" result which states that deeper models are exponentially more expressive than shallower models of similar size.

Formal studies of Depth Efficiency include the early work on boolean or thresholded circuits BID19 BID25 Håstad and Goldmann, 1991; Hajnal et al., 1993) , and the more recent studies covering the types of networks used in practice BID13 BID12 Eldan and Shamir, 2016; BID5 BID23 BID16 .

What makes the Depth Efficiency attribute so desirable, is that it brings exponential increase in expressive power through merely a polynomial change in the model, i.e. the addition of more layers.

Nevertheless, depth is merely one among many architectural attributes that define modern networks.

The deep networks used in practice consist of architectural features defined by various schemes of connectivity, convolution filter defined by size and stride, pooling geometry and activation functions.

Whether or not those relate to expressive efficiency, as depth has proven to be, remains an open question.

In order to study the effect of network design on expressive efficiency we should first define "efficiency" in broader terms.

Given two network architectures A and B, we say that architecture A is expressively efficient with respect to architecture B, if the following two conditions hold: (i) any function h realized by B of size r B can be realized (or approximated) by A with size r A ∈ O(r B ); (ii) there exist a function h realized by A with size r A , that cannot be realized (or approximated) by B, unless r B ∈ Ω(f (r A )) for some super-linear function f .

The exact definition of the sizes r A and r B depends on the measurement we care about, e.g. the number of parameters, or the number of "neurons".

The nature of the function f in condition (ii) determines the type of efficiency taking place -if f is exponential then architecture A is said to be exponentially efficient with respect to architecture B, and if f is polynomial so is the expressive efficiency.

Additionally, we say A is completely efficient with respect to B, if condition (ii) holds not just for some specific functions (realizable by A), but for all functions other than a negligible set.

In this paper we study the efficiency associated with the architectural attribute of convolutions, namely the size of convolutional filters (receptive fields) and more importantly its proportion to their stride.

We say that a network architecture is of the non-overlapping type when the size of the local receptive field in each layer is equal to the stride.

In that case, the sets of pixels participating in the computation of each two neurons in the same layer are completely separated.

When the stride is smaller than the receptive field we say that the network architecture is of the overlapping type.

In the latter case, the overlapping degree is determined by the total receptive field and stride projected back to the input layer -the implication being that for the overlapping architecture the total receptive field and stride can grow much faster than with the non-overlapping case.

As several studies have shown, non-overlapping convolutional networks do have some theoretical merits.

Namely, non-overlapping networks are universal BID5 , i.e. they can approximate any function given sufficient resources, and in terms of optimization, under some conditions they actually possess better convergence guaranties than overlapping networks.

Despite the above, there are only few instances of strictly non-overlapping networks used in practice (e.g. BID17 ; van den BID24 ), which raises the question of why are non-overlapping architectures so uncommon?

Additionally, when examining the kinds of architectures typically used in recent years, which employ a mixture of both overlapping and nonoverlapping layers, there is a trend of using ever smaller receptive fields, as well as non-overlapping layers having an ever increasing role BID10 BID20 BID21 .

Hence, the most common networks used practice, though not strictly non-overlapping, are increasingly approaching the non-overlapping regime, which raises the question of why having just slightly overlapping architectures seems sufficient for most tasks?In the following sections, we will shed some light on these questions by analyzing the role of overlaps through a surrogate class of convolutional networks called Convolutional Arithmetic Circuits (ConvACs) BID5 ) -instead of non-linear activations and average/max pooling layers, they employ linear activations and product pooling.

ConvACs, as a theoretical framework to study ConvNets, have been the focused of several works, showing, amongst other things, that many of the results proven on this class are typically transferable to standard ConvNets as well .

Though prior works on ConvACs have only considered non-overlapping architectures, we suggest a natural extension to the overlapping case that we call Overlapping ConvACs.

In our analysis, which builds on the known relation between ConvACs and tensor decompositions, we prove that overlapping architectures are in fact completely and exponentially more efficient than non-overlapping ones, and that their expressive capacity is directly related to their overlapping degree.

Moreover, we prove that having even a limited amount of overlapping is sufficient for attaining this exponential separation.

To further ground our theoretical results, we demonstrate our findings through experiments with standard ConvNets on the CIFAR10 image classification dataset.

In this section, we introduce a class of convolutional networks referred to as Overlapping Convolutional Arithmetic Circuits, or Overlapping ConvACs for short.

This class shares the same architectural features as standard ConvNets, including some that have previously been overlooked by similar attempts to model ConvNets through ConvACs, namely, having any number of layers and unrestricted receptive fields and strides, which are crucial for studying overlapping architectures.

For simplicity, we will describe this model only for the case of inputs with two spatial dimensions, e.g. color images, and limiting the convolutional filters to the shape of a square.

DISPLAYFORM0 Figure 1: An illustration of a GC Layer.

We begin by presenting a broad definition of a Generalized Convolutional (GC) layer as a fusion of a 1×1 linear operation with a pooling function -this view of convolutional layers is motivated by the all-convolutional architecture BID20 , which replaces all pooling layers with convolutions with stride hidden layer L-2 hidden layer 1 DISPLAYFORM1 Figure 2: An illustration of a Generalized Convolutional Network.greater than 1.

The input to a GC layer is a 3-order tensor (multi-dimensional array), having width and height equal to H (in) ∈ N and depth D (in) ∈ N, also referred to as channels, e.g. the input could be a 2D image with RGB color channels.

Similarly, the output of the layer has width and height equal to H (out) ∈ N and D (out) ∈ N channels, where DISPLAYFORM2 S for S ∈ N that is referred to as the stride, and has the role of a sub-sampling operation.

Each spatial location (i, j) at the output of the layer corresponds to a 2D window slice of the input tensor of size R × R × D (in) , extended through all the input channels, whose top-left corner is located exactly at (i·S, j ·S), where R ∈ N is referred to as its local receptive field, or filter size.

For simplicity, the parts of window slices extending beyond the boundaries have zero value.

Let y ∈ R D ((out) be a vector representing the channels at some location of the output, and similarly, let DISPLAYFORM3 be the set of vectors representing the slice, where each vector represents the channels at its respective location inside the R × R window, then the operation of a GC layer is defined as follows: DISPLAYFORM4 where DISPLAYFORM5 are referred to as the weights and biases of the layer, respectively, and g : R DISPLAYFORM6 is some point-wise pooling function.

See FIG7 for an illustration of the operation a GC layer performs.

With the above definitions, a GC network is simply a sequence of L GC layers, where for l ∈ [L] ≡ {1, . . .

, L}, the l'th layer is specified by a local receptive field R (l) , a stride DISPLAYFORM7 , and a pooling function g (l) .

For classification tasks, the output of the last layer of the network typically has 1×1 spatial dimensions, i.e. a vector, where each output channel DISPLAYFORM8 represents the score function of the y'th class, denoted by h y , and inference is perform by y * = arg max y h y (X).

Oftentimes, it is common to consider the output of the very first layer of a network as a low-level feature representation of the input, which is motivated by the observation that these learned features are typically shared across different tasks and datasets over the same domain (e.g. edge and Gabor filters for natural images).

Hence, we treat this layer as a separate fixed "zeroth" convolutional layer referred to as the representation layer, where the operation of the layer can be depicted as applying a set of fixed functions DISPLAYFORM9 to the window slices denoted by x 1 , . . .

, x N ∈ R s , i.e. the entries of the output tensor of this layer are given by [N ] .

With these notations, the output of a GC network can be viewed as a function h y (x 1 , . . .

, x N ).

The entire GC network is illustrated in fig. 2 .

Given a non-linear point-wise activation function σ(·) (e.g. ReLU), then setting all pooling functions to average pooling followed by the activation, i.e. g( DISPLAYFORM10 DISPLAYFORM11 , give rise to the common all-convolutional network with σ(·) activations, which served as the initial motivation for our formulation.

Alternatively, choosing instead a product pooling function, i.e. g( DISPLAYFORM12 , results in an Arithmetic Circuit, i.e. a circuit containing just product and sum operations, hence it is referred to as a Convolutional Arithmetic Circuit, or ConvAC.

It is important to emphasize that ConvACs, as originally introduced by BID5 , are typically described in a very different manner, through the language of tensor decompositions (see app.

A for background).

Since vanilla ConvACs can be seen as an alternating sequence of 1×1 convolutions and non-overlapping product pooling layers, then the two formulations coincide when all GC layers are non-overlapping, i.e. for all l ∈ [L], R (l) = S (l) .

If, however, some of the layers are overlapping, i.e. there exists l ∈ [L] such that R (l) > S (l) , then our formulation through GC layers diverges, and give rise to what we call Overlapping ConvACs.

Given that our model is an extension of the ConvACs framework, it inherits many of its desirable attributes.

First, it shares most of the same traits as modern ConvNets, i.e. locality, sharing and pooling.

Second, it can be shown to form a universal hypotheses space BID5 .

Third, its underlying operations lend themselves to mathematical analysis based on measure theory and tensor analysis BID5 .

Forth, through the concept of generalized tensor decompositions , many of the theoretical results proven on ConvACs could be transferred to standard ConvNets with ReLU activations.

Finally, from an empirical perspective, they tend to work well in many practical settings, e.g. for optimal classification with missing data BID17 , and for compressed networks BID6 ).While we have just established that the non-overlapping GC Network with a product pooling function is equivalent to vanilla ConvACs, one might wonder if using overlapping layers instead could diminish what these overlapping networks can represent.

We show that not only is it not the case, but prove the more general claim that a network of a given architecture can realize exactly the same functions as networks using smaller local receptive fields, which includes the non-overlapping case.

Proposition 1.

Let A and B be two GC Networks with a product pooling function.

If the architecture of B can be derived from A through the removal of layers with 1×1 stride, or by decreasing the local receptive field of some of its layers, then for any choice of parameters for B, there exists a matching set of parameters for A, such that the function realized by B is exactly equivalent to A. Specifically, A can realize any non-overlapping network with the same order of strides (excluding 1×1 strides).Proof sketch.

This follows from two simple claims: (i) a GC layer can produce an output equivalent to that of a GC layer with a smaller local receptive field, by "zeroing" its weights beyond the smaller local receptive field; and (ii) GC layers with 1×1 receptive fields can be set such that their output is equal to their input, i.e. realize the identity function.

With these claims, the local receptive fields of A can be effectively shrank to match the local receptive fields of B, and any additional layers of A with stride 1×1 could be set such that they are realizing the identity mapping, effectively "removing" them from A. See app.

C.2 for a complete proof.

Proposition 1 essentially means that overlapping architectures are just as expressive as nonoverlapping ones of similar structure, i.e. same order of non-unit strides.

As we recall, this satisfies the first condition of the efficiency property introduced in sec. 1, and does so regardless if we measure the size of a network as the number of parameters, or the number of "neurons"1 .

In the following section we will cover the preliminaries required to show that overlapping networks actually lead to an increase in expressive capacity, which under some settings results in an exponential gain, proving that the second condition of expressive efficiency holds as well.

In this section we describe our methods for analyzing the expressive efficiency of overlapping ConvACs that lay the foundation for stating our theorems.

A minimal background on tensor analysis required to follow our work can be found in sec. 3.1, followed by presenting our methods in sec. 3.2.

In this sub-section we cover the minimal background on tensors analysis required to understand our analysis.

A tensor A ∈ R M1⊗···⊗M N of order N and dimension DISPLAYFORM0 For simplicity, henceforth we assume that all dimensions are equal, i.e. M ≡ M 1 = . . .

= M N .

One of the central concepts in tensor analysis is that of tensor matricization, i.e. rearranging its entries to the shape of a matrix.

Let P · ∪ Q = [N ] be a disjoint partition of its indices, such that P = {p 1 , . . .

, p |P | } with p 1 < . . .

< p |P | , and Q = {q 1 , . . .

, q |Q| } with q 1 < . . .

< q |Q| .

The matricization of A with respect to the partition P · ∪ Q, denoted by A P,Q , is the M |P | -by-M |Q| matrix holding the entries of A, such that for all i ∈ [N ] and DISPLAYFORM1 We take here the broader definition of a "neuron", as any one of the scalar values comprising the output array of an arbitrary layer in a network.

In the case the output array is of width and height equal to H and C channels, then the number of such "neurons" for that layer is H 2 · C. DISPLAYFORM2 3.2 BOUNDING THE SIZE OF NETWORKS VIA GRID TENSORSWe begin with a discussion on how to have a well-defined measure of efficiency.

We wish to compare the efficiency of non-overlapping ConvACs to overlapping ConvACs, for a fixed set of M representation functions (see sec. 2 for definitions).

While all functions realizable by non-overlapping ConvACs with shared representation functions lay in the same function subspace (see BID5 ), this is not the case for overlapping ConvACs, which can realize additional functions outside the sub-space induced by non-overlapping ConvACs.

We cannot therefore compare both architectures directly, and need to compare them through an auxiliary objective.

Following the work of , we instead compare architectures through the concept of grid tensors, and specifically, the grid tensor defined by the output of a ConvAC, i.e. the tensor A(h) for h(x 1 , . . .

, x N ).

Unlike with the ill-defined nature of directly comparing the functions of realized by ConvACs, proved that assuming the fixed representation functions are linearly independent, then there exists template vectors x (1) , . . .

, x (M ) , for which any nonoverlapping ConvAC architecture could represent all possible grid tensors over these templates, given sufficient number of channels at each layer.

More specifically, if DISPLAYFORM3 , then these template vector are chosen such that F is non-singular.

Thus, once we fix a set of linearly independent representation functions, we can compare different ConvACs, whether overlapping or not, on the minimal size required for them to induce the same grid tensor, while knowing such a finite number always exists.

One straightforward direction for separating between the expressive efficiency of two network architectures A and B is by examining the ranks of their respective matricized grid tensors.

Specifically, Let A(h (A) ) and A(h (B) ) denote the grid tensors of A and B, respectively, and let (P, Q) be a partition of [N ], then we wish to find an upper-bound on the rank of A(h (A) ) P,Q as a function of its size on one hand, while showing on the other hand that rank A(h (B) ) P,Q can be significantly greater.

One benefit of studying efficiency through a matrix rank is that not only we attain separation bounds for exact realization, but also immediately gain access to approximation bounds by examining the singular values of the matricized grid tensors.

This brings us to the following lemma, which connects upper-bounds that were previously found for non-overlapping ConvACs BID4 , with the grid tensors induced by them (see app.

C.1 for proof): Lemma 1.

Let h y (x 1 , . . .

, x N ) be a score function of a non-overlapping ConvAC with a fixed set of M linearly independent and continuous representation functions, and L GC layers.

Let (P, Q) be a partition dividing the spatial dimensions of the output of the representation layer into two equal parts, either along the horizontal or vertical axis, referred to as the "left-right" and "top-bottom" partitions, respectively.

Then, for any template vectors such that F is non-singular and for any choice of the parameters of the network, it holds that rank ( DISPLAYFORM4 Lemma 1 essentially means that it is sufficient to show that overlapping ConvACs can attain ranks super-polynomial in their size to prove they are exponentially efficient with respect to nonoverlapping ConvACs.

In the next section we analyze how the overlapping degree is related to the rank, and under what cases it leads to an exponentially large rank.

In this section we analyze the expressive efficiency of overlapping architectures.

We begin by defining our measures of the overlapping degree that will used in our claims, followed by presenting our main results in sec. 4.2.

For the sake of brevity, an additional set of results, in light of the recent work by BID4 on "Pooling Geometry", is deferred to app.

B. To analyze the efficiency of overlapping architectures, we will first formulate more rigorously the measurement of the overlapping degree of a given architecture.

As mentioned in sec. 1, we do so by defining the concepts of the total receptive field and total stride of a given

S , respectively.

Both measurements could simply be thought of as projecting the accumulated local receptive fields (or strides) to the the first layer, as illustrated in FIG0 , which represent a type of global statistics of the architecture.

However, note that proposition 1 entails that a given architecture could have a smaller effective total receptive field, for some settings of its parameters.

This leads us to define the α-minimal total receptive field, for any α ∈ R + , as the smallest effective total receptive field still larger than α, which we denote by T (l,α) R .

The exact definitions of the above concepts are formulated as follows: DISPLAYFORM0 where we omitted the arguments of T Notice that for non-overlapping networks the total receptive field always equals the total stride, and that only at the end of the network, after the spatial dimension collapses to 1×1, does the the total receptive field grow to encompass the entire size of the representation layer.

For overlapping networks this is not the case, and the total receptive field could grow much faster.

Intuitively, this means that values in regions of the input layer that are far apart would be combined by non-overlapping networks only near the last layers of such networks, and thus non-overlapping networks are effectively shallow in comparison to overlapping networks.

Base on this intuition, in the next section we analyze networks with respect to the point at which their total receptive field is large enough.

With all the preliminaries in place, we are ready to present our main result: Theorem 1.

Assume a ConvAC with a fixed representation layer having M output channels and both width and height equal to H, followed by L GC layers, where the l'th layer has a local receptive field R (l) , a stride S (l) , and DISPLAYFORM0 be a layer with a total receptive DISPLAYFORM1 Then, for any choice of parameters, except a null set (with respect to the Lebesgue measure), and for any template vectors such that F is non-singular, the following equality holds: DISPLAYFORM2 where (P, Q) is either the "left-right" or the "top-bottom" partitions and DISPLAYFORM3 Proof sketch.

Because the entries of the matricized grid tensors are polynomials in the parameters, then according to a lemma by BID17 , if there is a single example that attains the above

Figure 4: A network architectures beginning with large local receptive fields greater than N /2 and at least M output channels.

According to theorem 1, for almost all choice of parameters we obtain a function that cannot be approximated by a non-overlapping architecture, if the number of channels in its next to last layer is less than M H 2 2 .lower-bound on the rank, then it occurs almost everywhere with respect to the Lebesgue measure on the Euclidean space of the parameters.

Given the last remark, the central part of our proof is simply the construction of such an example.

First we find a set of parameters for the simpler case where the first GC layer is greater than a quarter of the input, satisfying the conditions of the theorem.

The motivation behind the specific construction is the pairing of indices from each side of the partition, such that they are both in the same local receptive field, and designing the filters such that the output of each local application of them defines a mostly diagonal matrix of rank D, with respect to these two indices.

The rest of the parameters are chosen such that the output of the entire network results in a product of the entries of these matrices.

Under matricization, this results in a matrix who is equivalent 2 to a Kronecker product of mostly diagonal matrices.

Thus, the matricization rank is equal to the product of the ranks of these matrices, which results in the exponential form of eq. 4.

Finally, we extend the above example to the general case, by realizing the operation of the first layer of the above example through multiple layers with small local receptive fields.

See app.

C.1 for the definitions and lemmas we rely on, and see app.

C.3 for a complete proof.

Combined with Lemma 1, it results in the following corollary: Corollary 1.

Under the same setting as theorem 1, and for all choices of parameters of an overlapping ConvAC, except a negligible set, any non-overlapping ConvAC that realizes (or approximates) the same grid tensor must be of size at least: DISPLAYFORM0 While the complexity of the generic lower-bound above might seem incomprehensible at first, its generality gives us the tools to analyze practically any kind of feed-forward architecture.

As an example, we can analyze the lower bound for the well known GoogLeNet architecture BID21 , for which the lower bound equals 32 98 , making it clear that using a non-overlapping architecture for this case is infeasible.

Next, we will focus on specific cases for which we can derive more intelligible lower bounds.

According to theorem 1, the lower bound depends on the first layer for which its total receptive field is greater than a quarter of the input.

As mentioned in the previous section, for non-overlapping networks this only happens after the spatial dimension collapses to 1×1, which entails that both the total receptive field and total stride would be equal to the width H of the representation layer, and substituting this values in eq. 4 results simply in D -trivially meaning that to realize one nonoverlapping network by another non-overlapping network, the next to last layer must have at least half the channels of the target network.

On the other extreme, we can examine the case where the first GC layer has a local receptive field R greater than a quarter of its input, i.e. R > H /2.

Since the layers following the first GC layer do not affect the lower bound in this case, it applies to any arbitrary sequence of layers as illustrated in fig. 4 .

For simplicity we will also assume that the stride S is less than H /2, and that .

Consider the case of D = M and S = 1, then a non-overlapping architecture that satisfies this lower bound is of the order of magnitude at which it could already represent any possible grid tensor.

This demonstrate our point from the introduction, that through a a polynomial change in the architecture, i.e. increasing the receptive field, we get an exponential increase in expressivity.

Though the last example already demonstrates that a polynomially sized overlapping architecture could lead to an exponential separation, in practice, employing such large convolutions is very resource intensive.

The common best practice is to use multiple small local receptive fields of size B × B, where the typical values are B = 3 or B = 5, separated by a 2 × 2 "pooling" layers, i.e. layers with both stride and local receptive field equal to 2 × 2.

For simplicity, we assume that FIG2 for an illustration of such a network.

Analyzing the above network with theorem 1 results in the following proposition: DISPLAYFORM1 Proposition 2.

Consider a network comprising a sequence of GC blocks, each block begins with a layer whose local receptive field is B×B and its stride 1×1, followed by a layer with local receptive field 2×2 and stride 2×2, where the output channels of all layers are at least 2M , and the spatial dimension of the representation layer is H×H for H=2 L .

Then, the lower bound describe by eq. 4 for the above network is greater than or equal to: DISPLAYFORM2 .

Finally, assum- DISPLAYFORM3 Proof sketch.

We first find a closed-form expression for the total receptive field and stride of each of the B×B layers in the given network.

We then show that for layers whose total receptive field is greater than H 2 , its α-minimal total receptive field, for α= H 2 , is equal to H 2 +1.

We then use the above to find the first layer who satisfies the conditions of theorem 1, and then use our closed-forms expressions to simplify the general lower bound for this case.

See app.

C.4 for a complete proof.

In particular, for the typical values of M = 64, B = 5, and H ≥ 20, the lower bound is at least 64 20 , which demonstrates that even having a small amount of overlapping already leads to an exponential separation from the non-overlapping case.

When B grows in size, this bound approaches the earlier result we have shown for large local receptive fields encompassing more than a quarter of the image.

When H grows in size, the lower bound is dominated strictly by the local receptive fields.

Also notice that based on proposition 2, we could also derive a respective lower bound for a network following VGG style architecture BID18 , where instead of a single convolutional layer before every "pooling" layer, we have K layers, each with a local receptive field of C × C. Under this case, it is trivial to show that the bound from proposition 2 holds for B = K · (C − 1) + 1, and under the typical values of C = 3 and K = 2 it once again results in a lower bound of at least 64 20 .

Train Accuracy (%) B = 1 B = 2 B = 3 B = 4 B = 5Figure 6: Training accuracies of standard ConvNets on CIFAR-10 with data augmentations, where the results of spatial augmentations presented at the top row, and color augmentations at the bottom row.

Each network follows the architecture of proposition 2, with with receptive field B and using the same number of channels across all layers, as specified by the horizontal axis of left plot.

We plot the same results with respect to the total number of parameters in the right plot.

In this section we show that the theoretical results of sec. 4.2 indeed hold in practice.

In other words, there exists tasks that require the highly expressive power of overlapping architectures, on which non-overlapping architectures would have to grow by an exponential factor to achieve the same level of performance.

We demonstrate this phenomenon on standard ConvNets with ReLU activations that follow the same architecture that was outlined in proposition 2, while varying the number of channels and the size of the receptive field of the B×B "conv" layers.

The only change we made, was to replace the 2×2-"pooling" layers of the convolutional type, with the standard 2×2-max-pooling layers, and using the same number of channels across all layers.

This was done for the purpose of having all the learned parameters located only at the (possibly) overlapping layers.

More specifically, the network has 5 blocks, each starting with a B×B convolution with C channels, stride 1×1, and ReLU activation, and then followed by 2×2 max-pooling layer.

After the fifth "conv-pool", there is a final dense layer with 10 outputs and softmax activations.

We train each of these networks for classification over the CIFAR-10 dataset, with two types of data augmentation schemes: (i) spatial augmentations, i.e. randomly translating (up to 3 pixels in each direction) and horizontally flipping each image, and (ii) color augmentations following Dosovitskiy et al. FORMULA22 , i.e. randomly adding a constant shift (at most ±0.3) to the hue, saturation, and luminance, for each attribute separately, and in addition randomly sampling a multiplier (in the range [0.5, 1.5]) just to the saturation and luminance.

Though typically data augmentation is only used for the purpose of regularization, we employ it for the sole purpose of raising the hardness of the regular CIFAR-10 dataset, as even small networks can already overfit and effectively memorize its small dataset.

We separately test both the spatial and color augmentation schemes to emphasize that our empirical results cannot be explained simply by spatial-invariance type arguments.

Finally, the training itself is carried out for 300 epochs with ADAM (Kingma and Ba, 2015) using its standard hyper-parameters, at which point the loss of the considered networks have stopped decreasing.

We report the training accuracy over the augmented dataset in fig. 6 , where for each value of the receptive field B, we plot its respective training accuracies for variable number of channels C. The source code for reproducing the above experiments and plots can be found at https://github.com/HUJI-Deep/OverlapsAndExpressiveness.It is quite apparent that the greater B is chosen, the less channels are required to achieve the same accuracy.

Moreover, for the non-overlapping case of B=1, more than 2048 channels are required to reach the same performance of networks with B>2 and just 64 channels under the spatial aug-mentations -which means effectively exponentially more channels were required.

Even more so, under the color augmentations, we were not able to train non-overlapping networks to reach even the smallest overlapping network (B = 2 and C = 16).

In terms of total number of parameters, there is a clear separation between the overlapping and the non-overlapping types, and we once again see more than an order of magnitude increase in the number of parameters between an overlapping and non-overlapping architectures that achieve similar training accuracy.

As a somewhat surprising result, though based only on our limited experiments, it appears that for the same number of parameters, all overlapping networks attain about the same training accuracy, suggesting perhaps that having the smallest amount of overlapping already attain all the benefits overlapping provides, and that increasing it further does not affect the performance in terms of expressivity.

As final remark, we also wish to acknowledge the limitations of drawing conclusions strictly from empirical experiments, as there could be alternative explanations to these observations, e.g. the effects overlapping has on the optimization process.

Nevertheless, our theoretical results suggests this is less likely the case.

The common belief amongst deep learning researchers has been that depth is one of the key factors in the success of deep networks -a belief formalized through the depth efficiency conjecture.

Nevertheless, depth is one of many attributes specifying the architecture of deep networks, and each could potentially be just as important.

In this paper, we studied the effect overlapping receptive fields have on the expressivity of the network, and found that having them, and more broadly denser connectivity, results in an exponential gain in the expressivity that is orthogonal to the depth.

Our analysis sheds light on many trends and practices in contemporary design of neural networks.

Previous studies have shown that non-overlapping architectures are already universal BID5 , and even have certain advantages in terms of optimization BID0 , and yet, real-world usage of non-overlapping networks is scarce.

Though there could be multiple factors involved, our results clearly suggest that the main culprit is that non-overlapping networks are significantly handicapped in terms of expressivity compared to overlapping ones, explaining why the former are so rarely used.

Additionally, when examining the networks that are commonly used in practice, where the majority of the layers are of the convolutional type with very small receptive field, and only few if any fully-connected layers BID18 BID20 He et al., 2016) , we find that though they are obviously overlapping, their overlapping degree is rather low.

We showed that while denser connectivity can increase the expressive capacity, even in the most common types of modern architectures already exhibit exponential increase in expressivity, without relying on fully-connected layers.

This could partly explain that somewhat surprising observation, as it is probable that such networks are sufficiently expressive for most practical needs simply because they are already in the exponential regime of expressivity.

Indeed, our experiments seems to suggests the same, in which we saw that further increases in the overlapping degree beyond the most limited overlapping case seems to have insignificant effects on performance -a conjecture not quite proven by our current work, but one we wish to investigate in the future.

There are relatively few other works which have studied the role of receptive fields in neural networks.

Several empirical works BID9 BID8 have demonstrated similar behavior, showing that the classification accuracy of networks can sharply decline as the degree of overlaps is decreased, while also showing that gains from using very large local receptive fields are insignificant compared to the increase in computational resources.

Other works studying the receptive fields of neural networks have mainly focused on how to learn them from the data Jia et al., 2012) .

While our analysis has no direct implications to those specific works, it does lay the ground work for potentially guiding architecture design, through quantifying the expressivity of any given architecture.

Lastly, BID11 studied the effective total receptive field of different layers, a property of a similar nature to our total receptive field, where they measure the the degree to which each input pixel is affecting the output of each activation.

They show that under common random initialization of the weights, the effective total receptive field has a gaussian shape and is much smaller than the maximal total receptive field.

They additionally demonstrate that during training the effective total receptive field grows in size, and suggests that weights should be initialized such that the initial effective receptive field is large.

Their results strengthen our theory, by showing that trained networks tend to maximize their effective receptive field, taking full potential of their expressive capacity.

To conclude, we have shown both theoretically and empirically that overlapping architectures have an expressive advantage compared to non-overlapping ones.

Our theoretical analysis is grounded on the framework of ConvACs, which we extend to overlapping configurations.

Though are proofs are limited to this specific case, previous studies have already shown that such results could be transferred to standard ConvNets as well, using most of the same mathematical machinery.

While adapting our analysis accordingly is left for future work, our experiments on standard ConvNets (see sec. 5) already suggest that the core of our results should hold in this case as well.

Finally, an interesting outcome of moving from non-overlapping architectures to overlapping ones is that the depth of a network is no longer capped at log 2 (input size), as has been the case in the models investigated by Cohen et al. DISPLAYFORM0 Figure 7: The original Convolutional Arithmetic Circuits as presented by BID5 .

We base our analysis on the convolutional arithmetic circuit (ConvAC) architecture introduced by BID5 , which is illustrated by fig. 7 , and can be simply thought of as a regular ConvNet, but with linear activations and product pooling layers, instead of the more common non-linear activations (e.g. ReLU) and average/max pooling.

More specifically, each point in the input space of the network, denoted by X = (x1, . . .

, xN ), is represented as an N -length sequence of s-dimensional vectors x1, . . .

, xN ∈ R s .

X is typically thought of as an image, where each xi corresponds to a local patches from that image.

The first layer of the network is referred to as the representation layer, consisting of applying M representation functions f θ 1 , . . .

, f θ M : R s → R on each local patch xi, giving rise to M feature maps.

Under the common setting, where the representation functions are selected to be DISPLAYFORM0 s × R, the representation layer reduces to the standard convolutional layer.

Other possibilities, e.g. gaussian functions with diagonal covariances, have also been considered in BID5 .

Following the representation layer, are hidden layers indexed by l = 0, . . .

, L − 1, each begins with a 1 × 1 conv operator, which is just an r l−1 × 1 × 1 convolutional layer with r l−1 input channels and r l output channels, with the sole exception that parameters of each kernel could be spatially unshared (known as locally-connected layer BID22 ).

Following each conv layer is a spatial pooling, that takes products of non-overlapping two-dimensional windows covering the output of the previous layer, where for l = L − 1 the pooling window is the size of the entire spatial dimension (i.e. global pooling), reducing its output's shape to a rL−1 × 1 × 1, i.e. an rL−1-dimensional vector.

The final L layer maps this vector with a dense linear layer into the Y network outputs, denoted by hy(x1, . . .

, xN ), representing score functions classifying each X to one of the classes through: y * = argmax y hy(x1, . . .

, xN ).

As shown in BID5 , these functions have the following form: DISPLAYFORM1 where A y , called the coefficients tensor, is a tensor of order N and dimension M in each mode, which for the sake of discussion can simply be seen as a multi-dimensional array, specified by N indices d1, . . .

, dN each ranging in {1, . . .

, M }, with entries given by polynomials in the network's conv weights.

A byproduct of eq. 5 is that for a fixed set of M representation functions, all functions represented by ConvACs lay in the same subspace of functions.

From theorem 1 we learn that overlaps give rise to networks which almost always cannot be efficiently implemented by non-overlapping ConvAC with standard pooling geometry.

However, as proven by BID4 , a ConvAC that uses a different pooling geometry -i.e.

the input to the pooling layers are not strictly contiguous windows from the previous layer -also cannot be efficiently implemented by the standard ConvAC with standard pooling geometry.

This raises the question of whether overlapping operations are simply equivalent to a ConvAC with a different pooling geometry and nothing more.

We answer this question in two parts.

First, a ConvAC with a different pooling geometry might be able to implement some function more efficiently than ConvAC with standard pooling geometry, however, the reverse is also true, that a ConvAC with standard pooling can implement some functions more efficiently than ConvAC with alternative pooling.

In contrast, a ConvAC that uses overlaps is still capable to implement efficiently any function that a non-overlapping ConvAC with standard pooling can.

Second, we can also show that some overlapping architectures are exponentially efficient than any non-overlapping ConvAC regardless of its pooling geometry.

This is accomplished by first extending lemma 1 to this case: Lemma 2.

Under the same conditions as lemma 1, if for all partitions P · ∪ Q such that |P | = |Q| = N /2 it holds that rank ( A(hy) P,Q) ≥ T , then any non-overlapping ConvAC regardless of its pooling geometry must have at least T channels in its next to last layer to induce the same grid tensor.

Next, in theorem 2 below show that some overlapping architectures can induce grid tensors whose matricized rank is exponential for any equal partition of its indices, proving they are indeed exponentially more efficient:Theorem 2.

Under the same settings as theorem 1, consider a GC network whose representation layer is followed by a GC layer with local receptive field H × H, stride 1 × 1, and D ≥ M output channels, whose parameters are "unshared", i.e. unique to each spatial location in the output of the layer as opposed to shared across them, followed by (L − 1) arbitrary GC layers, whose final output is a scalar.

For any choice of the parameters, except a null set (with respect to the Lebesgue measure) and for any template vectors such that F is non-singular, then the matricized rank of the induced grid tensor is equal to M H 2 2 , for any equal partition of the indices.

The exact same result holds if the parameters of the first GC layers are "shared" and D ≥ M · H 2 .Proof sketch.

We follow the same steps of our proof of theorem 1, however, we do not construct just one specific overlapping network that attains a rank of D ≥ M · H 2 , for all possible matricizations of the induced grid tensor.

Instead, we construct a separate network for each possible matricization.

This proves that with respect to the Lebesgue measure over the network's parameters space, separately for each pooling geometry, the set of parameters for which the lower bound does not hold is of measure zero.

Since a finite union of zero measured sets is also of measure zero, then the lower bound with respect to all possible pooling geometries holds almost everywhere, which concludes the proof sketch.

See app.

C.5 for a complete proof.

It is important to note that though the above theorem shows that pooling geometry on its own is less expressive than overlapping networks with standard pooling, it does not mean that pooling geometry is irrelevant.

Specifically, we do not yet know the effect of combining both overlaps and alternative pooling geometries together.

Additionally, many times sufficient expressivity is not the main obstacle for solving a specific task, and the inductive bias induced by a carefully chosen pooling geometry could help reduce overfitting.

In this section we present our proofs for the theorems and claims stated in the body of the article.

In this section we lay out the preliminaries required to understand the proofs in the following sections.

We begin with a limited introduction to tensor analysis, followed by quoting a few relevant known results relating tensors to ConvACs.

We begin with basic definitions and operations relating to tensors.

Let A ∈ R M 1 ⊗···⊗M N be a tensor of order DISPLAYFORM0 (1) and A (2) of orders N (1) and N (2) , and dimensions M DISPLAYFORM1 , respectively, we define their tensor product A (1) ⊗ A (2) as the order DISPLAYFORM2 tensor, where DISPLAYFORM3 For a set of vectors DISPLAYFORM4 is called an elementary tensor, or rank-1 tensor.

More generally, any tensor can be represented as a linear combination of rank-1 tensors, i.e. A = Z z=1 v (Z,1) ⊗· · ·⊗v (Z,1) , known as rank-1 decomposition, or CP decomposition, where the minimal Z for which this equality holds is knows as the tensor rank of A. Given a set of matrices DISPLAYFORM5 such that for any elementary tensor A, with notations as above, it holds that: DISPLAYFORM6 F(A) is defined for a general tensor A through its rank-1 decomposition comprising elementary tensors and applying F on each of them, which can be shown to be equivalent to DISPLAYFORM7 A central concept in tensor analysis is that of tensor matricization.

Let P · ∪Q = [N ] be a disjoint partition of its indices, such that P = {p1, . . .

, p |P | } with p1 < . . .

< p |P | , and Q = {q1, . . .

, q |Q| } with q1 < . . .

< q |Q| .The matricization of A with respect to the partition P · ∪ Q, denoted by A P,Q, is the Applying the matricization operator · P,Q on the tensor product operator results in the Kronecker Product, i.e. for an N -ordered tensor A, a K-ordered tensor B, and the partition P · ∪ Q = [N + K], it holds that DISPLAYFORM8 where P − N and Q − N are simply the sets obtained by subtracting the number N from every element of P or Q, respectively.

In concrete terms, the Kronecker product for the matrices A ∈ R M 1 ×M 2 and B ∈ R N 1 ×N 2 results in the matrix A B ∈ R M 1 N 1 ×M 2 N 2 holding AijB kl in row index (i − 1)N1 + k and column index (j − 1)N2 + l. An important property of the Kronecker product is that rank (A B) = rank (A) · rank (B).

Typically, when wish to compute rank ( A P,Q), we will first decompose it to a Kronecker product of matrices.

For a linear transform F, as defined above in eq.6, and a partition P · ∪ Q, if F (1) , . . .

, F (N ) are non-singular matrices, then F is invertible and the matrix rank of A P,Q equals to the matrix rank of F(A) P,Q (see proof in Hackbusch (2012) ).

Finally, we define the concept of grid tensors: for a function f : R s × · · · × R s → R and a set of template vectors DISPLAYFORM9 In the context of ConvACs, circuits and the functions they can realize are typically examined through the matricization of the grid tensors they induce.

The following is a succinct summary of the relevant known results used in our proofs -for a more detailed discussion, see previous works BID5 .

Using the same notations from eq. 5 describing a general ConvAC, let A y be the coefficients tensor of order N and dimension M in each mode, and let f θ 1 , . . .

, f θ M :R s →R be a set of M representation functions (see app.

A).

Under the above definitions, a non-overlapping ConvAC can be said to decompose the coefficients tensor A y .

Different network architectures correspond to known tensor decompositions: shallow networks corresponds to rank-1 decompositions, and deep networks corresponds to Hierarchical Tucker decompositions.

In BID4 , it was found that the matrix rank of the matricization of the coefficients tensors A y could serve as a bound for the size of networks decomposing A y .

For the conventional non-overlapping ConvAC and the contiguous "low-high" partition P = {1, . . .

, N /2}, Q = { N /2 + 1, . . . , N } of [N ], the rank of the matricization A y P,Q serves as a lower-bound on the number of channels of the next to last layer of any network which decomposes the coefficients tensor A. In the common case of square inputs, i.e. the input is of shape H × H and N = H 2 , it is more natural to represent indices by pairs (j, i) denoting the spatial location of each "patch" x (j,i) , where the first argument denotes the vertical location and the second denotes the horizontal location.

Under such setting the equivalent "low-high" partitions are either the "left-right" partition, DISPLAYFORM10 }.

More generally, when considering networks using other pooling geometries, i.e. not strictly contiguous pooling windows, then for each pooling geometry there exists a corresponding partition P · ∪ Q such that rank ( A y P,Q) serves as its respective lower-bound.

Though the results in BID4 are strictly based on the matricization rank of the coefficients tensors, they can be transferred to the matricization rank of grid tensors as well.

Grid tensors were first considered for analyzing ConvACs in .

For a set of M template vectors DISPLAYFORM11 With the above notations in place, we can write the the grid tensor A(hy) for the function hy(x1, . . .

, xN ) as: DISPLAYFORM12 If the representation functions are linearly independent and continuous, then we can choose the template vectors such that F is non-singular (see ), which according to the previous discussion on tensor matricization, means that for any partition P · ∪ Q and any coefficients tensor A y , it holds that rank ( A y P,Q) = rank ( A(hy) P,Q).

Thus, any lower bound on the matricization rank of the grid tensor translates to a lower bound on the matricization rank of the coefficients tensors, which in turn serves as lower bound on the size of non-overlapping ConvACs.

The above discussion leads to the proof of lemma 1 and lemma 2 that were previously stated:Proof of lemma 1 and lemma 2.

For the proofs of the base results with respect to the coefficients tensor, see BID4 .

To prove it is possible to choose the template vectors such that F is non-singular, see .

To prove that if F is non-singular, then the grid tensor and the coefficients tensor have the same matricization rank, see lemma 5.6 in Hackbusch (2012).We additionally quote the following lemma regarding the prevalence of the maximal matrix rank for matrices whose entries are polynomial functions: Lemma 3.

Let M, N, K ∈ N, 1 ≤ r ≤ min{M, N } and a polynomial mapping A : DISPLAYFORM13 If there exists a point x ∈ R K such that rank (A(x)) ≥ r, then the set {x ∈ R K |rankA(x) < r} has zero measure (with respect to the Lebesgue measure over R K ).Proof.

See BID17 .Finally, we simplify the notations of the GC layer with product pooling function for the benefit of following our proofs below.

We will represent the parameters of the l'th GC layer by {(w (l,c) ∈ R DISPLAYFORM14 , where w (l,c) represents the weights and b (l,c) the biases.

Let X ∈ R DISPLAYFORM15 be the input to the layer and Y ∈ R DISPLAYFORM16 be the output, then the following equality holds: DISPLAYFORM17 The above treats the common case where the parameters are shared across all spatial locations, but sometimes we wish to consider the "unshared" case, in which there is are different weights and biases for each location, which we denote by {(w (l,c,u,v) ∈ R DISPLAYFORM18 With the above definitions and lemmas in place, we are ready to prove the propositions and theorems from the body of the article.

Proposition 1 is a direct corollary of the following two claims: DISPLAYFORM0 be a function realized by a single GC layer with R × R local receptive field, S × S stride, and D (out) output channels, that is parameterized by DISPLAYFORM1 .

For allR ≥ R, a GC layer withR×R local receptive field, S×S stride, and C output channels, parameterized by {( DISPLAYFORM2 , could also realize f .

The same is true for the unshared case of both layers.

Proof.

The claim is trivially satisfied by settingw (c) such that it is equal to w (c) in all matching coordinates, while using zeros for all other coordinates.

Similarly, we setb (c) to be equal to b (c) in all matching coordinates, while using ones for all other coordinates.

DISPLAYFORM3 be a function realized by a GC layer with R×R local receptive field and 1×1 stride, parameterized by {(w (c) , DISPLAYFORM4 .

Then there exists an assignment to (w, b) such that f is the identity function f (X) = X. The same is true for the unshared case of both layers.

Proof.

From claim 1 it is sufficient to show the above holds for R = 1.

Indeed, setting DISPLAYFORM5 and b (c) ≡ 0 satisfies the claim.

We wish to show that for all choices of parameters, except a null set (with respect to Lebesgue measure) the grid tensor induced by the given GC network has rank satisfying eq. 4.

Since the entries of the matricized grid tensor are polynomial function of its parameters, then according to lemma 3, it is sufficient to find a single example that achieves this bound.

Hence, our proof is simply the construction of such an example.

Recall that the template vectors must hold that for the matrix F , defined by Fij = fj(x (i) ), where {fj} M j=1are the representation matrices, is a non-singular matrix.

We additionally assume in the following claims that the output of the representation layer is of width and height equal to H ∈ N, where H is an even number -the claims and proofs can however be easily adapted to the more general case.

Assume a ConvAC as described in the theorem, with representation layer defined according to above, followed by L GC layers, where the l'th layer has a local receptive field of R (l) , a stride of S (l) , and D (l) output channels.

We first construct our example, that achieves the desired matricization rank, for the simpler case where the first layer following the representation layer has a local receptive field large enough, i.e. when it is larger than H 2 .

Recall that for the first layer the total receptive field is equal to its local receptive field.

In the context of theorem 1, this first layer satisfies the conditions necessary to produce the lower bound given in the theorem.

The specific construction is presented in the following claim, which relies on utilizing the large local receptive field to match each spatial location in the left side of the input with one from the right side, such that for each such pair, the respective output of the first layer will represent a mostly diagonal matrix.

We then set the rest of the parameters such that the output of the entire network is defined by a tensor product of mostly diagonal matrices.

Since the matricization rank of the tensor product of matrices is equal to the product of the individual ranks, it results in an exponential form of the rank as is given in the theorem.

Claim 3.

Assume a ConvAC as defined above, ending with a single scalar output.

For all l ∈ [L], the parameters of the l-th GC layer are denoted by {(w (l,c) ∈ R DISPLAYFORM0 .

Let h(x1, . . .

, xN ) be the function realized the output of network.

Additionally define R ≡ R(1) , S ≡ S (1) and DISPLAYFORM1 , and the weights w (1,c) and biases b (1,c) of the first GC layer layer are set to: DISPLAYFORM2 , then there exists an assignment to α and the parameters of the other GC layers such that DISPLAYFORM3 +1 · H S , where P · ∪ Q is either the "left-right" or "top-bottom" partition, and (ρ, τ ) equals to (1, R) or (R, 1), respectively.

Proof.

The proof for either the "left-right" or "top-bottom" partition is completely symmetric, thus it is enough to prove the claim for the "left-right" case, where (ρ, τ ) = (1, R).

We wish to compute the entry A(h) d FIG7 ,...,d (H,H) of the induced grid tensor for arbitrary indices d FIG7 , . . .

, d (H,H) .

Let O ∈ R M ×H×H be the 3-order tensor output of the representation layer, where Om,j,i = F d (j,i) ,m for the aforementioned indices and for all 1 ≤ i, j ≤ H and m ∈ [M ].We begin by setting the parameters of all layers following the first GC layer, such that they are equal to computing the sum along the channels axis of the output of the first GC layer, followed by a global product of all of the resulting sums.

To achieve this, we can first assume w.l.o.g.

that these layers are non-overlapping through proposition 1.

We then set the parameters of the second GC layer to w (2,c) = 1 and b (2,c) ≡ 0, i.e. all ones and all zeros, respectively, which is equivalent to taking the sum along the channels axis for each spatial location, followed by taking the products over non-overlapping local receptive fields of size R (2) .

For the other layers, we simply set them as to take just the output of the first channel of the output of the preceding layer, which is equal to setting their parameters to w DISPLAYFORM4 where we extended O with zero-padding for the cases where uS + j > H or vS + i > H, as mentioned in sec. 2.

Next, we go through the technical process of reducing eq. 8 to a product of matrices.

Substituting the values of w(1,c) dji and b(1,c) ji with those defined in the claim, and computing the value of g(u, v, c, j, i) results in: uS+j,vS+i) ,m c ≤ D and vS + R > H and (j, i) = (1, 1) β c ≤ D and vS + R > H and (j, i) DISPLAYFORM5 DISPLAYFORM6 c ≤ D and vS + R > H and (j, i) = (1, 1) DISPLAYFORM7 from which we derive: DISPLAYFORM8 At this point we branch into two cases.

If S divides R − 1, then for all u, v ∈ N such that vS + R ≤ H and uS < H, the above expression for f (u, v) and f (u, v +

) depends only on the indices d (uS+1,vS+1) and d (uS+1,vS+R) , while these two indices affect only the aforementioned expressions.

By denoting DISPLAYFORM0 ), we can write it as: , results in: DISPLAYFORM1 DISPLAYFORM2 , then A(h) P,Q equals to the Kronecker product of the matrices in {A (u,v) |0 ≤ uS < H, 0 ≤ vS ≤ H − R}, up to permutation of its rows and columns, which do not affect its matrix rank.

Thus, the matricization rank of A(h) satisfies: DISPLAYFORM3 which proves the claim for this case.

If S does not divide R − 1, then for all u, v ∈ N, such that vS + R ≤ H and uS < H, it holds that f (u, v) depends only on the indices d (uS+1,vS+1) and d (uS+1,vS+R) , and they affect only f (u, v).

Additionally, for all u, v ∈ N, such that H < vS + R, vS < H and uS < H, it holds that f (u, v) depends only on the index d (uS+1,vS+1) , and this index affects only f (u, v).

Let us denote A 1 [d=k] (1 < l < L) and (1 ≤ k ≤ D) and (j, i) = (1, 1) 1 [d=k] (1 < l < L) and (D < k ≤ 2D) and (j, i) = (1, R DISPLAYFORM4 Were we used the fact that η (l) = S (l) η (l−1) and ξ (l) = R (l) η (l−1) + ξ (l−1) − η (l−1) .

k,u+1,v+1 for k ≤ D and 0 ≤ u, v < H (L) equals to the output of the single GC layer specified in the claim: Finally, with the above two claims set in place, we can prove our main theorem: DISPLAYFORM0 Proof. (of theorem 1) Using claim 4 we can realize the networks from claim 3, for which the matricization rank for either partition equals to: DISPLAYFORM1 Since for any matricization [A (Ψ)] P,Q the entries of the matricization are polynomial functions with respect to the parameters of the network, then, according to lemma 3, the set of parameters of Φ, that does not attain the above rank, has zero measure.

Since the union of zero measured sets is also of measure zero, then all parameters except a set of zero measure attain this matricization rank for both partitions at once, concluding the proof.

Following theorem 1, to compute the lower bound for the network described in proposition 2, we need to find the first layer for which its total receptive field is greater than H /2, and then estimate its total stride and its Proof. (of proposition 2) From claim 5, we can infer which is the first B × B layer such that its receptive field is greater than H /2: DISPLAYFORM0 Combining the above with claim 7, results in: The limits and the special case for B ≤ H 5 DISPLAYFORM1

@highlight

We analyze how the degree of overlaps between the receptive fields of a convolutional network affects its expressive power.

@highlight

The paper studies the expressive power provided by "overlap" in convolution layers of DNNs by considering linear activations with product pooling.

@highlight

This paper analyzes the expressivity of convolutional arithmetic circuits and shows that an exponentialy large number of non-overlapping ConvACs are required to approximate the grid tensor of an overlapping ConvACs.