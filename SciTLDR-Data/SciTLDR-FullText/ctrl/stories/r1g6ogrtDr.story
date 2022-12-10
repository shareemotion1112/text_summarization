Equivariance is a nice property to have as it produces much more parameter efficient neural architectures and preserves the structure of the input through the feature mapping.

Even though some combinations of transformations might never appear (e.g. an upright face with a horizontal nose), current equivariant architectures consider the set of all possible transformations in a transformation group when learning feature representations.

Contrarily, the human visual system is able to attend to the set of relevant transformations occurring in the environment and utilizes this information to assist and improve object recognition.

Based on this observation, we modify conventional equivariant feature mappings such that they are able to attend to the set of co-occurring transformations in data and generalize this notion to act on groups consisting of multiple symmetries.

We show that our proposed co-attentive equivariant neural networks consistently outperform conventional rotation equivariant and rotation & reflection equivariant neural networks on rotated MNIST and CIFAR-10.

Thorough experimentation in the fields of psychology and neuroscience has provided support to the intuition that our visual perception and cognition systems are able to identify familiar objects despite modifications in size, location, background, viewpoint and lighting (Bruce & Humphreys, 1994) .

Interestingly, we are not just able to recognize such modified objects, but are able to characterize which modifications have been applied to them as well.

As an example, when we see a picture of a cat, we are not just able to tell that there is a cat in it, but also its position, its size, facts about the lighting conditions of the picture, and so forth.

Such observations suggest that the human visual system is equivariant to a large transformation group containing translation, rotation, scaling, among others.

In other words, the mental representation obtained by seeing a transformed version of an object, is equivalent to that of seeing the original object and transforming it mentally next.

These fascinating abilities exhibited by biological visual systems have inspired a large field of research towards the development of neural architectures able to replicate them.

Among these, the most popular and successful approach is the Convolutional Neural Network (CNN) (LeCun et al., 1989) , which incorporates equivariance to translation via convolution.

Unfortunately, in counterpart to the human visual system, CNNs do not exhibit equivariance to other transformations encountered in visual data (e.g. rotations).

Interestingly, however, if an ordinary CNN happens to learn rotated copies of the same filter, the stack of feature maps becomes equivariant to rotations even though individual feature maps are not (Cohen & Welling, 2016) .

Since ordinary CNNs must learn such rotated copies independently, they effectively utilize an important number of network parameters suboptimally to this end (see Fig. 3 in Krizhevsky et al. (2012) ).

Based on the idea that equivariance in CNNs can be extended to larger transformation groups by stacking convolutional feature maps, several approaches have emerged to extend equivariance to, e.g. planar rotations (Dieleman et al., 2016; Marcos et al., 2017; Weiler et al., 2018; Li et al., 2018) , spherical rotations (Cohen et al., 2018; Worrall & Brostow, 2018; Cohen et al., 2019) , scaling (Marcos et al., 2018; Worrall & Welling, 2019) and general transformation groups (Cohen & Welling, 2016) , such that transformed copies of a single entity are not required to be learned independently.

Figure 1: Our visual system infers object identities according to their size, location and orientation in a scene.

In this blurred picture, observers describe the scene as containing a car and a pedestrian in the street.

However, the pedestrian is in fact the same shape as the car, except for a 90

• rotation.

The atypicality of this orientation for a car within the context defined by the street scene causes the car to be recognized as a pedestrian.

Extracted from Oliva & Torralba (2007) .

Although incorporating equivariance to arbitrary transformation groups is conceptually and theoretically similar 1 , evidence from real-world experiences motivating their integration might strongly differ.

Several studies in neuroscience and psychology have shown that our visual system does not react equally to all transformations we encounter in visual data.

Take, for instance, translation and rotation.

Although we easily recognize objects independently of their position of appearance, a large corpus of experimental research has shown that this is not always the case for in-plane rotations.

Yin (1969) showed that mono-oriented objects, i.e. complex objects such as faces which are customarily seen in one orientation, are much more difficult to be accurately recognized when presented upsidedown.

This behaviour has been reproduced, among others, for magazine covers (Dallett et al., 1968) , symbols (Henle, 1942) and even familiar faces (e.g. from classmates) (Brooks & Goldstein, 1963) .

Intriguingly, Schwarzer (2000) found that this effect exacerbates with age (adults suffer from this effect much more than children), but, adults are much faster and accurate in detecting mono-oriented objects in usual orientations.

Based on these studies, we draw the following conclusions:

• The human visual system does not perform (fully) equivariant feature transformations to visual data.

Consequently, it does not react equally to all possible input transformations encountered in visual data, even if they belong to the same transformation group (e.g. in-plane rotations).

• The human visual system does not just encode familiarity to objects but seems to learn through experience the poses in which these objects customarily appear in the environment to assist and improve object recognition (Freire et al., 2000; Riesenhuber et al., 2004; Sinha et al., 2006) .

Complementary studies (Tarr & Pinker, 1989; Oliva & Torralba, 2007) suggest that our visual system encodes orientation atypicality relative to the context rather than on an absolute manner (Fig. 1) .

Motivated by the aforementioned observations we state the co-occurrence envelope hypothesis:

The Co-occurrence Envelope Hypothesis.

By allowing equivariant feature mappings to detect transformations that co-occur in the data and focus learning on the set formed by these co-occurrent transformations (i.e. the co-occurrence envelope of the data), one is able to induce learning of more representative feature representations of the data, and, resultantly, enhance the descriptive power of neural networks utilizing them.

We refer to one such feature mapping as co-attentive equivariant.

Identifying the co-occurrence envelope.

Consider a rotation equivariant network receiving two copies of the same face (Fig. 2a) .

A conventional rotation equivariant network is required to perform inference and learning on the set of all possible orientations of the visual patterns constituting a face regardless of the input orientation (Fig. 2b) .

However, by virtue of its rotation equivariance, it is able to recognize rotated faces even if it is trained on upright faces only.

A possible strategy to simplify the task at hand could be to restrict the network to react exclusively to upright faces (Fig. 2c) .

In this case, the set of relevant visual pattern orientations becomes much smaller, at the expense of disrupting equivariance to the rotation group.

Resultantly, the network would risk becoming unable to detect faces in any other orientation than those it is trained on.

A better strategy results from restricting the set of relevant pattern orientations by defining them relative to one another (e.g. mouth

Figure 2: Effect of multiple attention strategies for the prioritization of relevant pattern orientations in rotation equivariant networks for the task of face recognition.

Given that all attention strategies are learned exclusively from upright faces, we show the set of relevant directions for the recognition of faces in two orientations (Fig. 2a) obtained by: no attention (Fig. 2b) , attending to the pattern orientations of appearance independently (Fig. 2c) and, attending to the pattern orientations of appearance relative to one another (Fig. 2d ).

Built upon Figure 1 from Schwarzer (2000) .

orientation w.r.t.

the eyes) as opposed to absolutely (e.g. upright mouth) (Fig. 2d ).

In such a way, we are able to exploit information about orientation co-occurrences in the data without disrupting equivariance.

The set of co-occurrent orientations in Fig. 2d corresponds to the co-occurrence envelope of the samples in Fig. 2a for the transformation group defined by rotations.

In this work, we introduce co-attentive equivariant feature mappings and apply them on existing equivariant neural architectures.

To this end, we leverage the concept of attention (Bahdanau et al., 2014) and modify existing mathematical frameworks for equivariance, such that co-occurrent transformations can be detected.

It is critical not to disrupt equivariance in the attention procedure as to preserve it across the entire network.

To this end, we introduce cyclic equivariant self-attention, a novel attention mechanism able to preserve equivariance to a large set of transformation groups.

Experiments and results.

We explore the effects of co-attentive equivariant feature mappings for single and multiple symmetry groups.

Specifically, we replace conventional rotation equivariant mappings in p4-CNNs (Cohen & Welling, 2016) and DRENs (Li et al., 2018) with co-attentive ones.

We show that co-attentive rotation equivariant neural networks consistently outperform their conventional counterparts in fully (rotated MNIST) and partially (CIFAR-10) rotational settings.

Subsequently, we generalize cyclic equivariant self-attention to multiple similarity groups and apply it on p4m-CNNs (Cohen & Welling, 2016 ) (equivariant to rotation and mirror reflections).

Our results are in line with those obtained for single symmetry groups and support our stated hypothesis.

• We propose the co-occurrence envelope hypothesis and demonstrate that conventional equivariant mappings are consistently outperformed by our proposed co-attentive equivariant ones.

• We generalize co-attentive equivariant mappings to multiple symmetry groups and provide, to the best of our knowledge, the first attention mechanism acting generally on symmetry groups.

Equivariance.

We say that a feature mapping f : X → Y is equivariant to a (transformation) group G (or G-equivariant) if it commutes with actions of the group G acting on its domain and codomain:

where

g denotes a group action in the corresponding space.

In other words, the ordering in which we apply a group action T g and the feature mapping f is inconsequential.

There are multiple reasons as of why equivariant feature representations are advantageous for learning systems.

Since group actions T X g produce predictable and interpretable transformations T Y g in the feature space, the hypothesis space of the model is reduced (Weiler et al., 2018) and the learning process simplified (Worrall et al., 2017) .

Moreover, equivariance allows the construction of L-layered networks by stacking several equivariant feature mappings {f (1) , ..., f (l) , ..., f (L) } such that the input structure as regarded by the group G is preserved (e.g. CNNs and input translations).

As a result, an arbitrary intermediate

is able to take advantage of the structure of x as well.

Invariance is an special case of equivariance in which T Y g = Id Y , and thus all group actions in the input space are mapped to the same feature representation.

Equivariant neural networks.

In neural networks, the integration of equivariance to arbitrary groups G has been achieved by developing feature mappings f that utilize the actions of the group G in the feature mapping itself.

Interestingly, equivariant feature mappings encode equivariance as parameter sharing with respect to G, i.e. the same weights are reused for all g ∈ G. This makes the inclusion of larger groups extremely appealing in the context of parameter efficient networks.

Conventionally, the l-th layer of a neural network receives a signal x (l) (u, λ) (where u ∈ Z 2 is the spatial position and λ ∈ Λ l is the unstructured channel index, e.g. RGB channels in a color image), and applies a feature mapping

to generate the feature representation

T is defined by a convolution 2 ( R 2 ) between the input signal x (l) and a learnable convolutional filter W

By sliding W

λ ,λ across u, CNNs are able to preserve the spatial structure of the input x through the feature mapping f l T and successfully provide equivariance to the translation group T = (Z 2 , +).

The underlying idea for the extension of equivariance to larger groups in CNNs is conceptually equivalent to the strategy utilized by LeCun et al. (1989) for translation equivariance.

Consider, for instance, the inclusion of equivariance to the set of rotations by θ r degrees 3 :

.

To this end, we modify the feature mapping

λ ,λ (u, r) be the input and the convolutional filter with an affixed index r for rotation.

The roto-translational convolution

R is defined as:

R produces (dim(Θ) = r max ) times more output feature maps than f (l)

T , we need to learn much smaller convolutional filters W (l) λ ,λ to produce the same number of output feature channels.

Learning equivariant neural networks.

Consider the change of variables

Eq. 2 and Eq. 3, respectively.

In general, neural networks are learned via backpropagation (LeCun et al., 1989) by iteratively applying the chain rule of derivation to update the network parameters.

Intuitively, the networks outlined in Eq. 2 and Eq. 3 obtain feedback from all g ∈ G and, resultantly, are inclined to learn feature representations that perform optimally on the entire group G. However, as outlined in Fig. 2 and Section 1, several of those feature combinations are not likely to simultaneously appear.

Resultantly, the hypothesis space of the model as defined by Weiler et al. (2018) might be further reduced.

Note that this reasoning is tightly related to existing explanations for the large success of spatial (Xu et al., 2015; Woo et al., 2018; Zhang et al., 2018) and temporal (Luong et al., 2015; Vaswani et al., 2017; Mishra et al., 2017; Zhang et al., 2018) attention in deep learning architectures.

In this section we define co-attentive feature mappings and apply them in the context of equivariant neural networks (Figure 3) .

To this end, we introduce cyclic equivariant self-attention and utilize it to construct co-attentive rotation equivariant neural networks.

Subsequently, we show that cyclic equivariant self-attention is extendable to larger symmetry groups and make use of this fact to construct co-attentive neural networks equivariant to rotations and mirror reflections.

2 Formally it is as a correlation.

However, we hold on to the standard deep learning terminology.

3 The reader may easily verify that Θ (and hence Z 2 × Θ) forms a group.

Figure 3: Co-attentive equivariant feature mappings acting on the groups p4 (top) and p4m (bottom).

In order to learn co-attentive equivariant representations, cyclic equivariant self-attention A C is applied on top of the output of a conventional equivariant feature mapping (here p4 and p4m group convolutions, respectively).

Resultantly, the group convolution responses are modulated based on their assessed relevance.

For multiple symmetry groups, the group convolution responses must be rearranged in a vector structure so that the permutation laws of A C correspond to those of the composing group symmetries.

Same colors in A C denote equal weights.

The circulant (block) structure of A C ensures that equivariance to the corresponding group is preserved through the course of attention.

Consequently, if the input is rotated (or mirrored in p4m), the attention mask shown here is transformed accordingly.

Built upon Figures 1 and 2 from Cohen & Welling (2016) .

To allow rotation equivariant networks to utilize and learn co-attentive equivariant representations, we introduce an attention operator A (l) on top of the roto-translational convolution f

R with which discernment along the rotation axis r of the generated feature responses x (l) (u, r, λ) is possible.

Formally, our co-attentive rotation equivariant feature mapping f

R is defined as follows:

Theoretically, A (l) could be defined globally over f

R (x (l) ) (i.e. simultaneously along u, r, λ) as depicted in Eq. 4.

However, we apply attention locally to: (1) grant the algorithm enough flexibility to attend locally to the co-occurrence envelope of feature representations and, (2) utilize attention exclusively along the rotation axis r, such that our contributions are clearly separated from those possibly emerging from spatial attention.

To this end, we apply attention pixel-wise on top of f

λ to each learned feature representation and utilize it across the spatial dimension of the output feature maps 4 :

Attention and self-attention.

Consider a source vector x = (x 1 , ..., x n ) and a target vector y = (y 1 , ..., y m ).

In general, an attention operator A leverages information from the source vector x (or multiple feature mappings thereof) to estimate an attention matrix A ∈ [0, 1] n×m , such that: (1) the element A i,j provides an importance assessment of the source element x i with reference to the target element y j and (2) the sum of importance over all x i is equal to one:

i A i,j = 1.

Subsequently, the matrix A is utilized to modulate the original source vector x as to attend to a subset of relevant source positions with regard to y j :x j = (A :,j ) T x (where is the Hadamard product).

A special case of attention is that of self-attention (Cheng et al., 2016) , in which the target and the source vectors are equal (y := x).

In other words, the attention mechanism estimates the influence of the sequence x on the element x j for its weighting.

In general, the attention matrix 5 A ∈ [0, 1] n×m is constructed via nonlinear space transformations fÃ : R n → R n×m of the source vector x, on top of which the softmax function is applied: A :,j = softmax(fÃ(x) :,j ).

This ensures that the properties previously mentioned hold.

Typically, the mappings fÃ found in literature take feature transformation pairs of x as input (e.g. {s, H} in RNNs (Luong et al., 2015) , {Q, K} in self-attention networks (Vaswani et al., 2017) ), and perform (non)-linear mappings on top of it, ranging from multiple feed-forward layers (Bahdanau et al., 2014) to several operations between the transformed pairs (Luong et al., 2015; Vaswani et al., 2017; Mishra et al., 2017; Zhang et al., 2018) .

Due to the computational complexity of these approaches and the fact that we do extensive pixel-wise usage of fÃ on every network layer, their direct integration in our framework is computationally prohibitive.

To circumvent this problem, we modify the usual self-attention formulation as to enhance its descriptive power in a much more compact setting.

Compact local self-attention.

Initially, we relax the range of values of A from [0, 1] n×n to R n×n .

This allows us to encode much richer relationships between element pairs (x i , x j ) at the cost of less interpretability.

Subsequently, we define A = x T Ã , whereÃ ∈ R n×n is a matrix of learnable parameters.

Furthermore, instead of directly applying softmax on the columns of A, we first sum over the contributions of each element x i to obtain a vector a = { i A i,j } n j=1 , which is then passed to the softmax function.

Following Vaswani et al. (2017) , we prevent the softmax function from reaching regions of low gradient by scaling its argument by ( dim(A)) −1 = (1 / n): a = softmax((1 / n) a).

Lastly, we counteract the contractive behaviour of the softmax function by normalizingã before weighting x as to preserve the magnitude range of its argument.

This allows us to use A in deep architectures.

Our compact self-attention mechanism is summarized as follows:

The cyclic equivariant self-attention operator A C .

Consider {x(u, r, λ)} rmax r=1 , the vector of responses generated by a roto-translational convolution f R stacked along the rotation axis r. By applying self-attention along r, we are able to generate an importance matrix A ∈ R rmax×rmax relating all pairs of (θ i , θ j )-rotated responses in the rotational group Θ at a certain position.

We refer to this attention mechanism as full self-attention (A F ).

Although A F is able to encode arbitrary linear source-target relationships for each target position, it is not restricted to conserve equivariance to Θ. Resultantly, we risk incurring into the behavior outlined in Fig. 2c .

Before we further elaborate on this issue, we introduce the cyclic permutation operator P i , which induces a cyclic shift of i positions on its argument: σ

Consider a full self-attention operator A F acting on top of a roto-translational convolution f R .

Let p be an input pattern to which f R only produces a strong activation in the feature map x(r) = f R (p)(r),r ∈ {r} rmax r=1 .

Intuitively, during learning, only the corresponding attention coefficients A ;,r in A F would be significantly increased.

Now, consider the presence of the input pattern θ i p, a θ i -rotated variant of p. By virtue of the rotational equivariance property of the feature mapping f R , we obtain (locally) an exactly equal response to that of p up to a cyclic permutation of i positions on r, and thus, we obtain a strong activation in the feature map P i (x(r)) = x(σ P i (r)).

We encounter two problems in this setting: A F is not be able to detect that p and θ i p correspond to the exact same input pattern and, as each but the attention coefficientsÃ :,j is small, the network might considerably damp the response generated by θ i p.

As a result, the network might (1) squander important feedback information during learning and (2) induce learning of repeated versions of the same pattern for different orientations.

In other words, A F does not behave predictively as a function of θ i .

Interestingly, we are able to introduce prior-knowledge into the attention model by restricting the structure ofÃ. By leveraging the idea of equivariance to the cyclic group C n , we are able to solve the problems exhibited by A F and simultaneously reduce the number of additional parameters required by the self-attention mechanism (from r 2 max to r max ).

Consider again the input patterns p and θ i p.

We incorporate the intuition that p and θ i p are one and the same entity, and thus, f R (locally) generates the same output feature map up to a cyclic permutation

Consequently, the attention mechanism should produce the exact same output for both p and θ i p up to the same cyclic permutation P i .

In other words, A (and thusÃ) should be equivariant to cyclic permutations.

, 2001; Åhlander & Munthe-Kaas, 2005) .

We make use of this knowledge and leverage the concept of circulant matrices to impose cyclic equivariance to the structure ofÃ. Formally, a circulant matrix C ∈ R n×n is composed of n cyclic permutations of its defining vector c = {c i } n i=1 , such that its j-th column is a cyclic permutation of j − 1 positions of c: C :,j = P j−1 (c) T .

We construct our cyclic equivariant self-attention operator A C by defining A as a circulant matrix specified by a learnable attention vector a C = {a

and subsequently applying Eqs. 6 -8.

Resultantly, A C is able to assign the responses generated by f R for rotated versions of an input pattern p to a unique entity: f R (θ i p) = P i (f R (p)), and dynamically adjust its output to the angle of appearance θ i , such that the attention operation does not disrupt its propagation downstream the network:

.

Consequently, the attention weights a C are updated equally regardless of specific values of θ i .

Due to these properties, A C does not incur in any of the problems outlined earlier in this section.

Conclusively, our co-attentive rotation equivariant feature mapping f

R is defined as follows:

Note that a co-attentive equivariant feature mapping f R is approximately equal (up to a normalized softmax operation (Eq. 8)) to a conventional equivariant one f R , ifÃ = αI for any α ∈ R.

The self-attention mechanisms outlined in the previous section are easily extendable to larger groups consisting of multiple symmetries.

Consider, for instance, the group θ r m of rotations by θ r degrees and mirror reflections m defined analogously to the group p4m in Cohen & Welling (2016) .

Let p(u, r, m, λ) be an input signal with an affixed index m ∈ {m 0 , m 1 } for mirror reflections (m 1 indicates mirrored) and f θrm be a group convolution (Cohen & Welling, 2016) on the θ r m group.

The group convolution f θrm produces two times as many output channels (2r max : m 0 r max +m 1 r max ) as those generated by the roto-translational convolution f R (Eq. 3) (see Figure 3 ).

Full self-attention A F can be integrated directly by modulating the output of f θrm as depicted in Section 3.1 withÃ ∈ R 2rmax×2rmax .

Here, A F relates the group convolution responses with one another.

However, just as for f R , A F disrupts the equivariance property of f θrm to the θ r m group.

Similarly, the cyclic equivariant self-attention operator A C can be extended to multiple symmetry groups as well.

Before we continue, we introduce the cyclic permutation operator P i,t , which induces a cyclic shift of i positions on its argument along the transformation axis t. Consider the input patterns p and θ i p outlined in the previous section and mp, a mirrored instance of p. Let x(u, r, m, λ) = f θrm (p)(u, r, m, λ)

be the response of the group convolution f θrm for the input pattern p. By virtue of the rotation equivariance property of f θrm , the generated response for θ i p is equivalent to that of p up to a cyclic permutation of i positions along the rotation axis r: f θrm (θ i p)(u, r, m, λ) = P i,r (f θrm (p))(u, r, m, λ) = x(u, σ P i (r), m, λ).

Similarly, by virtue of the mirror equivariance property of f θrm , the response generated by mp is equivalent to that of p up to a cyclic permutation of one position along the mirroring axis m:

.

Note that if we take two elements from a group g, h, their composition (gh) is also an element of the group.

Resultantly,

In other words, in order to extend A C to the θ r m group, it is necessary to restrict the structure of A such that it respects the permutation laws imposed by the equivariant mapping f θrm .

Let us rewrite x(u, r, m, λ) as x(u, g, λ), g = (mr) ∈ {m 0 , m 1 } × {r} rmax r=1 .

In this case, we must impose a circulant block matrix structure onÃ such that: (1) the composing blocks permute internally as defined by P i,r and (2) the blocks themselves permute with one another as defined by P 1,m .

Formally,Ã is defined as:Ã

where {Ã i ∈ R rmax×rmax }, i ∈ {1, 2} are circulant matrices (Eq. 9).

Importantly, the ordering of the permutation laws inÃ is interchangeable if the input vector is modified accordingly, i.e. g = (rm).

Conclusively, cyclic equivariant self-attention A C is directly extendable to act on any G-equivariant feature mapping f G , and for any symmetry group G, if the group actions T Y g produce cyclic permutations on the codomain of f G .

To this end, one must restrict the structure ofÃ to that of a circulant block matrix, such that all permutation laws of T

Experimental Setup.

We validate our approach by exploring the effects of co-attentive equivariant feature mappings for single and multiple symmetry groups on existing equivariant architectures.

Specifically, we replace conventional rotation equivariant mappings in p4-CNNs (Cohen & Welling, 2016) and DRENs (Li et al., 2018) with co-attentive equivariant ones and evaluate their effects in fully (rotated MNIST) and partially (CIFAR-10) rotational settings.

Similarly, we evaluate coattentive equivariant maps acting on multiple similarity groups by replacing equivariant mappings in p4m-CNNs (Cohen & Welling, 2016 ) (equivariant to rotation and mirror reflections) likewise.

Unless otherwise specified, we replicate as close as possible the same data processing, initialization strategies, hyperparameter values and evaluation strategies utilized by the baselines in our experiments.

Note that the goal of this paper is to study and evaluate the relative effects obtained by co-attentive equivariant networks with regard to their conventional counterparts.

Accordingly, we do not perform any additional tuning relative to the baselines.

We believe that improvements on our reported results are feasible by performing further parameter tuning (e.g. on structure or hyperparameters) on the proposed co-attentive equivariant networks.

The additional learnable parameters, i.e. those associated to the cyclic self-attention operator (Ã) are initialized identically to the rest of the layer.

Subsequently, we replace the values ofÃ along the diagonal by 1 (i.e. diag(Ã init ) = 1) such thatÃ init approximately resembles the identity I and, hence, co-attentive equivariant layers are initially approximately equal to equivariant ones.

Rotated MNIST.

The rotated MNIST dataset (Larochelle et al., 2007) contains 62000 gray-scale 28x28 handwritten digits uniformly rotated on the entire circle [0, 2π).

The dataset is split into training, validation and tests sets of 10000, 2000 and 50000 samples, respectively.

We replace rotation equivariant layers in p4-CNN (Cohen & Welling, 2016) , DREN and DRENMaxPooling (Li et al., 2018) with co-attentive ones.

Our results show that co-attentive equivariant networks consistently outperform conventional ones (see Table 1 ).

CIFAR-10.

The CIFAR-10 dataset (Krizhevsky et al., 2009 ) consists of 60000 real-world 32x32 RGB images uniformly drawn from 10 classes.

Contrarily to the rotated MNIST dataset, this dataset does not exhibit rotation symmetry.

The dataset is split into training, validation and tests sets of 40000, 10000 and 10000 samples, respectively.

We replace equivariant layers in the p4 and p4m variations of the All-CNN (Springenberg et al., 2014) and the ResNet44 (He et al., 2016) proposed by Cohen & Welling (2016) with co-attentive ones.

Likewise, we modify the r x4-variations of the NIN (Lin et al., 2013) and ResNet20 (He et al., 2016 ) models proposed by Li et al. (2018) in the same manner.

Our results show that co-attentive equivariant networks consistently outperform conventional ones in this setting as well (see Table 1 ).

Training convergence of equivariant networks.

Li et al. (2018) reported that adding too many rotational equivariant (isotonic) layers decreased the performance of their models on CIFAR-10.

As a consequence, they did not report results on fully rotational equivariant networks for this setting and attributed this behaviour to the non-symmetricity of the data.

We noticed that, with equal initialization strategies, rotational equivariant CNNs were much more prone to divergence than ordinary CNNs.

This behaviour can be traced back to the additional feedback resulting from roto-translational convolutions (Eq. 3) compared to ordinary ones (Eq. 2).

After further analysis, we noticed that the data preprocessing strategy utilized by Li et al. (2018) leaves some very large outlier values in the data (|x| >100), which strongly contribute to the behaviour outlined before.

In order to evaluate the relative contribution of co-attentive equivariant neural networks we constructed fully equivariant DREN architectures based on their implementation.

Although the obtained results were much worse than those originally reported in Li et al. (2018) , we were able to stabilize training by clipping input values to the 99 percentile of the data (|x| ≤2.3) and reducing the learning rate to 0.01, such that the same hyperparameters could be used across all network types.

The obtained results (see Table 1 ) signalize that DREN networks are comparatively better than CNNs both in fully and partially rotational settings, contradictorily to the conclusions drawn in Li et al. (2018) .

This behaviour elucidates that although the inclusion of equivariance to larger transformation groups is beneficial both in terms of accuracy and parameter efficiency, one must be aware that such benefits are directly associated with an increase of the network susceptibility to divergence during training.

Our results show that co-attentive equivariant feature mappings can be utilized to enhance conventional equivariant ones.

Interestingly, co-attentive equivariant mappings are beneficial both in partially and fully rotational settings.

We attribute this to the fact that a set of co-occurring orientations between patterns can be easily defined (and exploited) in both settings.

It is important to note that we utilized attention independently over each spatial position u on the codomain of the corresponding group convolution.

Resultantly, we were restricted to mappings of the form xA, which, in turn, constraint our attention mechanism to have a circulant structure in order to preserve equivariance (since group actions acting in the codomain of the group convolution involve cyclic permutations and cyclic self-attention is applied in the codomain of the group convolution).

In future work, we want to extend the idea presented here to act on the entire group simultaneously (i.e. along u as well).

By doing so, we lift our current restriction to mappings of the form xA and therefore, may be able to develop attention instances with enhanced descriptive power.

Following the same line of though, we want to explore incorporating attention in the convolution operation itself.

Resultantly, one is not restricted to act exclusively on the codomain of the convolution, but instead, is able to impose structure in the domain of the mapping as well.

Naturally, such an approach could lead to enhanced descriptiveness of the incorporated attention mechanism.

Moreover, we want to utilize and extend more complex attention strategies (e.g. Bahdanau et al. (2014); Luong et al. (2015) ; Vaswani et al. (2017) ; Mishra et al. (2017) ) such that they can be applied to large transformation groups without disrupting equivariance.

As outlined earlier in Section 3.1, this becomes very challenging from a computational perspective as well, as it requires extensive usage of the corresponding attention mechanism.

Resultantly, an efficient implementation thereof is mandatory.

Furthermore, we want to extend co-attentive equivariant feature mappings to continuous (e.g. Worrall et al. (2017) ) and 3D space (e.g. Cohen et al. (2018) ; Worrall & Brostow (2018) ; Cohen et al. (2019) ) groups, and for applications other than visual data (e.g. speech recognition).

Published as a conference paper at ICLR 2020

Finally, we believe that our approach could be refined and extended to a first step towards dealing with the enumeration problem of large groups (Gens & Domingos, 2014) , such that functions acting on the group (e.g. group convolution) are approximated by evaluating them on the set of cooccurring transformations as opposed to on the entire group.

Such approximations are expected to be very accurate, as non-co-occurrent transformations are rare.

This could be though of as sharping up co-occurrent attention to co-occurrent restriction.

We have introduced the concept of co-attentive equivariant feature mapping and applied it in the context of equivariant neural networks.

By attending to the co-occurrence envelope of the data, we are able to improve the performance of conventional equivariant ones on fully (rotated MNIST) and partially (CIFAR-10) rotational settings.

We developed cyclic equivariant self-attention, an attention mechanism able to attend to the co-occurrence envelope of the data without disrupting equivariance to a large set of transformation groups (i.e. all transformation groups G, whose action in the codomain of a G-equivariant feature mapping produce cyclic permutations).

Our obtained results support the proposed co-occurrence envelope hypothesis.

A OBTAINING CO-OCCURRENT ATTENTION VIA EQUATION 5

Figure 4: Synchronous movement of feature mappings and attention masks as a function of input rotation in the group p4 (r max = 4).

In this section, we provide a meticulous description on how co-occurrent attention is obtained via the method presented in the paper.

Intuitively, a direct approach to address the problem illustrated in the introduction (Section 1) and Figure 2 requires an attention mechanism that acts simultaneously on r and λ (see Eq. 3).

However, we illustrate how the depicted problem can be simplified such that attention along r is sufficient by taking advantage of the equivariance property of the network.

Let p be the input of a roto-translational convolution f R : Z 2 × Θ × Λ 0 → Z 2 × Θ × Λ 1 as defined in Eq. 3, and Θ be the set of rotations by θ r degrees: Θ = {θ r = r 2π rmax } rmax r=1 .

Let f R (p)(u) ∈ R rmax×Λ1 be the matrix consisting of the r max oriented responses for each λ ∈ Λ 1 learned representation at a certain position u. Since the vectors f R (p)(u, λ) ∈ R rmax , λ ∈ Λ 1 permute cyclically as a result of the rotation equviariance property of f R , it is mandatory to ensure equivariance to cyclic permutations for each f R (p)(u, λ) during the course of the attention procedure (see Section 3).

At first sight, one is inclined to think that there is no connection between multiple vectors f R (p)(u, λ) in f R (p)(u), and, therefore, in order to exploit co-occurences, one must impose additional constraints along the λ axis.

However, there is indeed an implicit restriction in f R (p)(u) along λ resulting from the rotation equivariance property of the mapping f R , which we can take advantage from to simplify the problem at hand.

Consider, for instance, the input θ i p, a θ i -rotated version of p. By virtue of the equivariance property of f R , we have (locally) that f R (θ i p) = P i (f R (p)).

Furthermore, we know that this property must hold for all the learned feature representations f R (p)(u, λ),∀λ ∈ Λ 1 .

Resultantly, we have that:

In other words, if one of the learned mappings f R (p)(u, r, λ) experiences a permutation P i along r, all the learned representations f R (p)(u, r, λ), ∀λ ∈ Λ 1 must experience the exact same permutation P i as well.

Resultantly, the equivariance property of the mapping f R ensures that all the Λ 1 learned feature representations f R (p)(u, λ) "move synchronously" as a function of input rotation θ i .

Likewise, if we apply a cyclic equivariant attention mechanism A C λ independently on top of each λ learned representation f R (p)(u, λ), we obtain that the relation A C λ (f R (θ i p))(u, r, λ) = P i (A C λ (f R (p))(u, r, λ)) ∀λ ∈ Λ 1 (13) must hold as well.

Similarly to the case illustrated in Eq. 12 and given that A C λ is equivariant to cyclic permutations on the domain, we obtain that all the Λ 1 learned attention masks A C λ "move synchronously" as a function of input rotation θ i as well (see Fig. 4 ).

From Eq. 13 and Figure 4 , one can clearly see that by utilizing A C λ independently along r and taking advantage from the fact that all Λ 1 learned feature representations are tied with one another via f R , one is able to prioritize learning of feature representations that co-occur together as opposed to the much looser formulation in Eq. 12, where feedback is obtained from all orientations.

<|TLDR|>

@highlight

We utilize attention to restrict equivariant neural networks to the set or co-occurring transformations in data. 

@highlight

This paper combines attention with group equivariance, specifically looking at the p4m group of rotations, translations, and flips, and derives a form of self-attention that doesn't destroy the equivariance property.

@highlight

The authors propose a self-attention mechanism for rotation-equivariant neural nets that improves classification performance over regular rotation-equivariant nets.