The visual world is vast and varied, but its variations divide into structured and unstructured factors.

Structured factors, such as scale and orientation, admit clear theories and efficient representation design.

Unstructured factors, such as what it is that makes a cat look like a cat, are too complicated to model analytically, and so require free-form representation learning.

We compose structured Gaussian filters and free-form filters, optimized end-to-end, to factorize the representation for efficient yet general learning.

Our experiments on dynamic structure, in which the structured filters vary with the input, equal the accuracy of dynamic inference with more degrees of freedom while improving efficiency.



(Please see https://arxiv.org/abs/1904.11487 for the full edition.)

Although the visual world is varied, there is nevertheless ubiquitous structure.

Free-form learned representations are structure-agnostic, making them general, but their not harnessing structure is computationally and statistically inefficient.

Structured representations like steerable filtering BID5 BID6 , scattering BID0 , and steerable networks BID1 efficiently express certain structures, but are constrained.

We propose the semi-structured composition of Gaussian and free-form filters to blur the line between free-form and structured representations.

The effectiveness of strongly structured representations hinges on whether they encompass the true structure of the data.

If not, the representation is limiting, and subject to error.

At least, such is the case when structure substitutes for learning.

In this work we compose structured and free-form filters and learn both end-to-end ( FIG1 ).

The free-form parameters are not constrained by our composition for generality.

The structured parameters are low-dimensional for efficiency.

We choose Gaussian structure to represent the spatial structures of scale, aspect, and orientation through covariance BID7 .

Optimizing these structured covariance parameters carries out a form of differentiable architecture search over receptive fields.

Since this structure is lowdimensional, it is computationally efficient and could be learned from limited data.

Our composition f ?? ??? g ?? combines a free-form f ?? with a structured Gaussian g ?? .

This semistructured composition factorizes the representation into spatial Gaussian receptive fields and freeform features.

Composing filters in this fashion is a novel approach to making receptive field shape differentiable, low-dimensional, and decoupled from the number of parameters.

The structure of a Gaussian is controlled by its covariance ??, which for a spatial 2D Gaussian is Covariances come in families with progressively richer structure: spherical has one parameter for scale, diagonal has two parameters for scale and aspect, and full has three parameters for scale, aspect, and orientation/slant.

We compose free-form filters f ?? and structured Gaussian filters g ?? by convolution * to define a more general family of semi-structured filters than can be learned by either alone.

Our composition makes receptive field scale, aspect, and orientation differentiable in a low-dimensional parameterization for efficient end-to-end learning.heiro & Bates, 1996) is a good choice for iterative optimization because it is simple and quick to compute: ?? = U U for upper-triangular U with positive diagonal.

We can keep the diagonal positive by storing its log, hence log-Cholesky, and exponentiating it when forming ??.Composing with Convolution and Covariance The computation of our composition reduces to convolution, and so it inherits the efficiency of aggressively tuned convolution implementations.

Convolution is associative, so compositionally filtering an input I decomposes into two steps of convolution by DISPLAYFORM0 This decomposition has computational advantages.

The Gaussian step can be done by specialized filtering that harnesses separability, cascade smoothing, and other Gaussian structure.

Memory can be spared by only keeping the covariance parameters and recreating the Gaussian filters as needed (which is quick, although it is a space-time tradeoff).

Each compositional filter can always be explicitly formed by g ?? * f ?? for visualization (see FIG1 ) or other analysis.

Both ?? and ?? are differentiable for end-to-end learning.

Dynamic Gaussian Structure Semi-structured composition can learn a rich family of receptive fields, but visual structure is richer still, because structure locally varies while our filters are fixed.

Even a single image contains variations in scale and orientation, so one-size-and-shape-fits-all structure is suboptimal.

Dynamic inference replaces static, global parameters with dynamic, local parameters that are inferred from the input to adapt to these variations.

Composing with structure by convolution cannot locally adapt, since the filters are constant across the image.

We can nevertheless extend our composition to dynamic structure by representing local covariances and instantiating local Gaussians accordingly.

Our composition makes dynamic inference efficient by decoupling low-dimensional, Gaussian structure from high-dimensional, free-form filters.

There are two routes to dynamic Gaussian structure: local filtering and deformable sampling.

Local filtering has a different filter kernel for each position, as done by dynamic filter networks BID4 .

This ensures exact filtering for dynamic Gaussians, but is too computationally demanding for large-scale recognition networks.

Deformable sampling adjusts the position of filter taps by arbitrary offsets, as done by deformable convolution BID3 .

We exploit deformable sampling to dynamically form sparse approximations of Gaussians.

We constrain deformable sampling to Gaussian structure by setting the sampling points through covariance.

FIG3 illustrates these Gaussian deformations.

We relate the default deformation to the standard Gaussian by placing one point at the origin and circling it with a ring of eight points on the unit circle at equal distances and angles.

We consider the same progression of spherical, diagonal, and full covariance for dynamic structure.

This low-dimensional structure differs from the high degrees of freedom in a dynamic filter network, which sets free-form filter parameters, and deformable convolution, which sets free-form offsets.

In this way our semi-structured composition requires only a small, constant number of covariance parameters independent of the sampling resolution and the kernel size k, while deformable convolution has constant resolution and requires 2k 2 offset parameters for a k ?? k filter.

To infer the local covariances, we follow the deformable approach BID3 , and learn a convolutional regressor for each dynamic filtering step.

The regressor, which is simply a convolution layer, first infers the covariances which then determine the dynamic filtering that follows.

The lowdimensional structure of our dynamic parameters makes this regression more efficient than free-form deformation, as it only has three outputs for each full covariance, or even just one for each spherical covariance.

Since the covariance is differentiable, the regression is learned end-to-end from the task loss without further supervision.

We experiment on semantic segmentation of CityScapes BID2 , a challenging dataset of varied urban scenes captured by a car-mounted camera.

We score results by the common intersection-over-union metric (IU).

We choose the fully convolutional DRN-A BID9 as our base architecture.

We choose deformable convolution BID3 as strong baseline for local, dynamic inference without structure.

We train all methods with the same optimization settings for fair comparison.

Note that the backbone is an aggressively-tuned architecture which required significant model search and engineering effort.

Our composition is still able to deliver improvement through learning without further engineering.

We compare our static composition and our Gaussian deformation with free-form deformation in BID3 2k 2 73.6 TAB0 : Dynamic Gaussian deformation reduces parameters, improves computational efficiency, and rivals the accuracy of free-form deformation.

Even restricting the deformation to scale by spherical covariance suffices to equal the free-form accuracy.composition by convolution improves on the backbone by 1 point while dynamic Gaussian deformation gives a further +3 points.

Controlling deformable convolution by Gaussian structure improves efficiency while preserving accuracy to within one point.

While free-form deformations are more general in principle, in practice there is a penalty in efficiency.

Recall that the size of our structured parameterization is independent of the free-form filter size.

On the other hand the original, unstructured deformable convolution requires 2k 2 parameters for a k ?? k filter.

Our results show that making scale dynamic through spherical covariance suffices to achieve equal (or near equal) accuracy.

Scale is perhaps the most ubiquitous transformation in the distribution of natural images, so scale modeling might suffice to handle many transformations.

Our lowdimensional parameterization, needing only one scale parameter at the extreme, can be efficiently optimized on limited data.

<|TLDR|>

@highlight

Dynamic receptive fields with spatial Gaussian structure are accurate and efficient.

@highlight

This paper proposes a structured convolution operator to model deformations of local regions of an image, which significantly reduced the number of parameters.