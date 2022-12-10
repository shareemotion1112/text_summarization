In contrast to the monolithic deep architectures used in deep learning today for computer vision, the visual cortex processes retinal images via two functionally distinct but interconnected networks: the ventral pathway for processing object-related information and the dorsal pathway for processing motion and transformations.

Inspired by this cortical division of labor and properties of the magno- and parvocellular systems, we explore an unsupervised approach to feature learning that jointly learns object features and their transformations from natural videos.

We propose a new convolutional bilinear sparse coding model that (1) allows independent feature transformations and (2) is capable of processing large images.

Our learning procedure leverages smooth motion in natural videos.

Our results show that our model can learn groups of features and their transformations directly from natural videos in a completely unsupervised manner.

The learned "dynamic filters" exhibit certain equivariance properties, resemble cortical spatiotemporal filters, and capture the statistics of transitions between video frames.

Our model can be viewed as one of the first approaches to demonstrate unsupervised learning of primary "capsules" (proposed by Hinton and colleagues for supervised learning) and has strong connections to the Lie group approach to visual perception.

Current unsupervised models produce representations that either lack interpretability or hierarchical 27 depth.

Variational autoencoders and generative adversarial networks (GANs) typically produce non-28 interpretable features that do not match the object/parts hierarchy inherent in natural visual scenes.

In bilinear sparse coding [3, 1] , an image patch is modeled as a combination of features B ij with 54 two sets of coefficients r i (object coefficients) and x j (transformation coefficients) that interact 55 multiplicatively:

Let j x j B ij = B i (x) where x represents the transformation vector consisting of x j 's.

Then of objects, sparsity is enforced on either r or both r and x via some appropriate sparsity penalty.

Typically bilinear sparse coding models are trained using pairs of video frames I t+1 and I t , with r 65 fixed and x inferred separately to account for the difference between frames:

There is a strong connection to the Lie group approach to vision [2] where two consecutive frames 67 are modelled as I t+1 = T (∆x)I t where T is a transformation operator.

The first-order Taylor   68 series approximation of the Lie model [7, 6] is given by: I t+1 = I t + j ∆x t,j ∇x j I t which 69 means that ∆I = j ∆x t,j ∇x j I t .

Suppose I t i r i U i where U i ∈ R d×1 form an un-70 derlying feature set.

Replacing ∇x j with the transformation matrix G j ∈ R d×d , we obtain:

72 that B ij = G j U i .

We build on this model by allowing features to have independent pose parameters x ij so that features 74 can transform independently from frame to frame.

We also go beyond image patches to modeling 75 large images by using transposed convolutions ( * T ), resulting in a new bilinear model for images:

To distinguish our model from past models, we refer to traditional bilinear sparse coding as BSC 77 and our independent bilinear sparse coding model as IBSC.

The reconstruction-based loss function for consecutive frames of a video is given by:

(4) with r, x ≥ 0.

The first term is the mean-squared reconstruction error.

The other terms include a 81 sparsity penalty on r and weight decay for G and U .

To stabilize learning we project each B ij = 82 G j U i to unit 2 norm (P 2,1.0 ).

Inference for BSC is typically performed by initializing x to some canonical vector and then alter-84 natively optimizing r and x [3].

One of the issues with this approach is that the canonical vector 85 might be a poor approximation to the true underlying pose parameters, especially in the case of 86 independent features as in our model.

We convolve each feature B ij with the image to produce a 87 feature map α ijt = B ij * I t .

We then project onto some appropriately chosen norm ball to compute

2 Inference proceeds by alternatively optimizing r and x until convergence.

To 89 optimize r, we use iterative thresholding, while x is optimized by projected gradient descent.

Both 90 sets of coefficients are forced to be non-negative, using a rectifier for r and projecting on the positive 91 part of the norm ball for x.

3 Experiments

For our experiments, we used 1920 × 1080 resolution YouTube videos converted to gray scale

and scaled down to 236 × 176 pixels per frame.

The frames were normalized using subtractive 95 normalization 3 .

We extracted sequences of 5 consecutive frames, with r assumed to be constant 96 for each sequence during training.

We excluded sequences in the largest 5% of Euclidean norm 97 difference between frames to exclude sudden camera changes or changes between scenes.

We used

<|TLDR|>

@highlight

We extend bilinear sparse coding and leverage video sequences to learn dynamic filters.