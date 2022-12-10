Convolutional neural networks (CNNs) are inherently equivariant to translation.

Efforts to embed other forms of equivariance have concentrated solely on rotation.

We expand the notion of equivariance in CNNs through the Polar Transformer Network (PTN).

PTN combines ideas from the Spatial Transformer Network (STN) and canonical coordinate representations.

The result is a network invariant to translation and equivariant to both rotation and scale.

PTN is trained end-to-end and composed of three distinct stages: a polar origin predictor, the newly introduced polar transformer module and a classifier.

PTN achieves state-of-the-art on rotated MNIST and the newly introduced SIM2MNIST dataset, an MNIST variation obtained by adding clutter and perturbing digits with translation, rotation and scaling.

The ideas of PTN are extensible to 3D which we demonstrate through the Cylindrical Transformer Network.

Whether at the global pattern or local feature level BID8 , the quest for (in/equi)variant representations is as old as the field of computer vision and pattern recognition itself.

State-of-the-art in "hand-crafted" approaches is typified by SIFT (Lowe, 2004) .

These detector/descriptors identify the intrinsic scale or rotation of a region BID19 BID1 and produce an equivariant descriptor which is normalized for scale and/or rotation invariance.

The burden of these methods is in the computation of the orbit (i.e. a sampling the transformation space) which is necessary to achieve equivariance.

This motivated steerable filtering which guarantees transformed filter responses can be interpolated from a finite number of filter responses.

Steerability was proved for rotations of Gaussian derivatives BID6 and extended to scale and translations in the shiftable pyramid BID31 .

Use of the orbit and SVD to create a filter basis was proposed by BID26 and in parallel, BID29 proved for certain classes of transformations there exists canonical coordinates where deformation of the input presents as translation of the output.

Following this work, BID25 and BID10 ; Teo & BID33 proposed a methodology for computing the bases of equivariant spaces given the Lie generators of a transformation.

and most recently, BID30 proposed the scattering transform which offers representations invariant to translation, scaling, and rotations.

The current consensus is representations should be learned not designed.

Equivariance to translations by convolution and invariance to local deformations by pooling are now textbook BID17 , p.335) but approaches to equivariance of more general deformations are still maturing.

The main veins are: Spatial Transformer Network (STN) BID13 which similarly to SIFT learn a canonical pose and produce an invariant representation through warping, work which constrains the structure of convolutional filters BID36 and work which uses the filter orbit BID3 to enforce an equivariance to a specific transformation group.

In this paper, we propose the Polar Transformer Network (PTN), which combines the ideas of STN and canonical coordinate representations to achieve equivariance to translations, rotations, and dilations.

The three stage network learns to identify the object center then transforms the input into logpolar coordinates.

In this coordinate system, planar convolutions correspond to group-convolutions in rotation and scale.

PTN produces a representation equivariant to rotations and dilations without http://github.com/daniilidis-group//polar-transformer-networks Figure 1 : In the log-polar representation, rotations around the origin become vertical shifts, and dilations around the origin become horizontal shifts.

The distance between the yellow and green lines is proportional to the rotation angle/scale factor.

Top rows: sequence of rotations, and the corresponding polar images.

Bottom rows: sequence of dilations, and the corresponding polar images.the challenging parameter regression of STN.

We enlarge the notion of equivariance in CNNs beyond Harmonic Networks BID36 and Group Convolutions BID3 by capturing both rotations and dilations of arbitrary precision.

Similar to STN; however, PTN accommodates only global deformations.

We present state-of-the-art performance on rotated MNIST and SIM2MNIST, which we introduce.

To summarize our contributions:• We develop a CNN architecture capable of learning an image representation invariant to translation and equivariant to rotation and dilation.• We propose the polar transformer module, which performs a differentiable log-polar transform, amenable to backpropagation training.

The transform origin is a latent variable.• We show how the polar transform origin can be learned effectively as the centroid of a single channel heatmap predicted by a fully convolutional network.

One of the first equivariant feature extraction schemes was proposed by BID25 who suggested the discrete sampling of 2D-rotations of a complex angle modulated filter.

About the same time, the image and optical processing community discovered the Mellin transform as a modification of the Fourier transform BID39 BID0 .

The Fourier-Mellin transform is equivariant to rotation and scale while its modulus is invariant.

During the 80's and 90's invariances of integral transforms were developed through methods based in the Lie generators of the respective transforms starting from one-parameter transforms BID5 and generalizing to Abelian subgroups of the affine group BID29 .Closely related to the (in/equi)variance work is work in steerability, the interpolation of responses to any group action using the response of a finite filter basis.

An exact steerability framework began in BID6 , where rotational steerability for Gaussian derivatives was explicitly computed.

It was extended to the shiftable pyramid BID31 , which handle rotation and scale.

A method of approximating steerability by learning a lower dimensional representation of the image deformation from the transformation orbit and the SVD was proposed by BID26 .for the largest Abelian subgroup and incrementally steering for the remaining subgroups.

Cohen & Welling (2016a); Jacobsen et al. (2017) recently combined steerability and learnable filters.

The most recent "hand-crafted" approach to equivariant representations is the scattering transform BID30 which composes rotated and dilated wavelets.

Similar to SIFT (Lowe, 2004) this approach relies on the equivariance of anchor points (e.g. the maxima of filtered responses in (translation) space).

Translation invariance is obtained through the modulus operation which is computed after each convolution.

The final scattering coefficient is invariant to translations and equivariant to local rotations and scalings.

BID15 achieve transformation invariance by pooling feature maps computed over the input orbit, which scales poorly as it requires forward and backward passes for each orbit element.

Within the context of CNNs, methods of enforcing equivariance fall to two main veins.

In the first, equivariance is obtained by constraining filter structure similarly to Lie generator based approaches BID29 BID10 .

Harmonic Networks BID36 use filters derived from the complex harmonics achieving both rotational and translational equivariance.

The second requires the use of a filter orbit which is itself equivariant to obtain group equivariance.

BID3 convolve with the orbit of a learned filter and prove the equivariance of group-convolutions and preservation of rotational equivariance in the presence of rectification and pooling.

BID4 process elements of the image orbit individually and use the set of outputs for classification.

BID7 produce maps of finite-multiparameter groups, BID38 and BID21 use a rotational filter orbit to produce oriented feature maps and rotationally invariant features, and BID18 propose a transformation layer which acts as a group-convolution by first permuting then transforming by a linear filter.

Our approach, PTN, is akin to the second vein.

We achieve global rotational equivariance and expand the notion of CNN equivariance to include scaling.

PTN employs log-polar coordinates (canonical coordinates in BID29 ) to achieve rotation-dilation group-convolution through translational convolution subject to the assumption of an image center estimated similarly to the STN.

Most related to our method is BID11 , which achieves equivariance by warping the inputs to a fixed grid, with no learned parameters.

When learning features from 3D objects, invariance to transformations is usually achieved through augmenting the training data with transformed versions of the inputs BID37 , or pooling over transformed versions during training and/or test BID22 BID27 .

BID28 show that a multi-task approach, i.e. prediction of both the orientation and class, improves classification performance.

In our extension to 3D object classification, we explicitly learn representations equivariant to rotations around a family of parallel axes by transforming the input to cylindrical coordinates about a predicted axis.

This section is divided into two parts, the first offers a review of equivariance and groupconvolutions.

The second offers an explicit example of the equivariance of group-convolutions through the 2D similarity transformations group, SIM(2), comprised of translations, dilations and rotations.

Reparameterization of SIM(2) to canonical coordinates allows for the application of the SIM(2) group-convolution using translational convolution.

Equivariant representations are highly sought after as they encode both class and deformation information in a predictable way.

Let G be a transformation group and L g I be the group action applied to an image I. A mapping Φ : E → F is said to be equivariant to the group action DISPLAYFORM0 where L g and L g correspond to application of g to E and F respectively and satisfy DISPLAYFORM1 Invariance is the special case of equivariance where L g is the identity.

In the context of image classification and CNNs, g ∈ G can be thought of as an image deformation and Φ a mapping from the image to a feature map.

The inherent translational equivariance of CNNs is independent of the convolutional kernel and evident in the corresponding translation of the output in response to translation of the input.

Equivariance to other types of deformations can be achieved through application of the group-convolution, a generalization of translational convolution.

Letting f (g) and φ(g) be real valued functions on G DISPLAYFORM2 A slight modification to the definition is necessary in the first CNN layer since the group is acting on the image.

The group-convolution reduces to translational convolution when G is translation in R n with addition as the group operator, DISPLAYFORM3 Group-convolution requires integrability over a group and identification of the appropriate measure dg.

It can be proved that given the measure dg, group-convolution is always group equivariant: DISPLAYFORM4 This is depicted in response of an equivariant representation to input deformation ( Figure 2 (left)).

A similarity transformation, ρ ∈ SIM(2), acts on a point in x ∈ R 2 by DISPLAYFORM0 where SO(2) is the rotation group.

To take advantage of the standard planar convolution in classical CNNs we decompose a ρ ∈ SIM(2) into a translation, t in R 2 and a dilated-rotation r in SO(2)×R + .Equivariance to SIM FORMULA2 is achieved by learning the center of the dilated rotation, shifting the original image accordingly then transforming the image to canonical coordinates.

In this reparameterization the standard translational convolution is equivalent to the dilated-rotation group-convolution.

The origin predictor is an application of STN to global translation prediction BID13 , the centroid of the output is taken as the origin of the input.

Transformation of the image L t I = I(t − t 0 ) (canonization in Soatto FORMULA0 ) reduces the SIM(2) deformation to a dilated-rotation if t o is the true translation.

After centering, we perform SO(2) × R + convolutions on the new image I o = I(x − t o ): DISPLAYFORM1 and the feature maps f in subsequent layers DISPLAYFORM2 where r, s ∈ SO(2) × R + .

We compute this convolution through use of canonical coordinates for Abelian Lie-groups BID29 .

The centered image I o (x, y) 1 is transformed to logpolar coordinates, I(e ξ cos(θ), e ξ sin(θ)) hereafter written λ(ξ, θ) with (ξ, θ) ∈ SO(2) × R + for Figure 2 : Left: Group-convolutions in SO(2).

The images in the left most column differ by 90• rotation, the filters are shown in the top row.

Application of the rotational group-convolution with an arbitrary filter results is shown to produce an equivariant representation.

The inner-product each of filter orbit (rotated from 0 − 360 • ) and the image is plotted in blue for the top image and red for the bottom image.

Observe how the filter response is shifted by 90• .

Right: Group-convolutions in SO(2) × R + .

Images in the left most column differ by a rotation of π/4 and scaling of 1.2.

Careful consideration of the resulting heatmaps (shown in canonical coordinates) reveals a shift corresponding to the deformation of the input image.notational convenience.

The shift of the dilated-rotation equivariant representation in response to input deformation is shown in Figure 2 (right) using canonical coordinates.

In canonical coordinates s −1 r = ξ r − ξ, θ r − θ and the SO(2) × R + group-convolution 2 can be expressed and efficiently implemented as a planar convolution DISPLAYFORM3 To summarize, we (1) construct a network of translational convolutions, (2) take the centroid of the last layer, (3) shift the original image to accordingly, (4) convert to log-polar coordinates, and (5) apply a second network 3 of translational convolutions.

The result is a feature map equivariant to dilated-rotations around the origin.

PTN is comprised of two main components connected by the polar transformer module.

The first part is the polar origin predictor and the second is the classifier (a conventional fully convolutional network).

The building block of the network is a 3 × 3 × K convolutional layer followed by batch normalization, an ReLU and occasional subsampling through strided convolution.

We will refer to this building block simply as block.

Figure 3 shows the architecture.

The polar origin predictor operates on the original image and comprises a sequence of blocks followed by a 1 × 1 convolution.

The output is a single channel feature map, the centroid of which is taken as the origin of the polar transform.

There are some difficulties in training a neural network to predict coordinates in images.

Some approaches BID35 attempt to use fully connected layers to directly regress the coordinates with limited success.

A better option is to predict heatmaps BID34 BID24 , and take their argmax.

However, this can be problematic since backpropogation gradients are zero in all but one point, which impedes learning.1 we abuse the notation here and momentarily we use x as the x-coordinate instead of x ∈ R 2 .

2 abuse of the term, SO(2) × R + is not a group because the dilation ξ is not compact.

3 the network employs rectifier and pooling which have been shown to preserve equivariance BID3 Figure 3 : Network architecture.

The input image passes through a fully convolutional network, the polar origin predictor, which outputs a heatmap.

The centroid of the heatmap (two coordinates), together with the input image, goes into the polar transformer module, which performs a polar transform with origin at the input coordinates.

The obtained polar representation is invariant with respect to the original object location; and rotations and dilations are now shifts, which are handled equivariantly by a conventional classifier CNN.The usual approach to heatmap prediction is evaluation of a loss against some ground truth.

In this approach the argmax gradient problem is circumvented by supervision.

In PTN the the gradient of the output coordinates must be taken with respect to the heatmap since the polar origin is unknown and must be learned.

Use of argmax is avoided by using the centroid of the heatmap as the polar origin.

The gradient of the centroid with respect to the heatmap is constant and nonzero for all points, making learning possible.

The polar transformer module takes the origin prediction and image as inputs and outputs the logpolar representation of the input.

The module uses the same differentiable image sampling technique as STN BID13 , which allows output coordinates V i to be expressed in terms of the input U and the source sample point coordinates (x s i , y s i ).

The log-polar transform in terms of the source sample points and target regular grid (x t i , y t i ) is: DISPLAYFORM0 DISPLAYFORM1 where (x 0 , y 0 ) is the origin, W, H are the output width and height, and r is the maximum distance from the origin, set to 0.5 √ H 2 + W 2 in our experiments.

To maintain feature map resolution, most CNN implementations use zero-padding.

This is not ideal for the polar representation, as it is periodic about the angular axis.

A rotation of the input result in a vertical shift of the output, wrapping at the boundary; hence, identification of the top and bottom most rows is most appropriate.

This is achieved with wrap-around padding on the vertical dimension.

The top most row of the feature map is padded using the bottom rows and vice versa.

Zero-padding is used in the horizontal dimension.

TAB5 shows a performance evaluation.

To improve robustness of our method, we augment the polar origin during training time by adding a random shift to the regressed polar origin coordinates.

Note that this comes for little computational cost compared to conventional augmentation methods such as rotating the input image.

TAB5 quantifies the performance gains of this kind of augmentation.

We briefly define the architectures in this section, see A for details.

CCNN is a conventional fully convolutional network; PCNN is the same, but applied to polar images with central origin.

STN is our implementation of the spatial transformer networks BID13 .

PTN is our polar transformer networks, and PTN-CNN is a combination of PTN and CCNN.

The suffixes S and B indicate small and big networks, according to the number of parameters.

The suffixes + and ++ indicate training and training+test rotation augmentation.

We perform rotation augmentation for polar-based methods.

In theory, the effect of input rotation is just a shift in the corresponding polar image, which should not affect the classifier CNN.

In practice, interpolation and angle discretization effects result in slightly different polar images for rotated inputs, so even the polar-based methods benefit from this kind of augmentation.

TAB1 shows the results.

We divide the analysis in two parts; on the left, we show approaches with smaller networks and no rotation augmentation, on the right there are no restrictions.

Between the restricted approaches, the Harmonic Network BID36 outperforms the PTN by a small margin, but with almost 4x more training time, because the convolutions on complex variables are more costly.

Also worth mentioning is the poor performance of the STN with no augmentation, which shows that learning the transformation parameters is much harder than learning the polar origin coordinates.

Between the unrestricted approaches, most variants of PTN-B outperform the current state of the art, with significant improvements when combined with CCNN and/or test time augmentation.

Finally, we note that the PCNN achieves a relatively high accuracy in this dataset because the digits are mostly centered, so using the polar transform origin as the image center is reasonable.

Our method, however, outperforms it by a high margin, showing that even in this case, it is possible to find an origin away from the image center that results in a more distinctive representation.

FORMULA0 6 Test time performance is 8x slower when using test time augmentation

We also perform experiments in other MNIST variants.

MNIST R, RTS are replicated from BID13 .

We introduce SIM2MNIST, with a more challenging set of transformations from SIM(2).

See B for more details about the datasets.

TAB2 shows the results.

We can see that the PTN performance mostly matches the STN on both MNIST R and RTS.

The deformations on these datasets are mild and data is plenty, so the performance may be saturated.

On SIM2MNIST, however, the deformations are more challenging and the training set 5x smaller.

The PCNN performance is significantly lower, which reiterates the importance of predicting the best polar origin.

The HNet outperforms the other methods (except the PTN), thanks to its translation and rotation equivariance properties.

Our method is more efficient both in number of parameters and training time, and is also equivariant to dilations, achieving the best performance by a large margin.

BID13 0 DISPLAYFORM0 .28 (0.05) 44k 31.42 TI-Pooling BID15 0.8 DISPLAYFORM1 No augmentation is used with SIM2MNIST, despite the + suffixes 2 Our modified version, with two extra layers with subsampling to account for larger input

We visualize network activations to confirm our claims about invariance to translation and equivariance to rotations and dilations.

Figure 4 (left) shows some of the predicted polar origins and the results of the polar transform.

We can see that the network learns to reject clutter and to find a suitable origin for the polar transform, and that the representation after the polar transformer module does present the properties claimed.

We proceed to visualize if the properties are preserved in deeper layers.

FIG1 (right) shows the activations of selected channels from the last convolutional layer, for different rotations, dilations, and translations of the input.

The reader can verify that the equivariance to rotations and dilations, and the invariance to translations are indeed preserved during the sequence of convolutional layers.

We extend our model to perform 3D object classification from voxel occupancy grids.

We assume that the inputs are transformed by random rotations around an axis from a family of parallel axes.

Then, a rotation around that axis corresponds to a translation in cylindrical coordinates.

In order to achieve equivariance to rotations, we predict an axis and use it as the origin to transform to cylindrical coordinates.

If the axis is parallel to one of the input grid axes, the cylindrical transform amounts to channel-wise polar transforms, where the origin is the same for all channels and each channel is a 2D slice of the 3D voxel grid.

In this setting, we can just apply the polar transformer layer to each slice.

We use a technique similar to the anisotropic probing of BID27 to predict the axis.

Let z denote the input grid axis parallel to the rotation axis.

We treat the dimension indexed by z as channels, and run regular 2D convolutional layers, reducing the number of channels on each layer, eventually collapsing to a single 2D heatmap.

The heatmap centroid gives one point of the axis, and the direction is parallel to z. In other words, the centroid is the origin of all channel-wise polar transforms.

We then proceed with a regular 3D CNN classifier, acting on the cylindrical representation.

The 3D convolutions are equivariant to translations; since they act on cylindrical coordinates, the learned representation is equivariant to input rotations around axes parallel to z.

We run experiments on ModelNet40 BID37 , which contains objects rotated around the gravity direction (z).

FIG2 shows examples of input voxel grids and their cylindrical coordinates representation, while table 3 shows the classification performance.

To the best of our knowledge, our method outperforms all published voxel-based methods, even with no test time augmentation.

However, the multi-view based methods generally outperform the voxel-based.

BID27 .Note that we could also achieve equivariance to scale by using log-cylindrical or log-spherical coordinates, but none of these change of coordinates would result in equivariance to arbitrary 3D rotations.

Cylindrical Transformer (Ours) 86.5 89.9 3D ShapeNets BID37 77.3 -VoxNet BID22 83 -MO-SubvolumeSup BID27 86.0 89.2 MO-Aniprobing BID27 85.6 89.9

We have proposed a novel network whose output is invariant to translations and equivariant to the group of dilations/rotations.

We have combined the idea of learning the translation (similar to the spatial transformer) but providing equivariance for the scaling and rotation, avoiding, thus, fully connected layers required for the pose regression in the spatial transformer.

Equivariance with respect to dilated rotations is achieved by convolution in this group.

Such a convolution would require the production of multiple group copies, however, we avoid this by transforming into canonical coordinates.

We improve the state of the art performance on rotated MNIST by a large margin, and outperform all other tested methods on a new dataset we call SIM2MNIST.

We expect our approach to be applicable to other problems, where the presence of different orientations and scales hinder the performance of conventional CNNs.

We implement the following architectures for comparison,• Conventional CNN (CCNN), a fully convolutional network, composed of a sequence of convolutional layers and some rounds of subsampling .•

Polar CNN (PCNN), same architecture as CCNN, operating on polar images.

The logpolar transform is pre-computed at the image center before training, as in BID11 .

The fundamental difference between our method and this is that we learn the polar origin implicitly, instead of fixing it.• Spatial Transformer Network (STN), our implementation of BID13 , replacing the localization network by four blocks of 20 filters and stride 2, followed by a 20 unit fully connected layer, which we found to perform better.

The transformation regressed is in SIM(2), and a CCNN comes after the transform.• Polar Transformer Network (PTN), our proposed method.

The polar origin predictor comprises three blocks of 20 filters each, with stride 2 on the first block (or the first two blocks, when input is 96 × 96).

The classification network is the CCNN.• PTN-CNN, we classify based on the sum of the per class scores of instances of PTN and CCNN trained independently.

The following suffixes qualify the architectures described above:• S, "small" network, with seven blocks of 20 filters and one round of subsampling (equivalent to the Z2CNN in Cohen & Welling (2016b)).• B, "big" network, with 8 blocks with the following number of filters: 16, 16, 32, 32, 32, 64, 64, 64 .

Subsampling by strided convolution is used whenever the number of filters increase.

We add up to two 2 extra blocks of 16 filters with stride 2 at the beginning to handle larger input resolutions (one for 42 × 42 and two for 96 × 96).• +, training time rotation augmentation by continuous angles.• ++, training and test time rotation augmentation.

We input 8 rotated versions the the query image and classify using the sum of the per class scores.

The axis prediction part of the cylindrical transformer network is composed of four 2D blocks, with 5 × 5 kernels and 32, 16, 8, and 4 channels, no subsampling.

The classifier is composed of eight 3D convolutional blocks, with 3 × 3 × 3 kernels, the following number of filters: 32, 32, 32, 64, 64, 64, 128, 128 , and subsampling whenever the number of filters increase.

Total number of params is approximately 1M.

• Rotated MNIST The rotated MNIST dataset BID16 is composed of 28 × 28, 360• rotated images of handwritten digits.

The training, validation and test sets are of sizes 10k, 2k, and 50k, respectively.• MNIST R, we replicate it from BID13 .

It has 60k training and 10k testing samples, where the digits of the original MNIST are rotated between [−90 DISPLAYFORM0 It is also know as half-rotated MNIST BID15 .•

MNIST RTS, we replicate it from BID13 .

It has 60k training and 10k testing samples, where the digits of the original MNIST are rotated between [−45 DISPLAYFORM1 • ], scaled between 0.7 and 1.2, and shifted within a 42 × 42 black canvas.• SIM2MNIST, we introduce a more challenging dataset, based on MNIST, perturbed by random transformations from SIM(2).

The images are 96 × 96, with 360 • rotations; the scale factors range from 1 to 2.4, and the digits can appear anywhere in the image.

The training, validation and test set have size 10k, 5k, and 50k, respectively.

In order to demonstrate the efficacy of PTN on real-world RGB images, we run experiments on the Street View House Numbers (SVHN) dataset BID23 , and a rotated version that we introduce (ROTSVHN) .

The dataset contains cropped images of single digits, as well as the slightly larger images from where the digits are cropped.

Using the latter, we can extract the rotated digits without introducing artifacts.

FIG3 shows some examples from the ROTSVHN.We use a 32 layer Residual Network BID9 as a baseline (ResNet32).

The PTN-ResNet32 has 8 residual convolutional layers as the origin predictor, followed by a ResNet32.In contrast with handwritten digits, the 6s and 9s in house numbers are usually indistinguishable.

To remove this effect from our analysis, we also run experiments removing those classes from the datasets (which is denoted by appending a minus to the dataset name).

TAB4 shows the results.

The reader will note that rotations cause a significant performance loss on the conventional ResNet; the error increases from 2.09% to 5.39%, even when removing 6s and 9s from the dataset.

With PTN, on the other hand, the error goes from 2.85% to 3.96%, which shows our method is more robust to the perturbations, although the performance on the unperturbed datasets is slightly worse.

We expect the PTN to be even more advantageous when large scale variations are also present.

We quantify the performance boost obtained with wrap around padding, polar origin augmentation, and training time rotation augmentation.

Results are based on the PTN-B variant trained on Rotated MNIST.

We remove one operation at a time and verify that the performance consistently drops, which indicates that all operations are indeed helpful.

TAB5 shows the results.

@highlight

We learn feature maps invariant to translation, and equivariant to rotation and scale.