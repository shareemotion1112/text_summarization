Encoding the input scale information explicitly into the representation learned by a convolutional neural network (CNN) is beneficial for many vision tasks especially when dealing with multiscale input signals.

We study, in this paper, a scale-equivariant CNN architecture with joint convolutions across the space and the scaling group, which is shown to be both sufficient and necessary to achieve scale-equivariant representations.

To reduce the model complexity and computational burden, we decompose the convolutional filters under two pre-fixed separable bases and truncate the expansion to low-frequency components.

A further benefit of the truncated filter expansion is the improved deformation robustness of the equivariant representation.

Numerical experiments demonstrate that the proposed scale-equivariant neural network with decomposed convolutional filters (ScDCFNet) achieves significantly improved performance in multiscale image classification and better interpretability than regular CNNs at a reduced model size.

Convolutional neural networks (CNNs) have achieved great success in machine learning problems such as image classification (Krizhevsky et al., 2012) , object detection (Ren et al., 2015) , and semantic segmentation (Long et al., 2015; Ronneberger et al., 2015) .

Compared to fully-connected networks, CNNs through spatial weight sharing have the benefit of being translation-equivariant, i.e., translating the input leads to a translated version of the output.

This property is crucial for many vision tasks such as image recognition and segmentation.

However, regular CNNs are not equivariant to other important group transformations such as rescaling and rotation, and it is beneficial in some applications to also encode such group information explicitly into the network representation.

Several network architectures have been designed to achieve (2D) roto-translation-equivariance (SE(2)-equivariance) (Cheng et al., 2019; Marcos et al., 2017; Weiler et al., 2018b; Worrall et al., 2017; Zhou et al., 2017) , i.e., roughly speaking, if the input is spatially rotated and translated, the output is transformed accordingly.

The feature maps of such networks typically include an extra index for the rotation group SO(2).

Building on the idea of group convolutions proposed by Cohen & Welling (2016) for discrete symmetry groups, Cheng et al. (2019) and Weiler et al. (2018b) constructed SE(2)-equivariant CNNs by conducting group convolutions jointly across the space and SO(2) using steerable filters (Freeman & Adelson, 1991) .

Scaling-translation-equivariant (ST -equivariant) CNNs, on the other hand, have typically been studied in a less general setting in the existing literature (Kanazawa et al., 2014; Marcos et al., 2018; Xu et al., 2014; Ghosh & Gupta, 2019) .

In particular, to the best of our knowledge, a joint convolution across the space and the scaling group S has yet been proposed to achieve equivariance in the most general form.

This is possibly because of two difficulties one encounters when dealing with the scaling group: First, unlike SO(2), it is an acyclic and unbounded group; second, an extra index in S incurs a significant increase in model parameters and computational burden.

Moreover, since the scaling transformation is rarely perfect in practice (due to changing view angle or numerical discretization), one needs to quantify and promote the deformation robustness of the equivariant representation (i.e., is the model still "approximately" equivariant if the scaling transformation is "contaminated" by a nuisance input deformation), which, to the best of our knowledge, has yet been studied in prior works.

The purpose of this paper is to address the aforementioned theoretical and practical issues in the construction of ST -equivariant CNN models.

Specifically, our contribution is three-fold:

1.

We propose a general ST -equivariant CNN architecture with a joint convolution over R 2 and S, which is proved in Section 4 to be both sufficient and necessary to achieve ST -equivariance.

2.

A truncated decomposition of the convolutional filters under a pre-fixed separable basis on the two geometric domains (R 2 and S) is used to reduce the model size and computational cost.

3.

We prove the representation stability of the proposed architecture up to equivariant scaling action of the input signal.

Our contribution to the family of group-equivariant CNNs is non-trivial; in particular, the scaling group unlike SO(2) is acyclic and non-compact.

This poses challenges both in theory and in practice, so that many previous works on group-equivariant CNNs cannot be directly extended.

We introduce new algorithm design and mathematical techniques to obtain the first general ST -equivariant CNN in literature with both computational efficiency and proved representation stability.

Mixed-scale and ST -equivariant CNNs.

Incorporating multiscale information into a CNN representation has been studied in many existing works.

The Inception net (Szegedy et al., 2015) and its generalizations (Szegedy et al., 2017; 2016; Li et al., 2019) stack filters of different sizes in a single layer to address the multiscale salient features.

Dilated convolutions (Pelt & Sethian, 2018; Wang et al., 2018; Yu & Koltun, 2016; Yu et al., 2017) , pyramid architectures (Ke et al., 2017; Lin et al., 2017) , and multiscale dense networks (Huang et al., 2017) have also been proposed to take into account the multiscale feature information.

Although the effectiveness of such models have been empirically demonstrated in various vision tasks, there is still a lack of interpretability of their ability to encode the input scale information.

Group-equivariant CNNs, on the other hand, explicitly encode the group information into the network representation.

Cohen & Welling (2016) proposed CNNs with group convolutions that are equivariant to several finite discrete symmetry groups.

This idea is later generalized in Cohen et al. (2018) and applied mainly to the rotation groups SO(2) and SO(3) (Cheng et al., 2019; Weiler et al., 2018a; b) .

Although ST -equivariant CNNs have also been proposed in the literature (Kanazawa et al., 2014; Marcos et al., 2018; Xu et al., 2014; Ghosh & Gupta, 2019) , they are typically studied in a less general setting.

In particular, none of these previous works proposed to conduct joint convolutions over R 2 ?? S as a necessary and sufficient condition to impose equivariance, for which reason they are thus variants of a special case of our proposed architecture where the convolutional filters in S are Dirac delta functions (c.f.

Remark 1.)

The scale-space semi-group correlation proposed in the concurrent work (Worrall & Welling, 2019) bears the most resemblance to our proposed model, however, their approach is only limited to discrete semigroups, whereas our model does not have such restriction.

Representation stability to input deformations.

Input deformations typically induce noticeable variabilities within object classes, some of which are uninformative for the vision tasks.

Models that are stable to input deformations are thus favorable in many applications.

The scattering transform (Bruna & Mallat, 2013; Mallat, 2010; 2012) computes translation-invariant representations that are Lipschitz continuous to deformations by cascading predefined wavelet transforms and modulus poolings.

A joint convolution over R 2 ?? SO(2) is later adopted in Sifre & Mallat (2013) to build roto-translation scattering with stable rotation/translation-invariant representations.

These models, however, use pre-fixed wavelet transforms in the networks, and are thus nonadaptive to the data.

DCFNet (Qiu et al., 2018 ) combines a pre-fixed filter basis and learnable expansion coefficients in a CNN architecture, achieving both data adaptivity and representation stability inherited from the filter regularity.

This idea is later extended by Cheng et al. (2019) to produce SE(2)-equivariant representations that are Lipschitz continuous in L 2 norm to input deformations modulo a global rotation, i.e., the model stays approximately equivariant even if the input rotation is imperfect.

To the best of our knowledge, a theoretical analysis of the deformation robustness of a ST -equivariant CNN has yet been studied, and a direct generalization of the result in Cheng et al. (2019) is futile because the feature maps of a ST -equivariant CNN is typically not in L 2 (c.f.

Remark 2.)

3 ST -EQUIVARIANT CNN AND FILTER DECOMPOSITION Group-equivariance is the property of a mapping f : X ??? Y to commute with the group actions on the domain X and codomain Y .

More specifically, let G be a group, and D g , T g , respectively, be group actions on X and Y .

A function f : X ??? Y is said to be G-equivariant if

G-invariance is thus a special case of G-equivariance where T g = Id Y .

For learning tasks where the feature y ??? Y is known a priori to change equivariantly to a group action g ??? G on the input x ??? X, e.g. image segmentation should be equivariant to translation, it would be beneficial to reduce the hypothesis space to include only G-equivaraint models.

In this paper, we consider mainly the scaling-translation group ST ??? = S ?? R 2 ??? = R ?? R 2 .

Given g = (??, v) ??? ST and an input image x (0) (u, ??) (u ??? R 2 is the spatial position, and ?? is the unstructured channel index, e.g. RGB channels of a color image), the scaling-translation group action

Constructing ST -equivariant CNNs thus amounts to finding an architecture A such that each trained network f ??? A commutes with the group action D ??,v on the input and a similarly defined group action T ??,v (to be explained in Section 3.1) on the output.

Inspired by Cheng et al. (2019) and Weiler et al. (2018b) , we consider ST -equivariant CNNs with an extra index ?? ??? S for the the scaling group S ??? = R: for each l ??? 1, the l-th layer output is denoted as x (l) (u, ??, ??), where u ??? R 2 is the spatial position, ?? ??? S is the scale index, and ?? ??? [M l ] := {1, . . .

, M l } corresponds to the unstructured channels.

We use the continuous model for formal derivation, i.e., the images and feature maps have continuous spatial and scale indices.

In practice, the images are discretized on a Cartesian grid, and the scales are computed only on a discretized finite interval.

Similar to Cheng et al. (2019) , the group action T ??,v on the l-th layer output is defined as a scaling-translation in space as well as a shift in the scale channel:

A feedforward neural network is said to be scaling-translation-equivariant, i.e., equivariant to ST , if

where we slightly abuse the notation x (l) [x (0) ] to denote the l-th layer output given the input x (0) .

The following Theorem shows that ST -equivariance is achieved if and only if joint convolutions are conducted over S ?? R 2 as in (5) and (6).

Theorem 1.

A feedforward neural network with an extra index ?? ??? S for layerwise output is ST -equivariant if and only if the layerwise operations are defined as (5) and (6):

where ?? : R ??? R is a pointwise nonlinear function.

We defer the proof of Theorem 1, as well as those of other theorems, to the appendix.

We note that the joint-convolution in Theorem 1 is a generalization of the group convolution proposed by Cohen & Welling (2016) to a non-compact group ST in the continuous setting.

?? ,?? (u)??(??), the joint convolution (6) over R 2 ?? S reduces to only a (multiscale) spatial convolution

i.e., the feature maps at different scales do not transfer information among each other (see Figure 1a) .

The previous works (Kanazawa et al., 2014; Marcos et al., 2018; Xu et al., 2014; Ghosh & Gupta, 2019) on ST -equivariant CNNs are all based on this special case of Theorem 1.

Although the joint convolutions (6) on R 2 ?? S provide the most general way of imposing STequivariance, they unfortunately also incur a significant increase in the model size and computational burden.

Following the idea of Cheng et al. (2019) and Qiu et al. (2018) , we address this issue by taking a truncated decomposition of the convolutional filters under a pre-fixed separable basis, which will be discussed in detail in the next section.

We consider decomposing the convolutional filters W (l) ?? ,?? (u, ??) under the product of two function bases, {?? k (u)} k and {?? m (??)} m , which are the eigenfunctions of the Dirichlet Laplacian on, respectively, the unit disk D ??? R 2 and [???1, 1], i.e.,

In particular, the spatial basis {?? k } k satisfying (8) is the Fourier-Bessel (FB) basis (Abramowitz & Stegun, 1965) .

In the continuous formulation, the spatial "pooling" operation is equivalent to rescaling the convolutional filters in space.

We thus assume, without loss of generality, that the convolutional filters are compactly supported as follows

?? ,?? (k, m) are the expansion coefficients of the filters.

During training, the basis functions are fixed, and only the expansion coefficients are updated.

In practice, we truncate the expansion to only low-frequency components (i.e., a

, which are kept as the trainable parameters.

Similar idea has also been considered in the prior works (Qiu et al., 2018; Cheng et al., 2019; Jacobsen et al., 2016) .

This directly leads to a reduction of network parameters and computational burden.

More specifically, let us compare the l-th convolutional layer (6) of a ST -equivariant CNN with and without truncated basis decomposition:

Number of trainable parameters: Suppose the filters W

On the other hand, in an ScDCFNet with truncated basis expansion up to K leading coefficients for u and K ?? coefficients for ??, the number of parameters is instead KK ?? M l???1 M l .

Hence a reduction to a factor of KK ?? /L 2 L ?? in trainable parameters is achieved for ScDCFNet via truncated basis decomposition.

In particular, if L = 5, L ?? = 5, K = 8, and K ?? = 3, then the number of parameters is reduced to (8 ?? 3)/(5 2 ?? 5) = 19.2%.

Computational cost: Suppose the size of the input x (l???1) (u, ??, ??) and output x (l) (u, ??, ??) at the l-th layer are, respectively, W ?? W ?? N ?? ?? M l???1 and W ?? W ?? N ?? ?? M l , where W ?? W is the spatial dimension, N ?? is the number of scale channels, and M l???1 (M l ) is the number of the unstructured input (output) channels.

Let the filters W

The following theorem shows that, compared to a regular ST -equivariant CNN, the computational cost in a forward pass of ScDCFNet is reduced again to a factor of KK ?? /L 2 L ?? .

e., the number of the output channels is much larger than the size of the convolutional filters in u and ??, then the computational cost of an ScDCFNet is reduced to a factor of KK ?? /L 2 L ?? when compared to a ST -equivariant CNN without basis decomposition.

Apart from reducing the model size and computational burden, we demonstrate in this section that truncating the filter decomposition has the further benefit of improving the deformation robustness of the equivariant representation, i.e., the equivaraince relation (4) still approximately holds true even if the spatial scaling of the input D ??,v x (0) is contaminated by a local deformation.

The analysis is motivated by the fact that scaling transformations are rarely perfect in practice -they are typically subject to local distortions such as changing view angle or numerical discretization.

To quantify the distance between different feature maps at each layer, we define the norm of x (l) as

Remark 2.

The definition of x (l) is different from that of RotDCFNet (Cheng et al., 2019) , where an L 2 norm is taken for the ?? index as well.

The reason why we adopt the L ??? norm for ?? in (11) is that x (l) is typically not L 2 in ??, since the scaling group S, unlike SO(2), has infinite Haar measure.

We next quantify the representation stability of ScDCFNet under three mild assumptions on the convolutional layers and input deformations.

First,

The pointwise nonlinear activation ?? : R ??? R is non-expansive.

Next, we need a bound on the convolutional filters under certain norms.

where the Fourier-Bessel (FB) norm a FB of a sequence {a(k)} k???0 is a weighted l 2 norm defined as a

, where ?? k is the k-th eigenvalue of the Dirichlet Laplacian on the unit disk defined in (8).

We next assume that each A l is bounded:

The boundedness of A l is facilitated by truncating the basis decomposition to only low-frequency components (small ?? k ), which is one of the key idea of ScDCFNet explained in Section 3.2.

After a proper initialization of the trainable coefficients, (A2) can generally be satisfied.

The assumption (A2) implies several bounds on the convolutional filters at each layer (c.f.

Lemma 2 in the appendix), which, combined with (A1), guarantees that an ScDCFNet is layerwise non-expansive: Proposition 1.

Under the assumption (A1) and (A2), an ScDCFNet satisfies the following.

(a) For any l ??? 1, the mapping of the l-th layer, (5) and (6), is non-expansive, i.e.,

0 be the l-th layer output given a zero bottom-layer input, then x

Finally, we make an assumption on the input deformation modulo a global scale change.

Given a C 2 function ?? : R 2 ??? R 2 , the spatial deformation D ?? on the feature maps x (l) is defined as

where ??(u) = u ??? ?? (u).

We assume a small local deformation on the input:

, where ?? is the operator norm.

The following theorem demonstrates the representation stability of an ScDCFNet to input deformation modulo a global scale change.

] (rescaled l-th layer feature of the original input), and the difference (

It is clear that even after numerical discretization, ST -equivariance still approximately holds for ScDCFNet, i.e.,

Theorem 3.

Let D ?? be a small spatial deformation defined in (14), and let D ??,v , T ??,v be the group actions corresponding to an arbitrary scaling 2 ????? ??? R + centered at v ??? R 2 defined in (2) and (3).

In an ScDCFNet satisfying (A1), (A2), and (A3), we have, for any L,

Theorem 3 gauges how approximately equivariant is ScDCFNet if the input undergoes not only a scale change D ??,v but also a nonlinear spatial deformation D ?? , which is important both in theory and in practice because the scaling of an object is rarely perfect in reality.

In this section, we conduct several numerical experiments for the following three purposes.

1.

To verify that ScDCFNet indeed achieves ST -equivariance (4).

2.

To illustrate that ScDCFNet significantly outperforms regular CNNs at a much reduced model size in multiscale image classification.

3.

To show that a trained ScDCFNet auto-encoder is able to reconstruct rescaled versions of the input by simply applying group actions on the image codes, demonstrating that ScDCFNet indeed explicitly encodes the input scale information into the representation.

The experiments are tested on the Scaled MNIST (SMNIST) and Scaled Fashion-MNIST (SFashion) datasets, which are built by rescaling the original MNIST and Fashion-MNIST (Xiao et al., 2017) images by a factor randomly sampled from a uniform distribution on [0.3, 1].

A zero-padding to a size of 28 ?? 28 is conducted after the rescaling.

If mentioned explicitly, for some experiments, the images are resized to 64 ?? 64 for better visualization.

The implementation details of ScDCFNet are explained in Appendix B.1.

In particular, we discuss how to discretize the integral (6) and truncate the scale channel to a finite interval for practical implementation.

We also explain how to mitigate the boundary "leakage" effect incurred by the truncation.

Moreover, modifications to the spatial pooling and batch-normalization modules of ScDCFNet to maintain ST -equivariance are also explained.

We first verify that ScDCFNet indeed achieves ST -equivariance (4).

Specifically, we compare the feature maps of a two-layer ScDCFNet with randomly generated truncated filter expansion coefficients and those of a regular CNN.

The exact architectures are detailed in Appendix B.2.

Figure 2 displays the first-and second-layer feature maps of an original image x (0) and its rescaled version D ??,v x (0) using the two comparing architectures.

Feature maps at different layers are rescaled to the same spatial dimension for visualization.

The four images enclosed in each of the dashed rectangle correspond to:

] (rescaled l-th layer feature of the original input, where T ??,v is understood as D ??,v for a regular CNN due to the lack of a scale index ??), and the difference (a) Zero-padding for the scale channel.

(b) Replicate-padding for the scale channel.

Figure 3: The numerical error in equivariance (i.e., the boundary "leakage" effect incurred by scale channel truncation) as a function of network depth.

Either (a) zero-padding or (b) replicate-padding is used for the convolution in scale.

The error is unavoidable as depth becomes larger, but it can be mitigated by (1) using joint convolutional filters with a smaller support in scale (i.e., a smaller number of "taps" after discretization), and (2) using a replicate-padding instead of zero-padding.

See Appendix B.1 for detailed explanation.

It is clear that even with numerical discretization, which can be modeled as a form of input deformation, ScDCFNet is still approximately ST -equivariant, i.e.,

, whereas a regular CNN does not have such a property.

We also examin how the numerical error in equivariance (incurred by the the boundary "leakage" effect after the scale channel truncation) evolves as the network gets deeper.

The error in equivariance is measured in a relative L 2 sense at a particular scale ??, i.e.,

It is clear from Figure 3 that the boundary "leakage" effect is unavoidable as the network becomes deeper.

However, the error can be alleviated by either choosing joint convolutional filters with a smaller support in scale (i.e., the filter size, or the number of "taps", after discretization in scale is much smaller compared to the number of scale channels in the feature map), or using a replicatepadding in the scale channel for the joint convolution.

See Appendix B.1 for detailed explanation.

We next demonstrate the improved performance of ScDCFNet in multiscale image classification.

The experiments are conducted on SMNIST and SFashion, and a regular CNN is used as a performance benchmark.

Both networks are comprised of three convolutional layers with the exact architectures (Table 2 ) detailed in Appendix B.3.

Since the scaling group S ??? = R is unbounded, we compute only the feature maps x (l) (u, ??, ??) with the ?? index restricted to the truncated scale interval [???1.6, 0] (2 ???1.6 ??? 0.3), which is discretized uniformly into N ?? = 9 channels (again, see Appendix B.1 for implementation details.)

The performance of the comparing architectures with and without batch-normalization is shown in Table 1 .

It is clear that, by limiting the hypothesis space to STequivaraint models and taking truncated basis decomposition to reduce the model size, ScDCFNet achieves a significant improvement in classification accuracy with a reduced number of trainable parameters.

The advantage of ScDCFNet is more pronounced when the number of training samples is small (N tr = 2000), suggesting that, by hardwiring the input scale information directly into its representation, ScDCFNet is less susceptible to overfitting the limited multiscale training data.

We also observe that even when a regular CNN is trained with data augmentation (random cropping and rescaling), its performance is still inferior to that of an ScDCFNet without manipulation of the training data.

In particular, although the accuracies of the regular CNNs trained on 2000 SMNIST and SFashion images after data augmentation are improved to, respectively, 93.85% and 79.41%, they still underperform the ScDCFNets without data augmentation (93.91% and 79.94%) using only a fraction of trainable parameters.

Moreover, if ScDCFNet is trained with data augmentation, the accuracies can be further improved to 94.30% and 80.62% respectively.

This suggests that ScDCFNet can be combined with data augmentation for optimal performance in multiscale image classification.

Table 1 : Classification accuracy on the SMNIST and SFashion dataset with and without batch-normalization.

The architectures are detailed in Table 2 .

In particular, M stands for the number of the first-layer (unstructured) output channels, which is doubled after each layer, and K/K?? is the number of basis function in u/?? used for filter decomposition.

The networks are tested with different training data size, Ntr = 2000, 5000, and 10000, and the means and standard deviations after three independent trials are reported.

The column "ratio" stands for the ratio between the number of trainable parameters of the current architecture and that of the baseline CNN.

according to the group action (3).

The first two images on the left are the original inputs; Decoder(C) denotes the reconstruction using the (unchanged) image code C; Decoder(D ??,v C) and Decoder(T ??,v C) denote the reconstructions using the "rescaled" image codes D ??,v C and T ??,v C respectively according to (2) and (3).

Unlike the regular CNN auto-encoder, the ScDCFNet auto-encoder manages to generate rescaled versions of the original input, suggesting that it successfully encodes the scale information directly into the representation.

In the last experiment, we illustrate the ability of ScDCFNet to explicitly encode the input scale information into the representation.

To achieve this, we train an ScDCFNet auto-encoder on the SMNIST dataset with images resized to 64 ?? 64 for better visualization.

The encoder stacks two STequivaraint convolutional blocks with 2 ?? 2 average-pooling, and the decoder contains a succession of two transposed convolutional blocks with 2 ?? 2 upsampling.

A regular CNN auto-encoder is also trained for comparison (see Table 3 in Appendix B.4 for the detailed architecture.)

Our goal is to demonstrate that the image code produced by the ScDCFNet auto-encoder contains the scale information of the input, i.e., by applying the group action T ??,v (3) to the code C of a test image before feeding it to the decoder, we can reconstruct rescaled versions of original input.

This property can be visually verified in Figure 4 .

In contrast, a regular CNN auto-encoder fails to do so.

We propose, in this paper, a ST -equivaraint CNN with joint convolutions across the space R 2 and the scaling group S, which we show to be both sufficient and necessary to impose ST -equivariant network representation.

To reduce the computational cost and model complexity incurred by the joint convolutions, the convolutional filters supported on R 2 ?? S are decomposed under a separable basis across the two domains and truncated to only low-frequency components.

Moreover, the truncated filter expansion leads also to improved deformation robustness of the equivaraint representation, i.e., the model is still approximately equivariant even if the scaling transformation is imperfect.

Experimental results suggest that ScDCFNet achieves improved performance in multiscale image classification with greater interpretability and reduced model size compared to regular CNN models.

For future work, we will study the application of ScDCFNet in other more complicated vision tasks, such as object detection/localization and pose estimation, where it is beneficial to directly encode the input scale information into the deep representation.

Moreover, the memory usage of our current implementation of ScDCFNet scales linearly to the number of the truncated basis functions in order to realize the reduced computational burden explained in Theorem 2.

We will explore other efficient implementation of the model, e.g., using filter-bank type of techniques to compute convolutions with multiscale spatial filters, to significantly reduce both the computational cost and memory usage.

Proof of Theorem 1.

We note first that (4) holds true if and only if the following being valid for all l ??? 1,

where

.

We also note that the layer-wise operations of a general feedforward neural network with an extra index ?? ??? S can be written as

and, for l > 1,

To prove the sufficient part: when l = 1, (2), (3), and (5) lead to

and

Hence

When l > 1, we have

and

To prove the necessary part: when l = 1, we have

and

Hence for (17) to hold when l = 1, we need ??? u, ??, ??, u , ?? , v, ??. (26) Keeping u, ??, ??, u , ?? , ?? fixed while changing v in (26), we obtain that W (1) (u , ?? , u, ??, ??) does not depend on the third variable u. Thus

Then, for any given u , ?? , u, ??, ??, setting ?? = ?? in (26) leads to

(28) Hence (18) can be written as (5).

For l > 1, a similar argument leads to

Again, keeping u, ??, ??, u , ?? , ?? , ?? fixed while changing v in (29) leads us to the conclusion that W (l) (u , ?? , ?? , u, ??, ??) does not depend on the fourth variable u. Define

After setting ?? = ?? in (29), for any given u , ?? , ?? , u, ??, ??, we have

This concludes the proof of the Theorem.

Proof of Theorem 2.

In a regular ST -equivariant CNN, the l-th convolutional layer (6) is computed as follows:

The spatial convolutions in (32) take

The summation over ?? , adding the bias, and applying the nonlinear activation in (34) requires an additional W 2 N ?? M l (2 + M l???1 ) flops.

Thus the total number of floating point computations in a forward pass through the l-th layer of a regular ST -equivariant CNN is

On the other hand, in an ScDCFNet with separable basis truncation up to KK ?? leading coefficients, (6) can be computed via the following steps:

The convolutions in ?? (36) require

2 L 2 flops.)

The last step (38) requires an additional

Hence the total number of floating point computation for an ScDCFNet is

In particular, when M l L 2 , L ?? , the dominating terms in (35) and (39) are, respectively,

Thus the computational cost in an ScDCFNet has been reduced to a factor of KK?? L 2 L?? .

Before proving Proposition 1, we need the following two lemmas.

Lemma 1.

Suppose that {?? k } k are the FB bases, and

This is Lemma 3.5 and Proposition 3.6 in Qiu et al. (2018) after rescaling u. Lemma 1 easily leads to the following lemma.

Lemma 2.

Let a

We have

where

We thus have

where

?? ,?? ,

and, for l > 1,

In particular, (A2) implies that

Proof of Proposition 1.

To simplify the notation, we omit (l) in W

?? ,??,m , and b (l) , and let M = M l , M = M l???1 .

The proof of (a) for the case l = 1 is similar to Proposition 3.1(a) of Qiu et al. (2018) after noticing the fact that

and we include it here for completeness.

From the definition of B 1 in (45), we have

?? ,?? ??? B 1 , and sup

Thus, given two arbitrary functions x 1 and x 2 , we have

Therefore, for any ??,

where the last inequality makes use of the fact that B 1 ??? A 1 ??? 1 under (A2) (Lemma 2.)

Therefore

This concludes the proof of (a) for the case l = 1.

To prove the case for any l > 1, we first recall from (46) that

Thus, for two arbitrary functions x 1 and x 2 , we have

where

We claim (to be proved later in Lemma 3) that

Thus, for any ??,

Therefore

To prove (b), we use the method of induction.

When l = 0,

Part (c) is an easy corollary of part (a).

More specifically, for any l > 1,

and ?? L 2 = 1, and x is a function of three variables

with

Then we have

Proof of Lemma 3.

Notice that, for any ??, we have

Proof of Proposition 2.

Just like Proposition 1(a), the proof of Proposition 2(a) for the case l = 1 is similar to Lemma 3.2 of Qiu et al. (2018) after the change of variable (47).

We thus focus only on the proof for the case l > 1.

To simplify the notation, we denote

, and replace

Thus

where the second equality results from the fact that x(u, ??, ??) ??? x c (u, ??, ??) = x 0 (??) depends only on ?? (Proposition 1(b).)

Just like the proof of Proposition 1(a), we take the integral of ?? first, and define

where |E 1 (u, ??, ??) + E 2 (u, ??, ??)| 2 du

Hence

We thus seek to estimate E 1 and E 2 individually.

To bound E 2 , we let

Then

and, for any given v and ??

where the last inequality comes from (74).

Moreover, for any given u and ??,

where the last inequality is again because of (74).

Thus, for any given ??,

?? ,??,m (v, u, ??) du dv gradient descent (SGD) with decreasing learning rate from 10 ???2 (10 ???1 ) to 10 ???4 (10 ???3 ) is used to train all networks without (with) batch-normalization for 160 epochs.

Layer (Regular) CNN ScDCFNet 1 c3x3x1xM ReLU ap2x2 sc(9)9x9x1xM ReLU sap2x2 2 c3x3xMx2M ReLU ap2x2 sc(9)9x9xL??xMx2M ReLU sap2x2 3 c3x3x2Mx4M ReLU ap2x2 sc(9)9x9xL??x2Mx4M ReLU sap2x2 4 fc64 ReLU fc10 softmax-loss fc64 ReLU fc10 softmax-loss Table 2 : Network architectures used for the experiments in Section 5.2.

cLxLxM'xM: a regular convolutional layer with M' input channels, M output channels, and LxL spatial kernels.

sc(N??)LxLxM'xM: the first-layer convolution operation (5) in ScDCFNet, where N?? is the number of the uniform grid points to discretize the scale interval [???1.6, 0], and LxL is the spatial kernel size on the largest scale ?? = 0.

sc(N??)LxLxL??xM'xM: the l-th layer (l > 1) convolution operation (6) in ScDCFNet, where the extra symbol L?? stands for the filter size in ??.

apLxL(sapLxL): the regular (ST -equivariant) LxL average-pooling.

fcM: a fully connected layer with M output channels.

Batch-normalization layers are added to each convolutional layer if adopted during training.

The network architectures for the SDCFNet and regular CNN auto-encoders are shown in Table 3 .

The filter expansion in the SDCFNet auto-encoder is truncated to K = 8 and K ?? = 3.

SGD with decreasing learning rate from 10 ???2 to 10 ???4 is used to train both networks for 20 epochs.

Table 3 : Architectures of the auto-encoders used for the experiment in Section 5.3.

The encoded representation is the output of the second layer.

ctLxLxM'xM: transposed-convolutional layers with M' input channels, M output channels, and LxL spatial kernels.

us2x2: 2x2 spatial upsampling.

See the caption of Table 2 for the definitions of other symbols.

Batch-normalization (not shown in the table) is used after each convolutional layer.

@highlight

We construct scale-equivariant convolutional neural networks in the most general form with both computational efficiency and proved deformation robustness.

@highlight

The authors propose a CNN architecture that is theoretically equivariant to isotropic scalings and translations by adding an extra scale-dimension to activation tensors.