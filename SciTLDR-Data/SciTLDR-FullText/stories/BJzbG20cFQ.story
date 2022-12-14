The problem of visual metamerism is defined as finding a family of perceptually indistinguishable, yet physically different images.

In this paper, we propose our NeuroFovea metamer model, a foveated generative model that is based on a mixture of peripheral representations and style transfer forward-pass algorithms.

Our gradient-descent free model is parametrized by a foveated VGG19 encoder-decoder which allows us to encode images in high dimensional space and interpolate between the content and texture information with adaptive instance normalization anywhere in the visual field.

Our contributions include: 1) A framework for computing metamers that resembles a noisy communication system via a foveated feed-forward encoder-decoder network – We observe that metamerism arises as a byproduct of noisy perturbations that partially lie in the perceptual null space; 2) A perceptual optimization scheme as a solution to the hyperparametric nature of our metamer model that requires tuning of the image-texture tradeoff coefficients everywhere in the visual field which are a consequence of internal noise; 3) An ABX psychophysical evaluation of our metamers where we also find that the rate of growth of the receptive fields in our model match V1 for reference metamers and V2 between synthesized samples.

Our model also renders metamers at roughly a second, presenting a ×1000 speed-up compared to the previous work, which now allows for tractable data-driven metamer experiments.

The history of metamers originally started through color matching theory, where two light sources were used to match a test light's wavelength, until both light sources are indistinguishable from each other producing what is called a color metamer.

This leads to the definition of visual metamerism: when two physically different stimuli produce the same perceptual response (See Figure 1 for an example).

Motivated by BID1 's work of local texture matching in the periphery as a mechanism that explains visual crowding, BID7 were the first to create such point-of-fixation driven metamers through such local texture matching models that tile the entire visual field given log-polar pooling regions that simulate the V1 and V2 receptive field sizes, as well as having global image statistics that match the metamer with the original image.

The essence of their algorithm is to use gradient descent to match the local texture BID22 ) and image statistics of the original image throughout the visual field given a point of fixation until convergence thus producing two images that are perceptually indistinguishable to each other.

However, metamerism research currently faces 2 main limitations: The first is that metamer rendering faces no unique solution.

Consider the potentially trivial examples of having an image I and its metamer M where all pixel values are identical except for one which is set to zero (making this difference unnoticeable), or the case where the metameric response arises from an imperceptible equal perturbation across all pixels as suggested in BID16 ; BID7 .

This is a concept similar to Just Noticeable Differences BID21 BID5 ).

However, like the work of BID7 ; BID17 ; BID23 ; BID1 , we are interested in creating point-of-fixation driven metamers, which create images that preserve information in the fovea, yet lose spatial information in the periphery such that this loss is unnoticeable contingent of a point of fixation (Figure 1 ).

The second issue is that the current state of the art for a full field of view rendering of a 512px × 512px metamer takes 6 hours for a grayscale image and roughly a day for a color image.

This computational constraint makes data-

Same or Different?Figure 1: Two visual metamers are physically different images that when fixated on the orange dot (center), should remain perceptually indistinguishable to each other for an observer.

Colored circles highlight different distortions in the visual field that observers do not perceive in our model.

driven experiments intractable if they require thousands of metamers.

From a practical perspective, creating metamers that are quick to compute may lead to computational efficiency in rendering of VR foveated displays and creation of novel neuroscience experiments that require metameric stimuli such as gaze-contingent displays, or metameric videos for fMRI, EEG, or Eye-Tracking.

We think there is a way to capitalize metamer understanding and rendering given the developments made in the field of style transfer.

We know that the original model of BID7 consists of a local texture matching procedure for multiple pooling regions in the visual field as well as global image content matching.

If we can find a way to perform localized style transfer with proper texture statistics for all the pooling regions in the visual field, and if the metamerism via texture-matching hypothesis is correct -we can in theory successfully render a metamer.

Within the context of style transfer, we would want a complete and flexible framework where a single network can encode any style (or texture) without the need to re-train, and with the power of producing style transfer with a single forward pass, thus enabling real-time applications.

Furthermore, we would want such framework to also control for spatial and scale factors ) to enable foveated pooling BID0 ; BID6 ) which is critical in metamer rendering.

The very recent work of BID14 , provides such framework through adaptive instance normalization (AdaIN), where the content image is stylized by adjusting the mean and standard deviation of the channel activations of the encoded representation to match with the style.

They achieve results that rival those of BID27 BID16 , with the added benefit of not being limited to a single texture in a feed-forward pipeline.

In our model: we stack a peripheral architecture on top of a VGGNet BID25 ) in its encoded feature space, to map an image into a perceptual space.

We then add internal noise in the encoded space of our model as a characterization that perceptual systems are noisy.

We find that inverting such modified image representation via a decoder results in a metamer.

This breaks down our model into a foveated feed-forward 'auto' style transfer network, where the input image plays the role both of the content and the style, and internal network noise (stylized with the content statistics) serves as a proxy for intrinsic image texture.

While our model uses AdaIN for style transfer and a VGGNet for texture statistics, our pipeline is extendible to other models that successfully execute style transfer and capture proper texture statistics BID28 ).

To construct our metamer we propose the following statement: A metamer M can be rendered by transferring k localized styles over a content image I, controlled by a set of style-to-content ratios α i for every pooling region (i-th receptive field).

More formally, our goal is to find a Metamer function M(•) : I → M, where an input image I ∈ R L is fed through a VGG-Net encoder E(·) : R L → R D which is both the content and the style image, to produce the content feature C ∈ R D , where C = E(I) as shown in Figure 2 .

Let L = C × H × W, and D = C × H × W where {C,C }, {H, H }, {W, W } are the image/layer channels, height, width given the convolutional structure of the encoder (we drop fully connected layers).

A noise patch colored via ZCA BID3 Figure 2: The NeuroFovea metamer generation schematic: An input image and a noise patch are fed through a VGG-Net encoder into a new feature space.

Through spatial control we can produce an interpolation for each pooling region in such feature space between the stylized-noise (texture), and the content (the input image).

This is how we successfully impose both global image and local texture-like constraints in every pooling region.

The metamer is the output of the pooled (and interpolated) feature vector through the Meta VGG-Net Decoder.

content image's mean and variance N ∼ (µ I , σ 2 I ) ∈ R L is also fed through the same VGG-Net encoder producing the noise feature N ∈ R D , where N = E(N).

This is the internal perceptual noise of the system which will later on serve us as a proxy for texture encoding.

These vectors are masked through spatial control a la , and the noise is stylized via S(·) : R D → R D with the content which encodes the texture representation of the content in the feature space through Adaptive Instance Normalization (AdaIN).

A target feature T i ∈ R D is defined as an interpolation between the stylized noise S(N i ) and the content C i modulated by α, in the feature space R D for every i-th pooling region: DISPLAYFORM0 In other words, in our quest to probe for metamerism, we are finding an intermediate representation (the convex combination) between two vectors representing the image and its texturized version (the stylized noise) in R D per pooling region as seen in Figure 3 .

Within the framework of style transfer, we could think of this as a content-vs-style or structure-vs-texture tradeoff, since the style and the content image are the same.

Similar interpolations have been explored in BID13 via a joint pixel and network space minimization.

The final target feature vector T is the masked sum of every T i with spatial control masks w i s.t.

T = w i T i .

The metamer is the output of the Meta VGG-Net decoder D(·) on T, where the decoder receives only one vector (T) and produces a global decoded output.

Our Meta VGG-Net Decoder compensates for small artifacts by stacking a pix2pix BID15 U-Net refinement module which was trained on the Encoder-Decoder outputs to map to the original high resolution image.

Figure 2 fully describes our model, and the metamer transform is computed via: DISPLAYFORM1 where E is the foveated encoder that is defined as the sum of encoder outputs over all the k pooling regions (our spatial controls masks w i ) in the visual field.

Note that the decoder was not trained to generate metamers, but rather to invert the encoded image and act as E −1 .

It happens to be the Figure 3 : Interpolating between an image's intrinsic content and texture via a convex combination in the output of the VGG19 Encoder E.

Here we are treating the patch as a single pooling region.

In our model, this interpolation given Eq. 1 is done for every pooling region in the visual field.case that perturbing the encoded representation in the direction of the stylized noise by an amount specified by the size of the pooling regions, outputs a metamer.

Additional specifications and training of our model can be seen in the Supplementary Material.

Within the framework of metamerism where distortions lie on the perceptual null space as proposed initially in color matching theory, and also in BID7 for images, we can think of our model as a direct transform that is maximizing how much information to discard depending on the texture-like properties of the image and the size of the receptive fields.

Consider the following: if our interpolation is projected from the encoded space to the perceptual space via P, from Eq. 1 we get PT i = P(1 − α)C i (I) + P(α)S(N i ), it follows that for each receptive field: DISPLAYFORM0 where S is the projection of the difference vector on the perceptual space, and S ⊥ (N i ) is the orthogonal component perpendicular to such vector which lies in the perceptual null space (PS ⊥ (N i ) = 0).

The value of these components will change depending on the location of C i and S(N i ), and the geometry of the encoded space.

If ||S (N i )|| 2 2 < , (i.e. the image patch has strong texture-like properties), then α can vary above its critical value given that S ⊥ (N i ) is in the null space of P and the distortion term will still be small; but if ||S (N i )|| 2 2 > , α can not exceed its critical value for the metamerism condition to hold (PT i ≈ PC i ).

Thus our interest is in computing the maximal average amount of distortion (driven by α) given human sensitivity before observers can tell the difference.

This is illustrated in FIG0 via the blue circle around C i in the perceptual space which shows the metameric boundary for any distortion.

One can also see the resemblance of the model to a noisy communication system in the context of information theory.

The information source is the image I, the transmitter and the receiver are the encoder and decoders (E, D) respectively, and the noise source is the encoded noise patch E(N) imposing texture distortions in the visual field, and the destination is the metamer M. Highlighting this equivalence is important as metamerism can also be explored within the context of image compression and rate-distortion theory as in .

Such approaches are beyond the scope of this paper, however they are worth exploring in future work as most metamer models purely involve texture and image analysis-synthesis matching paradigms that are gradient-descent based.3 Hyperparameteric nature of our model Similar to our model, the BID7 model (hereto be abbreviated FS) requires a scale parameter s which controls the rate of growth of the receptive fields as a function of eccentricity.

This parameter should be maximized such that an upperbound for perceptual discrimination is found.

Given that texture and image matching occurs in each one of the pooling regions: a high scaling factor will likely make the image rapidly distinguishable from the original as distortions are more apparent in the periphery.

Conversely, a low scaling factor might gaurantee metamerism even if the texture statistics are not fully correct given that smaller pooling regions will simulate weak effects of crowding.

Low scaling factors in that sense are potentially uninteresting -it is the value up until humans can tell the difference that is critical BID21 ).

FS set out to find such critical value via a psychophysical experiment where they perform the following single-variable optimization to find such upper bound: DISPLAYFORM1 DISPLAYFORM2 is the index of detectability for each observer θ obs , Φ is the cumulative of the gaussian distribution, and HR and FA are the hit rate and false alarm rates as defined in BID11 .

However, our model is different in regards to a set of s = 0.25 s=0.50 s=0.75 s=1.0Figure 5: Potential issues of psychophysical intractability for the joint estimation of (s) and γ(·) as described by our model.

Running a psychophysical experiment that runs an exhaustive search for upper bounds for the scale and distortion parameters for every receptive field is intractable.

The goal of Experiment 1 is to solve this intractabitilty posed formally in Eq. 6 via a simulated experiment.hyperparametersᾱ that we must estimate everywhere in the visual field as summarized by the γ function, where we assume α to be tangentially isotropic: DISPLAYFORM3 where each α represents the maximum amount of distortion (Eq. 1) that is allowed for every receptive field in the visual periphery before an observer will notice.

At a first glance, it is not trivial to know if α should be a function of scale, retinal eccentricity, receptive field size, image content or potentially a combination of the before-mentioned (hence the • in the γ function's argument).Thus, the motivation of α seems uncertain and perhaps un-necessary from the Occam's razor perspective of model simplicity.

This raises the question: Why does the FS model not require any additional hyperparameters, requiring only a single scale (s) parameter?

The answer lies in the nature of their model which is gradient descent based and where local texture statistics are matched for every pooling region in the visual field, while preserving global image structural information.

When such condition is reached, no further synthesis steps are required as it is an equilibrium point.

Indeed, the experiments of BID30 have shown that images do not remain metameric if the structural information of a pooling region is discarded while purely retaining the texture statistics of BID22 .

This motivates the purpose of α where we interpolate between structural and texture representation.

Thus our goal is to find that equilibirum point in one-shot, given that our model is purely feed-forward and requires no gradient-descent (Eq. 2).

At the expense of this artifice, we run into the challenge of facing a multi-variable optimization problem that has the risk of being psychophysically intractable.

Analogous to FS, we must solve: Figure 5 shows the potential intractability: each observer would have to run multiple rounds of an ABX experiment for a collection of many scales and α values for each location in the visual field.

Consider: (S scales) × (k pooling regions) × (α m step size for each α) × (N images) × (w trials): S kNα m w trials per observer.

DISPLAYFORM4 We will show in Experiment 1 that one solution to Eq. 6 is to find a relationship between each set of α's and the scale, expressed via the γ function.

This requires a two stage process: 1) Showing that such γ exists; 2) Estimate γ given s. If this is achieved, we can relax the multi-variable optimization into a single variable optimization problem, where 0 < d (s, γ(•; s)|θ obs ) < , and: DISPLAYFORM5 4 ExperimentsThe goal of Experiment 1 is to estimate γ as a function of s via a computational simulation as a proxy for running human psychophysics.

Once it is computed, we have reduced our minimization to a tractable single variable optimization problem.

We will then proceed to Experiment 2 where we will perform an ABX experiment on human observers by varying the scale to render visual metamers as originally proposed by FS.

We will use the images shown in FIG1 for both our experiments.

Existence and shape of γ:

Given some biological priors, we would like γ to satisfy these properties: DISPLAYFORM0 , where z ∈ Z is parametrized by the size (radius) of each receptive field (pooling region) which grows with eccentricity in humans.

2.

γ is continuous and monotonically non-decreasing since more information should not be gained given larger crowding effects as receptive field size increases in the periphery.

3.

γ has a unique zero at γ(0) = 0.

Under ideal assumptions there is no loss of information in the fovea, where the size of the receptive fields asymptotes to zero.

Indeed, we found that γ is sigmoidal, and is a function of z, parametrized by s: DISPLAYFORM1 Figure 7: Perceptual optimization.

To numerically estimate the amount of α-noise distortion for each receptive field in our metamer model we need to find a way to simulate the perceptual loss made by a human observer when trying to discriminate between metamers and original images.

We will define a perceptual loss L that has the goal of matching the distortions via SSIM of a gradient descent based method such as the FS metamers, and the NeuroFovea metamers (NF) with their reference images -a strategy similar to Laparra et al. FORMULA0 used for perceptual rendering.

We chose SSIM as it is a standard IQA metric that is monotonic with human judgements, although other metrics such as MS-SSIM and IW-SSIM show similar tuning properties for γ as shown in the Supplementary Material.

Indeed the reference image I for the NF metamer is limited by the autoencoder-like nature of the model where the bottleneck usually limits perfect reconstruction s.t.

I = D(E(I))| (α=0) , where I → I, and they are only equal if the encoder-decoder pair (E, D) allows for lossless compression.

Since we can not define a direct loss function L between the metamers, we will need their reference images to define a convex surrogate loss function L R .

The goal of this function should be to match the perceptual loss of both metamers for each receptive field k when compared to their reference images: the original image I for the FS model, and the decoded image I for the NF model: DISPLAYFORM0 and α i should be minimized for each k pooling region via: α 0 = arg min α L R (α|k) for the collection of N images.

The intuition behind this procedure is shown in Figure 7 .

Note that if I = I, i.e. there is perfect lossless compression and reconstruction given the choice of encoder and decoder, then the optimization is performed with reference to the same original image.

This is an important observation as the reconstruction capacity of our decoder is limited despite E(MS-SSIM(I, I ) = 0.86 ± 0.04.

Only using the original image in the optimization yields poor local minima at α = 0.

Despite such limitation, we show that reference metamers can still be achieved for our lossy compression model.

Results: A collection of 10 images were used in our experiments.

We then computed the SSIM score for each FS and NF image paired with their reference image across each receptive field (R.F.) and averaged those that belonged to the same retinal eccentricity.

FIG2 (top) shows these results, as well as the convex nature of the loss function displayed in the bottom.

This procedure was repeated for all the eccentricity-dependent receptive fields for a collection of 5 values of scale: {0.3, 0.4, 0.5, 0.6, 0.7}. A sigmoid to estimate γ was then fitted to each α per R.F. parametrized by scale via least squares.

This gave us a collection of d values that control the slope rate of the sigmoid (Eq. 8).

These were d : {1.240, 1.196, 1.363, 1.311, 1.355} respectively per scale, and {d} = 1.281 for the ensemble of all scales.

We then conducted a 10000 sample permutation test between the pair of (z s , α s ) points per scale and the ensemble of points across all scales ({z}, {α}) that verified that their variation is statistically non-significant (p ≥ 0.05).

FIG3 illustrates the results from such procedure.

We can conclude that the parameters of γ do not vary as we vary scale.

In other words, the α = γ(z) function is fixed, and the scale parameter itself which controls receptive field size will implicitly modulate the maximum α-noise distortion with a unique γ function.

If the scale factor is small, the maximum noise distortion in the far periphery will be small and vice versa if the scale is large.

We should point out that FIG3 might suggest that the maximal noise distortion is contingent on image content as the scores are not uniform tangentially for the receptive fields that lie on the same eccentricity ring.

Indeed, we did simplify our model by computing an average and fitting the sigmoid.

However, computing an average should approximate the maximal distortion for the receptive field size on that eccentricity in the perceptual space for the human observer i.e. the metameric boundary.

We elaborate more on this idea in the discussion section.

Given that we have estimated the value of α anywhere in the visual field via the γ function, we can now render our metamers as a function of the single scaling parameter (s), as the receptive field size z is also a function of s as shown in FIG4 .

The psychophysical optimization procedure is now tractable on human observers and has the following form where 0 < d (s, γ(z(s); s)|θ obs ) < : DISPLAYFORM0 Inspired by the evaluations of Wallis et al. FORMULA0 , we wanted to test our metamers on a group of observers performing two different ABX discrimination tasks in a roving design: We had a group of 3 observers agnostic to the peripheral distortions and purposes of the experiment performed an interleaved Synth vs Synth and Synth vs Reference experiment for NF metamers for the previous set of images FIG1 ).

An SR EyeLink 1000 desk mount was used to monitor their gaze for the center forced fixation ABX task as shown in Figure 11 .

In each trial, observers were shown 3 images where their task is to match the third image to the 1st or the 2nd.

Each observer saw each of the 10 images 30 times per scaling factor (5) per discriminability type (2) totalling 3000 trials per observer.

Images were rendered at 512 × 512 px, and we fixed the monitor at 52cm viewing distance and 800 × 600px resolution so that the stimuli subtended 26 deg ×26 deg.

The monitor was linearly calibrated with a maximum luminance of 115.83 ± 2.12 cd/m 2 .

We then estimated the critical scaling factor s 0 , and absorbing factors β 0 of the roving ABX task to fit a psychometric function for Proportion Correct ( DISPLAYFORM1 Results: Absorbing gain factors β 0 and critical scales s 0 per observer are shown in Figure 12 , where the fits were made using a least squares curve fitting model and bootstrap sampling n = 10000 times to produce the 68% confidence intervals.

Lapse rates (λ) were also included for robustness of fit as in BID33 .

Analogous to BID7 , we find that the critical scaling factor is 0.51 when doing the Synth vs Synth experiment which match V2, a critical region in the brain that has been identified to respond to texture as in BID19 BID34 .

This suggests that the parameters we use to capture and transfer texture statistics which are different from the correlations of a steerable pyramid decomposition as proposed in BID22 , might the match perceptual discrimination rates of the FS metamers.

This does not imply that the models are perceptually equivalent, but it aligns with the results of BID28 which shows that even a basis of random filters can also capture texture statistics, thus different flavors of metamer models can be created with different statistics.

In addition, we find that the critical scaling factor for the Synth vs Reference experiment is less than 0.5 (∼ 0.25, matching V1) for the pooled observer as validated recently by BID29 for their CNN synthesis and FS model for the Synth vs Reference condition.

There has been a recent surge in interest with regards to developing and testing new metamer models: The SideEye model developed by BID8 , uses a fully convolutional network (FCN) as in BID20 and learns to map an input image into a Texture Tiling Model (TTM) mongrel BID23 ).

Their end-to-end model is also feedforward like ours, but no use of noise is incorporated in the generation pipeline making their model fully deterministic.

At first glance this seems to be an advantage rather a limitation, however it limits the biological plausilibility of metameric response as the same input image should be able to create more than one metamer.

Another model which has recently been proposed is the CNN synthesis model developed by BID29 .

The CNN synthesis model is gradient-descent based and is closest in flavor to the FS model, with the difference that their texture statistics are provided by a gramian matrix of filter activations of multiple layers of a VGGNet, rather than those used in BID22 .The question of whether the scaling parameter is the only parameter to be optimized for metamerism still seems to be open.

This has been questioned early in BID23 , and recently proposed and studied by BID29 , who suggest that metamers are driven by image content, rather than bouma's law (scaling factor).

FIG3 suggests that on average, it does seem that α must increase in proportion to retinal eccentricity, but this is conditioned by the image content of each receptive field.

We believe that the hyperparametric nature of our model sheds some light into reconciling these two theories.

Recall that in FIG0 , we found that certain images can be pushed stronger in the direction of it's texturized version versus others given their location in the encoded space, the local geometry of the surface, and their projection in the perceptual space.

This suggests that the average maximal distortion one can do is fixed contingent on the size of the receptive field, but we are allowed to push further (increase α) for some images more than others, because the direction of the distortion lies closer to the perceptual null space (making this difference perceptually un-noticeable to the human observer).

This is usually the case for regions of images that are periodic like skies, or grass.

Along the same lines, we elaborate in the Supplementary Material on how our model may potentially explain why creating synthesized samples are metameric to each other at the scales of (V1;V2), but only generated samples at the scale of V1 (s = 0.25) are metameric to the reference image.

Our model is also different to others (FS and recently Wallis et al. FORMULA0 ) given the role of noise in the computational pipeline.

The previously mentioned models used noise as an initial seed for the texture matching pipeline via gradient-descent, while we use noise as a proxy for texture distortion that is directly associated with crowding in the visual field.

One could argue that the same response is achieved via both approaches, but our approach seems to be more biologically plausible at the algorithmic level.

In our model an image is fed through a non-linear hierarchical system (simulated through a deep-net), and is corrupted by noise that matches the texture properties of the input image (via AdaIN).

This perceptual representation is perturbed along the direction of the texture-matched patch for each receptive field, and inverting such perturbed representation results in a metamer.

FIG7 illustrates such perturbations which produce metamers when projected to a 2D subspace via the locally linear embedding (LLE) algorithm (Roweis & Saul FORMULA1 ).

Indeed, the 10 encoded images do not fully overlap to each other and they are quite distant as seen in the 2D projection.

However, foveated representations when perturbed with texture-like noise seem to finely tile the perceptual space, and might act as a type of biological regularizer for human observers who are consistently making eye-movements when processing visual information.

This suggests that robust representations might be achieved in the human visual system given its foveated nature as non-uniform high-resolution imagery does not map to the same point in perceptual space.

If this holds, perceptually invariant data-augmentation schemes driven by metamerism may be a useful enhancement for artificial systems that react oddly to adversarial perturbations that exploit coarse perceptual mappings (Goodfellow et al. FORMULA0 ; BID26 ; Berardino et al. FORMULA0 ).Understanding the underlying representations of metamerism in the human visual system still remains a challenge.

In this paper we propose a model that emulates metameric responses via a foveated feed-forward style transfer network.

We find that correctly calibrating such perturbations (a consequence of internal noise that match texture representation) in the perceptual space and inverting such encoded representation results in a metamer.

Though our model is hyper-parametric in nature we propose a way to reduce the parametrization via a perceptual optimization scheme.

Via a psychophysical experiment we empirically find that the critical scaling factor also matches the rate of growth of the receptive fields in V2 (s = 0.5) as in BID7 when performing visual discrimination between synthesized metamers, and match V1 (0.25) for reference metamers similar to BID29 .

Finally, while our choice of texture statistics and transfer is relu4_1 of a VGG19 and AdaIN respectively, our ×1000-fold accelerated feed-forward metamer generation pipeline should be extendible to other models that correctly compute texture/style statistics and transfer.

This opens the door to rapidly generating multiple flavors of visual metamers with applications in neuroscience and computer vision.6 Supplementary Material FIG0 : Reference Metamers at the scale of s = 0.25, at which they are indiscriminable to the human observer.

The color coding scheme matches the data points of the optimization in Experiment 1 and the psychophysics of Experiment 2.

All images used in the experiments were generated originally at 512 × 512 px subtending 26 × 26 d.v.a (degrees of visual angle).

for each α ∈ [0 : DISPLAYFORM0 Compute metamer M NF (I) 9:end for 10:Find the α for each receptive field that minimizes: E(∆-SSIM) 2 .

11:Fit the γ s (•) function to collection of α values.

12:endfor 13: end for 14: Perform Permutation test on γ s for all s. 15: if γ s is independent of s then 16: γ s = γ 17: else 18:Perform regression of parameters of γ s as a function f of s. 19: DISPLAYFORM1 end if 21: end procedure

Algorithm 1 fully describes the outline of Experiment 1.

We use k = k p + k f spatial control windows, k p pooling regions (θ r receptive fields × θ t eccentricity rings) and k f = 1 fovea (at an approximate 3 deg radius).

Computing the metamers for the scales of {0.3, 0.4, 0.5, 0.6, 0.7} required {300, 186, 125, 102, 90} pooling regions excluding the fovea where we applied local style transfer.

Details regarding the decoder network architecture and training can be seen in BID14 .

We used the publicly available code by Huang and Belongie for our decoder which was trained on ImageNet and a collection of publicly available paintings to learn how to invert texture as well.

In their training pipeline, the encoder is fixed and the decoder is trained to learn how to invert the structure of the content image, and the texture of the style image, thus when the content and style image are the same, then the decoder approximates the inverse of the encoder (D ∼ E −1 ).

We also re-trained another decoder on a set of 100 images all being scenes (as a control to check for potential differences), and achieved similar outputs (visual inspection) to the publicly available one of Huang & Belongie.

The dimensionality of the input of the encoder is 1 × 512 × 512, and the dimensionality of the output (relu4_1) is 512 × 64 × 64, it is at the 64 × 64 resolution that we are applying foveated pooling from the initial guidance channels of the 512 × 512 input.

Constructions of biologically-tuned peripheral representations are explained in detail in BID7 BID0 ; BID6 , and are governed by the following equations: DISPLAYFORM0 g n (e) = f log(e) − [log(e 0 ) + w e (n + 1)] w e ; w e = log(e r ) − log(e 0 ) N e ; n = 0, ..., N e − 1where f (x) is a cosine profiling function that smoothes a regular step function, and h n (θ),g n (e), are the averaging values of the pooling region w i at a specific angle θ and radial eccentricity e in the visual field.

In addition we used the default values of visual radius of e r = 26 deg, and e 0 = 0.25 deg 1 , and t 0 = 1/2.

The scale s defines the number of eccentricities N e , as well as the number of polar pooling regions N θ from 0, 2π].

We perform the foveated pooling operation on the output of the Encoder.

Since the encoder is fully convolutional with no fully connected layers, guidance channels can be used to do localized (foveated) style transfer.

Our pix2pix U-Net refinement module took 3 days to train on a Titan X GPU, and was trained with 64 crops (256 × 256) per image on 100 images, including horizontally mirrored versions.

We ran 200 training epochs of these 12800 images on the U-Net architecture proposed by BID15 which preserves local image structure given an adversarial and L2 loss.

The following table summarizes the main similarities and differences across all current models: FORMULA0 SideEye FORMULA0 NF ( The FS model can not directly compute an inverse of the encoded representation to generate a metamer, requiring an iterative gradient descent procedure.

Our NF model is limited by the capacity of the encoder-decoder architecture as it does not achieve lossless compression (perfect reconstruction).

The distance in green between a V1 metamer and the content image is the same as the two V2 metamers, potentially explaining how they are perceptually indistinguishable to each other at different scaling factors given the type of ABX task.

DISPLAYFORM0 A B FIG1 : Decomposition and overview of the metamer generation process in the Image space, the Encoded space and the Perceptual space.

The original image patch is coded in blue, the V1 metamers are coded in purple, and the V2 metamers are coded in pink.

Dark brown represents the initial white noise that is later stylized via AdaIN through S(•).

Note that these two points are far away to each other in image space, but quite closeby in perceptual space as they are also 'metameric' to each other.

They are not placed on the actual encoded manifold since these points are not in the near vicinity of either C nor S(N), as they have no scene-like structure.

The interpolation for maximal distortion is done along the line between C and S(N), these are the points in blue and red in the encoded space which represent the extremes of α = 0.0 and α = 1.0 respectively.

In FIG1 , we illustrate the metamer generation process for two sample metamers, given different noise perturbations.

Here, we decompose FIG0 into two separate ones for each metamer given each noise perturbation, and provide an additional visualization of the projection of the metamers in perceptual space, gaining theoretical insight on how and why metamerism arises for the synth-vs-synth condition in V2, and the synth-vs-reference condition in V1 as we demonstrated experimentally.

Proportion Correct In a preliminary psychophysical study, we ran an experiment with a collection of 50 images and 6 observers on the FS metamers.

Observers performed a single session of 200 trials of the FS metamers where the scale was fixed at s = 0.5.

We found the following: While we found that the synthesized images were metameric to each other for the scaling factor of 0.5, the FS metamers were not metameric to their reference high-quality images at the scale of 0.5.

Only a sub-group of observers: 'LR','SO','DS' scored well above chance in terms of discriminating the images in the ABX task.

These results are in synch with the evalutions done by BID29 , which varied scale and found a critical value to be less than 0.5 and rather closer to 0.25 within the range of V1.

The motivation behind estimating the lapse rate is to quantify how engaged was the observer in the experiment, as well as providing a robust estimate of the parameters in the fit of the psychometric functions.

Not accounting for lapse rate may dramatically affect the estimation of these parameters as suggested in BID33 .

In general lapse rates are computed by penalizing a psychometric function ψ(•) that ranges between some lower bound and upper bound usually BID35 1] .

To estimate the lapse rate λ, a new ψ (•) is defined to have the following form: DISPLAYFORM0 Recall that for us, our psychometric fitting function ψ(•) = PC ABX (s) is defined by Equation 11 and parametrized by both the absorbing factor β 0 and the critical scaling factor s 0 : DISPLAYFORM1 where we have: DISPLAYFORM2 To compute the new ψ (•), we notice first that our ψ is bounded between [0.5, 1] , and that the new ψ will be a linear combination of a correct guess for a lapse, and a correct decision for a non-lapse from which we obtain: DISPLAYFORM3 as derived in Hénaff (2018) which includes lapse rates for an AXB task.

When fitting the curves for each of the n = 10000 bootstrapped samples, we restricted the lapse rate to vary between λ = [0.00, 0.06] as suggested in BID33 , and found the following lapse rates:Observer 1: λ RS ZQ = 0.0248 ± 0.0209, λ S S ZQ = 0.0430 ± 0.0228.

Observer 2: λ RS AL = 0.0008 ± 0.0062, λ S S AL = 0.0166 ± 0.0215.

Observer 3: λ RS AG = 0.0141 ± 0.0243, λ S S AG = 0.0218 ± 0.0236.

We later averaged these lapse rates as there is an equal probability of each type of trial to appear (Synth vs Synth, or Reference vs Synth), and refitted each curve with the new pooled lapse rate estimates λ .

Indeed, each observer did both experiments in a roving paradigm, rather than doing one experiment after the other -thus we should only have one estimate for lapse rate per observer.

It is worth mentioning that re-performing the fits with separate lapse rates did not significantly affect the estimates of critical scaling values, as one might argue that higher lapse rates will significantly move the critical scaling factor estimates.

This is not the case as the absorbing factor β does not place an upper bound for the psychometric function at 1.Our critical estimates of lapse rates were: λ ZQ = 0.0339, λ AL = 0.0087, λ AG = 0.0179, as shown in Figure 12 .The estimates (critical scale (s 0 ), absorbing factor (β 0 ) and lapse rate (λ 0 )) shown for the pooled observer were obtained by averaging the estimates over the 3 observers.

In this subsection we show how the perceptual optimization pipeline is robust to a selection of IQA metrics such as MS-SSIM (multi-scale SSIM 2 ) from BID32 and IW-SSIM (information content weighted SSIM) from BID31 .There are 3 key observations that stem from these additional results:1.

The sigmoidal natural of the γ function is found again and is also scale independent, showing the broad applicability of our perceptual optimization scheme and how it is extendable to other IQA metrics that satisfy SSIM-like properties (upper bounded, symmetric and unique maximum).2.

The tuning curves of MS-SSIM and IW-SSIM look almost identical, given that IW-SSIM is not more than a weighted version of MS-SSIM where the weighting function is the mutual information between the encoded representations of the reference and distortion image across multiple resolutions.

Differences are stronger in IW-SSIM when the region over which it is evaluated is quite large (i.e. an entire image), however given that our pooling regions are quite small in size, the IW-SSIM score asymptotes to the MS-SSIM score.

In addition both scores converge to very similar values given that we are averaging these scores over the images and over all the pooling regions that lie within the same eccentricity ring.

We found that ∼ 90% of the maximum α's had the same values given the 20 point sampling grid that we use in our optimization.

Perhaps a different selection of IW hyperparameters (we used the default set), finer sampling schemes for the optimal value search, as well as averaging over more images, may produce visible differences between both metrics.3.

The sigmoidal slope is smaller for both IW-SSIM and MS-SSIM vs SSIM, which yields more conservative distortions (as α is smaller for each receptive field).

This implies that the model can still create metamers at the estimated found scaling factors of 0.21 and 0.50, however they may have different critical scaling factors for the reference vs synth experiment, and for the synth vs synth experiment.

Future work should focus on psychophysically finding these critical scaling factors, and if they still are within the range of rate of growth of receptive field sizes of V1 and V2.

The maximum α-noise distortion computed per pooling region, and collapsed over all images for each IQA metric.

Bottom: When averaging across all the pooling regions for each retinal eccentricity, we find that the γ function is invariant to scale as in our original experimentsuggesting that our perceptual optimization scheme is flexible across IQA metrics.

Figure 20: A permutation test was ran and determined that each γ function is also scale independent under the 99% confidence interval (CI), as we increased the CI to account for false discovery rates (FDR).

Indeed, when we perform the permutation tests and use a 95% confidence interval (shown in the figure with the vertical lines in cyan), all curves except for MS-SSIM and IW-SSIM only for the scaling factor of 0.3 show a significant difference p ∼ 0.02 (non FDR-corrected), potentially due to small receptive field sizes, which bias the estimates.

All other differences in the d parameter of the sigmoid function, with respect to the average fitted sigmoid, are statistically insignificant.

@highlight

We introduce a novel feed-forward framework to generate visual metamers

@highlight

Proposes a NeuroFovea model for generation of point-of-fixation metamers by using a style transfer approach via and Encoder-Decoder style architecture

@highlight

An analysis of metamerism and a model capable of rapidly producing metamers of value for experimental psychophysics and other domains.

@highlight

The paper proposes a fast method for generating visual metamers – physically different images that cannot be told apart from an original – via foveated, fast, arbitrary style transfer