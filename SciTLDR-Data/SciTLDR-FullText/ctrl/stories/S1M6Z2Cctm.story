The recent direction of unpaired image-to-image translation is on one hand very exciting as it alleviates the big burden in obtaining label-intensive pixel-to-pixel supervision, but it is on the other hand not fully satisfactory due to the presence of artifacts and degenerated transformations.

In this paper, we take a manifold view of the problem by introducing a smoothness term over the sample graph to attain harmonic functions to enforce consistent mappings during the translation.

We develop HarmonicGAN to learn bi-directional translations between the source and the target domains.

With the help of similarity-consistency, the inherent self-consistency property of samples can be maintained.

Distance metrics defined on two types of features including histogram and CNN are exploited.

Under an identical problem setting as CycleGAN, without additional manual inputs and only at a small training-time cost, HarmonicGAN demonstrates a significant qualitative and quantitative improvement over the state of the art, as well as improved interpretability.

We show experimental results in a number of applications including medical imaging, object transfiguration, and semantic labeling.

We outperform the competing methods in all tasks, and for a medical imaging task in particular our method turns CycleGAN from a failure to a success, halving the mean-squared error, and generating images that radiologists prefer over competing methods in 95% of cases.

Image-to-image translation BID15 aims to learn a mapping from a source domain to a target domain.

As a significant and challenging task in computer vision, image-to-image translation benefits many vision and graphics tasks, such as realistic image synthesis BID15 BID41 , medical image generation BID39 BID9 , and domain adaptation BID13 .

Given a pair of training images with detailed pixel-to-pixel correspondences between the source and the target, image-to-image translation can be cast as a regression problem using e.g. Fully Convolutional Neural Networks (FCNs) BID23 by minimizing e.g. the per-pixel prediction loss.

Recently, approaches using rich generative models based on Generative Adaptive Networks (GANs) BID11 BID27 BID0 have achieved astonishing success.

The main benefit of introducing GANs BID11 to image-to-image translation BID15 is to attain additional image-level (often through patches) feedback about the overall quality of the translation, and information which is not directly accessible through the per-pixel regression objective.

The method by BID15 is able to generate high-quality images, but it requires paired training data which is difficult to collect and often does not exist.

To perform translation without paired data, circularity-based approaches BID41 BID17 BID37 have been proposed to learn translations from a set to another set, using a circularity constraint to establish relationships between the source and target domains and forcing the result generated from a sample in the source domain to map back and generate the original sample.

The original image-to-image translation problem BID15 ) is supervised in pixel-level, whereas the unpaired image-to-image translation task BID41 ) is considered unsupervised, with pixel-level supervision absent but with adversarial supervision at the image-level (in the target domain) present.

By using a cycled regression for the pixel-level prediction (source→target→source) plus a term for the adversarial difference between the transferred images and the target images, CycleGAN is able to successfully, in many cases, train a translation model without paired source→target supervision.

However, lacking a mechanism to enforce regularity in the translation creates problems like in Fig To combat the above issue, in this paper we look at the problem of unpaired image-to-image translation from a manifold learning perspective BID33 BID28 .

Intuitively, the problem can be alleviated by introducing a regularization term in the translation, encouraging similar contents (based on textures or semantics) in the same image to undergo similar translations/transformations.

A common principle in manifold learning is to preserve local distances after the unfolding: forcing neighboring (similar) samples in the original space to be neighbors in the new space.

The same principle has been applied to graph-based semisupervised learning BID44 where harmonic functions with graph Laplacians BID45 BID2 are used to obtain regularized labels of unlabeled data points.

During the translation/transformation, some domain-specific attributes are changed, such as the colors, texture, and semantics of certain image regions.

Although there is no supervised information for these changes, certain consistency during the transformation is desirable, meaning that for image contents similar in the source space should also be similar in the target space.

Inspired by graphbased semi-supervised learning BID45 BID44 , we introduce smoothness terms to unpaired image-to-image translation BID41 by providing a stronger regularization for the translation/transformation between the source and target domains, aiming to exploit the "manifold structure" of the source and target domains.

For a pair of similar samples (two different locations in an image; one can think of them as two patches although the receptive fields of CNN are quite large), we add the smoothness term to minimize a weighted distance of the corresponding locations in the target image.

Note that two spatially distant samples might be neighbors in the feature space.

We name our algorithm HarmonicGAN as it behaves harmonically along with the circularity and adversarial constraints to learn a pair of dual translations between the source and target domains, as shown in FIG0 .

Distance metrics defined on two alternative features are adopted: (1) a low-level soft RGB histograms; and (2) CNN (VGG) features with pre-trained semantics.

We conduct experiments in a number of applications, showing that in each of them our method outperforms existing methods quantitatively, qualitatively, and with user studies.

For a medical imaging task BID6 that was recently calling attention to a major CycleGAN failure case (learning to accidentally add/remove tumors in an MRI image translation task), our proposed method provides a large improvement over CycleGAN, halving the mean-squared error, and generating images that radiologists prefer over competing methods in 95% of cases.

CONTRIBUTIONS 1.

We introduce smooth regularization over the graph for unpaired image-to-image translation to attain harmonic translations.2.

When building an end-to-end learning pipeline, we adopt two alternative types of feature measures to compute the weight matrix for the graph Laplacian, one based on a soft histogram BID35 and another based on semantic CNN (VGG) features BID31 .3.

We show that this method results in significantly improved consistency for transformations.

With experiments on multiple translation tasks, we demonstrate that HarmonicGAN outperforms the state-of-the-art.

As discussed in the introduction, the general image-to-image translation task in the deep learning era was pioneered by BID15 , but there are prior works such as image analogies BID12 ) that aim at a similar goal, along with other exemplar-based methods BID10 BID8 BID1 .

After BID15 , a series of other works have also exploited pixel-level reconstruction constraints to build connections between the source and target domains BID34 .

The image-to-image translation framework BID15 ) is very powerful but it requires a sufficient amount of training data with paired source to target images, which are often laborious to obtain in the general tasks such as labeling BID23 , synthesis BID5 , and style transfer BID14 .Unpaired image-to-image translation frameworks BID41 BID22 BID29 BID17 such as CycleGAN remove the requirement of having detailed pixellevel supervision.

In CycleGAN this is achieved by enforcing a bi-directional prediction from source to target and target back to source, with an adversarial penalty in the translated images in the target domain.

Similar unsupervised circularity-based approaches BID17 BID37 have also been developed.

The CycleGAN family models BID41 point to an exciting direction of unsupervised approaches but they also create artifacts in many applications.

As shown in FIG1 , one reason for this is that the circularity constraint in CycleGAN lacks the straightforward description of the target domain, so it may change the inherent properties of the original samples and generate unexpected results which are inconsistent at different image locations.

These failures have been prominently explored in recent works, showing that CycleGAN BID41 ) may add or remove tumors accidentally in cross-modal medical image synthesis BID6 , and that in the task of natural image transfiguration, e.g. from a horse to zebra, regions in the background may also be translated into a zebra-like texture ) (see FIG0 ).Here we propose HarmonicGAN that introduces a smoothness term into the CycleGAN framework to enforce a regularized translation, enforcing similar image content in the source space to also be similar in the target space.

We follow the general design principle in manifold learning BID33 BID28 and the development of harmonic functions in the graph-based semi-supervised learning literature BID45 BID2 BID44 BID36 .

There has been previous work, DistanceGAN BID3 , in which distance preservation was also implemented.

However, DistanceGAN differs from HarmonicGAN in (1) motivation, (2) formulation, (3) implementation, and (4) performance.

The primary motivation of DistanceGAN demonstrates an alternative loss term for the per-pixel difference in CycleGAN.In HarmonicGAN, we observe that the cycled per-pixel loss is effective and we aim to make the translation harmonic by introducing additional regularization.

The smoothness term acts as a graph Laplacian imposed on all pairs of samples (using random samples in implementation).

In the experimental results, we show that the artifacts in CycleGAN are still present in DistanceGAN, whereas HarmonicGAN provides a significant boost to the performance of CycleGAN.In addition, it is worth mentioning that the smoothness term proposed here is quite different from the binary term used in the Conditional Random Fields literature BID20 BID19 , either fully supervised BID4 BID40 or weakly-supervised BID32 BID21 .

The two differ in (1) output space (multi-class label vs. highdimensional features), (2) mathematical formulation (a joint conditional probably for the neighboring labels vs. a Laplacian function over the graph), (3) application domain (image labeling vs. image translation), (4) effectiveness (boundary smoothing vs. manifold structure preserving), and FORMULA7 the role in the overall algorithm (post-processing effect with relatively small improvement vs. large-area error correction).

Following the basic formulation in CycleGAN BID41 , for the source domain X and the target domain Y , we consider unpaired training samples {x k } N k=1 where x k ∈ X, and DISPLAYFORM0 where y k ∈ Y .

The goal of image-to-image translation is to learn a pair of dual mappings, including forward mapping G : X → Y and backward mapping F : Y → X. Two discriminators D X and D Y are adopted in BID41 to distinguish between real images and generated images.

In particular, the discriminator D X aims to distinguish real image {x} from the generated image DISPLAYFORM1 Therefore, the objective of adversarial constraint is applied in both source and target domains, expressed in BID41 as: DISPLAYFORM2 and DISPLAYFORM3 (2) For notational simplicity, we denote the GAN loss as DISPLAYFORM4 Since the data in the two domains are unpaired, a circularity constraint is introduced in BID41 to establish relationships between X and Y .

The circularity constraint enforces that G and F are a pair of inverse mappings, and that the translated sample can be mapped back to the original sample.

The circularity constraint contains consistencies in two aspects: the forward cycle DISPLAYFORM5 Thus, the circularity constraint is formulated as BID41 : DISPLAYFORM6 Here we rewrite the overall objective in BID41 to minimize as: DISPLAYFORM7 where the weights λ GAN and λ cyc control the importance of the corresponding objectives.

The full objective of circularity-based approach contains adversarial constraints and a circularity constraint.

The adversarial constraints ensure the generated samples are in the distribution of the source or target domain, but ignore the relationship between the input and output of the forward or backward translations.

The circularity constraint establishes connections between the source and target domain by forcing the forward and backward translations to be the inverse of each other.

However, CycleGAN has limitations: as shown in FIG1 , the circular projection might perfectly match the input, and the translated image might look very well like a real one, but the translated image may not maintain the inherent property of the input and contain a large artifact that is not connected to the input.

Here we propose a smoothness term to enforce a stronger correlation between the source and target domains that focuses on providing similarity-consistency between image patches during the translation.

The smoothness term defines a graph Laplacian with the minimal value achieved as a harmonic function.

We define the set consisting of individual image patches as the nodes of the graph G. x i is referred to as the feature vector of the i-th image patch in x ∈ X. For the image set X, we define the set that consists of individual samples (image patches) of source image set X as S = { x(i), i = 1..M } where M is the total number of the samples/patches.

An affinity measure (similarity) computed on image patch x(i) and image patch x(j), w ij (X) (a scalar), defines the edge on the graph G of S. The smoothness term acts as a graph Laplacian imposed on all pairs of image patches.

Therefore, we define a smoothness term over the graph as DISPLAYFORM0 where BID45 defines the affinity between two patches x(i) and x(j) based on their distances (e.g. measured on histogram or CNN features).

Dist[G( y)(i), G( y)(j)] defines the distance between two image patches after translation at the same locations.

In implementation, we first normalize the features to the scale of [0,1] and then use the L1 distance of normalized features as the Dist function (for both histogram and CNN features).

Similarly, we define a smoothness term for the backward part as DISPLAYFORM1 DISPLAYFORM2 The combined loss for the smoothness thus becomes smoothness term provides a stronger similarity-consistency between patches to maintain inherent properties of the images.

DISPLAYFORM3 Combining Eqn.

FORMULA7 and Eqn. (8), the overall objective for our proposed HarmonicGAN under the smoothness constraint becomes DISPLAYFORM4 Similar to the graph-based semi-supervised learning definition BID45 BID44 , the solution to Eqn.

(9) leads to a harmonic function.

The optimization process during training obtains: DISPLAYFORM5 The effectiveness of the smoothness term of Eqn.

FORMULA11 is evident.

In Fig. 4 , we show (using t-SNE BID24 ) that the local neighborhood structure is being preserved by HarmonicGAN, whereas CycleGAN results in two similar patches being far apart after translation.

In the smoothness constraint, the similarity of a pair of patches is measured on the features for each patch (sample point).

All the patches in an image form a graph.

Here we adopt two types of features: (1) a low-level soft histogram, and (2) pre-trained CNN (VGG) features that carry semantic information.

Soft histogram features are lightweight and easy to implement but without much semantic information; VGG requires an additional CNN network but carries more semantics.

We first design a weight matrix based on simple low-level RGB histograms.

To make the end-to-end learning system work, it is crucial to make the computation of gradient in the histograms derivable.

We adopt a soft histogram representation proposed in BID35 but fix the means and the bin size.

This histogram representation is differentiable and its gradient is back-propagateable.

This soft histogram function contains a family of linear basis functions ψ b , b = 1, . . .

, B, where B is the number of bins in the histogram.

As x i represents the i-th patch in image domain X, for each pixel j in x i , ψ b ( x i (j)) represents pixel j voting for the b-th bin, expressed as: DISPLAYFORM0 where µ b and w b are the center and width of the b-th bin.

The representation of x i in the RGB space is the linear combination of linear basis functions on all the pixels in x i , expressed as: DISPLAYFORM1 where φ h is the RGB histogram feature, b is the index of dimension of the RGB histogram representation, and j represents any pixel in the patch x i .

The RGB histogram representation φ h (X, i) of x i is a B-dimensional vector.

For some domains we instead use semantic features to acquire higher-level representations of patches.

The semantic representations are extracted from a pre-trained Convolutional Neural Network (CNN).

The CNN encodes semantically relevant features from training on a large-scale dataset.

It extracts semantic information of local patches in the image through multiple pooling or stride operators.

Each point in the feature maps of the CNN is a semantic descriptor of the corresponding image patch.

Additionally, the semantic features learned from the CNN are differentiable and the CNN can be integrated into HarmonicGAN and be trained end-to-end.

We instantiate the semantic feature φ s as a pre-trained CNN model e.g. VGGNet BID30 .

In implementation, we select the layer 4 3 after ReLU from VGG-16 network for computing the semantic features.

We evaluate the proposed method on three different applications: medical imaging, semantic labeling, and object transfiguration.

We compare against several unpaired image-to-image translation methods: CycleGAN BID41 , DiscoGAN BID17 , DistanceGAN BID3 , and UNIT BID22 .

We also provide two user studies as well as qualitative results.

The appendix provides additional results and analysis.

Medical imaging.

This task evaluates cross-modal medical image synthesis, Flair ↔ T1.

The models are trained on the BRATS dataset BID26 which contains paired MRI data to allow quantitative evaluation.

Similar to the previous work BID6 , we use a training set of 1400 image slices (50% healthy and 50% tumors) and a test set of 300, and use their unpaired training scenario.

We adopt the Mean Absolute Error (MAE) and the Mean Squared Error (MSE) between the generated images and the real images to evaluate the reconstruction errors, and further use the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) to evaluate the reconstruction quality of generated images.

Semantic labeling.

We also test our method on the labels ↔ photos task using the Cityscapes dataset BID7 under the unpaired setting as in the original CycleGAN paper.

For quantitative evaluation, in line with previous work, for labels → photos we adopt the "FCN score" BID15 , which evaluates how interpretable the generated photos are according to a semantic segmentation algorithm.

For photos → labels, we use the standard segmentation metrics, including per-pixel accuracy, per-class accuracy, and mean class Intersection-Over-Union (Class IOU).Object transfiguration.

Finally, we test our method on the horse ↔ zebra task using the standard CycleGAN dataset (2401 training images, 260 test images).

This task does not have a quantitative evaluation measure, so we instead provide a user study together with qualitative results.

We apply the proposed smoothness term on the framework of CycleGAN BID41 .

Similar with CycleGAN, we adopt the architecture of BID16 as the generator and the PatchGAN BID15 as the discriminator.

The log likelihood objective in the original GAN is replaced with a least-squared loss BID25 for more stable training.

We resize the input images to the size of 256 × 256.

For the histogram feature, we equally split the RGB range of [0, 255] to 16 bins, each with a range of 16.

Images are divided into non-overlapping patches of 8 × 8 and the histogram feature is computed on each patch.

For the semantic feature, we adopt a VGG network pre-trained on ImageNet to obtain semantic features.

We select the feature map of layer relu4 3 in VGG.

The loss weights are set as λ GAN = λ Smooth = 1, λ cyc = 10.

Following CycleGAN, we adopt the Adam optimizer BID18 with a learning rate of 0.0002.

The learning rate is fixed for the first 100 epochs and linearly decayed to zero over the next 100 epochs.

Medical imaging.

Table 1 shows the reconstruction performance on medical image synthesis, Flair ↔ T1.

The proposed method yields a large improvement over CycleGAN, showing lower MAE and MSE reconstruction losses, and higher PSNR and SSIM reconstruction scores, highlighting the significance of the proposed smoothness regularization.

HarmonicGAN based on histogram and VGG features shows similar performance; the reconstruction losses of histogram-based HarmonicGAN are slightly lower than the VGG-based one in Flair → T1, while they are slightly higher in T1 → Flair, indicating that both low-level RGB values and high-level CNN features can represent the inherent property of medical images well and help to maintain the smoothness-consistency of samples.

Table 1 : Reconstruction evaluation of cross-modal medical image synthesis on the BRATS dataset.

Semantic labeling.

We report semantic labeling results in TAB1 .

The proposed method using VGG features yields a 3% improvement in Pixel Accuracy in translation scores for photo ↔ label and also shows stable improvements in other metrics, clearly outperforming all competing methods.

The performance using a histogram is slightly lower than CycleGAN; we hypothesize that the reason is that the objects in photos have a large intra-class variance and inter-class similarity in appearance, e.g. cars have different colors, while vegetation and terrain have similar colors, thus the regularization of the RGB histogram is not appropriate to extract the inherent property of photos.

DISPLAYFORM0

Medical imaging.

We randomly selected 100 images from BRATS test set.

For each image, we showed one radiologist the real ground truth image, followed by images generated by CycleGAN, DistanceGAN and HarmonicGAN (different order for each image set to avoid bias).

The radiologist was told to evaluate similarity by how likely they would lead to the same clinical diagnosis, and was asked to rate similarity of the generation methods on a Likert scale from 1 to 5 (1 is not similar at all, 5 is exactly same).

Results are in shown in TAB2 .

In 95% of cases, the radiologist preferred images generated by our method over the competing methods, and the average Likert score was 4.00 compared to 1.68 for CycleGAN, confirming that our generated images are significantly better.

This is significant as it confirms that we solve the issue presented in a recent paper BID6 showing that CycleGAN can learn to accidentally add/remove tumors in images.

Object transfiguration.

We evaluate our algorithm on horse ↔ zebra with a human perceptual study.

We randomly selected 50 images from the horse2zebra test set and showed the input images and three generated images from CycleGAN, DistanceGAN and HarmonicGAN (with generated images in random order).

10 participants were asked to score the generated images on a Likert scale from 1 to 5 (as above).

As shown in TAB3 , the participants give the highest score to the proposed method (in 72% of cases), significantly more often than CycleGAN (in 28% of cases).

Additionally, the average Likert score of our method was 3.60, outperforming 3.16 of CycleGAN and 1.08 of DistanceGAN, indicating that our method generates better results.

Medical imaging.

Object transfiguration.

FIG4 shows a qualitative comparison of our method on the horse ↔ zebra task.

We observe that we correct several problems in CycleGAN, including not changing the background and performing more complete transformations.

More results and analysis are shown in FIG6 and FIG0 .

We introduce a smoothness term over the sample graph to enforce smoothness-consistency between the source and target domains.

We have shown that by introducing additional regularization to enforce consistent mappings during the image-to-image translation, the inherent self-consistency property of samples can be maintained.

Through a set of quantitative, qualitative and user studies, we have demonstrated that this results in a significant improvement over the current state-of-the-art methods in a number of applications including medical imaging, object transfiguration, and semantic labeling.

In a medical imaging task in particular our method provides a very significant improvement over CycleGAN.

(1) They show different motivations and formulations.

The distance constraint aims to preserve the distance between samples in the mapping in a direct way, so it minimizes the expectation of differences between distances in two domains.

The distance constraint in DistanceGAN is not doing a graph-based Laplacian to explicitly enforce smoothness.

In contrast, the smoothness constraint is designed from a graph Laplacian to build the similarity-consistency between image patches.

Thus, the smoothness constraint uses the affinity between two patches as weight to measure the similarityconsistency between two domains.

The whole idea is based on manifold learning.

The smoothness term defines a Laplacian ∆ = D − W , where W is our weight matrix and D is a diagonal matrix with D i = j w ij , thus, the smoothness term defines a graph Laplacian with the minimal value achieved as a harmonic function.(2) They are different in implementation.

The smoothness constraint in HarmonicGAN is computed on image patches while the distance constraint in DistanceGAN is computed on whole image samples.

Therefore, the smoothness constraint is fine-grained compared to the distance constraint.

Moreover, the distances in DistanceGAN is directly computed from the samples in each domain.

They scale the distances with the precomputed means and stds of two domains to reduce the effect of the gap between two domains.

Differently, the smoothness constraint in HarmonicGAN is measured on the features (Histogram or CNN features) of each patch, which maps samples in two domains into the same feature space and removes the gap between two domains.(3) They show different results.

FIG5 shows the qualitative results of CycleGAN, DistanceGAN and the proposed HarmonicGAN on the BRATS dataset.

As shown in FIG5 , the problem of randomly adding/removing tumors in the translation of CycleGAN is still present in the results of Distance-GAN, while HarmonicGAN can correct the location of tumors.

Table 1 shows the quantitative results on the whole test set, which also yields the same conclusion.

The results of DistanceGAN on four metrics are even worse than CycleGAN, while HarmonicGAN yields a large improvement over CycleGAN.

There are some fundamental differences between the CRF literature and our work.

They differ in output space, mathematical formulation, application domain, effectiveness, and the role in the over-all algorithm.

The similarity between CRF and HarmonicGAN lies the adoption of a regularization term: a binary term in the CRF case and a Laplacian term in HarmonicGAN.The smoothness term in HarmonicGAN is not about obtaining 'smoother' images/labels in the translated domain, as seen in the experiments; instead, HarmonicGAN is about preserving the overall integrity of the translation itself for the image manifold.

This is the main reason for the large improvement of HarmonicGAN over CycleGAN.To further demonstrate the difference of HarmonicGAN and CRF, we perform an experiment applying the pairwise regularization of CRFs to the CycleGAN framework.

For each pixel of the generated image, we compute the unary term and binary term with its 8 neighbors, and then minimize the objective function of CRF.

The results are shown in TAB6 .

The pairwise regularization of CRF is unable to handle the problem of CycleGAN illustrated in FIG0 .

What's worse, using the pairwise regularization may over-smooth the boundary of generated images, which results in extra artifacts.

In contrast, HarmonicGAN aims at preserving similarity from the overall view of the image manifold, and can thus exploit similarity-consistency of the generated images, rather than over-smooth the boundary.

human to a zebra-like texture.

In contrast, HarmonicGAN does better in background region and achieves an improvement in some regions of of the human (Putin's face), but it still fails on the human body.

We hypothesize this is because the semantic features used by HarmonicGAN have not been trained on humans without a shirt.

to apple, facade to label, label to facade, aerial to map, map to aerial, summer to winter, winter to summer.

<|TLDR|>

@highlight

Smooth regularization over sample graph for unpaired image-to-image translation results in significantly improved consistency