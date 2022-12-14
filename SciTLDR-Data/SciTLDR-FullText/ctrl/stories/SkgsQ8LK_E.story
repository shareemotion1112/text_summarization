It is challenging to disentangle an object into two orthogonal spaces of content and style since each can influence the visual observation in a different and unpredictable way.

It is rare for one to have access to a large number of data to help separate the influences.

In this paper, we present a novel framework to learn this disentangled representation in a completely unsupervised manner.

We address this problem in a two-branch Autoencoder framework.

For the structural content branch, we project the latent factor into a soft structured point tensor and constrain it with losses derived from prior knowledge.

This encourages the branch to distill geometry information.

Another branch learns the complementary style information.

The two branches form an effective framework that can disentangle object's content-style representation without any human annotation.

We evaluate our approach on four image datasets, on which we demonstrate the superior disentanglement and visual analogy quality both in synthesized and real-world data.

We are able to generate photo-realistic images with 256x256 resolution that are clearly disentangled in content and style.

Content and style are the two most inherent attributes that characterize an object visually.

Computer vision researchers have devoted decades of efforts to understand object shape and extract features that are invariant to geometry change BID11 BID33 BID36 BID26 .

Learning such disentangled deep representation for visual objects is an important topic in deep learning.

The main objective of our work is to disentangle object's style and content in an unsupervised manner.

Achieving this goal is non-trivial due to three reasons: 1) Without supervision, we can hardly guarantee the separation of different representations in the latent space.

2) Although some methods like InfoGAN are capable of learning several groups of independent attributes from objects, attributes from these unsupervised frameworks are uninterpretable since we cannot pinpoint which portion of the disentangled representation is related to the content and which to the style.

3) Learning structural content from a set of natural real-world images is difficult.

To overcome the aforementioned challenges, we propose a novel two-branch Autoencoder framework, of which the structural content branch aims to discover semantically meaningful structural points (i.e., y in Fig 2) to represent the object geometry, while the other style branch learns the complementary style representation.

The settings of these two branches are asymmetric.

For the structural content branch, we add a layer-wise softmax operator to the last layer.

We could regard this as a projection of a latent content to a soft structured point tensor space.

Specifically designed prior losses are used to constrain the structured point tensors so that the discovered points have high repeatability across images yet distributed uniformly to cover different parts of the object.

To encourage the framework to learn a disentangled yet complementary representation of both content and style, we further introduce a Kullback-Leibler (KL) divergence loss and skip-connections design to the framework.

In FIG0 , we show the latent space walking results on cat face dataset, which demonstrates a reasonable coverage of the manifold and an effective disentanglement of the content and style space of our approach.

Extensive experiments show the effectiveness of the proposed method in disentangling the content and style of natural images.

We also conduct qualitative and quantitative experiments on MNISTColor, 3D synthesized data and several real-world datasets which demonstrate the superior performance of our method to state-of-the-art algorithms.

The architecture of our model is shown in Fig. 2 .

In the absence of annotation on the structure of object content, we rely on prior knowledge on how object landmarks should distribute to constrain the learning and disentanglement of structural content information.

Our experiments show that this is possible given appropriate prior losses and learning architecture.

We first formulate our loss function with special consideration on prior.

Specifically, we follow the VAE framework and assume 1) the two latent variables z and y, which represent the style and content, are generated from some prior distributions.

2) x follows the conditional distribution p(x|y, z).

We start with a Bayesian formulation and maximize the log-likelihood over all observed samples x ??? X. log p(x) = log p(y) + log p(x|y) ??? log p(y|x) ??? log p(y) + log p(x, z|y) dz ??? log p(y) + E q log p(x, z|y) q(z|x, y)= log p(y) + E q log p(x|y, z)p(z|y) q(z|x, y) .Figure 2:

Architecture: Our framework follows an Autoencoder framework.

It contains two branches: 1) the structural content branch forces the representation into a Gaussian spatial probability distribution with an hourglass network e??.

2) the style branch E ?? learns a complementary style representation to the content.

Equation 1 learns a deterministic mapping e(??; ??) from x to y, which we assume y is following a Gaussian distribution over N (e(x; ??), ??).

Term ??? log p(y|x) is non-negative.

In the second line of the equation, we start to consider the factor z. Similar to VAE, we address the issue of intractable integral by introducing an approximate posterior q(y, z|x; ??) to estimate the integral using evidence lower bound (ELBO).

By splitting the p(x|y, z) from the second term of the last expression, we obtain our final loss as, DISPLAYFORM0 The first term is about the prior on y. The second term describes the conditional distribution of x given all representation.

Ideally, if the decoder can perfectly reconstruct the x, the second term would be a delta function over x. The third term represents the Kullback-Leibler divergence between approximate.

In the rest of this paper we name these three terms respectively as prior loss L prior , reconstruction loss L recon and KL loss L KL .

We firstly formulate the content representation y as a soft latent structured point tensor.

Then, a re-projecting operator is applied here to force y to lie on a Gaussian spatial probability distribution space.

Following the notations from BID22 , we denote the direct outputs of the hourglass network e ?? as landmark heatmaps h, and each channel of which represents the spatial location of a structural point.

Instead of using max activations across each heatmap as landmark coordinates, we weighted average all activations across each heatmap.

We then re-project landmark coordinates to spatial features with the same size as heatmaps by a fixed Gaussian-like function centered at predicted coordinates with a fixed standard deviation.

As a result, we obtain a new tensor y with the structure prior on content representation.

Nevertheless, we find that training the structural content branch with general random initialization tend to locate all structural points around the mean location at the center of the image.

This could lead to a local minimum from which optimizer might not escape.

As such, we introduce a Separation Loss to encourage each heatmap to sufficiently cover the object of interest.

This is achieved by the first term in Eq. 3, where we encourage each pair of i th and j th heatmaps to share different activations.

?? can be regarded as a normalization factor here.

Another prior constraint is that we wish the structural point to behave like landmarks to encode geometry structure information.

To achieve this goal, we add a Concentration Loss to encourage the variance of activations h to be small so that it could concentrate at a single location, which corresponds to the second term in Eq. 3.

It is noteworthy that some recent works have considered the prior of latent factor.

BID4 proposed a Joint-??-VAE by adding different prior distribution over several latent factors to disentangle continuous and discrete factors from data.

Our work differs in that we investigate a different prior to disentangle visual content and style.

DISPLAYFORM0

For the second term we optimize the reconstruction loss of whole model, which will be denoted as generator G in the following context.

We assume that the decoder D ?? is able to reconstruct original input x from latent representation y and z, which isx = G(y, z).

Consequently, we can design the reconstruction loss as DISPLAYFORM0 However, minimizing L 1 / L 2 loss at pixel-level only does not model the perceptual quality well and makes the prediction look blurry and implausible.

This phenomenon has been well-observed in the literature of super-resolution BID1 BID27 .

We consequently define the reconstruction loss as DISPLAYFORM1 , where ?? l is the feature obtained from l-th layer of a VGG-19 model BID32 pre-trained on ImageNet.

It is also possible to add adversarial loss to further improve the perceptual reconstruction quality.

Since the goal of this work is disentanglement rather than reconstruction, we only adopt the L recon described above.

We model q(z|x, y) as a parametric Gaussian distribution which can be estimated by the encoder network E ?? .

Therefore, the style code z can be sampled from q(z|x, y).

Meanwhile, the prior p(z|y) can be estimated by the encoder network E ?? .

By using the reparametrization trick BID16 , these networks can be trained end-to-end.

We only estimate mean value here for the stability of learning.

By modeling the two distributions as Gaussian with identity covariances, the KL Loss is simply equal to the Euclidean distance between their means.

Thus, z is regularized by minimizing the KL divergence between q(z|x, y) and p(z|y).Notice that with only prior and reconstruction loss.

The framework only makes sure z is from x and the Decoder D ?? will recover as much information of x as possible.

There is no guarantee that z will learn a complementary of y. Towards this end, as shown in Fig. 2 , we design the network as fusing the encoded content representation by E ?? with the inferred style code z. Then, the fused representation is decoded together by D ?? .

Meanwhile, skip-connections between E ?? and D ?? are used to pass multi-level content information to the decoder.

Therefore, enough content information can be obtained from prior and any information about content encoded in z incurs a penalty of the likelihood p(x|y, z) with no new information (i.e. style information) is captured.

This design of the network and the KL Loss result in a constraint to guide z to encode more information about the style which is complementary to content.

Each of the input images x is cropped and resized to 256 ?? 256 resolution.

A one-stack hourglass network BID22 ) is used as a geometry extractor e ?? to project input image to the heatmap y ??? R 256??256??30 , in which each channel represents one point-centered 2D-Gaussian map (with ?? = 4).

y is drawn in a single-channel map for visualization in Fig. 2 .

Same network (with stride-2 convolution for downsampling) is used for both E ?? and E ?? to obtain style representation z and the embedded content representation as two 128-dimension vectors.

A symmetrical deconvolution network with skip connection is used as the decoder D ?? to get the reconstructed result x. All of the networks are jointly trained from scratch end-to-end.

We detail the architectures and hyperparameters used for our experiments in appendix A.

Unsupervised Feature Disentangle: Several pioneer works focus on unsupervised disentangled representation learning.

Following the propose of GANs BID7 , purpose InfoGAN to learn a mapping from a group of latent variables to the data in an unsupervised manner.

Many similar methods were purposed to achieve a more stable result BID8 BID18 .

However, these works suffer to interpret, and the meaning of each learned factor is uncontrollable.

There are some following works focusing on dividing latent factors into different sets to enforce better disentangling.

BID21 assign one code to the specified factors of variation associated with the labels, and left the remaining as unspecified variability.

Similar to BID21 , BID9 then propose to obtain disentanglement of feature chunks by leveraging Autoencoders, with the supervision of some same/different class pairs.

BID30 rely on a 3DMM model together with GANs to disentangle representation of face properties.

BID4 divides latent variable into discrete and continuous one, and distribute them in different prior distribution.

In our work, we give one branch of representation are more complicated prior, to force it to represent only the shape information for the object.

Supervised Pose Synthesis: Recently the booming of GANs research improves the capacity of pose-guided image generation.

BID20 firstly try to synthesize pose images with U-Netlike networks.

Several works soon follow this appealing topic and obtain better results on human pose or face generation.

BID5 applied a conditional U-Net for shape-guided image generation.

BID17 BID24 incorporat geometric information into the image generation process.

Nevertheless, existing works rely on massive annotated datas, they need a strong pre-trained landmark estimator, or treat landmarks of a object as input.

Unsupervised Structure Learning: Unsupervised learning structure from objects is one of the essential topics in computer vision.

The rudimentary works focus on keypoints detection and learning a strong descriptor to match BID33 BID26 .

Recently, BID12 and BID36 , show the possibility of end-to-end learning of structure in Autoencoder formulations.

BID31 follow the deformable template paradigm to represent shape as a deformation between a canonical coordinate system and an observed image.

Our work is diffenent from the aforementioned methods mainly in the explicitly learned complementary style representations and the formulation in a two-branch VAE framework which leads to a clear disentanglement.

Datasets: We evaluate our method on four datasets that cover both synthesized and real world data: 1).

MNIST-Color: we extend MNIST by either colorizing the digit (MNIST-CD) or the background (MNIST-CB) with a randomly chosen color following BID6 .

We use the standard split of training (50k) and testing (10k) set.

2).

3D Chair: BID0 offers rendered images of 1393 CAD chair models.

We take 1343 chairs for training and the left 50 chairs for testing.

For each chair, 12 rendered images with different views are selected randomly.

3).

Cat & Dog Face, we collect 6k (5k for training and 1k for testing) images of cat and dog from YFCC100M BID14 and Stanford Dog BID15 datasets respectively.

All images are center cropped around the face and scaled to the same size.

4).

CelebA: it supplies plenty of celebrity faces with different attributes.

The training and testing sizes are 160K and 20K respectively.

Evaluation Metric: We perform both qualitative and quantitative evaluations to study the disentanglement ability and generation quality of the proposed framework: 1).

Qualitative: we provide four kinds of qualitative results to show as many usages of the disentangled space as possible, i.e. conditional sampling, interpolation, retrieval, and visual analogy.

2).

Quantitative: we apply several metrics that are widely employed in image generation (a) Content consistency: content similarity metric BID19 and mean-error of landmarks BID2 .

(b) Style consistency: style similarity metric BID13 ) (c).

Disentangled ability: retrieval recall@K BID29 .

(d).

Reconstruction and generation quality: SSIM BID34 and Inception Score BID28 .

Diverse Generation.

We first demonstrate the diversity of conditional generation results on MNISTColor with the successfully disentangled content and style in FIG1 .

It can be observed that, given an image as a content condition, same digit information with different style can be generated by sampling the style condition images randomly.

While given an image as style condition, different digits with the same color can be generated by sampling different structural conditional images.

Note that the model has no prior knowledge of the digit in the image as no label is provided, it effectively learns the disentanglement spontaneously.

Interpolation.

FIG1 , the linear interpolation results show reasonable coverage of the manifold.

From left to right, the color is changed smoothly from blue to red with interpolated style latent space while maintaining the digit information.

Analogously, the color stays stable while one digit transforms into the other smoothly from top to down.

Retrieval.

To demonstrate the disentangled ability of the representation learned by the model, we perform nearest neighbor retrieval experiments following Mathieu et al. FORMULA0 on MNIST-Color.

With content and style representation used, both semantic and visual retrieval can be performed respectively.

The Qualitative results are shown in FIG2 .

Quantitatively, We use a commonly used retrieval metric Recall@K as in BID29 BID23 , where for a particular query digit, Recall@K is 1 if the corresponding digit is within the top-K retrieved results and 0 otherwise.

We report the most challenging Recall@1 by averaging over all queries on the test set in TAB2 (a).

It can be observed that the content representation shows the best performance and clearly outperforms image pixel and style representation.

In addition to the disentangled ability, this result also shows that the content representation learned by our model is useful for visual retrieval.

Visual Analogy.

The task of visual analogy is that the particular attribute of a given reference image can be transformed to a query one BID25 .

We show the visual analogy results on MNIST-Color and 3D Chair in FIG3 .

Note that even for the detail component (e.g. wheel and leg of 3D chair) the content can be maintained successfully, which is a rather challenging task in previous unsupervised works BID8 .Comparison.

We compare perceptual quality with the three most related unsupervised representation learning methods in FIG4 e., VAE (Kingma & Welling, 2014), ??-VAE BID8 and InfoGAN .

It can be observed that from left to right, all of the three methods can rotate the chairs successfully, which demonstrates the automatical learning of disentangling the factor of azimuth on 3D Chair dataset.

However, it can be perceived that the geometry shape can be maintained much better in our approach than all the other methods, owing to the informative prior supplied by our structural content branch.

BID8 and InfoGAN .

We demonstrate the disentanglement ability of our method of the azimuth factor for 3D Chair dataset and much better geometry maintaining ability from left to right than state-of-the-arts.

We have so far only discussed results on the synthesized benchmarks.

In this section, we will demonstrate the scalable performance of our model on several real-world datasets, i.e., Cat, Dog Face and CelebA. To the best knowledge of ours, there is no literature of unsupervised disentanglement before can successfully extend to photo-realistic generation with 256 ?? 256 resolution.

Owing to the structural prior which accurately capture the structural information of images, our model can transform style information while faithfully maintain the geometry shapes.

Qualitative evaluation is performed by visually examining the perceptual quality of the generated images.

In Fig. 8 , the swapping results along with the learned structural heatmaps y are illustrated on Cat dataset.

In can be seen that the geometry information, i.e., expression, head-pose, facial action, and style information i.e., hair texture, can be swapped between each other arbitrarily.

The learned structural heatmaps can be shown as a map with several 2D Gaussian points, which successfully encode the geometry cues of a image by the location of its points and supply an effective prior for identity, head pose and expression) of query image can be faithfully maintained while the style (e.g. the color of hair, beard and illumination) of reference image can be precisely transformed.

As concrete examples, the output of the dog in the third column is still tongue-sticking while the hair color is changed, and in the last column of CelebA, even the fine-grain eye make-up is successfully transformed to the query image surprisingly.

FIG5 .

We observe that our model can successfully generalize to various real-world images with large variations, such as mouth-opening, eye-closing, tongue-sticking and exclusive style.

For quantitative measurement, there is no standard evaluation metric of the quality of the visual analogy results for real-world datasets since ground-truth targets are absent.

We propose to evaluate the content and style consistency of the analogy predictions respectively instead.

We use content similarity metric for the evaluation of content consistency between a condition input x s and its guided generated images (e.g., for each column of images in Fig. 8 ).

We use style similarity metric to evaluate the style consistency between a condition input x a and its guided generated images (e.g., each row of images in Fig. 8 ).

These two metrics are used widely in image generation applications as an objective for training to maintain content and texture information BID19 BID13 .Since content similarity metric is less sensitive to the small variation of images, we further propose to use the mean-error of landmarks detected by a landmark detection network, which is pre-trained on manually annotated data, to evaluate the content consistency.

Since the public cat facial landmark annotations are too sparse to evaluate the content consistency (e.g. 9-points BID35 ), we manually annotated 10k cat face with 18-points to train a landmark detection network for evaluation purpose.

As for the evaluation of celebA, a state-of-the-art model BID2 with 68-landmarks is used.

The results on the testing set of the two real-world datasets are reported in TAB1 .

For each test image, 1k other images in the testing set are all used as the reference of content or style for generation, in which mean value is calculated.

In the baseline "Ours + random" setting, for one test image, the mean value is calculated by sampling randomly among the generated images guided by each image.

Results of two state-of-the-art unsupervised structure learning methods BID12 BID36 are also reported for comparison.

Content consistency is evaluated by content similarity metric and landmark detection error, while style consistency is evaluated by style similarity metric as mentioned above.

Structural Similarities (SSIM) and Inception Scores (IS) are utilized to evaluate the reconstruction quality and the analogy quality.

Superior performance of both content/style consistency and generation quality of our method can be obviously observed in TAB1 .

To study the effects of VGG loss (Sec. 2.2) and KL loss (Sec. 2.3) of our method on generated images.

We evaluate the aforementioned metrics of our method on Cat dataset.

As reported in

We propose a novel model based on Autoencoder framework to disentangle object's representation by content and style.

Our framework is able to mine structural content from a kind of objects and learn content-invariant style representation simultaneously, without any annotation.

Our work may also reveal several potential topics for future research: 1) Instead of relying on supervision, using strong prior to restrict the latent variables seems to be a potential and effective tool for disentangling.

2) In this work we only experiment on near-rigid objects like chairs and faces, learning on deformable objects is still an opening problem.3) The content-invariant style representation may have some potentials on recognition tasks.

<|TLDR|>

@highlight

We present a novel framework to learn the disentangled representation of content and style in a completely unsupervised manner. 

@highlight

Propose model based on autoencoder framework to disentangle an object's representation, results show that model can produce representations capturing content and style.