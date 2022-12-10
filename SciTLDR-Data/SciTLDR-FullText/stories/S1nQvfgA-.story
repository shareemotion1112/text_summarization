We propose a new algorithm for training generative adversarial networks to jointly learn latent codes for both identities (e.g. individual humans) and observations (e.g. specific photographs).

In practice, this means that by fixing the identity portion of latent codes, we can generate diverse images of the same subject, and by fixing the observation portion we can traverse the manifold of subjects while maintaining contingent aspects such as lighting and pose.

Our algorithm features a pairwise training scheme in which each sample from the generator consists of two images with a common identity code.

Corresponding samples from the real dataset consist of two distinct photographs of the same subject.

In order to fool the discriminator, the generator must produce images that are both photorealistic, distinct, and appear to depict the same person.

We augment both the DCGAN and BEGAN approaches with Siamese discriminators to accommodate pairwise training.

Experiments with human judges and an off-the-shelf face verification system demonstrate our algorithm’s ability to generate convincing, identity-matched photographs.

In many domains, a suitable generative process might consist of several stages.

To generate a photograph of a product, we might wish to first sample from the space of products, and then from the space of photographs of that product.

Given such disentangled representations in a multistage generative process, an online retailer might diversify its catalog, depicting products in a wider variety of settings.

A retailer could also flip the process, imagining new products in a fixed setting.

Datasets for such domains often contain many labeled identities with fewer observations of each (e.g. a collection of face portraits with thousands of people and ten photos of each).

While we may know the identity of the subject in each photograph, we may not know the contingent aspects of the observation (such as lighting, pose and background).

This kind of data is ubiquitous; given a set of commonalities, we might want to incorporate this structure into our latent representations.

Generative adversarial networks (GANs) learn mappings from latent codes z in some low-dimensional space Z to points in the space of natural data X BID9 .

They achieve this power through an adversarial training scheme pitting a generative model G : Z → X against a discriminative model D : X → [0, 1] in a minimax game.

While GANs are popular, owing to their ability to generate high-fidelity images, they do not, in their original form, explicitly disentangle the latent factors according to known commonalities.

In this paper, we propose Semantically Decomposed GANs (SD-GANs), which encourage a specified portion of the latent space to correspond to a known source of variation.1,2 The technique Figure 1 : Generated samples from SD-BEGAN.

Each of the four rows has the same identity code z I and each of the fourteen columns has the same observation code z O .decomposes the latent code Z into one portion Z I corresponding to identity, and the remaining portion Z O corresponding to the other contingent aspects of observations.

SD-GANs learn through a pairwise training scheme in which each sample from the real dataset consists of two distinct images with a common identity.

Each sample from the generator consists of a pair of images with common z I ∈ Z I but differing z O ∈ Z O .

In order to fool the discriminator, the generator must not only produce diverse and photorealistic images, but also images that depict the same identity when z I is fixed.

For SD-GANs, we modify the discriminator so that it can determine whether a pair of samples constitutes a match.

As a case study, we experiment with a dataset of face photographs, demonstrating that SD-GANs can generate contrasting images of the same subject ( Figure 1 ; interactive web demo in footnote on previous page).

The generator learns that certain properties are free to vary across observations but not identity.

For example, SD-GANs learn that pose, facial expression, hirsuteness, grayscale vs. color, and lighting can all vary across different photographs of the same individual.

On the other hand, the aspects that are more salient for facial verification remain consistent as we vary the observation code z O .

We also train SD-GANs on a dataset of product images, containing multiple photographs of each product from various perspectives FIG2 ).We demonstrate that SD-GANs trained on faces generate stylistically-contrasting, identity-matched image pairs that human annotators and a state-of-the-art face verification algorithm recognize as depicting the same subject.

On measures of identity coherence and image diversity, SD-GANs perform comparably to a recent conditional GAN method (Odena et al., 2017) ; SD-GANs can also imagine new identities, while conditional GANs are limited to generating existing identities from the training data.

Before introducing our algorithm, we briefly review the prerequisite concepts.

GANs leverage the discriminative power of neural networks to learn generative models.

The generative model G ingests latent codes z, sampled from some known prior P Z , and produces G(z), a sample of an implicit distribution P G .

The learning process consists of a minimax game between G, parameterized by θ G , and a discriminative model D, parameterized by θ D .

In the original formulation, the discriminative model tries to maximize log likelihood, yielding DISPLAYFORM0 Training proceeds as follows: For k iterations, sample one minibatch from the real distribution P R and one from the distribution of generated images P G , updating discriminator weights θ D to increase V (G, D) by stochastic gradient ascent.

Then sample a minibatch from P Z , updating θ G to decrease V (G, D) by stochastic gradient descent.

Sample two observation vectors z DISPLAYFORM0 5: DISPLAYFORM1 Generate pair of images G(z 1 ), G(z 2 ), adding them to the minibatch with label 0 (fake).

for m in 1:MinibatchSize do 8:Sample one identity i ∈

I uniformly at random from the real data set.

Sample two images of i without replacement x 1 , x 2 ∼ P R (x|I = i).

Add the pair to the minibatch, assigning label 1 (real).

Update discriminator weights by DISPLAYFORM0 12:Sample another minibatch of identity-matched latent vectors z 1 , z 2 .

Update generator weights by stochastic gradient descent BID17 propose energy-based GANs (EBGANs), in which the discriminator can be viewed as an energy function.

Specifically, they devise a discriminator consisting of an autoencoder: DISPLAYFORM0 DISPLAYFORM1 ).

In the minimax game, the discriminator's weights are updated to minimize the reconstruction error L(x) = ||x − D(x)|| for real data, while maximizing the error L(G(z)) for the generator.

More recently, BID2 extend this work, introducing Boundary Equilibrium GANs (BEGANs), which optimize the Wasserstein distance (reminiscent of Wasserstein GANs BID1 ) between autoencoder loss distributions, yielding the formulation: DISPLAYFORM2 Additionally, they introduce a method for stabilizing training.

Positing that training becomes unstable when the discriminator cannot distinguish between real and generated images, they introduce a new hyperparameter γ, updating the value function on each iteration to maintain a desired ratio between the two reconstruction errors: DISPLAYFORM3 The BEGAN model produces what appear to us, subjectively, to be the sharpest images of faces yet generated by a GAN.

In this work, we adapt both the DCGAN (Radford et al., 2016) and BEGAN algorithms to the SD-GAN training scheme.

Consider the data's identity as a random variable I in a discrete index set I. We seek to learn a latent representation that conveniently decomposes the variation in the real data into two parts: 1) due to I, and 2) due to the other factors of variation in the data, packaged as a random variable O. Ideally, the decomposition of the variation in the data into I and O should correspond exactly to a decomposition of the latent space Z = Z I × Z O .

This would permit convenient interpolation and other operations on the inferred subspaces Z I and Z O .A conventional GAN samples I, O from their joint distribution.

Such a GAN's generative model samples directly from an unstructured prior over the latent space.

It does not disentangle the variation in O and I, for instance by modeling conditional distributions P G (O | I = i), but only models their average with respect to the prior on I.Our SD-GAN method learns such a latent space decomposition, partitioning the coordinates of Z into two parts representing the subspaces, so that any z ∈ Z can be written as the concatenation DISPLAYFORM0 SD-GANs achieve this through a pairwise training scheme in which each sample from the real data consists of x 1 , x 2 ∼ P R (x | I = i), a pair of images with a common identity i ∈ I. Each sample from the generator consists of G(z 1 ), G(z 2 ) ∼ P G (z | Z I = z I ), a pair of images generated from a common identity vector DISPLAYFORM1 We assign identity-matched pairs from P R the label 1 and z I -matched pairs from P G the label 0.

The discriminator can thus learn to reject pairs for either of two primary reasons: 1) not photorealistic or 2) not plausibly depicting the same subject.

See Algorithm 1 for SD-GAN training pseudocode.

With SD-GANs, there is no need to alter the architecture of the generator.

However, the discriminator must now act upon two images, producing a single output.

Moreover, the effects of the two input images x 1 , x 2 on the output score are not independent.

Two images might be otherwise photorealistic but deserve rejection because they clearly depict different identities.

To this end, we devise two novel discriminator architectures to adapt DCGAN and BEGAN respectively.

In both cases, we first separately encode each image using the same convolutional neural network D e ( FIG0 ).

We choose this Siamese setup BID3 BID5 as our problem is symmetrical in the images, and thus it's sensible to share weights between the encoders.

To adapt DCGAN, we stack the feature maps D e (x 1 ) and D e (x 2 ) along the channel axis, applying one additional strided convolution.

This allows the network to further aggregate information from the two images before flattening and fully connecting to a sigmoid output.

For BEGAN, because the discriminator is an autoencoder, our architecture is more complicated.

After encoding each image, we concatenate the representations DISPLAYFORM0 O with linear activation.

In alignment with BEGAN, the SD-BEGAN bottleneck has the same dimensionality as the tuple of latent codes (z I , z 1 O , z 2 O ) that generated the pair of images.

Following the bottleneck, we apply a second FC layer DISPLAYFORM1 , taking the first d I + d O components of its output to be the input to the first decoder and the second d I + d O components to be the input to the second decoder.

The shared intermediate layer gives SD-BEGAN a mechanism to push apart matched and unmatched pairs.

We specify our exact architectures in full detail in Appendix E.

We experimentally validate SD-GANs using two datasets: 1) the MS-Celeb-1M dataset of celebrity face images BID10 and 2) a dataset of shoe images collected from Amazon (McAuley et al., 2015) .

Both datasets contain a large number of identities (people and shoes, respectively) with multiple observations of each.

The "in-the-wild" nature of the celebrity face images offers a richer test bed for our method as both identities and contingent factors are significant sources of variation.

In contrast, Amazon's shoe images tend to vary only with camera perspective for a given product, making this data useful for sanity-checking our approach.

Faces From the aligned face images in the MS-Celeb-1M dataset, we select 12,500 celebrities at random and 8 associated images of each, resizing them to 64x64 pixels.

We split the celebrities into subsets of 10,000 (training), 1,250 (validation) and 1,250 (test).

The dataset has a small number of duplicate images and some label noise (images matched to the wrong celebrity).

We detect and remove duplicates by hashing the images, but we do not rid the data of label noise.

We scale the pixel values to [−1, 1], performing no additional preprocessing or data augmentation.

Shoes Synthesizing novel product images is another promising domain for our method.

In our shoes dataset, product photographs are captured against white backgrounds and primarily differ in orientation and distance.

Accordingly, we expect that SD-GAN training will allocate the observation latent space to capture these aspects.

We choose to study shoes as a prototypical example of a category of product images.

The Amazon dataset contains around 3,000 unique products with the category "Shoe" and multiple product images.

We use the same 80%, 10%, 10% split and again hash the images to ensure that the splits are disjoint.

There are 6.2 photos of each product on average.

We train SD-DCGANs on both of our datasets for 500,000 iterations using batches of 16 identitymatched pairs.

To optimize SD-DCGAN, we use the Adam optimizer (Kingma & Ba, 2015) with hyperparameters α = 2e−4, β 1 = 0.5, β 2 = 0.999 as recommended by Radford et al. (2016) .

We also consider a non-Siamese discriminator that simply stacks the channels of the pair of real or fake images before encoding (SD-DCGAN-SC).As in (Radford et al., 2016) , we sample latent vectors z ∼ Uniform([−1, 1] 100 ).

For SD-GANs, we partition the latent codes according to 25, 50, 75] .

Our algorithm can be trivially applied with k-wise training (vs. pairwise).

To explore the effects of using k > 2, we also experiment with an SD-DCGAN where we sample k = 4 instances each from P G (z | Z I = z I ) for some z I ∈ Z I and from P R (x | I = i) for some i ∈ I. For all experiments, unless otherwise stated, we use d I = 50 and k = 2.

DISPLAYFORM0 We also train an SD-BEGAN on both of our datasets.

The increased complexity of the SD-BEGAN model significantly increases training time, limiting our ability to perform more-exhaustive hyperparameter validation (as we do for SD-DCGAN).

We use the Adam optimizer with the default hyperparameters from (Kingma & Ba, 2015) for our SD-BEGAN experiments.

While results from our SD-DCGAN k = 4 model are compelling, an experiment with a k = 4 variant of SD-BEGAN resulted in early mode collapse (Appendix F); hence, we excluded SD-BEGAN k = 4 from our evaluation.

We also compare to a DCGAN architecture trained using the auxiliary classifier GAN (AC-GAN) method (Odena et al., 2017) .

AC-GAN differs from SD-GAN in two key ways: 1) random identity codes z I are replaced by a one-hot embedding over all the identities in the training set (matrix of size 10000x50); 2) the AC-GAN method encourages that generated photos depict the proper identity by tasking its discriminator with predicting the identity of the generated or real image.

Unlike SD-GANs, the AC-DCGAN model cannot imagine new identities; when generating from AC-DCGAN (for our quantitative comparisons to SD-GANs), we must sample a random identity from those existing in the training data.

The evaluation of generative models is a fraught topic.

Quantitative measures of sample quality can be poorly correlated with each other (Theis et al., 2016) .

Accordingly, we design an evaluation to match conceivable uses of our algorithm.

Because we hope to produce diverse samples that humans deem to depict the same person, we evaluate the identity coherence of SD-GANs and baselines using both a pretrained face verification model and crowd-sourced human judgments obtained through Amazon's Mechanical Turk platform.

Recent advancements in face verification using deep convolutional neural networks (Schroff et al., 2015; Parkhi et al., 2015; Wen et al., 2016) have yielded accuracy rivaling humans.

For our evaluation, we procure FaceNet, a publicly-available face verifier based on the Inception-ResNet architecture (Szegedy et al., 2017 ).

The FaceNet model was pretrained on the CASIA-WebFace dataset BID13 and achieves 98.6% accuracy on the LFW benchmark BID12 .

FaceNet ingests normalized, 160x160 color images and produces an embedding f (x) ∈ R 128 .

The training objective for FaceNet is to learn embeddings that minimize the L 2 distance between matched pairs of faces and maximize the distance for mismatched pairs.

Accordingly, the embedding space yields a function for measuring the similarity between two faces x 1 and x 2 : D(x 1 , x 2 ) = ||f (x 1 ) − f (x 2 )|| 2 2 .

Given two images, x 1 and x 2 , we label them as a match if D(x 1 , x 2 ) ≤ τ v where τ v is the accuracy-maximizing threshold on a class-balanced set of pairs from MS-Celeb-1M validation data.

We use the same threshold for evaluating both real and synthetic data with FaceNet.

We compare the performance of FaceNet on pairs of images from the MS-Celeb-1M test set against generated samples from our trained SD-GAN models and AC-DCGAN baseline.

To match FaceNet's training data, we preprocess all images by resizing from 64x64 to 160x160, normalizing each image individually.

We prepare 10,000 pairs from MS-Celeb-1M, half identity-matched and half unmatched.

From each generative model, we generate 5,000 pairs each with z We also want to ensure that identity-matched images produced by the generative models are diverse.

To this end, we propose an intra-identity sample diversity (ID-Div) metric.

The multi-scale structural similarity (MS-SSIM) (Wang et al., 2004 ) metric reports the similarity of two images on a scale from 0 (no resemblance) to 1 (identical images).

We report 1 minus the mean MS-SSIM for all pairs of identity-matched images as ID-Div.

To measure the overall sample diversity (All-Div), we also compute 1 minus the mean similarity of 10k pairs with random identities.

In TAB1 , we report the area under the receiver operating characteristic curve (AUC), accuracy, and false accept rate (FAR) of FaceNet (at threshold τ v ) on the real and generated data.

We also report our proposed diversity statistics.

FaceNet verifies pairs from the real data with 87% accuracy compared to 86% on pairs from our SD-BEGAN model.

Though this is comparable to the accuracy achieved on pairs from the AC-DCGAN baseline, our model produces samples that are more diverse in pixel space (as measured by ID-Div and All-Div).

FaceNet has a higher but comparable FAR for pairs from SD-GANs than those from AC-DCGAN; this indicates that SD-GANs may produce images that are less semantically diverse on average than AC-DCGAN.We also report the combined memory footprint of G and D for all methods in TAB1 .

For conditional GAN approaches, the number of parameters grows linearly with the number of identities in the training data.

Especially in the case of the AC-GAN, where the discriminator computes a softmax over all identities, linear scaling may be prohibitive.

While our 10k-identity subset of MS-Celeb-1M requires a 131MB AC-DCGAN model, an AC-DCGAN for all 1M identities would be over 8GB, with more than 97% of the parameters devoted to the weights in the discriminator's softmax layer.

In contrast, the complexity of SD-GAN is constant in the number of identities.

In addition to validating that identity-matched SD-GAN samples are verified by FaceNet, we also demonstrate that humans are similarly convinced through experiments using Mechanical Turk.

For these experiments, we use balanced subsets of 1,000 pairs from MS-Celeb-1M and the most promising generative methods from our FaceNet evaluation.

We ask human annotators to determine if each pair depicts the "same person" or "different people".

Annotators are presented with batches of ten pairs at a time.

Each pair is presented to three distinct annotators and predictions are determined by majority vote.

Additionally, to provide a benchmark for assessing the quality of the Mechanical Turk ensembles, we (the authors) manually judged 200 pairs from MS-Celeb-1M.

Results are in TAB1 For all datasets, human annotators on Mechanical Turk answered "same person" less frequently than FaceNet when the latter uses the accuracy-maximizing threshold τ v .

Even on real data, balanced so that 50% of pairs are identity-matched, annotators report "same person" only 28% of the time (compared to the 41% of FaceNet).

While annotators achieve higher accuracy on pairs from AC-DCGAN than pairs from SD-BEGAN, they also answer "same person" 16% more often for AC-DCGAN pairs than real data.

In contrast, annotators answer "same person" at the same rate for SD-BEGAN pairs as real data.

This may be attributable to the lower sample diversity produced by AC-DCGAN.

Samples from SD-DCGAN and SD-BEGAN are shown in FIG1

Style transfer and novel view synthesis are active research areas.

Early attempts to disentangle style and content manifolds used factored tensor representations (Tenenbaum & Freeman, 1997; Vasilescu & Terzopoulos, 2002; BID7 BID19 , applying their results to face image synthesis.

More recent work focuses on learning hierarchical feature representations using deep convolutional neural networks to separate identity and pose manifolds for faces BID19 Reed et al., 2014; BID20 Yang et al., 2015; Kulkarni et al., 2015; Oord et al., 2016; BID4 and products BID6 .

BID8 use features of a convolutional network, pretrained for image recognition, as a means for discovering content and style vectors.

Since their introduction BID9 , GANs have been used to generate increasingly highquality images (Radford et al., 2016; BID17 BID2 .

Conditional GANs (cGANs), introduced by Mirza & Osindero (2014) , extend GANs to generate class-conditional data.

Odena et al. (2017) propose auxiliary classifier GANs, combining cGANs with a semi-supervised discriminator BID6 .

Recently, cGANs have been used to ingest text (Reed et al., 2016) and full-resolution images BID18 Liu et al., 2017; BID18 as conditioning information, addressing a variety of image-to-image translation and style transfer tasks.

BID4 devise an information-theoretic extension to GANs in which they maximize the mutual information between a subset of latent variables and the generated data.

Their unsupervised method BID16 all address synthesis of different body/facial poses conditioned on an input image (representing identity) and a fixed number of pose labels.

BID0 propose conditional GANs for synthesizing artificially-aged faces conditioned on both a face image and an age vector.

These approaches all require explicit conditioning on the relevant factor (such as rotation, lighting and age) in addition to an identity image.

In contrast, SD-GANs can model these contingent factors implicitly (without supervision).Mathieu et al. FORMULA0 combine GANs with a traditional reconstruction loss to disentangle identity.

While their approach trains with an encoder-decoder generator, they enforce a variational bound on the encoder embedding, enabling them to sample from the decoder without an input image.

Experiments with their method only address small (28x28) grayscale face images, and their training procedure is complex to reproduce.

In contrast, our work offers a simpler approach and can synthesize higher-resolution, color photographs.

One might think of our work as offering the generative view of the Siamese networks often favored for learning similarity metrics BID3 BID5 .

Such approaches are used for discriminative tasks like face or signature verification that share the many classes with few examples structure that we study here.

In our work, we adopt a Siamese architecture in order to enable the discriminator to differentiate between matched and unmatched pairs.

Recent work by Liu & Tuzel (2016) propose a GAN architecture with weight sharing across multiple generators and discriminators, but with a different problem formulation and objective from ours.

Our evaluation demonstrates that SD-GANs can disentangle those factors of variation corresponding to identity from the rest.

Moreover, with SD-GANs we can sample never-before-seen identities, a benefit not shared by conditional GANs.

In FIG1 , we demonstrate that by varying the observation vector z O , SD-GANs can change the color of clothing, add or remove sunnies, or change facial pose.

They can also perturb the lighting, color saturation, and contrast of an image, all while keeping the apparent identity fixed.

We note, subjectively, that samples from SD-DCGAN tend to appear less photorealistic than those from SD-BEGAN.

Given a generator trained with SD-GAN, we can independently interpolate along the identity and observation manifolds ( FIG4 ).On the shoe dataset, we find that the SD-DCGAN model produces convincing results.

As desired, manipulating z I while keeping z O fixed yields distinct shoes in consistent poses FIG2 .

The identity code z I appears to capture the broad categories of shoes (sneakers, flip-flops, boots, etc.) .

Surprisingly, neither original BEGAN nor SD-BEGAN can produce diverse shoe images (Appendix G).In this paper, we presented SD-GANs, a new algorithm capable of disentangling factors of variation according to known commonalities.

We see several promising directions for future work.

One logical extension is to disentangle latent factors corresponding to more than one known commonality.

We also plan to apply our approach in other domains such as identity-conditioned speech synthesis.

We estimate latent vectors for unseen images and demonstrate that the disentangled representations of SD-GANs can be used to depict the estimated identity with different contingent factors.

In order to find a latent vectorẑ such that G(ẑ) (pretrained G) is similar to an unseen image x, we can minimize the distance between x and G(ẑ): minẑ ||G(ẑ) − x|| In FIG5 , we depict estimation and linear interpolation across both subspaces for two pairs of images using SD-BEGAN.

We also display the corresponding source images being estimated.

For both pairs,ẑ I (identity) is consistent in each row andẑ O (observation) is consistent in each column.

In Section 3.1, we describe an AC-GAN (Odena et al., 2017) baseline which uses an embedding matrix over real identities as latent identity codes (G : i, z O →x).

In place of random identity vectors, we tried combining this identity representation with pairwise discrimination (in the style of SD-GAN).

In this experiment, the discriminator receives either either two real images with the same identity (x In Appendix C, we detail a modification of the DR-GAN (Tran et al., 2017) method which uses an encoding network G e to transform images to identity representations (G d : G e (x), z O →x).

We also tried combining this encoder-decoder approach with pairwise discrimination.

The discriminator receives either two real images with the same identity (x DISPLAYFORM0 We show results in FIG9 .

While these experiments are exploratory and not part of our principle investigation, we find the results to be qualitatively promising.

We are not the first to propose pairwise discrimination with pairs of (real, real) or (real, fake) images in GANs (Pathak et al., 2016; BID18 .

Tran et al. (2017) propose Disentangled Representation learning-GAN (DR-GAN), an approach to face frontalization with similar setup to our SD-GAN algorithm.

The (single-image) DR-GAN generator G (composition of G e and G d ) accepts an input image x, a pose code c, and a noise vector z. The DR-GAN discriminator receives either x orx = G d (G e (x), c, z).

In the style of BID6 , the discriminator is tasked with determining not only if the image is real or fake, but also classifying the pose c, suggesting a disentangled representation to the generator.

Through their experiments, they demonstrate that DR-GAN can explicitly disentangle pose and illumination (c) from the rest of the latent space (G e (x); z).

In addition to our AC-DCGAN baseline (Odena et al., 2017) , we tried modifying DR-GAN to only disentangle identity (rather than both identity and pose in the original paper).

We used the DCGAN (Radford et al., 2016) discriminator architecture (Table 4) as G e , linearly projecting the final convolutional layer to G e (x) ∈ R 50 (in alignment with our SD-GAN experiments).

We altered the discriminator to predict the identity of x orx, rather than pose information (which is unknown in our experimental setup).

With these modifications, G e (x) is analogous to z I in the SD-GAN generator, and z is analogous to z O .

Furthermore, this setup is identical to the AC-DCGAN baseline except that the embedding matrix is replaced by an encoding network G e .

Unfortunately, we found that the generator quickly learned to produce a single output imagex for each input x regardless of observation code z FIG10 ).

Accordingly, we excluded this experiment from our evaluation (Section 3.2).

Figure 10 : AC-DCGAN generation with random identity vectors that sum to one.

Each row shares an identity vector and each column shares an observation vector.

As stated in Section 3.1, AC-GANs Odena et al. (2017) provide no obvious way to imagine new identities.

For our evaluation (Section 3.2), the AC-GAN generator receives identity input z I ∈ [0, 1] 10000 : a one-hot over all identities.

One possible approach to imagining new identities would be to query a trained AC-GAN generator with a random vector z I such that 10000 i=1 z I [i] = 1.

We found that this strategy produced little identity variety (Figure 10 ) compared to the normal one-hot strategy ( FIG11 ) and excluded it from our evaluation.

We list here the full architectural details for our SD-DCGAN and SD-BEGAN models.

In these descriptions, k is the number of images that the generator produces and discriminator observes per identity (usually 2 for pairwise training), and d I is the number of dimensions in the latent space Z I (identity).

In our experiments, dimensionality of Z O is always 100 − d I .

As a concrete example, the bottleneck layer of the SD-BEGAN discriminator autoencoder ("fc2" in Table 6 ) with k = 2, d I = 50 has output dimensionality 150.We emphasize that generators are parameterized by k in the tables only for clarity and symmetry with the discriminators.

Implementations need not modify the generator; instead, k can be collapsed into the batch size.

For the stacked-channels versions of these discriminators, we simply change the number of input image channels from 3 to 3k and set k = 1 wherever k appears in the table.

TAB1 for qualitative comparison.

In each matrix, z I is the same across all images in a row and z O is the same across all images in a column.

We draw identity and observation vectors randomly for these samples.

Figure 18: Generated samples from SD-DCGAN trained with the Wasserstein GAN loss BID1 .

This model was optimized using RMS-prop (Hinton et al.) with α = 5e−5.

In our evaluation (Section 3.2), FaceNet had an AUC of .770 and an accuracy of 68.5% (at τ v ) on data generated by this model.

We excluded it from Table 1 for brevity.

@highlight

SD-GANs disentangle latent codes according to known commonalities in a dataset (e.g. photographs depicting the same person).

@highlight

This paper investigates the problem of controlled image generation and proposes an algorithm that produces a pair of images with the same identity.

@highlight

This paper proposes, SD-GAN, a method of training GANs to disentangle the identity and non-identity information in the latent vector input Z.