Unsupervised image-to-image translation aims to learn a mapping between several visual domains by using unpaired training pairs.

Recent studies have shown remarkable success in image-to-image translation for multiple domains but they suffer from two main limitations: they are either built from several two-domain mappings that are required to be learned independently and/or they generate low-diversity results, a phenomenon known as model collapse.

To overcome these limitations, we propose a method named GMM-UNIT based on a content-attribute disentangled representation, where the attribute space is fitted with a GMM.

Each GMM component represents a domain, and this simple assumption has two prominent advantages.

First, the dimension of the attribute space does not grow linearly with the number of domains, as it is the case in the literature.

Second, the continuous domain encoding allows for interpolation between domains and for extrapolation to unseen domains.

Additionally, we show how GMM-UNIT can be constrained down to different methods in the literature, meaning that GMM-UNIT is a unifying framework for unsupervised image-to-image translation.

Translating images from one domain into another is a challenging task that has significant influence on many real-world applications where data are expensive, or impossible to obtain and to annotate.

Image-to-Image translation models have indeed been used to increase the resolution of images (Dong et al., 2014) , fill missing parts (Pathak et al., 2016) , transfer styles (Gatys et al., 2016) , synthesize new images from labels (Liu et al., 2017) , and help domain adaptation (Bousmalis et al., 2017; Murez et al., 2018) .

In many of these scenarios, it is desirable to have a model mapping one image to multiple domains, while providing visual diversity (i.e. a day scene ↔ night scene in different seasons).

However, the existing models can either map an image to multiple stochastic results in a single domain, or map in the same model multiple domains in a deterministic fashion.

In other words, most of the methods in the literature are either multi-domain or multi-modal.

Several reasons have hampered a stochastic translation of images to multiple domains.

On the one hand, most of the Generative Adversarial Network (GAN) models assume a deterministic mapping (Choi et al., 2018; Pumarola et al., 2018; Zhu et al., 2017a) , thus failing at modelling the correct distribution of the data .

On the other hand, approaches based on Variational Auto-Encoders (VAEs) usually assume a shared and common zero-mean unit-variance normally distributed space Zhu et al., 2017b) , limiting to two-domain translations.

In this paper, we propose a novel image-to-image translation model that disentangles the visual content from the domain attributes.

The attribute latent space is assumed to follow a Gaussian mixture model (GMM), thus naming the method: GMM-UNIT (see Figure 1 ).

This simple assumption allows four key properties: mode-diversity thanks to the stochastic nature of the probabilistic latent model, multi-domain translation since the domains are represented as clusters in the same attribute spaces, scalability because the domain-attribute duality allows modeling a very large number of domains without increasing the dimensionality of the attribute space, and few/zero-shot generation since the continuity of the attribute representation allows interpolating between domains and extrapolating to unseen domains with very few or almost no observed data from these domains.

The code and models will be made publicly available. : GMM-UNIT working principle.

The content is extracted from the input image (left, purple box), while the attribute (turquoise box) can be either sampled (top images) or extracted from a reference image (bottom images).

Either way, the generator (blue box) is trained to output realistic images belonging to the domain encoded in the attribute vector.

This is possible thanks to the disentangled attribute-content latent representation of GMM-UNIT and the generalisation properties associated to Gaussian mixture modeling.

Our work is best placed in the literature of image-to-image translation, where the challenge is to translate one image from a visual domain (e.g. summer) to another one (e.g. winter).

This problem is inherently ill-posed, as there could be many mappings between two images.

Thus, researchers have tried to tackle the problem from many different perspectives.

The most impressive results on this task are undoubtedly related to GANs, which aim to synthesize new images as similar as possible to the real data through an adversarial approach between a Discriminator and a Generator.

The former continuously learns to recognize real and fake images, while the latter tries to generate new images that are indistinguishable from the real data, and thus to fool the Discriminator.

These networks can be effectively conditioned and thus generate new samples from a specific class ) and a latent vector extracted from the images.

For example, Isola et al. (2017) and trained a conditional GAN to encode the latent features that are shared between images of the same domain and thus decode the features to images of the target domain in a one-toone mapping.

However, this approach is limited to supervised settings, where pairs of corresponding images in different domains are available (e.g. a photos-sketch image pair).

In many cases, it is too expensive and unrealistic to collect a large amount of paired data.

Unsupervised Domain Translation.

Translating images from one domain to another without a paired supervision is particularly difficult, as the model has to learn how to represent both the content and the domain.

Thus, constraints are needed to narrow down the space of feasible mappings between images.

Taigman et al. (2017) proposed to minimize the feature-level distance between the generated and input images.

Liu et al. (2017) created a shared latent space between the domains, which encourages different images to be mapped in the same latent space.

Zhu et al. (2017a) proposed CycleGAN, which uses a cycle consistency loss that requires a generated image to be translated back to the original domain.

Similarly, Kim et al. (2017) used a reconstruction loss applying the same approach to both the target and input domains.

Mo et al. (2019) later expanded the previous approach to the problem of translating multiple instances of objects in the same image.

All these methods, however, are limited to a one-to-one domain mapping, thus requiring training multiple models for cross-domain translation.

Recently, Choi et al. (2018) proposed StarGAN, a unified framework to translate images in a multi-domain setting through a single GAN model.

To do so, they used a conditional label and a domain classifier ensuring network consistency when translating between domains.

However, StarGAN is limited to a deterministic mapping between domains.

Style transfer.

A related problem is style transfer, which aims to transform the style of an image but not its content (e.g. from a photo to a Monet painting) to another image (Gatys et al., 2015; Huang & Belongie, 2017; Tenenbaum & Freeman, 1997; Donahue et al., 2018) .

Differently from domain translation, usually the style is extracted from a single reference image.

We willshow that our model could be applied to style transfer as well.

Multi-modal Domain Translation.

Most existing image-to-image translation methods are deterministic, thus limiting the diversity of the translated outputs.

However, even in a one-to-one domain translation such as when we want to translate people's hair from blonde to black, there could be multiple hair styles that are not modeled in a deterministic mapping.

The straightforward solution would be to inject noise in the model, but it turned out to be worthless as GANs tend to ignore this injected noise (Mathieu et al., 2015; Isola et al., 2017; Zhu et al., 2017b) .

To address this prob-lem, Zhu et al. (2017b) proposed BicycleGAN, which encourages the multi-modality in a paired setting through GANs and Variational Auto-Encoders (VAEs).

Almahairi et al. (2018) have instead augmented CycleGAN with two latent variables for the input and target domains and showed that it is possible to increase diversity by marginalizing over these latent spaces.

proposed MUNIT, which assumes that domains share a common content space but different style spaces.

Then, they showed that by sampling from the style space and using Adaptive Instance Normalization (AdaIN) (Huang & Belongie, 2017) , it is possible to have diverse and multimodal outputs.

In a similar vein, focused on the semantic consistency during the translation, and applied AdaIN to the feature-level space.

Recently, proposed a mode seeking loss to encourage GANs to better explore the modes and help the network avoiding the mode collapse.

Altogether, the models in the literature are either multi-modal or multi-domain.

Thus, one has to choose between generating diverse results and training one single model for multiple domains.

Here, we propose a unified model to overcome this limitation.

Concurrent to our work, DRIT++ ) also proposed a multi-modal and multi-domain model using a discrete domain encoding and assuming, however, a zero-mean unit-variance Gaussian shared space for multiple modes.

We instead propose a content-attribute disentangled representation, where the attribute space fits a GMM distribution.

A variational loss forces the latent representation to follow this GMM, where each component is associated to a domain.

This is the key to provide for both multi-modal and multi-domain translation.

In addition, GMM-UNIT is the first method proposing a continuous encoding of the domains, as opposed to the discrete encoding used in the literature.

This is important because it allows for domain interpolation and extrapolation with very few or no data (few/zero-shot generation).

The main properties of GMM-UNIT compared to the literature are shown in Table 1 . (Zhu et al., 2017b) None MUNIT None StarGAN (Choi et al., 2018) Discrete DRIT++ Discrete GMM-UNIT (Proposed) Continuous 3 GMM-UNIT GMM-UNIT is an image-to-image translation model that maps an image to multiple domains in a stochastic fashion.

Following recent seminal works Lee et al., 2018) , our model assumes that each image can be decomposed in a domain-invariant content space and a domainspecific attribute space.

In this paper, we model the attribute latent space through Gaussian Mixture Models (GMMs), formally with a K-component Z-dimensional GMM:

where z ∈ R Z denotes a random attribute vector sample, φ k , µ µ µ k and Σ Σ Σ k denote respectively the weight, mean vector and covariance matrix of the k-th GMM component (φ k ≥ 0,

Z×Z is symmetric and positive definite).

p(z) denotes the probability density of this GMM at z. In the proposed representation, the domains are Gaussian components in a mixture.

This simple yet effective model has two prominent advantages.

Differently from previous works where each domain is a category and the one-hot vector representation grows linearly with the number of domains, we can encode many more domains than the dimension of the latent attribute space Z. Moreover, the continuous encoding of the domains we are introducing in this paper allows us to navigate in the attribute latent space, thus generating images corresponding to domains that have never (or very little) been observed and allowing to interpolate between two domains.

We note that the state of the art models can be traced back as a particular case of GMMs.

Existing multi-domain models such as Choi et al. (2018) or Pumarola et al. (2018) can be modelled with K = |domain in the training data| and ∀k Σ Σ Σ k = 0, thus only allowing the generation of a single result per domain translation.

Then, when K = 1, µ µ µ = 0, and Σ Σ Σ = I it is possible to model the state of the art approaches in multi-modal translation Zhu et al., 2017b) , which share a unique latent space where every domain is overlapped, and it is thus necessary to train N (N − 1) models to achieve the multi-domain translation.

Finally, we can obtain the approach of by separating the latent space from the domain code.

The former is a GMM with K = 1, µ µ µ = 0, and Σ Σ Σ = I, while the latter is another GMM with K = |domain in the training data| and ∀k Σ Σ Σ k = 0.

Thus, our GMM-UNIT is a generalization of the existing state of the art.

the In the next sections, we formalize our model and show that the use of GMMs for the latent space allows learning multi-modal and multi-domain mappings, and also few/zero-shot image generation.

GMM-UNIT follows the generative-discriminative philosophy.

The generator inputs a content latent code c ∈ C C C = R C and an attribute latent code z ∈ Z Z Z = R Z , and outputs a generated image G(c, z).

This image is then fed to a discriminator that must discern between "real" or "fake" images (D r/f ), and must also recognize the domain of the generated image (D dom ).

For an image x n from domain X X X n (i.e. x n ∼ p X X X n ), its latent attribute z n is assumed to follow the n-th Gaussian component of

The attribute and content latent representations need to be learned, and they will be modeled by two architectures, namely a content extractor E c and an attribute extractor E z .

See Figure 2 for a graphical representation of GMM-UNIT.

In addition to tackling the problem of multi-domain and multi-modal translation, we would like these two extractors, content and attribute, to be disentangled.

This would constrain the learning and hopefully yield better domain translation, since the content would be as independent as possible from the attributes.

Formally, the following two properties must hold:

Extracted attribute translation

The encoders E c and E z , and the generator G need to be learned to satisfy three main properties.

Consistency: When traveling through the network, the generated/extracted codes and images must be consistent with the original samples.

Fit: The distribution of the attribute latent space must follow a GMM.

Realism: The generated images must be indistinguishable of real images.

In the following we discuss different losses used to force the overall pipeline to satisfy these properties.

In the textbfconsistency term, we include image, attribute and content reconstruction, as well as cycle consistency.

More formally, we use the following losses:

Self-reconstruction of any input image from its extracted content and attribute vectors:

Content reconstruction from an image, translated into any domain:

Attribute reconstruction from an image translated with any content:

In practice, this loss needs to be complemented with an isometry loss:

Cycle consistency when translating an image back to the original domain:

In the fit term we encourage both the attribute latent variable to follow the Gaussian mixture distribution and the generated images to follow the domain's distribution.

We set two loss functions.

Kullback-Leibler divergence between the extracted latent code and the model.

Since the KL divergence between two GMMs is not analytically tractable, we resort on the fact that we know from which domain are we sampling and define:

where

q(t) dt is the Kullback-Leibler divergence.

Domain classification of generated and original images.

For any given input image x, we would like the method to classify it as its original domain, and to be able to generate from its content an image in any domain.

Therefore, we need two different losses, one directly applied to the original images, and a second one applied to the generated images:

where d X X X n is the label of domain n. Importantly, while the generator is trained using the second loss only, the discriminator D dom is trained using both.

The realism term tries to making the generated images indistinguishable from real images; we adopt the adversarial loss to optimize both the real/fake discriminator D r/f and the generator G:

The full objective function of our network is:

where {λ GAN , λ s/rec , λ c/rec , λ a/rec , λ cyc , λ KL , λ iso , λ dom } are hyper-parameters of weights for corresponding loss terms.

The value of most of these parameters come from the literature.

We refer to Appendix A for the details.

We perform extensive quantitative and qualitative analysis in three real-world tasks, namely: edgesshoes, digits and faces.

First, we test GMM-UNIT on a simple task such as a one-to-one domain translation.

Then, we move to the problem of multi-domain translation where each domain is independent from each other.

Finally, we test our model on multi-domain translation where each domain is built upon different combinations of lower level attributes.

Specifically, for this task, we test GMM-UNIT in a dataset containing over 40 labels related to facial attributes such as hair color, gender, and age.

Each domain is then composed by combinations of these attributes, which might be mutually exclusive (e.g. either male or female) or mutually inclusive (e.g. blonde and black hair).

Additionally, we show how the learned GMM latent space can be used to interpolate attributes and generate images in previously unseen domains, thus showing the first example of few-or zero-shot generation in image-to-image translation.

Finally, GMM-UNIT will be applied to the Style transfer task.

We compare our model to the state of the art of both multi-modal and multi-domain image translation problems.

In the former, we select BicycleGAN (Zhu et al., 2017b) , MUNIT (Zhu et al., 2017a) and MSGAN .

In the latter, we compare with StarGAN (Choi et al., 2018) and DRIT++ , which is the only multi-modal and multi-domain method in the literature.

However, since StarGAN is not multi-modal we additionally test a simple modification of the model where we inject noise in the network.

We call this version StarGAN*. More details are in Appendix A.

We quantitatively evaluate the performance of our method through image quality and diversity of generated images.

The former is evaluated through the Fréchet Inception Distance (FID) Heusel et al. (2017) and the Inception Score (Salimans et al., 2016) .

We evaluate the latter through the LPIPS Distance (Zhang et al., 2018) , NDB and JSD (Richardson & Weiss, 2018) metrics.

In addition, we also show the overall number of parameters used for all domains (Params).

FID We use FID to measure the distance between the generated and real distributions.

Lower FID values indicate better quality of the generated images.

We estimate the FID using 100 input images and 100 samples per input v.s. randomly selected 10000 images from the target domain.

IS To estimate the IS, we use Inception-v3 (Szegedy et al., 2016) fine-tuned on our specific datasets as classifier for 100 input images and 100 samples per input image.

Higher IS means higher generated image quality.

LPIPS The LPIPS distance is defined as the L 2 distance between the features extracted by a deep learning model of two images.

This distance has been demonstrated to match well the human perceptual similarity (Zhang et al., 2018) .

Thus, following Zhu et al. (2017b) ; ; Lee et al. (2018) , we randomly select 100 input images and translate them to different domains.

For each domain translation, we generate 10 images for each input image and evaluate the average LPIPS distance between the 10 generated images.

Finally, we get the average of all distances.

Higher LPIPS distance indicates better diversity among the generated images.

NDB and JSD These are measuring the similarity between the distributions of real and generated images.

We use the same testing data as for FID.

Lower NDB and JSD mean the generated data distribution approaches better the real data distribution.

We first evaluate our model on a simpler task than multi-domain translation: two-domain translation (e.g. edges to shoes).

We use the dataset provided by Isola et al. (2017) ; Zhu et al. (2017a) containing images of shoes and their edge maps generated by HED (Xie & Tu, 2015) .

We train a single model for edges ↔ shoes without using paired information.

Figure 3 displays examples of shoes generated from the same sketch by GMM-UNIT.

Table 2 shows the quantitative evaluation and comparison with the state-of-the-art.

Our model generates images with high diversity and quality using half the parameters of the state of the art.

We refer to Appendix B.1 for additional results on this task.

Figure 3: Examples of edges → shoes translation with the proposed GMM-UNIT.

We then evaluate our model in a multi-domain translation problem where each domain is composed by digits collected in different scenes.

We use the Digits-Five dataset introduced in Xu et al. (2018) , from which we select three different domains, namely MNIST (LeCun et al., 1998) , MNIST-M (Ganin & Lempitsky, 2014) , a colorized version of MNIST for domain adaptation, and Street View House Numbers (SVHN) (Netzer et al., 2011) .

We compare our model with the state-of-theart on multi-domain translation, and we show in Figure 4 and Table 3 the qualitative and quantitative results respectively.

From these results we conclude that StarGAN* fails at generating diversity, thus confirming the findings of previous studies that adding noise does not increase diversity (Mathieu et al., 2015; Isola et al., 2017; Zhu et al., 2017b) .

GMM-UNIT instead generates images with higher quality and diversity than all the state-of-the-art models.

We note, however, that StarGAN* achieves a higher IS, probably due to the fact that it solves a simpler task.

Additional experiments carried out implementing a StarGAN*-like GMM-UNIT (i.e. setting σ σ σ k = 0, ∀k) indeed produced similar results.

Specifically, the StarGAN*-like GMM-UNIT tends to generate for each input image one single (deterministic) output and thus the corresponding LPIPS scores are around zero.

We refer to Appendix B.2 for additional results on this task.

We also evaluate GMM-UNIT in the complex setting of multi-domain translation in a dataset of facial attributes.

We use the CelebFaces Attributes (CelebA) dataset (Liu et al., 2015) , which contains 202,599 face images of celebrities where each face is annotated with 40 binary attributes.

We resize the initial 178×218 size images to 128×128.

We randomly select 2,000 images for testing and use all remaining images for training.

This dataset is composed of some attributes that are mutually exclusive (e.g. either male or female) and those that are mutually inclusive (e.g. people could have both blonde and black hair).

Thus, we model each attribute as a different GMM component.

For this reason, we can generate new images for all the combinations of attributes by sampling from the GMM.

As aforementioned, this is not possible for state-of-the-art models such as StarGAN and DRIT++, as they use one-hot domain codes to represent the domains.

For the purpose of this experiment we show five binary attributes: hair color (black, blond, brown), gender (male/female), and age (young/old).

These five attributes allow GMM-UNIT to generate 32 domains.

Figure 5 shows some generated results of our model.

We can see that GMM-UNIT learns to translate images to simple attributes such as blonde hair, but also to translate images with combinations of them (e.g. blonde hair and male).

Moreover, we can see that the rows show different realizations of the model thus demonstrating the stochastic approach of GMM-UNIT.

These results are corroborated by Table 4 that shows that our model is superior to StarGAN* in both quality and diversity of generated images.

We also note in this experiment that the IS is higher in StarGAN*. Additional results are on Appendix B.3.

We evaluate our model on style transfer, which is a specific task where the style is usually extracted from a single reference image.

Thus, we randomly select two input images and synthesize new images where, instead of sampling from the GMM distribution, we extract the style (through E z ) from some reference images.

Figure 6 shows that the generated images are sharp and realistic, showing that our method can also be effectively applied to Style transfer.

Figure 6: Examples of GMM-UNIT applied on the Style transfer task.

The style is here extracted from a single reference images provided by the user.

In addition, we evaluate the ability of GMM-UNIT to synthesize new images with attributes that are extremely scarce or non present in the training dataset.

To do so, we select three combinations of attributes consisting of less than two images in the CelebA dataset: Black hair+Blonde hair+Male+Young, Black hair+Blonde hair+Female+Young and Black hair+Blonde hair+Brown+Young.

Figure 7: Generated images in previously unseen combinations of attributes.

Figure 7 shows that learning the continuous and multi-modal latent distribution of attributes allow to effectively generate images as zero-or few-shot generation.

At the best of our knowledge, we are the first ones being able to translate images in previously unseen domains.

This can be extremely important in tasks that are extremely imbalanced.

Finally, we show that by learning the full latent distribution of the attributes we can do attribute interpolation both intra-and inter-domains.

In contrast, state of the art methods such as can only do intra-domain interpolations due to their discrete domain encoding.

Figure 8 shows some generated images through a linear interpolation between two given attributes, while in Appendix B.3 we show that we can also do intra-domain interpolations.

We compare GMM-UNIT with three variants of the model that ablate L cyc , L d/rec and L iso in the Digits dataset.

Table 5 shows the results of the ablation.

As expected, L cyc is needed to have higher image quality.

When L d/rec is removed image quality decreases, but L iso still helps to learn the attributes space.

Finally, without L iso we observe that both diversity and quality decrease, thus confirming the need of all these losses.

We refer to Appendix B.4 for the additional ablation results broken down by domain.

In this paper, we present a novel image-to-image translation model that maps images to multiple domains and provides a stochastic translation.

GMM-UNIT disentangles the content of an image from its attributes and represents the attribute space with a GMM, which allows us to have a continuous encoding of domains.

This has two main advantages: first, it avoids the linear growth of the dimension of the attribute space with the number of domains.

Second, GMM-UNIT allows for interpolation across-domains and the translation of images into previously unseen domains.

We conduct extensive experiments in three different tasks, namely two-domain translation, multidomain translation and multi-attribute multi-domain translation.

We show that GMM-UNIT achieves quality and diversity superior to state of the art, most of the times with fewer parameters.

Future work includes the possibility to thoroughly learn the mean vectors of the GMM from the data and extending the experiments to a higher number of GMM components per domain.

Our deep neural models are built upon the state-of-the-art methods MUNIT , BicycleGAN (Zhu et al., 2017b) and StarGAN (Choi et al., 2018) , as shown in Table 6 with details.

We apply Instance Normalization (IN) (Ulyanov et al., 2017) to the content encoder E c and Adaptive Instance Normalization (AdaIN) (Huang & Belongie, 2017) and Layer Normalization (LN) (Ba et al., 2016) for the decoder G. For the discriminator network, we use Leaky ReLU (Xu et al., 2015) with a negative slope of 0.2.

We use the following notations: D: the number of domains, N: the number of output channels, K: kernel size, S: stride size, P: padding size, CONV: a convolutional layer, GAP: a global average pooling layer, UPCONV: a 2×bilinear upsampling layer followed by a convolutional layer.

Note that we reduce the number of layers of the discriminator on the Digits dataset.

We use the Adam optimizer (Kingma & Ba, 2015) with β 1 = 0.5, β 2 = 0.999, and an initial learning rate of 0.0001.

The learning rate is decreased by half every 100,000 iterations.

In all experiments, we use a batch size of 1 for Edges2shoes and Faces and batch size of 32 for Digits.

And we set the loss weights to λ GAN = 1, λ s/rec = 10, λ c/rec = 1, λ a/rec = 1, λ cyc = 10, λ KL = 0.1, λ iso = 0.1 and λ dom = 1.

We use the domain-invariant perceptual loss with weight 0.1 in all experiments.

Random mirroring is applied during training.

, S1, P0)

A.1 GMM

In our experiments we use a simplified version of the GMM, which satisfies the following properties:

• The mean vectors are placed on the vertices of (N − 1)-dimensional regular simplex, so that the mean vectors are equidistant.

• The covariance matrices are diagonal, with the same on all the components.

In other words, each Gaussian component is spherical, formally: Σ Σ Σ k = σ σ σ k I, where I is the identity matrix.

B ADDITIONAL RESULTS

In this section, we present the additional results for the one-to-one domain translation.

As shown in Figure 9 , we qualitatively compare GMM-UNIT with the state-of-the-art.

We observe that while all the methods (multi-domain and not) achieve acceptable diversity, it seems that DRIT++ suffers from problems of realism.

Figure 10 shows the qualitative comparison with the state of the art, while Figure 1 : Examples on using reference images to provide attribute representations.

2 Figure 9 : Visual comparisons of state of the art methods on Edge ↔ Shoes dataset.

We note that Bicycle-GAN, MUNIT and MSGAN are one-to-one domain translation models, while StarGAN* is a multi-domain (deterministic) model.

Finally DRIT++ and GMM-UNIT are multi-modal and multi-domain methods.

In Table 9 we show the quantitative results on the CelebA datset, broken down per domain.

In Figure 11 we show some generated images in comparison with StarGAN*. Figure 12 shows the possibility to do attribute interpolation inside a domain.

In Table 10 we show additional, per domain, ablation results on the Digits dataset.

C VISUALIZATION OF THE ATTRIBUTE LATENT SPACE Figure 13 shows that the attributes sampled from the distribution and those extracted by the encoder E z are mapped and well projected in the latent space of the attributes.

Figure 13 : Visualization of the attribute vectors in a 2D space via t-SNE method.

"S" refers to randomly sampling from GMM components (1: black hair, 2: blondehair, 3: brown hair) and "E" refers to extracting attribute vectors by the encoder Ez from the real data.

@highlight

GMM-UNIT is an image-to-image translation model that maps an image to multiple domains in a stochastic fashion.