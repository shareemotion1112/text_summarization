Most domain adaptation methods consider the problem of transferring knowledge to the target domain from a single source dataset.

However, in practical applications, we typically have access to multiple sources.

In this paper we propose the first approach for Multi-Source Domain Adaptation (MSDA) based on Generative Adversarial Networks.

Our method is inspired by the observation that the appearance of a given image depends on three factors: the domain, the style (characterized in terms of low-level features variations) and the content.

For this reason we propose to project the image features onto a space where only the dependence from the content is kept, and then re-project this invariant representation onto the pixel space using the target domain and style.

In this way, new labeled images can be generated which are used to train a final target classifier.

We test our  approach using common MSDA benchmarks, showing that it outperforms state-of-the-art methods.

A well known problem in computer vision is the need to adapt a classifier trained on a given source domain in order to work on another domain, i.e. the target.

Since the two domains typically have different marginal feature distributions, the adaptation process needs to align the one to the other in order to reduce the domain shift (Torralba & Efros (2011) ).

In many practical scenarios, the target data are not annotated and Unsupervised Domain Adaptation (UDA) methods are required.

While most previous adaptation approaches consider a single source domain, in real world applications we may have access to multiple datasets.

In this case, Multi-Source Domain Adaptation (MSDA) (Yao & Doretto (2010) ; Mansour et al. (2009) ; Xu et al. (2018) ; Peng et al. (2019) ) methods may be adopted, in which more than one source dataset is considered in order to make the adaptation process more robust.

However, despite more data can be used, MSDA is challenging as multiple domain shift problems need to be simultaneously and coherently solved.

In this paper we tackle MSDA (unsupervised) problem and we propose a novel Generative Adversarial Network (GAN) for addressing the domain shift when multiple source domains are available.

Our solution is based on generating artificial target samples by transforming images from all the source domains.

Then the synthetically generated images are used for training the target classifier.

While this strategy has been recently adopted in single-source UDA scenarios (Russo et al. (2018) ; ; Liu & Tuzel (2016) ; Murez et al. (2018) ; Sankaranarayanan et al. (2018) ), we are the first to show how it can be effectively exploited in a MSDA setting.

The holy grail of any domain adaptation method is to obtain domain invariant representations.

Similarly, in multi-domain image-to-image translation tasks it is very crucial to obtain domain invariant representations in order to reduce the number of learned translations from O(N 2 ) to O(N ), where N is the number of domains.

Several domain adaptation methods (Roy et al. (2019) ; Carlucci et al. (2017) ; ; Tzeng et al. (2014) ) achieve domain-invariant representations by aligning only domain specific distributions.

However, we postulate that style is the most important latent factor that describe a domain and need to be modelled separately for obtaining optimal domain invariant representation.

More precisely, in our work we assume that the appearance of an image depends on three factors: i.e. the content, the domain and the style.

The domain models properties that are shared by the elements of a dataset but which may not be shared by other datasets, whereas, the factor style represents a property that is shared among different parts of a single image and describes low-level features which concern a specific image.

Our generator obtains the do-main invariant representation in a two-step process, by first obtaining style invariant representations followed by achieving domain invariant representation.

In more detail, the proposed translation is implemented using a style-and-domain translation generator.

This generator is composed of two main components, an encoder and a decoder.

Inspired by (Roy et al. (2019) ) in the encoder we embed whitening layers that progressively align the styleand-domain feature distributions in order to obtain a representation of the image content which is invariant to these factors.

Then, in the decoder, we project this invariant representation onto a new domain-and-style specific distribution with Whitening and Coloring (W C) ) batch transformations, according to the target data.

Importantly, the use of an intermediate, explicit invariant representation, obtained through W C, makes the number of domain transformations which need to be learned linear with the number of domains.

In other words, this design choice ensures scalability when the number of domains increases, which is a crucial aspect for an effective MSDA method.

Contributions.

Our main contributions can be summarized as follows.

(i) We propose the first generative model dealing with MSDA.

We call our approach TriGAN because it is based on three different factors of the images: the style, the domain and the content. (ii) The proposed style-anddomain translation generator is based on style and domain specific statistics which are first removed from and then added to the source images by means of modified W C layers: Instance Whitening Transform (IW T ), Domain Whitening Transform (DW T ) (Roy et al. (2019) ), conditional Domain Whitening Transform (cDW T ) and Adaptive Instance Whitening Transform (AdaIW T ).

Notably, the IW T and AdaIW T are novel layers introduced with this paper. (iii) We test our method on two MSDA datasets, Digits-Five (Xu et al. (2018) ) and Office-Caltech10 (Gong et al. (2012) ), outperforming state-of-the-art methods.

In this section we review previous approaches on UDA, considering both single source and multisource methods.

Since, the proposed architecture is also related to deep models used for image-toimage translation, we also discuss related work on this topic.

Single Source UDA.

Single source UDA approaches assume a single labeled source domain and can be broadly classified under three main categories, depending upon the strategies adopted to cope with the domain-shift problem.

The first category utilizes the first and second order statistics to model the source and target feature distributions.

For instance, (Long & Wang (2015) ; Long et al. (2017) ; Venkateswara et al. (2017); Tzeng et al. (2014) ) minimize the Maximum Mean Discrepancy, i.e. the distance between the mean of feature distributions between the two domains.

On the other hand, ; Morerio et al. (2018); Peng & Saenko (2018) ) achieve domain invariance by aligning the second-order statistics through correlation alignment.

Differently, (Carlucci et al. (2017) ; Li et al. (2017) ; Mancini et al. (2018) ) reduce the domain shift by domain alignment layers derived from batch normalization (BN) (Ioffe & Szegedy (2015) ).

This idea has been recently extended in (Roy et al. (2019) ), where grouped-feature whitening (DWT) is used instead of feature standardization as in BN .

Contrarily, in our proposed encoder we use the W C transform, which we adapt to work in a generative network.

In addition, we also propose other style and domain dependent batch-based normalizations (i.e., IW T , cDW T and AdaIW T ).

The second category of methods aim to build domain-agnostic representations by means of an adversarial learning-based approach.

For instance, discriminative domain-invariant representations are constructed through a gradient reversal layer in (Ganin & Lempitsky (2015) ).

Similarly, the approach in ) utilizes a domain confusion loss to promote the alignment between the source and the target domain.

A third category of methods use adversarial learning in a generative framework (i.e., GANs (Goodfellow et al. (2014) ) to reconstruct artificial source and/or target images and perform domain adaptation.

Notable approaches are SBADA-GAN (Russo et al. (2018) ), CyCADA ), CoGAN (Liu & Tuzel (2016) ), I2I Adapt (Murez et al. (2018) ) and Generate To Adapt (GTA) (Sankaranarayanan et al. (2018) ).

While these generative methods have been shown to be very successful in UDA, none of them deals with a multi-source setting.

Indeed, extending these approaches to deal with multiple source domains is not trivial, because the construction of O(N 2 ) one-to-one translation generators and discriminator networks would most likely dramatically increase the number of parameters which need to be trained.

Figure 1: An overview of the TriGAN generator.

We schematically show 3 domains {T, S 1 , S 2 } -objects with holes, 3D objects and skewered objects, respectively.

The content is represented by the object's shape -square, circle or triangle.

The style is represented by the color: each image input to G has a different color and each domain has it own set of styles.

First, the encoder E creates a styleinvariant representation using IWT blocks.

DWT blocks are then used to obtain a domain-invariant representation.

Symmetrically, the decoder D brings back domain-specific information with cDWT blocks (for simplicity we show only a single domain, T ).

Finally, we apply a reference style.

The reference style is extracted using style path and it is applied using Adaptive IWT blocks.

Multi-source UDA.

Yao & Doretto (2010) deal with multiple-source knowledge transfer by borrowing knowledge from the target k nearest-neighbour sources.

Similarly, a distribution-weighed combining rule is proposed in (Mansour et al. (2009) ) to construct a target hypothesis as a weighted combination of source hypotheses.

Recently, Deep Cocktail Network (DCTN) (Xu et al. (2018) ) uses the distribution-weighted combining rule in an adversarial setting.

A Moment Matching Network (M 3 SDA) is introduced in (Peng et al. (2019) ) for reducing the discrepancy between the multiple source and the target domains.

Differently from these methods which operate in a discriminative setting, our method relies on a deep generative approach for MSDA.

Image-to-image Translation.

Image-to-image translation approaches, i.e. the methods that learn how to transform an image from one domain to another, possibly keeping its semantics, are the basis of our method.

In ) the pix2pix network translates images under the assumption that paired images in the two domains are available at training time.

In contrast, CycleGAN ) can learn to translate images using unpaired training samples.

Note that, by design, these methods work with two domains.

ComboGAN (Anoosheh et al. (2018) ) partially alleviates this issue by using N generators for translations among N domains.

Our work is also related to StarGAN (Choi et al. (2018) ) which handles unpaired image translation amongst N domains (N ≥ 2) through a single generator.

However, StarGAN achieves image translation without explicitly forcing representations to be domain invariant and this may lead to significant reduction of network representation power as the number of domains increases.

On the other hand, our goal is to obtain an explicit, intermediate image representation which is style-and-domain independent.

We use IWT and DWT to achieve this.

We also show that this invariant representation can simplify the re-projection process onto a desired style and target domain.

This is achieved through AdaIW T and cDW T which results into very realistic translations amongst domains.

Very recently, a whitening and colouring based image-to-image translation method was proposed in (Cho et al. (2019) ), where the whitening operation is weight-based.

Specifically, the whitening operation is approximated by enforcing the convariance matrix, computed from the intermediate features, to be equal to the identity matrix.

Conversely, our whitening layers are data dependent and they use the Cholesky decomposition (Dereniowski & Kubale (2003) ) to compute the whitening matrices from the input samples in a closed form, thereby eliminating the need for additional ad-hoc losses.

In this section we describe the proposed approach for MSDA.

We first provide an overview of our method and introduce the notation adopted throughout the paper (Sec. 3.1).

Then we describe the TriGAN architecture (Sec. 3.2) and our training procedure (Sec.3.3).

In the MSDA scenario we have access to N labeled source datasets {S j } N j=1 , where

, and a target unlabeled dataset T = {x k } nt k=1 .

All the datasets (target included) share the same categories and each of them is associated to a domain D

t , respectively.

Our final goal is to build a classifier for the target domain D t exploiting the data in {S j } N j=1 ∪ T .

Our method is based on two separate training steps.

We initially train a generator G which learns how to change the appearance of a real input image in order to adhere to a desired domain and style.

Importantly, our G learns mappings between every possible pair of image domains.

Once G is trained, we use it to generate target data having the same content of the source data, thus creating a new, labeled, target dataset, which is finally used to train a target classifier C. In more detail, G is trained using {S j } N j=1 ∪ T , however no class label is involved in this phase and T is treated in the same way as the other domain datasets.

As mentioned in Sec. 1, G is composed of an encoder E and a decoder D (Fig. 1) .

The role of E is to whiten, i.e., to remove, both domain-specific and style-specific aspects of the input image features in order to obtain domain and style invariant representations.

Conversely and symmetrically, D needs to progressively project the domain-andstyle invariant features generated by E onto a domain-and-style specific space.

At training time, G takes as input a batch of images B = {x 1 , ..., x m } with corresponding domain labels L = {l 1 , ..., l m }, where x i belongs to the domain D li and

, and a batch of style images

and has the same style of image x O i .

The TriGAN architecture is made of a generator network G and a discriminator D P .

As stated above, G comprises an encoder E and decoder D, which will be described in (Sec. 3.2.2-3.2.3).

The discriminator D P is based on the Projection Discriminator (Miyato & Koyama (2018) ).

Before describing the details of G, we briefly review the W C transform ) (Sec. 3.2.1) which is used as the basic operation in our proposed batch-based feature transformations.

h×w×d be the tensor representing the activation values of the convolutional feature maps in a given layer corresponding to the input image x, with d channels and h × w spatial locations.

We treat each spatial location as a d-dimensional vector, in this way each image x i contains a set of vectors X i = {v 1 , ..., v h×w }.

With a slight abuse of the notation, we use

.., v h×w×m }, which includes all the spatial locations in all the images in a batch.

The W C transform is a multivariate extension of the per-dimension normalization and shiftscaling transform (BN ) proposed in (Ioffe & Szegedy (2015) ) and widely adopted in both generative and discriminative networks.

W C can be described by:

where:

In Eq. 2, µ B is the centroid of the elements in B, while W B is such that:

B , where Σ B is the covariance matrix computed using B. The result of applying Eq. 2 to the elements of B, is a set of whitened featuresB = {v 1 , ...,v h×w×m }, which lie in a spherical distribution (i.e., with a covariance matrix equal to the identity matrix).

On the other hand, Eq. 1 performs a coloring transform, i.e. projects the elements inB onto a learned multivariate Gaussian distribution.

While µ B and W B are computed from the elements in B, Eq. 1 depends on the learnable d dimensional vector β and d × d dimensional matrix Γ. Eq. 1 is a linear operation and can be simply implemented using a convolutional layer with kernel size 1 × 1.

In this paper we use the WC transform in our encoder E and decoder D, in order to first obtain a style-and-domain invariant representation for each x i ∈ B, and then transform this representation

The encoder E is composed of a sequence of standard Convolution k×k -N ormalization -ReLU -AverageP ooling blocks and some ResBlocks (more details in Appendix B), in which we replace the common BN layers (Ioffe & Szegedy (2015) ) with our proposed normalization modules, which are detailed below.

Obtaining Style Invariant Representations.

In the first two blocks of E we whiten first and secondorder statistics of the low-level features of each X i ⊆ B, which are mainly responsible for the style of an image (Gatys et al. (2016) ).

To do so, we propose the Instance Whitening Transform (IW T ), where the term instance is inspired by Instance Normalization (IN ) (Ulyanov et al. (2016) ) and highlights that the proposed transform is applied to a set of features extracted from a single image

Eq. 3 implies that whitening is performed using an image-specific feature centroid µ Xi and covariance matrix Σ Xi , which represent the first and second-order statistics of the low-level features of x i .

Coloring is based on the parameters β and Γ, which do not depend on x i or l i .

The coloring operation is the analogous of the shift-scaling per-dimension transform computed in BN just after feature standardization (Ioffe & Szegedy (2015) ) and is necessary to avoid decreasing the network representation capacity ).

Obtaining Domain Invariant Representations.

In the subsequent blocks of E we whiten first and second-order statistics which are domain specific.

For this operation we adopt the Domain Whitening Transform (DW T ) proposed in (Roy et al. (2019) ).

Specifically, for each X i ⊆ B, let l i be its domain label (see Sec. 3.1) and let B li ⊆ B be the subset of feature which have been extracted from all those images in B which share the same domain label.

Then, for each v j ∈ B li :

Similarly to Eq. 3, Eq. 4 performs whitening using a subset of the current feature batch.

Specifically, all the features in B are partitioned depending on the domain label of the image they have been extracted from, so obtaining B 1 , B 2 , ..., etc, where all the features in B l belongs to the images from domain D l .

Then, first and second order statistics (µ B l , Σ B l ) are computed thus effectively projecting each v j ∈ B li onto a domain-invariant spherical distribution.

A similar idea was recently proposed in (Roy et al. (2019) ) in a discriminative network for single-source UDA.

However, differently from (Roy et al. (2019) ), we also use coloring by re-projecting the whitened features onto a new space governed by a learned multivariate distribution.

This is done using the learnable parameters β and Γ which do not depend on l i .

Our decoder D is functionally and structurally symmetric with respect to E: it takes as input domain and style invariant features computed by E and projects these features onto the desired domain

with style extracted from desired image x O i .

Similarly to E, D is a sequence of ResBlocks and a few U psampling -N ormalizationReLU -Convolution k×k blocks (more details in Appendix B).

Similarly to Sec. 3.2.2, in the N ormalization layers we replace BN with our proposed feature normalization approaches, which are detailed below.

Projecting Features onto a Domain-specific Distribution.

Apart from the last two blocks of D (see below), all the other blocks are dedicated to project the current set of features onto a domain-specific subspace.

This subspace is learned from data using domain-specific coloring parameters (β l , Γ l ), where l is the label of the corresponding domain.

To this purpose we introduce the conditional Domain Whitening Transform (cDW T ), where the term "conditional" specifies that the coloring step is conditioned on the domain label l. In more detail: Similarly to Eq. 4, we first partition B into B 1 , B 2 , ..., etc.

However, the membership of v j ∈ B to B l is decided taking into account the desired output domain label l O i for each image rather than its original domain as in case of to Eq. 4.

Specifically, let v j ∈ X i and the output domain is given by the label l

Once B has been partitioned we define cDW T as follows:

Note that, after whitening, and differently from Eq. 4, coloring in Eq. 5 is performed using domainspecific

Applying a Specific Style.

In order to apply a given style to x i , we extract the style from image x O i using the Style Path (see Fig. 1 ).

Style Path consists of two Convolution k×k -IW T -ReLU -AverageP ooling blocks (which shares the parameters with the first two layers of encoder) and a MultiLayer Perceptron (MLP) F. Following (Gatys et al. (2016)) we describe a style using first and second order statistics

, which are extracted using the IW T blocks.

Then we use F to adapt these statistics to the domain-specific representation obtained as the output of the previous step.

In fact, in principle, for each v j ∈ X O i , the W hitening() operation inside the IW T transform could be "inverted" using:

Indeed, the coloring operation (Eq. 1) is the inverse of whitening (Eq. 2).

However, the elements of X i now lie in a feature space different from the output space of Eq. 3, thus the transformation defined by Style Path needs to be adapted.

For this reason, we use a MLP (F) which implements this adaptation:

Note that, in Eq. 7,

has been generated, we use it as the coloring parameters of our Adaptive IWT (AdaIW T ):

Eq. 8 imposes style-specific first and second order statistics to the features of the last blocks of D in order to mimic the style of x O i .

GAN Training.

For the sake of clarity, in the rest of the paper we use a simplified notation for G, in which G takes as input only one image instead of a batch.

Specifically,

) be the generated image, starting from x i (x i ∈ D li ) and with desired output domain l O i and style image x O i .

G is trained using the combination of three different losses, with the goal of changing the style and the domain of x i while preserving its content.

First, we use an adversarial loss based on the Projection Discriminator (Miyato & Koyama (2018) ) (D P ), which is conditioned on labels (domain labels, in our case) and uses a hinge loss:

The second loss is the Identity loss proposed in ), and which in our framework is implemented as follows:

In Eq. 11, G computes an identity transformation, being the input and the output domain and style the same.

After that, a pixel-to-pixel L 1 norm is computed.

Finally, we propose to use a third loss which is based on the rationale that the generation process should be equivariant with respect to a set of simple transformations which preserve the main content of the images (e.g., the foreground object shape).

Specifically, we use the set of the affine transformations {h(x; θ)} of image x which are defined by the parameter θ (θ is a 2D transformation matrix).

The affine transformation is implemented by differentiable billinear kernel as in (Jaderberg et al. (2015) ).

The Equivariance loss is:

In Eq. 12, for a given image x i , we randomly choose a parameter θ i and we apply h(·; θ i ) tô

Then, using the same θ i , we apply h(·; θ i ) to x i and we get x i = h(x i ; θ i ), which is input to G in order to generate a second image.

The two generated images are finally compared using the L 1 norm.

This is a form of self-supervision, in which equivariance to geometric transformations is used to extract semantics.

Very recently a similar loss has been proposed in (Hung et al. (2019) ), where equivariance to affine transformations is used for image co-segmentation.

The complete loss for G is:

Classifier Training.

Once G is trained, we use it to artificially create a labeled training dataset for the target domain.

Specifically, for each S j and each (x i , y i ) ∈ S j , we randomly pick one image x t from T which is used as the style-image reference, and we generate:

where N + 1 is fixed and indicates the target domain D t label (see Sec. 3.1).

and the process is iterated.

Finally, we train a classfier C on T L using the cross-entropy loss:

In this section we describe the experimental setup and then we evaluate our approach using MSDA datasets.

We also present an ablation study in which we analyse the impact of each of TriGAN component on the classification accuracy.

In our experiments we consider two common domain adaptation benchmarks, namely the DigitsFive benchmark (Xu et al. (2018) ) and the Office-Caltech (Gong et al. (2012) ).

The Digits-Five (Xu et al. (2018) ) is composed of five digit-recognition datasets: USPS (Friedman et al. (2001) ), MNIST (LeCun et al. (1998) ), MNIST-M (Ganin & Lempitsky (2015) ), SVHN (Netzer et al. (2011)) and Synthetic numbers datasets (Ganin et al. (2016) ) (SYNDIGITS).

SVHN (Netzer et al. (2011)) contains images from Google Street View of real-world house numbers.

Synthetic numbers (Ganin et al. (2016) ) includes 500K computer-generated digits with different sources of variations (i.e. position, orientation, color, blur).

USPS (Friedman et al. (2001) ) is a dataset of digits scanned from U.S. envelopes, MNIST (LeCun et al. (1998) ) is a popular benchmark for digit recognition and MNIST-M (Ganin & Lempitsky (2015) ) is its colored counterpart.

We adopt the experimental protocol described in (Xu et al. (2018) ): in each domain the train/test split is composed of a subset of 25000 images for training and 9000 for testing.

For USPS the entire dataset is used.

The Office-Caltech (Gong et al. (2012) ) is a domain-adaptation benchmark obtained selecting the subset of 10 categories shared between the Office31 and the Caltech256 (Griffin et al. (2007) ) datasets.

It contains 2533 images, about half of which belong to Caltech256.

There are four different domains: Amazon (A), DSLR (D), Webcam (W) and Caltech256 (C).

We provide architecture details about our generator G and discriminator D P in the Appendix B. We train TriGAN for 100 epochs using the Adam optimizer (Kingma & Ba (2014)) with the learning rate set to 1e-4 for the G and 4e-4 for the D P as in (Heusel et al. (2017) ).

The loss weighing factor λ in Eqn.

13 is set to 10 as in ).

All other hyperparameters are chosen by crossvalidating on the MNIST-M, USPS, SVHN, SYNDIGITS → MNIST adaptation setting and are used in all the other settings.

For the Digits-Five experiments we use a mini-batch of size 256 for TriGAN training.

Due to the difference in image resolution and image channels, the images of all the domains are converted to 32 × 32 RGB.

For a fair comparison, for the final target classifier C we use exactly the same network architecture used in (Ganin et al. (2016) ).

In the Office-Caltech10 experiments we downsample the images to 164 × 164 to accommodate more samples in a mini-batch.

We use a mini-batch of size 24 for training with 1 GPU.

For the back-bone target classifier C we use the ResNet101 (He et al. (2016) ) architecture used by Peng et al. (2019) .

The weights are initialized with a network pre-trained on the ILSVRC-2012 dataset (Russakovsky et al. (2015) ).

In our experiments we remove the output layer and we replace it with a randomly initialized fully-connected layer that has 10 logits, one per each class of the OfficeCaltech10 dataset.

C is trained with Adam with an initial learning rate of 1e-5 for the randomly initialized last layer and 1e-6 for all other layers.

Since there are only a few training data in the T L dataset, we also use {S j } N j=1 for training C.

In this section we analyse our proposal using an ablation study and we compare with MSDA stateof-the-art methods.

In Appendix A we show our qualitative results.

In this section we compare our method with previous MSDA approaches.

Tab.

1 and Tab.

2 show the results on the Digits-Five and Office-Caltech10 datset, respectively.

Table 1 shows that TriGAN achieves an average accuracy of 90.08% which is higher than all other methods.

M 3 SDA is better in the mm, up, sv, sy → mt and in the mt, mm, sv, sy → up settings, where TriGAN is the second best.

In all the other settings, TriGAN outperforms all other approaches.

As an example, in the mt, up, sv, sy → mm setting, TriGAN is better than the second best method M 3 SDA by a significant margin of 10.38%.

For the StarGAN (Choi et al. (2018) ) baseline, synthetic images are generated in the target domain and a target classifier is trained using our protocol described in Sec. 3.3.

StarGAN, despite known to work well for aligned face translation, fails drastically when digits are concerned.

This shows the importance of a well-designed generator that enforces domain invariant representations in the MSDA setting when there is a significant domain shift.

Models Table 1 : Classification accuracy (%) on Digits-Five experiments.

MNIST-M, MNIST, USPS, SVHN, Synthetic Digits are abbreviated as mm, mt, up, sv and sy respectively.

The best value is in bold and the second best is underlined.

Finally, we also experimented using the Office-Caltech10, which is considered to be difficult for reconstruction-based GAN methods because of the high-resolution images.

Although the dataset is quite saturated, TriGAN achieves a classification accuracy of 97.0%, outperforming all the other methods and beating the previous state-of-the-art approach (M 3 SDA) by a margin of 0.6% on average (see Tab.

2).

Table 2 : Classification accuracy (%) on Office-Caltech10 dataset.

ResNet-101 pre-trained on ImageNet is used as the backbone network.

The best value is in bold and the second best is underlined.

In this section we analyse the different components of our method and study in isolation their impact on the final accuracy.

Specifically, we use the Digits-Five dataset and the following models: i) Model A, which is our full model containing the following components: IWT, DWT, cDWT, AdaIWT and L Eq .

ii) Model B, which is similar to Model A except we replace L Eq with the cycle-consistency loss L Cycle of CycleGAN ).

iii) Model C, where we replace IWT, DWT, cDWT and AdaIWT of Model A with IN (Ulyanov et al. (2016)), BN (Ioffe & Szegedy (2015)), conditional Batch Normalization (cBN) (Dumoulin et al. (2016)) and Adaptive Instance Normalization (AdaIN) (Huang et al. (2018) ).

This comparison highlights the difference between feature whitening and feature standardisation.

iv) Model D, which ignores the style factor.

Specifically, in Model D, the blocks related to the style factor, i.e., the IWT and the AdaIWT blocks, are replaced by DWT and cDWT blocks, respectively.

v) Model E, in which the style path differs from Model A in the way the style is applied to the domain-specific representation.

Specifically, we remove the MLP F(.) and we directly apply (µ

).

vi) Finally, Model F represents no-domain assumption (e.g. the DWT and cDWT blocks are replaced with standard WC blocks).

Tab.

3 shows that Model A outperforms all the ablated models.

Model B shows that L Cycle is detrimental for the accuracy because G may focus on meaningless information to reconstruct back the image.

Conversely, the affine transformations used in case of L Eq , force G to focus on the shape of the content of the images.

Model C is outperformed by model A, demonstrating the importance of feature whitening over feature standardisation, corroborating the findings of (Roy et al. (2019) ) in a pure-discriminative setting.

Moreover, the no-style assumption in Model D hurts the classification accuracy by a margin of 1.76% when compared with Model A. We believe this is due to the fact that, when only domain-specific latent factors are modeled but instance-specific style information is missing in the image translation process, then the diversity of translations decreases, consequently reducing the final accuracy.

Model E shows the need of using the proposed style path.

Finally, Model F shows that having a separate factor for domain yields better performance

Our proposed method can be used for multi-domain image-to-image translation tasks.

We conduct experiments on Alps Seasons dataset (Anoosheh et al. (2018) ) which consists of images of Alps mountain range belonging to four different domains.

Fig. 2 shows some images generated using our generator on the Alps Seasons.

For this experiment we compare our generator with StarGAN (Choi et al. (2018) ) and report the FID (Heusel et al. (2017) ) metrics for the generated images.

FID measures the realism of generated images and it is desirable to have a lower FID score.

The FID is computed considering all the real samples in the target domain and generating equivalent number of synthetic images in the target domain.

It can be observed from Tab.

4 that the FID scores of our approach is significantly lower than that of StarGAN.

This further highlights the fact that explicit enforcing of domain and style invariant representation is essential for multi-domain translation. (Choi et al. (2018)) 5 CONCLUSIONS

In this work we proposed TriGAN, an MSDA framework which is based on data-generation from multiple source domains using a single generator.

The underlying principle of our approach to to obtain domain-style invariant representations in order to simplify the generation process.

Specifically, our generator progressively removes style and domain specific statistics from the source images and then re-projects the so obtained invariant representation onto the desired target domain and styles.

We obtained state-of-the-art results on two MSDA datasets, showing the potentiality of our approach.

We performed a detailed ablation study which shows the importance of each component of the proposed method.

Some sample translations of our G are shown in Fig. 3 .

For example, in Fig. 3 (a) when the SVHN digit "six" with side-digits is translated to MNIST-M the cDWT blocks re-projects it to MNIST-M domain (i.e., single digit without side-digits) and the AdaIWT block applies the instance-specific style of the digit "three" (i.e., blue digit with red background) to yield a blue "six" with red background.

Similar trends are also observed in Fig. 3 (b) .

Adaptive Instance Whitening (AdaIWT) blocks.

The AdaIWT blocks are analogous to the IWT blocks except from the IWT which is replaced by the AdaIWT.

The AdaIWT block is a sequence: U psampling m×m − Convolution k×k − AdaIW T − ReLU , where m = 2 and k = 3.

AdaIWT also takes as input the coloring parameters (Γ, β) (See Sec. 3.2.3) and Fig. 4 (b) ).

Two AdaIWT blocks are consecutively used in D. The last AdaIWT block is followed by a Convolution 5×5 layer.

Style Path.

The Style Path is composed of: (Fig. 4 (c) ).

The output of the Style Path is (β 1 Γ 1 ) and (β 2 Γ 2 ), which are input to the second and the first AdaIWT blocks, respectively (see Fig. 4 (b) ).

The M LP is composed of five fully-connected layers with 256, 128, 128, 256 neurons, with the last fully-connected layer having a number of neurons equal to the cardinality of the coloring parameters (β Γ).

Domain Whitening Transform (DWT) blocks.

The schematic representation of a DWT block is shown in Fig. 5 (a) .

For the DWT blocks we adopt a residual-like structure He et al. (2016) :

We also add identity shortcuts in the DWT residual blocks to aid the training process.

Conditional Domain Whitening Transform (cDWT) blocks.

The proposed cDWT blocks are schematically shown in Fig. 5 (b) .

Similarly to a DWT block, a cDWT block contains the following layers: cDW T − ReLU − Convolution 3×3 − cDW T − ReLU − Convolution 3×3 .

Identity shortcuts are also used in the cDWT residual blocks.

All the above blocks are assembled to construct G, as shown in Fig. 6 .

Specifically, G contains two IWT blocks, one DWT block, one cDWT block and two AdaIWT blocks.

It also contains the Style Path and 2 Convolution 5×5 (one before the first IWT block and another after the last AdaIWT block), which is omitted in Fig. 6 for the sake of clarity.

{Γ 1 , β 1 , Γ 2 , β 2 } are computed using the Style Path.

For the discriminator D P architecture we use a Projection Discriminator (Miyato & Koyama (2018) ).

In D P we use projection shortcuts instead of identity shortcuts.

In Fig 7 we schematically show a discriminator block.

D P is composed of 2 such blocks.

We use spectral normalization (Miyato & Koyama (2018) ) in D P .

Since, our proposed TriGAN has a generic framework and can handle N -way domain translations, we also conduct experiments for Single-Source UDA scenario where N = 2 and the source domain is grayscale MNIST.

We consider the following UDA settings with the digits dataset:

C.1 DATASETS MNIST → USPS.

The MNIST dataset contains grayscale images of handwritten digits 0 to 9.

The pixel resolution of MNIST digits is 28 × 28.

The USPS contains similar grayscale handwritten digits except the resolution is 16 × 16.

We up-sample images from both domains to 32 × 32 during (Liu & Tuzel (2016)) 91.2 62.0 -ADDA 89.4 --PixelDA (Bousmalis et al. (2017)) 95.9 98.2 -UNIT 95.9 --SBADA-GAN (Russo et al. (2018))

97.6 99.4 61.1 GenToAdapt (Sankaranarayanan et al. (2018)) 92.5 -36.4 CyCADA 94.8 --I2I Adapt (Murez et al. (2018)) 92.1 --TriGAN (Ours) 98.0 95.7 66.3 Table 5 : Classification Accuracy (%) of GAN-based methods on the Single-source UDA setting for Digits Recognition.

The best number is in bold and the second best is underlined.

In this section we compare our proposed TriGAN with GAN-based state-of-the-art methods, both with adversarial learning based approaches and reconstruction-based approaches.

Tab.

5 reports the performance of our TriGAN alongside the results obtained from the following baselines: Domain Adversarial Neural Network (Ganin et al. (2016) ) (DANN), Coupled generative adversarial networks (Liu & Tuzel (2016) ) (CoGAN), Adversarial discriminative domain adaptation ) (ADDA), Pixel-level domain adaptation (Bousmalis et al. (2017) ) (PixelDA), Unsupervised image-to-image translation networks ) (UNIT), Symmetric bi-directional adaptive gan (Russo et al. (2018) ) (SBADA-GAN), Generate to adapt (Sankaranarayanan et al. (2018) ) (GenToAdapt), Cycle-consistent adversarial domain adaptation ) (CyCADA) and Image to image translation for domain adaptation (Murez et al. (2018) ) (I2I Adapt).

As can be seen from Tab.

5 TriGAN does better in two out of three adaptation settings.

It is only worse in the MNIST → MNIST-M setting where it is the third best.

It is to be noted that TriGAN does significantly well in MNIST → SVHN adaptation which is particularly considered as a hard setting.

TriGAN is 5.2% better than the second best method SBADA-GAN for MNIST → SVHN.

@highlight

In this paper we propose generative method for multisource domain adaptation based on decomposition of content, style and domain factors.