Unpaired image-to-image translation among category domains has achieved remarkable success in past decades.

Recent studies mainly focus on two challenges.

For one thing, such translation is inherently multimodal due to variations of domain-specific information (e.g., the domain of house cat has multiple fine-grained subcategories).

For another, existing multimodal approaches have limitations in handling more than two domains, i.e. they have to independently build one model for every pair of domains.

To address these problems, we propose the Hierarchical Image-to-image Translation (HIT) method which jointly formulates the multimodal and multi-domain problem in a semantic hierarchy structure, and can further control the uncertainty of multimodal.

Specifically, we regard the domain-specific variations as the result of the multi-granularity property of domains, and one can control the granularity of the multimodal translation by dividing a domain with large variations into multiple subdomains which capture local and fine-grained variations.

With the assumption of Gaussian prior, variations of domains are modeled in a common space such that translations can further be done among multiple domains within one model.

To learn such complicated space, we propose to leverage the inclusion relation among domains to constrain distributions of parent and children to be nested.

Experiments on several datasets validate the promising results and competitive performance against state-of-the-arts.

Image-to-image translation is the process of mapping images from one domain to another, during which changing the domain-specific aspect and preserving the domain-irrelevant information.

It has wide applications in computer vision and computer graphics Isola et al. (2017) ; Ledig et al. (2017) ; Zhu et al. (2017a) ; Liu et al. (2017) ; such as mapping photographs to edges/segments, colorization, super-resolution, inpainting, attribute and category transfer, style transfer, etc.

In this work, we focus on the task of attribute and category transfer, i.e. a set of images sharing the same attribute or category label is defined as a domain 1 .

Such task has achieved significant development and impressive results in terms of image quality in recent years, benefiting from the improvement of generative adversarial nets (GANs) Goodfellow et al. (2014) ; Mirza & Osindero (2014) .

Representative methods include pix2pix Isola et al. (2017) , UNIT Liu et al. (2017) , CycleGAN Zhu et al. (2017a) , DiscoGAN Kim et al. (2017) , DualGAN Kim et al. (2017) and DTN Taigman et al. (2017) .

More recently the study of this task mainly focus on two challenges.

The first is the ability of involving translation among several domains into one model.

It is quite a practical need for users.

Using most methods, we have to train a separate model for each pair of domains, which is obviously inefficient.

To deal with such problem, StarGAN Choi et al. (2018) leverages one generator to transform an image to any domain by taking both the image and the target domain label as conditional input supervised by an auxiliary domain classifier.

Another challenge is the multimodal problem, which is early addressed by BicycleGAN Zhu et al. (2017b) .

Most techniques including aforementioned StarGAN can only give a single determinate output in target domain given an image from source domain.

However, for many translation task, the mapping is naturally multimodal.

As shown in Fig.1 , a cat could have many possible appearances such as being a Husky, a Samoyed or other dogs when translated to the dog domain.

To address Figure 1: An illustration of a hierarchy structure and the distribution relationship in a 2D space among categories in such hierarchy.

Multi-domain translation is shown in the horizontal direction (blue dashed arrow) while multimodal translation is indicated in the vertical direction (red dashed arrow).

Since one child category is a special case of its parent, in the distribution space it is a conditional distribution of its parent, leading to the nested relationship between them.

this issue, recent works including BicycleGAN Zhu et al. (2017b) , MUNIT Huang et al. (2018) and DRIT Lee et al. (2018) model a continuous and multivariant distribution independently for each domain to represent the variations of domain-specific information, and they have achieved diverse and high-quality results for several two-domain translation tasks.

In this paper , we aim at involving the abilities of both multi-domain and multimodal translation into one model.

As shown in Fig.1 , it is noted that categories have the natural hierarchical relationships.

For instance, the cat, dog and bird are three special children of the animal category since they share some common visual attributes.

Furthermore, in the dog domain, some samples are named as husky and some of them are called samoyed due to the appearance variations of being the dog.

Of course, one can continue to divide the husky to be finer-grained categories based on the variations of certain visual attributes.

Such hierarchical relationships widely exist among categories in real world since it is a natural way for our human to understand objects according to our needs in that time.

We go back to the image translation task, the multi-domain and multimodal issues can be understood from horizontal and vertical views respectively.

From the horizontal view as the blue dashed arrow indicates, multi-domain translation is the transformation in a flat level among categories.

From the vertical view as the red dashed arrow indicates, multimodal translation further considers variations within target category, i.e. the multimodal issue is actually due to the multi-granularity property of categories.

In the extreme case, every instance is a variation mode of the domain-specific information.

Inspired by these observations, we propose a Hierarchical Image-to-image Translation (HIT) method which translates object images among both multiple category domains in a same hierarchy level and their children domains.

To this end, our method models the variations of all domains in forms of multiple continuous and multivariant Gaussian distributions in a common space.

This is different from previous methods which model the same Gaussian distribution for two domains in independent spaces and thus can not work with only one generator.

Due to the hierarchical relationships among domains, distribution of a child domain is the conditional one of its parent domain.

Take such principle into consideration, distributions of domains should be nested between a parent and its children, as a 2D illustration shown in Fig.1 .

To effectively supervise the learning of such distributions space, we further improve the traditional conditional GAN framework to possess the hierarchical discriminability via a hierarchical classifier.

Experiments on several categories and attributes datasets validate the competitive performance of HIT against state-of-the-arts.

Conditional Generative Adversarial Networks.

GAN Goodfellow et al. (2014) is probably one of the most creative frameworks recently for the deep learning community.

It contains a generator and a discriminator.

The generator is trained to fool the discriminator, while the discriminator in turn tries to distinguish the real and generated data.

Various GANs have been proposed to improve the training stability, including better network architectures ; Denton et al. (2015) ; ; Karras et al. (2017) ; Brock et al. (2019) , more reasonable distribution metrics Mao et al. (2017) ; ; Gulrajani et al. (2017) and normalization schemes Miyato et al. (2018) ; Karras et al. (2018) .

With these improvements, GANs have been applied to many conditional tasks Mirza & Osindero (2014) , such as image generation given class labels Odena et al. (2017) , super resolution Ledig et al. (2017) , text2image Reed et al. (2016) , 3D reconstruction from 2D input Wu et al. (2016) and image-to-image translation introduced in the following.

Image-to-image Translation.

Pix2pix Isola et al. (2017) is the first unified framework for the task of image-to-image translation based on conditional GANs, which combines the adversarial loss with a pixel-level L1 loss and thus requires the pairwise supervision information between two domains.

To address this issue, unpaired methods are proposed including UNIT Liu et al. (2017) , DiscoGAN Kim et al. (2017) , DualGAN Yi et al. (2017) and CycleGAN Zhu et al. (2017a) .

UNIT combines the varitional auto-encoder and GAN framework, and proposes to share partial network weights of two domains to learn a common latent space such that corresponding images in two domains can be matched in this space.

DiscoGAN, DualGAN and CycleGAN leverage a cycle consistency loss which enforces that we can re-translate the target image back to the original image.

More recently, works in this area mainly focus on the problems of multi-domain and multimodal.

To deal with translation among several domains in one generator, StarGAN Choi et al. (2018) takes target label and input image as conditions, and uses an auxiliary classifier to classify translated image into its belonged domain.

As for the multimodal issue, BicycleGAN Zhu et al. (2017b) proposes to model continuous and multivariant distributions.

However, BicycleGAN requires input-output paired annotations.

To overcome this problem, MUNIT Huang et al. (2018) and DRIT Lee et al. (2018) adopt a disentangled representation for learning diverse translation results from unpaired training data.

Chen et al. (2019) propose to interpolate the latent codes between input and referred image to realize generation of diverse images.

Different from all aforementioned works, we aim at realizing both multi-domain and multimodal translation in one model using the natural hierarchical relationships among domains defined by category or attribute.

Hierarchy-regularized Learning.

Hierarchical learning is a natural learning manner for human beings and we often describe objects in the world from abstract to detailed according to our needs of the time.

Zhao et al. (2017) propose to use generative models to disentangle the factors from low-level representations to high-level ones that can construct a specific object.

Singh et al. (2019) uses an unsupervised generative framework to hierarchically disentangle the background, object shape and appearance from an image.

In natural language processing, Athiwaratkun & Wilson (2018) propose a probabilistic word embedding method to capture the semantics described by the WordNet hierarchy.

Our method first introduces such semantic hierarchy to learn a both multi-domain and multimodal translation model.

3.1 PROBLEM FORMULATION Let x i ∈ X i be an image from domain i. Our goal is to estimate the conditional probability p(x j |x i ) by learning an image-to-image translation model p(x i→j |x i ), where x i→j is a sample produced by translating x i to domain X j .

Generally speaking, p(x j |x i ) are multimodal due to the intra-domain variations.

To deal with the multimodal problem, similar to , we assume that x i is disentangled by an encoder E into the content part c ∈ C that is shared by all domains (i.e. domain-irrelevant) and the style part s i ∈ S i that is specific to domain X i (i.e. domain-specific).

By modeling S j as a continuous distribution such as a Gaussian N j , x i can be simply translated to domain X j by G(c, s j ) where s j is randomly sampled from N j and G is a decoder.

We further assume G and E are deterministic and mutually inverse, i.e. E = G −1 and G = E −1 .

Besides, we assume that c is a high-dimensional feature map while s i is a low-dimensional vector such that the complex spatial structure of objects can be preserved and the style parts could focus more on the relatively small scale but discriminative domain-specific information.

Different from , we aim to translate not only between two domains but among multiple ones.

To this end, we need to model Gaussians of styles for all domains in a common space (not independently in two spaces like ) such that the single decoder G could generate target image based on which Gaussian is sampled.

In the multi-domain and multimodal Figure 2: Overview of the whole framework of the proposed method, which mainly consists of five modules: an encoder, a domain distributions modeling module, a decoder, a discriminator and a hierarchical classifier.

Given images from different categories, the encoder extracts domain-irrelevant and domain-specific features respectively from the content and style branches.

Then the decoder takes them as input to reconstruct the inputs supervised by the reconstruction losses.

To realize the multimodal and multi-domain translation, domain distributions are modeled in a common space based on the semantic hierarchy structure and elaborately designed nested loss.

Combining the domain-irrelevant features and sampled styles from any distribution, the decoder could translate them to the target domain, guided by the adversarial loss and hierarchical classification loss.

settings, it is noted that categories have the hierarchical relationships.

As we introduced in Fig.1 , multi-domain translation is in the horizontal direction among categories in a particular hierarchy level, and multimodal translation is in the vertical direction since samples can be further divided into multiple child modes.

Therefore, distribution of a parent domain covers several conditional distributions, leading to the nested relationship.

In this paper, we model all category domains in a given hierarchy structure as nested Gaussian distributions in a common space, realizing Hierarchical Image-to-image Translation (HIT) between any two domains.

In such settings, N .

The framework is trained with adversarial loss that ensures the translated images approximate the manifold of real images, hierarchical cross-entropy loss that makes the generation conditioned on the sampled domain, nested loss that constrains distributions of domains to satisfy their hierarchical relationships, as well as bidirectional reconstruction losses that ensure enough and meaningful information be encoded.

In math, the relation between a parent node u and a child node v in the hierarchy is called partial order relation Vendrov et al. (2016) , defined as v u. In the application of taxonomy, for concept u and v, v u means every instance of category v is also an instance of category u, but not vise versa.

We call such partial order on probability densities as the notion of nested (encapsulation called by Athiwaratkun & Wilson (2018) ).

Let g and f be the densities of u and v respectively, if v u, then f g, i.e. f is nested in g. Quantitatively measuring the loss violate the nested relation between f and g is not easy.

According to the definition of partial order, strictly measuring that can be done as:

where {x : f (x) > η} is the set where f is greater than a nonnegative threshold η.

Threshold η indicates the nested degree required by us.

Small value of η means high requirement for the overlap between f and g to satisfy f g. Eqn.

(1) describes how many regions with densities greater than η of f are not nested in those of g.

Eqn.

(1) is difficult to be computed for most distributions including Gaussians.

Inspired by the work in word embedding Athiwaratkun & Wilson (2018) , we turn to use a thresholded divergence:

where D(·||·) is a divergence measure between densities, we use the KL divergence considering its simple formulation for Gaussians.

Such loss is a soft measure of violation of the nested relation.

If f = g, then D(f ||g) = 0.

In case of f g, D(f ||g) would be positive but smaller than a threshold α.

As for the effectiveness of using α, please make a reference to Athiwaratkun & Wilson (2018) .

To learn the nested distributions for domains in the hierarchy shown in Fig.2 , the penalty described by Eqn.(2) between a positive pair of distributions (N i N j ) should be minimized, while that between a negative pair (N i N j ) should be greater than a margin m:

where P and N denote the numbers of positive and negative pairs respectively.

Apart from the proposed nested loss in Eqn.(3), our HIT is equipped with an adversarial loss and a hierarchical classification loss to distinguish which domain the generated images belong to, and two general reconstruction losses applied on both images and features.

Adversarial loss.

GAN is an effective objective to match the generated images to the real data manifold.

The discriminator D tries to classify natural images as real and distinguish generated ones as fake, while the generator G learns to improve image quality to fool D, defined as:

Hierarchical classification loss.

Similar to StarGAN Choi et al. (2018) , we introduce an auxiliary classifier D cls on top of D and impose the domain classification loss when optimizing G and D, i.e. using real images to train D cls and generated ones to optimize G with such classification loss.

Differently, our classifier is hierarchical.

In general, the deeper of categories in the hierarchy, the more difficult to distinguish.

To alleviate such problem, the loss is cumulative, i.e. classification loss of l-th level is the summation of losses of all levels above l-th with more than two categories.

where y k j is the label of domain X j in k-th level.

Bidirectional reconstruction loss.

To ensure meaningful information encoded and inverse between G and E, we encourage reconstruction of both images and latent features.

-Image reconstruction loss:

-Feature reconstruction loss:

Full objectives.

To learn E, G and N l j , we need to optimize the following terms:

where λ 1 , λ 2 and λ 3 are loss weights of different terms.

D is updated with the following losses: Karras et al. (2018) and translation works .

As shown in Fig.2 , we add a distribution modeling module where a pair of mean vector and diagonal covariance matrix of Gaussian for each domain is parameterized to learn.

More training details are given in the Appendix.

Style adversarial loss.

Eqn.

(4) and Eqn.(5) match the generated images to the distribution of a target domain.

Such loss functions can also be applied on the encoded style codes, i.e. matching s i (act as fake data) of input images to domain Gaussians N l i (act as real data) they belong to.

By doing so, it is found that the performance of style transfer between a pair of real images would become better.

However, such loss would lead to the training collapse on small scale datasets.

Therefore, it is recommended to equip it to our framework on datasets with enough training data.

We conduct experiments on hierarchical annotated data from CelebA Liu et al. (2015) , ImageNet Russakovsky et al. (2015) and ShapeNet Chang et al. (2015) .

Typical examples are shown in Fig.8, Fig.9 and Fig.10 in Appendix.

CelebA provides more than 200K face images with 40 attribute annotations.

Following the official train/test protocol and imitating the category hierarchy, we define a hierarchy based on attribute annotations.

Specifically, all faces are first clustered into male and female and are further classified according to the age and hair color in the next two levels.

Following , we collect images from 3 super domains including house cats, dogs and big cats of ImageNet.

Each super domain contains 4 fine-grained categories, which thus construct in a three-level hierarchy (root is animal).

All images split by official train/test protocol are processed by a pre-trained faster-rcnn head detector and then cropped as the inputs for translation.

ShapeNet is constitutive of 51,300 3D models covering 55 common and 205 finer-grained categories.

12 2D images with different poses are obtained for each 3D model.

A three-level hierarchy of furniture containing different kinds of tables and sofas are defined.

Ratio of train/test split is 4:1.

4.3 EVALUATION METRICS Frankly speaking, quantitatively evaluating the quality of generated images is not easy.

Recent proposed metrics may be fooled by artifacts in some extent.

In this paper, we use the Inception Score (IS) Salimans et al. (2016) and Frchet Inception Distance (FID) Heusel et al. (2017) to evaluate the semantics of generated images, and leverage the Learned Perceptual Image Patch Similarity Zhang et al. (2018) (LPIPS) to measure the semantic diversity of generated visual modes.

4.4 COMPARED BASELINES We mainly compared methods proposed for the objectives of either multi-domain or multimodal translation.

Considering the unpaired training settings, the multi-domain method StarGAN Choi et al. (2018) and multimodal method MUNIT Huang et al. (2018) are compared in this paper.

Since MUNIT needs to train a model for each pair of domains, it is trained for domain pairs of male/female, young/old and black/golden hair on CelebA, house cat/dog, house cat/big cat and big cat/dog on ImageNet, and sofa/table on ShapeNet, respectively.

The average of evaluations on all domain pairs for each dataset is reported.

As for StarGAN, it is trained on CelebA as done in its opened source codes.

Translations among house cat, dog and big cat domains on ImageNet, and between sofa and table domains on ShapeNet are learned for StarGAN.

As comparison, results of our HIT in corresponding domain levels for each dataset are reported.

4.5 RESULTS Table.

1 shows the quantitative comparisons of the proposed HIT with the baselines.

Fig.3 shows qualitative results on CelebA. It is observed that StarGAN achieves outstanding image quality espe- cially on the fine-grained translations among attribute domains, while the advantages of multimodal methods are generating multiple translations with intra-domain mode variations at the cost of image quality.

The image quality of MUNIT is not satisfactory on CelebA both in quantitatively in Table.

1 and in qualitatively in Fig.3 .

The reasons for this may be that using only the adversarial learning to find fine-grained attribute differences between domains is not stable while multi-domain classifier is good at such task.

Besides MUNIT obtains the best diversity performance.

It is reasonable as it only involves two domains in one model and equips a triplet of backbone including encoder, decoder and discriminator for each domain.

Our method considers both multimodal and multi-domain translation within only one triplet of such backbone, which has high requirement for capacity of networks.

It performs in trade-off between image quality and diversity.

From Fig.3 , artifacts accompanying the generated faces for MUNIT may overestimate the LPIPS on CelebA. Fig.4 and Fig.5 further shows the qualitative results of our HIT on ImageNet and ShapeNet datasets respectively.

Generally speaking, translation among such categories with large variations is much more challenging than that for face data (several times of increase of the FID in Table.

1 can be found).

Even so, our HIT achieves promising qualitative results.

Besides, using the fixed styles from a particular category distribution (same columns in Fig.4 and Fig.5 ), the generated images indeed have similar styles of that category and dissimilar content appearances (e.g. pose, expression), demonstrating good disentanglement of content and style of images.

Walking in the path towards leaf-level, translated images would have fewer variations with more conditions being specified by the categories in high levels.

In other words, distributions in low levels are local modes of its ancestor domains in high levels, leading to the nested relationship.

Results in Fig.6 validate the learned distributions of styles in different levels are exactly nested.

In the Appendix, we give an experimental parameter-sensitiveness analysis of m and α which constrain the nested relationships among distributions.

In Fig.7(a) , we further study the smoothness of learned distributions.

It is observed one can conduct smooth translation via interpolations between styles from different attribute domains.

Besides, with the help of additional style adversarial loss introduced in Sec.4.1, our method can provide users more controlled translation as done in , i.e. use the styles of referenced real images instead of sampling them from distributions.

Fig.7(b) shows some example results.

We can find that the semantics of gender, age and hair color are all correctly transferred to the input images.

In this paper we propose the Hierarchical Image-to-image Translation (HIT) method which incorporates multi-domain and multimodal translation into one model.

Experiments on three datasets especially on CelebA show that the proposed method can well achieve such granularity controlled translation objectives, i.e. the variation modes of outputs can be specified owe to the nested distributions.

However, current work has a limitation, i.e. the assumption of single Gaussian for each category domain.

On one hand, though Gaussian distribution prior is a good approximation for many data, it may not be applicable when scale of available training data is small but variations within domain are large such as the used hierarchical data on ImageNet and ShapeNet in this paper.

On the other hand, the parent distributions should be mixture of Gaussians given multiple single Gaussians of its children.

This issue would lead to sparse sampling around the centers of parent distributions and poor nested results if samples are not enough to fulfill the whole space.

We have made efforts to the idea of mixture of Gaussians and found that it is hard to compute the KL divergence between two mixture of Gaussians which does not have an analytical solution.

Besides, the re-parameterize trick for distribution sampling during SGD optimization can not be transferred to the case of mixture of Gaussians.

A better assumption to realize the nested relationships among parent-children distributions is a promising direction for our future research.

We use the Adam optimizer with β 1 = 0.5, β 2 = 0.999, and initial learning rate of 0.0001.

We train HIT on all datasets for 300K iterations and half decay the learning rate every 100K iterations.

We set batch size to 16.

The loss weights λ 1 , λ 2 andλ 3 in Eqn.(8) are set as 1, 10 and 1 respectively.

α and m in Eqn.(3) are empirically set as 50, 200 respectively.

Random mirroring is applied during training.

In this section, Fig.8 , Fig.9 and Fig.10 provide leaf-level examples for better understanding the nested relationships among categories in different hierarchy levels.

Take the CelebA for example, the root category face has two children distinguished by gender attribute.

For each of the two super categories, it includes two finer granular children which are further divided by the age attribute (young/old).

Finally, in the leaf-level, each local branch are classified according to their hair colors, i.e. black, golden and brown hair.

Within each leaf-level category, samples mainly contain intraclass variations caused by identities, expressions, poses, etc.

The impacts of hyper-parameters in the nested distribution learning on word embedding task have been studied in Athiwaratkun & Wilson (2018) .

In this section, we further make an analysis of them in current image generation task.

Fig.11 and Fig.12 show the impacts of hyper-parameters m and α in the nested loss of Eqn.(3).

It is observed that distribution margin m has larger impact than nested threshold α.

With too large settings of m, distributions which do not have nested relationship would be pushed far away, leading to sparse space.

Sampling in such space would make the learning of generator quite difficult.

In contrast, with too small settings of m, the discriminabilities of distributions may be poor.

Therefore, a trade-off value 200 is set for m in this paper.

As for nested threshold α, a relative small or large value performs well in terms of the image quality metric.

However, in theory, large value setting of α would relax the nested constraint too much, result in small overlap between parent and children distributions.

Therefore, we recommend to set α in the left half axis of α.

When α is set as 0, it means the parent and children distributions are all overlapped, which is too strict to learn.

Finally, we set α as 50, and the ratio of 1:4 between α and m is consistent with the observation in Athiwaratkun & Wilson (2018) .

<|TLDR|>

@highlight

Granularity controled multi-domain and multimodal image to image translation method