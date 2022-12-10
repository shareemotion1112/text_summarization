Style transfer usually refers to the task of applying color and texture information from a specific style image to a given content image while preserving the structure of the latter.

Here we tackle the more generic problem of semantic style transfer: given two unpaired collections of images, we aim to learn a mapping between the corpus-level style of each collection, while preserving semantic content shared across the two domains.

We introduce XGAN ("Cross-GAN"), a dual adversarial autoencoder, which captures a shared representation of the common domain semantic content in an unsupervised way, while jointly learning the domain-to-domain image translations in both directions.

We exploit ideas from the domain adaptation literature and define a semantic consistency loss which encourages the model to preserve semantics in the learned embedding space.

We report promising qualitative results for the task of face-to-cartoon translation.

The cartoon dataset we collected for this purpose will also be released as a new benchmark for semantic style transfer.

Image-to-image translation -learning to map images from one domain to another -covers several classical computer vision tasks such as style transfer (rendering an image in the style of a given input BID3 ), colorization (mapping grayscale images to color images (Zhang et al., 2016) ), super-resolution (increasing the resolution of an input image BID9 ), or semantic segmentation (inferring pixelwise semantic labeling of a scene BID14 ).

In many cases, one can rely on supervision in the form of labels or paired samples.

This assumption holds for instance for colorization, where ground-truth pairs are easily obtained by generating grayscale images from colored inputs.

Figure 1: On the left, we depict a high-level motivational example for semantic style transfer, the task of adapting an image to the visual appearance of an other domain without altering its semantic content.

The proposed XGAN applied on the face-to-cartoon task preserves important face semantics such as hair style or face shape (right).In this work, we consider the task of semantic style transfer: learning to map an image from one domain into the style of another domain without altering its semantic content (see Figure 1) .

In that sense, our goal is akin to style transfer: We aim to transfer style while keeping content consistent.

The key differences with traditional techniques are that (i) we work with image collections instead of having a single style image, and (ii) we aim to retain higher-level semantic content in the feature space rather than pixel-level structure.

In particular, we experiment on the task of translating faces to cartoons while preserving their various facial attributes (hair color, eye color, etc.).

Note that without loss of generality, a photo of a face can be mapped to many valid cartoons, and vice versa.

Semantic style transfer is therefore a many-to-many mapping problem, for which obtaining labeled examples is ambiguous and costly.

Although this paper specifically focuses on the face-to-cartoon setting, many other examples fall under this category: mapping landscape pictures to paintings (where the different scene objects and their composition describe the input semantics), transforming sketches to images, or even cross-domain tasks such as generating images from text.

In this setting, we only rely on two unlabeled training image collections or corpora, one for each domain, with no known image pairings across domains.

Hence, we are faced with a double domain shift, first in terms of global domain appearance, and second in terms of the content distribution of the two collections.

Recent work BID6 Zhu et al., 2017; Yi et al., 2017; BID1 report good performance using GAN-based models for unsupervised image-to-image translation when the two input domains share similar pixel-level structure (e.g., horses and zebras) but fail for more general transformations (e.g., dogs and cats).

Perhaps the best known recent example is CycleGAN (Zhu et al., 2017) .

Given two image domains D 1 and D 2 , the model is trained with a pixel-level cycleconsistency loss which ensures that the mapping g 1→2 from D 1 to D 2 followed by its inverse, g 2→1 , yields the identity function; i.e., g 1→2 • g 2→1 = id.

However, we argue that such a pixel-level constraint is not sufficient in our case; the category of transformations we are interested in requires a constraint in semantic space even though the transformation occurs in the pixel space.

To this end, we propose XGAN ("Cross-GAN"), a dual adversarial autoencoder which learns a shared semantic representation of the two input domains in an unsupervised way, while jointly learning both domain-to-domain translations.

In other words, the domain-to-domain translation g 1→2 consists of an encoder e 1 taking inputs in D 1 , followed by a decoder d 2 with outputs in D 2 (and likewise for g 2→1 ) such that e 1 and e 2 , as well as d 1 and d 2 , are partially shared.

The main novelty lies in how we constrain the shared embedding using techniques from the domain adaptation literature, as well as a novel semantic consistency loss.

The latter ensures that the domain-to-domain translations preserve the semantic representation, i.e., that e 1 ≈ e 2 •g 1→2 and e 2 ≈ e 1 •g 2→1 .

Therefore, it acts as a form of self-supervision which alleviates the need for paired examples and preserves semantic featurelevel information rather than pixel-level content.

In the following section, we review relevant recent work before discussing the XGAN model in more detail in Section 3.

In Section 4, we introduce CARTOONSET, our dataset of cartoon faces for research on semantic style transfer, which we are currently in the process of making publicly available.

Finally, in Section 5 we report experimental results of XGAN on the face-to-cartoon task, and discuss various ablation experiments.

Recent literature suggests two main directions for tackling the semantic style transfer task: traditional style transfer and pixel-level domain adaptation.

The first approach is inadequate as it only transfers texture information from a single style image, and therefore does not capture the style of an entire corpus.

The latter category also fails in practice as it assumes pixel-level similarity which does not allow for significant structural change of the input.

Instead, we draw inspiration from the domain adaptation and feature-level image-to-image translation literature.

Style Transfer.

Style transfer traditionally refers to the task of transferring the texture of a specific style image while preserving the pixel-level structure of an input content image BID3 BID5 .

Recently, BID10 BID11 proposed to compare the style and generated image via a dense local patch-based matching approach in the feature space, as opposed to global feature matching, allowing for transformations between visually dissimilar domains.

Still, these models only perform image-specific transfer rather than learning a global corpus-level style, and do not provide a meaningful joint semantic domain representation.

Domain adaptation.

XGAN relies on learning a shared semantic representation of both domains in an unsupervised setting.

For this purpose, we make use of the domain-adversarial training scheme BID2 .

Moreover, recent domain adaptation work BID0 Shrivastava et al., 2017; BID1 can be framed as semantic style transfer as they tackle the problem of mapping synthetic images, easy to generate, to natural images, which are more difficult to obtain.

The generated samples are then used to train a model that can be applied to natural images.

Contrary to our work however, they only consider pixel-level transformations.

Image-to-Image translation.

Recent work BID6 Zhu et al., 2017; Yi et al., 2017 ) tackle the unsupervised pixel-level image-to-image translation task by learning both cross-domain mappings jointly, each as a separate generative network, via a cycle-consistency loss which ensures that applying each mapping followed by its reverse yields the identity function.

This intuitive form of self-supervision leads to good results for pixel-level transformations, but often fails to capture significant structural changes Zhu et al. (2017) .

In comparison, our proposed semantic consistency loss acts at the feature-level, allowing for more flexible transformations.

Orthogonal to this work is UNIT BID13 .

While also trained with pixel-level cycle-consistency, it consists of a coupled VAEGAN Larsen et al. (2015) ; BID12 with a shared embedding bottleneck, similar to XGAN.

However, UNIT assumes that sharing high-level layers in the architecture is sufficient to learn a joint representation of both domains, while XGAN's objective explicitly introduces the semantic consistency component.

The Domain Transfer Network (DTN) (Taigman et al., 2016; Wolf et al., 2017 ) is closest to our work.

DTN is a single autoencoder trained to map images from a source to a target domain with self-supervised semantic-consistency feedback.

It was also successfully applied to the problem of feature-level image-to-image translation, in particular to the face-to-cartoon problem.

Contrary to XGAN however, the DTN encoder is pretrained and fixed, and is assumed to produce meaningful embeddings for both the face and the cartoon domains.

This assumption is very restrictive, as offthe-shelf models pretrained on natural images do not necessarily generalize to other domains.

In fact, while the reported results are convincing, we show in Section 5 that using a fixed encoder does not generalize well in the presence of large domain shift between the two input domains.

Let D 1 and D 2 be two domains that differ in terms of visual appearance but share common semantic content.

Note that while it is easier to think of domain semantics as a high-level notion, as for instance semantic attributes, we do not require such annotations in practice, but instead consider learning a feature-level representation that automatically captures these semantics without supervision.

Our goal is thus to learn in an unsupervised fashion, i.e., without paired examples, a joint domain-invariant embedding that is semantically-consistent and meaningful for both domains; i.e., semantically similar inputs in both domains will be embedded nearby in the learned semantic space.

Architecture-wise, XGAN is a dual autoencoder on domains D 1 and D 2 (Figure 2(A) ).

We denote by e 1 the encoder and by d 1 the decoder for domain D 1 ; likewise e 2 and d 2 for D 2 .

For simplicity, we also denote by DISPLAYFORM0 The training objective can be decomposed into five main components: the reconstruction loss, L rec , encourages the learned embedding to encode meaningful knowledge for each domain; the domain-adversarial loss, L dann , pushes embeddings from D 1 and D 2 to lie in the same subspace, bridging the domain gap at the semantic level; the semantic consistency loss, L sem , ensures that input semantics are preserved after domain translation; L gan is a simple generative adversarial (GAN) objective, encouraging the model to generate more realistic samples, and finally, L teach is an optional teacher loss that distills prior knowledge from a fixed pretrained teacher embedding, when available.

The total loss function is defined as: DISPLAYFORM1 where the ω hyper-parameters control the contributions from each of the individual objectives.

An overview of the model is given in Figure 2 , and we discuss each objective in more detail in the rest of this section.

Reconstruction loss.

L rec encourages the model to encode enough information on each domain for the input to be reconstructed by the autoencoder.

More specifically L rec = L rec,1 + L rec,2 is the sum of reconstruction losses for each domain.

DISPLAYFORM2 Domain-adversarial loss.

L dann is the domain-adversarial loss between D 1 and D 2 , as introduced in BID2 .

It encourages the embeddings learned by e 1 and e 2 to lie in the same subspace.

In particular, it guarantees the soundness of the cross-domain transformations g 1→2 and g 2→1 .

More formally, this is achieved by training a binary classifier, c dann , on top of the embedding layer to categorize encoded images from both domains as coming from either D 1 or D 2 (see Figure 2 (B1)).

c dann is trained to maximize its classification accuracy L dann while the encoders e 1 and e 2 simultaneously strive to minimize it, i.e., to confuse the domain-adversarial classifier.

Denoting model parameters by θ and a classification loss function by (e.g., cross-entropy), we have: DISPLAYFORM3 Semantic consistency loss, L sem .

Our key contribution is a semantic consistency feedback loop that acts as self-supervision for the cross-domain translations g 1→2 and g 2→1 .

It reinforces the action of the domain-adversarial loss L dann by mapping the embedding of an input image and the embedding of its translated counterpart to the same point.

Intuitively, we want the semantics of input x ∈ D 1 to be preserved when translated to the other domain, g 1→2 (x) ∈ D 2 , and similarly for the reverse mapping.

However this consistency property is hard to assess at the pixel-level as we do not have paired data and pixel-level metrics are suboptimal for image comparison.

Instead, we introduce a feature-level semantic consistency loss, which encourages the network to preserve the learned embedding during domain translation.

Formally, DISPLAYFORM4 , where: DISPLAYFORM5 L sem,2→1 is defined in the same way for the transformation from D 2 to D 1 .GAN objective, L gan .

Although the key aim of XGAN is to learn a joint meaningful and semantically consistent embedding, we find that generating realistic image transformations has a crucial positive effect as the produced samples are fed back through the encoders when computing the semantic consistency loss: Making the transformed distribution p(g 2→1 (D 2 )) as close as possible to the original domain p(D 1 ) ensures that the encoder e 1 does not have to cope with an additional domain shift.

Therefore, with the purpose of improving sample quality, we define DISPLAYFORM6 , where L gan,1→2 is a state-of-the-art GAN objective BID4 where the generator g 1→2 is paired against the discriminator D 1→2 (and likewise for g 2→1 and D 2→1 ).

The models are trained jointly in an adversarial scheme where D 1→2 strives to distinguish generated samples from real samples in D 2 , while the generator aims to produce samples that confuse the discriminator, i.e., DISPLAYFORM7 DISPLAYFORM8 Once again L gan,2→1 is the symmetric version for the transformation from D 2 to D 1 .Teacher loss, L teach .

We introduce an optional component to easily incorporate prior knowledge in the model when available, i.e., when working in a semi-supervised framework.

L teach encourages the learned embeddings to lie in a region of the subspace defined by the output of the representation layer of a teacher network T .

In other words, it distills knowledge from a pretrained teacher and constrains the embeddings to a more meaningful subregion (relative to the task on which T was trained), which can be seen as a form of regularization of the learned embedding.

L teach is asymmetric by definition.

It should not be used for both domains simultaneously as each term would potentially push the learned embedding in two different directions.

Assuming it is applied to domain D 1 , leads to the following definition: DISPLAYFORM9 , where · is a distance between vectors.

We use a simple mirrored convolutional architecture for the autoencoder.

It consists of 5 convolutional blocks for each encoder, the two last ones being shared across domains, and likewise for the decoders (5 deconvolutional blocks with the two first ones shared).

This encourages the model to learn shared representations at different levels of the architecture rather than only in the middle layer.

For the teacher network, we use the highest convolutional layer of FaceNet (Schroff et al., 2015) , a state-of-the-art model pretrained for the task of face recognition.

Note that FaceNet was trained on natural images only, i.e., it does not contain any prior knowledge of the cartoon domain.

A more detailed description is given in Appendix 7.1.The XGAN training objective is obtained by minimizing Equation FORMULA1 .

In particular, the two adversarial losses (L gan and L dann ) leads to minmax optimization problems that require careful optimization.

For the GAN loss L gan , we use a standard adversarial training scheme BID4 .

Note that in order to ease training, we only use one of the discriminators in practice, namely D 1→2 which corresponds to the face-to-cartoon path, our target application.

We first update the parameters of the generators g 1→2 and g 2→1 in one step.

We then keep these fixed and update the parameters for the discriminator D 1→2 .

Finally, we train the model by iterating this alternating process.

The adversarial training scheme for L dann can be easily implemented in practice by connecting the classifier c dann and the embedding layer via a gradient reversal layer BID2 : The feed-forward pass is unaffected, however the gradient is backpropagated to the encoders with a sign-inversion representing the minmax alternation.

This update is performed in the same step as for the generator parameters.

Finally, we use ADAM optimization BID7 with an initial learning rate of 0.0001 to train the model.

Although previous work has tackled the task of transforming frontal faces to a specific cartoon style, there is currently no such dataset publicly available.

For this purpose, we introduce a new dataset, CartoonSet, which we will release publicly to further aid research on this topic.

Each cartoon face is composed of 16 components including 12 facial attributes (e.g., facial hair, eye shape, etc) and 4 color attributes (such as skin or hair color) which are chosen from a discrete set of RGB values.

The number of options per attribute category ranges from 3, for short/medium/long chin length, to 111, for the largest category, hairstyle.

Each of these components and their variation were drawn by the same artist, resulting in approximately 250 cartoon components artworks and 10 8 possible combinations.

Furthermore, the artwork components are divided into a fixed set of layers that define a Z-ordering for rendering.

For instance, face shape is defined on a layer below eyes and glasses, so that the artworks are rendered in the correct order.

Hair style is a more complex case and needs to be defined on two layers, one behind the face and one in front.

There are 8 total layers: hair back, face, hair front, eyes, eyebrows, mouth, facial hair, and glasses.

The mapping from attribute to artwork is also defined by the artist such that any random selection of attributes produces a visually appealing cartoon without any misaligned artwork; this sometimes involves handling interaction between attributes.

For example, the proper way to display a "short beard" changes for different face shapes, which required the artist to create a "short beard" artwork for each face shape.

We create the CartoonSet dataset from arbitrary cartoon faces by randomly sampling a value for each attribute.

The corresponding artworks are rendered back-to-front.

We then filter out unusual hair colors (pink, green etc) or unrealistic attribute combinations, which results in a final dataset of approximately 9, 000 cartoons.

In particular, the filtering step guarantees that the dataset only contains realistic cartoons, while being completely unrelated to the source dataset.

We experimentally evaluate our XGAN model on semantic style transfer; more specifically, on the task of converting images of frontal faces (source domain) to images of cartoon avatars (target domain) given an unpaired collection of such samples in each domain.

Our source domain is composed of real-world frontal-face images from the VGG-Face dataset BID15 .

In particular, we use an image collection consisting of 18,054 uncropped celebrity frontal face pictures.

As a preprocessing step, we align the faces based on eyes and mouth location and remove the background.

The target domain is the cartoon style we introduced in Section 4.

The corresponding training image collection consists of 9,000 cartoon images that we center-align by localizing the center of the irises, the center of the mouth, and tip of the nose.

Finally, we randomly select and take out 20% of the images from each dataset for testing purposes, and use the remaining 80% for training.

For our experiments we also resize all images to 64×64.

As shown in FIG0 , the two domains vary significantly in appearance.

In particular, cartoon faces are rather simplistic compared to real faces, and do not display as much variety (e.g., noses or eyebrows only have a few shape options).

Furthermore, we observe a major content distribution shift between the two domains due to the way we collected the data: for instance, certain hair color shades (e.g., bright red, gray) are over-represented in the cartoon domain compared to real faces.

Similarly, the cartoon dataset contains many samples with eyeglasses while the source dataset only has a few.

Baseline comparison.

Our primary evaluation result is a qualitative comparison between the Domain Transfer Network (DTN) (Taigman et al., 2016) and XGAN on the semantic style transfer problem outlined above.

To the best of our knowledge, DTN is the current state of the art for semantic style transfer given unpaired image corpora from two domains with significant visual shift.

In particular, DTN was also applied to the task of transferring face pictures to cartoons (bitmojis) in the original paper 2 .

See Section 2 for a more detailed introduction.

FIG2 shows the performance of both DTN and XGAN applied to random VGG-Face samples from the test set to produce cartoon versions of each sample.

For both models, we present random samples produced with the best set of hyperparameters we found.

Evaluation metrics for style transfer are still an active research topic with no good solution yet.

Hence we choose optimal hyperparameters by manually evaluating the quality of resulting samples, focusing on accurate transfer of semantic attributes, similarity of the resulting sample to the target domain, and crispness of samples.

It is clear from FIG2 that DTN fails to capture the transformation function that semantically stylizes frontal faces to cartoons from our target domain.

In contrast, XGAN is able to produce sensible cartoons both in terms of the style domain -the resulting cartoons look crisp and respect the specific CartoonSet style -and in terms of semantic similarity to the input samples from VGGFace.

There are some failure cases such as hair or skin color mismatch, which emerge from the weakly supervised nature of the task and the significant content shift between the two domains (e.g., red hair is over-represented in the target cartoon dataset).

We also report selected XGAN samples that we think best illustrate its semantic consistency abilities in FIG3 .

Finally, additional random samples for both cross-domain mappings are available in Appendix 7.3.We believe the failure of DTN is primarily due to its assumption of a fixed joint encoder for both domains.

Although the decoder learns to reconstruct inputs from the target domain almost perfectly, the semantics are not well preserved across domains and the decoder yields samples of poor quality for the domain transfer.

In fact, FaceNet was originally trained on real faces inputs, hence there is no guarantee it can produce a meaningful representation for CartoonSet samples.

In contrast to our dataset, the target bitmoji domain in (Taigman et al., 2016) is visually closer to real faces, as bitmojis are more realistic and customizable than the cartoon style domain we introduce here.

This might explain the good reported performance even with a fixed encoder.

Our experiments suggest that using a fixed encoder is a very restrictive assumption that does not adapt well to new scenarios.

We also report results from a finetuned DTN in Appendix 7.2 and 7.3, which yields samples of better quality than the original DTN.

However, this setup is very sensitive to training hyperparameters and prone to mode collapse.

Ablation study.

We conduct a number of insightful ablation experiments on XGAN.

We first consider training only with the reconstruction loss L rec and domain-adversarial loss L dann .

In fact these form the core domain adaptation component in XGAN and, as we will show, are already able to capture basic semantic knowledge across domains in practice.

Secondly we experiment with the semantic consistency loss and teacher loss.

We show that both have a constraining effect on the embedding space which contributes to improving the sample consistency.

We also show in Appendix 7.4.1 that the GAN loss, even though it makes training more complex, is necessary for producing samples of good quality and cannot be replaced with simpler image smoothness objectives.

We first experiment on XGAN with only the reconstruction and domain-adversarial losses active.

This component prompts the model to (i) encode enough information for each decoder to correctly reconstruct images from the corresponding domain and (ii) to ensure that the embedding lies in a common subspace for both domains.

In practice in this setting, the model is robust to hyperparameter choice and does not require much tuning to converge to a good regime, i.e., low reconstruction error and around 50% accuracy for the domain-adversarial classifier.

As a result of (ii), applying each decoder to the output of the other domain's encoder yields reasonable cross-domain translations, albeit of low quality (see Figure 7) .

Furthermore, we observe that some simple semantics such as skin tone or gender are overall well preserved by the learned embedding due to the shared autoencoder structure.

For comparison, failure modes occur in extreme cases, e.g., when the model capacity is too small, in which case transferred samples are of poor quality, or when ω dann is too low.

In the latter case, the source and target embeddings are easily distinguishable and the cross-domain translations do not look realistic (see Appendix 7.4).source to target target to source Figure 7 : Test results for XGAN with the reconstruction and domain-adversarial losses only Secondly, we investigate the benefits of adding semantic consistency in XGAN via the following three components: Sharing high-level layers in the autoencoder leads the model to capture common semantics earlier in the architecture.

In general, high-level layers in convolutional neural networks are known to encode semantic information.

We perform a few experiments when sharing only the middle layer in the dual autoencoder.

As expected, the resulting embedding does not capture relevant shared domain semantics.

Second, we use the semantic consistency loss as self-supervision for the learned embedding, ensuring that it is preserved through the cross-domain transformations.

It also reinforces the action of the domain-adversarial loss as it constrains embeddings from the two input domains to lie close to each other.

Finally, the optional teacher loss leads the learned source embedding to lie near the teacher output (in our case, FaceNet's representation layer), which is meaningful for real faces.

It acts in conjunction which the domain-adversarial loss and semantic consistency loss which bring the source and target embedding distributions closer to each other.

In FIG4 we report random test samples for both domain-to-domain translations when ablating the teacher loss and semantic consistency loss respectively.

While it is hard to draw conclusions from qualitative results, it seems that the teacher network has a positive regularization effect on the learned embedding by guiding it to a more reasonable region of the space: Training the model without the teacher loss FIG4 ) yields more distorted samples, especially when the input is an outlier, e.g., person wearing a hat, or cartoons with unusual hairstyles FIG2 ).

Conversely, when the semantic consistency is inactive FIG4 ), the generated samples overall display less variety.

In particular, rare attributes (e.g., unusual hairstyle) are not as well preserved as when the semantic consistency loss is present.

Discussions and Limitations.

Our initial motivation for XGAN was to tackle the semantic style transfer problem in a fully unsupervised framework by combining techniques from domain adaptation and image-to-image translation.

We first observe that using a simple setup where a partially shared dual autoencoder is trained with reconstruction losses and a domain-adversarial loss already suffices to produce an embedding that captures basic semantics rather well (for instance, skin tone).

However, the generated samples are of poor quality and fine-grained attributes such as facial hair are not well captured.

These two problems are greatly diminished after adding the GAN loss and the proposed semantic consistency loss, respectively.

Failure cases still exist, especially on nonrepresentative input samples (e.g., a person wearing a hat) which are mapped to unrealistic cartoons.

Adding the teacher loss reduces this problem by regularizing the learned embedding, however it requires additional supervision and makes the model dependent on the specific representation provided by the teacher network.

Future work will focus on evaluating XGAN on more tasks.

In particular, , while we introduced XGAN as a solution to semantic style transfer, we think the model goes be-yond this scenario and could be applied to classical domain adaptation problems, where quantitative evaluation becomes possible.

In this work, we introduced XGAN, a model for unsupervised domain translation applied to the task of semantically-consistent style transfer.

In particular, we argue that learning image-to-image translation between two structurally different domains requires passing through a high-level joint semantic representation while discarding local pixel-level dependencies.

Additionally, we proposed a semantic consistency loss acting on both domain translations as a form of self-supervision.

We reported promising experimental results on the task of mapping the domain of face images to cartoon avatars that clearly outperform the current baseline.

We also showed that additional weak supervision, such as a pretrained feature representation, can easily be added to the model in the form of teacher knowledge.

While not necessary, it acts as a good regularizer for the learned embeddings and generated samples.

This can be particularly useful for natural image data as offthe-shelf pretrained models are abundant.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros.

Unpaired image-to-image translation using cycle-consistent adversarial networks.

In ICCV, 2017.

Autoencoder.

Encoders take 64x64 images as input, which are then fed through five 2D convolutional blocks.

Two fully-connected layers are applied to the last feature map in order to obtain the embedding vector.

Finally, we normalize the embedding vector so that it lies in the unit ball.

We use the cosine distance for all embedding comparisons (for the semantic consistency and teacher loss).

The architecture for the decoder is a mirrored version of the encoder.

From the initial flat embedding layer, we apply a sequence of five deconvolutions, the last block outputting an 64x64 color image.

For both the encoder and decoder, the two highest-level (de)convolutional blocks are shared across domains.

This encourages the model to learn shared representations at different levels of the architecture rather than only in the middle layer.

A detailed overview of the architecture is presented in Appendix 7.1.Discriminator.

The discriminator architecture is very similar to the encoder architecture with the difference that it only needs to output one logit for each input image, representing its binary classification decision.

In practice, we use a smaller architecture for the discriminator as it often tends to be too powerful and easily distinguish between real and transformed images.

We also report details of the XGAN architecture in TAB3 .

Note that all layers except the last ones are followed by batch normalization.

We also use ReLU as activation function for each of them, except for the last deconvolution of the decoders which uses hyperbolic tangent activation function.

DISPLAYFORM0

As we noted when experimenting with the DTN, its main drawback seems to come from the assumption to keep a fixed pretrained encoder in the model.

Following this observation, we perform another experiment in which we finetune the FaceNet encoder relatively to the semantic consistency loss, additionally to the decoder parameters.

While this yields visually better samples (see Figure 9 (b)), it also raises the classical domain adaptation issue of guaranteeing that the initial FaceNet embedding knowledge is preserved when retraining the embedding.

For comparison, XGAN exploits a teacher network that can be used to distill prior domain knowledge throughout training, when available.

Secondly, this finetuned DTN is prone to mode collapse.

In fact, the encoder is now only trained relatively to the semantic consistency loss which can be easily minizimed by mapping each domain to the same point in the embedding space, leading to the same cartoon being generated for all of them.

In XGAN, the source embeddings are regularized by the reconstruction loss on the source domain.

This allows us to learn a joint domain embedding from scratch in a proper domain adaptation framework.(a) Random generated samples (left) and reconstructions (right) with fixed FaceNet embedding (b) Random generated samples with a fine-tuned FaceNet encoder Figure 9 : Reproducing the Domain Transfer Network performs badly in our experimental setting (a); fine-tuning the encoder yields better results (b) but is unstable for training in practice.

As mentioned in the main text, the DTN baseline fails to capture a meaningful shared embedding for the two input domains.

Instead, we consider and experiment with three different models to tackle the semantic style transfer problem.

Selected samples are reported in FIG5 :• Finetuned DTN, as introduced previously.

In practice, this model yields satisfactory samples but is very sensitive to hyperparameter choice and often collapses to one model.

• XGAN with L rec and L dann active only corresponds to a simple domain-adaptation setting: the proposed XGAN model where only the reconstruction loss L rec and the domainadversarial loss L dann are active.

We observe that semantics are globally well preserved across domains although the model still makes some basic mistakes (e.g., gender misclassifications) and the samples quality is poor.• XGAN, the full proposed model, yields the best visual samples out of the models we experiment on.

In the rest of this section, we report a detailed study on its different components and possible failure modes.

In FIG6 we also report a more extensive random selection of samples produced by XGAN.

Note that we only used a discriminator for the source to target path (i.e., L gan,2→1 is inactive); in fact the GAN objective tends to make training more unstable so we only use one for the transformation we care most about for this specific application, i.e., faces to cartoons.

Other than the GAN objective, the model appears to be robust to the choice of hyperparameters.

Overall, the cartoon samples are visually very close to the original dataset and main identity characteristics such as face shape, hair style, skin tone, etc., are well preserved between the two domains.

The main failure mode appears to be mismatched hair color: in particular, bright red hair appear very often in generated samples which is likely due to its abundance in the training cartoon dataset.

In fact, when looking at the target to source generated samples, we observe that this color shade often gets mapped to dark brown hair in the real face domain.

One could expect the teacher network to regularize the hair color mapping, however FaceNet was originally trained for face identification, hence is most likely more sensitive to structural characteristics such as face shape.

More generally, most mistakes are due to the shift in content distribution rather than style distribution between the two domains.

Other examples include bald faces being mapped to cartoons with light hair (most likely due to the lack of bald cartoon faces and the model mistaking the white background for hair color).

Also, eyeglasses on cartoon faces disappear when mapped to the real face domain (only very few faces in the source dataset wear glasses).

In FIG7 we report examples of failure cases when ω dann is too high in the setting with the reconstruction and domain-adversarial loss only: The domain-adversarial classifier c dann reaches perfect accuracy and cross-domain translation fails.

As mentioned Section 3.1, we only use a GAN loss term for the source → target translation, to ease training.

This prompts the face-to-cartoon path to generate more realistic samples.

As expected, when the GAN loss is inactive, the generated samples are noisy and unrealistic (see FIG0 (a)).

For comparison, tackling the low quality problem with simpler regularization techniques such as using total variation smoothness loss leads to more uniform samples but significantly worsen their blurriness on the long term (see FIG0 b) ).

This shows the importance of the GAN objective for image generation applications, even though it makes the training process more complex.

@highlight

XGAN is an unsupervised model for feature-level image-to-image translation applied to semantic style transfer problems such as the face-to-cartoon task, for which we introduce a new dataset.

@highlight

This paper proposes a new GAN-based model for unpaired image-to-image translation similar to DTN