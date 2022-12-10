Unsupervised image-to-image translation has gained considerable attention due to the recent impressive progress based on generative adversarial networks (GANs).

However, previous methods often fail in challenging cases, in particular, when an image has multiple target instances and a translation task involves significant changes in shape, e.g., translating pants to skirts in fashion images.

To tackle the issues, we propose a novel method, coined instance-aware GAN (InstaGAN), that incorporates the instance information (e.g., object segmentation masks) and improves multi-instance transfiguration.

The proposed method translates both an image and the corresponding set of instance attributes while maintaining the permutation invariance property of the instances.

To this end, we introduce a context preserving loss that encourages the network to learn the identity function outside of target instances.

We also propose a sequential mini-batch inference/training technique that handles multiple instances with a limited GPU memory and enhances the network to generalize better for multiple instances.

Our comparative evaluation demonstrates the effectiveness of the proposed method on different image datasets, in particular, in the aforementioned challenging cases.

Code and results are available in https://github.com/sangwoomo/instagan

Cross-domain generation arises in many machine learning tasks, including neural machine translation BID2 BID21 , image synthesis BID33 BID48 , text style transfer BID34 , and video generation BID3 BID39 BID6 .

In particular, the unpaired (or unsupervised) image-to-image translation has achieved an impressive progress based on variants of generative adversarial networks (GANs) BID27 BID7 BID0 BID23 , and has also drawn considerable attention due to its practical applications including colorization , super-resolution BID22 , semantic manipulation BID40 , and domain adaptation BID5 BID35 BID15 .

Previous methods on this line of research, however, often fail on challenging tasks, in particular, when the translation task involves significant changes in shape of instances or the images to translate contains multiple target instances BID10 .

Our goal is to extend image-to-image translation towards such challenging tasks, which can strengthen its applicability up to the next level, e.g., changing pants to skirts in fashion images for a customer to decide which one is better to buy.

To this end, we propose a novel method that incorporates the instance information of multiple target objectsin the framework of generative adversarial networks (GAN); hence we called it instance-aware GAN (InstaGAN) .

In this work, we use the object segmentation masks for instance information, which may be a good representation for instance shapes, as it contains object boundaries while ignoring other details such as color.

Using the information, our method shows impressive results for multi-instance transfiguration tasks, as shown in FIG0 .Our main contribution is three-fold: an instance-augmented neural architecture, a context preserving loss, and a sequential mini-batch inference/training technique.

First, we propose a neural network architecture that translates both an image and the corresponding set of instance attributes.

Our architecture can translate an arbitrary number of instance attributes conditioned by the input, and is designed to be permutation-invariant to the order of instances.

Second, we propose a context preserv- ), and our proposed method, InstaGAN.

Our method shows better results for multi-instance transfiguration problems.ing loss that encourages the network to focus on target instances in translation and learn an identity function outside of them.

Namely, it aims at preserving the background context while transforming the target instances.

Finally, we propose a sequential mini-batch inference/training technique, i.e., translating the mini-batches of instance attributes sequentially, instead of doing the entire set at once.

It allows to handle a large number of instance attributes with a limited GPU memory, and thus enhances the network to generalize better for images with many instances.

Furthermore, it improves the translation quality of images with even a few instances because it acts as data augmentation during training by producing multiple intermediate samples.

All the aforementioned contributions are dedicated to how to incorporates the instance information (e.g., segmentation masks) for image-to-image translation.

However, we believe that our approach is applicable to numerous other cross-domain generation tasks where set-structured side information is available.

To the best of our knowledge, we are the first to report image-to-image translation results for multiinstance transfiguration tasks.

A few number of recent methods BID27 BID10 show some transfiguration results but only for images with a single instance often in a clear background.

Unlike the previous results in a simple setting, our focus is on the harmony of instances naturally rendered with the background.

On the other hand, CycleGAN show some results for multi-instance cases, but report only a limited performance for transfiguration tasks.

At a high level, the significance of our work is also on discovering that the instance information is effective for shape-transforming image-to-image translation, which we think would be influential to other related research in the future.

Mask contrast-GAN and Attention-GAN (Mejjati et al., 2018) use segmentation masks or predicted attentions, but only to attach the background to the (translated) cropped instances.

They do not allow to transform the shapes of the instances.

To the contrary, our method learns how to preserve the background by optimizing the context preserving loss, thus facilitating the shape transformation.

Given two image domains X and Y, the problem of image-to-image translation aims to learn mappings across different image domains, G XY : X → Y or/and G YX : Y → X , i.e., transforming target scene elements while preserving the original contexts.

This can also be formulated as a conditional generative modeling task where we estimate the conditionals p(y|x) or/and p(x|y).

The goal of unsupervised translation we tackle is to recover such mappings only using unpaired samples from marginal distributions of original data, p data (x) and p data (y) of two image domains.

The main and unique idea of our approach is to incorporate the additional instance information, i.e., augment a space of set of instance attributes A to the original image space X , to improve the image-to-image translation.

The set of instance attributes a ∈ A comprises all individual attributes of N target instances: DISPLAYFORM0 .

In this work, we use an instance segmentation mask only, but we remark that any useful type of instance information can be incorporated for the attributes.

Our approach then can be described as learning joint-mappings between attribute-augmented spaces X ×A and Y ×B. This leads to disentangle different instances in the image and allows the generator to perform an accurate and detailed translation.

We learn our attribute-augmented mapping in the framework of generative adversarial networks (GANs) BID11 , hence, we call it instance-aware GAN (InstaGAN).

We present details of our approach in the following subsections.

Recent GAN-based methods BID27 have achieved impressive performance in the unsupervised translation by jointly training two coupled mappings G XY and G YX with (c) , respectively.

Each network is designed to encode both an image and set of instance masks.

G is permutation equivariant, and D is permutation invariant to the set order.

To achieve properties, we sum features of all set elements for invariance, and then concatenate it with the identity mapping for equivariance.a cycle-consistency loss that encourages G YX (G XY (x)) ≈ x and G XY (G YX (y)) ≈ y. Namely, we choose to leverage the CycleGAN approach to build our InstaGAN.

However, we remark that training two coupled mappings is not essential for our method, and one can also design a single mapping following other approaches BID4 BID9 .

FIG1 illustrates the overall architecture of our model.

We train two coupled generators Our generator G encodes both x and a, and translates them into y and b .

Notably, the order of the instance attributes in the set a should not affect the translated image y , and each instance attribute in the set a should be translated to the corresponding one in b .

In other words, y is permutation-invariant with respect to the instances in a, and b is permutation-equivariant with respect to them.

These properties can be implemented by introducing proper operators in feature encoding BID43 .

We first extract individual features from image and attributes using image feature extractor f GX and attribute feature extractor f GA , respectively.

The attribute features individually extracted using f GA are then aggregated into a permutation-invariant set feature via summation: DISPLAYFORM0 As illustrated in FIG1 , we concatenate some of image and attribute features with the set feature, and feed them to image and attribute generators.

Formally, the image representation h GX and the n-th attribute representation h n GA in generator G can be formulated as: DISPLAYFORM1 where each attribute encoding h n GA process features of all attributes as a contextual feature.

Finally, h GX is fed to the image generator g GX , and h n GA (n = 1, . . .

, N ) are to the attribute generator g GA .

On the other hand, our discriminator D encodes both x and a (or x and a ), and determines whether the pair is from the domain or not.

Here, the order of the instance attributes in the set a should not affect the output.

In a similar manner above, our representation in discriminator D, which is permutation-invariant to the instances, is formulated as: DISPLAYFORM2 which is fed to an adversarial discriminator g DX .We emphasize that the joint encoding of both image x and instance attributes a for each neural component is crucial because it allows the network to learn the relation between x and a. For example, if two separate encodings and discriminators are used for x and a, the generator may be misled to produce image and instance masks that do not match with each other.

By using the joint encoding and discriminator, our generator can produce an image of instances properly depicted on the area consistent with its segmentation masks.

As will be seen in Section 3, our approach can disentangle output instances considering their original layouts.

Note that any types of neural networks may be used for sub-network architectures mentioned above such as f GX , f GA , f DX , f DA , g GX , g GA , and g DX .

We describe the detailed architectures used in our experiments in Appendix A.

Remind that an image-to-image translation model aims to translate a domain while keeping the original contexts (e.g., background or instances' domain-independent characteristics such as the looking direction).

To this end, we both consider the domain loss, which makes the generated outputs to follow the style of a target domain, and the content loss, which makes the outputs to keep the original contents.

Following our baseline model, CycleGAN , we use the GAN loss for the domain loss, and consider both the cycle-consistency loss BID42 and the identity mapping loss BID37 for the content losses.

1 In addition, we also propose a new content loss, coined context preserving loss, using the original and predicted segmentation information.

In what follows, we formally define our training loss in detail.

For simplicity, we denote our loss function as a function of a single training sample (x, a) ∈ X × A and (y, b) ∈ Y × B, while one has to minimize its empirical means in training.

The GAN loss is originally proposed by BID11 for generative modeling via alternately training generator G and discriminator D. Here, D determines if the data is a real one of a fake/generated/translated one made by G. There are numerous variants of the GAN loss BID32 BID31 , and we follow the LSGAN scheme BID28 , which is empirically known to show a stably good performance: DISPLAYFORM0 For keeping the original content, the cycle-consistency loss L cyc and the identity mapping loss L idt enforce samples not to lose the original information after translating twice and once, respectively: DISPLAYFORM1 Finally, our newly proposed context preserving loss L ctx enforces to translate instances only, while keeping outside of them, i.e., background.

Formally, it is a pixel-wise weighted 1 -loss where the weight is 1 for background and 0 for instances.

Here, note that backgrounds for two domains become different in transfiguration-type translation involving significant shape changes.

Hence, we consider the non-zero weight only if a pixel is in background in both original and translated ones.

Namely, for the original samples (x, a), (y, b) and the translated one (y , b ), (x , a ), we let the weight w(a, b ), w(b, a ) be one minus the element-wise minimum of binary represented instance masks, and we propose DISPLAYFORM2 where is the element-wise product.

In our experiments, we found that the context preserving loss not only keeps the background better, but also improves the quality of generated instance segmentations.

Finally, the total loss of InstaGAN is DISPLAYFORM3 where λ cyc , λ idt , λ ctx > 0 are some hyper-parameters balancing the losses.

While the proposed architecture is able to translate an arbitrary number of instances in principle, the GPU memory required linearly increases with the number of instances.

For example, in our experiments, a machine was able to forward only a small number (say, 2) of instance attributes during training, and thus the learned model suffered from poor generalization to images with a larger number of instances.

To address this issue, we propose a new inference/training technique, which allows to train an arbitrary number of instances without increasing the GPU memory.

We first describe the sequential inference scheme that translates the subset of instances sequentially, and then describe the corresponding mini-batch training technique.

Given an input (x, a), we first divide the set of instance masks a into mini-batches a 1 , . . .

, a M , i.e., a = i a i and a i ∩ a j = ∅ for i = j.

Then, at the m-th iteration for m = 1, 2, . . .

, M , we translate the image-mask pair (x m , a m ), where x m is the translated image y m−1 from the previous iteration, and x 1 = x. In this sequential scheme, at each iteration, the generator G outputs an intermediate translated image y m , which accumulates all mini-batch translations up to the current iteration, and a translated mini-batch of instance masks b m : DISPLAYFORM0 In order to align the translated image with mini-batches of instance masks, we aggregate all the translated mini-batch and produce a translated sample: DISPLAYFORM1 The final output of the proposed sequential inference scheme is (y M , b 1:M ).We also propose the corresponding sequential training algorithm, as illustrated in FIG3 .

We apply content loss (4-6) to the intermediate samples (y m , b m ) of current mini-batch a m , as it is just a function of inputs and outputs of the generator G.2 In contrast, we apply GAN loss (3) to the samples of aggregated mini-batches (y m , b 1:m ), because the network fails to align images and masks when using only a partial subset of instance masks.

We used real/original samples {x} with the full set of instance masks only.

Formally, the sequential version of the training loss of InstaGAN is DISPLAYFORM2 where DISPLAYFORM3 We detach every m-th iteration of training, i.e., backpropagating with the mini-batch a m , so that only a fixed GPU memory is required, regardless of the number of training instances.

sequential training allows for training with samples containing many instances, and thus improves the generalization performance.

Furthermore, it also improves translation of an image even with a few instances, compared to the one-step approach, due to its data augmentation effect using intermediate samples (x m , a m ).

In our experiments, we divided the instances into mini-batches a 1 , . . .

, a M according to the decreasing order of the spatial sizes of instances.

Interestingly, the decreasing order showed a better performance than the random order.

We believe that this is because small instances tend to be occluded by other instances in images, thus often losing their intrinsic shape information.

We first qualitatively evaluate our method on various datasets.

We compare our model, InstaGAN, with the baseline model, CycleGAN .

For fair comparisons, we doubled the number of parameters of CycleGAN, as InstaGAN uses two networks for image and masks, respectively.

We sample two classes from various datasets, including clothing co-parsing (CCP) (Yang et al., 2014), multi-human parsing (MHP) BID46 , and MS COCO datasets, and use them as the two domains for translation.

In visualizations, we merge all instance masks into one for the sake of compactness.

See Appendix B for detailed settings for our experiments.

The translation results for three datasets are presented in FIG4 , 5, and 6, respectively.

While CycleGAN mostly fails, our method generates reasonable shapes of the target instances and keeps the original contexts by focusing on the instances via the context preserving loss.

For example, see the results on sheep↔giraffe in FIG6 .

CycleGAN often generates sheep-like instances but loses the original background.

InstaGAN not only generates better sheep or giraffes, but also preserves the layout of the original instances, i.e., the looking direction (left, right, front) of sheep and giraffes are consistent after translation.

More experimental results are presented in Appendix E. Code and results are available in https://github.com/sangwoomo/instagan.On the other hand, our method can control the instances to translate by conditioning the input, as shown in FIG7 .

Such a control is impossible under CycleGAN.

We also note that we focus on complex (multi-instance transfiguration) tasks to emphasize the advantages of our method.

Nevertheless, our method is also attractive to use even for simple tasks (e.g., horse↔zebra) as it reduces false positives/negatives via the context preserving loss and enables to control translation.

We finally emphasize that our method showed good results even when we use predicted segmentation for inference, as shown in FIG8 , and this can reduce the cost of collecting mask labels in practice.

4 Finally, we also quantitatively evaluate the translation performance of our method.

We measure the classification score, the ratio of images predicted as the target class by a pretrained classifier.

Specifically, we fine-tune the final layers of the ImageNet BID8 ) pretrained VGG-16 (Simonyan & Zisserman, 2014 network, as a binary classifier for each domain.

TAB0 in Appendix D show the classification scores for CCP and COCO datasets, respectively.

Our method outperforms CycleGAN in all classification experiments, e.g., ours achieves 23.2% accuracy for the pants→shorts task, while CycleGAN obtains only 8.5%.

We now investigate the effects of each component of our proposed method in FIG9 .

Our method is composed of the InstaGAN architecture, the context preserving loss L ctx , and the sequential minibatch inference/training technique.

We progressively add each component to the baseline model, CycleGAN (with doubled parameters).

First, we study the effect of our architecture.

For fair comparison, we train a CycleGAN model with an additional input channel, which translates the mask-augmented image, hence we call it CycleGAN+Seg.

Unlike our architecture which translates the set of instance masks, CycleGAN+Seg translates the union of all masks at once.

Due to this, CycleGAN+Seg fails to translate some instances and often merge them.

On the other hand, our architecture keeps every instance and disentangles better.

Second, we study the effect of the context The left and right side of title indicates which method used for training and inference, respectively, where "One" and "Seq" indicate the one-step and sequential schemes, respectively.preserving loss: it not only preserves the background better (row 2), but also improves the translation results as it regularizes the mapping (row 3).

Third, we study the effect of our sequential translation: it not only improves the generalization performance (row 2,3) but also improves the translation results on few instances, via data augmentation (row 1).Finally, FIG0 reports how much the sequential translation, denoted by "Seq", is effective in inference and training, compared to the one-step approach, denoted by "One".

For the one-step training, we consider only two instances, as it is the maximum number affordable for our machines.

On the other hand, for the sequential training, we sequentially train two instances twice, i.e., images of four instances.

For the one-step inference, we translate the entire set at once, and for the sequential inference, we sequentially translate two instances at each iteration.

We find that our sequential algorithm is effective for both training and inference: (a) training/inference = One/Seq shows blurry results as intermediate data have not shown during training and stacks noise as the iteration goes, and (b) Seq/One shows poor generalization performance for multiple instances as the one-step inference for many instances is not shown in training (due to a limited GPU memory).

We have proposed a novel method incorporating the set of instance attributes for image-to-image translation.

The experiments on different datasets have shown successful image-to-image translation on the challenging tasks of multi-instance transfiguration, including new tasks, e.g., translating jeans to skirt in fashion images.

We remark that our ideas utilizing the set-structured side information have potential to be applied to other cross-domain generations tasks, e.g., neural machine translation or video generation.

Investigating new tasks and new information could be an interesting research direction in the future.

We adopted the network architectures of CycleGAN as the building blocks for our proposed model.

In specific, we adopted ResNet 9-blocks generator BID18 BID13 and PatchGAN discriminator.

ResNet generator is composed of downsampling blocks, residual blocks, and upsampling blocks.

We used downsampling blocks and residual blocks for encoders, and used upsampling blocks for generators.

On the other hand, PatchGAN discriminator is composed of 5 convolutional layers, including normalization and non-linearity layers.

We used the first 3 convolution layers for feature extractors, and the last 2 convolution layers for classifier.

We preprocessed instance segmentation as a binary foreground/background mask, hence simply used it as an 1-channel binary image.

Also, since we concatenated two or three features to generate the final outputs, we doubled or tripled the input dimension of those architectures.

Similar to prior works BID18 , we applied Instance Normalization (IN) BID38 for both generators and discriminators.

In addition, we observed that applying Spectral Normalization (SN) BID30 for discriminators significantly improves the performance, although we used LSGAN BID28 , while the original motivation of SN was to enforce Lipschitz condition to match with the theory of WGAN BID12 .

We also applied SN for generators as suggested in Self-Attention GAN BID44 , but did not observed gain for our setting.

For all the experiments, we simply set λ cyc = 10, λ idt = 10, and λ ctx = 10 for our loss (7).

We used Adam BID20 optimizer with batch size 4, training with 4 GPUs in parallel.

All networks were trained from scratch, with learning rate of 0.0002 for G and 0.0001 for D, and β 1 = 0.5, β 2 = 0.999 for the optimizer.

Similar to CycleGAN , we kept learning rate for first 100 epochs and linearly decayed to zero for next 100 epochs for multi-human parsing (MHP) BID46 and COCO dataset, and kept learning rate for first 400 epochs and linearly decayed for next 200 epochs for clothing co-parsing (CCP) BID41 dataset, as it contains smaller number of samples.

We sampled two classes from the datasets above, and used it as two domains for translation.

We resized images with size 300×200 (height×width) for CCP dataset, 240×160 for MHP dataset, and 200×200 for COCO dataset, respectively.

We tracked the trend of translation results over epoch increases, as shown in FIG0 .

Both image and mask smoothly adopted to the target instances.

For example, the remaining parts in legs slowly disappears, and the skirt slowly constructs the triangular shapes.

We evaluated the classification score for CCP and COCO dataset.

Unlike CCP dataset, COCO dataset suffers from the false positive problem, that the classifier fails to determine if the generator produced target instances on the right place.

To overcome this issue, we measured the masked classification score, where the input images are masked by the corresponding segmentations.

We note that CycleGAN and our method showed comparable results for the naïve classification score, but ours outperformed for the masked classification score, as it reduces the false positive problem.

We present more qualitative results in high resolution images.

FIG0 : Translation results for images searched from Google to test the generalization performance of our model.

We used a pix2pix model to predict the segmentation.

To demonstrate the effectiveness of our method further, we provide more comparison results with CycleGAN+Seg.

Since CycleGAN+Seg translates all instances at once, it often (a) fails to translate instances, or (b) merges multiple instances (see FIG1 , or (c) generates multiple instances from one instance (see FIG1 .

On the other hand, our method does not have such issues due to its instance-aware nature.

In addition, since the unioned mask losses the original shape information, our instance-aware method produces better shape results (e.g., see row 1 of FIG1 ).

To show that our model generalizes well, we searched the nearest training neighbors (in L 2 -norm) of translated target masks.

As reported in FIG1 , we observe that the translated masks (col 3,4) are often much different from the nearest neighbors (col 5,6 ).

This confirms that our model does not simply memorize training instance masks, but learns a mapping that generalizes for target instances.

For interested readers, we also present the translation results of the simple crop & attach baseline in FIG1 , that find the nearest neighbors of the original masks from target masks, and crop & attach the corresponding image to the original image.

Here, since the distance in pixel space (e.g., L 2 -norm) obviously does not capture semantics, the cropped instances do not fit with the original contexts as well.

For interested readers, we also report the translation and reconstruction results of our method in FIG3 .

One can observe that our method shows good reconstruction results while showing good translation results.

This implies that our translated results preserve the original context well.

@highlight

We propose a novel method to incorporate the set of instance attributes for image-to-image translation.

@highlight

This paper proposes a method -- InstaGAN -- which builds on CycleGAN by taking into account instance information in the form of per-instance segmentation masks, with results that compare favorably to CycleGAN and other baselines.

@highlight

 Proposes to add instance-aware segmentation masks for the problem of unpaired image-to-image translation.