Recently, Generative Adversarial Network (GAN) and numbers of its variants have been widely used to solve the image-to-image translation problem and achieved extraordinary results in both a supervised and unsupervised manner.

However, most GAN-based methods suffer from the imbalance problem between the generator and discriminator in practice.

Namely, the relative model capacities of the generator and discriminator do not match, leading to mode collapse and/or diminished gradients.

To tackle this problem, we propose a GuideGAN based on attention mechanism.

More specifically, we arm the discriminator with an attention mechanism so not only it estimates the probability that its input is real, but also does it create an attention map that highlights the critical features for such prediction.

This attention map then assists the generator to produce more plausible and realistic images.

We extensively evaluate the proposed GuideGAN framework on a  number of image transfer tasks.

Both qualitative results and quantitative comparison demonstrate the superiority of our proposed approach.

Generative Adversarial Networks (GANs) have drawn much attention during the past few years, due to their proven ability to generate realistic and sharp looking images.

Various computer vision problems are solved using this framework, such as super-resolution (Ledig et al., 2017) , colorization (Cao et al., 2017) , denoising (Yang et al., 2018) and style transfer .

All these problems can be considered as an image-to-image translation problem, mapping an image from source domain to target domain, for instance, the super-resolution problem of trying to transfer a low-resolution image (source domain) to a corresponding high-resolution image (target domain).

Existing literatures have shows that variants of GAN achieved impressive results in both a supervised and unsupervised setting.

Choi et al., 2018; Huang et al., 2018) Even with such great success, most GAN-based approaches are suffering from the imbalance between the generator and discriminator (Arjovsky & Bottou, 2017) .

In practice, the discriminator is usually too powerful for its task.

Thus, the generator obtains very small gradients from discriminator and is hard to converge.

Most state-of-the-art solutions are trying to find a new objective or add some new regularization terms to the cost function, which mainly affect the generator Arjovsky & Bottou, 2017; Mao et al., 2017; Nowozin et al., 2016; Hu et al., 2018) .

To address this problem from a different direction, we want to borrow some power from the discriminator by incorporating the attention mechanism to help the generator.

In this paper, we propose that the critical locating areas are more significant in the translation.

The generator should pay more attention to some particular areas rather than the whole image.

Imagine a student is learning how to draw a horse.

The standard discriminator, as a painting master, merely grades the student's painting and hopes that can help the student improve his work.

On the other hand, another master will provide additional information.

For instance, an error canvas circling each incorrect region.

That is exactly our idea: we suggest that the student (generator) gains benefit from the second master (attention embedded discriminator).

Our main contribution is threefold:

• A flexible attention-augmented discriminator: such discriminator provides not only the probability of realness, but an attention map from its perspective.

Both trainable attention module and post hoc attention are implemented.

• A unified GAN framework using attention information: to improve the translation of the generator, we combine the attention map with raw input via two concatenate methods: 1) convert the input to a RGBA image by adding an alpha channel; 2) compute the residual Hadamard production of the attention map and corresponding original input, based on RAM; • Extensive experiment validation on different benchmarks: we provide extensive experimental validation of GuideGAN on different benchmarks; both the qualitative results and quantitative comparisons against state-of-the-art methods demonstrate the effectiveness of our approach.

To the best of our knowledge, we are the first to report image-to-image translation results using the attention information from discriminator.

Different with previous approaches, our framework strengthens the communication and guidance between the generator and discriminator.

At a high level, the significance of our work is also on discovering that the attention information from auxiliary network affects the result of image-to-image translation, which we think would be influential to other related research in the future.

Generative Adversarial Network GANs have achieved impressive results in image translation tasks (Denton et al., 2015; Radford et al., 2015; Kim et al., 2017; Ledig et al., 2017) .

Typically, GAN consists of two components: a generator and a discriminator.

The generator is trained to fool the discriminator which in turn tries to distinguish between real and synthesised samples.

Various improvements to GANs have been proposed, like improved objective function (Mao et al., 2017; and advanced training strategies (Gulrajani et al., 2017; Nowozin et al., 2016) .

A recent framework, FAL (Huh et al., 2019) , iteratively improves the synthesized image using a spatial discriminator.

However, they either didn't collect enough information from the discriminator or need more forward pass to stabilize the result.

Image Translation Image to image translation can be considered as a generative process conditioned on an input image.

pix2pix was the first unified framework for supervised image-to-image translation based on conditional GAN (cGAN) (Mirza & Osindero, 2014) .

TextureGAN solves the sketch-to-image problem using user defined texture patch.

Gonzalez-Garcia et al. (2018) adopted disentanglement representation to improve the rendering process and Tang et al. (2019) utilized the extra semantic information to guide the generation.

Despite the promising results they achieved, the above methods are generally not applicable in practice due to the lack of paired data.

Several interesting frameworks have be proposed to solve this unsupervised image-to-image translation problem.

Cycle consistency loss was first proposed in CycleGAN and is widely used by other unsupervised image translation frameworks.

UNIT improves the translation with shared latent space assumption, which is the fundamental of MUNIT (Huang et al., 2018 ) that handles multi-modal translation.

In contrast, our flexible framework can be applied on both supervised and unsupervised settings.

Attention Learning Generally, attention can be viewed as guidance to bias the allocation of available processing resources towards the most informative components of an input.

Contemporary approaches are divided into two categories: post hoc network analysis and trainable attention module.

The former scheme has been predominantly employed to access network reasoning for the visual object recognition task (Simonyan et al., 2013; Zhou et al., 2016; Selvaraju et al., 2017; Chattopadhay et al., 2018) .

Trainable attention models fall into two main sub-categories, hard (stochastic) that requires reinforcement training and soft (deterministic) that can be trained end-to-end Hu et al., 2018; .

Attention is also widely used in image translation.

Ma et al. (2018) proposed the DA-GAN framework, which learns a deep attention encoder to discover the instance level correspondences.

Mejjati et al. (2018) separates the object and background using a trainable attention network.

InstaGAN (Mo et al., 2018) incorporates the instance information, like segmentation masks, to improve the multi-instance transfiguraiton.

Generally, these methods are trying to boost the attention embedded component, while we are using the attention mechanism to transfer more information from the discriminator to the generator.

We directly compare against several state-of-the-art approaches in Section 4.

3 METHOD Figure 1 : Overview of our framework.

Left block is a standard GAN with an attention embedded discriminator.

M x is the attention map provided by the discriminator.

The L1 loss between generated y i and corresponding ground truth y i is computed.

Right side is the framework for unsupervised translation using cycle consistency.

Ground truth y i is not available and the L1 loss between x and x is calculated instead.

Consider images from two different domains, source domain X and target domain Y. Data instances in source domain x ∈ X follow the distribution P x , whereas instances in target domain y ∈ Y follow the distribution P y .

Notice that we do not have labels in both X and Y. Our goal, in the problem setting of image-to-image translation, aims to learn mapping functions Gs across these two different image domains, G X : x → y and/or G Y : y → x, such that the differences between P x and G Y • P y and the difference between P y and G X • P x are minimized.

From the perspective of statistics, learning those two mapping functions can also be formulated as estimating the conditional distribution P (x|y) and P (y|x).

The main and unique idea of our approach is to incorporate the attention map generated by the discriminator, i.e., augment a space of attention information A to the original input space X, to improve the image-to-image translation.

The attention map can be further transformed to an extra alpha channel α (a mask channel with weight) or be interpreted as a pixel weight map.

In this paper, different attention mechanisms and concatenation methods have been studied and achieve promising results based on a different task setting.

Formally, our approach can be described as a joint-mapping learning from attention-augmented space X ⊕ A X to Y , and Y ⊕ A Y to X if cycle consistency applied, where ⊕ is the concatenate operation.

Our method explicitly forces the generator to put more processing resources to attended areas so it can conduct a sharp and clear translation.

Generally, this approach can be applied to any conditional GAN-based translation, hence, we call it GuideGAN.

We will present the detail of our approach in the following subsections.

Our framework, as illustrated in Figure 1 , is built upon GAN and attention mechanism.

For the supervised learning setting, it consists of three components, a generator G X , a discriminator D Y and an attention transfer block T .

It can be extended to unsupervised setting using CycleGAN framework easily, which now has five components: including two generators G X and G Y , two domain adversarial discriminators D Y and D X , and one shared attention transfer component T .

The training is based on each generator-discriminator pair.

Considering a standard GAN, the generator G X translates an image x i in X to an image in domain Y and the discriminator D Y tries to distinguish whether its input is a real or fake image in domain Y. Here, we denote y i = G X (x i ) as the output of generator, given x i .

Our attention embedded discriminator not only returns the probability of realness, D Y (y i ) ∈ [0, 1], but also an attention map A xi that highlights the focusing area of D Y .

This attention map A xi will be passed to the attention transfer block T to create an alpha channel or a pixel weight map, depending on the concatenation method, which will be described in Section 3.3.

For simplicity, the constructed term is denoted as M xi given A xi , despite its actual interpolation.

Noteworthy is the input of our generator G X is actually the concatenation of x i and M xi , which is represented as x i = x i ⊕ M xi .

At the start of the training, the attention map of each image is not available so we initialize it as an all-ones matrix A xi ∈ R m×n , where m × n is the shape of the input image.

Other initialization methods, like random noise, have also be examined but have limited impact on the final result.

The translation process of generator G X can be formulated as: y

where k and k + 1 denote the index of iteration and θ is the parameter of G X .

We emphasize that the attention map from D Y is crucial because it allows G X to focus on informative areas.

For example, if we only give the generator the raw input, G X may waste its processing resources on some inessential locations and D Y will beat it easily.

As a consequence, the loss of the discriminator quickly converges to zero and the generator can no longer efficiently update its parameter.

Alternatively, by concatenating the raw input with M x , the generator knows exactly where the discriminator is looking and allocates its processing resources properly on those areas.

As illustrated in Figure 1 , we can easily extend this framework to perform the unsupervised translation by adding another GAN component and enforcing cycle consistency.

Remember that our discriminator provides an extra attention map A xi for each image generated from x i .

Therefore, we consider both post hoc attention (PHA) that does not change the capacity of the discriminator, and trainable attention module (TAM), which enhances the discriminator's distinguishing power.

Given input x, the post hoc attention map is constructed from the backward gradients, forward activation, or the mix of them.

We use PatchGAN as the bone of our discriminator.

The network can be formulated as D = {l 0 , l 1 , . . .

, l m } where l i denotes i-th convolution layer in the network, and Act D = {a 1 , a 2 , . . .

, a m } is the set of activation map of corresponding layer.

This kind of attention map is sensitive to layer selection; different layer selection leads to different attention map (Mei et al., 2019) .

More specifically, if t is the chosen layer, the attention map can be described as:

where c is the number of channels in t-th layer and g(·) applies the min-max normalization.

This attention map only requires minor computation and works surprisingly well in most cases, but it may not achieve promising results when handling complex images.

On the contrary, a TAM is suitable for such complex input since it simultaneously increases the capacity of generator and discriminator.

Our TAM follows the same 2-branch architecture of the attention block in RAM .

See Appendix A for implementation details.

However, each branch in their implementation contains several ResBlock He et al. (2016) , which makes it impractical in our framework due to two reasons: 1) The discriminator conducts a simple binary classification 2) The capacity gap between generator and discriminator is already significant.

We replaced the Resblock by a simple convolution layer to simplify the network structure.

First few layers of the discriminator extract the low-level information of the input, and passes it to following branches.

Given the trunk branch output T (x) with the input x, the mask branch learns an attention map M (x) that softly weights the output of trunk branch.

Put these two outputs together:

where i ranges over all spatial positions and c ∈ {1, 2, ..., C} is the index of channels.

Finally, a few consecutive convolutional layers will do the final prediction based on E and attention map 1 C c M c (x) constructed from mask branch output will be returned.

In this section, we propose two methods to blend the attention map M x with its corresponding input x. The first one is based on the aforementioned attention module in the RAM .

We compute the Residual Hadamard Production (RHP) of the attention map and original input.

The reason of this operation are 1) Dot production with the attention range from zero to one will degrade the pixel value and cause fractional pixel problem (Mejjati et al., 2018) , 2) Attention mask can potentially break good property of the raw input.

For example, some pixels are not crucial in distinguish real and fake image, but they are still important for the image translation process.

This RHP can be formulated as:

where g(; φ) is a transfer function that up-sample the attention map to the shape of original input.

Another more intuitive concatenation is converting an RGB image to its RGBA version.

RGBA, as a color space, stands for red-green-blue-alpha.

It is the three-channel RGB color model supplemented with a 4-th alpha channel that indicates how opaque each pixel is.

This concatenation somehow makes nonessential areas more transparent thus highlighting the crucial locations.

Formally, it is described as:

where g(; φ) is a transfer function that maps attention map to alpha channel.

Follow the standard image pre-processing step, this concatenation can also be applied on gray scale image.

Gray scale image can be transformed into RGB image by repeating its intensity for each RGB channel.

Let's start with supervised translation.

The adversarial loss of the generator G and its discriminator D can be expressed as:

which is the adversarial loss of vanilla GAN.

G aims to minimize this objective while an adversary D tries to maximize it, i.e., min G max D L GAN (G, D).

However, this cost function is well known for its training difficulty.

We adopt the modified least-squares loss, proposed in LSGAN (Mao et al., 2017) , to further stabilize the training process and improve the quality of generated images.

The adversarial loss now becomes:

Adversarial loss alone does not guarantee a sound translation.

It is beneficial to mix traditional loss like L1 distance or L2 distance between synthesized image and ground truth.

Based on the suggestion from pix2pix ) that L1 loss encourages less blurry, we chose L1 loss as part of our training objective:

The final objective function in this setting is:

We can easily extend this framework to conduct unsupervised translation by adding another pair of generator and discriminator and enforcing cycle consistency.

Assume the generator G X simulates Table 1 : KID×100 ± std.×100 (Lower is better) computed for different combination on apple2orange and summer2winter.

Left 4 columns shown the target only KID and the rest 4 columns shown the mean KID (Lower the better).

Best results are bolded.

the map function G : X → Y and discriminator D Y are trying to distinguish between G(x) and y, the objective of this GAN component is L GAN (G X , D Y ) .

The generator G Y and discriminator D X is doing the same task in an opposite direction, their loss function is L GAN (G Y , D X ).

Cycle consistency is employed in such unsupervised setting because it alleviate the shortness of paired data.

It assumes that if a image x from domain X has be translated to a fake imageŷ in domain Y, we should get the same image x by applying G Y : Y → X. This behavior is formally presented as:

The final objective function for the unsupervised translation is:

A crucial point of our framework is how we perform the inference in test phase.

The attention map of same input from previous iterations can be used at training phase.

However, this information is not available in testing, and some placeholders are required.

To alleviate the problem that lead to this phenomenon, the generator should not rely too much on the attention map.

Our proposed concatenation methods naturally handle this problem, since the attention map can only amplify the information but never hurt the original input.

An all-one attention map is used as the placeholder based on the assumption that whole image is important.

Recall that we implemented two attention mechanisms and two concatenations.

Then the problem is how to combine them properly.

First, qualitative results are presented in Figure 2 .

As discussed in Section 3, TAM is not good at handling simple datasets, e.g. apple2orangewhile the results are more attractive for more complex summer2winter dataset.

By comparing alpha concatenation with RHP concatenation under post hoc attention, we find that the contrast ratio of the synthesized image is normally too high and makes the image look unrealistic.

We also present a quantitative evaluation for each method combination in Table 1 .

Kernel Inception Distance (KID) (Bińkowski et al., 2018 ) is used as evaluation metric, which computes the squared MMD (Maximum Mean Discrepancy) between feature representations of real and generated images.

Different from the Frchet Inception Distance (Heusel et al., 2017) , KID is more reliable because of the unbiased estimator.

While KID is unbounded, the lower its value, the more shared visual similarities there are between real and generated images.

Numeric results in the table justified our previous observations.

Based on the overall performance across different tasks, most experiments are using PHA and RHP by default.

We first evaluate our method on four benchmark datasets.

orange2apple, horse2zebra (Deng et al., 2009 ) are for object transfer and day2night, summer2winter are two challenging scenery transfer tasks.

day2night is cropped from BDD110k , which contains 7870 Table 2 : KID×100 ± std.×100 (Lower is better) computed using only target domain for different methods and on different datasets.

Best results are bolded.

images of daytime street traffic signs and 8592 night street traffic signs.

All data were split into train and test randomly (80%/20% split).

Then for all tasks, we present target KID in Table 2 and mean KID in Table 3 .

The target only KID is meaningful when the background of an image is not important.

For example, in apple2orange and horse2zebra tasks, we only care about objects in those images.

Under such scenarios, our proposed framework outperforms all baselines in all tasks except one task in day2night.

Still, our result is very close to the best.

This observation is consistent with our qualitative evaluation in Figure 3 , where our fake horse (zebra) is much more realistic than the counterparts produced by baselines.

However, we can clearly see the background changed using our method, even though we still have leading performance on apple2orange and summer2winter regarding the mean KID between source and target domain.

It's surprising to see that simplest CycleGAN model got the first place in day2night, which is a very hard task compared to two object transfer datasets.

Another interesting observation is our framework got 6.96 for the task night→day, while CycleGAN got 7.68, given CycleGAN outperforms our method in the easier day→night direction.

Figure 3 : Image-to-Image translation results generated by different approaches on apple2orange and horse2zebra.

Every two rows from top: apple→orange, orange→apple, zebra→horse, horse→zebra.

We then evaluate our method on Cityscape (Cordts et al., 2016) using FCN scores.

Appendix B is shown for detailed evaluation protocol.

We train photo→label and label→photo tasks on the Cityscape, and compare the output label images with the ground truth.

We used only RHP concatenation for this task.

We find that our method significantly outperforms the baselines in these experiment, especially when PHA and RHP work together, as shown in Table 4 .

The image translation result is also presented in Figure 6 .

The significant improvement in the pixel-level accuracy comes from the guidance of the attention map, which aligns with our expectations.

However, the improvement for metrics, Table 5 : FCN-scores (Higher is better) for different methods, evaluated on Cityscape label↔photos in supervised setting.

See Appendix C for qualitative result.

for this phenomenon, since we cannot increase the accuracy for nonexistent class objects.

Empirical justifications are available in Appendix D.

Meanwhile, the improvement of the supervised translation is not as sharp as the unsupervised translation according to Table 5 .

Yet it still shows that we can further improve the translation results with little extra computation, especially when PHA has been chosen.

We believe that the major reason actually due to strong regularizations, which are from the L1 distance between paired images.

The generator receives two feedbacks when paired image is available.

1) The L1 loss between paired image and 2) The prediction from the discriminator.

Recall that the idea behind our framework is letting the discriminator provide more useful information, but maybe the information from L1 loss is already sufficient.

Appendix C offers qualitative result.

we have proposed a novel method incorporating attention map from discriminator for image-toimage translation.

The experiments on different datasets have shown successful translation in both supervised and unsupervised setting.

We remark that our idea can apply on any GAN-based model with little modification, such as those baselines in the paper.

Nonetheless, the results are sensitive to the selection of attention module and concatenation.

Investigating the impact of different attention mechanism and new tasks could be an interesting research direction in the future.

For the unsupervised Cityscape translation, we adopted the network architectures of CycleGAN as the basic of our proposed model.

In specific, we adopted the ResNet 6-blocks (He et al., 2016) generator and the PatchGAN discriminator.

This generator contains 2 down-sampling blocks, 6 residual blocks and 2 up-sampling blocks.

For the supervised translation, we adopted the UNet-128 (Ronneberger et al., 2015) generator and a same PatchGAN discriminator.

The PatchGAN discriminator is composed of 5 convolution layers, including normalization and ReLU layers.

Before diving into the detail of our modified discriminator, let us first describe the details of RAM's 2-branch architecture .

They built a very deep network with numbers of attention blocks.

Each attention block contains two branches: mask branch and trunk branch.

Mask branch cascades the input features through a bottom-up top-down architecture that mimics human attention.

Trunk branch is applied as feature processing.

To build a TAM discriminator with this 2-branch architecture, we replaced the ResBlock by a simple convolution layer, as presented in the left part of Figure 5 .

In this TAM discriminator, we use the first convolution layer as feature extractor, three consecutive convolution layers for trunk branch and the last one convolution layer for classifier.

The mask branch is composed of two downsampling layers, two convolution layers and one upsampling layer.

As presented in the right side of Figure 5 , we selected the 4-th convolution layer to compute the post hoc attention map, based on the formula in Section 3.2.

All attention maps will be detached from the computation graph and be resized to the shape of original input.

Either by a bilinear upsampling layer or by a small 3-layer neural network.

Figure 5: Left: PatchGAN discriminator using TAM, the attention map is denoted as A x ; Right: Patch discriminator using post hoc attention, the attention map A x is computed from 4-th conv layer.

horse2zebra, apple2orange and day2night tasks are performed under the unsupervised setting.

For this three tasks, we adopted ResNet 9-blocks generator and aforementioned PatchGAN discriminator.

Similar to prior works, we applied Instance Normalization (IN) for both generators and discriminators.

In the preprocessing step, we resized the input image to 143 × 143 then randomly cropping back to 128 × 128 for all Cityscape related tasks.

We resized the input image to 286 × 286 then randomly cropping back to 256 × 256 for the rest tasks.

For all the experiments, we simply set the weight factor of the GAN loss to 10 and the weight factor of L1 loss to 10 for our objective.

For example, our implementation uses following objective for supervised training.

We used Adam optimizer with batch size 1, training on a Quadro 8000 GPU.

All networks were trained from scratch, with learning rate of 0.0002 for both the generator and discriminator, and β 1 = 0.5, β 2 = 0.999 for the optimizer.

Similar to CycleGAN, we kept learning rate for first 100 epochs and linearly decayed to 0 for next 100 epochs for apple2orange and Cityscape related tasks, and kept learning rate for first 50 epochs and linearly decayed to 0 for next 50 epochs for horse2zebra and day2night datasets.

Evaluating the quality of synthesized images is an open and difficult problem.

In this paper, we trained a network to perform label→photo and photo→label translation in both supervised and unsupervised manner.

Classical metrics such L1, or L2, distance between the real image and synthesized image are not suitable since they do not assess joint statistics of the result.

Researchers in image segmentation are widely using a pretrained semantic classifier to measure the discriminability of the generated image as a surrogate metric.

The assumption behind such measurement is that if the generated images are indeed realistic, classifiers pretrained on real images should classify the synthesized image correctly as well.

For the Cityscapes dataset, we used the FCN-8s (Long et al., 2015) network released by , which is pretrained on the Cityscape dataset.

The metrics we used in our experiment are per-pixel accuracy, per-class accuracy.

and Intersection over Union (IoU).

Per-pixel accuracy, namely, is the ratio between the number of correctly predicted pixels and total number of pixels.

It can be presented as:

where P (x) denotes the number of correctly predicted pixels and M ×N is the sharp of input image.

Per-class accuracy, also known as macro-average, is self explanatory.

It computes the the accuracy for each class and then compute the average.

It's formulated as:

where K is the set of classes, P (x, k) denotes the number of correctly predicted pixels for class k and G(k) is the total number of pixels that belongs to class k in ground truth.

Intersection over Union (IoU) is another often used metric for image segmentation.

It computes the ratio between the number of pixels seat in the intersection between predicted segmentation mask and ground truth, and the union of them.

Let P (x) be the prediction and GT (x) be the ground truth.

For all three aforementioned metric, the highest score is one, and the closer to one, the better.

Cityscape translation result is presented in Figure 6 .

In order to empirically justify our hypothesis for the limited improvement over per-class accuracy and IoU in Section 4.

We conduct two additional experiments and show its result here.

We first compute the statistic information of each class.

Figure 7 shows the statistic frequency of each class.

Namely, it tell us how many images contains the specific class in the dataset.

Another useful statistic information, which tell us the average frequency for each class, is provided in Figure 8 .

This statistic justified our second hypothesis that some class objects merely presented in the image thus it's hard to improve the per-class accuracy and IoU. We then extracted the attention map of whole training set and computed the average per-class attention map intensity.

More specific, we first perform a binary normalization over all attention map using a threshold α (In this experiment we use α = 0.5).

So we assume a pixel is crucial if the attention value on it is larger than α.

For a specific class in one image, it's regarded as attended if at least half of its pixel is crucial.

In the Figure 9 , we show the average per-class attention map intensity in different epochs.

Based on aforementioned figures, it's not hard to find out that the attention map are focusing on small classes.

For example, rider and terrain.

If the generator tried too hard to fix those small wholes but ignore the major classes, like car, the per-class accuracy and IoU will also be affected.

Since the contribution from generating good riders and terrain is significantly less than the contribution from generating good cars.

This experiment also justified our first hypothesis.

We present some intermediate training results with its attention map in Figure 10 , Figure 11 and Figure 12 .

The white area in the attention map indicates that region is important.

Please note that the attention map indicates the behavior of the discriminator thus some of them may not make sense from human's perspective.

From left to right: real apple images, fake orange images, real orange images, fake apple images.

Figure 15: Additional translation results on horse2zebra dataset using RAM + RHP.

From left to right: real horse images, fake zebra images, real zebra images, fake horse images.

<|TLDR|>

@highlight

A general method that improves the image translation performance of GAN framework by using an attention embedded discriminator

@highlight

A feedback mechanism in the GAN framework which improves the quality of generated images in image-to-image translation, and whose discriminator outputs a map indicating where the generator should focus to make its results more convincing.

@highlight

Proposal for a GAN with an attention-based discriminator for I2I translation which provides the probability of real/fake and an attention map which reflects salience for image generation.