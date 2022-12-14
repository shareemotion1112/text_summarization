Image translation between two domains is a class of problems aiming to learn mapping from an input image in the source domain to an output image in the target domain.

It has been applied to numerous applications, such as data augmentation, domain adaptation, and unsupervised training.

When paired training data is not accessible, image translation becomes an ill-posed problem.

We constrain the problem with the assumption that the translated image needs to be perceptually similar to the original image and also appears to be drawn from the new domain, and propose a simple yet effective image translation model consisting of a single generator trained with a self-regularization term and an adversarial term.

We further notice that existing image translation techniques are agnostic to the subjects of interest and often introduce unwanted changes or artifacts to the input.

Thus we propose to add an attention module to predict an attention map to guide the image translation process.

The module learns to attend to key parts of the image while keeping everything else unaltered, essentially avoiding undesired artifacts or changes.

The predicted attention map also opens door to applications such as unsupervised segmentation and saliency detection.

Extensive experiments and evaluations show that our model while being simpler, achieves significantly better performance than existing image translation methods.

Figure 1: Horse→zebra image translation.

Our model learns to predict an attention map (b) and translates the horse to zebra while keeping the background untouched (c).

By comparison, Cycle- GAN Zhu et al. (2017a) significantly alters the appearance of the background together with the horse (d).Many computer vision problems can be cast as an image-to-image translation problem: the task is to map an image of one domain to a corresponding image of another domain.

For example, image colorization can be considered as mapping gray-scale images to corresponding images in RGB space ; style transfer can be viewed as translating images in one style to corresponding images with another style BID12 ; BID19 ; BID11 .

Other tasks falling into this category include semantic segmentation BID32 , super-resolution BID26 , image manipulation , etc.

Another important application of image translation is related to domain adaptation and unsupervised learning: with the rise of deep learning, it is now considered crucial to have large labeled training datasets.

However, labeling and annotating such large datasets are expensive and thus not scalable.

An alternative is to use synthetic or simulated data for training, whose labels are trivial to acquire BID59 ; BID50 ; BID43 ; BID40 ; BID38 ; BID34 ; BID20 ; BID6 .

Unfortunately, learning from synthetic data can be problematic and most of the time does not generalize to real-world data, due to the data distribution gap between the two domains.

Furthermore, due to the deep neural networks' capability of learning small details, it is anticipated that the trained model would easily over-fits to the synthetic domain.

In order to close this gap, we can either find mappings or domain-invariant representations at feature level BID3 ; BID10 ; BID33 ; ; BID51 ; BID14 ; BID5 ; BID0 ; BID21 or learn to translate images from one domain to another domain to create "fake" labeled data for training BID4 ; BID58 ; ; BID26 ; BID29 ; BID55 .

In the latter case, we usually hope to learn a mapping that preserves the labels as well as the attributes we care about.

Typically there exist two settings for image translation given two domains X and Y .

The first setting is supervised, where example image pairs x, y are available.

This means for the training data, for each image x i ∈ X there is a corresponding y i ∈ Y , and we wish to find a translator G : .

However, paired training data comes at a premium.

For example, for image stylization, obtaining paired data requires lengthy artist authoring and is extremely expensive.

For other tasks like object transfiguration, the desired output is not even well defined.

DISPLAYFORM0 Therefore, we focus on the second setting, which is unsupervised image translation.

In the unsupervised setting, X and Y are two independent sets of images, and we do not have access to paired examples showing how an image x i ∈ X could be translated to an image y i ∈ Y .

Our task is then to seek an algorithm that can learn to translate between X and Y without desired input-output examples.

The unsupervised image translation setting has greater potentials because of its simplicity and flexibility but is also much more difficult.

In fact, it is a highly under-constrained and ill-posed problem, since there could be unlimited many number of mappings between X and Y : from the probabilistic view, the challenge is to learn a joint distribution of images in different domains.

As stated by the coupling theory BID27 , there exists an infinite set of joint distributions that can arrive the two marginal distributions in two different domains.

Therefore, additional assumptions and constraints are needed for us to exploit the structure and supervision necessary to learn the mapping.

Existing works that address this problem assume that there are certain relationships between the two domains.

For example, CycleGAN Zhu et al. (2017a) assumes cycle-consistency and the existence of an inverse mapping F that translates from Y to X. It then trains two generators which are bijections and inverse to each other and uses adversarial constraint BID13 to ensure the translated image appears to be drawn from the target domain and the cycle-consistency constraint to ensure the translated image can be mapped back to the original image using the inverse mapping (F (G(x) ) ≈ x and G(F (y)) ≈ y).

UNIT Liu et al. (2017) , on the other hand, assumes shared-latent space, meaning a pair of images in different domains can be mapped to some shared latent representations.

The model trains two generators G X , G Y with shared layers.

Both G X and G Y maps an input to itself, while the domain translation is realized by letting x i go through part of G X and part of G Y to get y i .

The model is trained with an adversarial constraint on the image, a variational constraint on the latent code BID23 ; BID39 , and another cycle-consistency constraint.

Assuming cycle consistency ensures 1-1 mapping and avoids mode collapses BID44 , both models generate reasonable image translation and domain adaptation results.

However, there are several issues with existing approaches.

First, such approaches are usually agnostic to the subjects of interest and there is little guarantee it reaches the desired output.

In fact, approaches based on cycle-consistency BID58 could theoretically find any arbitrary 1-1 mapping that satisfies the constraints, and this renders the training unstable and the results random.

This is problematic in many image translation scenarios.

For example, when translating from a horse image to a zebra image, most likely we only wish to draw the particular black-white stripes on top of the horses while keeping everything else unchanged.

However, what we observe is that existing approaches BID58 ; do not differentiate between the horse/zebra from the scene background, and the colors and appearances of the background often significantly change during translation (Fig. 1) .

Second, most of the time we only care about one-way translation, while existing methods like CycleGAN Zhu et al. (2017a) and UNIT Liu (2017) always require training two generators of bijections.

This is not only cumbersome but it is also hard to balance the effects of the two generators.

Third, there is a sensitive trade-off between the faithfulness of the translated image to the input image and how similar it resembles the new domain, and it requires excessive manual tuning of the weight between the adversarial loss and the reconstruction loss to get satisfying results.

To address the aforementioned issues, we propose a simpler yet more effective image translation model that consists of a single generator with an attention module.

We first re-consider what the desired outcome of an image translation task should be: most of the time the desired output should not only resemble the target domain but also preserve certain attributes and share similar visual appearance with input.

For example, in the case of horse-zebra translation BID58 , the output zebra should be similar to the input horse in terms of the scene background, the location and the shape of the zebra and horse, etc.

In the domain adaptation task that translates MNIST LeCun et al. (2010) to USPS Denker et al. (1989) , we expect the output is visually similar to the input in terms of the shape and structure of the digit such that it preserves the label.

Based on such observation, our model proposes to use a single generator that maps X to Y and is trained with a self-regularization term that enforces perceptual similarity between the output and the input, together with an adversarial term that enforces the output to appear like drawn from Y .

Furthermore, in order to focus the translation on key components of the image and avoid introducing unnecessary changes to irrelevant parts, we propose to add an attention module that predicts a probability map as to which part of the image it needs to attend to when translating.

Such probability maps, which are learned in a completely unsupervised fashion, could further facilitate segmentation or saliency detection ( Fig. 1 ).

Third, we propose an automatic and principled way to find the optimal weight between the self-regularization term and the adversarial term such that we do not have to manually search for the best hyper-parameter.

Our model does not rely on cycle-consistency or shared representation assumption, and it only learns one-way mapping.

Although the constraint is susceptible to oversimplify certain scenarios, we found that the model works surprisingly well.

With the attention module, our model learns to detect the key objects from the background context and is able to correct artifacts and remove unwanted changes from the translated results.

We apply our model on a variety of image translation and domain adaptation tasks and show that our model is not only simpler but also works better than existing methods, achieving superior qualitative and quantitative performance.

To demonstrate its application in real-world tasks, we show our model can be used to improve the accuracy of face 3D morphable model BID1 prediction by augmenting the training data of real images with adapted synthetic images.

We begin by explaining our model for unsupervised image translation.

Let X and Y be two image domains, our goal is to train a generator G θ : X → Y , where θ are the function parameters.

For simplicity, we omit θ and use G instead.

We are given unpaired samples x ∈ X and y ∈ Y , and the unsupervised setting assumes that x and y are independently drawn from the marginal distributions P x∼X (x) and P y∼Y (y).

Let y ′ = G(x) denote the translated image, the key requirement is that y ′ should appear like drawn from domain Y , while preserving the low-level visual characteristics of x. The translated images y ′ can be further used for other downstream tasks such as unsupervised learning.

However, in our case, we decouple image translation from its applications.

Based on the requirements described, we propose to learn θ by minimizing the following loss: Here DISPLAYFORM0 DISPLAYFORM1 where G 0 is the vanilla generator and G attn is the attention branch.

G 0 outputs a translated image while G attn predicts a probability map that is used to composite G 0 (x) with x to get the final output.

The first part of the loss, ℓ adv , is the adversarial loss on the image domain that makes sure that G(x) appears like domain Y .

The second part of the losses ℓ reg makes sure that G(x) is visually similar to x. In our case, ℓ adv is given by a discriminator D trained jointly with G, and ℓ reg is measured with perceptual loss.

We illustrate the model in FIG2 .The model architectures: Our model consists of a generator G and a discriminator D. The generator G has two branches: the vanilla generator G 0 and the attention branch G attn .

G 0 translates the input x as a whole to generate a similar image G 0 (x) in the new domain, and G attn predicts a probability map G attn (x) as the attention mask.

G attn (x) has the same size as x and each pixel is a probability value between 0-1.

In the end, we composite the final image G(x) by adding up x and G 0 (x) based on the attention mask.

G 0 is based on Fully Convolutional Network (FCN) and leverages properties of convolutional neural networks, such as translation invariance and parameter sharing.

Similar to BID58 , the generator G is built with three components: a down-sampling front-end to reduce the size, followed by multiple residual blocks BID16 , and an up-sampling back-end to restore the original dimensions.

The down-samping front-end consists of two convolutional blocks, each with a stride of 2.

The intermediate part contains nine residual blocks that keep the height/width constant, and the up-sampling back-end consists of two deconvolutional blocks, also with a stride of 2.

Each convolutional layer is followed by batch normalization and ReLU activation, except for the last layer whose output is in the image space.

Using down-sampling at the beginning increases the receptive field of the residual blocks and makes it easier to learn the transformation at a smaller scale.

Another modification is that we adopt the dilated convolution in all residual blocks, and set the dilation factor to 2.

Dilated convolutions use spaced kernels, enabling it to compute each output value with a wider view of input without increasing the number of parameters and computational burden.

G attn consists of the initial layers of the VGG-19 network BID47 (up to conv3_3), followed by two deconvolutional blocks.

In the end it is a convolutional layer with sigmoid that outputs a single channel probability map.

During training, the VGG-19 layers are warm-started with weights pretrained on ImageNet BID42 .For the discriminator, we use a five-layer convolutional network.

The first three layers have a stride of 2 followed by two convolution layers with stride 1, which effectively down-samples the networks three times.

The output is a vector of real/fake predictions and each value corresponds to a patch of the image.

Classifying each patch as real/fake introduces PatchGAN, and is shown to work better than the global GAN Zhu et al. (2017a) ; .

DISPLAYFORM2 The adversarial loss used to update the generator G is defined as: DISPLAYFORM3 By minimizing the loss function, the generator G learns to create translated image that fools the network D into classifying the image as drawn from Y .Self-regularization loss: Theoretically, adversarial training can learn a mapping G that produces outputs identically distributed as the target domain Y .

However, if the capacity is large enough, a network can map the input images to any random permutations of images in the target domain.

Thus, adversarial loses alone cannot guarantee that the learned function G maps the input to the desired output.

To further constrain the learned mapping such that it is meaningful, we argue that G should preserve visual characteristics of the input image.

In other words, the output and the input need to share perceptual similarities, especially regarding the low-level features.

Such features may include color, edges, shape, objects, etc.

We impose this constraint with the self-regularization term, which is modeled by minimizing the distance between the translated image y ′ and the input x:

DISPLAYFORM4 Here d is some distance function d, which can be ℓ 2 , ℓ 1 , SSIM, etc.

However, recent research suggests that using perceptual distance based on a pre-trained network corresponds much better to human perception of similarity comparing with traditional distance measures BID57 .

In particular, we defined the perceptual loss as: DISPLAYFORM5 HereF is VGG pretrained on ImageNet used to extract the neural features; we use l to represent each layer, and H l , W l are the height and width of featureF l .

We extract neural features withF across multiple layers, compute the ℓ 2 difference at each location h, w ofF l and average over the feature height and width.

We then scale it with layer-wise weight w l .

We did extensive experiments to try different combinations of feature layers and obtained the best results by only using the first three layers of VGG and setting w 1 , w 2 , w 3 to be 1.0/32, 1.0/16, 1.0/8 respectively.

This conforms to the intuition that we would like to preserve the low-level traits of the input during translation.

Note that this may not always be true (such as in texture transfer), but it is a hyper-parameter that could be easily adjusted based on different problem settings.

We also experimented with using different pre-trained networks such as AlexNet to extract neural features as suggested by BID57 but do not observe much difference in results.

Training scheme: In our experiment, we found that training the attention branch and the vanilla generator branch is difficult as it is hard to balance the learned translation and mask.

In our practice, we train the two branches separately.

First, we train the vanilla generator G 0 without the attention branch.

After it converges, we train the attention branch G attn while keeping the trained generator G 0 fixed.

In the end, we jointly fine-tune them with a smaller learning rate.

Adaptive weight induction: Like other image translation methods, the resemblance to the new domain and faithfulness to the original image is a trade-off.

In our model, it is determined by the weight λ of the self-regularization term relative to the image adversarial term.

If λ is too large, the translated image will be close to the input but does not look like the new domain.

If λ is too small, the translated image would fail to pertain the visual traits of the input.

Previous approaches usually decide the weight heuristically.

Here we propose an adaptive scheme to search for the best λ: we start by setting λ = 0, which means we only use the adversarial constraint to train the generator.

Then we gradually increase λ.

This would lead to the increase of the adversarial loss as the output would shift away from Y to X, which makes it easier for D to classify.

We stop increasing λ when the adversarial loss sinks below some threshold ℓ t adv .

We then keep λ constant and continue to train the network until converging.

Using the adaptive weight induction scheme avoids manual tuning of λ for each specific task and gives results that are both similar to the input x and the new domain Y .

Note that we repeat such process both when training G 0 and G attn .

Analysis: Our model is related to CycleGAN in that if we assume 1-1 mapping, we can define an inverse mapping F : Y → X such that F (G(x)) = x. This satisfies the constraints of CycleGAN in that the cycle-consistency loss is zero.

This shows that our learned mapping belongs to the set of possible mappings given by CycleGAN.

On the other hand, although CycleGAN tends to learn the mapping such that the visual distance between y ′ and x is small possibly due to cycleconsistency constraint, it does not guarantee to minimize the perceptual distance between G(x) and x. Comparing with UNIT, if we add another constraint that G(y) = y, then it is a special case of the UNIT model where all layers of the two generators are shared which leads to a single generator G.

).

However, we observe that adding the additional self-mapping constraint for domain Y does not improve the results.

Even though our approach assumes the perceptual distance between x and its corresponding y ∈ Y is small, our approach generalizes well to tasks where the input and output domains are significantly different, such as translation of photo to map, day to night, etc., as long as our assumption generally holds.

For example, in the case of photo to map, the park (photo) is labeled as green (map) and the water (photo) is labeled as blue (map), which provides certain low-level similarities.

Experiments show that even without the attention branch, our model produces results consistently similar or better than other methods.

This indicates that the cycle-consistency assumption may not be necessary for image translation.

Note that our approach is a meta-algorithm, and we could potentially improve the results by using new/more advanced components.

For example, the generator and discriminator could be easily replaced with the latest GAN architectures such as LSGAN Mao et al. (2017) , WGAN-GP Gulrajani et al. (2017) , or adding spectral normalization BID36 .

We may also improve the results by employing a more specific self-regularizaton term that is fine-tuned on the datasets we work on.

We tested our model on a variety of datasets and tasks.

In the following, we show the qualitative results of image translation, as well as quantitative results in several domain adaptation settings.

In our experiments, all images are resized to 256x256.

We use Adam solver BID22 to update the model weights during training.

In order to reduce model oscillation, we update the discriminators using a history of generated images rather than the ones produced by the latest generative models BID46 : we keep an image buffer that stores the 50 previously generated images.

All networks were trained from scratch with a learning rate of 0.0002.

Starting from 5k iteration, we linearly decay the learning rate over the remaining 5k iterations.

Most of our training takes about 1 day to converge on a single Titan X GPU.

FIG3 shows visual results of image translation of horse to zebra.

For each image, we show the initial translation G 0 (x), the attention map G attn (x) and the final result G(x) composited using G 0 (x) and x based on G attn (x).

We also compare the results with CycleGAN Zhu et al. (2017a) and UNIT Liu (2017) , and all models are trained using the same number of iterations.

For the baseline implementation, we use the original authors' implementations.

We can see from the examples that without the attention branch, our simple translation model G 0 already gives results similar or better than BID58 .

However, all these results suffer from perturbations of background color/texture and artifacts near the region of interest.

With the predicted attention map which learns to segment the horses, our final results have much higher visual quality, with the background keeping untouched and artifacts near the ROI removed (row 2, 4).

Complete results of horse-zebra translations and comparisons are available online 1 .

FIG4 shows more results on a variety of datasets.

We can see that for all these tasks, our model can learn the region of interest and generate compositions that are not only more faithful to the input, but also have fewer artifacts.

For example, in dog to cat translation, we notice most attention maps BID7 .

Given the source and target domains are globally different, the initial translation and final result are similar with the attention maps focusing on the entire images.

have large values around the eyes, indicating the eyes are key ROI to differentiate cats from dogs.

In the examples of photo to DSLR, the ROI should be the background that we wish to defocus, while the initial translation changes the color of the foreground flower in the photo.

The final result, on the other hand, learns to keep the color of the foreground flower.

In the second example of summer to winter translation, we notice the initial result incorrectly changes color of the person.

With the guidance of attention map, the final result removes such artifacts.

In a few scenarios, the attention map is less useful as the image does not explicitly contain region of interest and should be translated everywhere.

In this case, the composited results largely rely on the initial prediction given by G 0 .

This is true for tasks like edges to shoes/handbags, SYNTHIA to cityscape FIG5 ) and photo to map (Fig. 8) .

Although many of these tasks have very different source and target domains, our method is general and can be applied to get satisfying results.

To better demonstrate the effectiveness of our simple model, Fig. 6 shows several results before training with the attention branch and compares with baseline.

We can see that even without the attention branch, our model generates better qualitative results comparing with Cycle-GAN and UNIT.User study: To more rigorously evaluate the performance, we perform a user study to compare the results.

The procedure is as following: we asked for feedbacks from 10 users (all are graduate students).

Each user is given 30 sets of images to compare.

Each set has 5 images, which are the input, initial result (w/o attention), final result (with attention), CycleGAN results and UNIT results.

In total there are 300 different image sets randomly selected from several image translation tasks.

The images in each set are in random order.

The user is then asked to rank the four results from highest visual quality to lowest.

The user is fully informed about the task and is aware of the goal as to translate the input image into a new domain while avoiding unnecessary changes.

DISPLAYFORM0 Figure 7: Effects of using different layers as feature extractors.

From left to right: input (a), using the first two layers of VGG (b), using the last two layers of VGG (c) and using the first three layers of VGG (d).

Effects of using different layers as feature extractors: We experimented using different layers of VGG-19 as feature extractors to measure the perceptual loss.

Fig. 7 shows visual example of the horse to zebra image translation results trained with different perceptual terms.

We can see that only using high-level features as regularization leads to results that are almost identical to the input (Fig. 7 (c) ) while only using low-level features as regularization leads to results that are blurry and noisy ( Fig. 7 (b) ).

We find the balance by adopting the first three layers of VGG-19 as feature extractor which does a good job of image translation and also avoids introducing too many noise or artifacts ( Fig. 7 (d) ).

Map prediction: We translate images from satellite photos to maps with unpaired training data and compute the pixel accuracy of predicted maps.

The original photo-map dataset consists of 1096 training pairs and 1098 testing pairs, where each pair contains a satellite photo and the corresponding map.

To enable unsupervised learning, we take the 1096 photos from the training set and the 1098 maps from the test set, using them as the training data.

Note that no attention is used here since the change is global and we observe training with attention yields similar results.

At test time, we translate the test set photos to maps and again compute the accuracy.

If the total RGB difference between the color of a pixel on the predicted map and that on the ground truth is larger than 12, we mark the pixel as wrong.

Figure 8 and TAB3 show the visual results and the accuracy results, and we can see our approach achieves highest map prediction accuracy.

Note that Pix2Pix is trained with paired data.

Unsupervised classification: We show unsupervised classification results on USPS Denker et al. (1989) and MNIST-M Ganin et al. (2016) in FIG7 and TAB5 .

On both tasks, we assume we have access to labeled MNIST dataset.

We first train a generator that maps MNIST to USPS or MNIST-M and then use the translated image and original label to train the classifier (we do not apply the attention branch here as we did not observe much difference after training with attention).

We can see from the results that we achieve the highest accuracy on both tasks, advancing stateof-the-art.

The qualitative results clearly show that our MNIST-translated images both preserve the original label and are also visually similar to USPS/MNIST-M.3DMM face shape prediction: As a real-world application, we study the problem of estimating 3D face shape, which is modeled with the 3D morphable model (3DMM) BID2 .

For a given face, the 3DMM encodes its shape with a 100 dimension vector.

The goal of 3DMM regression is to predict the 100 dimension vector and we compare them with the ground truth using mean squared error (MSE).

BID49 proposes to train a very deep neural network BID16 for 3DMM regression.

However, in reality, the labeled training data for real faces are expensive to collect.

We propose to use rendered faces instead, as their 3DMM parameters are readily available.

We first rendered 200k faces as the source domain and use human selfie photo data of 645 face images we collected as the target domain.

For test, we use our collected 112 3D-scanned faces as test data.

For the purpose of domain adaptation, we first use our model to translate the rendered faces to real faces and use the results as the training data, assuming the 3DMM parameters stay unchanged.

The 3DMM regression model structure is 102-layer Resnet BID16 as in BID49 , and was trained with the translated faces.

FIG8 and TAB6 show the qualitative results and the final accuracy of 3DMM regression.

From the visual results, we see that our translated face preserves the shape of the original rendered face and has higher quality than using CycleGAN.

We also reduced the 3DMM regression error compared with baseline (where we trained on rendered faces and tested on real faces) and the CycleGAN results.

We propose to use a simple model with attention for image translation and domain adaption and achieve superior performance in a variety of tasks demonstrated by both qualitative and quantitative measures.

We show that the attention module is particularly helpful to focus the translation on region of interest, remove unwanted changes or artifacts, and may also be used for unsupervised segmentation or saliency detection.

@highlight

We propose a simple generative model for unsupervised image translation and saliency detection.