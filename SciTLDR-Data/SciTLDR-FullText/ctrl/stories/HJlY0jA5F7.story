In this paper, we propose an improved quantitative evaluation framework for Generative Adversarial Networks (GANs) on generating domain-specific images, where we improve conventional evaluation methods on two levels: the feature representation and the evaluation metric.

Unlike most existing evaluation frameworks which transfer the representation of ImageNet inception model to map images onto the feature space, our framework uses a specialized encoder to acquire fine-grained domain-specific representation.

Moreover, for datasets with multiple classes, we propose Class-Aware Frechet Distance (CAFD), which employs a Gaussian mixture model on the feature space to better fit the multi-manifold feature distribution.

Experiments and analysis on both the feature level and the image level were conducted to demonstrate improvements of our proposed framework over the recently proposed state-of-the-art FID method.

To our best knowledge, we are the first to provide counter examples where FID gives inconsistent results with human judgments.

It is shown in the experiments that our framework is able to overcome the shortness of FID and improves robustness.

Code will be made available.

Generative Adversarial Networks (GANs) have shown outstanding abilities on many computer vision tasks including generating domain-specific images BID7 , style transfer , super resolution BID20 , etc.

The basic idea of GANs is to hold a two-player game between generator and discriminator, where the discriminator aims to distinguish between real and fake samples while the generator tries to generate samples as real as possible to fool the discriminator.

Researchers have been continuously exploring better GAN architectures.

However, developing a widely-accepted GAN evaluation framework remains to be a challenging topic BID35 .

Due to a lack of GAN benchmark results, newly proposed GAN variants are validated on different evaluation frameworks and therefore incomparable.

Because human judgements are inherently limited by manpower resource, good quantitative evaluation frameworks are of very high importance to guide future research on designing, selecting, and interpreting GAN models.

There have been varieties of efforts on designing sample-based evaluation for GANs on its ability of generating domain-specific images.

The goal is to measure the distance between the generated samples and the real in the dataset.

Most existing methods utilized the ImageNet BID29 inception model to map images onto the feature space.

The most widely used criteria is probably the Inception Score BID31 , which measures the distance via Kullback-Leiber Divergence (KLD).

However, it is probability based and is unable to report overfitting.

Recently, Frechet Inception Distance (FID) was proposed BID11 on improving Inception Score.

It directly measures Frechet Distance on the feature space with the Gaussian assumption.

It has been proved that FID is far better than Inception Score BID13 BID15 BID24 .

However, we argue that assuming normality on the whole feature distribution may lose class information on labeled datasets.

In this work, we propose an improved quantitative sample-based evaluating criteria.

We improve conventional evaluation methods on two levels: the feature representation and the evaluation metric.

Unlike most existing methods including the Inception Score BID31 and FID BID11 , our framework uses a specialized encoder trained on the dataset to get domain-specific representation.

We argue that applying the ImageNet model to either labeled or unlabeled datasets is ineffective.

Moreover, we propose Class-Aware Frechet Distance (CAFD) in our framework to measure the distribution distance of each class (mode) respectively on the feature space to include class information.

Instead of the single Gaussian assumption, we employ a Gaussian mixture model (GMM) to better fit the feature distribution.

We also include KL divergence (KLD) between mode distribution of real data and generated samples into the framework to help detect mode dropping.

Experiments and analysis on both the feature level and the image level were conducted to demonstrate the improved effectiveness of our proposed framework.

To our best knowledge, we are the first BID4 to provide counter examples where FID is inconsistent with human judgements (See FIG0 ).

It is shown in the experiments that our framework is able to overcome the shortness of existing methods.

Evaluation Methods.

Several GAN evaluation methods have been proposed by researchers.

While model-based methods including Parzen window estimation and the annealed importance sampling (AIS) BID36 ) require either density estimation or observation on the inner structure of the decoder, model-agnostic methods BID11 BID31 BID15 BID23 are more popular in the GAN community.

These methods are sample based.

Most of them map images onto the feature space via an ImageNet pretrained model and measure the similarity of the distribution between the dataset and the generated data.

Maximum mean discrepancy (MMD) was proposed by and it has been further used in classifier two-sample tests BID23 , where statistical hypothesis testing is used to assess whether two sample sets are from the same distribution.

Inception Score BID31 , along with its improved version Mode Score , was the most widely used metric in the last two years.

FID BID11 was proposed on improving the Inception Score.

Recently, several interesting methods were also proposed including classification accuracy BID33 , precisionrecall measuring BID30 and skill rating BID26 .

These metrics give complementary perspectives towards sample-based methods.

Studies on Existing Frameworks.

It is common BID2 in the literature to see algorithms which use existing metrics to optimize early stopping, hyperparameter tuning, and even model architecture.

Thus, comparison and analysis on previous evaluation methods have been attracting more and more attention recently BID35 BID13 BID15 BID24 .

While Inception Score was the most popular metric in the last two years, it was believed to be misleading in recent literature BID11 BID13 BID24 BID4 BID2 .

Applying the ImageNet model to encode features in Inception Score is ineffective BID35 BID2 BID28 .

The recently proposed FID has been proved to be far better than Inception Score BID11 BID13 BID15 .

And its robustness was experimentally demonstrated recently in a technical report BID24 .

However, we argue that FID still has problems and provide counter examples where FID gives inconsistent results with human judgements.

Moreover, we propose an improved version of evaluation which overcomes its shortness.

The evaluation problem can be formulated as modeling the distance between two distributions P r and P g , where P r denotes the distribution of real samples in the dataset and P g denotes the distributions of new samples generated by GAN models.

The main difficulties for GANs on generating domain-specific images can be summarized into three types below.• Lack of generating ability.

Either the generator cannot generate useful samples or the GAN training cannot diverge.• Mode collapse.

Different modes collapse to a new mixed mode in the generated samples. (e.g. An animal resembling both a horse and a deer.)• Mode dropping.

Only part of the modes in the dataset are generated while some modes are implicitly ignored.

(e.g. The handwritten 5 can hardly be generated by GAN trained on MNIST.)

Therefore, a good evaluation framework should be consistent to human judgements, penalize on mode collapse and mode dropping.

Most of the conventional methods utilized an ImageNet pretrained inception model to map images onto the feature space.

Inception Score, which was originally formulated as Eq. FORMULA0 , ignored information in the dataset completely.

Thus, its original formulation was considered to be relatively misleading.

DISPLAYFORM0 The Mode Score was proposed to overcome this shortness.

Its formulation is shown in Eq. (2).

By including the prior distribution of the ground truth labels, Mode Score improved Inception Score (Che et al., 2017) on reporting mode dropping.

DISPLAYFORM1 FID BID11 , which was formulated in Eq. (3), was proposed on improving Inception Score BID31 .

DISPLAYFORM2 (µ g , C g ), (µ r , C r ) are the first-order and second-order statistics for generated samples and real data respectively.

Unlike the previous two metrics which are probability-based, FID directly measures Frechet distance on the feature space.

It uses an ImageNet model for encoding features and assumes normality on the whole feature distribution.

FID was believed to be better than Inception Score BID13 BID15 BID24 .

However, we argue that FID still has two major problems (See Section 3.1 and 3.2).

As both Inception Score BID31 and Mode Score ) is probabilitybased, applying the ImageNet pretrained model on non-ImageNet dataset is relatively meaningless.

This misuse of representation on Inception Score was mentioned previously BID28 .

However, we argue that applying the ImageNet model to map the generated images to the feature space in FID can also be misleading.

While both of the BID2 BID28 mentioned that applying the ImageNet pretrained model to the probability-based metric Inception Score BID31 ) is inadequate, the trend for applying it to feature-based metric such as FID BID11 ) is widely followed.

BID2 pointed out that because classes are unmatched, the p(y|x) and p(y * ) in the formulation of Inception Score are meaningless.

However, we argue that applying the ImageNet model for mapping the generated images to the feature space in FID can also be misleading for the two reasons below.

First, On labeled datasets with multiple classes, the class labels unmatch those in ImageNet.

For example, the class 'Bird' in CIFAR-10 ( BID19 ) is divided into several sophisticated category labels in ImageNet.

This will make the CNN representations trained on ImageNet is either meaningless or over-complicated.

Specifically, some features distinguishing the "acoustic guitar" from "electric guitar" are hardly useful on CIFAR-10 while fine-grained features distinguishing "African hunting dog" from "Cape hunting dog" (which all belong to the category "dog" in CIFAR-10) are not needed as well.

On unlabeled datasets with images from a single class such as CelebA , applying the ImageNet inception model is also inappropriate.

The categories of ImageNet labels are so sophisticated that the trained model needs to encode diverse features on various objects.

However, this will get encoded features limited to a relatively low-dimensional subspace lack of fine-grained information.

For example, the ImageNet models can hardly distinguish different faces.

In Section 5.1, we designed experiments on both the feature level and the image level to demonstrate the effects of using different representations.

We argue that the single Gaussian assumption in FID is over-simplified.

As the training decreases intra-class distance and increases inter-class distance, the features are distributed in groups by their class labels.

Thus, we propose that on datasets with multiple classes, the feature distribution is better fitted by a Gaussian mixture model.

Considering the specific Gaussian mixture model where x ∼ N (µ i , C i ) with probability p i , we can derive the first and second moment of the feature distribution in Eq. (4) and Eq. (5).

DISPLAYFORM0 It should be noted that when the feature is n-dimensional and there are K classes in total, there are a total of K( n 2 +n 2 + n + 1) variables in the model.

However, directly modeling the whole distribution Gaussian as in FID will result in n 2 +n 2 + n degrees of freedom, which is a relatively small number.

Thus, FID detects mode-related problems in an implicit way.

Either simply dropping a mode or linearly combining images increases FID by unintentionally changing the mean µ. However, FID gets to be misleading when the deficiency type becomes more complicated (See FIG1 .

As discussed in Section 3.1, applying the ImageNet inception model to either labeled or unlabeled datasets is ineffective.

We argue that a specialized domain-specific encoder should be used for sample-based evaluation.

While the features encoded by the ImageNet model are limited within a low-dimensional subspace, the domain-specific model could encode more fine-grained information, making the encoded features much more effective.

Specifically, we propose to use the widely used variational autoencoder (VAE) BID17 to acquire the specialized embedding for a specific dataset.

In labeled datasets, we can add a cross-entropy loss for training the VAE model.

In Section 5.1, we show that simply training an autoencoder can already get better domain-specific representations on CelebA .

Before introducing our improved evaluation metric, we would firstly take a step back towards existing popular metrics.

Both Inception Score BID31 and Mode Score measure distance between probability distribution while FID BID11 directly measures distance on the feature space.

Probability-based metrics better handle mode-related problems (with the correct use of a domain-specific encoder), while directly measuring distance between features better models the generating ability.

In fact, we believe these two perspectives are complementary.

Thus, we propose a class-aware metric on the feature space to combine the two perspectives together.

For datasets with multiple classes, the feature distribution is better fit with mixture Gaussian (See Section 3.2).

Thus, we propose Class-Aware Frechet Distance (CAFD) to include class information.

Specifically, we compute probability-based Frechet Distance between real data and generated samples in each class respectively.

class 0 1 2 3 4 5 dist 64.8 ± 0.5 18.9 ± 0.2 80.5 ± 1.1 81.3 ± 0.3 64.5 ± 0.6 79.0 ± 0.4 class 6 7 8 9 average dist 65.2 ± 0.3 46.8 ± 0.3 90.4 ± 0.3 59.8 ± 0.2 65.1 ± 0.4As previously discussed in Section 4.1, we train a domain-specific VAE along with the cross entropy on datasets with multiple classes and use its learned representations.

In our evaluation framework, we also made use of the predicted probability p(y|x).

To calculate the expected mean of each class in a specific set S of generated samples, we can derive the formulation below in Eq. (6).

DISPLAYFORM0 where DISPLAYFORM1 Similarly, The covariance matrix in each class is shown in Eq. (8).

DISPLAYFORM2 We compute Frechet distance in each of the K classes and average the results to get Class-Aware Frechet Distance (CAFD) in Eq. (9).

DISPLAYFORM3 This improved form based on mixture Gaussian assumption can better evaluate the actual distance compared to the original FID.

Moreover, when CAFD is applied to evaluating a specific GAN model, we could get better class-aware understanding towards the generating ability.

For example, as shown in TAB0 , the selected model generates digit 1 well but struggles on other classes.

This information will provide guidance for researchers on how well their generative models perform on each mode and may explain what specific problems exist.

As both FID and CAFD aim to model how well domain-specific images are generated, they are not designed to deal with mode dropping, where some of the modes are missed in the generated samples.

Thus, motivated by Mode Score (Che et al., 2017), we propose that KL divergence KL(p(y * )||p(y)) should be included as auxiliary scores into the evaluation framework.

To sum up, the correct use of encoder, the CAFD and the KL divergence term combine for a complete sample-based evaluation framework.

Our proposed method combines the advantages of Inception Score BID31 , Mode Score (Che et al., 2017) and FID BID11 and overcomes their shortness.

Our method is sensitive to different representations.

Different selection of encoders can result in changes on the evaluation results.

Experiments in Section 5.1 demonstrate that the ImageNet inception model will give misleading results (See FIG0 .

Thus, a domain-specific encoder should be used in each evaluation pipeline.

Because the representation is not fixed, the correct use (with

In this section, we study the representation for mapping the generated images onto the feature space.

As discussed in Section 4.1, applying the pretrained ImageNet inception model to sample-based evaluation methods is inappropriate.

We firstly investigated the features generated by different encoders on CelebA , which is a widely used dataset containing more than 200k face images.

Then, we gave an intuitive demonstration where FID BID11 using ImageNet pretrained representations gives inconsistent results with human judgements.

We give two proposals of domain-specific encoders in the experiment: an autoencoder and a VAE BID17 .

Both proposed encoders share a similar network architecture which is the inverse structure of the 4-conv DCGAN .

The embedding is dimensioned 2048, which is the same as the dimension of ImageNet features.

We train both models for 25 epochs.

The loss weight of the KLD term in VAE is 1e-5.

We conducted principle component analysis (PCA) on three feature sets encoded on CelebA : 1) ImageNet inception model.

2) proposed autoencoder 3) proposed VAE.

TAB1 shows the percent of explained variance on the first 5 components.

Although the ImageNet model should have much greater representation capability than the 4-conv encoder, its first two components has much higher explained variance (9.35% and 7.04%).

This supports our claim that the features encoded by ImageNet are limited in a low-dimensional subspace.

It can be also noted that VAE better makes use of the feature space compared to the naive autoencoder.

To better demonstrate the deficiency of the ImageNet model, we performed three different types of adjustments on the first 10,000 images on CelebA : a) Random noise uniformly distributed in [-33,33 ] was applied on each pixel.

b) Each image was divided into 8x8=64 regions and seven of them were sheltered by a pixel sampled from the face.

c) Each image was first divided into 4x4=16 regions and random exchanges were performed twice. .

The ImageNet inception model fails to encode fine-grained features on faces.

a) Random noise uniformly distributed in [-33,33] was applied on each pixel.

b) Each image was divided into 8x8=64 regions and seven of them were sheltered by a pixel sampled from the face.

c) Each image was first divided into 4x4=16 regions and random exchanges were performed twice.

Results are shown in FIG0 .

With the ImageNet inception model, it is obvious that FID gave inconsistent results with human judgements (See TAB4 ).

In fact, when similar adjustments were conducted with the overall color maintained, FID fluctuated within only a small range.

The ImageNet model mainly extracts general features on color, shape to better classify objects in the world while domain-specific facial textures cannot be well represented.

For comparison, we applied the trained autoencoder and VAE onto the case.

Also, we tried to apply the representation of the discriminator after GAN training, which was previously proposed in .

Specifically, we use the features right before the final fc layer for the discriminator.

Results are shown in TAB2 .

It is shown that only representations derived from the domain-specific encoder including the autoencoder and VAE are effective and give results consistent with human judgements.

The discriminator which learns to discriminate fake samples from the real cannot learn good representation for distance measurement.

Thus, for datasets where images are from a single class such as CelebA and LSUN Bedrooms BID38 , the representation should be acquired via training a domain-specific encoder such as a VAE.

In this way our sample-based evaluation employs specialized representations, which can provide more fine-grained information related to the specific domain.

In this section, we used the domain-specific representations and studied the improvements of the evaluation metric CAFD proposed in our framework against the state-of-the-art metric FID BID11 .

In datasets with multiple classes, the Gaussian mixture model in CAFD will better fit the feature distribution.

First, we performed user study to demonstrate the improved consistency of our method.

Then, An intuitive case for further demonstration is given where CAFD shows great robustness while FID fails to give consistent results with human judgements.

For implementation details, on the MNIST dataset, we trained a variational autoencoder (VAE) BID17 with the kl loss weight 1e-5 for the specialized encoder and added the cross-entropy term with a loss weight of 1.0.

BID8 .

We use the domain-specific representation of VAE for embedding images. (See TAB6 5.2.1 USER STUDY Evaluating the evaluation metrics is a non-trivial task, as the best criterion is the consistency with human judgements.

Therefore, we performed user study to compare our proposed method with the existing ones including Inception Score BID31 , Mode Score and FID BID11 .

Our setting is consistent with BID15 .

15 volunteers were first trained to tell generated samples from the groundtruth in the dataset.

Then, paired image sets were randomly sampled and volunteers were asked to tell the better sets.

Finally, we counted pairs where the metric agreed the voted results by the volunteers.

We conducted experiments on MNIST with two settings for the experiments: 'easy' and 'hard'.

The 'easy' setting is where random pairs are sampled from the intermediate results of GAN training, while the 'hard' setting is where only random pairs with the difference of FID of two sampled sets within a threshold are included.

TAB3 shows the results.

It is worth noting that in hard cases, the results of Inception Score BID31 are relatively meaningless (50%), which makes it hard to be applied as guidance for improving the quality of generated images by GANs.

In both 'easy' and 'hard' settings, our method gets consistent gain compared to baseline approaches.

In this experiment, we gave an intuitive case where FID fails to give consistent results with human judgements.

We used two different settings of representations and focused on the evaluation metric within each setting.

Specifically, Besides the VAE, we also train a classifier on MNIST and use its representation as a supporting experimental setting.

BID7 .

We use two setting of different representations in this experiment: a domain-specific classifier and a VAE.

For VAE, 'generated' and 'hack' are the sampled images in FIG1 .

Compared to FID, CAFD are more robust to feature-level adjustments.

FID, as an overall statistical measure, is able to detect either a single mode dropping or a trivial linear combination of two images.

However, as its formulation has relatively limited constraints, it can be hacked in complicated scenarios.

Considering the features extracted from MNIST test data, which has a zero FID with itself.

We performed operations below on the features.

Step 1 Performed principle component analysis (PCA) on the original features.

Step 2 Normalized each axis to zero mean and unit variance.

Step 3 Switched the normalized projection of the first two component.

Step 4 Unnormalized the data and reconstructed features.

The adjusted features are completely different with the original one with zero FID maintained.

The over-simplified Gaussian assumption on overall distribution cannot tell the differences while our proposed method is able to report the changes with CAFD raising from 0 to 246.2 (539.8) for VAE (classifier). (See TAB6 Furthermore, We used FGSM BID8 to reconstruct the images from the adjusted features in both settings.

Specifically, we first trained an decoder for initialization via an AutoEncoder with the encoder fixed.

Then, we performed pixelwise adjustment via FGSM BID8 to lower the reconstruction error.

Because the used encoder has a relatively simple structure, the final reconstruction error is still relatively high after optimized.

For comparison, We trained a simple WGAN-GP model and took samples (generated by intermediate models during training) with comparable FID with our constructed images.

Visualization for the VAE setting are shown in FIG1 .It is obvious that the quality of constructed images are much worse than the generated samples.

After axis permutation, the constructed images suffers from mode collapse.

There are many pictures in the right which resemble more than one digits and are hard to recognize.

However, for the VAE (classifier) setting, it still received a FID of 25.4 (72.8) lower than 49.9 (73.1) received by generated samples.

For comparison, The results of CAFD on these cases are shown in TAB6 .

While FID gives misleading results, CAFD are much more robust on the adjusted features.

Compared to the constructed images (211.6 (468.6)), the generated images received a much lower CAFD (80.7 (201.4)), which is consistent with human judgements. (See Table 6 ) Thus, results for both settings demonstrates the improved effectiveness of the evaluation metric in our proposed evaluation framework.

In this paper, we aimed to tackle the very important problem of evaluating the Generative Adversarial Networks.

We presented an improved sample-based evaluation, which improves conventional methods on both representation and evaluation metric.

We argue that a domain-specific encoder is needed and propose Class-Aware Frechet Distance to better fit the feature distribution.

To our best knowledge, we are the first to provide counter examples where the state-of-the-art FID method is inconsistent with human judgements.

Experiments and analysis on both the feature level and the image level have shown that our framework is more effective.

Therefore, the encoder should be specifically trained for datasets of which the labels are different from ImageNet.

To attain effective representations on non-ImageNet datasets, we need to ensure that the class labels of data used for training GAN models are consistent with those of data used for training the encoder.

The Gaussian assumption on the features were commonly used in the literature.

Although there are non-linear operations such as relu and max-pooling in the neural network, assuming the normality simplifies the model and enables numerical expression.

However, in labeled dataset with multiple classes, the Gaussian assumption is relatively over-simplified.

In this experiment, we performed Anderson-Darling test (AD-test) BID32 to quantatively study the normality of the data.

Specifically, to test the multivariate normality on a set of features, we first performed principle component analysis (PCA) on the data, and then applied AD-test to the first 10 components and averaged the results.

We compared the test results on each class and the whole training set on MNIST.

We used a simple 2-conv structure trained on the MNIST classification task as our feature encoder with a output dimension 1024.

To reduce the influence of sample number on the results, we divided the whole features randomly into 10 sets to study the normality of the mixed features.

Results are shown in Table 9 .

Although the p-value of both features are small, features within a single class get much greater results than the mixed features.

It can be inferred that compared to the whole training set, features within each class are much more Gaussian.

Thus, the basic assumption of CAFD in our proposed framework is more reasonable compared to the FID BID11 method.

The idea of Generative Adversarial Network was originally proposed in BID7 .

It has been applied to various computer vision tasks BID20 BID40 .

Researchers have been continuously developing better GAN architectures BID10 BID14 and training strategies BID1 BID12 on generating domain-specific images.

Deep convolutional networks were firstly in-

Table 9 : P-value results of AD-test BID32 on features of each class and the whole training images.

The whole features were randomly divided into 10 sets.

Compared to the mixed features, features encoding images from a single class are more Gaussian.

<|TLDR|>

@highlight

This paper improves existing sample-based evaluation for GANs and contains some insightful experiments.