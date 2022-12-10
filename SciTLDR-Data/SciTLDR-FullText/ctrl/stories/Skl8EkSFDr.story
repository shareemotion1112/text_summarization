Deep learning's success has led to larger and larger models to handle more and more complex tasks; trained models can contain millions of parameters.

These large models are compute- and memory-intensive, which makes it a challenge to deploy them with minimized latency, throughput, and storage requirements.

Some model compression methods have been successfully applied on image classification and detection or language models, but there has been very little work compressing generative adversarial networks (GANs) performing complex tasks.

In this paper, we show that a standard model compression technique, weight pruning, cannot be applied to GANs using existing methods.

We then develop a self-supervised compression technique which uses the trained discriminator to supervise the training of a compressed generator.

We show that this framework has a compelling performance to high degrees of sparsity, generalizes well to new tasks and models, and enables meaningful comparisons between different pruning granularities.

Deep Neural Networks (DNNs) have proved successful in various tasks like computer vision, natural language processing, recommendation systems, and autonomous driving.

Modern networks are comprised of millions of parameters, requiring significant storage and computational effort.

Though accelerators such as GPUs make realtime performance more accessible, compressing networks for faster inference and simpler deployment is an active area of research.

Compression techniques have been applied to many networks, reducing memory requirements and improving their performance.

Though these approaches do not always harm accuracy, aggressive compression can adversely affect the behavior of the network.

Distillation (Schmidhuber, 1991; Hinton et al., 2015) can improve the accuracy of a compressed network by using information from the original, uncompressed network.

Generative Adversarial Networks (GANs) (Schmidhuber, 1990; Goodfellow et al., 2014) are a class of DNN that consist of two sub-networks: a generative model and a discriminative model.

Their training process aims to achieve a Nash Equilibrium between these two sub-models.

GANs have been used in semi-supervised and unsupervised learning areas, such as fake dataset synthesis (Radford et al., 2016; Brock et al., 2019) , style transfer (Zhu et al., 2017b; Azadi et al., 2018) , and image-to-image translation (Zhu et al., 2017a; Choi et al., 2018) .

As with networks used in other tasks, GANs have millions of parameters and nontrivial computational requirements.

In this work, we explore compressing the generative model of GANs for more efficient deployment.

We show that applying standard pruning techniques, with and without distillation, can cause the generator's behavior to no longer achieve the network's goal.

Similarly, past work targeted at compressing GANs for simple image synthesis fall short when they are applied to large tasks.

In some cases, this result is masked by loss curves that look identical to the original training.

By modifying the loss function with a novel combination of the pre-trained discriminator and the original and compressed generators, we can overcome this behavioral degradation and achieve compelling compression rates with little change in the quality of the compressed generator's ouput.

We apply our technique to several networks and tasks to show generality.

Finally, we study the behavior of compressed generators when pruned with different amounts and types of sparsity, finding that filter pruning, a technique commonly used for accelerating image classification networks, is not trivially applicable to GANs.

A complementary method of network compression is quantization.

Sharing weight values among a collection of similar weights by hashing (Chen et al., 2015) or clustering (Han et al., 2016) can save storage and bandwidth at runtime.

Changing fundamental data types adds the ability to accelerate the arithmetic operations, both in training (Micikevicius et al., 2018) and inference regimes (Jain et al., 2019) .

Several techniques have been devised to combat lost accuracy due to compression, since there is always the chance that the behavior of the network may change in undesirable ways when the network is compressed.

Using GANs to generate unique training data (Liu et al., 2018b) and extracting knowledge from an uncompressed network, known as distillation (Hinton et al., 2015) , can help keep accuracy high.

Since the pruning process involves many hyperparameters, Lin et al. (2019) use a GAN to guide pruning, and Wang et al. (2019a) structure compression as a reinforcement learning problem; both remove some of the burden from the user.

Though there are two networks in a single GAN, the main workload at deployment is usually from the generative model, or generator.

For example, in image synthesis and style transfer tasks, the final output images are created solely by the generator.

The discriminative model (discriminator) is vital in training, but it is abandoned afterward for many tasks.

So, when we try to apply state-of-the-art compression methods to GANs, we focus on the generator for efficient deployment.

As we will see, the generative performance of the compressed generators is quite poor for the selected image-toimage translation task.

We look at two broad categories of baseline approaches: standard pruning techniques that have been applied to other network architectures, and techniques that were devised to compress the generator of a GAN performing image synthesis.

We compare to the dense baseline [a], our technique [b] , as well as a small, dense network with the same number of parameters [c] .

(Labels correspond to entries in Table 1 , the overview of all techniques, and Figure 1 , results of each technique).

Standard Pruning Techniques.

To motivate GAN-specific compression methods, we try variations of two state-of-the-art pruning methods: manually pruning and fine tuning (Han et al., 2015 ) a trained dense model [d] , and AGP (Zhu & Gupta, 2018) from scratch [e] and during fine-tuning [f].

We also include distillation (Hinton et al., 2015) to improve the performance of the pruned network with manual pruning [g] and AGP fine-tuning [h] .

Distillation is typically optional for other network types, since it is possible to get decent accuracy with moderate pruning in isolation.

For very aggressive compression or challenging tasks, distillation aims to extract knowledge for the compressed (student) network from original (teacher) network's behavior.

We also fix the discriminator of [g] to see if the discriminator was being weakened by the compressed generator [i] .

Targeted GAN Compression.

There has been some work in compressing GANs with methods other than pruning, and only one technique applied to an image-to-image translation task.

We first examine two approaches similar to ours.

Adversarial training [j] posits that during distillation of a classification network, the student network can be thought of as a generative model attempting to produce features similar to that of the teacher model.

So, a discriminator was trained alongside the student network, trying to distinguish between the student and the teacher.

One could apply this technique to compress the generator of a GAN, but we find that its key shortcoming is that it trains a discriminator from scratch.

Similarly, distillation has been used to compress GANs in Aguinaldo et al. (2019) [k], but again, the "teacher" discriminator was not used when teaching the "student" generator.

Learned Intermediate Representation Training (LIT) (Koratana et al., 2019) [l] compresses StarGAN by a factor of 1.8× by training a shallower network.

Crucially, LIT does not use the pre-trained discriminator in any loss function.

Quantized GANs (QGAN) (Wang et al., 2019b) [m] use a training process based on Expectation-Maximization to achieve impressive compression results on small generative tasks with output images of 32x32 or 64x64 pixels.

Liu et al. (2018a) find that maintaining a balance between discriminator and generator is key: their approach is to selectively binarize parts of both networks in the training process on the Celeb-A generative task, up to 64x64 pixels.

So, we try pruning both networks during the training process [n] .

Experiments.

For these experiments, we use StarGAN (Choi et al., 2018) trained with the Distiller (Zmora et al., 2018) library for the pruning.

StarGAN 1 extends the image-to-image translation capability from two domains to multiple domains within a single unified model.

It uses the CelebFaces Attributes (CelebA) (Liu et al., 2015) as the dataset.

CelebA contains 202,599 images of celebrities' faces, each annotated with 40 binary attributes.

As in the original work, we crop the initial images from size 178 × 218 to 178 × 178, then resize them to 128 × 128 and randomly select 2,000 images as the test dataset and use remaining images for training.

The aim of StarGAN is facial attribute translation: given some image of a face, it generates new images with five domain attributes changed: 3 different hair colors (black, blond, brown), different gender (male/female), and different age (young/old).

Our target sparsity is 50% for each approach.

We stress that we attempted to find good hyperparameters when using the existing techniques, but standard approaches like reducing the learning rate for fine-tuning (Han et al., 2015) , etc., were not helpful.

Further, the target sparsity, 50%, is not overly aggressive, and we do not impose any structure; other tasks readily achieve 80%-90% fine-grained sparsity with minimal accuracy impact.

The results of these trials are shown in Figure 1 .

Subjectively, it is easy to see that the existing approaches (1c through 1n) produce inferior results to the original, dense generator.

Translated facial images from pruning & naïve fine-tuning (1d and 1e) do give unique results for each latent variable, but the images are hardly recognizable as faces.

These fine-tuning procedures, along with AGP from scratch (1f) and distillation from intermediate representations (1l), simply did not converge.

One-shot pruning and traditional distillation (1g), adversarial learning (1j), knowledge distillation (1k), training a "smaller, dense" half-sized network from scratch (1c) and pruning both generator and discriminator (1n) keep facial features intact, but the image-to-image translation effects are lost to mode collapse (see below).

There are obvious mosaic textures and color distortion on the translated images from fine-tuning & distillation (1h), without fine-tuning the original loss (1i), and from the pruned model based on the Expectation-Maximization (E-M) algorithm (1m).

However, the translated facial images from a generator compressed with our proposed self-supervised GAN compression method (1b) are more natural, nearly indistinguishable from the dense baseline (1a), matching the quantitative Frechet Inception Distance (FID) scores (Heusel et al., 2017) in Table 1 .

While past approaches have worked to prune some networks on other tasks (DCGAN generating MNIST digits, see the supplementary material), we show that they do not succeed on larger imageto-image translation tasks, while our approach works on both.

Similarly, though LIT (Koratana et al., 2019) [l] was able to achieve a compression rate of 1.8× on this task by training a shallower network, it does not see the same success at network pruning.

Discussion.

It is tempting to think that the loss curves of the experiment for each technique can tell us if the result is good or not.

We found that for many of these experiments, the loss curves correctly predicted that the final result would be poor.

However, the curves for [h] and [m] look very good -the compressed generator and discriminator losses converge at 0, just as they did for baseline training.

It is clear from the results of querying the generative models (Figures 1h and 1m ), though, that this promising convergence is a false positive.

In contrast, the curves for our technique predict good performance, and, as we prune more aggressively in Section 6, higher loss values correlate well with worsening FID scores.

(Loss curves are provided in the Appendix.)

As pruning and distillation are very effective when compressing models for image classification tasks, why do they fail to compress this generative model?

We share three potential reasons:

1.

Standard pruning techniques need explicit evaluation metrics; softmax easily reflects the probability distribution and classification accuracy.

GANs are typically evaluated subjectively, though some imperfect quantitative metrics have been devised.

2. GAN training is relatively unstable (Arjovsky et al., 2017; Liu et al., 2018a) and sensitive to hyperparameters.

The generator and discriminator must be well-matched, and pruning can disrupt this fine balance.

3.

The energy of the input and output of a GAN is roughly constant, but other tasks, such as classification, produce an output (1-hot label vector) with much less entropy than the input (three-channel color image of thousands of pixels).

Elaborating on this last point, there is more tolerance in the reduced-information space for the compressed classification model to give the proper output.

That is, even if the probability distribution inferred by the original and compressed classification models are not exactly the same, the classified labels can be the same.

On the other hand, tasks like style-transfer and dataset synthesis have no obvious energy reduction.

We need to keep entropy as high as possible (Kumar et al., 2019) during the compression process to avoid mode collapse -generating the same output for different inputs or tasks.

Attempting to train a new discriminator to make the compressed generator behave more like the original generator suffers from this issue -the new discriminator quickly falls into a low-entropy solution and cannot escape.

Not only does this preclude its use on generative tasks, but it means that the compressed network for any task must also be trained from scratch during the distillation process, or the discriminator will never be able to learn.

We seek to solve each of the problems highlighted above.

Let us restate the general formulation of GAN training: the purpose of the generative model is to generate new samples which are very similar to the real samples, but the purpose of the discriminative model is to distinguish between real samples and those synthesized by the generator.

A fully-trained discriminator is good at spotting differences, but a well-trained generator will cause it to believe that the a generated sample is both real and generated with a probability of 0.5.

Our main insight follows:

By using this powerful discriminator that is already well-trained on the target data set, we can allow it to stand in as a quantitative subjective judge (point 1, above) -if the discriminator can't tell the difference between real data samples and those produced by the compressed generator, then the compressed generator is of the same quality as the uncompressed generator.

A human no longer needs to inspect the results to judge the quality of the compressed generator.

This also addresses our second point: by starting with a trained discriminator, we know it is well-matched to the generator and will not be overpowered.

Since it is so capable (there is no need to prune it to), it also helps to avoid mode collapse.

As distillation progresses, it can adapt to and induce fine changes in the compressed generator, which is initialized from the uncompressed generator.

Since the original discriminator is used as a proxy for a human's subjective evaluation, we refer to this as "self-supervised" compression.

We illustrate the workflow in Figure 2 , using a GAN charged with generating a map image from a satellite image in a domain translation task.

In the right part of Figure 2 , the real satellite image (x) goes through the original generative model (G O ) to produce a fake map image (ŷ o ).

The corresponding generative loss value is l-G O .

Accordingly, in the left part of Figure 2 , the real satellite image (x) goes through the compressed generative model (G C ) to produce a fake map image (ŷ c ).

The corresponding generative loss value is l-G C .

This is the inference process of the original and compressed generators, expressed as follows:

The overall generative difference is measured between the two corresponding generative losses 2 .

We use a generative consistent loss function (L GC ) in the bottom of Figure 2 to represent this process.

Since the GAN training process aims to reduce the differences between real and generated samples, we stick to this principle in the compression process.

In the upper right of Figure

So the discriminative difference is measured between two corresponding discriminative losses.

We use the discriminative consistent loss function L DC in the top of Figure 2 to represent this process.

The generative and discriminative consistent loss functions (L GC and L DC ) use the weighted normalized Euclidean distance.

Taking the StarGAN task as the example (other tasks may use different losses):

where l-Gen is the generation loss term, l-Cla is the classification loss term, and l-Rec is the reconstruction loss term.

α and β are the weight ratios among three loss types.

(We use the same values of α and β used in the original StarGAN baseline.)

where l-Dis is the discriminative loss item, l-GP is the gradient penalty loss item, and δ is a weighting factor (again, we use the same value as the baseline).

The overall loss function of GAN compression consists of generative and discriminative differences:

where λ is the parameter to adjust the percentages between generative and discriminative losses.

We showed promising results with this method above in the context of prior methods.

In the following experiments, we investigate how well the method applies to other networks and tasks (Section 5) and how well the method works on different sparsity ratios and pruning granularities (Section 6).

For the experiments in this section, we choose to prune individual weights in the generator.

The final sparsity rate is 50% for all convolution and deconvolution layers in the generator (more aggressive sparsities are discussed in Section 6).

Following AGP (Zhu & Gupta, 2018) , we gradually increase the sparsity from 5% at the beginning to our target of 50% halfway through the self-supervised training process, and we set the loss adjustment parameter λ to 0.5 in all experiments.

We use PyTorch (Paszke et al., 2017) , implement the pruning and training schedules with Distiller (Zmora et al., 2018) , and train and generate results with a V100 GPU (NVIDIA, 2017) using FP32 to match public baselines.

In all experiments, the data sets, data preparation, and baseline training all follow from the public repositories -details are summarized in Table 2 .

We start by assuming an extra 10% of the original number of epochs will be required; in some cases, we reduced the overhead to only 1% while maintaining subjective quality.

We include representative results for each task, but a more comprehensive collection of outputs for each experiment is included in the Appendix.

Image Synthesis.

We apply the proposed compression method to DCGAN (Radford et al., 2016) 3 , a network that learns to synthesize novel images belonging to a given distribution.

We task DCGAN with generating images that could belong to the MNIST data set, with results shown in Figure 3 .

Domain Translation.

We apply the proposed compression method to pix2pix (Isola et al., 2017) 4 , an approach to learn the mapping between paired training examples by applying conditional adversarial networks.

In our experiment, the task is synthesizing fake satellite images from label maps and vice-versa.

Representative results of this bidirectional task are shown in Figure 4 .

Style Transfer.

We apply the proposed compression method to CycleGAN (Zhu et al., 2017a) , used to exchange the style of images from a source domain to a target domain in the absence of paired training examples.

In our experiment, the task is to transfer the style of real photos with that Image-to-image Translation.

In addition to the StarGAN results above (Section 3, Figure 1 ), we apply the proposed compression method to CycleGAN (Zhu et al., 2017a) performing bidirectional translation between zebra and horse images.

Results are shown in Figure 6 .

Super Resolution.

We apply self-supervised compression to SRGAN (Ledig et al., 2017) 5 , which uses a discriminator network trained to differentiate between upscaled and the original highresolution images.

We trained SRGAN on the DIV2K data set Agustsson & Timofte (2017) , and use the DIV2K validation images, as well as Set5 Bevilacqua et al. (2012) and Set14 Zeyde et al. (2010) to report deployment quality.

In this task, quality is often evaluated by two metrics: Peak Signal-toNoise Ratio (PSNR) (Huynh-Thu & Ghanbari, 2008) and Structural Similarity (SSIM) (Wang et al., 2004) .

We also show FID scores (Heusel et al., 2017) for our results in the results summarized in Table 3 , and a representative output is shown in Figure 7 .

These results also include filter-pruned generators (see Section 6).

After showing that self-supervised compression applies to many tasks and networks with a moderate, fine-grained sparsity of 50%, we expand the scope of the investigation to include different pruning granularities and rates.

From coarse to fine, we can compress and remove the entire filters (3D-level), kernels (2D-level), vectors (1D-level) or individual elements (0D-level).

In general, finer-grained pruning results in higher accuracy for a given sparsity rate, but coarser granularities are easier to exploit for performance gains due to their regular structure.

Similarly, different sparsity rates, leaving many nonzero weights or few, can result in varying levels of quality in the final network.

We pruned all tasks by removing both single elements (0D) and entire filters (3D).

Further, for each granularity, we pruned to final sparsities of 25%, 50%, 75%, and 90%.

Representative results for CycleGAN (Monet → Photo) are shown in Figure 8 , but in general, 0D pruning is less invasive, even at higher sparsities.

Up to 90% fine-grained sparsity, some fine details faded away in pix2pix, but filter pruning results in drastic color shifts and loss of details at even 25% sparsity.

In this paper, we propose using a pre-trained discriminator to self-supervise the compression of a generative adversarial network.

We show that it is effective and applies to many tasks commonly solved with GANs, unlike traditional compression approaches.

Comparing the compressed generators with the baseline models on different tasks, we can conclude that the compression method performs well both in subjective and quantitative evaluations.

Advantages of the proposed method include:

• The results from the compressed generators are greatly improved over past work.

• The self-supervised compression is much shorter than the original GAN training process.

It only takes 1%-10% training effort to get an optimal compressed generative model.

• It is an end-to-end compression schedule that does not require objective evaluation metrics.

• We introduce a single optional hyperparameter (fixed to 0.5 for all our experiments).

We use self-supervised GAN compression to show that pruning whole filters, which can work well for image classification models (Li et al., 2017) , may perform poorly for GAN applications.

Even pruned at a moderate sparsity (e.g. 25% in Figure 8 ), the generated image has an obvious color shift and does not transfer the photorealistic style.

In contrast, the fine-grained compression stategy works well for all tasks we explored.

SRGAN seems to be an exception to filter-pruning's poor results; we have to look closely to see differences, and it's not clear which is subjectively better.

We have not tried to achieve extremely aggressive compression rates with complicated pruning strategies.

Different models may be able to tolerate different amounts of pruning when applied to a task, which we leave to future work.

Similarly, we have used network pruning to show the importance and utility of the proposed method, but self-supervised compression is general to other techniques, such as quantization, weight sharing, etc.

There are other tasks for which GANs can provide compelling results, and newer networks for tasks we have already explored; future work will extend our self-supervised compression method to these new areas.

Finally, self-supervised compression may apply to other network types and tasks if a discriminator is trained alongside the teacher and student networks.

The loss curves for the comparative experiment in Figure 1 are shown in Figure 9 .

Figures 10-12 show outputs of StarGAN compressed with various existing techniques (c-n), and the proposed self-supervised method (b).

The baseline output is at the top (a) of each figure for comparison.

Each row shows one input face translated to have black hair, blond hair, brown hair, the opposite gender, and a different age, and each row is a different method of compressing the network (the key is identical to that of Figure 9 ).

Figure 13: Image synthesis on MNIST dataset with DCGAN pruned to 50% with fine-grained sparsity.

Column 1: Handwritten numbers generated by the original generator, 2: Handwritten numbers generated by the generator pruned with our method, 3: Handwritten numbers generated by the pruned generator with traditional knowledge distillation adapted for GANs (Aguinaldo et al., 2019) .

FID: 37.1296 118.3160 172.9123 Figure 14 : Image synthesis on MNIST dataset with DCGAN of 75% fine-grained sparsity.

Column 1: Handwritten numbers generated by the original generator, 2: Handwritten numbers generated by the generator pruned with our method, Column 3: Handwritten numbers generated by the pruned generator with traditional knowledge distillation adapted for GANs.

CycleGAN: zebra to horse image-to-image translation.

Figure 34: Image-to-image translation: filter pruning to different sparsities.

Row 1: Baseline generator output.

Rows 2-5: Generated real photo style images by generators pruned to sparsities of 25%, 50%, 75%, 90%.

Figure 35 : Image-to-image translation: fine-grained pruning to different sparsities.

Row 1: Baseline generator output.

Rows 2-5: Generated real photo style images by generators pruned to sparsities of 25%, 50%, 75%, 90%.

CycleGAN: horse to zebra image-to-image translation.

The loss curves for the comparative experiment in Figure 39 and 41 are shown in Figure 44 .

<|TLDR|>

@highlight

Existing pruning methods fail when applied to GANs tackling complex tasks, so we present a simple and robust method to prune generators that works well for a wide variety of networks and tasks.

@highlight

The authors propose a modification to the classic distillation method for the task of compressing a network to address the failure of previous solutions when applied to generative adversarial networks.