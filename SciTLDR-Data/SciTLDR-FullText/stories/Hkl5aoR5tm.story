Training Generative Adversarial Networks (GANs) is notoriously challenging.

We propose and study an architectural modification, self-modulation, which improves GAN performance across different data sets, architectures, losses, regularizers, and hyperparameter settings.

Intuitively, self-modulation allows the intermediate feature maps of a generator to change as a function of the input noise vector.

While reminiscent of other conditioning techniques, it requires no labeled data.

In a large-scale empirical study we observe a relative decrease of 5%-35% in FID.

Furthermore, all else being equal, adding this modification to the generator leads to improved performance in 124/144 (86%) of the studied settings.

Self-modulation is a simple architectural change that requires no additional parameter tuning, which suggests that it can be applied readily to any GAN.

Generative Adversarial Networks (GANs) are a powerful class of generative models successfully applied to a variety of tasks such as image generation BID20 Miyato et al., 2018; Karras et al., 2017) , learned compression BID15 , super-resolution (Ledig et al., 2017) , inpainting (Pathak et al., 2016) , and domain transfer BID13 BID23 .Training GANs is a notoriously challenging task BID6 BID15 as one is searching in a high-dimensional parameter space for a Nash equilibrium of a non-convex game.

As a practical remedy one applies (usually a variant of) stochastic gradient descent, which can be unstable and lack guarantees Salimans et al. (2016) .

As a result, one of the main research challenges is to stabilize GAN training.

Several approaches have been proposed, including varying the underlying divergence between the model and data distributions Mao et al., 2016) , regularization and normalization schemes BID7 Miyato et al., 2018) , optimization schedules (Karras et al., 2017) , and specific neural architectures (Radford et al., 2016; BID21 .

A particularly successful approach is based on conditional generation; where the generator (and possibly discriminator) are given side information, for example class labels Mirza & Osindero (2014) ; Odena et al. (2017) ; Miyato & Koyama (2018) .

In fact, state-of-the-art conditional GANs inject side information via conditional batch normalization (CBN) layers BID3 Miyato & Koyama, 2018; BID21 .

While this approach does help, a major drawback is that it requires external information, such as labels or embeddings, which is not always available.

In this work we show that GANs benefit from self-modulation layers in the generator.

Our approach is motivated by Feature-wise Linear Modulation in supervised learning (Perez et al., 2018; BID3 , with one key difference: instead of conditioning on external information, we condition on the generator's own input.

As self-modulation requires a simple change which is easily applicable to all popular generator architectures, we believe that is a useful addition to the GAN toolbox.

We provide a simple yet effective technique that can added universally to yield better GANs.

We demonstrate empirically that for a wide variety of settings (loss functions, regularizers and normalizers, neural architectures, and optimization settings) that the proposed approach yields between a 5% and 35% improvement in sample quality.

When using fixed hyperparameters settings our approach outperforms the baseline in 86%(124/144) of cases.

Further, we show that self-modulation still helps even if label information is available.

Finally, we discuss the effects of this method in light of recently proposed diagnostic tools, generator conditioning (Odena et al., 2018) and precision/recall for generative models (Sajjadi et al., 2018) .

Several recent works observe that conditioning the generative process on side information (such as labels or class embeddings) leads to improved models (Mirza & Osindero, 2014; Odena et al., 2017; Miyato & Koyama, 2018) .

Two major approaches to conditioning on side information s have emerged: (1) Directly concatenate the side information s with the noise vector z (Mirza & Osindero, 2014), i.e. z = [s, z].

(2) Condition the hidden layers directly on s, which is usually instantiated via conditional batch normalization BID3 Miyato & Koyama, 2018)

.Despite the success of conditional approaches, two concerns arise.

The first is practical; side information is often unavailable.

The second is conceptual; unsupervised models, such as GANs, seek to model data without labels.

Including them side-steps the challenge and value of unsupervised learning.

We propose self-modulating layers for the generator network.

In these layers the hidden activations are modulated as a function of latent vector z. In particular, we apply modulation in a feature-wise fashion which allows the model to re-weight the feature maps as a function of the input.

This is also motivated by the FiLM layer for supervised models (Perez et al., 2018; BID3 in which a similar mechanism is used to condition a supervised network on side information.

Batch normalization BID12 can improve the training of deep neural nets, and it is widely used in both discriminative and generative modeling Radford et al., 2016; Miyato et al., 2018) .

It is thus present in most modern networks, and provides a convenient entry point for self-modulation.

Therefore, we present our method in the context of its application via batch normalization.

In batch normalization the activations of a layer, h, are transformed as DISPLAYFORM0 where µ and σ 2 are the estimated mean and variances of the features across the data, and γ and β are learnable scale and shift parameters.

Self-modulation for unconditional (without side information) generation.

In this case the proposed method replaces the non-adaptive parameters β and γ with input-dependent β(z) and γ(z), respectively.

These are parametrized by a neural network applied to the generator's input FIG0 ).

In particular, for layer , we compute In general, it suffices that γ (·) and β (·) are differentiable.

In this work, we use a small onehidden layer feed-forward network (MLP) with ReLU activation applied to the generator input z. Specifically, given parameter matrices U ( ) and V ( ) , and a bias vector b ( ) , we compute DISPLAYFORM1 DISPLAYFORM2 We do the same for β(z) with independent parameters.

Self-modulation for conditional (with side information) generation.

Having access to side information proved to be useful for conditional generation.

The use of labels in the generator (and possibly discriminator) was introduced by Mirza & Osindero (2014) and later adapted by Odena et al. (2017); Miyato & Koyama (2018) .

In case that side information is available (e.g. class labels y), it can be readily incorporated into the proposed method.

This can be achieved by simply composing the information y with the input z ∈ R d via some learnable function g, i.e. z = g(y, z).

In this work we opt for the simplest option and instantiate g as a bi-linear interaction between z and two trainable embedding functions E, E : Y → R d of the class label y, as DISPLAYFORM3 This conditionally composed z can be directly used in Equation 1.

Despite its simplicity, we demonstrate that it outperforms the standard conditional models.

Discussion.

TAB0 summarizes recent techniques for generator conditioning.

While we choose to implement this approach via batch normalization, it can also operate independently by removing the normalization part in the Equation 1.

We made this pragmatic choice due to the fact that such conditioning is common (Radford et al., 2016; Miyato et al., 2018; Miyato & Koyama, 2018) .The second question is whether one benefits from more complex modulation architectures, such as using an attention network BID18 whereby β and γ could be made dependent on all upstream activations, or constraining the elements in γ to (0, 1) which would yield a similar gating mechanism to an LSTM cell BID10 .

Based on initial experiments we concluded that this additional complexity does not yield a substantial increase in performance.

We perform a large-scale study of self-modulation to demonstrate that this method yields robust improvements in a variety of settings.

We consider loss functions, architectures, discriminator regularization/normalization strategies, and a variety of hyperparameter settings collected from recent studies (Radford et al., 2016; BID7 Miyato et al., 2018; BID15 Kurach et al., 2018) .

We study both unconditional (without labels) and conditional (with labels) generation.

Finally, we analyze the results through the lens of the condition number of the generator's Jacobian as suggested by Odena et al. (2018) , and precision and recall as defined in Sajjadi et al. (2018) .

Loss functions.

We consider two loss functions.

The first one is the non-saturating loss proposed in BID6 : DISPLAYFORM0 The second one is the hinge loss used in Miyato et al. (2018) : DISPLAYFORM1 Controlling the Lipschitz constant of the discriminator.

The discriminator's Lipschitz constant is a central quantity analyzed in the GAN literature (Miyato et al., 2018; BID22 .

We consider two state-of-the-art techniques: gradient penalty BID7 , and spectral normalization (Miyato et al., 2018) .

Without normalization and regularization the models can perform poorly on some datasets.

For the gradient penalty regularizer we consider regularization strength λ ∈ {1, 10}.Network architecture.

We use two popular architecture types: one based on DCGAN (Radford et al., 2016) , and another from Miyato et al. (2018) which incorporates residual connections BID8 .

The details can be found in the appendix.

Optimization hyper-parameters.

We train all models for 100k generator steps with the Adam optimizer (Kingma & Ba, 2014) (We also perform a subset of the studies with 500K steps and discuss it in.

We test two popular settings of the Adam hyperparameters (β 1 , β 2 ): (0.5, 0.999) and (0, 0.9).

Previous studies find that multiple discriminator steps per generator step can help the training BID6 Salimans et al., 2016 ), thus we also consider both 1 and 2 discriminator steps per generator step 2 .

In total, this amounts to three different sets of hyperparameters for (β 1 , β 2 , disc iter): (0, 0.9, 1), (0, 0.9, 2), (0.5, 0.999, 1).

We fix the learning rate to 0.0002 as in Miyato et al. (2018) .

All models are trained with batch size of 64 on a single nVidia P100 GPU.

We report the best performing model attained during the training period; although the results follow the same pattern if the final model is report.

Datasets.

We consider four datasets: CIFAR10, CELEBA-HQ, LSUN-BEDROOM, and IMAGENET.

The LSUN-BEDROOM dataset BID19 contains around 3M images.

We partition the images randomly into a test set containing 30588 images and a train set containing the rest.

CELEBA-HQ contains 30k images (Karras et al., 2017) .

We use the 128 × 128 × 3 version obtained by running the code provided by the authors 3 .

We use 3000 examples as the test set and the remaining examples as the training set.

CIFAR10 contains 70K images (32 × 32 × 3), partitioned into 60000 training instances and 10000 testing instances.

Finally, we evaluate our method on IMAGENET, which contains 1.3M training images and 50K test images.

We re-size the images to 128 × 128 × 3 as done in Miyato & Koyama (2018) and BID21 .Metrics.

Quantitative evaluation of generative models remains one of the most challenging tasks.

This is particularly true in the context of implicit generative models where likelihood cannot be effectively evaluated.

Nevertheless, two quantitative measures have recently emerged: The Inception Score and the Frechet Inception Distance.

While both of these scores have some drawbacks, they correlate well with scores assigned by human annotators and are somewhat robust.

Inception Score (IS) (Salimans et al., 2016) posits that that the conditional label distribution p(y|x) of samples containing meaningful objects should have low entropy, while the marginal label distribution p(y) should have high entropy.

Formally, DISPLAYFORM2 The score is computed using an Inception classifier .

Drawbacks of applying IS to model comparison are discussed in BID2 .An alternative score, the Frechet Inception Distance (FID), requires no labeled data BID9 .

The real and generated samples are first embedded into a feature space (using a specific layer of InceptionNet).

Then, a multivariate Gaussian is fit each dataset and the distance is computed as DISPLAYFORM3 ), where µ and Σ denote the empirical mean and covariance and subscripts x and g denote the true and generated data, respectively.

FID was shown to be robust to various manipulations and sensitive to mode dropping BID9 .

2 We also experimented with 5 steps which didn't outperform the 2 step setting.

3 Available at https://github.com/tkarras/progressive_growing_of_gans.

In the unpaired setting (as defined in Section 3.2), we compute the median score (across random seeds) and report the best attainable score across considered optimization hyperparameters.

SELF-MOD is the method introduced in Section 2 and BASELINE refers to batch normalization.

We observe that the proposed approach outperforms the baseline in 30 out of 32 settings.

The relative improvement is detailed in TAB3 .

The standard error of the median is within 3% in the majority of the settings and is presented in TAB7

To test robustness, we run a Cartesian product of the parameters in Section 3.1 which results in 36 settings for each dataset (2 losses, 2 architectures, 3 hyperparameter settings for spectral normalization, and 6 for gradient penalty).

For each setting we run five random seeds for self-modulation and the baseline (no self-modulation, just batch normalization).

We compute the median score across random seeds which results in 1440 trained models.

We distinguish between two sets of experiments.

In the unpaired setting we define the model as the tuple of loss, regularizer/normalization, neural architecture, and conditioning (self-modulated or classic batch normalization).

For each model compute the minimum FID across optimization hyperparameters (β 1 , β 2 , disc iters).

We therefore compare the performance of self-modulation and baseline for each model after hyperparameter optimization.

The results of this study are reported in TAB1 , and the relative improvements are in TAB3 and FIG1 .We observe the following: (1) When using the RESNET style architecture, the proposed method outperforms the baseline in all considered settings.

(2) When using the SNDCGAN architecture, it outperforms the baseline in 87.5% of the cases.

The breakdown by datasets is shown in FIG1 .

FORMULA3 The improvement can be as high as a 33% reduction in FID.

(4) We observe similar improvement to the inception score, reported in the appendix.

In the second setting, the paired setting, we assess how effective is the technique when simply added to an existing model with the same set of hyperparameters.

In particular, we fix everything except the type of conditioning -the model tuple now includes the optimization hyperparameters.

This results in 36 settings for each data set for a total of 144 comparisons.

We observe that selfmodulation outperforms the baseline in 124/144 settings.

These results suggest that self-modulation can be applied to most GANs even without additional hyperparameter tuning.

Conditional Generation.

We demonstrate that self-modulation also works for label-conditional generation.

Here, one is given access the class label which may be used by the generator and the .

We observe that the majority "good" models utilize self-modulation.

Figure (c) shows that applying self-conditioning is more beneficial on the later layers, but should be applied to each layer for optimal performance.

This effect persists across all considered datasets, see the appendix.discriminator.

We compare two settings: (1) Generator conditioning is applied via label-conditional Batch Norm BID3 Miyato & Koyama, 2018) with no use of labels in the discriminator (G-COND).

(2) Generator conditioning applied as above, but with projection based conditioning in the discriminator (intuitively it encourages the discriminator to use label discriminative features to distinguish true/fake samples), as in Miyato & Koyama (2018) (P-CGAN).

The former can be considered as a special case of the latter where discriminator conditioning is disabled.

For P-CGAN, we use the architectures and hyper-parameter settings of Miyato & Koyama (2018) .

See the appendix, Section B.3 for details.

In both cases, we compare standard label-conditional batch normalization to self-modulation with additional labels, as discussed in Section 2, Equation 3.The results are shown in TAB4 .

Again, we observe that the simple incorporation of self-modulation leads to a significant improvement in performance in the considered settings.

Training for longer on IMAGENET.

To demonstrate that self-modulation continues to yield improvement after training for longer, we train IMAGENET for 500k generator steps.

Due to the increased computational demand we use a single setting for the unconditional and conditional settings models following Miyato et al. (2018) and Miyato & Koyama (2018) , but using only two discriminator steps per generator.

We expect that the results would continue to improve if training longer.

However, currently results from 500k steps require training for ∼10 days on a P100 GPU.We compute the median FID across 3 random seeds.

After 500k steps the baseline unconditional model attains FID 60.4, self-modulation attains 53.7 (11% improvement).

In the conditional setting Where to apply self-modulation?

Given the robust improvements of the proposed method, an immediate question is where to apply the modulation.

We tested two settings: (1) applying modulation to every batch normalization layer, and (2) applying it to a single layer.

The results of this ablation are in FIG1 .

These results suggest that the benefit of self-modulation is greatest in the last layer, as may be intuitive, but applying it to each layer is most effective.

Self-modulation is a simple yet effective complementary addition to this line of work which makes a significant difference when no side information is available.

In addition, when side information is available it can be readily applied as discussed in Section 2 and leads to further improvements.

Conditional Modulation.

Conditional modulation, using side information to modulate the computation flow in neural networks, is a rich idea which has been applied in various contexts (beyond GANs).

Multiplicative and Additive Modulation.

Existing conditional modulations mentioned above are usually instantiated via Batch Normalization, which include both multiplicative and additive modulation.

These two types of modulation also link to other techniques widely used in neural network literature.

The multiplicative modulation is closely related to Gating, which is adopted in LSTM BID10 , gated PixelCNN (van den Oord et al., 2016), Convolutional Sequence-to-sequence networks BID5 and Squeeze-and-excitation Networks BID11 .

The additive modulation is closely related to Residual Networks BID8 .

The proposed method adopts both types of modulation.

We present a generator modification that improves the performance of most GANs.

This technique is simple to implement and can be applied to all popular GANs, therefore we believe that selfmodulation is a useful addition to the GAN toolbox.

Our results suggest that self-modulation clearly yields performance gains, however, they do not say how this technique results in better models.

Interpretation of deep networks is a complex topic, especially for GANs, where the training process is less well understood.

Rather than purely speculate, we compute two diagnostic statistics that were proposed recently ignite the discussion of the method's effects.

First, we compute the condition number of the generators Jacobian.

Odena et al. (2018) provide evidence that better generators have a Jacobian with lower condition number and hence regularize using this quantity.

We estimate the generator condition number in the same was as Odena et al. (2018) .

We compute the Jacobian (J z ) i,j = δG(z)i δzj at each z in a minibatch, then average the logarithm of the condition numbers computed from each Jacobian.

Second, we compute a notion of precision and recall for generative models.

Sajjadi et al. (2018) define the quantities, F 8 and F 1/8 , for generators.

These quantities relate intuitively to the traditional precision and recall metrics for classification.

Generating points which have low probability under the true data distribution is interpreted as a loss in precision, and is penalized by the F 8 score.

Failing to generate points that have high probability under the true data distributions is interpreted as a loss in recall, and is penalized by the F 1/8 score.

FIG4 shows both statistics.

The left hand plot shows the condition number plotted against FID score for each model.

We observe that poor models tend to have large condition numbers; the correlation, although noisy, is always positive.

This result corroborates the observations in (Odena et al., 2018) .

However, we notice an inverse trend in the vicinity of the best models.

The cluster of the best models with self-modulation has lower FID, but higher condition number, than the best models without self-modulation.

Overall the correlation between FID and condition number is smaller for self-modulated models.

This is surprising, it appears that rather than unilaterally reducing the condition number, self-modulation provides some training stability, yielding models with a small range of generator condition numbers.

The right-hand plot in FIG4 shows the F 8 and F 1/8 scores.

Models in the upper-left quadrant cover true data modes better (higher precision), and models in the lower-right quadrant produce more modes (higher recall).

Self-modulated models tend to favor higher recall.

This effect is most pronounced on IMAGENET.

Overall these diagnostics indicate that self-modulation stabilizes the generator towards favorable conditioning values.

It also appears to improve mode coverage.

However, these metrics are very new; further development of analysis tools and theoretical study is needed to better disentangle the symptoms and causes of the self-modulation technique, and indeed of others.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen.

Progressive growing of gans for improved quality, stability, and variation.

A ADDITIONAL RESULTS

We describe the model structures that are used in our experiments in this section.

The SNDCGAN architecture we follows the ones used in Miyato et al. (2018) .

Since the resolution of images in CIFAR10is 32 × 32 × 3, while resolutions of images in other datasets are 128 × 128 × 3.There are slightly differences in terms of spatial dimensions for both architectures.

The proposed self-modulation is applied to replace existing BN layer, we term it sBN (self-modulated BN) for short in TAB8 , 8, 9, 10.

The ResNet architecture we also follows the ones used in Miyato et al. (2018) .

Again, due to the resolution differences, two ResNet architectures are used in this work.

The proposed self-modulation is applied to replace existing BN layer, we term it sBN (self-modulated BN) for short in TAB0 , 12, 13, 14.

For the conditional setting with label information available, we adopt the Projection Based Conditional GAN (P-cGAN) (Miyato & Koyama, 2018) .

There are both conditioning in generators as well ad discriminators.

For generator, conditional batch norm is applied via conditioning on label information, more specifically, this can be expressed as follows, DISPLAYFORM0 Where each label y is associated with a scaling and shifting parameters independently.

For discriminator label conditioning, the dot product between final layer feature φ(x) and label embedding E(y) is added back to the discriminator output logits, i.e. D(x, y) = ψ(φ(x)) + φ(x) T E(y) where φ(x) represents the final feature representation layer of input x, and ψ(·) is the linear transformation maps the feature vector into a real number.

Intuitively, this type of conditional discriminator encourages discriminator to use label discriminative features to distinguish true/fake samples.

Both the above conditioning strategies do not dependent on the specific architectures, and can be applied to above architectures with small modifications.

We use the same architectures and hyper-parameter settings 4 as in Miyato & Koyama (2018) .

More specifically, the architecture is the same as ResNet above, and we compare in two settings: (1) only generator label conditioning is applied, and there is no projection based conditioning in the discriminator, and (2) both generator and discriminator conditioning are applied, which is the standard full P-cGAN.

@highlight

A simple GAN modification that improves performance across many losses, architectures, regularization schemes, and datasets. 