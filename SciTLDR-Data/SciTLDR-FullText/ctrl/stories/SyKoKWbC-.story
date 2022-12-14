In most current formulations of adversarial training, the discriminators can be expressed as single-input operators, that is, the mapping they define is separable over observations.

In this work, we argue that this property might help explain the infamous mode collapse phenomenon in adversarially-trained generative models.

Inspired by discrepancy measures and two-sample tests between probability distributions, we propose distributional adversaries that operate on samples, i.e., on sets of multiple points drawn from a distribution, rather than on single observations.

We show how they can be easily implemented on top of existing models.

Various experimental results show that generators trained in combination with our distributional adversaries are much more stable and are remarkably less prone to mode collapse than traditional models trained with observation-wise prediction discriminators.

In addition, the application of our framework to domain adaptation results in strong improvement over recent state-of-the-art.

Adversarial training of neural networks, especially Generative Adversarial Networks (GANs) BID10 , has proven to be a powerful tool for learning rich models, leading to outstanding results in various tasks such as realistic image generation, text to image synthesis, 3D object generation, and video prediction BID25 BID33 BID32 .

Despite their success, GANs are known to be difficult to train.

The generator and discriminator can oscillate significantly from iteration to iteration, and slight imbalances in their capacities frequently cause the training objective to diverge.

Another common problem suffered by GANs is mode collapse, where the distribution learned by the generator concentrates on a few modes of the true data distribution, ignoring the rest of the space.

In the case of images, this failure results in generated images that albeit realistic, lack diversity and reduce to a handful of prototypes.

A flurry of recent research seeks to understand and address the causes of instability and mode collapse in adversarially-trained models.

The first insights come from BID10 , who note that one of the main causes of training instability is saturation of the discriminator.

BID1 formalize this idea by showing that if the two distributions have supports that are disjoint or concentrated on low-dimensional manifolds that do not perfectly align, then there exists an optimal discriminator with perfect classification accuracy almost everywhere and the usual divergences (Kullback-Leibler, Jensen-Shannon) max-out for this discriminator.

In follow-up work, propose an alternative training scheme (WGAN) based on estimating the Wasserstein distance instead of the Jensen-Shannon divergence between real and generated distributions.

In this work, we highlight a further view on mode collapse.

The discriminator part of GANs and of variations like WGANs is separable over observations, which, as we will illustrate, can result in serious problems, even when minibatches are used.

The underlying issue is that the stochastic gradients are essentially (sums of) functions of single observations (training points).

Despite connections to two-sample tests based on Jensen-Shannon divergence, ultimately the updates based on gradients from different single observations are completely independent of each other.

We show how this lack of sharing information between observations may explain mode collapses in GANs.

Motivated by these insights, we take a different perspective on adversarial training and propose a framework that brings the discriminator closer to a truly distributional adversary, i.e., one that 1 (a set of observations) in its entirety, retaining and sharing global information between gradients.

The key insight is that a carefully placed nonlinearity in the form of specific population comparisons can enable information-sharing and thereby stabilize training.

We develop and test two such models, and also connect them to other popular ideas in deep learning and statistics.

Contributions.

The main contributions of this work are as follows:??? We introduce a new distributional framework for adversarial training of neural networks that operates on a genuine sample, i.e., a collection of points, rather than an observation.

This choice is orthogonal to modifications of the types of loss (e.g. logistic vs. Wasserstein) in the literature.??? We show how off-the-shelf discriminator networks can be made distribution-aware via simple modifications to their architecture and how existing models can seamlessly fit into this framework.??? Empirically, our distributional adversarial framework leads to more stable training and significantly better mode coverage than common single-observation methods.

A direct application of our framework to domain adaptation results in strong improvements over state-of-the-art.

To motivate our distributional approaches, we illustrate with the example of the original GAN how training with single-observation-based adversaries might lead to an unrecoverable mode collapse in the generator.

The objective function for a GAN with generator G and discriminator D is DISPLAYFORM0 where DISPLAYFORM1 maps an observation to the probability that it comes from data distribution P x , and G : DISPLAYFORM2 is a network parametrized by ?? G that maps a noise vector z ??? R l , drawn from a simple distribution P z , to the original data space.

The aim of G is to make the distribution P G of its generated outputs indistinguishable from the training distribution.

In practice, G is trained to instead maximize log D(G(z)) to prevent loss saturation early in the training.

BID10 showed that the discriminator converges in the limit to D * (x) = P x (x)/(P x (x) + P G (x)) for a fixed G. In this limit, given a sample Z = {z(1) , . . .

, z (B) } from the noise distribution P z , the gradient of G's loss with respect to its parameters ?? G is DISPLAYFORM3 where we slightly abuse the notation DISPLAYFORM4 G to denote G's Jacobian matrix.

Thus, the gradient with respect to each observation is weighted by terms of the form ???D(x DISPLAYFORM5 .

These terms can be interpreted as relative slopes of the discriminator's confidence function.

Their magnitude and sign depend on dD dx G (the slope of D around x G ) and on D(x G ), the confidence that x G is drawn from P x .

In at least one notable case this ratio is unambiguously low: values of x where the discriminator has high confidence of real samples and flat slope.

This implies that given the definition of D * , the weighting term will vanish in regions of the space where the generator's distribution has constant, low probability compared to the real distribution, such as neighborhoods of the support of P x where P G is missing a mode.

FIG0 exemplifies this situation for a simple case in 1D where the real distribution is bimodal and the generator's distribution is currently concentrated around one of the modes.

The cyan dashed line is the weighting term ???D(x G )/D(x G ), confirming our analysis above that gradients for points around the second mode will vanish.

The effect of this vanishing of mode-seeking gradients during training is that it prevents G from spreading mass to other modes, particularly to distant ones.

Whenever G does generate a point in a region far away from where most of its mass is concentrated (an event that by definition already occurs with low probability), this example's gradient (which would update G's parameters to move mass to this region) will be heavily down-weighted and therefore dominated by high-weighted gradients of other examples in the batch, such as those in spiked high-density regions.

The result is a catastrophic averaging-out.

A typical run of a GAN suffering from this phenomenon on a simple distribution is shown in Figure 2 (top row), where the generator keeps generating points from the same mode across the training procedure and is unable to recover from mode collapse.

Intuition behind modecollapsing behavior in singleobservation discriminators with logistic loss.

The current generated distribution (solid green line) covers only one of the true distribution's modes.

Gradients with respect to generated points x are weighted by the term ??? 1 D dD dx (cyan dashed line), so gradients corresponding to points close to the second mode will be dominated by those coming from the first mode.

At a high level, this mode-collapsing phenomenon is a consequence of a myopic discriminator that bases its predictions on a single observation, leading to gradients that are not harmonized by global information.

Discriminators that instead predict based on an entire sample are a natural way to address this issue (Figure 2 , bottom two rows), as we will show in the remainder of this work.

To mitigate the above noted failure, we seek a discriminator that considers a sample (multiple observations) instead of a single observation when predicting whether the input is drawn from P x .

More concretely, our discriminator is a set function M : 2 DISPLAYFORM0 } of potentially varying size.

Next, we construct this discriminator step by step.

Neural Mean Embedding.

Despite its simplicity, the mean is a surprisingly useful statistic for discerning between distributions, and is central to the Maximum Mean Discrepancy (MMD) BID11 BID8 BID27 , a powerful discrepancy measure between distributions that enjoys strong theoretical properties.

Instead of designing a kernel explicitly as in MMD, here we learn it in a fully data-driven fashion.

Specifically, we define a neural mean embedding (NME) ??, where ?? is learned as a neural network: DISPLAYFORM0 In practice, ?? only has access to P through samples of finite size {x DISPLAYFORM1 ??? P, thus one effectively uses the empirical estimate DISPLAYFORM2 This distributional encoder forms one pillar of our adversarial learning framework.

We propose two alternative adversary models that build on this NME to discriminate between samples and produce a rich training signal for the generator.

Sample Classifier.

First, we combine the NME with a classifier to build a discriminator for adversarial training.

That is, given an empirical estimate ??(X) from a sample X (drawn from the real data P x or the generated distribution P G ), the classifier ?? S outputs 1 to indicate the sample was drawn from P x and 0 otherwise.

Scoring the discriminator's predictions via the logistic loss leads to the following value function for the adversarial game: DISPLAYFORM3 where X and Z are samples from P x and P z respectively, and D S (??) = ?? S ??? ??(??) is the full discriminator (NME and classifier), which we refer to as the sample classifier.

Eq. (3.3) is similar to the original GAN objective (2.1), but differs in a crucial aspect: here, the expectation is inside ?? S , and the classifier is only predicting one label for the entire sample instead of one label for each observation.

In other words, while in a GAN D operates on single observations and then aggregates its predictions, here D S first aggregates the sample and then operates on this aggregated representation.

Two-sample Discriminator.

Inspired by two-sample tests, we alternatively propose to shift from a classification to a discrepancy objective, that is, given two samples X, Z drawn independently, the discriminator predicts whether they are drawn from the same or different distributions.

Concretely, given two NMEs ??(P 1 ) and ??(P 2 ), the two-sample discriminator ?? 2S uses their absolute difference to output ?? 2S (|??(P 1 ) ??? ??(P 2 )|) ??? [0, 1], interpreted as the confidence that the two samples were indeed drawn from different distributions.

Again with the logistic loss, we arrive at the objective function DISPLAYFORM4 where we split each of the two samples X and Z into halves (X 1 , X 2 from X and Z 1 , Z 2 from Z), and the second line evaluates the discrepancy between the two parts of the same sample, respectively.

As before, we use DISPLAYFORM5 to denote the full model and refer to it as a two-sample discriminator.

Choice of Objective Function.

Despite many existing adversarial objective functions in the literature BID19 BID3 , here we use the logistic loss to keep the presentation simple, to control for varying factors and to extract the effect of just our main contribution: namely, that the adversary is distribution-based and returns only one label per sample, a crucial departure from existing, single-observation adversary models such as the well-understood vanilla GAN.

This choice is orthogonal to modifications of the types of loss (e.g., logistic vs. Wasserstein).

It can, indeed, be seamlessly combined with most existing models to enhance their ability to match distributions.

We use the novel distributional adversaries proposed above in a new training framework that we name Distributional Adversarial Network (DAN).

This framework can easily be combined with existing adversarial training algorithms by a simple modification of their adversaries.

In this section, we examine in detail an example application of DAN to the generative adversarial setting.

In the experiments section we provide an additional application to adversarial domain adaptation.

Using distributional adversaries within the context of GANs yields a saddle-point problem analogous to the original GAN (2.1): DISPLAYFORM0 where ?? ??? {S, 2S} and the objective function V ?? is either (3.3) or (3.4); we refer to these as DAN-??.

As with GAN, we can optimize (3.5) via alternating updates on G and D ?? .Although optimizing (3.5) directly yields generators with remarkable performance in simple settings (see empirical results in Section 5.1), when the data distribution is complex, the distributional adversaries proposed here can easily overpower G, particularly early in the training when P G and P x differ substantially.

Thus, we propose to optimize instead a regularized version of (3.5): DISPLAYFORM1 where D is a weaker single-observation discriminator (such as the one in the original GAN), and ?? is a parameter that trades off the strength of this regularization.

Note that for ?? ??? ??? we recover the original GAN objective, while ?? = 0 yields the purely distributional loss (3.5).

In between these two extremes, the generator receives both local and global training signals for every generated sample.

We show empirically that DAN training is stable across a reasonable range of ?? (Appendix B).During training, all expectations with respect to data and noise distributions are approximated via finite sample averages.

In each training iteration, we draw samples from the data and noise distributions.

While for DAN-S the training procedure is similar to that of GAN, DAN-2S requires a modified training scheme.

Due to the form of the two-sample discriminator, we want a balanced exposure to pairs of samples drawn from the same and different distributions.

Thus, every time we update D 2S , we draw samples X = {x (1) , . . .

, x (B) } ??? P x and Z = {z (1) , . . . , z (B) } ??? P z from data and noise distributions.

We then split each sample into two parts, DISPLAYFORM2 and use the discriminator D 2S to predict on each pair of (X 1 , G(Z 2 )), (G(Z 1 ), X 2 ), (X 1 , X 2 ) and (G(Z 1 ), G(Z 2 )) with target outputs 1, 1, 0 and 0, respectively.

A detailed training procedure and its visualization are shown in Appendix A.

We close this section discussing why the distributional adversaries proposed above overcome the vanishing gradients phenomenon described in Section 2.

In the case of the sample classifier (?? = S), a derivation as in Section 2 shows that for a sample Z of B points from the noise distribution, the gradient of the loss with respect to G's parameters is DISPLAYFORM0 where we use ?? B := ??({G(z (1) ), . . .

, G(x (B) )}) for ease of notation.

Note that, as opposed to (2.2), the gradient for each variable z DISPLAYFORM1 is weighted by the same left-most discriminator confidence term.

This has the effect of sharing information across observations when computing gradients: whether a sample (encoded as a vector ?? B ) can fool the discriminator or not will have an effect on every observation's gradient.

The benefit is clearly revealed in Figure 2 (bottom two rows), where, in contrast to the vanilla GAN which remains stuck in mode collapse, DAN is able to recover all modes.

The gradient (3.7) suggests that the true power of this sample-based setting lies in choosing a discriminator ?? S that, through non-linearities, enforces interaction between the points in the sample.

The notion of sharing information across examples occurs also in batch normalization (BN) BID14 , although the mechanism to achieve this interaction and the underlying motivation for doing it are very different.

While the analysis here is not rigorous, the intuitive justification for sample-based aggregation is clear, and is confirmed by our experimental results.

Distributional Adversaries as Discrepancy Measures.

Our sample classifier implicitly defines a general notion of discrepancy between distributions: DISPLAYFORM0 for some function classes ?? and ??, a monotone function ?? : R ??? R, and constant c. This generalized discrepancy includes many existing adversarial objectives as special cases.

For example, if ?? and ?? S are identity functions, ?? the space of 1-Lipschitz functions and ?? an embedding into R, we obtain the 1-Wasserstein distance used by WGAN BID12 .Similarly, our two-sample discriminator implicitly defines the following discrepancy: DISPLAYFORM1 where ??, ?? are as before.

This form can be thought of as generalizing other discrepancy measures like MMD BID11 BID8 BID27 , defined as: DISPLAYFORM2 where ??(??) is some feature mapping, and k(u, v) = ??(u) ??(v) is the corresponding kernel function.

Letting ?? be the identity function corresponds to computing the distance between the sample means.

More complex kernels result in distances that use higher-order moments of the two samples.

In adversarial models, MMD and its variants (e.g., central moment discrepancy) have been used either as a direct objective for the target model or as a form of adversary BID36 BID28 BID17 BID7 .

However, most of them require hand-picking the kernel.

An adaptive feature function, in contrast, may be able to adapt to the given distributions and thereby discriminate better.

Our distributional adversary addresses this drawback and, at the same time, generalizes MMD, owing to the fact that neural networks are universal approximators.

Since it is trainable, the underlying witness function evolves as training proceeds, taking a simpler form when the two distributions (generated and true data) are easily distinguishable, and becoming more complex as they start to resemble.

In this sense our distributional adversary bears similarity to the recent work by .

However, their learned sample embedding is used as input to a Gaussian kernel to compute the MMD, inheriting the quadratic time complexity (in sample size) of kernel MMD.

Our model uses a neural network on top of the mean embedding differences to compute the discrepancy, resulting in linear time complexity and additional flexibility.

A more general family of discrepancy measures of interest in the GAN literature are Integral Probability Metrics (IPM), defined as DISPLAYFORM3 where F is a set of measurable and bounded real valued functions.

Comparing this to Eq. (4.1) shows that our two-sample discriminator discrepancy generalizes IPMs too.

In this sense, our model is also connected to GAN frameworks based on IPMs .Minibatch Discrimination.

Initiated by BID26 to stabilize training, a line of work in the GAN literature considers statistics of minibatches to train generative models, known as minibatch discrimination.

Batch normalization BID14 can be viewed as a form of minibatch discrimination, and is known to aid GAN training BID23 .

Zhao et al. (2017) proposed a repelling regularizer that operates on a minibatch and orthogonalizes the pairwise sample representation, keeping the model from concentrating on only a few modes.

We derive our adversaries as acting on full distributions instead of batches, but approximate population expectations with samples in practice.

In this sense, our implementation can be viewed as generalizing minibatch discrimination: we leverage neural networks for the design of different forms of minibatch discrimination.

Such discrimination does not need hand-crafted objectives as in existing work, and will be able to adapt to the data distribution and target models.

Permutation Invariant Networks.

The NME (3.2) is a permutation invariant operator on (unordered) samples.

Recent work has explored neural architectures that operate on sets and are likewise invariant to permutations on their inputs.

BID31 propose a content attention mechanism for unordered inputs of variable length.

Later work BID35 embeds all samples into a fixed-dimensional latent space, and then sums them to obtain a fixeddimensional vector representation, used as input to another network.

BID18 use a similar network for embedding a set of images into a latent space, but aggregate using a (learned) weighted sum.

Although the structure of our NEM resembles these networks in terms of permutation invariance, it differs in its motivation-discrepancy measures-as well as its usage within discriminators in adversarial training settings.

Other Related Work.

Various other approaches have been proposed to address training instability and mode collapse in GANs.

Many such methods resort to more complex network architectures or better-behaving objectives BID23 BID13 BID33 BID4 Zhao et al., 2017; , while others add more discriminators or generators BID6 BID30 in the hope that training signals from multiple sources lead to more stable training and better mode coverage.

We demonstrate the effectiveness of DAN training by applying it to generative models and domain adaptation.3In generative models, we observe remarkably better mode recovery than nondistributional models on both synthetic and real datasets, through both qualitative and quantitative evaluation of generated samples.

In domain adaptation, we leverage distributional adversaries to align latent spaces of source and target domains, and see strong improvements over state-of-the-art.

Figure 2 : Mixture of 4 Gaussians.

GAN training leads to unrecoverable mode collapse, with only one of the true distribution's modes being recovered.

The two distributional training approaches (bottom 2 rows, ?? = 0) capture all modes, and are able to recover from a missing mode (second column).

We first test DAN in a simple generative setting, where the true data distribution is a simple twodimensional mixture of Gaussians.

A similar task was used as a proof of concept for mode recovery by BID20 .

We generate a mixture of four Gaussians with means equally spaced on a circle of radius 6, and variances of 0.02.

We compare our distributional adversaries against various discriminator frameworks, including GAN, WGAN and WGAN-GP BID12 .

We use equivalent 4 simple feed-forward networks with ReLU activations for all discriminators.

For these synthetic distributions, we use the pure distributional objective for DAN (i.e., setting ?? = 0 in (3.5)).

Figure 2 displays the results for GAN and our methods; in the appendix, we show results for other models, different ?? settings and other synthetic distributions.

Overall, while other methods suffer from mode collapse throughout (GAN) or are sensitive to network architectures and hyperparameters (WGAN, WGAN-GP), our DANs consistently recover all modes of the true distribution and are stable across a reasonable range of ??'s.

Mode recovery entails not only capturing a mode, i.e., generating samples that lie in the corresponding mode, but also recovering the true probability mass of the mode.

Next, we evaluate our model on this criterion: we train DAN on MNIST and Fashion-MNIST BID34 , both of which have 10-class balanced distributions.

Since the generated samples are unlabeled, we train an external classifier on (Fashion-)MNIST to label the generated data.

Besides the original GAN, we compare to recently proposed generative models: RegGAN (Che et al., 2017), EBGAN (Zhao et al., 2017) , WGAN and its variant WGAN-GP BID12 , and GMMN BID17 .

To keep the approaches comparable, we use a similar neural network architecture in all cases, and did not tailor it to any particular model.

We trained models without Batch Normalization (BN), except for RegGAN and EBGAN, which we observed to consistently benefit from BN.

For DAN, we use the the regularized objective (3.5) with ?? > 0.

Both adversaries in this formulation use the same architecture except for the averaging layer of the distributional adversary, and share weights of the pre-averaging layers.

FIG1 shows the results.

Training with the vanilla GAN, RegGAN or GMMN leads to generators that place too much mass on some modes and ignore others, leading to a large TV distance between the generated label distribution and the correct one.

EBGAN and WGAN perform slightly worse than WGAN-GP and DAN on MNIST, and significantly worse on the (harder) Fashion-MNIST dataset.

WGAN-GP performs on par with DAN on MNIST, and slightly worse than DAN on Fashion-MNIST.

Moreover, WGAN-GP is in general more sensitive to hyperparameter selection (Appendix B).

Total variation distances (in log-scale) between generated and true (uniform) label distributions over 5 repetitions.

DAN achieves the best and most stable mode frequency recovery.

We also test DAN on a harder task, generating faces, and compare against the samples generated by DCGAN.

The generated examples in FIG2 show that DCGAN+BN exhibits an obvious mode-collapse, with most faces falling into a few clusters.

DCGAN without BN generates better images, but they still lack feature diversity (e.g., backgrounds, hairstyle).

The images generated by DAN, in contrast, exhibit a much richer variety of features, indicating a better coverage of the data distribution's modes.

In the last set of experiments, we test DAN in the context of domain adaptation.

We compare against DANN BID9 , an adversarial training algorithm for domain adaptation that uses a domain-classifier adversary to enforce similar source and target representations, thus allowing for the source classifier to be used in combination with the target encoder.

We use the DAN framework to further encourage distributional similarities between representations from different domains.

We first compare the algorithms on the Amazon reviews dataset preprocessed by BID5 .

It consists of four domains: books, dvd, electronics and kitchen appliances, each of which contains Table 1 .

Our models outperform the GAN-based DANN on most source-target pairs, with an average improvement in accuracy of 1.41% for DAN-S and 0.92% for DAN-2S.

Note that we directly integrated DAN without any network structure tuning -we simply set the network structure of the distributional adversary to be the same as that of original discriminator in DANN (except that it takes an average of latent representations in the middle).Lastly, we test the effectiveness of DAN on a domain adaptation task for image label prediction.

The task is to adapt from MNIST to MNIST-M BID9 , which is obtained by blending digits over patches randomly extracted from color photos from BSDS500 BID0 .

The results are shown in TAB1 .

DAN-S improves over DANN by ??? 5%.

Again, these results were obtained by simply plugging in the DAN objective, demonstrating the ease of using DAN.

In this work, we propose a distributional adversarial framework that, as opposed to common approaches, does not rely on a sum of observation-wise functions, but considers a sample (collection of observations) as a whole.

We show that when used within generative adversarial networks, this different approach to distribution discrimination has a stabilizing effect and remedies the well-known problem of mode collapse.

One likely reason for this is its ability to share information across observations when computing gradients.

The experimental results obtained with this new approach offer a promising glimpse of the advantages of genuine sample-based discriminators over common alternatives that are separable over observations, while the simplicity and ease of implementation make this approach an appealing plug-in, easily compatible with most existing models.

The framework proposed here is fairly general and opens the door for various possible extensions.

The two types of discriminators proposed here are by no means the only options.

There are many other approaches in the distributional discrepancy literature to draw inspiration from.

One aspect that warrants additional investigation is the effect of sample size on training stability and mode coverage.

It is sensible to expect that in order to maintain global discrimination power in settings with highly multimodal distributions, the size of samples fed to the discriminators should grow, at least with the number of modes.

Formalizing this relationship is an interesting avenue for future work.

We show the complete training procedure for DAN in Algorithm 1 and a visual illustration in Figure 5 .

Note that we set a step number k such that every other k iterations we update the distributional adversary, otherwise we update the single-observation adversary.

Algorithm 1 Training Procedure for DAN-S/2S.

Input:total number of iterations T , size of minibatch B, step number k, model mode ?? ??? {S, DISPLAYFORM0 Update distributional adversary D S = {?? S , ??} with one gradient step on loss: DISPLAYFORM1 then Divide X and Z evenly into X 1 , X 2 and Z 1 , Z 2 respectively Update distributional adversary D 2S = {?? 2S , ??} with one gradient step on loss: DISPLAYFORM2 Update discriminator D with one gradient step on loss: DISPLAYFORM3 Update G with gradient step on loss: DISPLAYFORM4 then Divide X and Z into X 1 , X 2 and Z 1 , Z 2 Update G with gradient step on loss: DISPLAYFORM5 end if end for

We first show the full result for mode recovery when the true data distribution is a mixture of 4 Gaussians with variances of 0.02 on a circle of radius 6.

The generator consists of a fully connected network with 3 hidden layers of size 128 with ReLU activations, followed by a linear projection to 2 dimensions.

The discriminator consists of a fully connected network with 3 hidden layers of size 32 with ReLU activations, followed by a linear projection to 1 dimension.

Latent vectors are sampled Figure 5 : DAN-S and DAN-2S models and corresponding losses, where DISPLAYFORM0 DISPLAYFORM1 uniformly from [???1, 1]

.

For WGAN, the weight clipping parameter is set to 0.01.

For WGAN-GP, ?? the weight for the gradient penalty is set to 0.1.

For DAN, the distributional adversaries have two initial hidden layers of size 32, after which the latent representations are averaged across the batch.

The mean representation is then fed to a fully connected layer of size 32 and a linear projection to 1 dimension.

WGAN is optimized using RMSProp BID29 with learning rate of 5 ?? 10 ???5.

All other models are optimized using Adam BID15 with learning rate of 10 ???4 and ?? 1 = 0.5.

Minibatch size is fixed to 512.The intuitive argument presented in Sec. 2 suggests that GAN training would result in the same single mode being recovered throughout, which is also confirmed in the first row of Figure 6 : the generator's distribution is concentrated around a single mode, from which it cannot due to its inherited disadvantage of single observation-based discriminator and logistic loss.

WGAN does not stuck at a single mode but the mode collapse still happens.

On the other hand, WGAN-GP and DAN is able to constantly recover all 4 far-apart modes.

Note that the gradient scaling argument in Sec. 2 only says that points far away (i.e. with low probability under G) from G's modes will be down weighted.

But points close to the boundary of D's decision will have a large ???D(x (i) G ) term, and thus up-weighting gradients and can cause the G's whole mode to shift towards the boundary.

If after this shift, enough of G's mass falls closer to another mode, then this can become the attracting point and further concentrate G's mass.

This suggests that, for distributions with spiked, but close modes, G's mass should concentrate but possibly traverse from mode to mode.

This effect is indeed commonly observed for the popular 8-gaussians example shown in FIG5 .

All other models follow the same argument as in Figure 6 .

Though WGAN-GP is able to recover all modes as DAN, we show that the training for WGAN-GP is unstable -a slight change in hyperparameter may end up with totally different generator distribution.

We consider the same network architecture as in public available WGAN-GP code 5 : The generator consists of a fully connected network with 3 hidden layers of size 512 with ReLU activations, followed by a linear projection to 2 dimensions.

The discriminator consists of a fully connected For the first set of experiments we optimize using Adam BID15 with learning rate of 10 ???4, ?? 1 = 0.5 and ?? 2 = 0.9, and train the network for 30, 000 iterations, where each iteration consists of 5 updates for discriminator and 1 for generator for WGAN-GP.

The result is shown in Figure 8 .

All models successfully capture all modes, despite their separation.

However, a slight change in the optimizer's parameters to ?? 2 = 0.999 (the default value) causes WGAN-GP to fail sometime even if we run for 50, 000 iterations, while DAN still constantly recovers the true distribution (cf.

Figure 9 ).

We also experiment with WGAN optimized using RMSProp with learning rate of 5 ?? 10 DISPLAYFORM0 under this setting and in contrast to previous experiments where it fails to recover any mode, it is able to recover all modes using this network architecture.

This shows that both WGAN and WGAN-GP are sensitive to network structure or hyperparameter settings, while DAN is more stable.

Finally, we show that DAN is stable across a reasonable range of ??, the trade-off parameter for single observation and distributional adversaries.

We set ?? = {0, 0.2, 0.5, 1, 2, 5, 10} for DAN and the results are shown in FIG0 and 11.

While both DAN-S and DAN-2S are stable, DAN-2S is stable in larger range of ?? than DAN-S.To summarize, the trends we observe in these these synthetic experiments are: (i) GAN irremediably suffers from mode collapse for this simple dataset, (ii) WGAN(+GP) can fail in recovering modes or are not stable against hyperparameter choices and (iii) DAN constantly recovers all modes and is stable for all reasonable hyperparameter settings.

The right-most plot shows the true data distribution.

Since modes are closer, the generator in GAN may not get stuck at generating the same single mode, but oscillate between modes, as confirmed by the 1st row.

WGAN (rows 2) do not get stuck at the same single mode since they do not use the logistic loss, but the mode collapse still happens.

WGAN-GP and DAN is able to constantly recover all 8 modes.

, ?? 1 = 0.5 and ?? 2 = 0.999.

The first 3 rows show random runs for WGAN-GP -it does not constantly recover all modes, even if we run it for longer time.

We use the same architecture for the generator across all models: three fully-connected layers of sizes [256, 512, 1024] with ReLU activations, followed by a fully connected linear layer and a Sigmoid activation to map it to a 784-dimensional vector.

Latent vectors are uniform samples from [???1, 1]

.

For GAN, RegGAN and DAN the single observation discriminators consist of three fully-connected layers of sizes [1024, 512, 256] with ReLU activations, followed by a fully connected linear layer and Sigmoid activation to map it to a 1-dimensional value.

Both adversaries in DAN's use the same architecture except for the averaging layer of the distributional adversary, and share weights of the pre-averaging layers.

This is done to limit the number of additional parameters of the regularizer, so as to make training more efficient.

The architecture for the decoder in EBGAN is the same with the generator, while the ones for encoders in RegGAN and EBGAN are the same as the discriminator except for the last layer where it maps to 256-dimensional vectors.

For GMMN we use a mixture of Gaussian kernels with bandwidths in {0.1, 0.5, 1, 5, 10, 50}. Throughout the experiments, we set the hyperparameters as shown in TAB3 .

Hypaerparameters We use Adam with learning rate of 0.0005 and ?? 1 = 0.5, a fixed minibatch size of 256 and 100 epochs of training.

For models except WGAN and DAN we train by alternating between generator and adversary (and potentially encoder and decoder) updates.

Generated digits by each model are shown in FIG0 , where we observe a clear advantage of DAN-S and DAN-2S over all other baselines in terms of mode coverage and generation quality.

DISPLAYFORM0 Under review as a conference paper at ICLR 2018Figure 10: DAN-S with ?? set to {0, 0.2, 0.5, 1, 2, 5} (top-down order).

We use the same experimental settings as in Section C. Generated examples by each model are shown in FIG0 , where we observe a clear advantage of DAN-S and DAN-2S over all other baselines in terms of mode coverage and generation quality.

We vary the batch-size in DAN's to see how it affects model performance.

Results are shown in FIG0 .

We observe that both methods suffer from too small or too large batch-sizes.

However, there is a clear distinction between the two: while DAN-S outperforms DAN-2S with smaller batchsizes, this trend reverses for larger batch-sizes, where DAN-2S achieves better performance and is more stable across repetitions.

We present here additional experiments on the SVHN dataset.

We compare DAN's against GAN, WGAN, WGAN-GP and GMMN.

We use the same architecture for the generator across all models: a latent vector first goes through two fully connected layers with number of hidden units 1, 024 and 8,192 respectively.

Then it is reshaped to sizes [8 ?? 8 ?? 128] and goes through 2 transposed

.

For GAN, WGAN variants and DAN the single observation discriminators consist of 2 convolutional layers of kernel size 4 and stride 2, followed by 2 fully connected linear layer with number hidden unites 8,192 and 1,024 respectively, and a Sigmoid activation to map it to a 1-dimensional value.

Both adversaries in DAN use the same architecture except for the averaging layer of the distributional adversary, and share weights of the pre-averaging layers.

For both generators and discriminators we also add batch-normalization layers for all but DAN models.

Other hyperparameters and training dynamics are the same as in Section C.We again compute the generated label distribution and compare it against true (uniform) distribution.

TV distances are shown in FIG0 .

DAN achieves one of the best and most stable mode frequency recovery.

While GMMN performs slightly better than DAN, it suffers from poor generating quality.

Specifically, we show generated figures in FIG0 , where DAN's present one of the best mode coverage and generation quality among all.

We use the same experimental settings as in Section E and run all models on CIFAR10.

Generated samples from various models are shown in FIG0 .

Again DAN presents one of the best mode coverage and generation quality.

Figure 15: Total variation distances between generated and true (uniform) label distributions over 5 repetitions.

Performances are sorted in increasing order, i.e., better performing models stay on the right.

DAN achieves one of the best and most stable mode frequency recovery.

While GMMN performs slightly better than DAN, it suffers from poor generating quality as seen in FIG0 G EXPERIMENT DETAILS ON CELEBAWe use a publicly available implementation 6 of DCGAN.

The network architecture is kept as in the default configuration.

We preprocess the image data by first cropping each image to 160 ?? 160 (instead of 108 ?? 108 in default setting) and then resize them to 64 ?? 64.

The generator consists of a fully connected linear layer mapping from latent space of [???1, 1] 100 to dimension 8,192, followed by 4 deconvolution layers, three with ReLU activations and the last one followed by tanh.

The discriminator is the "reverse" of generator, except that the activation function is Leaky ReLU the last layer being a linear mapping and a Sigmoid activation to 1D value.

Both adversaries in DAN use the same architecture except for the averaging layer of the distributional adversary, and share weights of the pre-averaging layers.

<|TLDR|>

@highlight

We show that the mode collapse problem in GANs may be explained by a lack of information sharing between observations in a training batch, and propose a distribution-based framework for globally sharing information between gradients that leads to more stable and effective adversarial training.

@highlight

Proposes to replace single-sample discriminators in adversarial training with discriminators that explicitly operate on distributions of examples.

@highlight

Theory on two-sample tests and MMD and how can be beneficially incorporated into GAN framework.