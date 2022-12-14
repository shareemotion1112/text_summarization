Due to the phenomenon of “posterior collapse,” current latent variable generative models pose a challenging design choice that either weakens the capacity of the decoder or requires altering the training objective.

We develop an alternative that utilizes the most powerful generative models as decoders, optimize the variational lower bound, and ensures that the latent variables preserve and encode useful information.

Our proposed δ-VAEs achieve this by constraining the variational family for the posterior to have a minimum distance to the prior.

For sequential latent variable models, our approach resembles the classic representation learning approach of slow feature analysis.

We demonstrate our method’s efficacy at modeling text on LM1B and modeling images: learning representations, improving sample quality, and achieving state of the art log-likelihood on CIFAR-10 and ImageNet 32 × 32.

Deep latent variable models trained with amortized variational inference have led to advances in representation learning on high-dimensional datasets BID25 BID33 .

These latent variable models typically have simple decoders, where the mapping from the latent variable to the input space is unimodal, for example using a conditional Gaussian decoder.

This typically results in representations that are good at capturing the global structure in the input, but fail at capturing more complex local structure (e.g. texture BID28 ).

In parallel, advances in autoregressive models have led to drastic improvements in density modeling and sample quality without explicit latent variables BID39 .

While these models are good at capturing local statistics, they often fail to produce globally coherent structures BID30 .Combining the power of tractable densities from autoregressive models with the representation learning capabilities of latent variable models could result in higher-quality generative models with useful latent representations.

While much prior work has attempted to join these two models, a common problem remains.

If the autoregressive decoder is expressive enough to model the data density, then the model can learn to ignore the latent variables, resulting in a trivial posterior that collapses to the prior.

This phenomenon has been frequently observed in prior work and has been referred to as optimization challenges of VAEs by BID5 , the information preference property by , and the posterior collapse problems by several others (e.g. van den BID40 BID14 .

Ideally, an approach that mitigates posterior collapse would not alter the evidence lower bound (ELBO) training objective, and would allow the practitioner to leverage the most recent advances in powerful autoregressive decoders to improve performance.

To the best of our knowledge, no prior work has succeeded at this goal.

Most common approaches either change the objective BID20 BID1 Zhao et al., 2017; BID29 Goyal et al., 2017) , or weaken the decoder BID5 BID18 .

Additionally, these approaches are often challenging to tune and highly sensitive to hyperparameters BID1 .In this paper, we propose δ-VAEs, a simple framework for selecting variational families that prevent posterior collapse without altering the ELBO training objective or weakening the decoder.

By restricting the parameters or family of the posterior, we ensure that there is a minimum KL divergence, δ, between the posterior and the prior.

We demonstrate the effectiveness of this approach at learning latent-variable models with powerful decoders on images (CIFAR-10, and ImageNet 32 × 32), and text (LM1B).

We achieve state of the art log-likelihood results with image models by additionally introducing a sequential latent-variable model with an anti-causal encoder structure.

Our experiments demonstrate the utility of δ-VAEs at learning useful representations for downstream tasks without sacrificing performance on density modeling.

Our proposed δ-VAE builds upon the framework of variational autoencoders (VAEs) BID25 BID33 for training latent-variable models with amortized variational inference.

Our goal is to train a generative model p(x, z) to maximize the marginal likelihood log p(x) on a dataset.

As the marginal likelihood requires computing an intractable integral over the unobserved latent variable z, VAEs introduce an encoder network q(z|x) and optimize a tractable lower bound (the ELBO): log p(x) ≥ E z∼q(z|x) [log p(x|z)] − D KL (q(z|x) p(z)).

The first term is typically referred to as the reconstruction term, and the second term (KL) the rate term, as it measures how many nats on average are required to send the latent variables from the encoder (q(z|x)) to the decoder (p(z|x)) using a code designed for the prior (p(z)) BID21 BID1 .The problem of posterior collapse is that the rate term, D KL (q(z|x) p(z)) reduces to 0.

In this case, the approximate posterior q(z|x) equals the prior p(z), thus the latent variables do not carry any information about the input x. A necessary condition if we want representations to be meaningful is to have the rate term be positive.

In this paper we address the posterior collapse problem with structural constraints so that the KL divergence between the posterior and prior is lower bounded by design.

This can be achieved by choosing families of distributions for the prior and approximate posterior, p θ (z) and q φ (z|x) such that min θ,φ D KL (q φ (z|x) p θ (z)) ≥ δ.

We refer to δ as the committed rate of the model.

Note that a trivial choice for p and q to have a non-zero committed rate is to set them to Gaussian distributions with fixed (but different) variance term.

We study a variant of this case in the experiments, and provide more details of this setup in Appendix D. In the following section we describe our choices for p θ and q φ , but others should also be explored in future work.

Data such as speech, natural images and text exhibit strong spatio-temporal continuity.

Our aim is to model variations in such data through latent variables, so that we have control over not just the global characteristics of the generated samples (e.g., existence of an object), but also can influence their finer, often shifting attributes such as texture and pose in the case of natural images, tone, volume and accent in the case of speech, or style and sentiment in the case of natural language.

Sequences of latent variables can be an effective modeling tool for expressing the occurrence and evolution of such features throughout the sequence.

To construct a δ-VAE in the sequential setting, we combine a mean field posterior with a correlated prior in time.

We model the posterior distribution of each timestep as q(z t |x) = N (z t ; µ t (x), σ t (x)).

For the prior, we use a first-order linear autoregressive process (AR(1)), where z t = αz t−1 + t with t zero mean Gaussian noise with constant variance σ 2 .

The conditional probability for the latent variable can be expressed as p(z t |z <t ) = N (z t ; αz t−1 , σ ).

This process is wide-sense stationary (that is, having constant sufficient statistics through its time evolution) if |α| < 1.

If so, then z t has zero mean and variance of σ 2 1−α 2 .

It is thus convenient to choose σ = √ 1 − α 2 so that the variance is constant over time.

The mismatch in the correlation structure of the prior and the posterior results in the following positive lower bound on the KL-divergence between the two distributions (see Appendix C for derivation): DISPLAYFORM0 where n is the length of the sequence and d is the dimension of the latent variable at each timestep.

The committed rate between the prior and the posterior is easily controlled by equating the right hand side of the inequality in equation 1 to a given rate δ and solving for α.

In Fig. 1 , we show the scaling of the minimum rate as a function of α and the behavior of δ-VAE in 2d.

Figure 1: Effect of δ in a toy model.

Fitting an uncorrelated Gaussian for the posterior, q φ (z), to a correlated Gaussian prior, p α (z), by minimizing D KL (q φ (z) p α (z)) over φ.

Left: committed rate (δ) as a function of the prior squared correlation α and the dimensionality n. Right: contours of the optimal posterior and prior in 2d.

As the correlation increases, the minimum rate grows.

The AR(1) prior over the latent variables specifies the degree of temporal correlation in the latent space.

As the correlation α approaches one, the prior trajectories get smoother .

On the other hand in the limit of α approaching 0, the prior becomes the same as the independent standard Gaussian prior where there are no correlations between timesteps.

This pairing of independent posterior with a correlated prior is related to the probabilistic counterpart to Slow Feature Analysis BID44 in BID37 .

SFA has been shown to be an effective method for learning invariant spatio-temporal features BID44 .

In our models, we infer latent variables with multiple dimensions per timestep, each with a different slowness filter imposed by a different value of α, corresponding to features with different speed of variation.

Having a high capacity autoregressive network as the decoder implies that it can accurately estimate p(x t |x <t ).

Given this premise, what kind of complementary information can latent variables provide?

Encoding information about the past seems wasteful as the autoregressive decoder has full access to past observations already.

On the other hand, if we impose conditional independence between observations and latent variables at other timesteps given the current one (i.e., p(x t |z) = p(x t |z t )), there will then be at best (by the data processing inequality BID10 ) a break-even situation between the KL cost of encoding information in z t and the resulting improvement in the reconstruction loss.

There is therefore no advantage for the model to utilize the latent variable even if it would transmit to the decoder the unobserved x t .

The situation is different when z t can inform the decoder at multiple timesteps, encoding information about x t and x >t .

In this setting, the decoder pays the KL cost for the mutual information once, but is able to leverage the transmitted information multiple times to reduce its entropy about future predictions.

To encourage the generative model to leverage the latents for future timesteps, we introduce an anti-causal structure for the encoder where the parameters of the variational posterior for a timestep cannot depend on past observations FIG1 .

Alternatively, one can consider a non-causal structure that allows latents be inferred from all observations.

In this non-causal setup there is no temporal order in either the encoder or the decoder, thus the model resembles a standard non-temporal latentvariable model.

While the anti-causal structure is a subgraph of the non-causal structure, we find that the anti-causal structure often performs better, and we compare both approaches in different settings in Appendix F.1.

The main focus of our work is on representation learning and density modeling in latent variable models with powerful decoders.

Earlier work has focused on this kind of architecture, but has addressed the problem of posterior collapse in different ways.

In terms of our architecture, the decoders for our image models build on advances in autoregressive modeling from van den BID39 ; BID35 ; ; .

Unlike prior models, we use sequential latent variables to generate the image row by row.

This differs from BID17 , where the latent variables are sequential but the entire image is generated at each timestep.

Our sequential image generation model resembles latent variable models used for timeseries BID9 BID4 BID13 but does not rely on KL annealing, and has an additional autoregressive dependence of the outputs over time (rows of the image).

Another difference between our work and previous sequential latent variable models is our proposed anti-causal structure for the inference network (see Sect.

2.2).

We motivate this structure from a coding efficiency and representation learning standpoint and demonstrate its effectiveness empirically in Sect.

4.

For textual data, we use the Transformer architecture from BID42 as our main blueprint for the decoder.

As shown in Sect.

4, our method is able to learn informative latent variables while preserving the performance of these models in terms of likelihoods.

To prevent posterior collapse, most prior work has focused on modifying the training objective.

Bowman et al. FORMULA0 ; BID46 ; and BID18 use an annealing strategy, where they anneal the weight on the rate from 0 to 1 over the course of training.

This approach does not directly optimize a lower bound on likelihood for most of training, and tuning the annealing schedule to prevent collapse can be challenging (see Sect.

4).

Similarly, BID20 proposes using a fixed coefficient > 1 on the rate term to learn disentangled representations.

Zhao et al. (2017) adds a term to the objective to pick the model with maximal rate.

Other works use auxiliary tasks such as secondary low-resolution reconstructions with non-autoregressive decoders BID29 or predicting the state of the backward LSTM in the encoder (Goyal et al., 2017) to encourage the utilization of latent variables. ; BID7 use free-bits to allow the model to hit a target minimum rate, but the objective is non-smooth which leads to optimization difficulties in our hands, and deviations from a lower bound on likelihood when the soft version is used with a coefficient less than 1.

BID1 argue that the ELBO is a defective objective function for representation learning as it does not distinguish between models with different rates, and advocate for model selection based on downstream tasks.

Their method for sweeping models was to use β-VAE with different coefficients, which can be challenging as the mapping from β to rate is highly nonlinear, and model-and data-dependent.

While we adopt the same rate-distortion perspective as Alemi et al. FORMULA0 , we present a new way of achieving a target rate while optimizing the vanilla ELBO objective.

Most similar to our approach is work on constraining the variational family to regularize the model.

VQ-VAE (van den Oord et al., 2017) uses discrete latent variables obtained by vector quantization of the latent space that, given a uniform prior over the outcome, yields a fixed KL divergence equal to log K, where K is the size of the codebook.

A number of recent papers have also used the von Mises-Fisher (vMF) distribution to obtain a fixed KL divergence and mitigate the posterior collapse problem.

In particular, BID19 ; BID45 ; BID11 use vMF(µ, κ) with a fixed κ as their posterior, and the uniform distribution (i.e. vMF(·, 0)) as the prior.

The mismatching prior-posterior thus give a constant KL divergence.

As such, this approach can be considered as the continuous analogue of VQ-VAE.

Unlike the VQ-VAE and vMF approaches which have a constant KL divergence for every data point, δ-VAE can allow higher KL for different data points.

This allows the model to allocate more bits for more complicated inputs, which has been shown to be useful for detecting outliers in datasets BID2 .

As such, δ-VAE may be considered a generalisation of these fixed-KL approaches.

The Associative Compression Networks (ACN) BID16 ) is a new method for learning latent variables with powerful decoders that exploits the associations between training examples in the dataset by amortizing the description length of the code among many similar training examples.

ACN deviates from the i.i.d training regime of the classical methods in statistics and machine learning, and is considered a procedure for compressing whole datasets rather than individual training examples.

GECO (Jimenez Rezende & Viola, 2018 ) is a recently proposed method to stabilize the training of β-VAEs by finding an automatic annealing schedule for the KL that satisfies a tolerance constraint for maximum allowed distortion, and solving the resulting Lagrange multiplier for the KL penalty.

The value of β, however, does not necessarily approach one, which means that the optimized objective may not be a lower bound for the marginal likelihood.

We applied our method to generative modeling of images on the CIFAR-10 (Krizhevsky et al.) and downsampled ImageNet BID12 ) (32 × 32 as prepared in van den Oord et al. FORMULA0 datasets.

We describe the main components in the following.

The details of our hyperparameters can be found in Appendix E.Decoder: Our decoder network is closest to PixelSNAIL but also incorporates elements from the original GatedPixelCNN (van den BID39 .

In particular, as introduced by BID35 and used in , we use a single channel network to output the components of discretised mixture of logistics distributions for each channel, and linear dependencies between the RGB colour channels.

As in PixelSNAIL, we use attention layers interleaved with masked gated convolution layers.

We use the same architecture of gated convolution introduced in van den BID39 .

We also use the multi-head attention module of BID42 .

To condition the decoder, similar to Transformer and unlike PixelCNN variants that use 1x1 convolution, we use attention over the output of the encoder.

The decoder-encoder attention is causally masked to realize the anti-causal inference structure, and is unmasked for the non-causal structure.

Encoder.

Our encoder also uses the same blueprint as the decoder.

To introduce the anti-causal structure the input is reversed, shifted and cropped by one in order to obtain the desired future context.

Using one latent variable for each pixel is too inefficient in terms of computation so we encode each row of the image with a multidimensional latent variable. (2018) show that VAE performance can suffer when there is a significant mismatch between the prior and the aggregate posterior, q(z) = E x∼D [q(z|x)].

When such a gap exists, the decoder is likely to have never seen samples from regions of the prior distribution where the aggregate posterior assigns small probability density.

This phenomenon, also known as the "posterior holes" problem BID22 , can be exacerbated in δ-VAEs, where the systematic mismatch between the prior and the posterior might induce a large gap between the prior and aggregate posterior.

Increasing the complexity of the variational family can reduce this gap BID32 , but require changes in the objective to control the rate and prevent posterior collapse BID7 .

To address this limitation, we adopt the approaches of van den Oord et al. FORMULA0 ; BID34 and train an auxiliary prior over the course of learning to match the aggregate posterior, but that does not influence the training of the encoder or decoder.

We used a simple autoregressive model for the auxiliary prior p aux : a single-layer LSTM network with conditional-Gaussian outputs.

We begin by comparing our approach to prior work on CIFAR-10 and downsampled ImageNet 32x32 in Table 1 .

As expected, we found that the capacity of the employed autoregressive decoder had a large impact on the overall performance.

Nevertheless, our models with latent variables have a negligible gap compared to their powerful autoregressive latent-free counterparts, while also learning informative latent variables.

In comparison, ) had a 0.03 bits per dimension gap between their latent variable model and PixelCNN++ architecture 1 .

On ImageNet 32x32, our latent variable model achieves on par performance with purely autoregressive Image Transformer .

On CIFAR-10 we achieve a new state of the art of 2.83 bits per dimension, again matching the performance of our autoregressive baseline.

Note that the values for KL appear quite small as they are reported in bits per dimension (e.g. 0.02 bits/dim translates to 61 bits/image encoded in the latents).

The results on CIFAR-10 also demonstrate the effect of the auxiliary prior on improving the efficiency of the latent code; it leads to more than 50% (on average 30 bits per image) reduction in the rate of the model to achieve the same performance.

BID17 ) ≤ 3.85 -DenseNet VLAE Table 1 : Estimated upper bound on negative log-likelihood along with KL-divergence (in parenthesis) in bits per dimension for CIFAR-10 and downsampled ImageNet.

32 × 32 Valid Latent Variable Models ConvDraw (

In this section, we aim to demonstrate that our models learn meaningful representations of the data in the latent variables.

We first investigate the effect of z on the generated samples from the model.

Fig. 3 depicts samples from an ImageNet model (see Appendix for CIFAR-10), where we sample from the decoder network multiple times conditioned on a fixed sample from the auxiliary prior.

We see similar global structure (e.g. same color background, scale and structure of objects) but very different details.

This indicates that the model is using the latent variable to capture global structure, while the autoregressive decoder is filling in local statistics and patterns.

For a more quantitative assessment of how useful the learned representations are for downstream tasks, we performed linear classification from the representation to the class labels on CIFAR-10.

We also study the effect of the chosen rate of the model on classification accuracy as illustrated in FIG4 , along with the performance of other methods.

We find that generally a model with higher rate gives better classification accuracy, with our highest rate model, encoding 92 bits per image, giving the best accuracy of 68%.

However, we find that improved log-likelihood does not necessarily lead to better linear classification results.

We caution that an important requirement for this task is the linear separability of the learned feature space, which may not align with the desire to learn highly compressed representations.

We performed more extensive comparisons of δ-VAE with other approaches to prevent posterior collapse on the CIFAR-10 dataset.

We employ the same medium sized encoder and decoder for Fig. 3a shows multiple samples from p(x|z) for a fixed z ∼ p aux (z).

Each image in Fig. 3b is decoded using a different sample from p aux (z).evaluating all methods as detailed in Appendix E. FIG4 reports the rate-distortion results of our experiments for the CIFAR-10 test set.

To better highlight the difference between models and to put into perspective the amount of information that latent variables capture about images, the rate and distortion results in FIG4 are reported in bits per images.

We only report results for models that encode a non-negligible amount information in latent variables.

Unlike the committed information rate approach of δ-VAE, most alternative solutions required considerable amount of effort to get the training converge or prevent the KL from collapsing altogether.

For example, with linear annealing of KL BID5 , despite trying a wide range of values for the end step of the annealing schedule, we were not able to train a model with a significant usage of latent variables; the KL collapsed as soon as β approached one.

A practical advantage of our approach is its simple formula to choose the target minimum rate of the model.

Targeting a desired rate in β-VAE, on the other hand, proved to be difficult, as many of our attempts resulted in either collapsed KL, or very large KL values that led to inefficient inference.

As reported in , we also observed that optimising models with the free-bits loss was challenging and sensitive to hyperparameter values.

To assess each methods tendency to overfit across the range of rates, we also report the rate-distortion results for CIFAR-10 training sets in Appendix F. While β-VAEs do find points along the ratedistortion optimal frontier on the training set, we found that they overfit more than δ-VAEs, with δ-VAEs dominating the rate-distortion frontier on heldout data.

Next, we compare the performance of the anti-causal encoder structure with the non-causal structure on the CIFAR-10 dataset discussed in Sect.

2.2.

The results for several configurations of our model are reported in the Appendix TAB3 .

In models where the decoder is not powerful enough (such as our 6-layer PixelCNN that has no attention and consequently a receptive field smaller than the causal context for most pixels), the anti-causal structure does not perform as well as the noncausal structure.

The performance gap is however closed as the decoder becomes more powerful and its receptive field grows by adding self-attention and more layers.

We observed that the anticausal structure outperforms the non-causal encoder for very high capacity decoders, as well as for medium size models with a high rate.

We also repeated these experiments with both anti-causal and non-causal structures but without imposing a committed information rate or using other mitigation strategies, and found that neither structure by itself is able to mitigate the posterior collapse issue; in both cases the KL divergence drops to negligible values (< 10 −8 bits per dimension) only after a few thousand training steps, and never recovers.

For our experiments on natural language, we used the 1 Billion Words or LM1B BID6 dataset in its processed form in the Tensor2Tensor ) codebase 2 .

Our employed architecture for text closely follows the Transformer network of BID42 .

Our sequence of latent variables has the same number of elements as in the number of tokens in the input, each having two dimensions with α = 0.2 and 0.4.

Our decoder uses causal self-attention as in BID42 .

For the anti-causal structure in the encoder, we use the inverted causality masks as in the decoder to only allow looking at the current timestep and the future.

Quantitatively, our model achieves slightly worse log-likelihood compared to its autoregressive counterpart TAB2

In this work, we have demonstrated that δ-VAEs provide a simple, intuitive, and effective solution to posterior collapse in latent variable models, enabling them to be paired with powerful decoders.

Unlike prior work, we do not require changes to the objective or weakening of the decoder, and we can learn useful representations as well as achieving state-of-the-art likelihoods.

While our work presents two simple posterior-prior pairs, there are a number of other possibilities that could be explored in future work.

Our work also points to at least two interesting challenges for latentvariable models: (1) can they exceed the performance of a strong autoregressive baseline, and (2) can they learn representations that improve downstream applications such as classification?

DISPLAYFORM0 B DERIVATION OF THE KL-DIVERGENCE BETWEEN AR(1) AND DIAGONAL GAUSSIAN, AND ITS LOWER-BOUND DISPLAYFORM1 Noting the analytic form for the KL-divergence for two uni-variate Gaussian distributions: DISPLAYFORM2 we now derive the lower-bound for KL-divergence.

To avoid clutter we assume a single dimension per timestep but extend the results to the general multivariate case at the end of this section.

DISPLAYFORM3 C DERIVATION OF THE LOWER-BOUND Removing non-negative quadratic terms involving µ i in equation 3 and expanding back f inside the summation yields DISPLAYFORM4 Consider f a (x) = ax − ln(x) − 1 and its first and second order derivatives, f a (x) = a − 1 x and f a (x) ≥ 0.

Thus, f a is convex and obtains its minimum value of ln(a) at x = a −1 .

Substituting σ DISPLAYFORM5 When using multi-dimensional z i at each timestep, the committed rate is the sum of the KL for each individual dimension: DISPLAYFORM6 The most common choice for variational families is to assume that the components of the posterior are independent, for example using a multivariate Gaussian with a diagonal covariance: q φ (z|x) = N (z; µ q (x), σ q (x)).

When paired with a standard Gaussian prior, p(z) = N (z; 0, 1), we can guarantee a committed information rate δ by constraining the mean and variance of the variational family (see Appendix C) DISPLAYFORM7 We can, thus, numerically solve DISPLAYFORM8 where the above equation has a solution for µ q , and the committed rate δ.

Posterior parameters can thus be parameterised as: DISPLAYFORM9 Where φ parameterises the data-dependent part of µ q ad σ q , which allow the rate to go above the designated lower-bound δ.

We compare this model with the temporal version of δ-VAE discussed in the paper and report the results in Table 3 .

While independent δ-VAE also prevents the posterior from collapsing to prior, its performance in density modeling lags behind temporal δ-VAE.

Test ELBO (KL) Accuracy Independent δ-VAE (δ = 0.08) 3.08 (0.08) 66% Temporal δ-VAE (δ = 0.08) 3.02 (0.09) 65% Table 3 : Comparison of independent Gaussian delta-VAE and temporal delta-VAE with AR(1) prior on CIFAR-10 both targeting the same rate.

While both models achieve a KL around the target rate and perform similarly in the downstream linear classification task, the temporal model with AR(1) prior achieves significantly better marginal likelihood.

In this section we provide the details of our architecture used in our experiments.

The overall architecture diagram is depicted in FIG8 .

To establish the anti-causal context for the inference network we first reverse the input image and pad each spatial dimension by one before feeding it to the encoder.

The output of the encoder is cropped and reversed again.

As show in FIG8 , this gives each pixel the anti-causal context (i.e., pooling information from its own value and future values).

We then apply average pooling to this representation to give us row-wise latent variables, on which the decoder network is conditioned.

The exact hyper-parameters of our network is detailed in Table 4 .

We used dropout only in our decoder and applied it the activations of the hidden units as well as the attention matrix.

As in BID42 , we used rectified linear units and layer normalization BID3 after the multi-head attention layers.

We found layer normalization to be essential for stabilizing training.

We trained with the Adam optimizer BID24 and used the learning rate schedule proposed in BID42 ) with a few tweaks as in the following formulae: DISPLAYFORM0 We use multi-dimensional latent variables per each timestep, with different slowness factors linearly spaced between a chosen interval.

For our ablation studies, we chose corresponding hyperparameters of each method we compare against to target rates between 25-100 bits per image.

Table 4 : Hyperparameter values for the models used for experiments.

The subscripts e, d, aux respectively denote the encoder, the decoder, and the LSTM auxiliary prior.

l is the number of layers, h is the hidden size of each layer, r is the size of the residual filter, a is the number of attention layers interspersed with gated convolution layers of PixelCNN, n dmol is the number of components in the discrete mixture of logistics distribution, do d is the probability of dropout applied to the decoder, z is the dimensionality of the latent variable used for each row, and the alpha column gives the range of the AR(1) prior hyper-parameter for each latent.

DISPLAYFORM1 We developed our code using Tensorflow BID0 .

Our experiments on natural images were conducted on Google Cloud TPU accelerators.

For ImageNet, we used 128 TPU cores with batch size of 1024.

We used 8 TPU cores for CIFAR-10 with batch size of 64.

The architecture of our model for text experiment is closely based on the Transformer network of BID42 .

We realize the encoder anti-causal structure by inverting the causal attention masks to upper triangular bias matrices.

The exact hyper-parameters are summarized in Table 5 .

Table 5 : Hyperparameter values for our LM1B experiments.

l is the number of layers, h is the hidden size of each layer, r is the size of the residual filters, do is the probability of dropout, z is the dimensionality of the latent variable, and the alpha column gives the range of the AR(1) prior hyper-parameter for each latent dimension.

For our ablation studies on CIFAR-10, we trained our model with the configuration listed in Table 4 .

After training the model, we inferred the mean of the posterior distribution corresponding to each training example in the CIFAR-10 test set, and subsequently trained a multi-class logistic regression classifier on top of it.

For each model, the linear classifier was optimized for 100 epochs using the Adam optimizer with the starting learning rate of 0.003.

The learning rate was decayed by a factor of 0.3 every 30 epochs.

We also report the rate-distortion curves for the CIFAR-10 training set in FIG11 .

In contrast to the graph of FIG4 for the test set, δ-VAE achieves relatively higher negative log-likelihood on the training set compared to other methods, especially for larger rates.

This suggests that δ-VAE is less prone to overfitting compared to β-VAE and free-bits.

In TAB3 , we report the details of evaluating our proposed anti-causal encoder architecture (discussed in Sect.

2.2) against the non-causal architecture in which there is no restriction on the connectivity of the encoder network.

The reported experiments are conducted on the CIFAR-10 dataset.

We trained 4 different configurations of our model to provide comparison in different capacity and information rate regimes, using the temporal δ-VAE approach to prevent posterior collapse.

We found that the anti-causal structure is beneficial when the decoder has sufficiently large receptive field, and also when encoding relatively high amount of information in latent variables.

DISPLAYFORM0

It is generally expected that images from the same class are mapped to the same region of the latent space.

FIG13 illustrates the t-SNE (van der Maaten & Hinton, 2008) plot of latent variables inferred from 3000 examples from the test set of CIFAR-10 colour coded based on class labels.

As can also be seen on the right hand plot, classes that are closest are also mostly the one that have close semantic and often visual relationships (e.g., cat and dog, or deer and horse).H ADDITIONAL SAMPLES Figure 10: Random samples from the auxiliary (left) and AR(1) (right) priors of our high-rate (top) and low-rate(bottom) CIFAR-10 models.

The high-rate (low-rate) model has -ELBO of 2.90 (2.83) and KL of 0.10 (0.01) bits/dim.

Notice that in the high rate model that has a larger value of α, samples from the AR(1) prior can turn out too smooth compared to natural images.

This is because of the gap between the prior and the marginal posterior, which is closed by the auxiliary prior.

==== Interpolating dimension 0 ==== The company's stock price is also up for a year-on-year rally, when the The company's shares are trading at a record high for the year, when they were trading at The company's shares were trading at $3.00, down from their 52-week low The company's shares fell $1.14, or 5.7 percent, to $ UNK The company, which is based in New York, said it would cut 1,000 jobs in the The two-day meeting, held at the White House, was a rare opportunity for the United States The company, which is based in New York, said it was looking to cut costs, but added The company is the only company to have a significant presence in China.

The company is the only company to have a significant presence in the North American market.

The two men, who were arrested, have been released.==== Interpolating dimension 1 ==== In the meantime, however, the company is taking the necessary steps to keep the company in the UNK In the meantime, however, the company is expected to take some of the most aggressive steps in the In the meantime, the company is expected to report earnings of $2.15 to $4.

The two men, who were both in their 20s , were arrested on suspicion of causing death by dangerous The company said it was "disappointed" by a decision by the U.S. Food and Drug The company said it would continue to provide financial support to its business and financial services clients.

The new plan would also provide a new national security dimension to U.S.-led efforts to "I've always been a great customer and thereś always thereś a good chance "

It's a great personal decision...

FIG1 : One at a time interpolation of latent dimensions of a sample from the AR(1) prior.

The sentences of each segment are generated by sampling a 32 element sequence of 2D random vectors from the autoregressive prior, fixing one dimension interpolating the other dimension linearly between µ ± 3σ.==== Interpolating dimension 0 ==== "I'll be in the process of making the case," he said.

"

I've got to take the best possible shot at the top," he said "I'm not going to take any chances," he said.

"I'm not going to take any chances," he said.

"I'm not going to take any chances," he said.

We are not going to get any more information on the situation," said a spokesman for the U. N. mission in Afghanistan, which is to be formally We are not going to get the money back," he said.

We are not going to get the money back," said one of the co -workers. "

We are not going to get the money back," said the man. "

We are not going to get a lot of money back," said the man.==== Interpolating dimension 1 ==== The company said the company, which employs more than 400 people, did not respond to requests for comment, but did not respond to an email seeking comment, which The new rules, which are expected to take effect in the coming weeks, will allow the government to take steps to ensure that the current system does not take too "The only thing that could be so important is the fact that the government is not going to be able to get the money back, so the people are taking "I'm not sure if the government will be able to do that," he said.

"

We are not going to get any more information about the situation," said Mr. O'Brien.

"

It's a very important thing to have a president who has a strong and strong relationship with our country," said Mr. Obama, who has been the "It's a very important thing to have a president who has a great chance to make a great president," said Mr. Obama, a former senator from "It's a very important decision," said Mr. Obama.

"

It's a very difficult decision to make," said Mr. McCain.

Figure 13: One at a time interpolation of latent dimensions of a sample from the auxiliary prior.

The generation procedure is identical to FIG1 with the exception that the initial vector is sampled from the auxiliary prior.

The company is now the world's cheapest for consumers .

The company is now the world's biggest producer of oil and gas, with an estimated annual revenue of $2.2 billion.

The company is now the world's third-largest producer of the drug, after Pfizer and AstraZeneca, which is based in the UK.The company is now the world's biggest producer of the popular games console, with sales of more than $1bn (312m) in the US and about $3bn in the UK.

The company is now the world's largest company, with over $7.5 billion in annual revenue in 2008, and has been in the past for more than two decades.

The company is now the world's second-largest, after the cellphone company, which is dominated by the iPhone, which has the iPhone and the ability to store in -store, rather than having to buy, the product, said, because of the Apple-based device.

The company is now the world's biggest manufacturer of the door-to-door design for cars and the auto industry.

The company is now the world's third-largest maker of commercial aircraft, behind Boeing and Airbus.

The company is now the world's largest producer of silicon, and one of the biggest producers of silicon in the world.

The company is now the world's largest maker of computer -based software, with a market value of $4.2 billion (2.6 billion) and an annual turnover of $400 million (343 million).

FIG4 : Text completion samples.

For each sentence we prime the decoder with a fragment of a random sample from the validation set (shown in bold), and condition the decoder on interpolations between two samples from the latent space.

@highlight

 Avoid posterior collapse by lower bounding the rate.

@highlight

Presents an approach to preventing posterior collapse in VAEs by limiting the family of the variational approximation to the posterior

@highlight

This paper introduces a constraint on the family of variational posteriors such that the KL term can be controlled to combat posterior collapse in deep generative models such as VAEs