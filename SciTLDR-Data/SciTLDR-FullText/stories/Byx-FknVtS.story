Despite promising progress on unimodal data imputation (e.g. image inpainting), models for multimodal data imputation are far from satisfactory.

In this work, we propose variational selective autoencoder (VSAE) for this task.

Learning only from partially-observed data, VSAE can model the joint distribution of observed/unobserved modalities and the imputation mask, resulting in a unified model for various down-stream tasks including data generation and imputation.

Evaluation on synthetic high-dimensional and challenging low-dimensional multimodal datasets shows significant improvement over state-of-the-art imputation models.

Modern deep learning techniques rely heavily on extracting information from large scale datasets of clean and complete training data, such as labeled data or images with all pixels.

Practically these data is costly due to the limited resources or privacy concerns.

Having a model that learns and extracts information from partially-observed data will largely increase the application spectrum of deep learning models and provide benefit to down-stream tasks, e.g. data imputation, which has been an active research area.

Despite promising progress, there are still challenges in learning effective imputation models: 1) Some prior works focus on learning from fully-observed data and then performing imputation on partially-observed data (Suzuki et al., 2016; Ivanov et al., 2019) ; 2) They usually have strong assumptions on missingness mechanism (see Appendix A.1) such as data is missing completely at random (MCAR) (Yoon et al., 2018) ; 3) Some other works explore only unimodal imputation such as image in-painting for high-dimensional data (Ivanov et al., 2019; Mattei and Frellsen, 2019) .

Modeling any combination of data modalities has not been well-established yet.

This can limit the potential of such models since raw data in real-life is usually acquired in a multimodal manner (Ngiam et al., 2011) .

A class of prior works focus on learning the conditional likelihood of the modalities (Sohn et al., 2015; Pandey and Dukkipati, 2017) .

However, they require complete data during training and cannot handle arbitrary conditioning.

In practice, one or more of the modalities maybe be missing, leading to a challenging multimodal data imputation task.

For more on related works, see Appendix A.2.

The unimodal/multimodal proposal networks are employed by selection indicated by the arrows.

Standard normal prior is ignored for simplicity.

φ, ψ, θ and are the parameters of each modules.

All components are trained jointly.

We propose Variational Selective Autoencoder (VSAE) for multimodal data generation and imputation.

It can model the joint distribution of data and mask and avoid the limited assumptions such as MCAR.

VSAE is optimized efficiently with a single variational objective.

The contributions are summarized as: (1) A novel variational framework to learn from partially-observed multimodal data; (2) VSAE can learn the joint distribution of observed/unobserved modalities and the mask, resulting in a unified model for various down-stream tasks including data generation/imputation with relaxed assumptions on missigness mechanism; (3) Evaluation on both synthetic high-dimensional and challenging low-dimensional multimodal datasets shows improvement over the state-of-the-art data imputation models.

Problem Statement.

Let x = [x 1 , x 2 ..., x M ] be the complete data with M modalities.

The size of each x i may vary.

A binary mask variable m ∈ {0, 1} M : m i = 1 indicates x i is observed and m i = 0 indicates x i is unobserved.

The set of observed modalities O = {i|m i = 1} and unobserved modalities U = {i|m i = 0} are complementary.

Accordingly, we denote the representation of observed and unobserved modalities with x o = [x i |m i = 1] and x u = [x i |m i = 0].

Assuming x and m are dependent, we aim to model the joint distribution p(x, m).

As a result, VSAE can be used for both imputation and generation.

Proposed Model.

The high-level overview of VSAE (see Figure 1) is that the multimodal data is encoded to a latent space factorized w.r.t.

the modalities.

The latent variable of each modality is selectively chosen between a unimodal encoder (if the modality is observed) or a multimodal encoder (if the modality is unobserved).

Next all the modalities and mask are reconstructed by decoding the aggregated latent codes.

Mathematically, we aim to model the joint distribution of the data x = [x o , x u ] and mask m. Following VAE formulation (see Appendix A.3), we derive the ELBO for log p(x, m) with approximate posterior q(z|x, m):

(1)

Decoder The probability distribution factorizes over modalities assuming that reconstructions are conditionally independent given complete latent variables of all modalities:

Selective proposal distribution for encoders Following Tsai et al. (2019), we assume latent variables factorizes w.r.t the modalities

.

Given this, we define the proposal distribution for each modality as

This is based on the intuitive assumption that the latent space of each modality is independent of the others given its data is observed.

If the modality is missing, its latent variable is selectively inferred from other observed modalities.

For partially-observed setting, x u is unavailable even during training.

Thus, we define the objective function for training by taking expectation of the ELBO in Equation (1) over x u .

Only one term in Equation (2) is dependent on x u , so the final objective is derived as

We approximate E x j [log p θ (x j |m, z)], j ∈ U by sampling x j from the prior network (standard normal) and passing through the decoder.

Our experiments show even a single sample is sufficient to learn the model effectively.

In fact, the prior network can be used as a self-supervision mechanism to find the most likely samples which dominate the other samples when taking the expectation.

See Appendix B for more details.

We evaluate our model on high-dimensional multi-modal data and low-dimensional tabular data in comparison with state-of-the-art latent variable models.

To test the robustness of our model, we evaluate our model under various challenging missingness mechanisms.

Table 1 : Data Imputation.

Missing ratio is 0.5.

For all lower is better.

Last two rows are trained with fully-observed data.

We show mean/std over 3 independent runs.

∆ ≤ 0.001.

Low-dimensional tabular data.

We choose UCI repository datasets (contains both numerical and categorical data, training/test split as 4/1 and 20% of training set for validation).

We randomly sample from independent Bernoulli distributions with pre-defined missing ratio to generate masks that are fixed during training and test.

In Table 1 , we observe that VSAE trained with partially-observed data outperforms other baselines, even those models trained with fully-observed data on some datasets.

We argue this is due to two potential reasons: (1) the mask provides a natural way of dropout on the data space, thereby, helping the model to generalize; (2) If the data is noisy or has outliers, which is common in low-dimensional data, learning from partially-observed data can improve the results by ignoring these data.

Figure 2 indicates VSAE are more robust to missing ratio.

High-dimensional multimodal data.

We synthesize two bimodal datasets using MNIST and SVHN datasets: Synthetic1 contains randomly paired two digits in MNIST as [0, 9] , [1, 8] , [2, 7] , [3, 6] , [4, 5] ; Synthetic2 contains pairs of one digit in MNIST with a random same digit in SVHN.

VSAE has better performance with lower variance (see Table 1 ).

Experiments also indicate that VSAE is robust under different missing ratios, whereas other baselines are sensitive to the missing ratio, which is consistent as UCI experiments.

We believe this is because of the underlying mechanism of proper proposal distribution selection and prior sharing.

The separation of unimodal/multimodal encoders helps the model to attend to the observed data, while baselines only have single proposal distribution inferred from the whole input.

Thus, VSAE can easily ignore unobserved noisy modalities and attends on observable useful modalities, but baselines rely on neural networks to extract useful information from the whole data (which is dominated by missing information in case of high missing ratio).

Figure 3: Data Generation.

Generated w/o conditional information.

As shown, the correspondence between modalities (pre-defined pairs) are preserved.

After training, we sample from prior to generate the data and mask.

In UCI experiments, We calculate the proportion of 0 in generated mask vectors over 100 samples and average on all experiments, we get 0.3123 ± 0.026, 0.4964 ± 0.005, 0.6927 ± 0.013 for missing ratio of 0.3, 0.5, 0.7.

It indicates VSAE can learn mask distribution.

We observe that conditions on the reconstructed mask in the data decoders improve the performance.

We believe this is because the mask vector can inform the data decoder about the missingness in data space since the latent space is shared by all modalities thereby allowing it to generate data from the selective proposal distribution.

Figure 3 Following (Little and Rubin, 1986) , the imputation process is to learn a generative distribution for unobserved missing data.

To be consistent with notations in Section 2, let x = [x 1 , x 2 , ..., x M ] be the complete data of all modalities where x i denote the feature representation for the i-th modality.

We also define m ∈ {0, 1} M as the binary mask vector, where m i = 1 indicates if the i-th modality is observed, and m i = 0 indicates if it is unobserved:

Given this, the observed data x o and unobserved data x u are represented accordingly:

In the standard maximum likelihood setting, the unknown parameters are estimated by maximizing the following marginal likelihood, integrating over the unknown missing data values:

Little and Rubin (1986) characterizes the missingness mechanism p(m|x o , x u ) in terms of independence relations between the complete data x = [x o , x u ] and the mask m:

• Missing completely at random (MCAR):

• Missing at random (MAR):

• Not missing at random (NMAR):

Most previous data imputation methods works under MCAR or MAR assumptions since

With such decoupling, we do not need missing information to marginalize the likelihood, and it provides a simple but approximate framework to learn from partially-observed data.

Data Imputation.

Classical imputation methods such as MICE (Buuren and GroothuisOudshoorn, 2010) and MissForest (Stekhoven and Bühlmann, 2011) learn discriminative models to impute missing features from observed ones.

With recent advances in deep learning, several deep imputation models have been proposed based on autoencoders (?

Gondara and Wang, 2017; Ivanov et al., 2019) , generative adversarial nets (GANs) (Yoon et al., 2018; Li et al., 2019) , and autoregressive models (?).

GAN-based imputation method GAIN proposed by Yoon et al. (2018) assumes that data is missing completely at random.

Moreover, this method does not scale to high-dimensional multimodal data.

Several VAE based data imputation methods (Ivanov et al., 2019; Nazabal et al., 2018; Mattei and Frellsen, 2019) have been proposed in recent years.

Ivanov et al. (2019) formulated variational autoencoders with arbitrary conditioning (VAEAC) for data imputation which allows generation of missing data conditioned on any combination of observed data.

This algorithm needs complete data during training cannot learn from partially-observed data only.

Nazabal et al. (2018) and Mattei and Frellsen (2019) modified VAE formulation to model the likelihood of the observed data only.

However, they require training of a separate generative network for each dimension thereby increasing computational requirements.

In contrast, our method aims to model joint distribution of observed and unobserved data along with the missingness pattern (imputation mask).

This enables our model to perform both data generation and imputation even under relaxed assumptions on missingness mechanism (see Appendix A.1).

A class of prior works such as conditional VAE (Sohn et al., 2015) and conditional multimodal VAE (Pandey and Dukkipati, 2017) focus on learning the conditional likelihood of the modalities.

However, these models requires complete data during training and cannot handle arbitrary conditioning.

Alternatively, several generative models aim to model joint distribution of all modalities (Ngiam et al., 2011; ?; Sohn et al., 2014; Suzuki et al., 2016) .

However, multimodal VAE based methods such as joint multimodal VAE (Suzuki et al., 2016 ) and multimodal factorization model (MFM) (Tsai et al., 2019) require complete data during training.

On the other hand, Wu and Goodman (2018) proposed another multimodal VAE (namely MVAE) can be trained with incomplete data.

This model leverages a shared latent space for all modalities and obtains an approximate joint posterior for the shared space assuming each modalities to be factorized.

However, if training data is complete, this model cannot learn the individual inference networks and consequently does not learn to handle missing data during test.

Building over multimodal VAE approaches, our model aims to address the shortcomings above within a flexible framework.

In particular, our model can learn multimodal representations from partially observed training data and perform data imputation from arbitrary subset of modalities during test.

By employing a factorized multimodal representations in the latent space it resembles disentangled models which can train factors specialized for learning from different parts of data (Tsai et al., 2019) .

Variational Autoencoder (VAE) (Kingma and Welling, 2013 ) is a probabilistic generative model, where data is constructed from a latent variable z with a prior distribution p(z).

It is composed of an inference network and a generation network to encode and decode data.

To model the likelihood of data, the true intractable posterior p(z|x) is approximated by a proposal distribution q φ (z|x), and the whole model is trained until ideally the decoded reconstructions from the latent codes sampled from the approximate posterior match the training data.

In the generation module, p θ (x|z), a decoder realized by a deep neural network parameterized by θ, maps a latent variable z to the reconstructionx of observation x.

In the inference module, an encoder parameterized by φ produces the sufficient statistics of the approximation posterior q φ (z|x) (a known density family where sampling can be readily done).

In vanilla VAE setting, by simplifying approximate posterior as a parameterized diagonal normal distribution and prior as a standard diagonal normal distribution N (0, I), the training criterion is to maximize the following evidence lower bound (ELBO) w.r.t.

θ and φ.

where D KL denotes the Kullback-Leibler (KL) divergence.

Usually the prior p(z) and the approximate q φ (z|x) are chosen to be in simple form, such as a Gaussian distribution with diagonal covariance, which allows for an analytic calculation of the KL divergence.

While VAE approximates p(x), conditional VAE (Sohn et al., 2015) approximates the conditional distribution p(x|y).

By simply introducing a conditional input, CVAE is trained to maximize the following ELBO: (9) B.1.

Architecture

We construct each module of our model using neural networks and optimize the parameters via backpropagation techniques.

Following the terms in standard VAE, our model is composed of encoders and decoders.

The architecture is shown in Figure 1 with different modalities denoted by different colors.

The data space of unobserved modalities is shaded to differentiate from observed modalities.

The whole architecture can be viewed as an integration of two auto-encoding structures: the top-branch data-wise encoders/decoders and the bottom-branch mask-wise encoders/decoder.

The selective proposal distribution chooses between the unimodal and multimodal encoders if the data is observed or not.

The outputs of all encoders are aggregated and a common latent space is shared among all decoders.

In the rest of this section we explain different modules in the proposed model.

For more details about architecture and implementation see Appendix B.

Selective Factorized Encoders Standard proposal distribution of VAEs depends on the whole data and can not handle incomplete input when the data is partially-observed.

To overcome this, we introduce our selective proposal distribution, which is factorized with respect to the modalities.

As defined in Equation (3), q φ (z i |x i ), named as the unimodal proposal distribution, is inferred only from each observed individual modality of data.

However, if the modality is unobserved, the multimodal proposal distribution q ψ (z i |x o , m) is used to infer corresponding latent variables from other observed modalities and mask.

Hence, the learned model can impute the missing information by combining unimodal proposal distribution of observed modalities and multimodal proposal distribution of the unobserved modalities.

The condition on the mask could make the model aware of the missing pattern and could help the model to attend to observed modalities.

For each modality x i , we have a separate encoder to infer its unimodal proposal distribution parameterized by φ.

For the multimodal proposal distribution, however, we use a single encoder parameterized by ψ.

This encoder outputs the latent codes for all modalities, and we simply obtain the latent variable for each modality by slicing the output vector to M sequential vectors.

We simply model all the proposal distributions as normal distributions by setting the outputs of all encoders as mean and variance of a normal distribution.

For the unimodal proposal distributions, we have

where µ φ and Σ φ are deterministic neural networks parameterized by φ that output the mean and covariance respectively.

Similarly, the multimodal proposal distribution m) ) can be modeled by a neural network with x o and m as the inputs.

The reparameterization in standard VAE is used for end-to-end training.

Decoding through Latent Variable Aggregator F After selecting and sampling from proper proposal distributions for all modalities, the variational latent codes can be fed to the downstream decoders even when the observation is incomplete.

To do this, the information from different modalities interact by aggregating their stochastic latent codes before going through the decoders:

Here we simply choose the aggregator F(·) = concat(·), i.e., concatenating the latent codes as one single vector.

One may also use other aggregation functions such as max/mean pooling or matrix fusion (Veit et al., 2018) to combine latent codes from all modalities.

The decoders take the shared aggregated variational latent codes as input to generate data and mask.

Mask Vector Encoding and Decoding The mask variable m is encoded into the latent space through the multimodal proposal network.

The latent space is shared by the mask and data decoders.

The mask decoder is an MLP parameterized by in our implementation.

It maps the aggregated latent codes from the selective proposal distributions to a reconstruction of the M -dimensional binary mask vector.

We assume each dimension of the mask variable is an independent Bernoulli distribution.

Training With reparameterization trick (Kingma and Welling, 2013), we can jointly optimize the objective derived in Equation Equation (4) with respect to these parameters defined above on training set:

Since Equation (11) only requires the mask and observed data during training, this modified ELBO L φ,θ,ψ, (x o , m) can be optimized without the presence of unobserved modalities.

The KL-divergence term is calculated analytically for each factorized term.

The conditional log-likelihood term is computed by negating reconstruction loss function.

(See Section 3 and Appendix B.3.)

Inference The learned model can be used for both data imputation and generation.

For imputation, the observed modalities x o and mask m are fed through the encoders to infer the selective proposal distributions.

Then the sampled latent codes are decoded to estimate the unobserved modalities x u .

All the modules in Figure 1 are used for imputation.

For generation, since no data is available at all, the latent codes are sampled from the prior and go through the decoders to generate the data and the mask.

In this way, only modules after the aggregator are used without any inference modules.

In all models, all the layers are modeled by MLP without any skip connections or resnet modules.

Basically, the unimodal encoders take single modality data vector as input to infer the unimodal proposal distribution; the multimodal encoders take the observed data vectors and mask vector as as input to infer the multimodal proposal distributions.

The input vector to multimodal encoders should have same length for the neural network.

Here we just concatenate all modality vectors and replace the unobserved modality vectors with some noise.

In UCI repository experiment, we replace the unobserved modality vectors as standard normal noise.

In Bimodal experiment, we simply replace the pixels of unobserved modality as zero.

Note that all the baselines has encoders/decoders with same or larger number of parameters than our method.

We implement our model using PyTorch.

Unimodal Encoders In UCI repository experiment, the unimodal encoders for numerical data are modeled by 3-layer 64-dim MLPs and the unimodal encoders for categorical data are modeled by 3-layer 64-dim MLPs, all followed by Batch Normalization and Leaky ReLU nonlinear activations.

In MNIST+MNIST bimodal experiment, the unimodal encoders are modeled by 3-layer 128-dim MLPs followed by Leaky ReLU nonlinear activations; In MNIST+SVHN bimodal experiment, the unimodal encoders are modeled by 3-layer 512-dim MLPs followed by Leaky ReLU nonlinear activations.

We set the latent dimension as 20-dim for every modality in UCI repository experiments and 256-dim for every modality in Bimodal experiments.

UCI data unimodal encoder: Linear(1, 64)→ BatchNorm1d (64) Multimodal Encoders In general, any model capable of multimodal fusion (Zadeh et al., 2017; Morency et al., 2011) can be used here to map the observed data x o and the mask m to the latent variables z. However, in this paper we simply use an architecture similar to unimodal encoders.

The difference is that the input to unimodal encoders are lower dimensional vectors of an individual modalities.

But, the input to the multimodal encoders is the complete data vector with unobserved modalities replaced with noise or zeros.

As the input to the multimodal encoders is the same for all modalities (i.e., q(z i |x o , m) ∀i), we model the multimodal encoders as one single encoder to take advantage of the parallel matrix calculation speed.

Thus the multimodal encoder for every experiment has the same structure as its unmidal encoder but with full-dimensional input.

Aggregator In our models, we simply use vector concatenation as the way of aggregating.

We use Adam optimizer for all models.

For UCI numerical experiment, learning rate is 1e-3 and use validation set to find a best model in 1000 epochs.

For UCI categorical experiment, learning rate is 1e-2 and use validation set to find a best model in 1000 epochs.

For bimodal experiments, learning rate is 1e-4 and use validation set to find a best model in 1000 epochs.

All modules in our models are trained jointly.

In our model, we calculate the conditional log-likelihood of unobserved modality by generating corresponding modalities from prior.

We initially train the model for some (empirically we choose 20) epochs without calculating the conditional log-likelihood of x u .

And then first feed the partially-observed data to the model and generate the unobserved modalitỹ x u without calculating any loss; then feed the same batch for another pass, calculate the conditional log-likelihood using real x o and generated x u as ground truth.

Table 3 : Imputation on Numerical datasets.

Missing ratio is 0.5.

Last two rows are trained with fully-observed data.

Evaluated by NRMSE, lower is better.

Table 4 : Imputation on MNIST+MNIST.

Missing ratio is 0.3, 0.5 and 0.7.

Last two rows are trained with fully-observed data.

Evaluated by combined errors of two modalities, lower is better.

Table 6 : Imputation on MNIST+SVHN.

Missing ratio is 0.3, 0.5 and 0.7.

Last two rows are trained with fully-observed data.

Evaluated by combined errors of two modalities, lower is better.

@highlight

We propose a novel VAE-based framework learning from partially-observed data for imputation and generation. 