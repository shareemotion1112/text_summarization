There have been multiple attempts with variational auto-encoders (VAE) to learn powerful global representations of complex data using a combination of latent stochastic variables and an autoregressive model over the dimensions of the data.

However, for the most challenging natural image tasks the purely autoregressive model with stochastic variables still outperform the combined stochastic autoregressive models.

In this paper, we present simple additions to the VAE framework that generalize to natural images by embedding spatial information in the stochastic layers.

We significantly improve the state-of-the-art results on MNIST, OMNIGLOT, CIFAR10 and ImageNet when the feature map parameterization of the stochastic variables are combined with the autoregressive PixelCNN approach.

Interestingly, we also observe close to state-of-the-art results without the autoregressive part.

This opens the possibility for high quality image generation with only one forward-pass.

In representation learning the goal is to learn a posterior latent distribution that explains the observed data well BID0 .

Learning good representations from data can be used for various tasks such as generative modelling and semi-supervised learning (Kingma, 2013; BID14 BID14 BID23 .

The decomposition of variational auto-encoders (VAE) (Kingma, 2013; BID14 provides the potential to disentangle the internal representation of the input data from local to global features through a hierarchy of stochastic latent variables.

This makes the VAE an obvious candidate for learning good representations.

However, in order to make inference tractable VAEs contain simplifying assumptions.

This limits their ability to learn a good posterior latent representation.

In complex data distributions with temporal dependencies (e.g. text, images and audio), the VAE assumption on conditional independence in the input distribution limits the ability to learn local structures.

This has a significant impact on its generative performance, and thereby also the learned representations.

Additionally, the one-layered VAE model with a N (0, I) latent prior poses serious constraints on the posterior complexity that the model is able to learn.

A deep hierarchy of stochastic latent variables should endow the model with more expressiveness, but the VAE has a tendency to skip the learning of the higher representations since they pose a direct cost in its optimization term.

There have been several attempts to eliminate the limitations of the VAE.

Some concern formulating a more expressive variational distribution BID3 BID25 BID30 where other concerns learning a deeper hierarchy of latent variables .

These contributions have resulted in better performance, but are still limited when modelling complex data distributions where a conditional independence does not apply.

When parameterizing the VAE decoder with recurrent neural networks BID17 BID1 BID7 , the decoding architecture gets too powerful which results in unused latent stochastic variables .The limitations of the VAE have spawned interest towards other generative models such as Generative Adversarial Networks (GAN) BID8 and the autoregressive Pixel-CNN/PixelRNN models BID33 .

These methods have proven powerful in learning good generative models, but the lack of stochastic latent variables makes them less suitable for representation learning purposes .

Lately, we have seen several successful attempts to combine VAEs with PixelCNNs BID11 .

This results Figure 1 : A visualization of FAME where the solid lines denote the variational approximation (inference/encoder/recognition) network and dashed lines denote the generative model (decoder) network for training.

When performing reconstructions during training, the input image is concatenated with the output of the generative model (blue) and when generating the model follows a normal autoregressive sampling flow (red) while also using the stochastic latent variables z = z 1 , ..., z L .

Both the variational approximation and the generative model follow a top-down hierarchical structure which enables precision weighted stochastic variables in the variational approximation.in a model where the global structure of the data is learned in the stochastic latent variables of the VAE and the local structure is learned in the PixelCNN.

However, despite the additional complexity and potential extra expressiveness, these models do not outperform a simple autoregressive model BID32 .In this paper we present the Feature Map Variational Auto-Encoder (FAME) that combines the top-down variational approximation presented in the Ladder Variational Auto-Encoder (LVAE) ) with a spatial (feature map) representation of the stochastic latent variables and an autoregressive decoder.

We show that (i) FAME outperforms previously state-of-the-art loglikelihood on MNIST, OMNIGLOT, CIFAR10 and ImageNet, (ii) FAME learns a deep hierarchy of stochastic latent variables without inactivated latent units, (iii) by removing the autoregressive decoder FAME performs close to previous state-of-the-art log-likelihood suggesting that it is possible to get good quality generation with just one forward pass.

The VAE BID14 Kingma, 2013 ) is a generative model with a hierarchy of stochastic latent variables: DISPLAYFORM0 where z = z 1 , ..., z L , ?? denotes the parameters, and L denotes the number of stochastic latent variable layers.

The stochastic latent variables are usually modelled as conditionally independent Gaussian distributions with a diagonal covariance: DISPLAYFORM1 Since the posterior p(z|x) often is intractable we introduce a variational approximation q ?? (z|x) with parameters ??.

In the original VAE formulation q ?? (z|x) is decomposed as a bottom-up inference path through the hierarchy of the stochastic layers: DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 We optimize an evidence lower-bound (ELBO) to the log-likelihood log p ?? (x) = log z p ?? (x, z)dz.

BID2 introduced the importance weighted bound: DISPLAYFORM5 and proved that DISPLAYFORM6 For K = 1 the bound co-incides with the standard ELBO: L(??, ??; x) = L 1 (??, ??; x).

The hierarchical structure of both the variational approximation and generative model give the VAE the expressiveness to learn different representations of the data throughout its stochastic variables, going from local (e.g. edges in images) to global features (e.g. class specific information).

However, we can apply as recursive argument BID22 to show that when optimizing with respect to the parameters ?? and ?? the VAE is regularized DISPLAYFORM7 .

This is evident if we rewrite Equation 6 for K = 1: DISPLAYFORM8 is a local maxima and learning a useful representation in z L can therefore be disregarded throughout the remainder of the training.

The same argumentation can be used for all subsequent layers z 2:L , hence the VAE has a tendency to collapse towards not using the full hierarchy of latent variables.

There are different ways to get around this tendency, where the simplest is to down-weight the KL-divergence with a temperature term BID1 .

This term is applied during the initial phase of optimization and thereby downscales the regularizing effect.

However, this only works for a limited number of hierarchically stacked latent variables .Formulating a deep hierarchical VAE is not the only cause of inactive latent variables, it also occurs when the parameterization of the decoder gets too powerful BID17 BID7 .

This can be caused by using autoregressive models such as p( circumvent this by introducing the Variational Lossy AutoEncoder (VLAE) where they define the architecture for the VAE and autoregressive model such that they capture global and local structures.

They also utilize the power of more expressive posterior approximations using inverse autoregressive flows BID25 BID15 .

In the PixelVAE, BID11 takes a similar approach to defining the generative model but makes a simpler factorizing decomposition in the variational approximation q ?? (z|x) = L i q ?? (z i |x), where the terms have some degree of parameter sharing.

This formulation results in a less flexible model.

DISPLAYFORM9 In BID15 ; BID11 we have seen that VAEs with simple decompositions of the stochastic latent variables and a powerful autoregressive decoder can result in good generative performance and representation learning.

However, despite the additional cost of learning a VAE we only see improvement in the log-likelihood over the PixelCNN for small gray-scale image datasets .

We propose FAME that extends the VAE with a top-down variational approximation similar to the LVAE (S??nderby et al., 2016) combined with spatial stochastic latent layers and an autoregressive decoder, so that we ensure expressive latent stochastic variables learned in a deep hierarchy (cf.

Figure 1 ).

The LVAE does not change the generative model but changes the variational distribution to be top-down like the generative model.

Furthermore the variational distribution shares parameters with the generative model which can be viewed as a precision-weighted (inverse variance) combination of information from the prior and data distribution.

The variational approximation is defined as: DISPLAYFORM0 The stochastic latent variables are all fully factorized Gaussian distributions and are therefore modelled by q ?? (z i |z i+1 , x) = N (z i |?? i , diag(?? separate parameters (as in the VAE), the LVAE let the mean and variance be defined in terms of a function of x (the bottom-up data part) and the generative model (the top-down prior): DISPLAYFORM1 DISPLAYFORM2 where ?? ??,i = ?? ??,i (x) and ?? ??,i = ?? ??,i (z i+1 ) and like-wise for the variance functions.

This precision weighted parameterization has previously yielded excellent results for densely connected networks .

We have seen multiple contributions (e.g. BID11 ) where VAEs (and similar models) have been parameterized with convolutions in the deterministic layers h i j , for j = 1, ..., M , and M is the number of layers connecting the stochastic latent variables z i .

The size of the spatial feature maps decreases towards higher latent representations and transposed convolutions are used in the generative model.

In FAME we propose to extend this notion, so that each of the stochastic latent layers z i , ..., z L???1 are also convolutional.

This gives the model more expressiveness in the latent layers, since it will keep track of the spatial composition of the data (and thereby learn better representations).

The top stochastic layer z L in FAME is a fully-connected dense layer, which makes it simpler to condition on a non-informative N (0, I) prior and sample from a learned generative model p ?? (x, z).

For the i = 1, ..., L ??? 1 stochastic latent variables, the architecture is as follows: DISPLAYFORM0 where CNN and CONV denote a convolutional neural network and convolutional layer respectively.

The top-most latent stochastic layer z L is computed by: DISPLAYFORM1 This new feature map parameterization of the stochastic layers should be viewed as a step towards a better variational model where the test ELBO and the amount of activated stochastic units are direct meaures hereof.

From van den Oord et al. FIG0 ; we have seen that the PixelCNN architecture is very powerful in modelling a conditional distribution between pixels.

In FAME we introduce a PixelCNN in the input dimension of the generative model p ?? (x|z) (cf.

Figure 1) .

During training we concatenate the input with the reconstruction data in the channel dimension and propagate it through the PixelCNN, similarly to what is done in BID11 .

When generating samples we fix a sample from the stochastic latent variables and generate the image pixel by pixel autoregressively.

We test FAME on images from which we can compare with a wide range of generative models.

First we evaluate on gray-scaled image datasets: statically and dynamically binarized MNIST (LeCun et al., 1998) and OMNIGLOT (Lake et al., 2013) .

The OMNIGLOT dataset is of particular interest due to the large variance amongst samples.

Secondly we evaluate our models on natural image datasets: CIFAR10 BID18 Table 2 : Negative log-likelihood performance on dynamically (left) and statically (right) binarized MNIST in nats.

For the dynamically binarized MNIST results show the results for the FAME No Concatenation that has no dependency on the input image.

The evidence lower-bound is computed with 5000 importance weighted samples L 5000 (??, ??; x).modelling the gray-scaled images we assume a Bernoulli B distribution using a Sigmoid activation function as the output and for the natural images we assume a Categorical distribution ?? by applying the 256-way Softmax approach introduced in van den BID33 .

We evaluate the grayscaled images with L 5000 (cf.

Equation 6) and due to runtime and space complexity we evaluate the natural images with L 1000 .We use a hierarchy of 5 stochastic latent variables.

In case of gray-scaled images the stochastic latent layers are dense with sizes 64, 32, 16, 8, 4 (equivalent to S??nderby et al. FORMULA0 ) and for the natural images they are spatial (cf.

Table 1 ).

There was no significant difference when using feature maps (as compared to dense layers) for modelling gray-scaled images.

We apply batchnormalization BID12 and ReLU activation functions as the non-linearity between all hidden layers h i,j and use a simple PixelCNN as in van den BID33 with 4 residual blocks.

Because of the concatenation in the autoregressive decoder (cf.

Figure 1) , generation is a cumbersome process that scales linearly with the amount of pixels in the input image.

Therefore we have defined a slightly changed parameterization denoted FAME No Concatenation, where the concatenation with the input is omitted.

The generation has no dependency on the input data distribution and can therefore be performed in one forward-pass through the generative model.

For optimization we apply the Adam optimizer (Kingma & Ba, 2014 ) with a constant learning rate of 0.0003.

We use 1 importance weighted sample and temperature scaling from .3 to 1.

during the initial 200 epochs for gray-scaled images and .01 to 1.

during the first 400 epochs for natural images.

All models are trained using the same optimization scheme.

The MNIST dataset serves as a good sanity check and has a myriad of previously published generative modelling benchmarks.

We experienced much faster convergence rate on FAME compared to training a regular LVAE.

On the dynamically binarized MNIST dataset we see a sig- Table 1 : The convolutional layer (Conv), filter size (F), depth (K), stride (S), dense layer (Dense) and dimensionality (D) used in defining FAME for gray-scaled and natural images.

The architecture is defined such that we ensure dimensionality reduction throughout the hierarchical stochastic layers.

The autoregressive decoder is a PixelCNN (van den Oord et al., 2016b) with a mask A convolution F=7x7, K=64, S=1 followed by 4 residual blocks of convolutions with mask B, F=3x3, K=64, S=1.Finally there are three non-residual layers of convolutions with mask B where the last is the output layer with a Sigmoid activation for gray-scaled images and a 256-way Softmax for natural images.nificant improvement (cf.

Table 2 ).

However, on the statically binarized MNIST, the parameterization and current optimization strategy was unsuccessful in achieving state-of-the-art results (cf.

Table 1 ).

In FIG1 we see random samples drawn from a N (0, I) distribution and propagated through the decoder parameters ??.

We also trained the FAME No Concatenation which performs nearly on par with the previously state-of-the-art VLAE model ) that in comparison utilizes a skip-connection from the input distribution to the generative decoder: DISPLAYFORM0 .

This proves that a better parameterization of the VAE improves the performance without the need of tedious autoregressive generation.

There was no significant difference in the KL q(z|x)||p(z) between FAME and FAME No Concatenation.

FAME use 10.85 nats in average to encode images, whereas FAME No Concatenation use 12.29 nats.

NLL IWAE (BURDA ET AL., 2015A) 103.38 LVAE (S??NDERBY 102.11 RBM (BURDA ET AL., 2015B) 100.46 DVAE BID26 97.43 DRAW (GREGOR 96.50 CONV DRAW (GREGOR ET AL., 2016) 91.00 VLAE CHEN 89.83 FAME 82.54Figure 3: Negative log-likelihood performance on OMNIGLOT in nats.

The evidence lower-bound is computed with 5000 importance weighted samples L 5000 (??, ??; x).OMNIGLOT consists of 50 alphabets of handwritten characters, where each character has a limited amount of samples.

Each character has high variance which makes it harder to fit a good generative model compared to MNIST.

TAB3 presents the negative log-likelihood of FAME for OMNIGLOT and demonstrates significant improvement over previously published state-of-the-art.

FIG1 shows generated samples from the learned ?? parameter space.

From S??nderby et al. FORMULA0 we have seen that the LVAE is able to learn a much tighter L 1 ELBO compared to the VAE.

For the MNIST experiments, the L 1 ELBO is at 80.11 nats compared to the L 5000 77.82 nats.

Similarly the OMNIGLOT L 1 ELBO is 86.62 nats compared to 82.54 nats.

This shows significant improvements when using importance weighted samples and indicates that the parameterization of the FAME can be done in a way so that the bound is even tighter.

We also find that the top-most latent stochastic layer is not collapsing into its prior, since the KL q(z 5 |x)||p(z 5 ) is 5.04 nats for MNIST and 3.67 nats for OMNIGLOT.

In order to analyze the contribution from the autoregressive decoder we experimented on masking the contribution from either the concatenated image or the output of the FAME decoder before feeding it into the PixelCNN layers (cf.

Figure 1 ).

In FIG2 we see the results of reconstructing MNIST images when masking out the contribution from the stochastic variables and in FIG2 we mask out the contribution from the concatenated input image.

We investigate the performance of FAME on two natural image datasets: CIFAR10 and ImageNet.

Learning a generative model on natural images is more challenging, which is also why there are many tricks that can be done in regards to the autoregressive decoding BID32 .

However, since we are interested in the additional expressiveness of a LVAE parameterization with convolutional stochastic latent variables, we have chosen a suboptimal architecture for the autoregressive decoding (cf.

Table 1 ) BID33 ).

An obvious improvement to the decoder would be to incorporate the PixelCNN++ , but by using the simpler architecture we ensure that the improvements in log-likelihood is not a result of a strong autoregressive model.

From TAB3 we see the performance from FAME and FAME No Concatenation on the CIFAR10 dataset.

Similarly to the gray-scaled images, FAME outperforms current state-of-the-art results sig- Table 4 : Negative log-likelihood performance on ImageNet in bits/dim.

The evidence lower-bound is computed with 1000 importance weighted samples L 1000 (??, ??; x).nificantly.

It is also interesting to see how FAME No Concatenation performs close to the previously published state-of-the-art results.

Especially in the image space, this could prove interesting, since the FAME No Concatenation has no additional autoregressive runtime complexity.

We only investigated the 32x32 ImageNet dataset, since the training time is significant and it outperformed the 64x64 models (cf.

Table 4 ), whereas the previously published 64x64 ImageNet models consistently outperform their 32x32 counterpart.

In FIG0 we show samples from FAME on the CIFAR10 dataset.

Similarly to previously published results it is difficult to analyze the performance from the samples.

However, we can conclude that FAME is able to capture spatial correlations in the images for generating sharp samples.

It is also interesting to see how it captures the contours of objects in the images.

We have presented FAME, an extension to the VAE that significantly improve state-of-the-art performance on standard benchmark datasets.

By introducing feature map representations in the latent stochastic variables in addition to top-down inference we have shown that the model is able to capture representations of complex image distributions while utilizing a powerful autoregressive architecture as a decoder.

In order to analyze the contribution from the VAE as opposed to the autoregressive model, we have presented results without concatenating the input image when reconstructing and generating.

This parameterization shows on par results with the previously state-of-the-art results without depending on the time consuming autoregressive generation.

Further directions for FAME is to (i) test it on larger image datasets with images of a higher resolution, (ii) expand the model to capture other data modalities such as audio and text, (iii) combine the model in a semi-supervised framework.

@highlight

We present a generative model that proves state-of-the-art results on gray-scale and natural images.