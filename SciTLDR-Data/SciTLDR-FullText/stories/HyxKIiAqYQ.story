We propose a context-adaptive entropy model for use in end-to-end optimized image compression.

Our model exploits two types of contexts, bit-consuming contexts and bit-free contexts, distinguished based upon whether additional bit allocation is required.

Based on these contexts, we allow the model to more accurately estimate the distribution of each latent representation with a more generalized form of the approximation models, which accordingly leads to an enhanced compression performance.

Based on the experimental results, the proposed method outperforms the traditional image codecs, such as BPG and JPEG2000, as well as other previous artificial-neural-network (ANN) based approaches, in terms of the peak signal-to-noise ratio (PSNR) and multi-scale structural similarity (MS-SSIM) index.

The test code is publicly available at https://github.com/JooyoungLeeETRI/CA_Entropy_Model.

Recently, artificial neural networks (ANNs) have been applied in various areas and have achieved a number of breakthroughs resulting from their superior optimization and representation learning performance.

In particular, for various problems that are sufficiently straightforward that they can be solved within a short period of time by hand, a number of ANN-based studies have been conducted and significant progress has been made.

With regard to image compression, however, relatively slow progress has been made owing to its complicated target problems.

A number of works, focusing on the quality enhancement of reconstructed images, were proposed.

For instance, certain approaches BID4 BID17 BID24 have been proposed to reduce artifacts caused by image compression, relying on the superior image restoration capability of an ANN.

Although it is indisputable that artifact reduction is one of the most promising areas exploiting the advantages of ANNs, such approaches can be viewed as a type of post-processing, rather than image compression itself.

Regarding ANN-based image compression, the previous methods can be divided into two types.

First, as a consequence of the recent success of generative models, some image compression approaches targeting the superior perceptual quality BID0 BID16 BID15 have been proposed.

The basic idea here is that learning the distribution of natural images enables a very high compression level without severe perceptual loss by allowing the generation of image components, such as textures, which do not highly affect the structure or the perceptual quality of the reconstructed images.

Although the generated images are very realistic, the acceptability of the machine-created image components eventually becomes somewhat applicationdependent.

Meanwhile, a few end-to-end optimized ANN-based approaches (Toderici et al., 2017; BID1 BID18 , without generative models, have been proposed.

In these approaches, unlike traditional codecs comprising separate tools, such as prediction, transform, and quantization, a comprehensive solution covering all functions has been sought after using end-to-end optimization.

Toderici et al. (2017) 's approach exploits a small number of latent binary representations to contain the compressed information in every step, and each step increasingly stacks the additional latent representations to achieve a progressive improvement in quality of the reconstructed images.

improved the compression performance by enhancing operation methods of the networks developed by Toderici et al. (2017) .

Although Toderici et al. (2017) ; provided novel frameworks suitable to quality control using a single trained network, the increasing number of iteration steps to obtain higher image quality can be a burden to certain applications.

In contrast to the approaches developed by Toderici et al. (2017) and , which extract binary representations with as high an entropy as possible, BID1 , BID18 , and regard the image compression problem as being how to retrieve discrete latent representations having as low an entropy as possible.

In other words, the target problem of the former methods can be viewed as how to include as much information as possible in a fixed number of representations, whereas the latter is simply how to reduce the expected bit-rate when a sufficient number of representations are given, assuming that the low entropy corresponds to small number of bits from the entropy coder.

To solve the second target problem, BID1 , BID18 , and adopt their own entropy models to approximate the actual distributions of the discrete latent representations.

More specifically, BID1 and BID18 proposed novel frameworks that exploit the entropy models, and proved their performance capabilities by comparing the results with those of conventional codecs such as JPEG2000.

Whereas BID1 and BID18 assume that each representation has a fixed distribution, introduced an input-adaptive entropy model that estimates the scale of the distribution for each representation.

This idea is based on the characteristics of natural images in which the scales of the representations vary together in adjacent areas.

They provided test results that outperform all previous ANN-based approaches, and reach very close to those of BPG BID3 , which is known as a subset of HEVC (ISO/IEC 23008-2, ITU-T H.265), used for image compression.

One of the principle elements in end-to-end optimized image compression is the trainable entropy model used for the latent representations.

Because the actual distributions of latent representations are unknown, the entropy models provide the means to estimate the required bits for encoding the latent representations by approximating their distributions.

When an input image x is transformed into a latent representation y and then uniformly quantized intoŷ, the simple entropy model can be represented by pŷ(ŷ), as described by .

When the actual marginal distribution ofŷ is denoted as m(ŷ), the rate estimation, calculated through cross entropy using the entropy model, pŷ(ŷ), can be represented as shown in equation FORMULA0 , and can be decomposed into the actual entropy ofŷ and the additional bits owing to a mismatch between the actual distributions and their approximations.

Therefore, decreasing the rate term R during the training process allows the entropy model pŷ(ŷ) to approximate m(ŷ) as closely as possible, and let the other parameters transform x into y properly such that the actual entropy ofŷ becomes small.

DISPLAYFORM0 In terms of KL-divergence, R is minimized when pŷ(ŷ) becomes perfectly matched with the actual distribution m(ŷ).

This means that the compression performance of the methods essentially depends on the capacity of the entropy model.

To enhance the capacity, we propose a new entropy model that exploits two types of contexts, bit-consuming and bit-free contexts, distinguished according to whether additional bit allocation is required.

Utilizing these two contexts, we allow the model to more accurately estimate the distribution of each latent representation through the use of a more generalized form of the entropy models, and thus more effectively reduce the spatial dependencies among the adjacent latent representations.

FIG0 demonstrates a comparison of the compression results of our method to those of other previous approaches.

The contributions of our work are as follows:• We propose a new context-adaptive entropy model framework that incorporates the two different types of contexts.• We provide the test results that outperform the widely used conventional image codec BPG in terms of the PSNR and MS-SSIM.• We discuss the directions of improvement in the proposed methods in terms of the model capacity and the level of the contexts.

Note that we follow a number of notations given by because our approach can be viewed as an extension of their work, in that we exploit the same rate-distortion (R-D) optimization framework.

The rest of this paper is organized as follows.

In Section 2, we introduce the key approaches of end-to-end optimized image compression and propose the context-adaptive entropy model.

Section 3 demonstrates the structure of the encoder and decoder models used, and the experimental setup and results are then given in section 4.

Finally, in Section 5, we discuss the current state of our work and directions for improvement.

Since they were first proposed by BID1 and BID18 , entropy models, which approximate the distribution of discrete latent representations, have noticeably improved the image compression performance of ANN-based approaches.

BID1 assumes the entropy models of the latent representations as non-parametric models, whereas BID18 adopted a Gaussian scale mixture model composed of six weighted zero-mean Gaussian models per representation.

Although they assume different forms of entropy models, they have a common feature in that both concentrate on learning the distributions of the representations without considering input adaptivity.

In other words, once the entropy models are trained, the trained model parameters for the representations are fixed for any input during the test time.

, in contrast, introduced a novel entropy model that adaptively estimates the scales of the representations based on input.

They assume that the scales of the latent representations from the natural images tend to move together within an adjacent area.

To reduce this redundancy, they use a small amount of additional information by which the proper scale parameters (standard deviations) of the latent representations are estimated.

In addition to the scale estimation, have also shown that when the prior probability density function (PDF) for each representation in a continuous domain is convolved with a standard uniform density function, it approximates the prior probability mass function (PMF) of the discrete latent representation, which is uniformly quantized by rounding, much more closely.

For training, a uniform noise is added to each latent representation so as to fit the distribution of these noisy representations into the mentioned PMF-approximating functions.

Using these approaches, achieved a state-of-the-art compression performance, close to that of BPG.

The latent representations, when transformed through a convolutional neural network, essentially contain spatial dependencies because the same convolutional filters are shared across the spatial regions, and natural images have various factors in common within adjacent regions.

successfully captured these spatial dependencies and enhanced the compression performance by input-adaptively estimating standard deviations of the latent representations.

Taking a step forward, we generalize the form of the estimated distribution by allowing, in addition to the standard deviation, the mean estimation utilizing the contexts.

For instance, assuming that certain representations tend to have similar values within a spatially adjacent area, when all neighborhood representations have a value of 10, we can intuitively guess that, for the current representation, the chance of having a value equal or similar to 10 is relatively high.

This simple estimation will consequently reduce the entropy.

Likewise, our method utilizes the given contexts for estimating the mean, as well as the standard deviation, of each latent representation.

Note that Toderici et al. (2017) , , and BID15 also apply context-adaptive entropy coding by estimating the probability of each binary representation.

However, these context-adaptive entropy-coding methods can be viewed as separate components, rather than one end-to-end optimization component because their probability estimation does not directly contribute to the rate term of the R-D optimization framework.

FIG1 visualizes the latent variablesŷ and their normalized versions of the two different approaches, one estimating only the standard deviation parameters and the other estimating both the mu and standard deviation parameters with the two types of mentioned contexts.

The visualization shows that the spatial dependency can be removed more effectively when the mu is estimated along with the given contexts.

The optimization problem described in this paper is similar with , in that the input x is transformed into y having a low entropy, and the spatial dependencies of y are captured intô z.

Therefore, we also use four fundamental parametric transform functions: an analysis transform g a (x; φ g ) to transform x into a latent representation y, a synthesis transform g s (ŷ; θ g ) to reconstruct imagex, an analysis transform h a (ŷ; φ h ) to capture the spatial redundancies ofŷ into a latent representation z, and a synthesis transform h s (ẑ; θ h ) used to generate the contexts for the model estimation.

Note that h s does not estimate the standard deviations of the representations directly as in 's approach.

In our method, instead, h s generates the context c , one of the two types of contexts for estimating the distribution.

These two types of contexts are described in this section.

analyzed the optimization problem from the viewpoint of the variational autoencoder (Kingma & Welling FORMULA0 ; Rezende et al. FORMULA0 ), and showed that the minimization of the KL-divergence is the same problem as the R-D optimization of image compression.

Basically, we follow the same concept; however, for training, we use the discrete representations on the conditions instead of the noisy representations, and thus the noisy representations are only used as the inputs to the entropy models.

Empirically, we found that using discrete representations on the conditions show better results, as shown in appendix 6.2.

These results might come from removing the mis-matches of the conditions between the training and testing, thereby enhancing the training capacity by limiting the affect of the uniform noise only to help the approximation to the probability mass functions.

We use the gradient overriding method with the identity function, as in BID18 , to deal with the discontinuities from the uniform quantization.

The resulting objective function used in this paper is given in equation (2).

The total loss consists of two terms representing the rates and distortions, and the coefficient λ controls the balance between the rate and distortion during the R-D optimization.

Note that λ is not an optimization target, but a manually configured condition that determines which to focus on between rate and distortion: DISPLAYFORM0 with R = E x∼px Eỹ ,z∼q − log pỹ |ẑ (ỹ |ẑ) − log pz(z) , DISPLAYFORM1 Here, the noisy representations ofỹ andz follow the standard uniform distribution, the mean values of which are y and z, respectively, when y and z are the result of the transforms g a and h a , repectively.

Note that the input to h a isŷ, which is a uniformly quantized representation of y, rather than the noisy representationỹ.

Q denotes the uniform quantization function, for which we simply use a rounding function: DISPLAYFORM2 DISPLAYFORM3 The rate term represents the expected bits calculated using the entropy models of pỹ |ẑ and pz.

Note that pỹ |ẑ and pz are eventually the approximations of pŷ |ẑ and pẑ, respectively.

Equation FORMULA5 represents the entropy model for approximating the required bits forŷ.

The model is based on the Gaussian model, which not only has the standard deviation parameter σ i , but also the mu parameter µ i .

The values of µ i and σ i are estimated from the two types of given contexts based on the function f , the distribution estimator, in a deterministic manner.

The two types of contexts, bit-consuming and bit-free contexts, for estimating the distribution of a certain representation are denoted as c i and c i .

E extracts c i from c , the result of transform h s .

In contrast to c i , no additional bit allocation is required for c i .

Instead, we simply utilize the known (already entropy-coded or decoded) subset ofŷ, denoted as ŷ .

Here, c i is extracted from ŷ by the extractor E .

We assume that the entropy coder and the decoder sequentially processŷ i in the same specific order, such as with raster scanning, and thus ŷ given to the encoder and decoder can always be identical when processing the sameŷ i .

A formal expression of this is as follows: DISPLAYFORM4 DISPLAYFORM5 In the case ofẑ, a simple entropy model is used.

We assumed that the model follows zero-mean Gaussian distributions which have a trainable σ.

Note thatẑ is regarded as side information and it contributes a very small amount of the total bit-rate, as described by , and thus we use this simpler version of the entropy model, rather than a more complex model, for end-to-end optimization over all parameters of the proposed method: Note that actual entropy coding or decoding processes are not necessarily required for training or encoding because the rate term is not the amount of real bits, but an estimation calculated from the entropy models, as mentioned previously.

We calculate the distortion term using the mean squared error (MSE) 1 , assuming that p x|ŷ follows a Gaussian distribution as a widely used distortion metric.

DISPLAYFORM6

This section describes the basic structure of the proposed encoder-decoder model.

On the encoder side, an input image is transformed into latent representations, quantized, and then entropy-coded using the trained entropy models.

In contrast, the decoder first applies entropy decoding with the same entropy models used for the encoder, and reconstructs the image from the latent representations, as illustrated in FIG2 .

It is assumed that all parameters that appear in this section were already trained.

The structure of the encoder-decoder model basically includes g a and g s in charge of the transform of x into y and its inverse transform, respectively.

The transformed y is uniformly quantized intoŷ by rounding.

Note that, in the case of approaches based on the entropy models, unlike traditional codecs, tuning the quantization steps is usually unnecessary because the scales of the representations are optimized together through training.

The other components between g a and g s carry out the role of entropy coding (or decoding) with the shared entropy models and underlying context preparation processes.

More specifically, the entropy model estimates the distribution of eachŷ i individually, in which µ i and σ i are estimated with the two types of given contexts, c i and c i .

Among these contexts, c can be viewed as side information, which requires an additional bit allocation.

To reduce the required bit-rate for carrying c , the latent representation z, transformed fromŷ, is quantized and entropy-coded by its own entropy model, as specified in section 2.3.

On the other hand, c i is extracted from ŷ , without any additional bit allocation.

Note that ŷ varies as the entropy coding or decoding progresses, but is always identical for processing the sameŷ i in both the encoder and decoder, as described in 2.3.

The parameters of h s and the entropy models are simply shared by both the encoder and the decoder.

Note that the inputs to the entropy models during training are the noisy representations, as illustrated with the dotted line in FIG2 , to allow the entropy model to approximate the probability mass functions of the discrete representations.

We basically use the convolutional autoencoder structure, and the distribution estimator f is also implemented using convolutional neural networks.

The notations of the convolutional layer follow : the number of filters × filter height × filter width / the downscale or upscale factor, where ↑ and ↓ denote the up and downscaling, respectively.

For up or downscaling, we use the transposed convolution.

For the networks, input images are normalized into a scale between -1 and 1.

We use a convolutional neural networks to implement the analysis transform and the synthesis transform functions, g a , g s , h a , and h s .

The structures of the implemented networks follow the same structures of , except that we use the exponentiation operator instead of an absolute operator at the end of h s .

Based on 's structure, we added the components to estimate the distribution of eachŷ i , as shown in FIG3 .

Herein, we represent a uniform quantization (round) as "Q," entropy coding as "EC," and entropy decoding as "ED."

The distribution estimator is denoted as f , and is also implemented using the convolutional layers which takes channel-wise concatenated c i and c i as inputs and provides estimated µ i and σ i as results.

Note that the same c i and c i are shared for allŷ i s located at the same spatial position.

In other words, we let E extract all spatially adjacent elements from c across the channels to retrieve c i and likewise let E extract all adjacent known elements from ŷ for c i .

This could have the effect of capturing the remaining correlations among the different channels.

In short, when M is the total number of channels of y, we let f estimate all M distributions ofŷ i s, which are located at the same spatial position, using only a single step, thereby allowing the total number of estimations to be reduced.

Furthermore, the parameters of f are shared for all spatial positions ofŷ, and thus only one trained f per λ is necessary to process any sized images.

In the case of training, however, collecting the results from the all spatial positions to calculate the rate term becomes a significant burden, despite the simplifications mentioned above.

To reduce this burden, we designate a certain number (32 and 16 for the base model and the hybrid model, respectively) of random spatial points as the representatives per training step, to calculate the rate term easily.

Note that we let these random points contribute solely to the rate term, whereas the distortion is still calculated over all of the images.

Because y is a three-dimensional array in our implementation, index i can be represented as three indexes, k, l, and m, representing the horizontal index, the vertical index, and the channel index, respectively.

When the current position is given as (k, l, m), E extracts c To keep the dimensions of the estimation results to the inputs, the marginal areas of c and ŷ are also set to zeros.

Note that when training or encoding, c i can be extracted using simple 4×4×M windows and binary masks, thereby enabling parallel processing, whereas a sequential reconstruction is inevitable for decoding.

Another implementation technique used to reduce the implementation cost is combining the lightweight entropy model with the proposed model.

The lightweight entropy model assumes that the representations follow a zero-mean Gaussian model with the estimated standard deviations, which is very similar with Ballé et al. FORMULA0 's approach.

We utilize this hybrid approach for the top four cases, in bit-rate descending order, of the nine λ configurations, based on the assumption that for the higher-quality compression, the number of sparse representations having a very low spatial dependency increases, and thus a direct scale estimation provides sufficient performance for these added representations.

For implementation, we separate the latent representation y into two parts, y 1 and y 2 , and two different entropy models are applied for them.

Note that the parameters of g a , g s , h a , and h s are shared, and all parameters are still trained together.

The detailed structure and experimental settings are described in appendix 6.1.The number of parameters N and M are set to 128 and 192, respectively, for the five λ configurations for lower bit-rates, whereas 2-3 times more parameters, described in appendix 6.1, are used for the four λ configurations for higher bit-rates.

Tensorflow and Python were used to setup the overall network structures, and for the actual entropy coding and decoding using the estimated model parameters, we implemented an arithmetic coder and decoder, for which the source code of the "Reference arithmetic coding" project 2 was used as the base code.

We optimized the networks using two different types of distortion terms, one with MSE and the other with MS-SSIM.

For each distortion type, the average bits per pixel (BPP) and the distortion, PSNR and MS-SSIM, over the test set are measured for each of the nine λ configurations.

Therefore, a total of 18 networks are trained and evaluated within the experimental environments, as explained below:• For training, we used 256×256 patches extracted from 32,420 randomly selected YFCC100m BID19 ) images.

We extracted one patch per image, and the extracted regions were randomly chosen.

Each batch consists of eight images, and 1M iterations of the training steps were conducted, applying the ADAM optimizer (Kingma & Ba FORMULA0 ).

We set the initial learning rate to 5×10 − 5, and reduced the rate by half every 50,000 iterations for the last 200,000 iterations.

Note that, in the case of the four λ configurations for high bpp, in which the hybrid entropy model is used, 1M iterations of pre-training steps were conducted using the learning rate of 1×10 − 5.

Although we previously indicated that the total loss is the sum of R and λD for a simple explanation, we tuned the balancing parameter λ in a similar way as BID18 , as indicated in equation FORMULA8 .

We used the λ parameters ranging from 0.01 to 0.5.

DISPLAYFORM0 • For the evaluation, we measured the average BPP and average quality of the reconstructed images in terms of the PSNR and MS-SSIM over 24 PNG images of the Kodak PhotoCD image dataset BID10 .

Note that we represent the MS-SSIM results in the form of decibels, as in , to increase the discrimination.

We compared the test results with other previous methods, including traditional codecs such as BPG and JPEG2000, as well as previous ANN-based approaches such as BID18 and BalléFigure 5: Rate-distortion curves of the proposed method and competitive methods.

The top plot represents the PSNR values as a result of changes in bpp, whereas the bottom plot shows MS-SSIM values in the same manner.

Note that MS-SSIM values are converted into decibels(−10 log 10 (1 − MS-SSIM)) for differentiating the quality levels, in the same manner as in .

Because two different quality metrics are used, the results are presented with two separate plots.

As shown in figure 5 , our methods outperform all other previous methods in both metrics.

In particular, our models not only outperform 's method, which is believed to be a state-of-the-art ANN-based approach, but we also obtain better results than the widely used conventional image codec, BPG.More specifically, the compression gains in terms of the BD-rate of PSNR over JPEG2000, 's approach (MSE-optimized), and BPG are 34.08%, 11.97%, and 6.85%, respectively.

In the case of MS-SSIM, we found wider gaps of 68.82%, 13.93%, and 49.68%, respectively.

Note that we achieved significant gains over traditional codecs in terms of MS-SSIM, although this might be because the dominant target metric of the traditional codec developments have been the PSNR.In other words, they can be viewed as a type of MSE-optimized codec.

Even when setting aside the case of MS-SSIM, our results can be viewed as one concrete evidence supporting that ANN-based image compression can outperform the existing traditional image codecs in terms of the compression performance.

Supplemental image samples are provided in appendix 6.3.

Based on previous ANN-based image compression approaches utilizing entropy models BID1 BID18 , we extended the entropy model to exploit two different types of contexts.

These contexts allow the entropy models to more accurately estimate the distribution of the representations with a more generalized form having both mean and standard deviation parameters.

Based on the evaluation results, we showed the superiority of the proposed method.

The contexts we utilized are divided into two types.

One is a sort of free context, containing the part of the latent variables known to both the encoder and the decoder, whereas the other is the context, which requires additional bit allocation.

Because the former is a generally used context in a variety of codecs, and the latter was already verified to help compression using 's approach, our contributions are not the contexts themselves, but can be viewed as providing a framework of entropy models utilizing these contexts.

Although the experiments showed the best results in the ANN-based image compression domain, we still have various studies to conduct to further improve the performance.

One possible way is generalizing the distribution models underlying the entropy model.

Although we enhanced the performance by generalizing the previous entropy models, and have achieved quite acceptable results, the Gaussian-based entropy models apparently have a limited expression power.

If more elaborate models, such as the non-parametric models of or BID11 , are combined with the context-adaptivity proposed in this paper, they would provide better results by reducing the mismatch between the actual distributions and the approximation models.

Another possible way is improving the level of the contexts.

Currently, our methods only use low-level representations within very limited adjacent areas.

However, if the sufficient capacity of the networks and higher-level contexts are given, a much more accurate estimation could be possible.

For instance, if an entropy model understands the structures of human faces, in that they usually have two eyes, between which a symmetry exists, the entropy model could approximate the distributions more accurately when encoding the second eye of a human face by referencing the shape and position of the first given eye.

As is widely known, various generative models BID5 BID13 BID25 learn the distribution p(x) of the images within a specific domain, such as human faces or bedrooms.

In addition, various in-painting methods BID12 BID22 BID23 learn the conditional distribution p(x | context) when the viewed areas are given as context.

Although these methods have not been developed for image compression, hopefully such high-level understandings can be utilized sooner or later.

Furthermore, the contexts carried using side information can also be extended to some high-level information such as segmentation maps or any other information that helps with compression.

Segmentation maps, for instance, may be able to help the entropy models estimate the distribution of a representation discriminatively according to the segment class the representation belongs to.

Traditional codecs have a long development history, and a vast number of hand-crafted heuristics have been stacked thus far, not only for enhancing compression performance, but also for compromising computational complexities.

Therefore, ANN-based image compression approaches may not provide satisfactory solutions as of yet, when taking their high complexity into account.

However, considering its much shorter history, we believe that ANN-based image compression has much more potential and possibility in terms of future extension.

Although we remain a long way from completion, we hope the proposed context-adaptive entropy model will provide an useful contribution to this area.

The structure of the hybrid network for higher bit-rate environments.

The same notations as in the figure 4 are used.

The representation y is divided into two parts and quantized.

One of the resulting parts,ŷ 1 , is encoded using the proposed model, whereas the other,ŷ 2 , is encoded using a simpler model in which only the standard deviations are estimated using side information.

The detailed structure of the proposed model is illustrated in FIG3 .

All concatenation and split operations are performed in a channel-wise manner.

We combined the lightweight entropy model with the context-adaptive entropy model to reduce the implementation costs for high-bpp configurations.

The lightweight model exploits the scale (standard deviation) estimation, assuming that the PMF approximations of the quantized representations follow zero-mean Gaussian distributions convolved with a standard uniform distribution.

FIG5 illustrates the network structure of this hybrid network.

The representation y is split channel-wise into two parts, y 1 and y 2 , which have M 1 and M 2 channels, respectively, and is then quantized.

Here,ŷ 1 is entropy coded using the proposed entropy model, whereasŷ 2 is coded with the lightweight entropy model.

The standard deviations ofŷ 2 are estimated using h a and h s .

Unlike the context-adaptive entropy model, which uses the results of h a (ĉ ) as the input source to the estimator f , the lightweight entropy model retrieves the estimated standard deviations from h a directly.

Note that h a takes the concatenatedŷ 1 andŷ 2 as input, and h s generatesĉ as well as σ 2 , at the same time.

The total loss function also consists of the rate and distortion terms, although the rate is divided into three parts, each of which is forŷ 1 ,ŷ 2 , andẑ, respectively.

The distortion term is the same as before, but note thatŷ is the channel-wise concatenated representation ofŷ 1 andŷ 2 : DISPLAYFORM0 with R = E x∼px Eỹ 1 ,ỹ2,z∼q − log pỹ 1|ẑ (ỹ 1 |ẑ) − log pỹ 2 |ẑ (ỹ 2 |ẑ) − log pz(z) , DISPLAYFORM1 Here, the noisy representations ofỹ 1 ,ỹ 2 , andz follow a standard uniform distribution, the mean values of which are y 1 , y 2 , and z, respectively.

In addition, y 1 and y 2 are channel-wise split representations from y, the results of the transform g a , and have M 1 and M 2 channels, respectively: DISPLAYFORM2 with y 1 , y 2 = S(g a (x; φ g )),ŷ = Q(y 1 ) ⊕ Q(y 2 ), z = h a (ŷ; φ h ).The rate term forŷ 1 is the same model as that of equation FORMULA5 .

Note thatσ 2 does not contribute here, but does contribute to the model forŷ 2 : DISPLAYFORM3 DISPLAYFORM4 The rate term forŷ 2 is almost the same as , except that noisy representations are only used as the inputs to the entropy models for training, and not for the conditions of the models.

DISPLAYFORM5 The model of z is the same as in equation FORMULA7 .

For implementation, we used this hybrid structure for the top-four configurations in bit-rate descending order.

We set N , M 1 , and M 2 to 400, 192, and 408, respectively, for the top-two configurations, and to 320, 192, and 228, respectively, for the next two configurations.

In addition, we measured average execution time per image, spent for encoding and decoding Kodak PhotoCD image dataset BID10 , to clarify benefit of the hybrid model.

The test was conducted under CPU environments, Intel i9-7900X.

Note that we ignored time for actual entropy coding because all models with the same values of N and M spend the same amount of time for entropy coding.

As shown in figure 7 , the hybrid models clearly reduced execution time of the models.

Setting N and M to 320 and 420, respectively, we obtained 46.83% of speed gain.

With the higher number of parameters, 400 of N and 600 of M , we obtained 57.28% of speed gain.

In this section, we provide test results of the two models, the proposed model trained using discrete representations as inputs to the synthesis transforms, g s and h s , and the same model but trained using noisy representations following the training process of 's approach.

In detail, in training phase of the proposed model, we used quantized representationsŷ andẑ as inputs to the transforms g s and h s , respectively, to ensure the same conditions of training and testing phases.

On the other hand, for training the compared model, representationsỹ andz are used as inputs to the transforms.

An additional change of the proposed model is usingŷ, instead of y, as inputs to h a , but note that this has nothing to do with the mismatches between training and testing.

We used them to match inputs to h a to targets of model estimation via f .

As shown in figure 8 , the proposed model, trained using discrete representations, was 5.94% better than the model trained using noisy representations, in terms of the BD-rate of PSNR.

Compared with 's approach, the performance gains of the two models, trained using discrete representations and noisy representations, were 11.97% and 7.20%, respectively.

@highlight

Context-adaptive entropy model for use in end-to-end optimized image compression, which significantly improves compression performance