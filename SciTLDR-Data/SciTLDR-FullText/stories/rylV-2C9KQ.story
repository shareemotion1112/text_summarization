Deep neural networks, in particular convolutional neural networks, have become highly effective tools for compressing images and solving inverse problems including denoising, inpainting, and reconstruction from few and noisy measurements.

This success can be attributed in part to their ability to represent and generate natural images well.

Contrary to classical tools such as wavelets, image-generating deep neural networks have a large number of parameters---typically a multiple of their output dimension---and need to be trained on large datasets.

In this paper, we propose an untrained simple image model, called the deep decoder, which is a deep neural network that can generate natural images from very few weight parameters.

The deep decoder has a simple architecture with no convolutions and fewer weight parameters than the output dimensionality.

This underparameterization enables the deep decoder to compress images into a concise set of network weights, which we show is on par with wavelet-based thresholding.

Further, underparameterization provides a barrier to overfitting, allowing the deep decoder to have state-of-the-art performance for denoising.

The deep decoder is simple in the sense that each layer has an identical structure that consists of only one upsampling unit, pixel-wise linear combination of channels, ReLU activation, and channelwise normalization.

This simplicity makes the network amenable to theoretical analysis, and it sheds light on the aspects of neural networks that enable them to form effective signal representations.

Data models are central for signal and image processing and play a key role in compression and inverse problems such as denoising, super-resolution, and compressive sensing.

These data models impose structural assumptions on the signal or image, which are traditionally based on expert knowledge.

For example, imposing the assumption that an image can be represented with few non-zero wavelet coefficients enables modern (lossy) image compression BID1 and efficient denoising BID6 .In recent years, it has been demonstrated that for a wide range of imaging problems, from compression to denoising, deep neural networks trained on large datasets can often outperform methods based on traditional image models BID19 BID0 BID18 BID4 BID22 .

This success can largely be attributed to the ability of deep networks to represent realistic images when trained on large datasets.

Examples include learned representations via autoencoders BID12 and generative adversarial models BID8 .

Almost exclusively, three common features of the recent success stories of using deep neural network for imaging related tasks are i) that the corresponding networks are over-parameterized (i.e., they have much more parameters than the dimension of the image that they represent or generate), ii) that the networks have a convolutional structure, and perhaps most importantly, iii) that the networks are trained on large datasets.

An important exception that breaks with the latter feature is a recent work by Ulyanov et al. BID20 , which provides an algorithm, called the deep image prior (DIP), based on deep neural networks, that can solve inverse problems well without any training.

Specifically, Ulyanov et al. demonstrated that fitting the weights of an over-parameterized deep convolutional network to a single image, together with strong regularization by early stopping of the optimization, performs competitively on a variety of image restoration problems.

This result is surprising because it does not involve a training dataset, which means that the notion of what makes an image 'natural' is contained in a combination of the network structure and the regularization.

However, without regularization the proposed network has sufficient capacity to overfit to noise, preventing meaningful image denoising.

These prior works demonstrating the effectiveness of deep neural networks for image generation beg the question whether there may be a deep neural network model of natural images that is underparameterized and whose architecture alone, without algorithmic assistance, forms an efficient model for natural images.

In this paper, we propose a simple image model in the form of a deep neural network that can represent natural images well while using very few parameters.

This model thus enables image compression, denoising, and solving a variety of inverse problems with close to or state of the art performance.

We call the network the deep decoder, due to its resemblance to the decoder part of an autoencoder.

The network does not require training, and contrary to previous approaches, the network itself incorporates all assumptions on the data, is under-parameterized, does not involve convolutions, and has a simplicity that makes it amenable to theoretical analysis.

The key contributions of this paper are as follows:??? The network is under-parameterized.

Thus, the network maps a lower-dimensional space to a higher-dimensional space, similar to classical image representations such as sparse wavelet representations.

This feature enables image compression by storing the coefficients of the network after its weights are optimized to fit a single image.

In Section 2, we demonstrate that the compression is on-par with wavelet thresholding BID1 , a strong baseline that underlies JPEG-2000.

An additional benefit of underparameterization is that it provides a barrier to overfitting, which enables regularization of inverse problems.??? The network itself acts as a natural data model.

Not only does the network require no training (just as the DIP BID20 ); it also does not critically rely on regularization, for example by early stopping (in contrast to the DIP).

The property of not involving learning has at least two benefits: The same network and code is usable for a number of applications, and the method is not sensitive to a potential misfit of training and test data.??? The network does not use convolutions.

Instead, the network does have pixelwise linear combinations of channels, and, just like in a convolutional neural network, the weights are shared among spatial positions.

Nonetheless, these are not convolutions because they provide no spatial coupling between pixels, despite how pixelwise linear combinations are sometimes called '1x1 convolutions.'

In contrast, the majority of the networks for image compression, restoration, and recovery have convolutional layers with filters of nontrivial spatial extent BID19 ; BID0 ; BID18 ; BID4 BID22 .

This work shows that relationships characteristic of nearby pixels of natural images can be imposed directly by upsampling layers.??? The network only consists of a simple combination of few building blocks, which makes it amenable to analysis and theory.

For example, we prove that the deep decoder can only fit a small proportion of noise, which, combined with the empirical observation that it can represent natural images well, explains its denoising performance.

The remainder of the paper is organized as follows.

In Section 2, we first demonstrate that the deep decoder enables concise image representations.

We formally introduce the deep decoder in Section 3.

In Section 4, we show the performance of the deep decoder on a number of inverse problems such as denoising.

In Section 5 we discuss related work, and finally, in Section 6 we provide theory and explanations on what makes the deep decoder work.

Intuitively, a model describes a class of signals well if it is able to represent or approximate a member of the class with few parameters.

In this section, we demonstrate that the deep decoder, an untrained, non-convolutional neural network, defined in the next section, enables concise representation of an image-on par with state of the art wavelet thresholding.

The deep decoder is a deep image model G : R N ??? R n , where N is the number of parameters of the model, and n is the output dimension, which is (much) larger than the number of parameters (n N ).

The parameters of the model, which we denote by C, are the weights of the network, and not the input of the network, which we will keep fixed.

To demonstrate that the deep decoder enables concise image representations, we choose the number of parameters of the deep decoder, N , such that it is a small fraction of the output dimension of the deep decoder, i.e., the dimension of the images 1 .We draw 100 images from the ImageNet validation set uniformly at random and crop the center to obtain a 512x512 pixel color image.

For each image x * , we fit a deep decoder model G(C) by minimizing the loss DISPLAYFORM0 with respect to the network parameters C using the Adam optimizer.

We then compute for each image the corresponding peak-signal-to-noise ratio, defined as 10 log 10 (1/MSE), where DISPLAYFORM1 is the image generated by the network, and x * is the original image.

We compare the compression performance to wavelet compression BID1 by representing each image with the N -largest wavelet coefficients.

Wavelets-which underly JPEG 2000, a standard for image compression-are one of the best methods to approximate images with few coefficients.

In Fig. 1 we depict the results.

It can be seen that for large compression factors (3 ?? 512 2 /N = 32.3), the representation by the deep decoder is slightly better for most images (i.e., is above the red line), while for smalle compression factors (3 ?? 512 2 /N = 8), the wavelet representation is slightly better.

This experiment shows that deep neural networks can represent natural images well with very few parameters and without any learning.

The observation that, for small compression factors, wavelets enable more concise representations than the deep decoder is intuitive because any image can be represented exactly with sufficiently many wavelet coefficients.

In contrast, there is no reason to believe a priori that the deep decoder has zero representation error because it is underparameterized.

The main point of this experiment is to demonstrate that the deep decoder is a good image model, which enables applications like solving inverse problems, as in Section 4.

However, it also suggest that the deep decoder can be used for lossy image compression, by quantizing the coefficients C and saving the quantized coefficients.

In the appendix, we show that image representations of the deep decoder are not sensitive to perturbations of its coefficients, thus quantization does not have a detrimental effect on the image quality.

Deep networks were used successfully before for the compression of images BID19 BID0 BID18 .

In contrast to our work, which is capable of compressing images without any learning, the aforementioned works learn an encoder and decoder using convolutional recurrent neural networks BID19 and convolutional autoencoders BID18 based on training data.

We consider a decoder architecture that transforms a randomly chosen and fixed tensor B 0 ??? R n0??k0 consisting of k 0 many n 0 -dimensional channels to an n d ?? k out dimensional image, where k out = 1 for a grayscale image, and k out = 3 for an RGB image with three color channels.

Throughout, n i has two dimensions; for example our default configuration has n 0 = 16 ?? 16 and n d = 512 ?? 512.

The network transforms the tensor B 0 to an image by pixel-wise linearly combining the channels, upsampling operations, applying rectified linear units (ReLUs), and normalizing DISPLAYFORM0 . . .

lin.

comb., upsampling, ReLU, CN lin.

comb.

, sigmoid Figure 1: The deep decoder (depicted on the right) enables concise image representations, onpar with state-of-the-art wavelet based compression.

The crosses on the left depict the PSNRs for 100 randomly chosen ImageNet-images represented with few wavelet coefficients and with a deep decoder with an equal number of parameters.

A cross above the red line means the corresponding image has a smaller representation error when represented with the deep decoder.

The deep decoder is particularly simple, as each layer has the same structure, consisting of a pixel-wise linear combination of channels, upsampling, ReLU nonlinearities, and channelwise normalization (CN).

the channels.

Specifically, the channels in the (i + 1)-th layer are given by DISPLAYFORM1 Here, the coefficient matrices C i ??? R ki??ki+1 contain the weights of the network.

Each column of the tensor B i C i ??? R ni??ki+1 is formed by taking linear combinations of the channels of the tensor B i in a way that is consistent across all pixels.

Then, cn(??) performs a channel normalization operation which is equivalent to normalizing each channel individually, and can be viewed as a special case of the popular batch normalization proposed in BID13 .

Specifically, let Z i = relu(U i B i C i ) be the channels in the i-th layer, and let z ij be the j-th channel in the i-th layer.

Then channel normalization performs the following transformation: DISPLAYFORM2 , where mean and var compute the empirical mean and variance, and ?? ij and ?? ij are parameters, learned independently for each channel, and is a fixed small constant.

Learning the parameter ?? and ?? helps the optimization but is not critical.

This is a special case of batch normalization with batch size one proposed in BID13 , and significantly improves the fitting of the model, just like how batch norm alleviates problems encountered when training deep neural networks.

The operator U i ??? R ni+1??ni is an upsampling tensor, which we choose throughout so that it performs bi-linear upsampling.

For example, if the channels in the input have dimensions n 0 = 16??16, then the upsampling operator U 0 upsamples each channel to dimensions 32 ?? 32.

In the last layer, we do not upsample, which is to say that we choose the corresponding upsampling operator as the identity.

Finally, the output of the d-layer network is formed as DISPLAYFORM3 where Fig. 1 for an illustration.

Throughout, our default architecture is a d = 6 layer network with k i = k for all i, and we focus on output images of dimensions n d = 512 ?? 512 and number of channels k out = 3.

Recall that the parameters of the network are given by C = {C 0 , C 1 , . . .

, C d }, and the output of the network is only a function of C, since we choose the tensor B 0 at random and fix it.

Therefore, we write x = G(C).

Note that the number of parameters is given by DISPLAYFORM4 DISPLAYFORM5 where the term 2k i corresponds to the two free parameters associated with the channel normalization.

Thus, the number of parameters is N = dk 2 + 2dk + 3k.

In the default architectures with d = 6 and k = 64 or k = 128, we have that N = 25,536 (for k = 64) and N =100,224 (k = 128) out of an RGB image space of dimensionality 512 ?? 512 ?? 3 = 786,432 parameters.

We finally note that naturally variations of the deep decoder are possible; for example in a previous version of this manuscript, we applied upsampling after applying the relu-nonlinearity, but found that applying it before yields slightly better results.

While the deep decoder does not use convolutions, its structure is closely related to that of a convolutional neural network.

Specifically, the network does have pixelwise linear combinations of channels, and just like in a convolutional neural network, the weights are shared among spatial positions.

Nonetheless, pixelwise linear combinations are not proper convolutions because they provide no spatial coupling of pixels, despite how they are sometimes called 1 ?? 1 convolutions.

In the deep decoder, the source of spatial coupling is only from upsampling operations.

In contrast, a large number of networks for image compression, restoration, and recovery have convolutional layers with filters of nontrivial spatial extent BID19 ; BID0 ; BID18 ; BID4 ; BID22 .

Thus, it is natural to ask whether using linear combinations as we do, instead of actual convolutions yields better results.

Our simulations indicate that, indeed, linear combinations yield more concise representations of natural images than p ?? p convolutions, albeit not by a huge factor.

Recall that the number of parameters of the deep decoder with d layers, k channels at each layer, and 1 ?? 1 convolutions is N (d, k; 1) = dk 2 + 3k + 2dk.

If we consider a deep decoder with convolutional layers with filters of size p ?? p, then the number of parameters is: DISPLAYFORM0 If we fix the number of channels, k, but increase p to 3, the representation error only decreases since we increase the number of parameters (by a factor of approximately 32 ).

We consider image reconstruction as described in Section 2.

For a meaningful comparison, we keep the number of parameters fixed, and compare the representation error of a deep decoder with p = 1 and k = 64 (the default architecture in our paper) to a variant of the deep decoder with p = 3 and k = 22, so that the number of parameters is essentially the same in both configurations.

We find that the representation of the deep decoder with p = 1 is better (by about 1dB, depending on the image), and thus for concise image representations, linear combinations (1 ?? 1 convolutions) appear to be more effective than convolutions of larger spatial extent.

In this section, we use the deep decoder as a structure-enforcing model or regularizers for solving standard inverse problems: denoising, super-resolution, and inpainting.

In all of those inverse problems, the goal is to recover an image x from a noisy observation y = f (x) + ??.

Here, f is a known forward operator (possibly equal to identity), and ?? is structured or unstructured noise.

We recover the image x with the deep decoder as follows.

Motivated by the finding from the previous section that a natural image x can (approximately) be represented with the deep decoder as G(C), we estimate the unknown image from the noisy observation y by minimizing the loss DISPLAYFORM0 with respect to the model parameters C. Let?? be the result of the optimization procedure.

We estimate the image asx = G(??).We use the Adam optimizer for minimizing the loss, but have obtained comparable results with gradient descent.

Note that this optimization problem is non-convex and we might not reach a global minimum.

Throughout, we consider the least-squares loss (i.e., we take ?? 2 to be the 2 norm), but the loss function can be adapted to account for structure of the noise.

We remark that fitting an image model to observations in order to solve an inverse problem is a standard approach and is not specific to the deep decoder or deep-network-based models in general.

Specifically, a number of classical signal recovery approaches fit into this framework; for example solving a compressive sensing problem with 1 -norm minimization amounts to choosing the forward operator as f (x) = Ax and minimizing over x in a 1 -norm ball.

We start with the perhaps most basic inverse problem, denoising.

The motivation to study denoising is at least threefold: First, denoising is an important problem in practice, second, many inverse problem can be solved as a chain of denoising steps BID15 Figure 2: An application of the deep decoder for denoising the astronaut test image.

The deep decoder has performance on-par with state of the art untrained denoising methods, such as the DIP method BID20 and the BM3D algorithm BID5 .problem is simple to model mathematically, and thus a common entry point for gaining intuition on a new method.

Given a noisy observation y = x + ??, where ?? is additive noise, we estimate an image with the deep decoder by minimizing the least squares loss G(C) ??? y 2 2 , as described above.

The results in Fig. 2 and Table 1 demonstrate that the deep decoder has denoising performance onpar with state of the art untrained denoising methods, such as the related Deep Image Prior (DIP) method BID20 (discussed in more detail later) and the BM3D algorithm BID5 .

Since the deep decoder is an untrained method, we only compared to other state-of-the-art untrained methods (as opposed to learned methods such as BID22 ).Why does the deep decoder denoise well?

In a nutshell, from Section 2 we know that the deep decoder can represent natural images well even when highly underparametrized.

In addition, as a consequence of being under-parameterized, the deep decoder can only represent a small proportion of the noise, as we show analytically in Section 6, and as demonstrated experimentally in FIG2 .

Thus, the deep decoder "filters out" a significant proportion of the noise, and retains most of the signal.

How to choose the parameters of the deep decoder?

The larger k, the larger the number of latent parameters and thus the smaller the representation error, i.e., the error that the deep decoder makes when representing a noise-free image.

On the other hand, the smaller k, the fewer parameters, and the smaller the range space of the deep decoder G(C), and thus the more noise the method will remove.

The optimal k trades off those two errors; larger noise levels require smaller values of k (or some other form of regularization).

If the noise is significantly larger, then the method requires either choosing k smaller, or it requires another means of regularization, for example early stopping of the optimization.

For example k = 64 or 128 performs best out of {32, 64, 128}, for a PSNR of around 20dB, while for a PSNR of about 14dB, k = 32 performs best.

We next super-resolve images with the deep denoiser.

We define a forward model f that performs downsampling with the Lanczos filter by a factor of four.

We then downsample a given image by a factor of four, and then reconstruct it with the deep decoder (with k = 128, as before).

We compare performance to bi-cubic interpolation and to the deep image prior, and find that the deep decoder outperforms bicubic interpolation, and is on-par with the deep image prior (see Table 1 in the appendix).

Finally, we use the deep decoder for inpainting, where we are given an inpainted image y, and a forward model f mapping a clean image to an inpainted image.

The forward model f is defined by a mask that describes the inpainted region, and simply maps that part of the image to zero.

FIG1 and Table 1 demonstrate that the deep decoder performs well on the inpainting problems; however, the deep image prior performs slightly better on average over the examples considered.

For the impainting problem we choose a significantly more expressive prior, specifically k = 320.

Image compression, restoration, and recovery algorithms are either trained or untrained.

Conceptually, the deep decoder image model is most related to untrained methods, such as sparse representations in overcomplete dictionaries (for example wavelets BID6 and curvelets BID17 ).

A number of highly successful image restoration and recovery schemes are not directly based on generative image models, but rely on structural assumptions about the image, such as exploiting self-similarity in images for denoising BID5 and super-resolution BID7 ).Since the deep decoder is an image-generating deep network, it is also related to methods that rely on trained deep image models.

Deep learning based methods are either trained end-to-end for tasks ranging from compression BID19 BID0 BID18 BID4 BID22 to denoising BID4 BID22 , or are based on learning a generative image model (by training an autoencoder or GAN BID12 BID8 ) and then using the resulting model to solve inverse problems such as compressed sensing BID3 , denoising BID11 , phase retrieval , and blind deconvolution BID2 , by minimizing an associated loss.

In contrast to the deep decoder, where the optimization is over the weights of the network, in all the aforementioned methods, the weights are adjusted only during training and then are fixed upon solving the inverse problem.

Most related to our work is the Deep Image Prior (DIP), recently proposed by Ulyanov et al. BID20 .

The deep image prior is an untrained method that uses a network with an hourglass or encoder-decoder architecture, similar to the U-net and related architectures that work well as autoencoders.

The key differences to the deep decoder are threefold: i) the DIP is over-parameterized, whereas the deep decoder is under-parameterized.

ii) Since the DIP is highly over-parameterized, it critically relies on regularization through early stopping and adding noise to its input, whereas the deep decoder does not need to be regularized (however, regularization can enhance performance).

iii) The DIP is a convolutional neural network, whereas the deep decoder is not.

We further illustrate point ii) comparing the DIP and deep decoder by denoising the astronaut image from Fig. 2 .

In FIG2 we plot the Mean Squared Error (MSE) over the number of iterations of the optimizer for fitting the noisy astronaut image x + ??.

Note that to fit the model, we minimize the error G(C) ??? (x + ??) 2 2 , because we are only given the noisy image, but we plot the MSE between the representation and the actual, true image G(C t ) ??? x 2 2 at iteration t. Here, C t are the parameters of the deep decoder after t iterations of the optimizer.

In FIG2 and (c), we plot the loss or MSE associated with fitting the noiseless astronaut image, x ( G(C t ) ??? x 2 2 ) and the noise itself, ??, ( G(C t ) ??? ?? 2 2 ).

Models are fitted independently for the noisy image, the noiseless image, and the noise.

The plots in FIG2 show that with sufficiently many iterations, both the DIP and the DD can fit the image well.

However, even with a large number of iterations, the deep decoder can not fit the noise well, whereas the DIP can.

This is not surprising, given that the DIP is over-parameterized and the deep decoder is under-parameterized.

In fact, in Section 6 we formally show that due to the The third panel shows the MSE of the output of DD or DIP for an image consisting purely of noise, as computed relative to that noise.

Due to under-parameterization, the deep decoder can only fit a small proportion of the noise, and thus enables image denoising.

Early stopping can mildly enhance the performance of DD; to see this note that in panel (a), the minimum is obtained at around 5000 iterations and not at 50,000.

The deep image prior can fit noise very well, but fits an image faster than noise, thus early stopping is critical for denoising performance.underparameterization, the deep decoder can only fit a small proportion of the noise, no matter how and how long we optimize.

As a consequence, it filters out much of the noise when applied to a natural image.

In contrast, the DIP relies on the empirical observation that the DIP fits a structured image faster than it fits noise, and thus critically relies on early stopping.

In the previous sections we empirically showed that the deep decoder can represent images well and at the same time cannot fit noise well.

In this section, we formally show that the deep decoder can only fit a small proportion of the noise, relative to the degree of underparameterization.

In addition, we provide insights into how the components of the deep decoder contribute to representing natural images well, and we provide empirical observations on the sensitivity of the parameters and their distribution.

We start by showing that an under-parameterized deep decoder can only fit a proportion of the noise relative to the degree of underparameterization.

At the heart of our argument is the intuition that a method mapping from a low-to a high-dimensional space can only fit a proportion of the noise relative to the number of free parameters.

For simplicity, we consider a one-layer network, and ignore the batch normalization operation.

Then, the networks output is given by DISPLAYFORM0 Here, we take C = (C 0 , c 1 ), where C 0 is a k ?? k matrix and c 1 is a k-dimensional vector, assuming that the number of output channels is 1.

While for the performance of the deep decoder the choice of upsampling matrix is important, it is not relevant for showing that the deep decoder cannot represent noise well.

Therefore, the following statement makes no assumptions about the upsampling matrix U 0 .

Proposition 1.

Consider a deep decoder with one layer and arbitrary upsampling and input matrices.

That is, let B 0 ??? R n0??k and U 0 ??? R n??n0 .

Let ?? ??? R n be zero-mean Gaussian noise with covariance matrix ??I, ?? > 0.

Assume that k 2 log(n 0 )/n ??? 1/32.

Then, with probability at least DISPLAYFORM1 The proposition asserts that the deep decoder can only fit a small portion of the noise energy, precisely a proportion determined by its number of parameters relative to the output dimension, n. Our The blue curves show a one-dimensional piecewise smooth signal, and the red crosses show estimates of this signal by a one-dimensional deep decoder with either linear or convex upsampling.

We see that linear upsampling acts as an indirect signal prior that promotes piecewise smoothness.simulations and preliminary analytic results suggest that this statement extends to multiple layers in that the lower bound becomes 1 ??? c DISPLAYFORM2 , where c is a numerical constant.

Note that the lower bound does not directly depend on the noise variance ?? since both sides of the inequality scale with ?? 2 .

Upsampling is a vital part of the deep decoder because it is the only way that the notion of locality explicitly enters the signal model.

In contrast, most convolutional neural networks have spatial coupling between pixels both by unlearned upsampling, but also by learned convolutional filters of nontrivial spatial extent.

The choice of the upsampling method in the deep decoder strongly affects the 'character' of the resulting signal estimates.

We now discuss the impacts of a few choices of upsampling matrices U i , and their impact on the images the model can fit.

No upsampling:

If there is no upsampling, or, equivalently, if U i = I, then there is no notion of locality in the resulting image.

All pixels become decoupled, and there is then no notion of which pixels are near to each other.

Specifically, a permutation of the input pixels (the rows of B 0 ) simply induces the identical permutation of the output pixels.

Thus, if a deep decoder without upsampling could fit a given image, it would also be able to fit random permutations of the image equally well, which is practically equivalent to fitting random noise.

Nearest neighbor upsampling: If the upsampling operations perform nearest neighbor upsampling, then the output of the deep decoder consists of piecewise constant patches.

If the upsampling doubles the image dimensions at each layer, this would result in patches of 2 d ?? 2 d pixels that are constant.

While this upsampling method does induce a notion of locality, it does so too strongly in the sense that squares of nearby pixels become identical and incapable of fitting local variation within natural images.

Linear and convex, non-linear upsampling: The specific choice of upsampling matrix affects the multiscale 'character' of the signal estimates.

To illustrate this, FIG3 shows the signal estimate from a 1-dimensional deep decoder with upsampling operations given by linear upsampling (x 0 , x 1 , x 2 , . . .) ??? (x 0 , 0.5x 0 + 0.5x 1 , x 1 , 0.5x 1 + 0.5x 2 , x 2 , . . .) and convex nonlinear upsampling given by (x 0 , x 1 , x 2 , . . .) ??? (x 0 , 0.75x 0 + 0.25x 1 , x 1 , 0.75x 1 + 0.25x 2 , x 2 , . . .).

Note that while both models are able to capture the coarse signal structure, the convex upsampling results in a multiscale fractal-like structure that impedes signal representation.

In contrast, linear upsampling is better able to represent smoothly varying portions of the signal.

Linear upsampling in a deep decoder indirectly encodes the prior that natural signals are piecewise smooth and in some sense have approximately linear behavior at multiple scales 6.3 NETWORK INPUT Throughout, the network input is fixed.

We choose the network input B 1 by choosing its entries uniformly at random.

The particular choice of the input is not very important; it is however desirable that the rows are incoherent.

To see this, as an extreme case, if any two rows of B 1 are equal and if the upsampling operation preserves the values of those pixels exactly (for example, as with the linear upsampling from the previous section), then the corresponding pixels of the output image is Figure 6 : The left panel shows an image reconstruction after training a deep decoder on the MRI phantom image (PSNR is 51dB).

The right panel shows how the deep decoder builds up an image starting from a random input.

From top to bottom are the input to the network and the activation maps (i.e., relu(B i C i )) for eight out of the 64 channels in layers one to six.

Table 1 : Performance comparison of the deep decoder for denoising (DN), superresolution (SR), and inpainting (IP), in peak signal to noise ratio (PSNR).

Note that identity corresponds to the PSNR of the noise and corruption in the DN and IP experiments, respectively.

also exactly the same, which restricts the range space of the deep decoder unrealistically, since for any pair of pixels, the majority of natural images does not have exactly the same value at this pair of pixels.

The deep decoder is tasked with coverting multiple noise channels into a structured signal primarily using pixelwise linear combinations, ReLU activation funcions, and upsampling.

Using these tools, the deep decoder builds up an image through a series of successive approximations that gradually morph between random noise and signal.

To illustrate that, we plot the activation maps (i.e., relu(B i C i )) of a deep decoder fitted to the phantom MRI test image (see Fig. 6 ).

We choose a deep decoder with d = 5 layers and k = 64 channels.

This image reconstruction approach is in contrast to being a semantically meaningful hierarchical representation (i.e., where edges get combined into corners, that get combined into simple sample, and then into more complicated shapes), similar to what is common in discriminative networks.

RH is partially supported by NSF award IIS-1816986, an NVIDIA Academic GPU Grant, and would like to thank Ludwig Schmidt for helpful discussions on the deep decoder in general, and in particular for suggestions on the experiments in Section 2.Code to reproduce the results is available at https://github.com/reinhardh/ supplement_deep_decoder APPENDIX A PROOF OF PROPOSITION 1Suppose that the network has one layer, i.e., G(C) = relu(U 0 B 0 C 0 )c 1 .

We start by re-writing B 1 = relu(B 0 C 0 ) in a convenient form.

For a given vector x ??? R n , denote by diag(x > 0) the matrix that contains one on its diagonal if the respective entry of x is positive and zero otherwise.

Let c jci denote the i-th column of C j , and denote by W ji ??? {0, 1} k??k the corresponding diagonal matrix W ji = diag(U j B j c jci > 0).

With this notation, we can write DISPLAYFORM0 where [c 1 ] i denotes the i-th entry of c 1 .

Thus, G(C) lies in the union of at-most-k 2 -dimensional subspaces of R n , where each subspace is determined by the matrices {W 0j } k j=1 .

The number of those subspaces is bounded by n k 2 .

This follows from the fact that for the matrix A := U 0 B 0 , by Lemma 1 below, the number of different matrices W 0j is bounded by n k .

Since there are k matrices, the number of different sets of matrices is bounded by n k 2 .

Lemma 1.

For any A ??? R n??k and k ??? 5, |{diag(Av > 0)A|v ??? R k }| ??? n k .Next, fix the matrixes {W 0j } j .

As G(C) lies in an at-most-k 2 -dimensional subspace, let S be a k 2 -dimensional subspace that contains the range of G for these fixed {W 0j } j .

It follows that Now, we make use of the following bound on the projection of the noise ?? onto a subspace.

Lemma 2.

Let S ??? R n be a subspace with dimension .

Let ?? ??? N (0, I n ) and ?? ??? 1.

Then, P P S c ?? n , then P X ??? n ??? 2 ??? nx + 2x ??? e ???x , DISPLAYFORM1 With these, we obtain P [X ??? 5??n]

??? e ?????n if ?? ??? 1,P [X ??? n/2] ??? e ???n/16 .We have noiseless C 1 noisy C 2 noisy C 3 noisy C 4 noisy C 5 noisy C 6 noisy c 7 noisy Figure 7 : Sensitivity to parameter perturbations of the weights in each layer, and images generated by perturbing the weights in different layers, and keeping the weights in the other layers constant.

A.1 PROOF OF LEMMA 1Our goal is to count the number of sign patterns (Av > 0) ??? {0, 1}. Note that this number is equal to the maximum number of partitions one can get when cutting a k-dimensional space with n many hyperplanes that all pass through the origin, and are perpendicular to the rows of A. This number if well known (see for example BID21 ) and is upper bounded by DISPLAYFORM2 Thus, DISPLAYFORM3 where the last inequality holds for k ??? 5.

The deep decoder is not overly sensitive to perturbations of its coefficients.

To demonstrate this, fit the standard test image Barbara with a deep decoder with 6 layers and k = 128, as before.

We then perturb the weights in a given layer i (i.e., the matrix C i ) with Gaussian noise of a certain signal-tonoise ratio relative to C i and leave the other weights and the input untouched.

We then measure the peak signal-to-noise ratio in the image domain, and plot the corresponding curve for each layer (see Fig. 7 ).

It can be seen that the representation provided by the deep decoder is relatively stable with respect to perturbations of its coefficients, and that it is more sensitive to perturbations in higher levels.

Finally, in FIG7 we depict the distribution of the weights of the network after fitted to the Barbara test image, and note that the weights are approximately Gaussian distributed.

The distribution of the weighs is approximately Gaussian.

@highlight

We introduce an underparameterized, nonconvolutional, and simple deep neural network that can, without training, effectively represent natural images and solve image processing tasks like compression and denoising competitively.