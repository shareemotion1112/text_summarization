This paper proposes a dual variational autoencoder (DualVAE), a framework for generating images corresponding to multiclass labels.

Recent research on conditional generative models, such as the Conditional VAE, exhibit image transfer by changing labels.

However, when the dimension of multiclass labels is large, these models cannot change images corresponding to labels, because learning multiple distributions of the corresponding class is necessary to transfer an image.

This leads to the lack of training data.

Therefore, instead of conditioning with labels, we condition with latent vectors that include label information.

DualVAE divides one distribution of the latent space by linear decision boundaries using labels.

Consequently, DualVAE can easily transfer an image by moving a latent vector toward a decision boundary and is robust to the missing values of multiclass labels.

To evaluate our proposed method, we introduce a conditional inception score (CIS) for measuring how much an image changes to the target class.

We evaluate the images transferred by DualVAE using the CIS in CelebA datasets and demonstrate state-of-the-art performance in a multiclass setting.

Recent conditional generative models have shown remarkable success in generating and transferring images.

Specifically, a conditional variational autoencoder (CVAE) BID4 can generate conditional images by learning the latent space Z that corresponds to multiclass labels.

In addition, StarGAN BID1 and FaderNetworks BID5 can generate images corresponding to multiple domains by conditioning with domains such as attributes.

However, when the dimension of the multiclass is increased, these models cannot transfer the images corresponding to one arbitrary domain (an element of a multiclass label).

The possible reasons are the following.

For simplicity, we consider a binary multiclass classification.

To transfer an image of a certain class, it is necessary to learn the distributions of the corresponding class.

That is, assuming that the number of classes in the multiclass is N, conditional models need to create 2 N distributions.

However, when N is large, training is difficult as O(2 N ) training samples will be required.

Hence, instead of conditioning with labels, we propose DualVAE, which conditions with latent vectors that include label information.

DualVAE divides one distribution of the latent space by N linear decision boundaries which need to learn only O(N ) parameters by adding another decoder p w (y|z) to a variational autoencoder (VAE) BID3 .

DualVAE assumes that a label is a linear combination of vectors of the latent space and the dual latent space.

There are two advantages to the DualVAE decoder p w (y|z) being a linear model.

First, DualVAE can easily transfer an image by moving a latent vector toward a decision boundary.

Next, DualVAE is robust to the missing values of multiclass labels.

In addition to this method, we propose the conditional inception score (CIS), a new metric for conditional transferred images.

Although the evaluation methods often used in the generation models are the Inception Score (IS) BID9 and the Fr??chet Inception Distance BID2 , they are used for evaluating the diversity of images and not suitable for evaluating transferred images conditioned with domains such as attributes or classes.

Therefore, we propose a new metric to evaluate two properties: the first property pertains to whether images in one domain are transferred properly to images in another domain; the second property pertains to whether images in one domain Figure 1 : Conditional VAE learns 2 n distributions for each binary multiclass label when the number of class is n. DualVAE learns n decision boundaries for dividing a distribution of latent space.

u 1 is a parameter of a decision boundary, which we call a dual vector.transferred to images in another domain can preserve the original properties.

By using the CIS, we compare DualVAE with other methods that can perform image-to-image translations for multiple domains.

In summary, the contributions from this study are as follows: 1) We introduce DualVAE, a method for transferring images corresponding to multiclass labels and demonstrate that images can be transferred quantitatively and qualitatively.

2) We propose the CIS, a new metric that can evaluate transferred images corresponding to multiclass labels.

Conditional model Several studies have been conducted to generate or transfer images conditioned with labels.

For example, conditional VAE BID4 is an extension of a VAE BID3 where latent variables z are inferred using image x and label y, and image x is reconstructed with y,z.

Further, a CGAN BID7 ) is a conditional model using a GAN, where a noise z and a class label y are input to the generator, and learning is performed similarly to the GAN using image x corresponding to class label y. FaderNetworks BID5 learns latent variables from which label information is eliminated by using adversarial learning and assigns attributes to images by providing labels to the decoder.

Furthermore, StarGAN BID1 , a method of domain transfer, had succeeded in outputting a beautiful image corresponding to an attribute by conditioning with a domain (attribute).

However, all these methods are models conditioned with labels; therefore, as the dimension of the labels becomes larger, the number of training samples becomes insufficient.

Connection to the Information Bottleneck As with DualVAE, there are several papers related to finding a latent variable z that predicts label y. For example, Information Bottleneck (IB) BID11 is a method for obtaining a latent expression z that solves task y. IB is a method which leaves the latent information z for solving the task y by maximizing the mutual information amount I(Z; Y).

At the same time, extra information about input x is discarded by minimizing I(Z; X).

Variational Information Bottleneck (VIB) BID0 succeeded in parameterizing the IB with a neural network, by performing a variational approximation.

VIB can also be considered as a kind of extension of VAE.

VAE minimizes the mutual information I(Z; i) between individual data i and latent variable z while maximizing I(Z; X).

DualVAE can be regarded as a framework of VIB as well, and it minimizes I(Z; i) while maximizing I(Z; Y) and I(Z; X).

We can also regard DualVAE as a probabilistic matrix factorization (PMF) BID8 extended to a generative model.

A PMF is used in several application areas, primarily in collaborative filtering, which is a typical recommendation algorithm.

It can predict missing ratings of users by assuming that the user's ratings are modeled by a linear combination of the item and user latent factors.

Similarly, we experimentally show that DualVAE is also robust to missing labels.

We devised DualVAE by adding a decoder p w (y|z) = p(y|z, u) to the VAE to learn the decision boundaries between classes.

Here, z is a vector of the latent space Z, u is a vector of the dual latent space Z * and y is a label.

Unlike the CVAE, this model does not require label y at the time of inference of z corresponding to x, and the difference is shown in FIG0 .

The objective function of the VAE is as follows: DISPLAYFORM0 where ??, ?? are parameters of the encoder and decoder of the VAE, respectively.

The lower bound of DualVAE is as follows: DISPLAYFORM1 where p w (y|z) = Bern(y|??(Uz)).

Here, U is a domain feature matrix whose row vector is a dual vector u and Bern is a Bernoulli distribution.

As you can see from Equation 2, the objective function of DualVAE is the objective function of the VAE plus the expectation of log-likelihood of p w (y|z) Specifically, training is performed such that the inner product of z j ??? Z and u i ??? Z * predicts the label y ij where j is the index of a sample and i is the index of a domain.

At the same time, we find the values of ?? and ?? that maximize the lower bound in Equation 1.We transfer the images on domain i by performing the following operation.

We calculated the following vector w i : DISPLAYFORM2 where ??(??? R) is a parameter.

Image transfer can be demonstrated by changing ?? and decoding w i .

Equation 3 corresponds to moving a latent vector toward a decision boundary.

Require: images (x j ) m j=1 , batch size M , indicator function I ij VAE/encoder optimizers: g, g e , hyper parameter ??, and the label matrix Y = (y ij ).

Initialize encoder parameter, decoder parameter and dual vector: ??, ??, U = (u i ) DISPLAYFORM0

Although IS BID9 ) is a score for measuring generated images, it can only measure the diversity of the images, and cannot be used for evaluating the domain transfer of the images.

Therefore, we proposed using a CIS, a score for evaluating the transformation of images into multiclass target domains.

The CIS is a scalar value calculated from the sum of two elements.

The first is whether the domain transfer of the original image has been successful (transfer score), and the second is whether the features other than the transferred domain are retained (reconstruction score).

The computation flow of these scores can be found in FIG1 .We calculated the CIS using Algorithm 2.

First, we assumed that the number of domains is n and the domain that each image belonged to was known.

We finetuned Inception-v3 BID10 using train images as inputs and domains as outputs.

To enable the model to classify the images with the domains, we replaced the last layer of the model with a new layer that had n outputs.

Next, we transferred test images into n domain images and loaded the transferred images into the pretrained Inception-v3.

Through this process, we obtained an n ?? n matrix for every original image because one image was transferred into n domain images and each domain image was mapped to an n-dimension vector.

We subsequently mapped the original image into an n-dimension vector using Inception-v3 and subtracted this vector from each row of the n ?? n matrix.

We named this matrix M. The key points are the following: (1) the diagonal elements of M should be large because the specified domain should be changed significantly, and (2) the off-diagonal elements of M should be small because the transferred images should preserve the original features.

DISPLAYFORM0 In the algorithm, abs denotes taking the absolute value, diag denotes taking the diagonal elements of the matrix, notdiag denotes taking the nondiagonal elements, avg denotes taking the mean of the multiclass values.

x is an original image, and x (i) denotes a transferred image on domain i.

We performed a standard image transfer task with the 40 attributes in the CelebA BID6 dataset, which comprises approximately 200,000 images of faces of celebrities.

Comparison of DualVAE and several models DualVAE was compared with several models capable of performing image-to-image translations for multiclass labels using a single model.

In each model, we calculated the CIS several times when applying Algorithm 2 on 160 CelebA test images; subsequently, the average and standard deviation were obtained.

DualVAE obtained a higher CIS than the other models and the results are shown in TAB0 and Figure 4 .

Robustness to sparsity To demonstrate experimentally that DualVAE is robust to the missing values of multiclass labels, the following steps were performed.

We calculated the rs and ts values when applying Algorithm 2 on 160 CelebA test images and plotted the figure below when we changed the missing ratio of CelebA's domain labels and the ?? in Equation 3.

StarGAN (b) s = 0.9.

All identical images were generated, and image transfer was not conducted properly.

As shown in FIG3 , DualVAE is robust in terms of the sparseness of domain labels, and the CIS does not decrease even when 90% of the labels are missing.

Meanwhile, we found that StarGAN is not as robust as DualVAE with respect to sparseness.

When 90% of the domain labels are missing, StarGAN cannot learn at all and generates identical images.

We proposed DualVAE, a simple framework for generating and transferring images corresponding to multiclass labels.

Further, we introduced the CIS, a new metric for measuring how much of an image corresponding to the change of labels could be generated.

The decoder of DualVAE was a simple linear model in this study; however, we would like to test more complex models in the future.

<|TLDR|>

@highlight

 a new framework using dual space for generating images corresponding to multiclass labels when the number of class is large