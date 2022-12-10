Recent findings show that deep generative models can judge out-of-distribution samples as more likely than those drawn from the same distribution as the training data.

In this work, we focus on variational autoencoders (VAEs) and address the problem of misaligned likelihood estimates on image data.

We develop a novel likelihood function that is based not only on the parameters returned by the VAE but also on the features of the data learned in a self-supervised fashion.

In this way, the model additionally captures the semantic information that is disregarded by the usual VAE likelihood function.

We demonstrate the improvements in reliability of the estimates with experiments on the FashionMNIST and MNIST datasets.

Deep Generative Models (DGMs) have gained in popularity due to their ability to model the density of the observed training data from which one can draw novel samples.

However, as Nalisnick et al. (2018) pointed out in their recent paper, the inferences made by likelihood-based models, such as Variational Autoencoders (VAEs) (Kingma and Welling, 2015; Rezende et al., 2014) and flow-based models (Kingma and Dhariwal, 2018; van den Oord et al., 2016) , are not always reliable.

They can judge out-of-distribution (OOD) samples to be more likely than in-distribution (ID) samples that are drawn from the same distribution as the training data.

Concretely, a DGM trained on the FashionMNIST dataset will on average assign higher likelihoods to images from the MNIST dataset than to test images from the FashionMNIST dataset (see for example top left image in Figure 1(a) ).

In this work we tackle the problem of misaligned likelihood estimates produced by VAEs on image data and propose a novel likelihood estimation during test time.

Our method leverages findings reported in our earlier work Bütepage et al. (2019) , which are summarised in Section 2, and is based on the idea to evaluate a given test image not only locally, using individual parameters returned by a VAE as it is usually done, but also globally using learned feature representations of the data.

The main contribution of this paper is the introduction of a feature-based likelihood trained in a self-supervised fashion.

This likelihood evaluates the model also based on the semantics of a given image and not solely on the values of each pixel.

We elaborate on this idea in Section 3 and demonstrate the improvements with an empirical evaluation presented in Section 4.

We emphasise that the aim of our work is exclusively to improve the reliability of the likelihood estimation produced by VAEs.

We focus on image data in particular as we have not observed the misalignment in our earlier experiments on various non-image datasets from UCI Machine Learning Repository (Dua and Graff, 2017) .

We plan to investigate this further in the future work.

Due to the lack of space we omit the experiments on non-image data as well as the specifics of VAEs for which we refer the reader to Kingma and Welling (2015) ; Rezende et al. (2014) .

This section provides a background on the evaluation of VAEs and summarizes our earlier work presented in (Bütepage et al., 2019) .

In VAEs, the observed random variable X is assumed to be generated from the joint distribution p(X, Z) = p(X|Z)p(Z) where Z denotes the latent variables.

Using variational inference the intractable true posterior distribution p * (Z|X) is approximated with a simpler parametrised distribution q(Z|X).

VAEs employ amortized inference where encoder and decoder neural networks, φ z (X) and φ x (Z), are jointly trained to represent the approximate posterior distribution q(Z|φ z (X)) and likelihood function p(X|φ x (Z)), respectively.

From a Bayesian perspective, we can evaluate a successfully trained VAE using two different evaluation schemes

where p P R V AE denotes the prior predictive (PR) and p AP O V AE the approximate posterior predictive (APO) distribution.

Bütepage et al. (2019) argue that the likelihood estimates produced by a trained VAE are influenced by both 1) the choice of the above listed evaluation scheme and 2) the choice of the parametrisation of the likelihood function p(X|φ x (Z)).

Here, two common choices are a Gaussian distribution in the case of colored images or a Bernoulli distribution in the case of black and white (or grey-scaled) images.

The effect of both 1) and 2) is best demonstrated in Figure 1 (a) where we visualise the log likelihood estimates from a VAE V 1 , parametrised by a Bernoulli likelihood (top row), and a VAE V 2 , parametrised by a Gaussian likelihood (bottom row), using both PR (left column) and APO (right column) evaluation schemes from Equations (1) and (2).

Both VAEs were trained on the FashionMNIST dataset and tested on test images from both the FashionMNIST and MNIST datasets.

In the case of V 1 the pixel values of the images were binarised with threshold 0.5, and in the case of V 2 scaled to the interval [0, 1] .

The choice of the evaluation scheme influences the variance of the estimates of the training data as it directly affects the variability of the parameters φ x (z) returned by the VAE (see left vs right column in Figure 1(a) ).

Namely, PR produces more diverse parameters corresponding to the latent representations of the whole training data while APO generates more homogeneous samples corresponding to the latent representation of a given test point x. On the other hand, the choice of the likelihood parametrisation (top vs bottom row in Figure 1(a) ) influences the actual values of the estimates since images are evaluated under distributions of different shapes.

We refer the interested reader to (Bütepage et al., 2019) for a detailed discussion.

Note that only the top-left combination in Figure 1(a) reproduces the results reported in (Nalisnick et al., 2018) .

(b) Using the improved p F EV AE (x|φ x (z)) likelihood from Equation (4).

This section describes the self-supervised feature-based likelihood function which is the main contribution of this work.

In addition to the influencing factors discussed in Section 2, we hypothesise that the likelihood estimates are also affected by the assumption that image pixels are independent and identically distributed (iid) around the likelihood function parameterised by the decoder.

Let a test image x be represented as a concatenated vector of length D and let x d denote its d-th component.

Using the assumption of iid pixels, the likelihood function becomes a product of individual pixel-wise likelihoods:

Therefore, when computing the probability of x, the likelihood only captures pixel-wise errors that are evaluated locally under the parameters φ x (z) returned by the VAE and does not take into account the "global" information contained in the image (such as the semantics of the dataset).

To mitigate the lack of the global evaluation, we propose to weight the likelihood term during test time with an additional term that relates the semantic information of both the test point x and the parameters φ x (z) to the semantics of the whole training dataset.

We define the details below.

We separately train a self-supervised classifier Γ and use its l-th layer to extract a low dimensional feature representation f x = l(x) of an image x.

We train Γ on the same training datasetX = {x 1 , . . .

, x N } as we train the VAE.

We then fit a Bayesian Gaussian Mixture (BGM) model with C components to the set F = {f x 1 , . . .

, f xn } of feature representations extracted from a randomly sampled subset ofX of size n < N (see also Section 4 for details).

Let f x be the feature representation of a test image x. During the evaluation of the BGM on f x each mixture component is assigned a weight that indicates its contribution to the generation of f x .

Let C x denote the mixture component with the highest weight.

Given likelihood parameters φ x (z) returned by the VAE, we define the global likelihood of x as the product

where p F E (f x |C φx(z) ) is the likelihood of the test point in feature space under the mixture component C φx(z) determined by the representation f φx(z) of the parameters φ x (z) and

) is the likelihood of f φx(z) under the same component C φx(z) .

The first term can be seen as a global likelihood of the test point under the decoded parameters and the second term represents a global likelihood of the parameters themselves.

We then propose to evaluate the test image x under the combined likelihood function

where p V AE as before captures local pixel-wise errors and p F E additionally captures the global (semantic) likelihood.

We evaluate our method with experiments on FashionMNIST and MNIST datasets and present the results below.

Feature extraction We obtained low dimensional features of the training data by deploying a self-supervised Jigsaw classifier Γ presented by Noroozi and Favaro (2016) .

The classifier receives a Jigsaw puzzle, which is a shuffled 3×3 grid of tiles extracted from a given image, and outputs (the class of) the permutation that was applied to the original unshuffled grid (see Appendix A for the implementation details).

Note that any self-supervised learning strategy could be deployed as long as the obtained low dimensional features are of high quality and represent the training data well.

After the completed training we randomly sampled n = 10000 training images {x 1 , . . .

, x n } and obtained their low dimensional representations {f 1 , . . .

, f n } from the first layer l 1 of the classifier Γ, to which we fitted a BGM model with C = 15 components.

The parameters n and C were determined using a hyperparameter grid search.

We used representations from the first layer because we hypothesise that the earlier layers of the classifier carry useful information about the training data while the later layers carry information about the task itself.

We leave experiments with representations obtained from different layers for the future work.

Experiment We trained two VAEs, V 1 and V 2 , and two Jigsaw classifiers, Γ 1 and Γ 2 with specifications described in Appendix A on the FashionMNIST dataset.

Here, the subscripts 1 and 2 denote that the model in consideration was trained on images binarised with threshold 0.5 and on images with pixel values scaled to the interval [0, 1], respectively.

As in the experiment producing the results in Figure 1 (a), V 1 additionally assumes a Bernoulli likelihood and V 2 a Gaussian likelihood.

For a given (binarised) test image x and parameters φ x (z) obtained from the trained VAE V i , we first calculate the VAE likelihood p V AE in the usual way using the assumption of iid pixels.

We then obtain their low dimensional features f x = l 1 i (x) and f φx(z) = l 1 i (φ x (z)) from the first layer l 1 i of the trained Jigsaw classifier Γ i and calculate p F E under the fitted BGM following Equation (3).

The product of the two likelihoods then equals the newly proposed likelihood p F EV AE from Equation (4).

Given this pipeline, VAE i + Γ i for i = 1, 2 and our likelihood p F EV AE , we compared the log likelihood estimates using the PR and APO evaluation schemes from Equations (1) and (2) on the images from the test splits of FashionMNIST and MNIST datasets.

The results are visualised in Figure 1(b) .

We see that our method significantly improves the estimates when using Gaussian likelihood parametrisation (bottom row) as it clearly separates the OOD samples from the ID samples.

Note that the VAE parameters φ x (z) in the PR evaluation always reflect the distribution of the entire training data.

This means that the global likelihood of a test point evaluates the test point under all classes that were presented during training time.

In practice this means that the PR evaluation of the global likelihood averages over all classes which results in a less distinct separation of the OOD samples.

When using Bernoulli likelihood (top row) our method increases the variance of the likelihood of OOD samples but fails to achieve the same separation as in the Gaussian case.

This is because a significant amount of the semantic information is lost during the binarisation process of the FashionMNIST dataset.

The resulting binarised images are often unrecognisable with a sparse pixel distribution which makes the task of solving Jigsaw puzzles more difficult.

Since digits in MNIST images are also sparse they become likely under p F E .

We observe their estimates fusing with FashionMNIST estimates if we corrupt the background using salt and pepper noise (see Figure 2 in Appendix B).

We therefore hypothesise that in this particular case OOD samples simply become too similar to the ID samples, suggesting that the Bernoulli likelihood is not the most appropriate modelling choice.

The inadequacy of the Bernoulli distribution in VAEs has also recently been discussed by Loaiza-Ganem and Cunningham (2019) who instead suggest to use their fully characterized continuous Bernoulli distribution.

We have discussed how the problematic assumption that the image pixels are iid around the decoded parameters narrows the focus of the VAE likelihood function p V AE to a local area of the data density.

Thus, the model likelihood function disregards the global data density, including the semantic information.

Our proposed likelihood function mitigates this problem by leveraging self-supervised feature learning.

In the future, we aim to evaluate our method on more complex datasets, such as CIFAR-10 and SVHN, and to design an end-to-end training procedure of VAEs using our proposed likelihood.

<|TLDR|>

@highlight

Improved likelihood estimates in variational autoencoders using self-supervised feature learning