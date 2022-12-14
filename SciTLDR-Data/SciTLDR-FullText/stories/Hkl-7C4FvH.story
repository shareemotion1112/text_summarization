While generative models have shown great success in generating high-dimensional samples conditional on low-dimensional descriptors (learning e.g. stroke thickness in MNIST, hair color in CelebA, or speaker identity in Wavenet), their generation out-of-sample poses fundamental problems.

The conditional variational autoencoder (CVAE) as a simple conditional generative model does not explicitly relate conditions during training and, hence, has no incentive of learning a compact joint distribution across conditions.

We overcome this limitation by matching their distributions using maximum mean discrepancy (MMD) in the decoder layer that follows the bottleneck.

This introduces a strong regularization both for reconstructing samples within the same condition and for transforming samples across conditions, resulting in much improved generalization.

We refer to the architecture as transformer VAE (trVAE).

Benchmarking trVAE on high-dimensional image and tabular data, we demonstrate higher robustness and higher accuracy than existing approaches.

In particular, we show qualitatively improved predictions for cellular perturbation response to treatment and disease based on high-dimensional single-cell gene expression data, by tackling previously problematic minority classes and multiple conditions.

For generic tasks, we improve Pearson correlations of high-dimensional estimated means and variances with their ground truths from 0.89 to 0.97 and 0.75 to 0.87, respectively.

The task of generating high-dimensional samples x conditional on a latent random vector z and a categorical variable s has established solutions (Mirza & Osindero, 2014; Ren et al., 2016) .

The situation becomes more complicated if the support of z is divided into different domains d with different semantic meanings: say d ∈ {men, women} and one is interested in out-of-sample generation of samples x in a domain and condition (d, s) that is not part of the training data.

If one predicts how a given black-haired man would look with blonde hair, which we refer to as transforming x men, black-hair → x men, blonde-hair , this becomes an out-of-sample problem if the training data does not have instances of blonde-haired men, but merely of blonde-and black-haired woman and blacked haired men.

In an application with higher relevance, there is strong interest in how untreated (s = 0) humans (d = 0) respond to drug treatment (s = 1) based on training data from in vitro (d = 1) and mice (d = 2) experiments.

Hence, the target domain of interest (d = 0) does not offer training data for s = 1, but only for s = 0.

In the present paper, we suggest to address the challenge of transforming out-of-sample by regularizing the joint distribution across the categorical variable s using maximum mean discrepancy (MMD) in the framework of a conditional variational autoencoder (CVAE) (Sohn et al., 2015) .

This produces a more compact representation of a distribution that displays high variance in the vanilla CVAE, which incentivizes learning of features across s and results in more accurate out-of-sample prediction.

MMD has proven successful in a variety of tasks.

In particular, matching distributions with MMD in variational autoencoders (Kingma & Welling, 2013) has been put forward for unsupervised domain adaptation (Louizos et al., 2015) or for learning statistically independent latent dimensions (Lopez et al., 2018b) .

In supervised domain adaptation approaches, MMD-based regularization has been shown to be a viable strategy of learning label-predictive features with domain-specific information removed (Long et al., 2015; Tzeng et al., 2014) .

In further related work, the out-of-sample transformation problem was addressed via hard-coded latent space vector arithmetics (Lotfollahi et al., 2019) and histogram matching (Amodio et al., 2018) .

The approach of the present paper, however, introduce a data-driven end-to-end approach, which does not involve hard-coded elements and generalizes to more than one condition.

In representation learning, one aims to map a vector x to a representation z for which a given downstream task can be performed more efficiently.

Hierarchical Bayesian models (Gelman & Hill, 2006) yield probabilistic representations in the form of sufficient statistics for the model's posterior distribution.

Let {X, S} denote the set of observed random variables and Z the set of latent variables (Z i denotes component i).

Then Bayesian inference aims to maximize the likelihood:

Because the integral is in general intractable, variational inference finds a distribution q φ (Z | X, S) that minimizes a lower bound on the data -the evidence lower bound (ELBO):

In the case of a variational auto-encoder (VAE), the variational distribution is parametrized by a neural network, both the generative model and the variational approximation have conditional distributions parametrized with neural networks.

The difference between the data likelihood and the ELBO is the variational gap:

The original AEVB framework is described in the seminal paper (Kingma & Welling, 2013) for the case Z = {z}, X = {x}, S = ∅. The representation z is optimized to "explain" the data x. The variational distribution can be used to meet different needs: q φ (y | x) is a classifier for a class label y and q φ (z | x) summarizes the data.

When using VAE, the empirical data distribution p data (X, S) is transformed to the representationq φ (Z) = E pdata(X,S) q φ (Z | X, S).

The case in which S = ∅ is referred to as the conditional variational autoencoder (CVAE) (Sohn et al., 2015) , and a straight-forward extension of the original framework.

Let (Ω, F, P) be a probability space.

Let X (resp.

X ) be a separable metric space.

Let x : Ω → X (resp.

x : Ω → X ) be a random variable.

Let k : X × X → R (resp.

k : X × X → R) be a continuous, bounded, positive semi-definite kernel.

Let H be the corresponding reproducing kernel Hilbert space (RKHS) and φ : Ω → H the corresponding feature mapping.

Consider the kernel-based estimate of distance between two distributions p and q over the random variables X and X .

Such a distance, defined via the canonical distance between their H-embeddings, is called the maximum mean discrepancy (Gretton et al., 2012) and denoted MMD(p, q), with an explicit expression:

where the sums run over the number of samples n 0 and n 1 for x and x , respectively.

Asymptotically, for a universal kernel such as the Gaussian kernel k(x, x ) = e

, MMD (X, X ) is 0 if and only if p ≡ q. For the implementation, we use multi-scale RBF kernels defined as:

Figure 1: The transformer VAE (trVAE) is an MMD-regularized CVAE.

It receives randomized batches of data (x) and condition (s) as input during training, stratified for approximately equal proportions of s. In contrast to a standard CVAE, we regularize the effect of s on the representation obtained after the first-layer g 1 (ẑ, s) of the decoder g. During prediction time, we transform batches of the source condition x s=0 to the target condition x s=1 by encodingẑ 0 = f (x 0 , s = 0) and decoding g(ẑ 0 , s = 1).

and γ i is a hyper-parameter.

Addressing the domain adaptation problem, the "Variational Fair Autoencoder" (VFAE) (Louizos et al., 2015) uses MMD to match latent distributions q φ (z|s = 0) and q φ (z|s = 1) -where s denotes a domain -by adapting the standard VAE cost function L VAE according to

where X and X are two high-dimensional observations with their respective conditions S and S .

In contrast to GANs (Goodfellow et al., 2014) , whose training procedure is notoriously hard due to the minmax optimization problem, training models using MMD or Wasserstein distance metrics is comparatively simple Arjovsky et al., 2017; Dziugaite et al., 2015a) as only a direct minimization of a straight forward loss is involved during the training.

It has been shown that MMD based GANs have some advantages over Wasserstein GANs resulting in a simpler and faster-training algorithm with matching performance (Bińkowski et al., 2018) .

This motivated us to choose MMD as a metric for regularizing distribution matching.

Let us adapt the following notation for the transformation within a standard CVAE.

High-dimensional observations x and a scalar or low-dimensional condition s are transformed using f (encoder) and g (decoder), which are parametrized by weight-sharing neural networks, and give rise to predictorsẑ,ŷ andx:ẑ

where we distinguished the first (g 1 ) and the remaining layers (g 2 ) of the decoder g = g 2 • g 1 (Fig.  1 ).

While z formally depends on s, it is commonly empirically observed Z ⊥ ⊥ S, that is, the representation z is disentangled from the condition information s.

By contrast, the original representation typically strongly covaries with S: X ⊥ ⊥ S. The observation can be explained by admitting that an efficient z-representation, suitable for minimizing reconstruction and regularization losses, should be as free as possible from information about s. Information about s is directly and explicitly available to the decoder (equation 7b), and hence, there is an incentive to optimize the parameters of f to only explain the variation in x that is not explained by s. Experiments below demonstrate that indeed, MMD regularization on the bottleneck layer z does not improve performance.

However, even if z is completely free of variation from s, the y representation has a strong s component, Y ⊥ ⊥ S, which leads to a separation of y s=1 and y s=0 into different regions of their support Y. In the standard CVAE, without any regularization of this y representation, a highly varying, non-compact distribution emerges across different values of s (Fig. 2) .

To compactify the distribution so that it displays only subtle, controlled differences, we impose MMD (equation 4) in the first layer of the decoder (Fig. 1 ).

We assume that modeling y in the same region of the support of Y across s forces learning common features across s where possible.

The more of these common features are learned, the more accurately the transformation task will performed, and the higher are chances of successful out-of-sample generation.

Using one of the benchmark datasets introduced, below, we qualitatively illustrate the effect (Fig. 2) .

During training time, all samples are passed to the model with their corresponding condition labels (x s , s).

At prediction time, we pass (x s=0 , s = 0) to the encoder f to obtain the latent representation z s=0 .

In the decoder g, we pass (ẑ s=0 , s = 1) and through that, let the model transform data tox s=1 .

The cost function of trVAE derives directly from the standard CVAE cost function, as introduced in the backgrounds section,

Through duplicating the cost function for X and adding an MMD term, the loss of trVAE becomes:

Figure 3: Out-of-sample style transfer for Morpho-MNIST dataset containing normal, thin and thick digits.

trVAE successfully transforms normal digits to thin (a) and thick ((b) for digits not seen during training (out-of-sample).

We demonstrate the advantages of an MMD-regularized first layer of the decoder by benchmarking versus a variety of existing methods and alternatives:

• Vanilla CVAE (Sohn et al., 2015) • CVAE with MMD on bottleneck (MMD-CVAE), similar to VFAE (Louizos et al., 2015) • MMD-regularized autoencoder (Dziugaite et al., 2015b; Amodio et al., 2019) • CycleGAN (Zhu et al., 2017) • scGen, a VAE combined with vector arithmetics (Lotfollahi et al., 2019) • scVI, a CVAE with a negative binomial output distribution (Lopez et al., 2018a) First, we demonstrate trVAE's basic out-of-sample style transfer capacity on two established image datasets, on a qualitative level.

We then address quantitative comparisons of challenging benchmarks with clear ground truth, predicting the effects of biological perturbation based on high-dimensional structured data.

We used convolutional layers for imaging examples in section 4.1 and fully connected layers for single-cell gene expression datasets in sections 4.2 and 4.3.

The optimal hyper-parameters for each application were chosen by using a parameter gird-search for each model.

The detailed hyper-parameters for different models are reported in tables 1-9 in appendix A.

Here, we use Morpho-MNIST (Castro et al., 2018) , which contains 60,000 images each of "normal" and "transformed" digits, which are drawn with a thinner and thicker stroke.

For training, we used all normal-stroke data.

Hence, the training data covers all domains (d ∈ {0, 1, 2, . . .

, 9}) in the normal stroke condition (s = 0).

In the transformed conditions (thin and thick strokes, s ∈ {1, 2}), we only kept domains d ∈ {1, 3, 6, 7}.

We train a convolutional trVAE in which we first encode the stroke width via two fully-connected layers with 128 and 784 features, respectively.

Next, we reshape the 784-dimensional into 28*28*1 images and add them as another channel in the image.

Such trained trVAE faithfully transforms digits of normal stroke to digits of thin and thicker stroke to the out-of-sample domains (Fig. 3) Figure 4: CelebA dataset with images in two conditions: celebrities without a smile and with a smile on their face.

trVAE successfully adds a smile on faces of women without a smile despite these samples completely lacking from the training data (out-of-sample).

The training data only comprises non-smiling women and smiling and non-smiling men.

Next, we apply trVAE to CelebA (Liu et al., 2015) , which contains 202,599 images of celebrity faces with 40 binary attributes for each image.

We focus on the task of learning a transformation that turns a non-smiling face into a smiling face.

We kept the smiling (s) and gender (d) attributes and trained the model with images from both smiling and non-smiling men but only with non-smiling women.

In this case, we trained a deep convolutional trVAE with a U-Net-like architecture (Ronneberger et al., 2015) .

We encoded the binary condition labels as in the Morpho-MNIST example and fed them as an additional channel in the input.

Predicting out-of-sample, trVAE successfully transforms non-smiling faces of women to smiling faces while preserving most aspects of the original image (Fig. 4) .

In addition to showing the model's capacity to handle more complex data, this example demonstrates the flexibility of the the model adapting to well-known architectures like U-Net in the field.

Accurately modeling cell response to perturbations is a key question in computational biology.

Recently, neural network models have been proposed for out-of-sample predictions of high-dimensional tabular data that quantifies gene expression of single-cells (Lotfollahi et al., 2019; Amodio et al., 2018) .

However, these models are not trained on the task relying instead on hard-coded transformations and cannot handle more than two conditions.

We evaluate trVAE on a single-cell gene expression dataset that characterizes the gut (Haber et al., 2017) after Salmonella or Heligmosomoides polygyrus (H. poly) infections, respectively.

For this, we closely follow the benchmark as introduced in (Lotfollahi et al., 2019) .

The dataset contains eight different cell types in four conditions: control or healthy cells (n=3,240), H.Poly infection a after three days (H.Poly.

Day3, n=2,121), H.poly infection after 10 days (H.Poly.

Day10, n=2,711) and salmonella infection (n=1,770) (Fig. 5a) .

The normalized gene expression data has 1,000 dimensions corresponding to 1,000 genes.

Since three of the benchmark models are only able to handle two conditions, we only included the control and H.Poly.

Day10 conditions for model comparisons.

In this setting, we hold out Tuft infected cells for training and validation, as these consitute the hardest case for out-of-sample generalization (least shared features, few training data).

Figure 5b-c shows trVAE accurately predicts the mean and variance for high-dimensional gene expression in Tuft cells.

We compared the distribution of Defa24, the gene with the highest change after H.poly infection in Tuft cells, which shows trVAE provides better estimates for mean and variance compared to other models.

Moreover, trVAE outperforms other models also when quantifying the correlation of the predicted 1,000 dimensional x with its ground truth (Fig. 5e ).

In particular, we note that the MMD regularization on the bottleneck layer of the CVAE does not improve performance, as argued above.

In order to show our model is able to handle multiple conditions, we performed another experiment with all three conditions included.

We trained trVAE holding out each of the eight cells types in all perturbed conditions.

Figure 5f shows trVAE can accurately predict all cell types in each perturbed condition, in contrast to existing models.

Similar to modeling infection response as above, we benchmark on another single-cell gene expression dataset consisting of 7,217 IFN-β stimulated and 6,359 control peripheral blood mononuclear cells (PBMCs) from eight different human Lupus patients (Kang et al., 2018) .

The stimulation with IFN-β induces dramatic changes in the transcriptional profiles of immune cells, which causes big shifts between control and stimulated cells (Fig. 6a) .

We studied the out-of-sample prediction of natural killer (NK) cells held out during the training of the model.

trVAE accurately predicts mean (Fig. 6b) and variance (Fig. 6c) for all genes in the held out NK cells.

In particular, genes strongly responding to IFN-β (highlighted in red in Fig. 6b-c) are well captured.

An effect of applying IFN-β is an increase in ISG15 for NK cells, which the model never sees during training.

trVAE predicts this change by increasing the expression of ISG15 as observed in real NK cells (Fig. 6d) .

A cycle GAN and an MMD-regularized auto-encoder (SAUCIE) and other models yield less accurate results than our model.

Comparing the correlation of predicted mean and variance of gene expression for all dimensions of the data, we find trVAE performs best (Fig. 6e) .

By arguing that the vanilla CVAE yields representations in the first layer following the bottleneck that vary strongly across categorical conditions, we introduced an MMD regularization that forces these representations to be similar across conditions.

The resulting model (trVAE) outperforms existing modeling approaches on benchmark and real-world data sets.

Within the bottleneck layer, CVAEs already display a well-controlled behavior, and regularization does not improve performance.

Further regularization at later layers might be beneficial but is numerically costly and unstable as representations become high-dimensional.

However, we have not yet systematically investigated this and leave it for future studies.

Further future work will concern the application of trVAE on larger and more data, focusing on interaction effects among conditions.

For this, an important application is the study of drug interaction effects, as previously noted by Amodio et al. (2018) .

Future conceptual investigations concern establishing connections to causal-inference-inspired models such as CEVAE (Louizos et al., 2017) : faithful modeling of an interventional distribution might possibly be re-framed as successful perturbation effect prediction across domains.

A HYPER-PARAMETERS

@highlight

Generates never seen data during training from a desired condition 