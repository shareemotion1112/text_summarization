We introduce the Convolutional Conditional Neural Process (ConvCNP), a new member of the Neural Process family that models translation equivariance in the data.

Translation equivariance is an important inductive bias for many learning problems including time series modelling, spatial data, and images.

The model embeds data sets into an infinite-dimensional function space, as opposed to finite-dimensional vector spaces.

To formalize this notion, we extend the theory of neural representations of sets to include functional representations, and demonstrate that any translation-equivariant embedding can be represented using a convolutional deep-set.

We evaluate ConvCNPs in several settings, demonstrating that they achieve state-of-the-art performance compared to existing NPs.

We demonstrate that building in translation equivariance enables zero-shot generalization to challenging, out-of-domain tasks.

Neural Processes (NPs; Garnelo et al., 2018b; a) are a rich class of models that define a conditional distribution p(y|x, Z, θ) over output variables y given input variables x, parameters θ, and a set of observed data points in a context set Z = {x m , y m } M m=1 .

A key component of NPs is the embedding of context sets Z into a representation space through an encoder Z → E(Z), which is achieved using a DEEPSETS function approximator (Zaheer et al., 2017 ).

This simple model specification allows NPs to be used for (i) meta-learning (Thrun & Pratt, 2012; Schmidhuber, 1987) , since predictions can be generated on the fly from new context sets at test time; and (ii) multi-task or transfer learning (Requeima et al., 2019) , since they provide a natural way of sharing information between data sets.

Moreover, conditional NPs (CNPs; Garnelo et al., 2018a) , a deterministic variant of NPs, can be trained in a particularly simple way with maximum likelihood learning of the parameters θ, which mimics how the system is used at test time, leading to strong performance (Gordon et al., 2019) .

Natural application areas of NPs include time series, spatial data, and images with missing values.

Consequently, such domains have been used extensively to benchmark current NPs (Garnelo et al., 2018a; b; Kim et al., 2019) .

Often, ideal solutions to prediction problems in such domains should be translation equivariant: if the data are translated in time or space, then the predictions should be translated correspondingly (Kondor & Trivedi, 2018; Cohen & Welling, 2016) .

This relates to the notion of stationarity.

As such, NPs would ideally have translation equivariance built directly into the modelling assumptions as an inductive bias.

Unfortunately, current NP models must learn this structure from the data set instead, which is sample and parameter inefficient as well as impacting the ability of the models to generalize.

The goal of this paper is to build translation equivariance into NPs.

Famously, convolutional neural networks (CNNs) added translation equivariance to standard multilayer perceptrons (LeCun et al., 1998; Cohen & Welling, 2016) .

However, it is not straightforward to generalize NPs in an analogous way: (i) CNNs require data to live "on the grid" (e.g. image pixels form a regularly spaced grid), while many of the above domains have data that live "off the grid" (e.g. time series data may be observed irregularly at any time t ∈ R). (ii) NPs operate on partially observed context sets whereas CNNs typically do not. (iii) NPs rely on embedding sets into a finite-dimensional vector space for which the notion of equivariance with respect to input translations is not natural, as we detail in Section 3.

In this work, we introduce the CONVCNP, a new member of the NP family that accounts for translation equivariance.

1 This is achieved by extending the theory of learning on sets to include functional representations, which in turn can be used to express any translation-equivariant NP model.

Our key contributions can be summarized as follows.

(i) We provide a representation theorem for translation-equivariant functions on sets, extending a key result of Zaheer et al. (2017) to functional embeddings, including sets of varying size. (ii) We extend the NP family of models to include translation equivariance. (iii) We evaluate the CONVCNP and demonstrate that it exhibits excellent performance on several synthetic and real-world benchmarks.

In this section we introduce the notation and precisely define the problem this paper addresses.

Notation.

In the following, let X = R d and Y ⊆ R d (with Y compact) be the spaces of inputs and outputs respectively.

To ease notation, we often assume scalar outputs Y ⊆ R. Define Z M = (X × Y)

M as the collection of M input-output pairs, Z ≤M = M m=1 Z m as the collection of at most M pairs, and Z = ∞ m=1 Z M as the collection of finitely many pairs.

Since we will consider permutation-invariant functions on Z (defined later in Property 1), we may refer to elements of Z as sets or data sets.

Furthermore, we will use the notation [n] = {1, . . .

, n}.

CNPs model predictive distributions as p(y|x, Z) = p(y|Φ(x, Z), θ), where Φ is defined as a composition ρ • E of an encoder E : Z → R e mapping into the embedding space R e and a decoder ρ : R e → C b (X , Y).

Here E(Z) ∈ R e is a vector representation of the set Z, and C b (X , Y) is the space of continuous, bounded functions X → Y endowed with the supremum norm.

While NPs (Garnelo et al., 2018b) employ latent variables to indirectly specify predictive distributions, in this work we focus on CNP models which do not.

Then a mapping Φ : Z → H is called translation equivariant if Φ(T τ Z) = T τ Φ(Z) for all τ ∈ X and Z ∈ Z.

Having formalized the problem, we now describe how to construct CNPs that translation equivariant.

We are interested in translation equivariance (Property 2) with respect to translations on X .

The NP family encoder maps sets Z to an embedding in a vector space R d , for which the notion of equivariance with respect to input translations in X is not well defined.

For example, a function f on X can be translated by τ ∈ X : f ( · − τ ).

However, for a vector x ∈ R d , which can be seen as a function [d] → R, x(i) = x i , the translation x( · − τ ) is not well-defined.

To overcome this issue, we enrich the encoder E : Z → H to map into a function space H containing functions on X .

Since functions in H map from X , our notion of translation equivariance (Property 2) is now also well defined for E(Z).

As we demonstrate below, every translation-equivariant function on sets has a representation in terms of a specific functional embedding.

Definition 1 (Functional mappings on sets and functional representations of sets).

Call a map E : Z → H a functional mapping on sets if it maps from sets Z to an appropriate space of functions H. Furthermore, call E(Z) the functional representation of the set Z.

Considering functional representations of sets leads to the key result of this work, which can be summarized as follows.

For Z ⊆ Z appropriate, a continuous function Φ : Z → C b (X , Y) satisfies Properties 1 and 2 if and only if it has a representation of the form Φ(Z) = ρ (E(Z)) , E(Z) = (x,y)∈Z φ(y)ψ( · − x) ∈ H,

for some continuous and translation-equivariant ρ : H → C b (X , Y), and appropriate φ and ψ.

Note that ρ is a map between function spaces.

We also remark that continuity of Φ is not in the usual sense; we return to this below.

Equation (1) defines the encoder used by our proposed model, the CONVCNP.

In Section 3.1, we present our theoretical results in more detail.

In particular, Theorem 1 establishes equivalence between any function satisfying Properties 1 and 2 and the representational form in Equation (1).

In doing so, we provide an extension of the key result of Zaheer et al. (2017) to functional representations on sets, and show that it can naturally be extended to handle varying-size sets.

The practical implementation of CONVCNPs -the design of ρ, φ, and ψ -is informed by our results in Section 3.1 (as well as the proofs, provided in Appendix A), and is discussed for domains of interest in Section 4.

In this section we establish the theoretical foundation of the CONVCNP.

We begin by stating a definition that is used in our main result.

We denote [m] = {1, . . .

, m}.

Definition 2 (Multiplicity).

A collection Z ⊆ Z is said to have multiplicity K if, for every set Z ∈ Z , every x occurs at most K times:

For example, in the case of real-world data like time series and images, we often observe only one (possibly multi-dimensional) observation per input location, which corresponds to multiplicity one.

We are now ready to state our key theorem.

3 , permutation invariant (Property 1), and translation

Apply CNN and predict

require: ρ = (CNN, ψρ), ψ, and density γ require: context (xn, yn)

10 end (b)

require: ρ = CNN and E = CONV θ require: image I, context Mc, and target mask Mt 1 begin 2 //

We discretize at the pixel locations.

Figure 1: (a) Illustration of the CONVCNP forward pass in the off-the-grid case and pseudo-code for (b) off-the-grid and (c) on-the-grid data.

The function pos : R → (0, ∞) is used to enforce positivity.

for some continuous and translation-equivariant ρ : H → C b (X , Y) and some continuous φ : Y → R K+1 and ψ : X → R, where H is an appropriate space of functions that includes the image of E. We call a function Φ of the above form CONVDEEPSET.

The proof of Theorem 1 is provided in Appendix A. We here discuss several key points from the proof that have practical implications and provide insights for the design of CONVCNPs: (i) For the construction of ρ and E, ψ is set to a flexible positive-definite kernel associated with a Reproducing Kernel Hilbert Space (RKHS; Aronszajn (1950)), which results in desirable properties for E. (ii) Using the work by Zaheer et al. (2017) , we set φ(y) = (y 0 , y 1 , · · · , y K ) to be the powers of y up to order K. (iii) Theorem 1 requires ρ to be a powerful function approximator of continuous, translation-equivariant maps between functions.

In Section 4, we discuss how these theoretical results inform our implementations of CONVCNPs.

Theorem 1 extends the result of Zaheer et al. (2017) discussed in Section 2 by embedding the set into an infinite-dimensional space -the RKHS -instead of a finite-dimensional space.

Beyond allowing the model to exhibit translation equivariance, the RKHS formalism allows us to naturally deal with finite sets of varying sizes, which turns out to be challenging with finite-dimensional embeddings.

Furthermore, our formalism requires φ(y) = (y 0 , y 1 , y 2 , . . .

, y K ) to expand up to order no more than the multiplicity of the sets K; if K is bounded, then our results hold for sets up to any arbitrarily large finite size M , while fixing φ to be only (K + 1)-dimensional.

In this section we discuss the architectures and implementation details for CONVCNPs.

Similar to NPs, CONVCNPs model the conditional distribution as

where Z is the observed data and Φ a CONVDEEPSET.

The key considerations are the design of ρ, φ, and ψ for Φ. We provide separate models for data that lie on-the-grid and data that lie off-the-grid.

Form of φ.

The applications considered in this work have a single (potentially multi-dimensional) output per input location, so the multiplicity of Z is one (i.e., K = 1).

It then suffices to let φ be a power series of order one, which is equivalent to appending a constant to y in all data sets, i.e. φ(y) = [1 y] .

The first output φ 1 thus provides the model with information regarding where data has been observed, which is necessary to distinguish between no observed datapoint at x and a datapoint at x with y = 0.

Denoting the functional representation as h, we can think of the first channel h (0) as a "density channel".

We found it helpful to divide the remaining channels h (1:) by h (0) (Figure 1b , line 5), as this improved performance when there is large variation in the density of input locations.

In the image processing literature, this is known as normalized convolution (Knutsson & Westin, 1993) .

The normalization operation can be reversed by ρ and is therefore not restrictive.

CONVCNPs for off-the-grid data.

Having specified φ, it remains to specify the form of ψ and ρ.

Our proof of Theorem 1 suggests that ψ should be a stationary, non-negative, positive-definite kernel.

The exponentiated-quadratic (EQ) kernel with a learnable length scale parameter is a natural choice.

This kernel is multiplied by φ to form the functional representation E(Z) ( Figure 1b , line 4; and Figure 1a , arrow 1).

Next, Theorem 1 suggests that ρ should be a continuous, translation-equivariant map between function spaces.

Kondor & Trivedi (2018) show that, in deep learning, any translation-equivariant model has a representation as a CNN.

However, CNNs operate on discrete (on-the-grid) input spaces and produce discrete outputs.

In order to approximate ρ with a CNN, we discretize the input of ρ, apply the CNN, and finally transform the CNN output back to a continuous function X → Y. To do this, for each context and test set, we space points (t i ) n i=1 ⊆ X on a uniform grid (at a pre-specified density) over a hyper-cube that covers both the context and target inputs.

We then evaluate (E(Z)(t i )) To map the output of the CNN back to a continuous function X → Y, we use the CNN outputs as weights for evenly-spaced basis functions (again employing the EQ kernel), which we denote by ψ ρ (Figure 1b , lines 7-8; Figure 1a , arrow 3).

The resulting approximation to ρ is not perfectly translation equivariant, but will be approximately so for length scales larger than the spacing of (E(Z)(t i ))

n i=1 .

The resulting continuous functions are then used to generate the (Gaussian) predictive mean and variance at any input.

This, in turn, can be used to evaluate the log-likelihood.

CONVCNP for on-the-grid data.

While CONVCNP is readily applicable to many settings where data live on a grid, in this work we focus on the image setting.

As such, the following description uses the image completion task as an example, which is often used to benchmark NPs (Garnelo et al., 2018a; Kim et al., 2019) .

Compared to the off-the-grid case, the implementation becomes simpler as we can choose the discretization (t i ) n i=1 to be the pixel locations.

Let I ∈ R H×W ×C be an image -H, W, C denote the height, width, and number of channels, respectively -and let M c be the context mask, which is such that [M c ] i,j = 1 if pixel location (i, j) is in the context set, and 0 otherwise.

To implement φ, we select all context points, Z c := M c I, and prepend the context mask: (Figure 1c, line 4) .

Next, we apply a convolution to the context mask to form the density channel: Figure 1c, line 4) .

To all other channels, we apply a normalized convolution: (Figure 1c , line 5), where the division is element-wise.

The filter of the convolution is analogous to ψ, which means that h is the functional representation, with the convolution performing the role of E (the summation in Figure 1b , line 4).

Although the theory suggests using a non-negative, positive-definite kernel, we did not find significant empirical differences between an EQ kernel and using a fully trainable kernel restricted to positive values to enforce non-negativity (see Appendices D.4 and D.5 for details).

Lastly, we describe the on-the-grid version of ρ(·), which consists of two stages.

First, we apply a CNN to E(Z) (Figure 1c, line 6 ).

Second, we apply a shared, pointwise MLP that maps the output of the CNN at each pixel location in the target set to R 2C , where we absorb MLP into the CNN (MLP can be viewed as an 1×1 convolution).

The first C outputs are the means of a Gaussian predictive distribution and the second C the standard deviations, which then pass through a positivity-enforcing function (Figure 1c, .

To summarise, the on-the-grid algorithm is given by

multiplies by ψ and sums

where (µ, σ) are the image mean and standard deviation, ρ is implemented with CNN, and E is implemented with the mask M c and convolution CONV.

Training.

Denoting the data set D = {Z n } N n=1 ⊆ Z and the parameters by θ, maximum-likelihood training involves (Garnelo et al., 2018a; b)

where we have split Z n into context (Z n,c ) and target (Z n,t ) sets.

This is standard practice in the NP (Garnelo et al., 2018a; b) and meta-learning settings (Finn et al., 2017; Gordon et al., 2019) and relates to neural auto-regressive models (Requeima et al., 2019) .

Note that the context set and target set are disjoint (Z n,c ∩ Z n,t = ∅), which differs from the protocol for the NP (Garnelo et al., 2018a) .

Practically, stochastic gradient descent methods (Bottou, 2010) can be used for optimization.

We evaluate the performance of CONVCNPs in both on-the-grid and off-the-grid settings focusing on two central questions: (i) Do translation-equivariant models improve performance in appropriate domains? (ii) Can translation equivariance enable CONVCNPs to generalize to settings outside of those encountered during training?

We use several off-the-grid data-sets which are irregularly sampled time series (X = R), comparing to Gaussian processes (GPs; Williams & Rasmussen (2006) ) and ATTNCNP(which is identical to the ANP (Kim et al., 2019) , but without the latent path in the encoder), the best performing member of the CNP family.

We then evaluate on several on-the-grid image data sets (X = Z 2 ).

In all settings we demonstrate substantial improvements over existing neural process models.

For the CNN component of our model, we propose a small and large architecture for each experiment (in the experimental sections named CONVCNP and CONVCNPXL, respectively).

We note that these architectures are different for off-the-grid and on-the-grid experiments, with full details regarding the architectures given in the appendices.

First we consider synthetic regression problems.

At each iteration, a function is sampled, followed by context and target sets.

Beyond EQ-kernel GPs (as proposed in Garnelo et al. (2018a) ; Kim et al. (2019) ), we consider more complex data arising from Matern-5 2 and weakly-periodic kernels, as well as a challenging, non-Gaussian sawtooth process with random shift and frequency (see Figure 2 , for example).

CONVCNP is compared to CNP (Garnelo et al., 2018a) and ATTNCNP.

Training and testing procedures are fixed across all models.

Full details on models, data generation, and training procedures are provided in Appendix C.

Params EQ Weak Periodic Matern Sawtooth CNP 66818 -1.49 ± 3e-3 -2.00 ± 2e-3 -1.61 ± 1e-3 -0.51 ± 1e-5 ATTNCNP 149250 1.08 ± 4e-3 -2.01 ± 2e-3 0.42 ± 2e-3 -0.50 ± 2e-3 CONVCNP 6537 1.03 ± 5e-3 -1.52 ± 2e-3 0.51 ± 4e-3 4.38 ± 4e-3 CONVCNPXL 50617 1.90 ± 4e-3 -1.19 ± 2e-3 0.81 ± 4e-3 6.01 ± 1e-3 Table 1 reports the log-likelihood means and standard errors of the models over 1000 tasks.

The context and target points for both training and testing lie within the interval [−2, 2] where training data was observed (marked "training data range" in Figure 2 ).

Table 1 demonstrates that, even when extrapolation is not required, CONVCNP significantly outperforms other models in all cases, despite having fewer parameters.

of CONVCNP and ATTNCNP when data is observed outside the range where the models were trained: translation equivariance enables CONVCNP to elegantly generalize to this setting, whereas ATTNCNP is unable to generate reasonable predictions.

The PLAsTiCC data set (Allam Jr et al., 2018 ) is a simulation of transients observed by the LSST telescope under realistic observational conditions.

The data set contains 3,500,734 "light curves", where each measurement is of an object's brightness as a function of time, taken by measuring the photon flux in six different astronomical filters.

The data can be treated as a six-dimensional time series.

The data set was introduced in a Kaggle competition, 4 where the task was to use these light curves to classify the variable sources.

The winning entry (Avocado, Boone, 2019) modeled the light curves with GPs and used these models to generate features for a gradient boosted decision tree classifier.

We compare a multi-input-multi-output CONVCNP with the GP models used in Avocado.

5 CONVCNP accepts six channels as inputs, one for each astronomical filter, and returns 12 outputs -the means and standard deviations of six Gaussians.

Full experimental details are given in Appendix C.3.

The mean squared error of both approaches is similar, but the held-out log-likelihood from the CONVCNP is far higher (see Table 2 ).

Table 2 : Mean and standard errors of log-likelihood and root mean squared error over 1000 test objects from the PLastiCC dataset.

Log-likelihood MSE Kaggle GP (Boone, 2019) -0.335 ± 0.09 0.037 ± 4e-3 ConvCP (ours)

1.31 ± 0.30 0.040 ± 5e-3

The CONVCNP model is well suited for applications where simulation data is plentiful, but real world training data is scarce (Sim2Real).

The CONVCNP can be trained on a large amount of simulation data and then be deployed with real-world training data as the context set.

We consider the Lotka-Volterra model (Wilkinson, 2011) , which is used to describe the evolution of predator-prey populations.

This model has been used in the Approximate Bayesian Computation literature where the task is to infer the parameters from samples drawn from the Lotka-Volterra process (Papamakarios & Murray, 2016) .

These methods do not simply extend to prediction problems such as interpolation or forecasting.

In contrast, we train CONVCNP on synthetic data sampled from the Lotka-Volterra model and can then condition on real-world data from the Hudson's Bay lynx-hare data set (Leigh, 1968) to perform interpolation (see Figure 3 ; full experimental details are given in Appendix C.4).

The CONVCNP performs accurate interpolation as shown in Figure 3 .

We were unable to successfully train the ATTNCNP for this task.

We suspect this is because the simulation data are variable lengthtime series, which requires models to leverage translation equivariance at training time.

As shown in Section 5.1, the ATTNCNP struggles to do this (see Appendix C.4 for complete details).

To test CONVCNP beyond one-dimensional features, we evaluate our model on on-the-grid image completion tasks and compare it to ATTNCNP.

Image completion can be cast as a prediction of pixel intensities y * i (∈ R 3 for RGB, ∈ R for greyscale) given a target 2D pixel location x * i conditioned on an observed (context) set of pixel values Z = (x n , y n ) N n=1 .

In the following experiments, the context set can vary but the target set contains all pixels from the image.

Further experimental details are in Appendix D.1.

Standard benchmarks.

We first evaluate the model on four common benchmarks: MNIST (LeCun et al., 1998) , SVHN (Netzer et al., 2011) , and 32 × 32 and 64 × 64 CelebA (Liu et al., 2018) .

Importantly, these data sets are biased towards images containing a single, well-centered object.

As a result, perfect translation-equivariance might hinder the performance of the model when the test data are similarly structured.

We therefore also evaluated a larger CONVCNP that can learn such non-stationarity, while still sharing parameters across the input space (CONVCNPXL).

Table 3 shows that CONVCNP significantly outperforms ATTNCNP when it has a large receptive field size, while being at least as good with a small receptive field size.

Qualitative samples for various context sets can be seen in Figure 5 .

Further qualitative comparisons and ablation studies can be found in Appendix D.3 and Appendix D.4 respectively.

Generalization to multiple, non-centered objects.

The data sets from the previous paragraphs were centered and contained single objects.

Here we test whether CONVCNPs trained on such data can generalize to images containing multiple, non-centered objects.

The last column of Table 3 evaluates the models in a zero shot multi-MNIST (ZSMM) setting, where images contain multiple digits at test time (Appendix D.2).

CONVCNP significantly outperforms ATTNCNP on such tasks.

Figure 4a shows a histogram of the image log-likelihoods for CONVCNP and ATTNCNP, as well as qualitative results at different percentiles of the CONVCNP distribution.

CONVCNP is able to extrapolate to this out-of-distribution test set, while ATTNCNP appears to model the bias of the training data and predict a centered "mean" digit independently of the context.

Interestingly, CONVCNPXL does not perform as well on this task.

In particular, we find that, as the receptive field becomes very large, performance on this task decreases.

We hypothesize that this has to do with behavior of the model at the edges of the image.

CNNs with larger receptive fields -the region of input pixels that affect a particular output pixel -are able to model non-stationary behavior by looking at the distance from any pixel to the image boundary.

We expand on this discussion and provide further experimental evidence regarding the effects of receptive field on the ZSMM task in Appendix D.6.

Although ZSMM is a contrived task, note that our field of view usually contains multiple independent objects, thereby requiring translation equivariance.

As a more realistic example, we took a CONVCNP model trained on CelebA and tested it on a natural image of different shape which contains multiple people (Figure 4b ).

Even with 95% of the pixels removed, the CONVCNP was able to produce a qualitatively reasonable reconstruction.

A comparison with ATTNCNP is given in Appendix D.3.

Computational efficiency.

Beyond the performance and generalization improvements, a key advantage of the CONVCNP is its computational efficiency.

The memory and time complexity of a single self-attention layer grows quadratically with the number of inputs M (the number of pixels for images) but only linearly for a convolutional layer.

Empirically, with a batch size of 16 on 32 × 32 MNIST, CONVCNPXL requires 945MB of VRAM, while ATTNCNP requires 5839 MB.

For the 56×56 ZSMM CONVCNPXL increases its requirements to 1443 MB, while ATTNCNP could not fit onto a 32GB GPU.

Ultimately, ATTNCNP had to be trained with a batch size of 6 (using 19139 MB) and we were not able to fit it for CelebA64.

Recently, restricted attention has been proposed to overcome this computational issue (Parmar et al., 2018 ), but we leave an investigation of this and its relationship to CONVCNPs to future work.

We have introduced CONVCNP, a new member of the CNP family that leverages embedding sets into function space to achieve translation equivariance.

The relationship to (i) the NP family, and (ii) representing functions on sets, each imply extensions and avenues for future work.

Deep sets.

Two key issues in the existing theory on learning with sets (Zaheer et al., 2017; Qi et al., 2017a; Wagstaff et al., 2019) are (i) the restriction to fixed-size sets, and (ii) that the dimensionality of the embedding space must be no less than the cardinality of the embedded sets.

Our work implies that by considering appropriate embeddings into a function space, both issues are alleviated.

In future work, we aim to further this analysis and formalize it in a more general context.

Point-cloud models.

Another line of related research focuses on 3D point-cloud modelling (Qi et al., 2017a; b) .

While original work focused on permutation invariance (Qi et al., 2017a; Zaheer et al., 2017) , more recent work has considered translation equivariance as well (Wu et al., 2019) , leading to a model closely resembling CONVDEEPSETS.

The key differences with our work are the following: (i) Wu et al. (2019) Correlated samples and consistency under marginalization.

In the predictive distribution of CON-VCNP (Equation (2)), predicted ys are conditionally independent given the context set.

Consequently, samples from the predictive distribution lack correlations and appear noisy.

One solution is to instead define the predictive distribution in an autoregressive way, like e.g. PixelCNN++ (Salimans et al., 2017) .

Although samples are now correlated, the quality of the samples depends on the order in which the points are sampled.

Moreover, the predicted ys are then not consistent under marginalization (Garnelo et al., 2018b; Kim et al., 2019) .

Consistency under marginalization is more generally an issue for neural autoregressive models (Salimans et al., 2017; Parmar et al., 2018) , although consistent variants have been devised (Louizos et al., 2019) .

To overcome the consistency issue for CONVCNP, exchangeable neural process models (e.g. Korshunova et al., 2018; Louizos et al., 2019) may provide an interesting avenue.

Another way to introduce dependencies between ys is to employ latent variables as is done in neural processes (Garnelo et al., 2018b) .

However, such an approach only achieves conditional consistency: given a context set, the predicted ys will be dependent and consistent under marginalization, but this does not lead to a consistent joint model that also includes the context set itself.

For each dataset, an image is randomly sampled, the first row shows the given context points while the second is the mean of the estimated conditional distribution.

From left to right the first seven columns correspond to a context set with 3, 1%, 5%, 10%, 20%, 30%, 50%, 100% randomly sampled context points.

In the last two columns, the context sets respectively contain all the pixels in the left and top half of the image.

CONVCNPXL is shown for all datasets besides ZSMM, for which we show the fully translation equivariant CONVCNP.

In this section, we provide the proof of Theorem 1.

Our proof strategy is as follows.

We first define an appropriate topology for fixed-sized sets (Appendix A.1).

With this topology in place, we demonstrate that our proposed embedding into function space is homeomorphic (Lemmas 1 and 2).

We then show that the embeddings of fixed-sized sets can be extended to varying-sized sets by "pasting" the embeddings together while maintaining their homeomorphic properties (Lemma 3).

Following this, we demonstrate that the resulting embedding may be composed with a continuous mapping to our desired target space, resulting in a continuous mapping between two metric spaces (Lemma 4).

Finally, in Appendix A.3 we combine the above-mentioned results to prove Theorem 1.

We begin with definitions that we will use throughout the section and then present our results.

Let X = R d and let Y ⊆ R be compact.

Let ψ be a symmetric, positive-definite kernel on X .

By the Moore-Aronszajn Theorem, there is a unique Hilbert space (H, · , · H ) of real-valued functions on X for which ψ is a reproducing kernel.

This means that (i) ψ( · , x) ∈ H for all x ∈ X and (ii) f, ψ( · , x) H = f (x) for all f ∈ H and x ∈ X (reproducing property).

For ψ : X × X → R, X = (x 1 , . . . , x n ) ∈ X n , and X = (x 1 , . . . , x n ) ∈ X n , we denote

Definition 3 (Interpolating RKHS).

Call H interpolating if it interpolates any finite number of points: for every ((x i , y i ))

For example, the RKHS induced by any strictly positive-definite kernel, e.g. the exponentiated quadratic (EQ) kernel ψ(x,

A.1 THE QUOTIENT SPACE A n / S n Let A be a Banach space.

For x = (x 1 , . . .

, x n ) ∈ A n and y = (y 1 , . . . , y n ) ∈ A n , let x ∼ y if x is a permutation of y; that is, x ∼ y if and only if x = πy for some π ∈ S n where πy = (y π(1) , . . .

, y π(n) ).

Let A n / S n be the collection of equivalence classes of ∼. Denote the equivalence class of

Proof.

We first show that d is well defined on A n / S n .

Assume x ∼ x and y ∼ y .

Then, x = π x x and y = π y y. Using the group properties of S n and the permutation invariance of · A n :

To show the triangle inequality, note that

using permutation invariance of · A n .

Hence, taking the minimum over π 1 ,

so taking the minimum over π 2 gives the triangle inequality for d.

Proposition 2.

The canonical map A n →

A n / S n is continuous under the metric topology induced by d.

Proposition 3.

Let A ⊆ A n be topologically closed and closed under permutations.

Then [A] is topologically closed in A n / S n under the metric topology.

Then there are permutations (π n ) ∞ n=1 ⊆

S n such that π n a n → x. Here π n a n ∈ A, because A is closed under permutations.

Thus x ∈ A, as A is also topologically closed.

We conclude that Proposition 5.

The quotient topology on A n / S n induced by the canonical map is metrizable with the metric d.

Proof.

Since the canonical map is surjective, there exists exactly one topology on A n / S n relative to which the canonical map is a quotient map: the quotient topology (Munkres, 1974).

Let p : A n →

A n / S n denote the canonical map.

It remains to show that p is a quotient map under the metric topology induced by d; that is, we show that U ⊂ A n / S n is open in A n / S n under the metric topology if and only if p

Whereas A previously denoted an arbitrary Banach space, in this section we specialize to A = X × Y.

We denote an element in A by (x, y) and an element in Z M = A M by ((x 1 , y 1 ) , . . .

, (x M , y M )).

Alternatively, we denote ((x 1 , y 1 ) , . . . , (x M , y M )) by (X, y) where X = (x 1 , . . .

, x M ) ∈ X M and y = (y 1 , . . . , y M ) ∈ Y M .

We clarify that an element in Z M = A M is permuted as follows: for π ∈ S M , π(X, y) = π ((x 1 , y 1 ) , . . .

, (x M , y M )) = ((x π(1) , y π(1) ), . . .

, (x π(n) , y π(n) )) = (πX, πy).

Note that permutation-invariant functions on Z M are in correspondence to functions on the quotient space induced by the equivalence class of permutations, Z M / S m The latter is a more natural representation.

Lemma 3 states that it is possible to homeomorphically embed sets into an RKHS.

This result is key to proving our main result.

Before proving Lemma 3, we provide several useful results.

We begin by demonstrating that an embedding of sets of a fixed size into a RKHS is continuous and injective.

Lemma 1.

Consider a collection Z M ⊆ Z M that has multiplicity K. Set

and let ψ be an interpolating, continuous positive-definite kernel.

Define

where H K+1 = H × · · · × H is the (K + 1)-dimensional-vector-valued-function Hilbert space constructed from the RKHS H for which ψ is a reproducing kernel and endowed with the inner product f, g

is injective, hence invertible, and continuous.

Proof.

First, we show that E M is injective.

Suppose that

Denote X = (x 1 , . . . , x M ) and y = (y 1 , . . . , y M ), and denote X and y similarly.

Taking the inner product with any f ∈ H on both sides and using the reproducing property of ψ, this implies that

for all f ∈ H. In particular, since by construction φ 1 ( · ) = 1,

for all f ∈ H. Using that H is interpolating, choose a particularx ∈ X ∪ X , and let f ∈ H be such that f (x) = 1 and f ( · ) = 0 at all other x i and x i .

Then

so the number of suchx in X and the number of suchx in X are the same.

Since this holds for everyx, X is a permutation of X : X = π(X ) for some permutation π ∈ S M .

Plugging in the permutation, we can write

Then, by a similar argument, for any particularx,

Let the number of terms in each sum equal S. Since Z M has multiplicity K, S ≤ K. By Lemma 4 from Zaheer et al. (2017) , the 'sum-of-power mapping' from {y i : x i =x} to the first S + 1 elements of i:xi=x φ(y i ), i.e. i:xi=x y 0 i , . . .

, i:xi=x y S i , is injective.

Therefore, (y i ) i:xi=x is a permutation of (y π(i) ) i:xi=x .

Note that x i =x for all above y i .

Furthermore, note that also x π(i) = x i =x for all above y π(i) .

We may therefore adjust the permutation π such that y i = y π(i) for all i such that x i =x whilst retaining that x = π(x ).

Performing this adjustment for allx, we find that y = π(y ) and x = π(x ).

Second, we show that E M is continuous.

Compute

Having established the injection, we now show that this mapping is a homeomorphism, i.e. that the inverse is continuous.

This is formalized in the following lemma.

Lemma 2.

Consider Lemma 1.

Suppose that Z M is also topologically closed in A M and closed under permutations, and that ψ also satisfies (i)

M is continuous.

Remark 1.

To define Z 2 with multiplicity one, one might be tempted to define

which indeed has multiplicity one.

Unfortunately, Z 2 is not closed: if [0, 1] ⊆ X and [0, 2] ⊆ Y, then ((0, 1), (1/n, 2)) ∞ n=1 ⊆

Z 2 , but ((0, 1), (1/n, 2)) → ((0, 1), (0, 2)) / ∈ Z 2 , because 0 then has two observations 1 and 2.

To get around this issue, one can require an arbitrarily small, but non-zero spacing > 0 between input locations:

This construction can be generalized to higher numbers of observations and multiplicities as follows:

Remark 2.

Before moving on to the proof of Lemma 2, we remark that Lemma 2 would directly follow if Z M were bounded: then Z M is compact, so E M is a continuous, invertible map between a compact space and a Hausdorff space, which means that E −1 M must be continuous.

The intuition that the result must hold for unbounded Z M is as follows.

Since φ 1 ( · ) = 1, for every f ∈ H M , f 1 is a summation of M "bumps" (imagine the EQ kernel) of the form ψ( · , x i ) placed throughout X .

If one of these bumps goes off to infinity, then the function cannot uniformly converge pointwise, which means that the function cannot converge in H (if ψ is sufficiently nice).

Therefore, if the function does converge in H, (x i ) M i=1 must be bounded, which brings us to the compact case.

What makes this work is the density channel φ 1 ( · ) = 1, which forces (x i ) M i=1 to be well behaved.

The above argument is formalized in the proof of Lemma 2.

First, we demonstrate that, assuming the claim, H M is closed.

Note that by boundedness of

is compact and therefore closed, since every compact subset of a metric space is closed.

Therefore, the image of E M | [Z J ] contains the limit f .

Since the image of E M | [Z J ] is included in H M , we have that f ∈ H M , which shows that H M is closed.

Next, we prove that, assuming the claim, E −1

−1 is continuous, because a continuous bijection from a compact space to a metric space is a homeomorphism.

Therefore

M is continuous.

It remains to show the claim.

Let f 1 denote the first element of f , i.e. the density channel.

Using the reproducing property of ψ,

→ f 1 in H means that it does so uniformly pointwise (over x).

Hence, we can let N ∈ N be such that n ≥ N implies that |f

At the same time, by pointwise non-negativity of ψ, we have that

Towards contradiction, suppose that (

which is a contradiction.

The following lemma states that we may construct an encoding for sets containing no more than M elements into a function space, where the encoding is injective and every restriction to a fixed set size is a homeomorphism.

and let ψ be an interpolating, continuous positive-definite kernel that satisfies

where H K+1 = H × · · · × H is the (K + 1)-dimensional-vector-valued-function Hilbert space constructed from the RKHS H for which ψ is a reproducing kernel and endowed with the inner product f, g

is injective, hence invertible.

Denote this inverse by E −1 , where

Proof.

Recall that E m is injective for every m ∈ [M ] .

Hence, to demonstrate that E is injective it remains to show that (H m ) M m=1 are pairwise disjoint.

To this end, suppose that

for m = m .

Then, by arguments like in the proof of Lemma 1,

Since φ 1 ( · ) = 1, this gives m = m , which is a contradiction.

Finally, by repeated application of Lemma 2, E −1

, the space of continuous bounded functions from X to Y, such that every restriction Φ| [Z m ] is continuous, and let E be from Lemma 3.

Then

is continuous.

is continuous.

From here on, we let ψ be a stationary kernel, which means that it only depends on the difference of its arguments and can be seen as a function X → R.

With the above results in place, we are finally ready to prove our central result, Theorem 1.

is closed under permutations, and (iv) is closed under translations.

Set

and let ψ be an interpolating, continuous positive-definite kernel that satisfies

where H K+1 = H × · · · × H is the (K + 1)-dimensional-vector-valued-function Hilbert space constructed from the RKHS H for which ψ is a reproducing kernel and endowed with the inner product f, g

Then a function Φ : Z ≤M → C b (X , Y) satisfies (i) continuity of the restriction Φ| Zm for every m ∈ [M ], (ii) permutation invariance (Property 1), and (iii) translation equivariance (Property 2) if and only if it has a representation of the form

is continuous and translation equivariant.

Proof of sufficiency.

To begin with, note that permutation invariance (Property 1) and translation equivariance (Property 2) for Φ are well defined, because Z ≤M is closed under permutations and translations by assumption.

First, Φ is permutation invariant, because addition is commutative and associative.

Second, that Φ is translation equivariant (Property 2) follows from a direct verification and that ρ is also translation equivariant:

Proof of necessity.

Our proof follows the strategy used by Zaheer et al. (2017); Wagstaff et al. (2019) .

To begin with, since Φ is permutation invariant (Property 1), we may define

for which we verify that every restriction Φ| [Z m ] is continuous.

By invertibility of E from Lemma 3, we have

is translation equivariant, because ψ is stationary.

Also, by assumption Φ is translation equivariant (Property 2).

Thus, their composition ρ is also translation equivariant.

Remark 3.

The function ρ : H ≤M → C b (X , Y) may be continuously extended to the entirety of H K+1 using a generalisation of the Tietze Extension Theorem by Dugundji et al. (1951) .

There are variants of Dugundji's Theorem that also preserve translation equivariance.

In both our 1d and image experiments, our main comparison is to conditional neural process models.

In particular, we compare to a vanilla CNP (1d only; Garnelo et al. (2018a) ) and an ATTNCNP (Kim et al., 2019) .

Our architectures largely follow the details given in the relevant publications.

CNP baseline.

Our baseline CNP follows the implementation provided by the authors.

6 The encoder is a 3-layer MLP with 128 hidden units in each layer, and RELU non-linearities.

The encoder embeds every context point into a representation, and the representations are then averaged across each context set.

Target inputs are then concatenated with the latent representations, and passed to the decoder.

The decoder follows the same architecture, outputting mean and standard deviation channels for each input.

Attentive CNP baseline.

The ATTNCNP we use corresponds to the deterministic path of the model described by Kim et al. (2019) for image experiments.

Namely, an encoder first embeds each context point c to a latent representation (x (c) ,

xy ∈ R 128 .

For the image experiments, this is achieved using a 2-hidden layer MLP of hidden dimensions 128.

For the 1d experiments, we use the same encoder as the CNP above.

Every context point then goes through two stacked self-attention layers.

Each self-attention layer is implemented with an 8-headed attention, a skip connection, and two layer normalizations (as described in Parmar et al. (2018) , modulo the dropout layer).

To predict values at each target point t, we embed x (t) → r (t)

x and x (c)

→ r (c)

x using the same single hidden layer MLP of dimensions 128.

A target representation r (t) xy is then estimated by applying cross-attention (using an 8-headed attention described above) with keys K := {r

, and query q := r (t)

x .

Given the estimated target representationr

xy , the conditional predictive posterior is given by a Gaussian pdf with diagonal covariance parametrised by

pre ∈ R 3 and decoder is a 4 hidden layer MLP with 64 hidden units per layer for the images, and the same decoder as the CNP for the 1d experiments.

Following Le et al. (2018), we enforce we set a minimum standard deviation σ (t) min = [0.1; 0.1; 0.1] to avoid infinite log-likelihoods by using the following post-processed standard deviation:

In this section, we give details regarding our experiments for the 1d data.

We begin by detailing model architectures, and then provide details for the data generating processes and training procedures.

The density at which we evaluate the grid differs from experiment to experiment, and so the values are given in the relevant subsections.

In all experiments, the weights are optimized using Adam (Kingma & Ba, 2015) and weight decay of 10 −5 is applied to all model parameters.

The learning rates are specified in the following subsections.

Throughout the experiments (Sections 5.1 to 5.3), we consider two models: CONVCNP (which utilizes a smaller architecture), and CONVCNPXL (with a larger architecture).

For all architectures, the input kernel ψ was an EQ (exponentiated quadratic) kernel with a learnable length scale parameter, as detailed in Section 4, as was the kernel for the final output layer ψ ρ .

When dividing by the density channel, we add ε = 10 −8 to avoid numerical issues.

The length scales for the EQ kernels are initialized to twice the spacing 1/γ 1/d between the discretization points (t i )

, where γ is the density of these points and d is the dimensionality of the input space X .

Moreover, we emphasize that the size of the receptive field is a product of the width of the CNN filters and the spacing between the discretization points.

Consequently, for a fixed width kernel of the CNN, as the number of discretization points increases, the receptive field size decreases.

One potential improvement that was not employed in our experiments, is the use of depthwise-separable convolutions (Chollet, 2017) .

These dramatically reduce the number of parameters in a convolutional layer, and can be used to increase the CNN filter widths, thus allowing one to increase the number of discretization points without reducing the receptive field.

The architectures for CONVCNP and CONVCNPXL are described below.

CONVCNP.

For the 1d experiments, we use a simple, 4-layer convolutional architecture, with RELU nonlinearities.

The kernel size of the convolutional layers was chosen to be 5, and all employed a stride of length 1 and zero padding of 2 units.

The number of channels per layer was set to [16, 32, 16, 2] , where the final channels where then processed by the final, EQ-based layer of ρ as mean and standard deviation channels.

We employ a SOFTPLUS nonlinearity on the standard deviation channel to enforce positivity.

This model has 6,537 parameters.

CONVCNPXL.

Our large architecture takes inspiration from UNet (Ronneberger et al., 2015) .

We employ a 12-layer architecture with skip connections.

The number of channels is doubled every layer for the first 6 layers, and halved every layer for the final 6 layers.

We use concatenation for the skip connections.

The following describes which layers are concatenated, where L i ← [L j , L k ] means that the input to layer i is the concatenation of the activations of layers j and k:

Like for the smaller architecture, we use RELU nonlinearities, kernels of size 5, stride 1, and zero padding for two units on all layers.

The kernels used for the Gaussian Processes which generate the data in this experiment are defined as follows:

• EQ: CNP when trained on an EQ kernel (with length scale parameter 1).

"True function" refers to the sample from the GP prior from which the context and target sets were sub-sampled.

"

Ground Truth GP" refers to the GP posterior distribution when using the exact kernel and performing posterior inference based on the context set.

The left column shows the predictive posterior of the models when data is presented in same range as training.

The centre column shows the model predicting outside the training data range when no data is observed there.

The right-most column shows the model predictive posteriors when presented with data outside the training data range.

• weakly periodic:

with f 1 (x) = cos(8πx) and f 2 (x) = sin(8πx), and

with

During the training procedure, the number of context points and target points for a training batch are each selected randomly from a uniform distribution over the integers between 3 and 50.

This number of context and target points are randomly sampled from a function sampled from the process (a Gaussian process with one of the above kernels or the sawtooth process), where input locations are uniformly sampled from the interval [−2, 2].

All models in this experiment were trained for 200 epochs using 256 batches per epoch of batch size 16.

We discretize E(Z) by evaluating 64 points per unit in this setting.

We use a learning rate of 3e−4 for all models, except for CONVCNPXL on the sawtooth data, where we use a learning rate of 1e−3 (this learning rate was too large for the other models).

The random sawtooth samples are generated from the following function:

where A is the amplitude, f is the frequency, and t is "time".

Throughout training, we fix the amplitude to be one.

We truncate the series at an integer K. At every iteration, we sample a frequency uniformly in [3, 5] , K in [10, 20] , and a random shift in [−5, 5] .

As the task is much harder, we sample context and target set sizes over [3, 100] .

Here the CNP and ATTNCNP employ learning rates of 10 −3 .

All other hyperparameters remain unchanged.

The CONVCNP was trained for 200 epochs using 1024 batches of batch size 4 per epoch.

For training and testing, the number of context points for a batch are each selected randomly from a uniform distribution over the integers between 1 and the number of points available in the series (usually between 10-30 per bandwidth).

The remaining points in the series are used as the target set.

For testing, a batch size of 1 was used and statistics were computed over 1000 evaluations.

We compare CONVCNP to the GP models used in (Boone, 2019) using the implementation in https:// github.com/kboone/avocado.

The data used for training and testing is normalized according to t(v) = (v − m)/s with the values in Table 4 .

These values are estimated from a batch sampled from the training data.

To remove outliers in the GP results, log-likelihood values less than −10 are removed from the evaluation.

These same datapoints were removed from the CONVCNP results as well.

For this dataset, we only used the CONVCNPXL, as we found the CONVCNP to underfit.

The learning rate was set to 10 −3 , and we discretize E(Z) by evaluating 256 points per unit.

We describe the way simulated training data for the experiment in Section 5.3 was generated from the Lotka-Volterra model.

The description is borrowed from (Wilkinson, 2011) .

Let X be the number of predators and Y the number of prey at any point in our simulation.

According to the model, one of the following four events can occur:

A: A single predator is born according to rate θ 1 XY , increasing X by one.

B: A single predator dies according to rate θ 2 X, decreasing X by one.

C: A single prey is born according to rate θ 3 Y , increasing Y by one.

A single prey dies (is eaten) according to rate θ 4 XY , decreasing Y by one.

The parameter values θ 1 , θ 2 , θ 3 , and θ 4 , as well as the initial values of X and Y govern the behavior of the simulation.

We choose θ 1 = 0.01, θ 2 = 0.5, θ 3 = 1, and θ 4 = 0.01, which are also used in (Papamakarios & Murray, 2016) and generate reasonable time series.

Note that these are likely not the parameter values that would be estimated from the Hudson's Bay lynx-hare data set (Leigh, 1968) , but they are used because they yield reasonably oscillating time series.

Obtaining oscillating time series from the simulation is sensitive to the choice of parameters and many parametrizations result in populations that simply die out.

Time series are simulated using Gillespie's algorithm (Gillespie, 1977):

1.

Draw the time to the next event from an exponential distribution with rate equal to the total rate θ 1 XY + θ 2 X + θ 3 Y + θ 4 XY .

2.

Select one of the above events A, B, C, or D at random with probability proportional to its rate.

3.

Adjust the appropriate population according to the selected event, and go to 1.

The simulations using these parameter settings can yield a maximum population of approximately 300 while the context set in the lynx-hare data set has an approximate maximum population of about 80 so we scaled our simulation population by a factor of 2/7.

We also remove time series which are longer than 100 units of time, which have more than 10000 events, or where one of the populations is entirely zero.

The number of context points n for a training batch are each selected randomly from a uniform distribution between 3 and 80, and the number of target points is 150 − n. These target and context points are then sampled from the simulated series.

The Hudson's Bay lynx-hare data set has time values that range from 1845 to 1935.

However, the values supplied to the model range from 0 to 90 to remain consistent with the simulated data.

For evaluation, an interval of 18 points is removed from the the Hudson's Bay lynx-hare data set to act as a target set, while the remaining 72 points act as the context set.

This construction highlights the model's interpolation as well as its uncertainty in the presence of missing data.

Models in this setting were trained for 200 epochs with 256 batches per epoch, each batch containing 50 tasks.

For this data set, we only used the CONVCNP, as we found the CONVCNPXL to overfit.

The learning rate was set to 10 −3 , and we discretize E(Z) by evaluating 100 points per unit.

We attempted to train an ATTNCNP for comparison, but due to the nature of the synthetic data generation, many of the training series end before 90 time units, the length of the Hudson's Bay lynx-hare series.

Effectively, this means that the ATTNCNP was asked to predict outside of its training interval, a task that it struggles with, as shown in Section 5.1.

The plots in Figure 9 show that the ATTNCNP is able to learn the first part of the time series but is unable to model data outside of the first 20 or so time units.

Perhaps with more capacity and training epochs the ATTNCNP training would be more successful.

Note from Figure 3 that our model does better on the synthetic data than on the real data.

This could be due to the parameters of the Lotka-Volterra model used being a poor estimate for the real data.

Training details.

In all experiments, we sample the number of context points uniformly from U( ntotal 100 , ntotal 2 ), and the number of target points is set to n total .

The context and target points are sampled randomly from each of the 16 images per batch.

The weights are optimised using Adam (Kingma & Ba, 2015) with learning rate 5 × 10 −4 .

We use a maximum of 100 epochs, with early stopping of 15 epochs patience.

All pixel values are divided by 255 to rescale them to the [0, 1] range.

In the following discussion, we assume that images are RGB, but very similar models can be used for greyscale images or other gridded inputs (e.g. 1d time series sampled at uniform intervals).

Proposed convolutional CNP.

Unlike ATTNCNP and off-the-grid CONVCNP, on-the-grid CON-VCNP takes advantage of the gridded structure.

Namely, the target and context points can be specified in terms of the image, a context mask M c , and a target mask M t instead of sets of input-value pairs.

Although this is an equivalent formulation, it makes it more natural and simpler to implement in standard deep learning libraries.

In the following, we dissect the architecture and algorithmic steps succinctly summarized in Section 4.

Note that all the convolutional layers are actually depthwise separable (Chollet, 2017) ; this enables a large kernel size (i.e. receptive fields) while being parameter and computationally efficient.

1.

Let I denote the image.

Select all context points signal := M c I and append a density channel density := M c , which intuitively says that "there is a point at this position": [signal, density] .

Each pixel value will now have 4 channels: 3 RGB channels and 1 density channel M c .

Note that the mask will set the pixel value to 0 at a location where the density channel is 0, indicating there are no points at this position (a missing value).

2.

Apply a convolution to the density channel density = CONV θ (density) and a normalized convolution to the signal signal := CONV θ (signal)/density .

The normalized convolution makes sure that the output mostly depends on the scale of the signal rather than the number of observed points.

The output channel size is 128 dimensional.

The kernel size of CONV θ depends on the image shape and model used (Table 5) .

We also enforce element-wise positivity of the trainable filter by taking the absolute value of the kernel weights θ before applying the convolution.

As discussed in Appendix D.4, the normalization and positivity constraints do not empirically lead to improvements for on-the-grid data.

Note that in this setting, E(Z) is [signal , density ] .

3.

Now we describe the on-the-grid version of ρ(·), which we decompose into two stages.

In the first stage, we apply a CNN to [signal , density ] .

This CNN is composed of residual blocks (He et al., 2016) , each consisting of 2 convolutional layers with ReLU activations and no batch normalization.

The number of output channels in each layer is 128.

The kernel size is the same across the whole network, but depends on the image shape and model used (Table 5 ).

4.

In the second stage of ρ(·), we apply a shared pointwise MLP : R 128 → R 2C (we use the same architecture as used for the ATTNCNP decoder) to the output of the first stage at each pixel location in the target set.

Here C denotes the number of channels in the image.

The first C outputs of the MLP are treated as the means of a Gaussian predictive distribution, and the last C outputs are treated as the standard deviations.

These then pass through the positivity-enforcing function shown in Equation (8).

Figure 10: Samples from our generated Zero Shot Multi MNIST (ZSMM) data set.

In the real world, it is very common to have multiple objects in our field of view which do not interact with each other.

Yet, many image data sets in machine learning contain only a single, well-centered object.

To evaluate the translation equivariance and generalization capabilities of our model, we introduce the zero-shot multi-MNIST setting.

The training set contains all 60000 28 × 28 MNIST training digits centered on a black 56 × 56 background. (Figure 10a ).

For the test set, we randomly sample with replacement 10000 pairs of digits from the MNIST test set, place them on a black 56 × 56 background, and translate the digits in such a way that the digits can be arbitrarily close but cannot overlap (Figure 10b) .

Importantly, the scale of the digits and the image size are the same during training and testing.

D.3 ATTNCNP AND CONVCNP QUALITATIVE COMPARISON Figure 11 shows the test log-likelihood distributions of an ATTNCNP and CONVCNP model as well as some qualitative comparisons between the two.

Although most mean predictions of both models look relatively similar for SVHN and CelebA32, the real advantage of CONVCNP becomes apparent when testing the generalization capacity of both models.

Figure 12 shows CONVCNP and ATTNCNP trained on CelebA32 and tested on a downscaled version of Ellen's famous Oscar selfie.

We see that CONVCNP generalizes better in this setting.

To understand the importance of the different components of the first layer, we performed an ablation study by removing the density normalization (CONVCNP no norm.), removing the density channel (CONVCNP no dens.), removing the positivity constraints (CONVCNP no abs.), removing the positivity constraints and the normalization (CONVCNP no abs.

norm.), and replacing the fully trainable first layer by an EQ kernel similar to the continuous case (CONVCNP EQ).

Table 6 shows the following: (i) Appending a density channel helps. (ii) Enforcing the positivity constraint is only important when using a normalized convolution. (iii) Using a less expressive EQ filter does not significantly decrease performance, suggesting that the model might be learning similar filters (Appendix D.5).

Figure 13:

First filter learned by CONVCNPXL, CONVCNP, and CONVCNP EQ for all our datasets.

In the case of RGB images, the plotted filters are for the first channel (red).

Note that not all filters are of the same size.

As discussed in Appendix D.4, using a less expressive EQ filter does not significantly decrease performance.

Figure 13 shows that this happens because the fully trainable kernel learns to approximate the EQ filter.

As seen in Table 3 , a CONVCNPXL with large receptive field performs significantly worse on the ZSMM task than CONVCNP, which has a smaller receptive field.

Figure 14 shows a more detailed comparison of the models, and suggests that CONVCNPXL learns to model non-stationary behaviour, namely that digits in the training set are centred.

We hypothesize that this issue stems from the the treatment of the image boundaries.

Indeed, if the receptive field is large enough and the padding values are significantly different than the inputs to each convolutional layer, the model can learn position-dependent behaviour by "looking" at the distance from the padded boundaries.

For ZSMM, Figure 15 suggests that "circular" padding, where the padding is implied by tiling the image, helps prevent the model from learning non-stationarities, even as the size of the receptive field becomes larger.

We hypothesize that this is due to the fact that "circularly" padded values are harder to distinguish from actual values than zeros.

We have not tested the effect of padding on other datasets, and note that "circular" padding could result in other issues.

Figure 15: Effect of the receptive field size on ZSMM's log-likelihood.

The line plot shows the mean and standard deviation over 6 runs.

The blue curve corresponds to a model with zero padding, while the orange one corresponds to "circular" padding.

<|TLDR|>

@highlight

We extend deep sets to functional embeddings and Neural Processes to include translation equivariant members