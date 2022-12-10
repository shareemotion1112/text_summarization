We introduce a novel framework for generative models based on Restricted Kernel Machines (RKMs) with multi-view generation and uncorrelated feature learning capabilities, called Gen-RKM.

To incorporate multi-view generation, this mechanism uses a shared representation of data from various views.

The mechanism is flexible to incorporate both kernel-based, (deep) neural network and convolutional based models within the same setting.

To update the parameters of the network, we propose a novel training procedure which jointly learns the features and shared representation.

Experiments demonstrate the potential of the framework through qualitative evaluation of generated samples.

In the past decade, interest in generative models has grown tremendously, finding applications in multiple fields such as, generated art, on-demand video, image denoising (Vincent et al., 2010) , exploration in reinforcement learning (Florensa et al., 2018) , collaborative filtering (Salakhutdinov et al., 2007) , inpainting (Yeh et al., 2017) and many more.

Some examples of graphical models based on a probabilistic framework with latent variables are Variational Auto-Encoders (Kingma & Welling, 2014) and Restricted Boltzmann Machines (RBMs) (Smolensky, 1986; Salakhutdinov & Hinton, 2009 ).

More recently proposed models are based on adversarial training such as Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) and its many variants.

Furthermore, auto-regressive models such as Pixel Recurrent Neural Networks (PixelRNNs) (Van Den Oord et al., 2016) model the conditional distribution of every individual pixel given previous pixels.

All these approaches have their own advantages and disadvantages.

For example, RBMs perform both learning and Bayesian inference in graphical models with latent variables.

However, such probabilistic models must be properly normalized, which requires evaluating intractable integrals over the space of all possible variable configurations (Salakhutdinov & Hinton, 2009) .

Currently GANs are considered as the state-of-the-art for generative modeling tasks, producing high-quality images but are more difficult to train due to unstable training dynamics, unless more sophisticated variants are applied.

Many datasets are comprised of different representations of the data, or views.

Views can correspond to different modalities such as sounds, images, videos, sequences of previous frames, etc.

Although each view could individually be used for learning tasks, exploiting information from all views together could improve the learning quality (Pu et al., 2016; Liu & Tuzel, 2016; Chen & Denoyer, 2017) .

Also, it is among the goals of the latent variable modelling to model the description of data in terms of uncorrelated or independent components.

Some classical examples are Independent Component Analysis; Hidden Markov models (Rabiner & Juang, 1986) ; Probabilistic Principal Component Analysis (PCA) (Tipping & Bishop, 1999) ; Gaussian-Process Latent variable model (Lawrence, 2005) and factor analysis.

Hence, when learning a latent space in generative models, it becomes interesting to find a disentangled representation.

Disentangled variables are generally considered to contain interpretable information and reflect separate factors of variation in the data for e.g. lighting conditions, style, colors, etc.

The definition of disentanglement in the literature is not precise, however many believe that a representation with statistically independent variables is a good starting point (Schmidhuber, 1992; Ridgeway, 2016) .

Such representations extract information into a compact form which makes it possible to generate samples with specific characteristics (Chen et al., 2018; Bouchacourt et al., 2018; Tran et al., 2017; Chen et al., 2016) .

Additionally, these representations have been found to generalize better and be more robust against adversarial attacks (Alemi et al., 2017) .

In this work, we propose an alternative generative mechanism based on the framework of Restricted Kernel Machines (RKMs) (Suykens, 2017) , called Generative RKM (Gen-RKM).

RKMs yield a representation of kernel methods with visible and hidden units establishing links between Kernel PCA, Least-Squares Support Vector Machines (LS-SVM) (Suykens et al., 2002) and RBMs.

This framework has a similar energy form as RBMs, though there is a non-probabilistic training procedure where the eigenvalue decomposition plays the role of normalization.

Recently, Houthuys & Suykens (2018) used this framework to develop tensor-based multi-view classification models and Schreurs & Suykens (2018) showed how kernel PCA fits into this framework.

Contributions.

1) A novel multi-view generative model based on the RKM framework where multiple views of the data can be generated simultaneously.

2) Two methods are proposed for computing the pre-image of the feature vectors: with the feature map explicitly known or unknown.

We show that the mechanism is flexible to incorporate both kernel-based, (deep) convolutional neural network based models within the same setting.

3) When using explicit feature maps, we propose a training algorithm that jointly performs the feature-selection and learns the common-subspace representation in the same procedure.

4) Qualitative and quantitative experiments demonstrate that the model is capable of generating good quality images of natural objects.

Further experiments on multi-view datasets exhibit the potential of the model.

Thanks to the orthogonality of eigenvectors of the kernel matrix, the learned latent variables are uncorrelated.

This resembles a disentangled representation, which makes it possible to generate data with specific characteristics.

This paper is organized as follows.

In Section 2, we discuss the Gen-RKM training and generation mechanism when multiple data sources are available.

In Section 3, we explain how the model incorporates both kernel methods and neural networks through the use of implicit and explicit feature maps respectively.

When the feature maps are defined by neural networks, the Gen-RKM algorithm is explained in Section 4.

In Section 5, we show experimental results of our model applied on various public datasets.

Section 6 concludes the paper along with directions towards the future work.

Additional supplementary materials are given in the Appendix A.

The proposed Gen-RKM framework consists of two phases: a training phase and a generation phase which occurs one after another.

Similar to Energy-Based Models (EBMs, see LeCun et al. (2004) for details), the RKM objective function captures dependencies between variables by associating a scalar energy to each configuration of the variables.

Learning consists of finding an energy function in which the observed configurations of the variables are given lower energies than unobserved ones.

Note that the schematic representation, as shown in Figure 1 is similar to Discriminative RBMs (Larochelle & Bengio, 2008) and the objective function J t (defined below) has an energy form similar to RBMs with additional regularization terms.

The latent space dimension in the RKM setting has a similar interpretation as the number of hidden units in a restricted Boltzmann machine, where in the specific case of the RKM these hidden units are uncorrelated.

We assume a dataset

, with x i ∈ R d , y i ∈ R p comprising of N data points.

Here y i may represent an additional view of x i , e.g., an additional image from a different angle, the caption of an image or a class label.

Starting from the RKM interpretation of Kernel PCA, which gives an upper bound on the equality constrained Least-Squares Kernel PCA objective function (Suykens, 2017) , and applying the feature-maps φ 1 :

Figure 1: Gen-RKM schematic representation modeling a common subspace H between two data sources X and Y. The φ 1 , φ 2 are the feature maps (F x and F y represent the feature-spaces) corresponding to the two data sources.

While ψ 1 , ψ 2 represent the pre-image maps.

The interconnection matrices U , V model dependencies between latent variables and the mapped data sources.

the training objective function J t for generative RKM is given by 1 :

where U ∈ R d f ×s and V ∈ R p f ×s are the unknown interaction matrices, and h i ∈ R s are the latent variables modeling a common subspace H between the two input spaces X and Y (see Figure  1 ).

The derivation of this objective function is given in the Appendix A.1.

Given η 1 > 0 and η 2 > 0 as regularization parameters, the stationary points of J t are given by:

Substituting U and V in the first equation above, denoting Λ = diag{λ 1 , . . .

, λ s } ∈ R s×s with s ≤ N , yields the following eigenvalue problem:

where H = h 1 , . . .

, h N ∈ R s×N with s ≤ N is the number of selected principal components and K 1 , K 2 ∈ R N ×N are the kernel matrices corresponding to data sources 2 .

Based on Mercer's theorem (Mercer, 1909) , positive-definite kernel functions k 1 :

. .

, N forms the elements of corresponding kernel matrices.

The feature maps φ 1 and φ 2 , mapping the input data to the high-dimensional feature space (possibly infinite) are implicitly defined by kernel functions.

Typical examples of such kernels are given by the Gaussian RBF kernel

− xi−xj 2/σ just to name a few (Scholkopf & Smola, 2001 ).

However, one can also define explicit feature maps, still preserving the positive-definiteness of the kernel function by construction (Suykens et al., 2002) .

In this section, we derive the equations for the generative mechanism.

RKMs resembling energybased models, the inference consists in clamping the value of observed variables and finding configurations of the remaining variables that minimizes the energy (LeCun et al., 2004) .

Given the

Otherwise, a centered kernel matrix could be obtained using Eq. 17 (Appendix A.4).

2 While in the above section we have assumed that only two data sources (namely X and Y) are available for learning, the above procedure could be extended to multiple data-sources.

For the M views or data-sources, this yields the training problem:

learned interconnection matrices U and V , and a given latent variable h , consider the following objective function:

with an additional regularization term on data sources.

Here J g denotes the objective function for generation.

The given latent variable h can be the corresponding latent code of a training point, a newly sampled hidden unit or a specifically determined one.

Above cases correspond to generating the reconstructed visible unit, generating a random new visible unit or exploring the latent space by carefully selecting hidden units respectively.

The stationary points of J g are characterized by:

Using U and V from Eq. 2, we obtain the generated feature vectors:

To obtain the generated data, one now needs to compute the inverse images of the feature maps φ 1 (·) and φ 2 (·) in the respective input spaces, i.e., solve the pre-image problem.

We seek to find the functions

where φ 1 (x ) and φ 2 (y ) are calculated using Eq. 6.

When using kernel methods, explicit feature maps are not necessarily known.

Commonly used kernels such as the radial-basis function and polynomial kernels map the input data to a very high dimensional feature space.

Hence finding the pre-image, in general, is known to be an ill-conditioned problem (Mika et al., 1999) .

However, various approximation techniques have been proposed (Bui et al., 2019; Kwok & Tsang, 2003; Honeine & Richard, 2011; Weston et al., 2004) which could be used to obtain the approximate pre-imagex of φ 1 (x ).

In section 3.1, we employ one such technique to demonstrate the applicability in our model, and consequently generate the multi-view data.

One could also define explicit pre-image maps.

In section 3.2, we define parametric pre-image maps and learn the parameters by minimizing the appropriately defined objective function.

The next section describes the above two pre-image methods for both cases, i.e., when the feature map is explicitly known or unknown, in greater detail.

As noted in the previous section, since x may not exist, we find an approximationx.

A possible technique is shown by Schreurs & Suykens (2018) .

Left multiplying Eq. 6 by φ 1 (x i ) and φ 2 (y i ) , ∀i = 1, . . .

, N , we obtain:

where,

represents the similarities between φ 1 (x ) and training data points in the feature space, and K 1 ∈ R N ×N represents the centered kernel matrix of X .

Similar conventions follow for Y respectively.

Using the kernel-smoother method (Hastie et al., 2001) , the pre-images are given by:

wherek 1 (x i , x ) andk 2 (y i , y ) are the scaled similarities (see Eq. 8) between 0 and 1 and n r the number of closest points based on the similarity defined by kernelsk 1 andk 2 .

While using an explicit feature map, Mercer's theorem is still applicable due to the positive semidefiniteness of the kernel function by construction, thereby allowing the derivation of Eq. 3.

In the experiments, we use a set of (convolutional) neural networks as the feature maps φ θ (·).

Another (transposed convolutional) neural network is used for the pre-image map ψ ζ (·) (Dumoulin & Visin, 2016) .

The network parameters {θ, ζ} are learned by minimizing the reconstruction errors defined by L 1 (x, ψ 1 ζ 1 (φ 1 θ 1 (x))) and L 2 (y, ψ 2 ζ 2 (φ 2 θ 2 (y))).

In our experiments, we use the mean-squared

, however, in principle, one can use any other loss appropriate to the dataset.

Here φ 1 θ 1 (x i ) and φ 2 θ 2 (y i ) are computed from Eq. 6, i.e., the generated points in feature space from the subspace H.

Adding the loss function directly into the objective function J t is not suitable for minimization.

Instead, we use the stabilized objective function defined as

is the regularization constant (Suykens, 2017) .

This tends to push the objective function J t towards zero, which is also the case when substituting the solutions λ i , h i back into J t (see Appendix A.3 for details).

The combined training objective is given by:

where c acc ∈ R + is a regularization constant to control the stability with reconstruction accuracy.

In this way, we combine feature-selection and subspace learning within the same training procedure.

There is also an intuitive connection between Gen-RKM and autoencoders.

Namely, the properties of kernel PCA resemble the objectives of the 3 variations of an autoencoder: standard (Kramer, 1991), VAE (Kingma & Welling, 2014) and β-VAE .

1) Similar to an autoencoder, Gen-RKM minimizes the reconstruction error in the loss function (see Eq. 9), where kernel PCA which acts as a denoiser (the information is compressed in the principal components).

2) By interpreting kernel PCA within the LS-SVM setting (Suykens et al., 2002) , the PCA analysis can take the interpretation of a one-class modeling problem with zero target value around which one maximizes the variance (Suykens et al., 2003) .

When choosing a good feature map, one expects the latent variables to be normally distributed around zero.

This property resembles the added regularization term in the objective of the VAE (Kingma & Welling, 2014) , which is expressed as the Kullback-Leibler divergence between the encoder's distribution and a unit Gaussian as a prior on the latent variables.

3) Kernel PCA gives uncorrelated components in feature space.

While it was already shown that PCA does not give a good disentangled representation for images (Eastwood & Williams, 2018; .

Hence by designing a good kernel (through appropriate feature-maps) and doing kernel PCA, it is possible to get a disentangled representation for images as we show on the example in Figure 5 .

The uncorrelated components enhances the interpretation of the model.

Based on the previous analysis, we propose a novel algorithm, called the Gen-RKM algorithm, combining kernel learning and generative models.

We show that this procedure is efficient to train and evaluate.

It is also scalable to large datasets when using explicit feature maps.

The training procedure simultaneously involves feature selection, common-subspace learning and pre-image map learning.

This is achieved via an optimization procedure where one iteration involves an eigendecomposition of the kernel matrix which is composed of the features from various views (see Eq. 3).

The latent variables are given by the eigenvectors, which are then passed via a pre-image map to reconstruct the sample.

Figure 1 shows a schematic representation of the algorithm when two data sources are available.

Thanks to training in m mini-batches, this procedure is scalable to large datasets (sample size N ) with training time scaling super-linearly with T m = c .

While using neural networks as feature maps, d f and p f correspond to the number of neurons in the output layer, which are chosen as hyperparameters by the practitioner.

Eigendecomposition of this smaller covariance matrix would yield U and V as eigenvectors (see Eq. 10 and Appendix A.2 for detailed derivation), where computing the h i involves only matrix-multiplication which is readily parallelizable on modern GPUs:

Algorithm 1 Gen-RKM

, η1, η2, feature map φj(·) -explicit or implicit via kernels kj(·, ·), for j ∈ {1, 2} Output: Generated data x , y 1: procedure TRAIN 2: if φj(·) = Implicit then 3:

Hyperparameters: kernel specific 4: Solve Eq. 3 5:

Select s principal components 6:

else if φj(·) = Explicit then 7:

while not converged do 8:

{x, y} ← {Get mini-batch} 9:

φ1(x) ← x; φ2(y)

← y 10:

do steps 4-5 11:

{φ1(x), φ2(y)} ← h (Eq. 6) 12:

{x, y} ← {ψ1(φ1(x)), ψ2(φ2(y))} 13:

∆θ1 ∝ −∇ θ 1 Jc; ∆θ2 ∝

−∇ θ 2 Jc 14: if φj(·) = Implicit then 4:

Hyperparameter: nr 5:

Compute kx * , ky * (Eq. 7) 6:

Getx,ŷ (Eq. 8) 7:

else if φj(·) = Explicit then 8:

do steps 11-12 9:

end if 10: end procedure

To demonstrate the applicability of the proposed framework and algorithm, we trained the Gen-RKM model on a variety of datasets commonly used to evaluate generative models: MNIST (LeCun & Cortes, 2010), Fashion-MNIST (Xiao et al., 2017) , CIFAR-10 (Krizhevsky, 2009), CelebA (Liu et al., 2015) , Dsprites and Teapot (Eastwood & Williams, 2018) .

The experiments were performed using both the implicit feature map defined by a Gaussian kernel and parametric explicit feature maps defined by deep neural networks, either convolutional or fully connected.

As explained in Section 2, in case of kernel methods, training only involves constructing the kernel matrix and solving the eigenvalue problem in Eq. 3.

In our experiments, we fit a Gaussian mixture model (GMM) with l components to the latent variables of the training set, and randomly sample a new point h for generating views using a kernel smoother.

In case of explicit feature maps, we define φ 1 θ 1 and ψ 1 ζ 1 as convolution and transposed-convolution neural networks, respectively (Dumoulin & Visin, 2016) ; and φ 2 θ 2 and ψ 1 ζ 2 as fully-connected networks.

The particular architecture details are outlined in Table 3 in the Appendix.

The training procedure in case of explicitly defined maps consists of minimizing J c using the Adam optimizer (Kingma & Ba, 2014) to update the weights and biases.

To speed-up learning, we subdivided the datasets into m mini-batches, and within each iteration of the optimizer, Eq. 3 is solved to update the value of H. Information on the datasets and hyperparameters used for the experiments is given in Table 4 in the Appendix.

Qualitative examples: Figure 2 shows the generated images using a convolutional neural network and transposed-convolutional neural network as the feature map and pre-image map respectively.

The first column in yellow-boxes shows the training samples and the second column on the right shows the reconstructed samples.

The other images shown are generated by random sampling from a GMM over the learned latent variables.

Notice that the reconstructed samples are of better quality visually than the other images generated by random sampling.

To elucidate that the model has not merely memorized the training examples, we show the generated images via bilinear-interpolations in the latent space in 2e and 2f.

Comparison: We compare the proposed model with the standard VAE (Kingma & Welling, 2014) .

For a fair comparison, the models have the same encoder/decoder architecture, optimization parameters and are trained until convergence, where the details are given in Table 3 .

We evaluate the performance qualitatively by comparing reconstruction and random sampling, the results are shown in Figure 8 in the Appendix.

In order to quantitatively assess the quality of the randomly generated samples, we use the Fréchet Inception Distance (FID) introduced by Heusel et al. (2017) .

The results are reported in Table 1 .

Experiments were repeated for different latent-space dimensions (h dim ), and we observe empirically that FID scores are better for the Gen-RKM.

This is confirmed by the qualitative evaluation in Table 8 , where the VAE generates smoother images.

An interesting trend could be noted that as the dimension of latent-space is increased, VAE gets better at generating images whereas the performance of Gen-RKM decreases slightly.

This is attributed to the eigendecomposition of the kernel matrix whose eigenvalue spectrum decreases rapidly depicting that most information is captured in few principal components, while the rest is noise.

The presence of noise hinders the convergence of the model.

It is therefore important to select the number of latent variables proportionally to the size of the mini-batch and the corresponding spectrum of the kernel matrix (the diversity within a mini-batch affects the eigenvalue spectrum of the kernel matrix).

Multi-view Generation: Figures 3 & 4 demonstrate the multi-view generative capabilities of the model.

In these datasets, labels or attributes are seen as another view of the image that provides extra information.

One-hot encoding of the labels was used to train the model.

Figure 4a shows the generated images and labels when feature maps are only implicitly known i.e. through a Gaussian kernel.

Figures 4b, 4c shows the same when using fully-connected networks as parametric functions to encode and decode labels.

We can see that both the generated image and the generated label matches in most cases, albeit not all.

Qualitative examples: The latent variables are uncorrelated, which gives an indication that the model could resemble a disentangled representation.

This is confirmed by the empirical evidence on Figure 5 , where we explore the uncorrelated features learned by the models on the Dsprites and celebA dataset.

In our experiments, the Dsprites training dataset comprised of 32 × 32 positions of oval and heart-shaped objects.

The number of principal components chosen were 2 and the goal was to findout whether traversing along the eigenvectors, corresponds to traversing the generated im-age in one particular direction while preserving the shape of the object.

Rows 1 and 2 of Figure 5 show the reconstructed images of an oval while moving along first and second principal component respectively.

Notice that the first and second components correspond to the y and x positions respectively.

Rows 3 and 4 show the same for hearts.

On the celebA dataset, we train the Gen-RKM with 15 components.

Rows 5 and 6 shows the reconstructed images while traversing along the principal components.

When moving along the first component from left-to-right, the hair-color of the women changes, while preserving the face structure.

Whereas traversal along the second component, transforms a man to woman while preserving the orientation.

When the number of principal components were 2 while training, the brightness and background light-source corresponds to the two largest variances in the dataset.

Also notice that, the reconstructed images are more blurry due to the selection of less number of components to model H.

Comparison: To quantitatively assess disentanglement performance, we compare Gen-RKM with VAE (Kingma & Welling, 2014) and beta-VAE on the Dsprites and Teapot datasets (Eastwood & Williams, 2018) .

The models have the same encoder/decoder architecture, optimization parameters and are trained until convergence, where the details are given in Table 3 .

The performance is measured using the proposed framework 3 of Eastwood & Williams (2018) , which gives 3 measures: disentanglement, completeness and informativeness.

The results are depicted in Table 2 .

Gen-RKM has good performance on the Dsprites dataset when the latent space dimension is equal to 2.

This is expected as the number of disentangled generating factors in the dataset is also equal to 2, hence there are no noisy components in the kernel PCA hindering the convergence.

The opposite happens in the case h dim = 10, where noisy component are present.

The above is confirmed by the Relative Importance Matrix on Figure 6 in the Appendix, where the 2 generating factors are well separated in the latent space of the Gen-RKM.

For the Teapot dataset, Gen-RKM has good performance when h dim = 10.

More components are needed to capture all variations in the dataset, where the number of generating factors is now equal to 5.

In the other cases, Gen-RKM has a performance comparable to the others.

The paper proposes a novel framework, called Gen-RKM, for generative models based on RKMs with extensions to multi-view generation and learning uncorrelated representations.

This allows for a mechanism where the feature map can be implicitly defined using kernel functions or explicitly by (deep) neural network based methods.

When using kernel functions, the training consists of only solving an eigenvalue problem.

In the case of a (convolutional) neural network based explicit feature map, we used (transposed) networks as the pre-image functions.

Consequently, a training procedure was proposed which involves joint feature-selection and subspace learning.

Thanks to training in mini-batches and capability of working with covariance matrices, the training is scalable to large datasets.

Experiments on benchmark datasets illustrate the merit of the proposed framework for generation quality as well as disentanglement.

Extensions of this work consists of adapting the model to more advanced multi-view datatsets involving speech, images and texts; further analysis on other feature maps, pre-image methods, loss-functions and uncorrelated feature learning.

Finally, this paper has demonstrated the applicability of the Gen-RKM framework, suggesting new research directions to be worth exploring.

, where Suykens et al., 2002) for the two data sources can be written as:

where U ∈ R d×s and V ∈ R p×s are the interconnection matrices.

Using the notion of conjugate feature duality introduced in Suykens (2017), the error variables e i are conjugated to latent variables h i using:

which is also known as the Fenchel-Young inequality for the case of quadratic functions (Rockafellar, 1974) .

By eliminating the variables e i from Eq. 11 and using Eq. 12, we obtain the Gen-RKM training objective function:

A.2 KERNEL PCA IN THE PRIMAL From Eq. 2, eliminating the variables h i yields the following:

Denote

. .

, λ s } ∈ R s×s with s ≤ N .

Now, composing the above equations in matrix form, we get the following eigen-decomposition problem:

Here the size of the covariance matrix is

The latent variables h i can be computed using Eq. 2, which simply involves matrix multiplications.

A.3 STABILIZING THE OBJECTIVE FUNCTION Proposition 1.

All stationary solutions for H,Λ in Eq. 3 of J t lead to J t = 0.

Proof.

Let λ i , h i are given by Eq. 3.

Using Eq. 2 to substitute V and U in Eq. 1 yields:

From Eq. 3, we get:

Proposition 2.

Let J(x) : R N − → R be a smooth function, for all x ∈ R N and for c ∈ R >0 , definē

2 .

Assuming (1 + cJ(x)) = 0, then x is the stationary points ofJ(x) iff

x is the stationary point for J(x).

Proof.

Let x be a stationary point of J(x), meaning that ∇J(x ) = 0.

The stationary points for J(x) can be obtained from:

It is easy to see from Eq. 2 that if x = x * , ∇J(x * ) = 0, we have that dJ dx x * = 0, meaning that all the stationary points of J(x) are stationary points ofJ(x).

To show the other way, let x be stationary point ofJ(x) i.e. ∇J(x ) = 0.

Assuming (1 + cJ(x )) = 0, then from Eq. 16 for all c ∈ R >0 , we have

Based on the above propositions, we stabilize our original objective function Eq. 1 to keep it bounded and hence is suitable for minimization with Gradient-descent methods.

Without the reconstruction errors, the stabilized objective function is

Since the derivatives of J t are given by Eq. 2, the stationary points of J are:

assuming 1 + c stab J t = 0.

Elimination of V and U yields 1 η1 K 1 + 1 η2 K 2 H = H Λ, which is indeed the same solution for c stab = 0 in Eq. 1 and Eq. 3.

Centering of the kernel matrix is done by the following equation:

where 1 denotes an N -dimensional vector of ones and K is either K 1 or K 2 .

See Table 3 and 4 for details on model architectures, datasets and hyperparameters used in this paper.

The PyTorch library in Python was used as the programming language with a 8GB NVIDIA QUADRO P4000 GPU.

Random Generation CelebA Figure 8 : Comparing Gen-RKM and standard VAE for reconstruction and generation quality.

In reconstruction MNIST and reconstruction CelebA, uneven columns correspond to the original image, even columns to the reconstructed image.

<|TLDR|>

@highlight

Gen-RKM: a novel framework for generative models using Restricted Kernel Machines with multi-view generation and uncorrelated feature learning.