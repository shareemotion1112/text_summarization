It has been shown that using geometric spaces with non-zero curvature instead of plain Euclidean spaces with zero curvature improves performance on a range of Machine Learning tasks for learning representations.

Recent work has leveraged these geometries to learn latent variable models like Variational Autoencoders (VAEs) in spherical and hyperbolic spaces with constant curvature.

While these approaches work well on particular kinds of data that they were designed for e.g.~

tree-like data for a hyperbolic VAE, there exists no generic approach unifying all three models.

We develop a Mixed-curvature Variational Autoencoder, an efficient way to train a VAE whose latent space is a product of constant curvature Riemannian manifolds, where the per-component curvature can be learned.

This generalizes the Euclidean VAE to curved latent spaces, as the model essentially reduces to the Euclidean VAE if curvatures of all latent space components go to 0.

Generative models are a growing area of unsupervised learning, that aim to model the data distribution p(x) over data points x in a high-dimensional space X (Doersch, 2016) , usually a subset of a high-dimensional Euclidean space R n , with all the associated benefits: a naturally definable scalar product, vector addition, and others.

Yet, many types of data have a strongly non-Euclidean latent structure (Bronstein et al., 2017) , like the set of human-interpretable images.

They are usually thought to live on a "natural image manifold" (Zhu et al., 2016) , a lower-dimensional subset of the space in which they are represented.

On this continuous manifold, one finds all the images that humans can interpret using their visual system.

By moving along the manifold, we can continuously change the content and appearance of interpretable images.

As mentioned in Nickel & Kiela (2017) , changing the geometry of the underlying latent space enables us to represent some data better than is possible in the equivalent Euclidean space.

Motivated by these observations, a range of methods to learn representations in different spaces of constant curvature have recently been introduced: learning embeddings in spherical spaces (Batmanghelich et al., 2016) , hyperbolic spaces (Nickel & Kiela, 2017; Tifrea et al., 2019; Sala et al., 2018) , and even in products of these spaces (Gu et al., 2019; Anonymous, 2020) .

By using a combination of different constant curvature spaces, it aims to match the underlying geometry of the data even closer than the others.

However, an open question that remains, is how to choose the dimensionality of partial spaces and their curvatures.

A popular approach to generative modeling is the Variational Autoencoder (Kingma & Welling, 2014, VAE) .

VAEs provide us with a way to sidestep the intractability of marginalizing a joint probability model of the input and latent space p(x, z) while allowing for a prior p(z) on the latent space.

Recently, variants of the VAE have been introduced for spherical (Davidson et al., 2018; Xu & Durrett, 2018) and hyperbolic (Mathieu et al., 2019; Nagano et al., 2019 ) latent spaces.

Our approach is a generalization of the VAE to products of constant curvature spaces, which have the advantage that we can obtain a better reduction in dimensionality while not making optimization of the model significantly more complex.

The resulting latent space is then a "non-constantly" curved manifold in an ambient Euclidean space.

Modeling the latent space as a single constant curvature manifold limits the flexibility of the space to assume a shape similar to that of the hypothetical intrinsic manifold.

Our contributions are the following: (i) we develop a principled framework for manipulating representations and modeling probability distributions in products of constant curvature spaces that smoothly transitions across different curvature signs, (ii) we show how to generalize Variational Au-toencoders to learn latent representations on products of constant curvature spaces with generalized Gaussian-like priors, and (iii) our approaches outperform current benchmarks on a synthetic tree dataset (Mathieu et al., 2019) and image reconstruction on MNIST (LeCun, 1998) , Omniglot (Lake et al., 2015) , and CIFAR (Krizhevsky, 2009) for some latent space dimensions.

To define constantly curved spaces, we first need to define the notion of sectional curvature K(?? x ) of two linearly independent vectors in the tangent space at a point x ??? M spanning a two-dimensional plane ?? x (Berger, 2012) .

Since we deal with constant curvature spaces where all the sectional curvatures are equal, we denote a manifold's curvature as K. Instead of curvature K, we sometimes use the generalized notion of a radius: R = 1/ |K|.

There are three different types of manifolds M we can define with respect to the sign of the curvature: a positively curved space, a "flat" space, and a negatively curved space.

Common realizations of those manifolds are the hypersphere S K , the Euclidean space E, and the hyperboloid H K :

x, x 2 = 1/K}, for K > 0

x, x L = 1/K}, for K < 0 where ??, ?? 2 is the standard Euclidean inner product, and ??, ?? L is the Lorentz inner product,

x, y L = ???x 1 y 1 + n+1 i=2

x i y i ???x, y ??? R n+1 .

We will need to define the exponential map, logarithmic map, and parallel transport in all spaces we consider.

The exponential map in Euclidean space is defined as exp x (v) = x + v, for all x ??? E n and v ??? T x E n .

Its inverse, the logarithmic map is log x (y) = y ??? x, for all x, y ??? E n .

Parallel transport in Euclidean space is simply an identity PT x???y (v) = v, for all x, y ??? E n and v ??? T x E n .

An overview of these operations in the hyperboloid H n K and the hypersphere S n K can be found in Table 1 .

For more details, refer to Petersen et al. (2006) , Cannon et al. (1997) , or Appendix A.

The above spaces are enough to cover any possible value of the curvature, and they define all the necessary operations we will need to train VAEs in them.

However, both the hypersphere and the hyperboloid have an unsuitable property, namely the non-convergence of the norm of points as the curvature goes to 0.

Both spaces grow as K ??? 0 and become locally "flatter", but to do that, their points have to go away from the origin of the coordinate space 0 to be able to satisfy the manifold's definition.

A good example of a point that diverges is the origin of the hyperboloid (or a pole of the hypersphere) ?? 0 = (1/ |K|, 0, . . . , 0)

T .

In general, we can see that ?? 0 2 = 1/K K???0 ???????????? ?????. Additionally, the distance and the metric tensors of these spaces do not converge to their Euclidean variants as K ??? 0, hence the spaces themselves do not converge to R d .

This makes both of these spaces unsuitable for trying to learn sign-agnostic curvatures.

Luckily, there exist well-defined non-Euclidean spaces that inherit most properties from the hyperboloid and the hypersphere, yet do not have these properties -namely, the Poincar?? ball and the projected sphere, respectively.

We obtain them by applying stereographic conformal projections, meaning that angles are preserved by this transformation.

Since the distance function on the hyperboloid and hypersphere only depend on the radius and angles between points, they are isometric.

We first need to define the projection function ?? K .

For a point (??; x T ) T ??? R n+1 and curvature K ??? R, where ?? ??? R, x, y ??? R ; 2y

Under review as a conference paper at ICLR 2020

The formulas correspond to the classical stereographic projections defined for these models (Lee, 1997) .

Note that both of these projections map the point ?? 0 = (1/ |K|, 0, . . . , 0) in the original space to ?? 0 = 0 in the projected space, and back.

Since the stereographic projection is conformal, the metric tensors of both spaces will be conformal.

In this case, the metric tensors of both spaces are the same, except for the sign of K: g

2 g E , for all x in the respective manifold (Ganea et al., 2018a) , and g E y = I for all y ??? E. The conformal factor ?? K x is then defined as ?? K x = 2/(1 + K x 2 2 ).

Among other things, this form of the metric tensor has the consequence that we unfortunately cannot define a single unified inner product in all tangent spaces at all points.

The inner product at x ??? M has the form of

We can now define the two models corresponding to K > 0 and K < 0.

The curvature of the projected manifold is the same as the original manifold.

An n-dimensional projected hypersphere

, along with the induced distance function.

The n-dimensional Poincar?? ball P n K (also called the Poincar?? disk when n = 2) for a given curvature K < 0 is defined as

, with the induced distance function.

An important analogy to vector spaces (vector addition and scalar multiplication) in non-Euclidean geometry is the notion of gyrovector spaces (Ungar, 2008) .

Both of the above spaces D K and P K (jointly denoted as M K ) share the same structure, hence they also share the following definition of addition.

The M??bius addition ??? K of x, y ??? M K (for both signs of K) is defined as

We can, therefore, define "gyrospace distances" for both of the above spaces, which are alternative curvature-aware distance functions

These two distances are equivalent to their non-gyrospace variants d M (x, y) = d Mgyr (x, y), as is shown in Theorem A.4 and its hypersphere equivalent.

Additionally, Theorem A.5 shows that

which means that the distance functions converge to the Euclidean distance function as K ??? 0.

We can notice that most statements and operations in constant curvature spaces have a dual statement or operation in the corresponding space with the opposite curvature sign.

The notion of duality is one which comes up very often and in our case is based on Euler's formula e ix = cos(x) + i sin(x) and the notion of principal square roots

This provides a connection between trigonometric, hyperbolic trigonometric, and exponential functions.

Thus, we can convert all the hyperbolic formulas above to their spherical equivalents and vice-versa.

Since Ganea et al. (2018a) and Tifrea et al. (2019) used the same gyrovector spaces to define an exponential map, its inverse logarithmic map, and parallel transport in the Poincar?? ball, we can reuse them for the projected hypersphere by applying the transformations above, as they share the same formalism.

For parallel transport, we additionally need the notion of gyration (Ungar, 2008) gyr [x, y]

Parallel transport in the both the projected hypersphere and the Poincar?? ball then is PT

we can summarize all the necessary operations in all manifolds compactly in Table 1 and Table 2 .

Previously, our space consisted of only one manifold of varying dimensionality and fixed curvature.

Like Gu et al. (2019) , we propose learning latent representations in products of constant curvature spaces, contrary to existing VAE approaches which are limited to a single Riemannian manifold.

Our latent space M consists of several component spaces

Ki , where n i is the dimensionality of the space, K i is its curvature, and M ??? {E, S, D, H, P} is the model choice.

Even though all components have constant curvature, the resulting manifold M has non-constant curvature.

Its distance function decomposes based on its definition d

where

Ki , corresponding to the part of the latent space representation of x belonging to M ni Ki .

All other operations we defined on our manifolds are element-wise.

Therefore, we again decompose the representations into parts x (i) , apply the operation on that part

Ki (x (i) ) and concatenate the resulting parts backx = k i=1x

The signature of the product space, i.e. its parametrization, has several degrees of freedom per component: (i) the model M, (ii) the dimensionality n i , and (iii) the curvature K i .

We need to select all of the above for every component in our product space.

To simplify, we use a shorthand notation for repeated components:

Ki .

In Euclidean spaces, the notation is redundant.

For n 1 , . . .

, n k ??? Z, such that

However, the equality does not hold for the other considered manifolds.

This is due to the additional constraints posed on the points in the definitions of individual models of curved spaces.

To be able to train Variational Autoencoders, we need to chose a probability distribution p as a prior and a corresponding posterior distribution family q. Both of these distributions have to be differentiable with respect to their parametrization, they need to have a differentiable KullbackLeiber (KL) divergence, and be "reparametrizable" (Kingma & Welling, 2014) .

For distributions where the KL does not have a closed-form solution independent on z, or where this integral is too hard to compute, we can estimate it using Monte Carlo estimation

where z (l) ??? q for all l = 1, . . .

, L. The Euclidean VAE uses a natural choice for a prior on its latent representations -the Gaussian distribution (Kingma & Welling, 2014) .

Apart from satisfying the requirements for a VAE prior and posterior distribution, the Gaussian distribution has additional properties, like being the maximum entropy distribution for a given variance (Jaynes, 1957).

There exist several fundamentally different approaches to generalizing the Normal distribution to Riemannian manifolds.

We discuss the following three generalizations based on the way they are constructed (Mathieu et al., 2019) .

Wrapping This approach leverages the fact that all manifolds define a tangent vector space at every point.

We simply sample from a Gaussian distribution in the tangent space at ?? 0 with mean 0, and use parallel transport and the exponential map to map the sampled point onto the manifold.

The PDF can be obtained using the multivariate chain rule if we can compute the determinant of the Jacobian of the parallel transport and the exponential map.

This is very computationally effective at the expense of losing some theoretical properties.

Restriction The "Restricted Normal" approach is conceptually antagonal -instead of expanding a point to a dimensionally larger point, we restrict a point of the ambient space sampled from a Gaussian to the manifold.

The consequence is that the distributions constructed this way are based on the "flat" Euclidean distance.

An example of this is the von Mises-Fisher (vMF) distribution (Davidson et al., 2018) .

A downside of this approach is that vMF only has a single scalar covariance parameter ??, while other approaches can parametrize covariance in different dimensions separately.

Maximizing entropy Assuming a known mean and covariance matrix, we want to maximize the entropy of the distribution (Pennec, 2006) .

This approach is usually called the Riemannian Normal distribution.

Mathieu et al. (2019) derive it for the Poincar?? ball, and Hauberg (2018) derive the Spherical Normal distribution on the hypersphere.

Maximum entropy distributions resemble the Gaussian distribution's properties the closest, but it is usually very hard to sample from such distributions, compute their normalization constants, and even derive the specific form.

Since the gains for VAE performance using this construction of Normal distributions over wrapping is only marginal, as reported by Mathieu et al. (2019) , we have chosen to focus on Wrapped Normal distributions.

To summarize, Wrapped Normal distributions are very computationally efficient to sample from and also efficient for computing the log probability of a sample, as detailed by Nagano et al. (2019) .

The Riemannian Normal distributions (based on geodesic distance in the manifold directly) could also be used, however they are more computationally expensive for sampling, because the only methods available are based on rejection sampling (Mathieu et al., 2019) .

First of all, we need to define an "origin" point on the manifold, which we will denote as ?? 0 ??? M K .

What this point corresponds to is manifold-specific: in the hyperboloid and hypersphere it corresponds to the point ?? 0 = (1/ |K|, 0, . . . , 0)

T , and in the Poincar?? ball, projected sphere, and Euclidean space it is simply ?? 0 = 0, the origin of the coordinate system.

The log-probability of samples can be computed by the reverse procedure:

The distribution can be applied to all manifolds that we have introduced.

The only differences are the specific forms of operations and the log-determinant in the PDF.

The specific forms of the log-PDF for the four spaces H, S, D, and P are derived in Appendix B. All variants of this distribution are reparametrizable, differentiable, and the KL can be computed using Monte Carlo estimation.

As a consequence of the distance function and operations convergence theorems for the Poincar?? ball A.5 (analogously for the projected hypersphere), A.17, A.18, and A.20 , we see that the Wrapped Normal distribution converges to the Gaussian distribution as K ??? 0.

To be able to learn latent representations in Riemannian manifolds instead of in R d as above, we only need to change the parametrization of the mean and covariance in the VAE forward pass, and the choice of prior and posterior distributions.

The prior and posterior have to be chosen depending on the chosen manifold and are essentially treated as hyperparameters of our VAE.

Since we have defined the Wrapped Normal family of distributions for all spaces, we can use WN (?? 0 , ?? 2 I) as the posterior family, and WN (?? 0 , I) as the prior distribution.

The forms of the distributions depend on the chosen space type.

The mean is parametrized using the exponential map exp In experiments, we sometimes use vMF(??, ??) for the hypersphere S n K (or a backprojected variant of vMF for D n K ) with the associated hyperspherical uniform distribution U (S n K ) as a prior (Davidson et al., 2018) , or the Riemannian Normal distribution RN (??, ?? 2 ) and the associated prior RN ?? 0 , 1 for the Poincare ball P n K (Mathieu et al., 2019) .

We have already seen approaches to learning VAEs in products of spaces of constant curvature.

However, we can also change the curvature constant in each of the spaces during training.

The individual spaces will still have constant curvature at each point, we just allow changing the constant in between training steps.

To differentiate between these training procedures, we will call them fixed curvature and learnable curvature VAEs respectively.

The motivation behind changing curvature of non-Euclidean constant curvature spaces might not be clear, since it is apparent from the definition of the distance function in the hypersphere and hyperboloid d(x, y) = R ?? ?? x,y , that the distances between two points that stay at the same angle only get rescaled when changing the radius of the space.

Same applies for the Poincar?? ball and the projected spherical space.

However, the decoder does not only depend on pairwise distances, but rather on the specific positions of points in the space.

It can be conjectured that the KL term of the ELBO indeed is only "rescaled" when we change the curvature, however, the reconstruction process is influenced in non-trivial ways.

Since that is hard to quantify and prove, we devise a series of practical experiments to show overall model performance is enhanced when learning curvature.

Fixed curvature VAEs In fixed curvature VAEs, all component latent spaces have a fixed curvature that is selected a priori and fixed for the whole duration of the training procedure, as well as during evaluation.

For Euclidean components it is 0, for positively or negatively curved spaces any positive or negative number can be chosen, respectively.

For stability reasons, we select curvature values from the range [0.25, 1.0], which corresponds to radii in [1.0, 2.0].

The exact curvature value does not have a significant impact on performance when training a fixed curvature VAE, as motivated by the distance rescaling remark above.

In the following, we refer to fixed curvature components with a constant subscript, e.g. H n 1 .

Learnable curvature VAEs In all our manifolds, we can differentiate the ELBO with respect to the curvature K. This enables us to treat K as a parameter of the model and learn it using gradientbased optimization, exactly like we learn the encoder/decoder maps in a VAE.

Learning curvature directly is badly conditioned -we are trying to learn one scalar parameter that influences the resulting decoder and hence the ELBO quite heavily.

Empirically, we have found that Stochastic Gradient Descent works well to optimize the radius of a component.

We constrain the radius to be strictly positive in all non-Euclidean spaces by applying a ReLU activation function before we use it in operations.

Universal curvature VAEs However, we must still a priori select the "partitioning" of our latent space -the number of components and for each of them select the dimension and at least the sign of the curvature of that component (signature estimation).

The simplest approach would be to just try all possibilities and compare the results on a specific dataset.

This procedure would most likely be optimal, but does not scale well.

To eliminate this, we propose an approximate method -we partition our space into 2-dimensional components (if the number of dimensions is odd, one component will have 3 dimensions).

We initialize all of them as Euclidean components and train for half the number of maximal epochs we are allowed.

Then, we split the components into 3 approximately equal-sized groups and make one group into hyperbolic components, one into spherical, and the last remains Euclidean.

We do this by changing the curvature of a component by a very small .

We then train just the encoder/decoder maps for a few epochs to stabilize the representations after changing the curvatures.

Finally, we allow learning the curvatures of all non-Euclidean components and train for the rest of the allowed epochs.

The method is not completely general, as it never uses components bigger than dimension 2, but the approximation has empirically performed satisfactorily.

We also do not constrain the curvature of the components to a specific sign in the last stage of training.

Therefore, components may change their type of space from a positively curved to a negatively curved one, or vice-versa.

Because of the divergence of points as K ??? 0 for the hyperboloid and hypersphere, the universal curvature VAE assumes the positively curved space is D and the negatively curved space is P. In all experiments, this universal approach is denoted as U n .

For our experiments, we use four datasets: (i) Branching diffusion process (Mathieu et al., 2019, BDP) -a synthetic tree-like dataset with injected noise, (ii) Dynamically-binarized MNIST digits (LeCun, 1998) -we binarize the images similarly to Burda et al. (2016); Salakhutdinov & Murray (2008) : the training set is binarized dynamically (uniformly sampled threshold per-sample bin(

, and the evaluation set is done with a fixed binarization (x > 0.5), (iii) Dynamically-binarized Omniglot characters (Lake et al., 2015) downsampled to 28 ?? 28 pixels, and (iv) CIFAR-10 (Krizhevsky, 2009).

All models in all datasets are trained with early stopping on training ELBO with a lookahead of 50 epochs and a warmup of 100 epochs (Bowman et al., 2016) .

All BDP models are trained for a 1000 epochs, MNIST and Omniglot models are trained for 300 epochs, and CIFAR for 200 epochs.

We compare models with a given latent space dimension using marginal log-likelihood with importance sampling (Burda et al., 2016) with 500 samples, except for CIFAR, which uses 50 due to memory constraints.

In all tables, we denote it as LL.

We run all experiments at least 3 times to get an estimate of variance when using different initial values.

In all the BDP, MNIST, and Omniglot experiments below, we use a simple feed-forward encoder and decoder architecture consisting of a single dense layer with 400 neurons and element-wise ReLU activation.

Since all the VAE parameters {??, ??} live in Euclidean manifolds, we can use standard gradient-based optimization methods.

Specifically, we use the Adam (Kingma & Ba, 2015) optimizer with a learning rate of 10 ???3 and standard settings for ?? 1 = 0.9, ?? 2 = 0.999, and = 10 ???8 .

For the CIFAR encoder map, we use a simple convolutional neural networks with three convolutional layers with 64, 128, and 512 channels respectively.

For the decoder map, we first use a dense layer

of dimension 2048, and then three consecutive transposed convolutional layers with 256, 64, and 3 channels.

All layers are followed by a ReLU activation function, except for the last one.

All convolutions have 4 ?? 4 kernels with stride 2, and padding of size 1.

The first 10 epochs for all models are trained with a fixed curvature starting at 0 and increasing in absolute value each epoch.

This corresponds to a burn-in period similarly to Nickel & Kiela (2017) .

For learnable curvature approaches we then use Stochastic Gradient Descent with learning rate 10 ???4 and let the optimizers adjust the value freely, for fixed curvature approaches it stays at the last burn-in value.

All our models use the Wrapped Normal distribution, or equivalently Gaussian in Euclidean components, unless specified otherwise.

All fixed curvature components are denoted with a M 1 or M ???1 subscript, learnable curvature components do not have a subscript.

The observation model for the reconstruction loss term were Bernoulli distributions for MNIST and Omniglot, and standard Gaussian distributions for BDP and CIFAR.

As baselines, we train VAEs with spaces that have a fixed constant curvature, i.e. assume a single Riemannian manifold (potentially a product of them) as their latent space.

It is apparent that our models with a single component, like S (2019), and E n is equivalent to the Euclidean VAE.

In the following, we present a selection of all the obtained results.

For more information see Appendix E. Bold numbers represent values that are particularly interesting.

Since the Riemannian Normal and the von MisesFischer distribution only have a spherical covariance matrix, i.e. a single scalar variance parameter per component, we evaluate all our approaches with a spherical covariance parametrization as well.

Binary diffusion process For the BDP dataset and latent dimension 6 (Table 3) , we observe that all VAEs that only use the von Mises-Fischer distribution perform worse than a Wrapped Normal.

However, when a VMF spherical component was paired with other component types, it performed better than if a Wrapped Normal spherical component was used instead.

Riemannian Normal VAEs did very well on their own -the fixed Poincar?? VAE (RN P 2 ???1 ) 3 obtains the best score.

It did not fare as well when we tried to learn curvature with it, however.

An interesting observation is that all single-component VAEs M 6 performed worse than product VAEs (M 2 ) 3 when curvature was learned, across all component types.

Our universal curvature VAE (U 2 ) 3 managed to get better results than all other approaches except for the Riemannian Normal baseline, but it is within the margin of error of some other models.

It also outperformed its singlecomponent variant U 6 .

However, we did not find that it converged to specific curvature values, only that they were in the approximate range of (???0.1, +0.1).

Dynamically-binarized MNIST reconstruction On MNIST (Table 3 ) with spherical covariance, we noticed that VMF again under-performed Wrapped Normal, except when it was part of a product like E 2 ??H 2 ??(vMF S 2 ).

When paired with another Euclidean and a Riemannian Normal Poincar?? disk component, it performed well, but that might be because the RN P ???1 component achieved best results across the board on MNIST.

It achieved good results even compared to diagonal covariance VAEs on 6-dimensional MNIST.

Several approaches are better than the Euclidean baseline.

That applies mainly to the above mentioned Riemannian Normal Poincar?? ball components, but also S 6 both with Wrapped Normal and VMF, as well as most product space VAEs with different curvatures (third section of the table).

Our (U 2 ) 3 performed similarly to the Euclidean baseline VAE.

With diagonal covariance parametrization (Table 4) , we observe similar trends.

With a latent dimension of 6, the Riemannian Normal Poincar?? ball VAE is still the best performer.

The Euclidean baseline VAE achieved better results than its spherical covariance counterpart.

Overall, the best result is achieved by the single-component spherical model, with learnable curvature S 6 .

Interestingly, all single-component VAEs performed better than their (M 2 ) 3 counterparts, except for the H 6 hyperboloid, but only by a tiny margin.

Products of different component types also achieve good results.

Noteworthy is that their fixed curvature variants seem to perform marginally better than learnable curvature ones.

Our universal VAEs perform at around the Euclidean baseline VAE performance.

Interestingly, all of them end up with negative curvatures ???0.3 < K < 0.

Secondly, we run our models with a latent space dimension of 12.

We immediately notice, that not many models can beat the Euclidean VAE baseline E 12 consistently, but several are within the margin of error.

Notably, the product VAEs of H, S, and E, fixed and learnable H 12 , and our universal VAE (U 2 ) 6 .

Interestingly, products of small components perform better when curvature is fixed, whereas single big component VAEs are better when curvature is learned.

Dynamically-binarized Omniglot reconstruction For a latent space of dimension 6 (Table 5) , the best of the baseline models is the Poincar?? VAE of (Mathieu et al., 2019) .

Our models that come

very close to the average estimated marginal log-likelihood, and are definitely within the margin of error, are mainly (S 2 ) 3 , D 2 ?? E 2 ?? P 2 , and U 6 .

However, with the variance of performance across different runs, we cannot draw a clear conclusion.

In general, hyperbolic VAEs seem to be doing a bit better on this dataset than spherical VAEs, which is also confirmed by the fact that almost all universal curvature models finished with negative curvature components.

CIFAR-10 reconstruction For a latent space of dimension 6, we can observe that almost all nonEuclidean models perform better than the euclidean baseline E 6 .

Especially well-performing is the fixed hyperboloid H 6 ???1 , and the learnable hypersphere S 6 .

Curvatures for all learnable models on this dataset converge to values in the approximate range of (???0.15, +0.15).

Summary In conclusion, a very good model seems to be the Riemannian Normal Poincar?? ball VAE RN P n .

However, it has practical limitations due to a rejection sampling algorithm and an unstable implementation.

On the contrary, von Mises-Fischer spherical VAEs have almost consistently performed worse than their Wrapped Normal equivalents.

Overall, Wrapped Normal VAEs in all constant curvature manifolds seem to perform well at modeling the latent space.

A key takeaway is that our universal curvature models U n and (U 2 ) n/2 seem to generally outperform their corresponding Euclidean VAE baselines in lower-dimensional latent spaces and, with minor losses, manage to keep most of the competitive performance as the dimensionality goes up, contrary to VAEs with other non-Euclidean components.

By transforming the latent space and associated prior distributions onto Riemannian manifolds of constant curvature, it has previously been shown that we can learn representations on curved space.

Generalizing on the above ideas, we have extended the theory of learning VAEs to products of constant curvature spaces.

To do that, we have derived the necessary operations in several models of constant curvature spaces, extended existing probability distribution families to these manifolds, and generalized VAEs to latent spaces that are products of smaller "component" spaces, with learnable curvature.

On various datasets, we show that our approach is competitive and additionally has the property that it generalizes the Euclidean variational autoencoder -if the curvatures of all components go to 0, we recover the VAE of Kingma & Welling (2014 An elementary notion in Riemannian geometry is that of a real, smooth manifold M ??? R n , which is a collection of real vectors x that is locally similar to a linear space, and lives in the ambient space R n .

At each point of the manifold x ??? M a real vector space of the same dimensionality as M is defined, called the tangent space at point x: T x M. Intuitively, the tangent space contains all the directions and speeds at which one can pass through x. Given a matrix representation G(x) ??? R n??n of the Riemannian metric tensor g(x), we can define a scalar product on the tangent space:

A Riemannian manifold is then the tuple (M, g).

The scalar product induces a norm on the tangent space T x M: ||a|| x = a, a x ???a ??? T x M (Petersen et al., 2006).

Although it seems like the manifold only defines a local geometry, it induces global quantities by integrating the local contributions.

The metric tensor induces a local infinitesimal volume element on each tangent space T x M and hence a measure is induced as well dM(x) = |G(x)|dx where dx is the Lebesgue measure.

The length of a curve ?? :

Straight lines are generalized to constant speed curves giving the shortest path between pairs of points x, y ??? M, so called geodesics, for which it holds that ?? * = arg min ?? L(??), such that ??(0) = x, ??(1) = y, and

.

Using this metric, we can go on to define a metric space (M, d M ).

Moving from a point x ??? M in a given direction v ??? T x M with constant velocity is formalized by the exponential map: exp x : T x M ??? M. There exists a unique unit speed geodesic ?? such that ??(0) = x and

The corresponding exponential map is then defined as exp x (v) = ??(1).

The logarithmic map is the inverse log x = exp ???1

x : M ??? T x M. For geodesically complete manifolds, i.e. manifolds in which there exists a length-minimizing geodesic between every x, y ??? M, such as the Lorentz model, hypersphere, and many others, exp x is well-defined on the full tangent space T x M.

To connect vectors in tangent spaces, we use parallel transport PT x???y : T x M ??? T y M, which is an isomorphism between the two tangent spaces, so that the transported vectors stay parallel to the connection.

It corresponds to moving tangent vectors along geodesics and defines a canonical way to connect tangent spaces.

We have seen five different models of constant curvature space, each of which has advantages and disadvantages when applied to learning latent representations in them using VAEs.

A big advantage of the hyperboloid and hypersphere is that optimization in the spaces does not suffer from as many numerical instabilities as it does in the respective projected spaces.

On the other hand, we have seen that when K ??? 0, the norms of points go to infinity.

As we will see in experiments, this is not a problem when optimizing curvature within these spaces in practice, except if we're trying to cross the boundary at K = 0 and go from a hyperboloid to a sphere, or vice versa.

Intuitively, the points are just positioned very differently in the ambient space of H ??? and S , for a small > 0.

Since points in the n-dimensional projected hypersphere and Poincar?? ball models can be represented using a real vector of length n, it enables us to visualize points in these manifolds directly for n = 2 or even n = 3.

On the other hand, optimizing a function over these models is not very well-conditioned.

In the case of the Poincar?? ball, a significant amount of points lie close to the boundary of the ball (i.e. with a squared norm of almost 1/K), which causes numerical instabilities even when using 64-bit float precision in computations.

A similar problem occurs with the projected hypersphere with points that are far away from the origin 0 (i.e. points that are close to the "South pole" on the backprojected sphere).

Unintuitively, all points that are far away from the origin are actually very close to each other with respect to the induced distance function and very far away from each other in terms of Euclidean distance.

Both distance conversion theorems (A.5 and its projected hypersphere counterpart) rely on the points being fixed when changing curvature.

If they are somehow dependent on curvature, the convergence theorem does not hold.

We conjecture that if points stay close to the boundary in P or far away from 0 in D as K ??? 0, this is exactly the reason for numerical instabilities (apart from the standard numerical problem of representing large numbers in floating-point notation).

Because of the above reasons, we do some of our experiments with the projected spaces and others with the hyperboloid and hypersphere, and aim to compare the performance of these empirically as well.

Distance function The distance function in E n is

Due to the Pythagorean theorem, we can derive that

Exponential map The exponential map in E n is

The fact that the resulting points belong to the space is trivial.

Deriving the inverse function, i.e. the logarithmic map, is also trivial: log x (y) = y ???

x.

Parallel transport We do not need parallel transport in the Euclidean space, as we can directly sample from a Normal distribution.

In other words, we can just define parallel transport to be an identity function.

A.4.1 HYPERBOLOID Do note, that all the theorems for the hypersphere are essentially trivial corollaries of their equivalents in the hypersphere (and vice-versa) (Section A.5.1).

Notable differences include the fact that

, and all the operations use the hyperbolic trigonometric functions sinh, cosh, and tanh, instead of their Euclidean counterparts.

Also, we often leverage the "hyperbolic" Pythagorean theorem, in the form cosh 2 (??) ??? sinh 2 (??) = 1.

Projections Due to the definition of the space as a retraction from the ambient space, we can project a generic vector in the ambient space to the hyperboloid using the shortest Euclidean distance by normalization:

Secondly, the n + 1 coordinates of a point on the hyperboloid are co-dependent; they satisfy the relation x, x L = 1/K. This implies, that if we are given a vector with n coordinatesx = (x 2 , . . .

, x n+1 ), we can compute the missing coordinate to place it onto the hyperboloid:

This is useful for example in the case of orthogonally projecting points from T ??0 H n K onto the manifold.

Remark A.1 (About the divergence of points in H n K ).

Since the points on the hyperboloid x ??? H n K are norm-constrained to

all the points on the hyperboloid go to infinity as K goes to 0 ??? from below:

This confirms the intuition that the hyperboloid grows "flatter", but to do that, it has to go away from the origin of the coordinate space 0.

A good example of a point that diverges is the origin of the hyperboloid ??

T .

That makes this model unsuitable for trying to learn sign-agnostic curvatures, similarly to the hypersphere.

and in the case of x := ?? 0 = (R, 0, . . . , 0) T :

For all x, y ??? H n K , the logarithmic map in H n K maps y to a tangent vector at x:

Proof.

We show the detailed derivation of the logarithmic map as an inverse function to the exponential map log x (y) = exp ???1

x (y), adapted from (Nagano et al., 2019).

As mentioned previously,

Solving for v, we obtain

However, we still need to rewrite ||v|| L in evaluatable terms:

Plugging the result back into the first equation, we obtain

, and the last equality assumes |??| > 1.

This assumption holds, since for all points x, y ??? H n K it holds that x, y L ??? ???R 2 , and x, y L = ???R 2 if and only if x = y, due to Cauchy-Schwarz (Ratcliffe, 2006, Theorem 3.1.6).

Hence, the only case where this would be a problem would be if x = y, but it is clear that the result in that case is u = 0.

Parallel transport Using the generic formula for parallel transport in manifolds for x, y ??? M and

and the logarithmic map formula from Theorem A.2

A special form of parallel transport exists for when the source vector is ?? 0 = (R, 0, . . . , 0) T :

Do note, that all the theorems for the projected hypersphere are essentially trivial corollaries of their equivalents in the Poincar?? ball (and vice-versa) (Section A.5.2).

Notable differences include the fact that

, and all the operations use the hyperbolic trigonometric functions sinh, cosh, and tanh, instead of their Euclidean counterparts.

Also, we often leverage the "hyperbolic" Pythagorean theorem, in the form cosh 2 (??) ??? sinh 2 (??) = 1.

Proof.

Distance function The distance function in P n K is (derived from the hyperboloid distance function using the stereographic projection ?? K ):

Theorem A.4 (Distance equivalence in P n K ).

For all K < 0 and for all pairs of points x, y ??? P n K , the Poincar?? distance between them equals the gyrospace distance

Proof.

Proven using Mathematica (File: distance limits.ws), proof involves heavy algebra.

Theorem A.5 (Gyrospace distance converges to Euclidean in P n K ).

For any fixed pair of points x, y ??? P n K , the Poincar?? gyrospace distance between them converges to the Euclidean distance in the limit (up to a constant) as K ??? 0 ??? :

Proof.

where the second equality holds because of the theorem of limits of composed functions, where

We see that

due to Theorem A.14, and

Additionally for the last equality, we need the fact that

Theorem A.6 (Distance converges to Euclidean as

For any fixed pair of points x, y ??? P n K , the Poincar?? distance between them converges to the Euclidean distance in the limit (up to a constant) as

Proof.

Theorem A.4 and A.5.

Exponential map As derived and proven in Ganea et al. (2018a) , the exponential map in P n K and its inverse is

In the case of x := ?? 0 = (0, . . . , 0) T they simplify to:

Parallel transport Kochurov et al. (2019); Ganea et al. (2018a) have also derived and implemented the parallel transport operation for the Poincar?? ball:

where

is the gyration operation (Ungar, 2008, Definition 1.11).

Unfortunately, on the Poincar?? ball, ??, ?? x has a form that changes with respect to x, unlike in the hyperboloid.

A.5.1 HYPERSPHERE All the theorems for the hypersphere are essentially trivial corollaries of their equivalents in the hyperboloid (Section A.4.1).

Notable differences include the fact that

, and all the operations use the Euclidean trigonometric functions sin, cos, and tan, instead of their hyperbolic counterparts.

Also, we often leverage the Pythagorean theorem, in the form sin 2 (??) + cos 2 (??) = 1.

Projections Due to the definition of the space as a retraction from the ambient space, we can project a generic vector in the ambient space to the hypersphere using the shortest Euclidean distance by normalization:

Secondly, the n + 1 coordinates of a point on the sphere are co-dependent; they satisfy the relation x, x 2 = 1/K. This implies, that if we are given a vector with n coordinatesx = (x 2 , . . .

, x n+1 ), we can compute the missing coordinate to place it onto the sphere:

This is useful for example in the case of orthogonally projecting points from T ??0 S n K onto the manifold.

Remark A.7 (About the divergence of points in S n K ).

Since the points on the hypersphere x ??? S n K are norm-constrained to x, x 2 = 1 K , all the points on the sphere go to infinity as K goes to 0 + from above:

This confirms the intuition that the sphere grows "flatter", but to do that, it has to go away from the origin of the coordinate space 0.

A good example of a point that diverges is the north pole of the sphere ??

T .

That makes this model unsuitable for trying to learn sign-agnostic curvatures, similarly to the hyperboloid.

Theorem A.8 (Logarithmic map in S n K ).

For all x, y ??? S n K , the logarithmic map in S n K maps y to a tangent vector at x:

where ?? = K x, y 2 .

Proof.

Analogous to the proof of Theorem A.2.

As mentioned previously,

Solving for v, we obtain

However, we still need to rewrite ||v|| 2 in evaluatable terms:

R 2 x, y 2 , and therefore

Plugging the result back into the first equation, we obtain

where ?? = 1 R 2 x, y 2 = K x, y 2 , and the last equality assumes |??| > 1.

This assumption holds, since for all points x, y ??? S n K it holds that x, y 2 ??? R 2 , and x, y 2 = R 2 if and only if x = y, due to Cauchy-Schwarz (Ratcliffe, 2006, Theorem 3.1.6).

Hence, the only case where this would be a problem would be if x = y, but it is clear that the result in that case is u = 0.

Parallel transport Using the generic formula for parallel transport in manifolds (Equation A.4.1) for x, y ??? S n K and v ??? T x S n K and the spherical logarithmic map formula

where ?? = K x, y 2 , we derive parallel transport in S n K :

A special form of parallel transport exists for when the source vector is ?? 0 = (R, 0, . . . , 0) T :

Do note, that all the theorems for the projected hypersphere are essentially trivial corollaries of their equivalents in the Poincar?? ball (and vice-versa) (Section A.4.2).

Notable differences include the fact that

, and all the operations use the Euclidean trigonometric functions sin, cos, and tan, instead of their hyperbolic counterparts.

Also, we often leverage the Pythagorean theorem, in the form sin 2 (??) + cos 2 (??) = 1.

Remark A.9 (Homeomorphism between S n K and R n ).

We notice that ?? K is not a homeomorphism between the n-dimensional sphere and R n , as it is not defined at ????? 0 = (???R; 0 T ) T .

If we additionally changed compactified the plane by adding a point "at infinity" and set it equal to ?? K (?? 0 ), ?? K would become a homeomorphism.

Proof.

Distance function The distance function in D n K is (derived from the spherical distance function using the stereographic projection ?? K ):

For all K > 0 and for all pairs of points x, y ??? D n K , the spherical projected distance between them equals the gyrospace distance

Proof.

Proven using Mathematica (File: distance limits.ws), proof involves heavy algebra.

Theorem A.12 (Gyrospace distance converges to Euclidean in D n K ).

For any fixed pair of points x, y ??? D n K , the spherical projected gyrospace distance between them converges to the Euclidean distance in the limit (up to a constant) as K ??? 0 + :

Proof.

where the second equality holds because of the theorem of limits of composed functions, where

We see that

due to Theorem A.14, and

Additionally for the last equality, we need the fact that

Theorem A.13 (Distance converges to Euclidean as

For any fixed pair of points x, y ??? D n K , the spherical projected distance between them converges to the Euclidean distance in the limit (up to a constant) as K ??? 0 + :

Proof.

Theorem A.11 and A.12.

Exponential map Analogously to the derivation of the exponential map in P

where we use the fact that tanh ???1 (ix) = i tan ???1 (x) and tanh(ix) = i tan(x).

We can easily see

Hence, the geodesic has the form of

and therefore the exponential map in D n K is:

Under review as a conference paper at ICLR 2020

The inverse formula can also be computed:

In the case of x := ?? 0 = (0, . . . , 0) T they simplify to:

Parallel transport Similarly to the Poincar?? ball, we can derive the parallel transport operation for the projected sphere:

where

is the gyration operation (Ungar, 2008, Definition 1.11).

Unfortunately, on the projected sphere, ??, ?? x has a form that changes with respect to x, similarly to the Poincar?? ball and unlike in the hypersphere.

Theorem A.14 (M??bius addition converges to Eucl.

vector addition).

Note: This theorem works from both sides, hence applies to the Poincar?? ball as well as the projected spherical space.

Observe that the M??bius addition has the same form for both spaces.

Proof.

where M ??? {S, H}.

Proof.

Proof.

hence the exponential map converges to its Euclidean variant.

due to several applications of the theorem of limits of composed functions, Lemma A.16, and the fact that

The negative case K < 0 is analogous.

hence the logarithmic map converges to its Euclidean variant.

Proof.

Firstly,

due to Theorem A.14.

For the positive case K > 0

due to several applications of the theorem of limits of composed functions, product rule for limits, Lemma A.16, and the fact that

The negative case K < 0 is analogous.

Proof.

due to Theorem A.14 and the theorem of limits of composed functions.

Proof.

Proof.

This was shown for the case K = 1 by Nagano et al. (2019) .

The difference is that we do not assume unitary radius R = 1 = 1/ ??? ???K. Hence, our tranformation function has the form f = exp

The derivative of parallel transport P T

Using the orthonormal basis (with respect to the Lorentz product) {?? 1 , . . .

?? n }, we can compute the determinant by computing the change with respect to each basis vector.

.

Since parallel transport preserves norms and vectors in the orthonormal basis have norm 1, the change is d PT

For computing the determinant of the exponential map Jacobian, we choose the orthonormal basis {?? 1 = u/ u L , ?? 2 , . . .

, ?? n }, where we just completed the basis based on the first vector.

We again look at the change with respect to each basis vector.

For the basis vector ?? 1 :

where the second equality is due to

For every other basis vector ?? k where k > 1:

where the third equality holds because

where the last equality relies on the fact that the basis is orthogonal, and u is parallel to ?? 1 = u/ u L , hence it is orthogonal to all the other basis vectors.

Because the basis is orthonormal the determinant is a product of the norms of the computed change for each basis vector.

Therefore,

Additionally, the following two properties hold:

Therefore, we obtain

Proof.

The theorem is very similar to Theorem B.1.

The difference is that in this one, our manifold changes from H n K to S n K , hence K > 0.

Our tranformation function has the form f = exp

The derivative of parallel transport P T

Using the orthonormal basis (with respect to the Lorentz product) {?? 1 , . . .

?? n }, we can compute the determinant by computing the change with respect to each basis vector.

.

Since parallel transport preserves norms and vectors in the orthonormal basis have norm 1, the change is d PT

For computing the determinant of the exponential map Jacobian, we choose the orthonormal basis {?? 1 = u/ u 2 , ?? 2 , . . .

, ?? n }, where we just completed the basis based on the first vector.

We again look at the change with respect to each basis vector.

For the basis vector ?? 1 :

where the second equality is due to

For every other basis vector ?? k where k > 1:

where the third equality holds because

where the last equality relies on the fact that the basis is orthogonal, and u is parallel to ?? 1 = u/ u 2 , hence it is orthogonal to all the other basis vectors.

Because the basis is orthonormal the determinant is a product of the norms of the computed change for each basis vector.

Therefore,

Additionally, the following two properties hold:

Therefore, we obtain

Proof.

Follows from Theorem B.1 and A.3.

Also proven by (Mathieu et al., 2019) in a slightly different form for a scalar scale parameter WN (z; ??, ?? 2 I).

Given

Proof.

Follows from Theorem B.2 and A.3 adapted from P to D.

Universal models of geometry Duality between spaces of constant curvature was first noticed by Lambert (1770), and later gave rise to various theorems that have the same or similar forms in all three geometries, like the law of sines (Bolyai, 1832)

where p K (r) = 2?? sin K (r) denotes the circumference of a circle of radius r in a space of constant curvature K, and

Other unified formulas for the law of cosines, or recently, a unified Pythagorean theorem has also been proposed (Foote, 2017) :

where A(r) is the area of a circle of radius r in a space of constant curvature K. Unfortunately, in this formulation A(r) still depends on the sign of K w.r.t.

the choice of trigonometric functions in its definition.

There also exist approaches defining a universal geometry of constant curvature spaces.

Li et al. (2001, Chapter 4 ) define a unified model of all three geometries using the null cone (light cone) of a Minkowski space.

The term "Minkowski space" comes from special relativity and is usually denoted as R 1,n , similar to the ambient space of what we defined as H n , with the Lorentz scalar product ??, ?? L .

The hyperboloid H n corresponds to the positive (upper, future) null cone of R 1,n .

All the other models can be defined in this space using the appropriate stereographic projections and pulling back the metric onto the specific sub-manifold.

Unfortunately, we found the formalism not useful for our application, apart from providing a very interesting theoretical connection among the models.

Concurrent VAE approaches The variational autoencoder was originally proposed in Kingma & Welling (2014) and concurrently in Rezende et al. (2014) .

One of the most common improvements on the VAE in practice is the choice of the encoder and decoder maps, ranging from linear parametrizations of the posterior to feed-forward neural networks, convolutional neural networks, etc.

For different data domains, extensions like the GraphVAE (Simonovsky & Komodakis, 2018) using graph convolutional neural networks for the encoder and decoder were proposed.

The basic VAE framework was mostly improved upon by using autoregressive flows (Chen et al., 2014) or small changes to the ELBO loss function (Matthey et al., 2017; Burda et al., 2016 ).

An important work in this area is ??-VAE, which adds a simple scalar multiplicative constant to the KL divergence term in the ELBO, and has shown to improve both sample quality and (if ?? > 1) disentanglement of different dimensions in the latent representation.

For more information on disentanglement, see Locatello et al. (2018) .

Geometric deep learning One of the emerging trends in deep learning has been to leverage nonEuclidean geometry to learn representations, originally emerging from knowledge-base and graph representation learning (Bronstein et al., 2017) .

Recently, several approaches to learning representations in Euclidean spaces have been generalized to non-Euclidean spaces (Dhingra et al., 2018; Ganea et al., 2018b; Nickel & Kiela, 2017) .

Since then, this research direction has grown immensely and accumulated more approaches, mostly for hyperbolic spaces, like Ganea et al. (2018a); Nickel & Kiela (2018); Tifrea et al. (2019); Law et al. (2019) .

Similarly, spherical spaces have also been leveraged for learning non-Euclidean representations (Batmanghelich et al., 2016; Wilson & Hancock, 2010) .

To be able to learn representations in these spaces, new Riemannian optimization methods were required as well (Wilson & Leimeister, 2018; Bonnabel, 2013; B??cigneul & Ganea, 2019) .

The generalization to products of constant curvature Riemannian manifolds is only natural and has been proposed by Gu et al. (2019) .

They evaluated their approach by directly optimizing a distancebased loss function using Riemannian optimization in products of spaces on graph reconstruction and word analogy tasks, in both cases reaping the benefits of non-Euclidean geometry, especially when learning lower-dimensional representations.

Further use of product spaces with constant curvature components to train Graph Convolutional Networks was concurrently with this work done by Anonymous (2020) .

One of the first attempts at leveraging geometry in VAEs was Arvanitidis et al. (2018) .

They examine how a Euclidean VAE benefits both in sample quality and latent representation distribution quality when employing a non-Euclidean Riemannian metric in the latent space using kernel transformations.

Hence, a potential improvement area of VAEs could be the choice of the posterior family and prior distribution.

However, the Gaussian (Normal) distribution works very well in practice, as it is the maximum entropy probability distribution with a known variance, and imposes no constraints on higher-order moments (skewness, kurtosis, etc.) of the distribution.

Recently, non-Euclidean geometry has been used in learning variational autoencoders as well.

Generalizing Normal distributions to these spaces is in general non-trivial.. Two similar approaches, Davidson et al. (2018) and Xu & Durrett (2018) , used the von MisesFischer distribution on the unit hypersphere to generalize VAEs to spherical spaces.

The von MisesFischer distribution is again a maximum entropy probability distribution on the unit hypersphere, but only has a spherical covariance parameter, which makes it less general than a Gaussian distribution.

Conversely, two approaches, Mathieu et al. (2019) and Nagano et al. (2019) , have generalized VAEs to hyperbolic spaces -both the Poincar?? ball and the hyperboloid, respectively.

They both adopt a non-maximum entropy probability distribution called the Wrapped Normal.

Additionally, Mathieu et al. (2019) also derive the Riemannian Normal, which is a maximum entropy distribution on the Poincar?? disk, but in practice performs similar to the Wrapped Normal, especially in higher dimensions.

Our approach generalizes on the afore-mentioned geometrical VAE work, by employing a "products of spaces" approach similar to Gu et al. (2019) and unifying the different approaches into a single framework for all spaces of constant curvature.

Even though we have shown that one can approximate the true posterior very well with Normallike distributions in Riemannian manifolds of constant curvature, there remain several promising directions of explorations.

First of all, an interesting extension of this work would be to try mixed-curvature VAEs on graph data, e.g. link prediction on social networks, as some of our models might be well suited for sparse and structured data.

Another very beneficial extension would be to investigate why the obtained results have a relatively big variance across runs and try to reduce it.

However, this is a problem that affects the Euclidean VAE as well, even if not as flagrantly.

Secondly, we have empirically noticed that it seems to be significantly harder to optimize our models in spherical spaces -they seem more prone to divergence.

In discussions, other researchers have also observed similar behavior, but a more thorough investigation is not available at the moment.

We have side-stepped some optimization problems by introducing products of spaces -previously, it has been reported that both spherical and hyperbolic VAEs generally do not scale well to dimensions greater than 20 or 40.

For those cases, we could successfully optimize a subdivided space (S 2 ) 36 instead of one big manifold S 72 .

However, that also does not seem to be a conclusive rule.

Especially in higher dimensions, we have noticed that our VAEs (S 2 ) 36 with learnable curvature and D

1 with fixed curvature seem to consistently diverge.

In a few cases S 72 with fixed curvature and even the product (E 2 ) 12 ?? (H 2 ) 12 ?? (S 2 ) 12 with learnable curvature seemed to diverge quite often as well.

The most promising future direction seems to be the use of "Normalizing Flows" for variational inference as presented by Rezende & Mohamed (2015) and Gemici et al. (2016) .

More recently, it was also combined with "autoregressive flows" in Huang et al. (2018) .

Using normalizing flows, one should be able to achieve the desired level of complexity of the latent distribution in a VAE, which should, similarly to our work, help to approximate the true posterior of the data better.

The advantage of normalizing flows is the flexibility of the modeled distributions, at the expense of being more computationally expensive.

Finally, another interesting extension would be to extend the defined geometrical models to allow for training generative adversarial networks (GANs) (Goodfellow et al., 2014) in products of constant curvature spaces and benefit from the better sharpness and quality of samples that GANs provide.

Finally, one could synthesize the above to achieve adversarially trained autoencoders in Riemannian manifolds similarly to Pan et al. (2018); Kim et al. (2017); Makhzani et al. (2015) and aim to achieve good sample quality and a well-formed latent space at the same time.

E EXTENDED RESULTS Figure 1 : Learned curvature across epochs (with standard deviation) with latent space dimension of 6, diagonal covariance parametrization, on the MNIST dataset.

Figure 2: Qualitative comparison of reconstruction quality of randomly selected runs of a selection of well-performing models on MNIST test set digits.

Table 13 : Summary of results (mean and standard-deviation) with latent space dimension of 6, diagonal covariance parametrization, on the CIFAR dataset.

All nan standard deviation values below indicate the repeated experiment was not stable enough to produce a meaningful estimate of spread.

LL ELBO BCE KL

<|TLDR|>

@highlight

Variational Autoencoders with latent spaces modeled as products of constant curvature Riemannian manifolds improve on image reconstruction over single-manifold variants.

@highlight

This paper introduces a general formulation of the notion of a VAE with a latent space composed by a curved manifold.

@highlight

This paper is about developing VAEs in non-Euclidean spaces.