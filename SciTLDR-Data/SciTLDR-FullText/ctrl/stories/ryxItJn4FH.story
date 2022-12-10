We consider the problem of unsupervised learning of a low dimensional, interpretable, latent state of a video containing a moving object.

The problem of distilling dynamics from pixels has been extensively considered through the lens of graphical/state space models that exploit Markov structure for cheap computation and structured graphical model priors for enforcing interpretability on latent representations.

We take a step towards extending these approaches by discarding the Markov structure; instead, repurposing the recently proposed Gaussian Process Prior Variational Autoencoder for learning sophisticated latent trajectories.

We describe the model and perform experiments on a synthetic dataset and see that the model reliably reconstructs smooth dynamics exhibiting U-turns and loops.

We also observe that this model may be trained without any beta-annealing or freeze-thaw of training parameters.

Training is performed purely end-to-end on the unmodified evidence lower bound objective.

This is in contrast to previous works, albeit for slightly different use cases, where application specific training tricks are often required.

We consider the problem of unsupervised learning of a low dimensional, interpretable, latent state of a video containing a moving object.

The problem of distilling interpretable dynamics from pixels has been extensively considered through the lens of graphical/state space models (Fraccaro et al., 2017; Lin et al., 2018; Pearce et al., 2018; Chiappa and Paquet, 2019 ) that exploit Markov structure for cheap computation and structured priors for enforcing interpretability on latent representations.

We take a step towards extending these approaches by discarding the Markov structure; inspired by Gaussian process dynamical models (Wang et al., 2006) , we instead repurpose the recently proposed Gaussian Process Prior Variational Autoencoder (Casale et al., 2018) for learning interpretable latent dynamics.

We describe the model and perform experiments on a synthetic dataset and see that the model reliably reconstructs smooth dynamics exhibiting U-turns and loops.

We also observe that this model may be trained without any β annealing or freeze-thaw of training parameters in contrast to previous works, albeit for slightly different use cases, where application specific training tricks are often required.

Assume we are given a dataset of images and each image has a corresponding feature vector, e.g. images of faces and features are pose angle, lighting intensity etc.

The Gaussian Process Prior Variational Autoencoder (GPP-VAE) may be used to learn intermediate latent representations from features (input) to images (output) where the input-output relationship is supervised but the latent representation is unsupervised.

In the use case we consider, we assume that we have a set of videos of a single object moving around on a pixel display.

Each video consists of a set of T images: v 1 , ..., v T ∈ [0, 1] 32×32 which are binary arrays and the corresponding feature of an image is its time stamp t.

The intermediate state we aim to learn is an interpretable 2D latent time-series of x 1:T , y 1:T coordindates for the object for each frame.

We have no ground truth data of object position hence this is unsupervised.

We place a Gaussian process prior over time on the functions x, y : [1, T ] → R. A Gaussian process may be viewed as a more general linear Gaussian state space model, this has the benefit of being more flexible, and smoothing (aggregating information across all time) is performed by default.

However sacrificing Markov structure typically increases computational complexity from linear to cubic.

A Gaussian process is fully specified by its mean and covariance functions.

Evaluating these functions at a discretization of the input domain, T = 1 : T ⊂ [1, T ], yields the mean vector and covariance matrix of a multivariate Gaussian distribution.

Hence without loss of generality, we adopt the notation of scalars x t , y t and vectors x 1:T , y 1:T instead of functions x(t), y(t).

The generative model is given by

Bernoulli over pixels

where B(v t |p θ (x t , y t )) is a product of 32 × 32 independent Bernoulli distributions over pixels parameterised by a neural network p θ (x t , y t ) with parameters θ.

The functions k x , k y :

→ R are positive semi-definite kernels with hyperparameters that we assume are known in this work for simplicity and N (x|µ, Σ) is the multivariate Gaussian density.

Given a video v 1:T we aim to learn x 1:T and y 1:T which ideally would be inferred by Bayes rule, P[x 1:T , y 1:T |v 1:T ].

However due to the B(v t |p θ (x t , y t )) terms, the true posterior has no normalized analytic form.

We desire a variational approximation with two properties.

Firstly for fast inference at test time we require amortization; we aim to learn a function from observed variables v 1:T directly to approximate posterior q(x 1:T , y 1:T |v 1:T ).

Secondly, at test time, video data is streaming and frames are accumulated over time thus we also require an approximate posterior that can handle variable length videos.

Satisfying both of the properties and exploiting the structure of the generative model, we propose the following variational approximation q(x 1:T , y 1:T |v 1:

which is simply the true generative model with only the troublesome B(v t |p θ (x t , y t )) terms replaced by Gaussian densities denoted q * φ (x t , y t |v t ).

These new Gaussian factors are conjugate to the GP prior thereby enabling normalization.

Each factor is parameterized by the output of a recognition network µ * xφ (v t ), σ * xφ 2 (v t ) (similarly for y t ) with parameters φ.

By noting the symmetry of the Gaussian distribution, N (x|µ, σ 2 ) = N (µ|x, σ 2 ), the denominator is exactly the form of the unnormalized likelihood × prior of standard GP regression.

The x 1:T are latent function values and {(t, µ * φx (v t )} T 1 are a set of observations (or pseudo-points) each with noise σ * φx 2 (v t ).

These points condition the generative prior GP yielding an (analytic) posterior GP that approximates the (non analytic) true posterior P[x 1:T , y 1:T |v 1:T ].

Thus the standard GP equations (see Appendix B) yield means, µ x (t), µ y (t), and variances plotted in Figure 1 .

Since, for simplicity, we assume x 1:T and y 1:T are independent, this may be viewed as two standard 1-D Gaussian process regression models and the term Z(v 1: .

Right: each latent dimension is a GP over time, using the generative prior GP conditioned on pseudo points from the recognition network.

and Z x (v 1:T ) is thus precisely the marginal likelihood of the x GP commonly used in GP regression for hyperparameter learning.

At test time, if there are missing frames or the video is shorter, T s < T , mathematically, only q * (v 1:Ts ) terms are included in Equation 3 and we have Z(v 1:Ts ).

Intuitively, the approximate posterior simply becomes a GP regression model with fewer points, {(t, µ * φx (v t )} Ts 1 .

In this work we assume that the kernels are the popular squared exponential

x ) with hyperparameters l x = l y = 5 that represent the time scale, or speed, of changes in x and y positions respectively and may be learnt or informed by prior knowledge, i.e. observing the volatility of object movement in videos.

This is a stationary kernel and enforces smoothness over latent trajectories and by setting l x , l y 1 recovers a standard factorised Gaussian prior of the traditional VAE model.

In general, any kernel may be used, quadratic kernel for parabolic motion, min kernel for Brownian motion, periodic kernels for oscillatory motion, or any sum or product of these kernels depending on the prior belief about a particular dataset or physical system.

For training the neural network parameters θ, φ, we maximize the evidence lower bound,

a full derivation is given in the Appendix.

The first (inner) term in Equation 4 is the reconstruction term, evaluated with the reparameterisation trick (Kingma and Welling, 2013; Rezende et al., 2014) , and the middle and final terms compose the analytically tractable Kullback-Leibler divergence between the GP prior and the inference model.

Alternatively, the first two terms together may be viewed as the "error" between the true posterior and approximate posterior introduced by replacing only the Bernoulli likelihoods with Gaussian approximations, and the final term is a surrogate marginal likelihood.

Bottom: x 1:T , y 1:T inferred using the the GPP-VAE.

In order to test the ability to learn latent dynamics we synthesize a controlled dataset.

We generate videos of length T = 30 by first sampling a time series x 1:T , y 1:T ∼ N · |0, k(T , T ) .

Each pair (x t , y t ) is rescaled to pixel indeces and rendered as a ball onto a binary canvas.

Example videos are shown in Figure 2 and Appendix A, network details and training are given in Appendix B.3.

As a naive baseline model, we train a standard Variational Autoencoder using the same inference and rendering networks.

For model evaluation, we consider how the latent space compares to the ground truth.

In Figure 2 we plot videos and approximate posterior means and see the GPP-VAE largely recovers the ground truth, note this is never used in training.

By construction, the generated images lie in a low dimensional subspace of pixel space, and we expect similar images (many overlapping white pixels) to be encoded into similar latents.

Thus we consider how a regular pattern of images is encoded into the latent space.

In Figure 3 we plot such patterns with the output of the recognition network (µ * φx (v), µ * φy (v)).

In this case, the VAE appears to learn a distorted and discontinous mapping from pixels to latents while the GPP-VAE learns a continuous mapping with mild distortion.

Both methods learn near perfect reconstruction of videos shown in Appendix A. Source code is available at https://github.com/scrambledpie/GPVAE/.

We present a simple model and show proof-of-concept results that a Gaussian Process Prior within a VAE may be used for learning complex but smooth latent dynamics without any Input VAE Latent GPP-VAE Latent Figure 3 : Left: top: 19 images, bottom: 25 images generated with the ball in a regular pattern.

Centre: the patterns output from the recognition network q * (x, y|v) from the trained VAE.

Ground truth in blue and recognition network means in orange (rotated onto ground truth).

Lines are for visual aid only.

Right: the output of q * (x, y|v) from the trained GPP-VAE (rotated onto ground truth).

There is no time correlation in the images hence we do not plot approximate posterior/apply smoothing.

The VAE latent space is a highly distorted and discontinuous transformation of the pixel space while the GPP-VAE latent space is much more coherent.

For training, see video https://www.youtube.com/watch?v=riVhb6K_iMo.

special training.

In this work we consider a toy dataset and the dynamics model generating the data was also used to fit the model removing miss-specification issues.

Hence future work is to apply the model to a wider variety of less controlled settings, and comparison with more sophisticated baselines.

By comparison, using similar data, the KalmanVariational Autoencoder learnt dynamics (also including sharp turns, hence non-smooth) using an LSTM and training required freeze-thaw of model parameters and re-weighting of objective terms.

Likewise extensions to this model (Chiappa and Paquet, 2019; Pearce et al., 2018) consider multiple objects constrained to parabolic motion and either require β annealing or other training tricks.

We assume independent factorised Gaussian process priors for the horizontal position x 1:T and vertical position y 1:T .

The recognition network returns fully factorised Gaussian densities therefore the approximate posterior is also factorised across vertical and horizontal positions.

As a consequence the log marginal likelihood is a sum of individual marginal likelihoods log Z(v 1:T ) = log Z x (v 1:T ) + log Z y (v 1:T ).

We therefore give the expression for a single term.

In our case, for a single video, v 1:T , the input-output pairs for Gaussian process regression is the set of time stamps and means from the recognition network {(1, µ * φx (v 1 )), ..., (T, µ * φx (v T ))} with noise variance for each observation given by (σ *

Denote the column vector of means as µ *

Due to the matrix inversion, this has cubic cost in the number of observed frames T .

The matrix inversion is done via Cholesky decomposition as suggested by Williams and Rasmussen (2006) .

Secondly to compute the approximate posterior mean and approximate posterior variance, we may use the standard Gaussian process regression equations,

where the dependence upon v 1:T is embedded in the µ * x and Σ * x terms.

The approximate posterior variance for a single point is given by σ 2 x (t|v 1:T ) = k x (t, t|v 1:T ) and the approximate posterior distribution for a single position x t , y t conditioned on all frames is q(x t , y t |v 1:

The objective function given in the main text is

For the next equations, we shall drop the time indices

= log

where expectations are over x, y. The first term must be evaluated by Monte-Carlo using the reparameterization trick (Kingma and Welling, 2013), a sample x i t , y i t is generated by taking the approximate posterior mean and standard deviation and sampling white-noise ∼ N (·|0, 1), x i t = µ x (t|v 1:T ) + * σ x (t|v 1:T ) which can then be passed to p θ (·) and the likelihood of the true image log B(v t |p θ (x i t , y i t )) can be computed.

Next we focus on the second term, the KL divergence from the approximate posterior to the prior.

Substituting in the from of q(x, y|v) = P[x, y]q * (x, y|v)/Z(v),

and note that the prior term that is common to both generative and inference model cancels out.

Combining terms yields the expression in Equation 9.

Recall that q * (x, y|v 1:T ) is a factorised Gaussian over all variables x 1 , .., x T , y 1 , ..., y T .

Therefore the first term of Equation 16 term is a sum of univariate cross-entropys of Gaussian distributions, again dropping time indices to minimize cluttering notation, the term for a single time is given by

where the expectation is over the approximate posterior given in Equation 8.

Let x ∼ N (x|µ, σ 2 ), then the univariate Guassian cross-entropy is given by

The Monte-Carlo integral of the reconstruction term and the analytic expression for the KL divergence can all be implemented in any modern machine learning framework and optimized by gradient ascent.

This is discussed below in Section B.3.

For training data, we generate videos of length T = 30 by first sampling two time series x 1:T , y 1:T ∼ N · |0, k(T , T ) where k(t, t ) = exp(−(t − t) 2 /(2 · 5 2 )).

Each pair (x t , y t ) is rescaled to pixel space (x t ,ỹ t ) = 7 * (x t , y t ) + (16, 16) and rendered as a ball with radius r = 3 onto a binary canvas.

v t ∈ {0, 1} 32×32 where

The recognition network q * : {0, 1} 1024 → R 4 is a fully connected network that takes as input a 32 * 32 = 1024 image flattened to a vector v t ∈ {0, 1} 1024 .

This is followed by a fully connected hidden layer of 500 nodes with the tanh() activation function, and finally the output layer of four nodes returning µ * φx (v t ), log σ * φx (v t ) and µ * φy (v t ), log σ * φx (v t ), the network parameters are therefore two weight matrices and two bias vectors φ = {W 1 q , B 1 q , W 2 q , B 2 q }.

The decoder, or rendering network, p θ : R 2 → [0, 1] 1024 , is almost the same architecture in reverse, the input layer has only two nodes, followed by a single fully connected layer of 500 nodes with the tanh() activation and finally 1024 nodes with the sigmoid() activation yielding a unique independent Bernoulli probability between 0 and 1 for each of the 1024 pixels.

The parameters are thus θ = {W 1 p , B 1 p , W 2 p , B 2 p }.

Training is performed using the Adam optimizer with Tensorflow default parameters α = 0.001, β 1 = 0.9, β 2 = 0.999, = 1e − 08 and a batchsize of 35 randomly generated videos.

We train the each method for 50,000 iterations.

In preliminary testing we applied β annealing of the prior KL term in the objective as it has been shown to stabilize learning however we found that a value of β > 1 would lead to the approximate model learning the prior distribution and never recovering, posterior collapse, and for β < 1 we found that training often became numerically unstable and overflow/underflow errors would cause training to halt.

Therefore we apply no β annealing, all results are with the objective unmodified and optimized end-to-end.

We maintain a held-out test set of latent trajectories x test 1:T , y test 1:T and their rendered counterparts v test 1:T .

We pass the images into the inference model to yield posterior mean vectors µ x (T |v test 1:T ), µ y (T |v test 1:T ) ∈ R T .

We then use linear regression to predict the ground truth trajectories.

Specifically we learn a rotation W ∈ R 2×2 and translation B ∈ R 2 that minimizes the mean squared error over the whole test set

The error minimizing W and B are used to rotate the latent trajectories onto the true trajectories for the figures.

The true trajectories are never used during training.

As a temporary aside for the interested reader, we may draw parallels between the Gaussian Process Prior Variational Autoencoder and Attentive Neural processes (Kim et al., 2019) .

A stochastic process is a collection of random variables (outputs) over an index set (inputs), i.e. a random function generator.

A video generator may be viewed as realisations of a stochastic process, over the index set of time, random outputs are images in pixel space.

At test time, given a set of frames from a video v 1:T and a new time stamp t , we may query a model to predict the new frame P[v t |v 1:T ], a simple regression problem, albeit with 1D input and high dimensional output.

Assume we are given the distribution of the stochastic process realisations, the full generative model of outputs given any inputs, and a subset of input-output pairs from a single realisation of a stochastic process.

For any new input, we desire statistical predictions of the corresponding output from the same realisation.

The Neural process architecture allows a user to make such new predictions where the generative stochastic process is not known, however instead one has access to a large corpus of input-output pairs from many realisations of the same generative process.

Firstly, each element from the set of observed points, (t, v t ), is encoded into a new representation r t , and the set r 1 , ..., r T is accumulated through summation to get R. Secondly, R is used in a decoder with the new input t to parameterise a distribution over the output P[v t |R, t ].

Attentive Neural processes augment this architecture in two ways.

First, a self-attention layer (Vaswani et al., 2017 ) is applied to the set r 1 , ..., r T yieldingr 1 , ...,r T .

Second, for prediction, the new input t is used as a query to inform attention weights overr 1 , ...,r T , such that the aggregated representation R t is informed by t , essentially augmenting Neural processes with a non-parametric memory.

In our use case, the encoder of (t, v t ) is the identity function for t and the recognition network for v t and the encoding r t is the time stamp and means and variances, encoder :R × {0, 1} 32×32 → R

<|TLDR|>

@highlight

We learn sohpisticated trajectories of an object purely from pixels with a toy video dataset by using a VAE structure with a Gaussian process prior.