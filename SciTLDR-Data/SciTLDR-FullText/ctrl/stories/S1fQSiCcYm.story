Autoencoders provide a powerful framework for learning compressed representations by encoding all of the information needed to reconstruct a data point in a latent code.

In some cases, autoencoders can "interpolate": By decoding the convex combination of the latent codes for two datapoints, the autoencoder can produce an output which semantically mixes characteristics from the datapoints.

In this paper, we propose a regularization procedure which encourages interpolated outputs to appear more realistic by fooling a critic network which has been trained to recover the mixing coefficient from interpolated data.

We then develop a simple benchmark task where we can quantitatively measure the extent to which various autoencoders can interpolate and show that our regularizer dramatically improves interpolation in this setting.

We also demonstrate empirically that our regularizer produces latent codes which are more effective on downstream tasks, suggesting a possible link between interpolation abilities and learning useful representations.

One goal of unsupervised learning is to uncover the underlying structure of a dataset without using explicit labels.

A common architecture used for this purpose is the autoencoder, which learns to map datapoints to a latent code from which the data can be recovered with minimal information loss.

Typically, the latent code is lower dimensional than the data, which indicates that autoencoders can perform some form of dimensionality reduction.

For certain architectures, the latent codes have been shown to disentangle important factors of variation in the dataset which makes such models useful for representation learning BID7 BID15 .

In the past, they were also used for pre-training other networks by being trained on unlabeled data and then being stacked to initialize a deep network BID1 BID44 .

More recently, it was shown that imposing a prior on the latent space allows autoencoders to be used for probabilistic or generative modeling BID18 BID34 BID27 .In some cases, autoencoders have shown the ability to interpolate.

Specifically, by mixing codes in latent space and decoding the result, the autoencoder can produce a semantically meaningful combination of the corresponding datapoints.

Interpolation has been frequently reported as a qualitative experimental result in studies about autoencoders BID5 BID35 BID30 BID29 BID14 and latent-variable generative models in general BID10 BID33 BID41 .

The ability to interpolate can be useful in its own right e.g. for creative applications (Carter & Nielsen, 2017) .

However, it also indicates that the autoencoder can "extrapolate" beyond the training data and has learned a latent space with a particular structure.

Specifically, if interpolating between two points in latent space produces a smooth semantic warping in data space, this suggests that nearby points in latent space are semantically similar.

A visualization of this idea is shown in FIG0 , where a smooth A critic network is fed interpolants and reconstructions and tries to predict the interpolation coefficient ?? corresponding to its input (with ?? = 0 for reconstructions).

The autoencoder is trained to fool the critic into outputting ?? = 0 for interpolants.

interpolation between a "2" and a "9" suggests that the 2 is surrounded by semantically similar points, i.e. other 2s.

This property may suggest that an autoencoder which interpolates well could also provide a good learned representation for downstream tasks because similar points are clustered.

If the interpolation is not smooth, there may be "discontinuities" in latent space which could result in the representation being less useful as a learned feature.

This connection between interpolation and a "flat" data manifold has been explored in the context of unsupervised representation learning BID3 and regularization BID43 .Given the widespread use of interpolation as a qualitative measure of autoencoder performance, we believe additional investigation into the connection between interpolation and representation learning is warranted.

Our goal in this paper is threefold: First, we introduce a regularization strategy with the specific goal of encouraging improved interpolations in autoencoders (section 2); second, we develop a synthetic benchmark where the slippery concept of a "semantically meaningful interpolation" is quantitatively measurable (section 3.1) and evaluate common autoencoders on this task (section 3.2); and third, we confirm the intuition that good interpolation can result in a useful representation by showing that the improved interpolation ability produced by our regularizer elicits improved representation learning performance on downstream tasks (section 4).

We also make our codebase available 1 which provides a unified implementation of many common autoencoders including our proposed regularizer.

Autoencoders, also called auto-associators BID4 , consist of the following structure:

First, an input x ??? R dx is passed through an "encoder" z = f ?? (x) parametrized by ?? to obtain a latent code z ??? R dz .

The latent code is then passed through a "decoder"x = g ?? (z) parametrized by ?? to produce an approximate reconstructionx ??? R dx of the input x. We consider the case where f ?? and g ?? are implemented as multi-layer neural networks.

The encoder and decoder are trained simultaneously (i.e. with respect to ?? and ??) to minimize some notion of distance between the input x and the outputx, for example the squared L 2 distance x ???x 2 .Interpolating using an autoencoder describes the process of using the decoder g ?? to decode a mixture of two latent codes.

Typically, the latent codes are combined via a convex combination, so that interpolation amounts to computingx ?? = g ?? (??z 1 +(1?????)z 2 ) for some ?? ??? [0, 1] where z 1 = f ?? (x 1 ) and z 2 = f ?? (x 2 ) are the latent codes corresponding to data points x 1 and x 2 .

We also experimented with spherical interpolation which has been used in settings where the latent codes are expected to have spherical structure BID17 BID45 BID35 , but found it made no discernible difference in practice for any autoencoder we studied.

Ideally, adjusting ?? from 0 to 1 will produce a sequence of realistic datapoints where each subsequentx ?? is progressively less semantically similar to x 1 and more semantically similar to x 2 .

The notion of "semantic similarity" is problem-dependent and ill-defined; we discuss this further in section 3.

As mentioned above, a high-quality interpolation should have two characteristics: First, that intermediate points along the interpolation are indistinguishable from real data; and second, that the intermediate points provide a semantically smooth morphing between the endpoints.

The latter characteristic is hard to enforce because it requires defining a notion of semantic similarity for a given dataset, which is often hard to explicitly codify.

So instead, we propose a regularizer which encourages interpolated datapoints to appear realistic, or more specifically, to appear indistinguishable from reconstructions of real datapoints.

We find empirically that this constraint results in realistic and smooth interpolations in practice (section 3.1) in addition to providing improved performance on downstream tasks (section 4).To enforce this constraint we introduce a critic network BID12 which is fed interpolations of existing datapoints (i.e.x ?? as defined above).

Its goal is to predict ?? fromx ?? , i.e. to predict the mixing coefficient used to generate its input.

When training the model, for each pair of training data points we randomly sample a value of ?? to producex ?? .

In order to resolve the ambiguity between predicting ?? and 1 ??? ??, we constrain ?? to the range [0, 0.5] when feedingx ?? to the critic.

In contrast, the autoencoder is trained to fool the critic to think that ?? is always zero.

This is achieved by adding an additional term to the autoencoder's loss to optimize its parameters to fool the critic.

In a loose sense, the critic can be seen as approximating an "adversarial divergence" BID23 BID0 between reconstructions and interpolants which the autoencoder tries to minimize.

Formally, let d ?? (x) be the critic network, which for a given input produces a scalar value.

The critic is trained to minimize DISPLAYFORM0 where, as above, DISPLAYFORM1 for some x (not necessarily x 1 or x 2 ), and ?? is a scalar hyperparameter.

The first term trains the critic to recover ?? from x ?? .

The second term serves as a regularizer with two functions: First, it enforces that the critic consistently outputs 0 for non-interpolated inputs; and second, by interpolating between x andx (the autoencoder's reconstruction of x) in data space it ensures the critic is exposed to realistic data even when the autoencoder's reconstructions are poor.

We found the second term was not crucial for our approach, but helped stabilize the convergence of the autoencoder and allowed us to use consistent hyperparameters and architectures across all datasets and experiments.

The autoencoder's loss function is modified by adding a regularization term: DISPLAYFORM2 where ?? is a scalar hyperparameter which controls the weight of the regularization term.

Note that the regularization term is effectively trying to make the critic output 0 regardless of the value of ??, thereby "fooling" the critic into thinking that an interpolated input is non-interpolated (i.e., having ?? = 0).

The parameters ?? and ?? are optimized with respect to L f,g (which gives the autoencoder access to the critic's gradients) and ?? is optimized with respect to L d .

We refer to the use of this regularizer as Adversarially Constrained Autoencoder Interpolation (ACAI).

A diagram of the ACAI is shown in FIG1 .

Assuming an effective critic, the autoencoder successfully "wins" this adversarial game by producing interpolated points which are indistinguishable from reconstructed data.

We find in practice that encouraging this behavior also produces semantically smooth interpolations and improved representation learning performance, which we demonstrate in the following sections.

Our loss function is similar to the one used in the Least Squares Generative Adversarial Network BID28 in the sense that they both measure the distance between a critic's output and a scalar using a squared L2 loss.

However, they are substantially different in that ours is used as a regularizer for autoencoders rather than for generative modeling and our critic attempts to regress the interpolation coefficient ?? instead of a fixed scalar hyperparameter.

Note that the only thing ACAI encourages is that interpolated points appear realistic.

The critic only ever sees a single reconstruction or interpolant at a time; it is never fed real datapoints or latent vectors.

It therefore will only be able to successfully recover ?? if the quality of the autoencoder's output degrades consistently across an interpolation as a function of ?? (as seen, for example, in fig. 3a where interpolated points become successively blurrier and darker).

ACAI's primary purpose is to discourage this behavior.

In doing so, it may implicitly modify the structure of the latent space learned by the autoencoder, but ACAI itself does not directly impose a specific structure.

Our goal in introducing ACAI is to test whether simply encouraging better interpolation behavior produces a better representation for downstream tasks.

Further, in contrast with the standard Generative Adversarial Network (GAN) setup BID12 , ACAI does not distinguish between "real" and "fake" data; rather, it simply attempts to regress the interpolation coefficient ??.

Furthermore, GANs are a generative modeling technique, not a representation learning technique; in this paper, we focus on autoencoders and their ability to learn useful representations.

How can we measure whether an autoencoder interpolates effectively and whether our proposed regularization strategy achieves its stated goal?

As mentioned in section 2, defining interpolation relies on the notion of "semantic similarity" which is a vague and problem-dependent concept.

For example, a definition of interpolation along the lines of "??z 1 + (1 ??? ??)z 2 should map to ??x 1 + (1 ??? ??)x 2 " is overly simplistic because interpolating in "data space" often does not result in realistic datapointsin images, this corresponds to simply fading between the pixel values of the two images.

Instead, we might hope that our autoencoder smoothly morphs between salient characteristics of x 1 and x 2 , even when they are dissimilar.

Put another way, we might hope that decoded points along the interpolation smoothly traverse the underlying manifold of the data instead of simply interpolating in data space.

However, we rarely have access to the underlying data manifold.

To make this problem more concrete, we introduce a simple benchmark task where the data manifold is simple and known a priori which makes it possible to quantify interpolation quality.

We then evaluate the ability of various common autoencoders to interpolate on our benchmark.

Finally, we test ACAI on our benchmark and show that it exhibits dramatically improved performance and qualitatively superior interpolations.

Given that the concept of interpolation is difficult to pin down, our goal is to define a task where a "correct" interpolation between two datapoints is unambiguous and well-defined.

This will allow us to quantitatively evaluate the extent to which different autoencoders can successfully interpolate.

Towards this goal, we propose the task of autoencoding 32 ?? 32 grayscale images of lines.

We consider 16-pixel-long lines beginning from the center of the image and extending outward at an angle ?? ??? [0, 2??] (or put another way, lines are radii of the circle circumscribed within the image borders).

An example of 16 such images is shown in FIG2 (appendix A.1).

In this task, the data manifold can be defined entirely by a single variable: ??. We can therefore define a valid interpolation from x 1 to x 2 as one which smoothly and linearly adjusts ?? from the angle of the line in x 1 to the angle in x 2 .

We further require that the interpolation traverses the shortest path possible along the data manifold.

We provide some concrete examples of good and bad interpolations, shown and described in appendix A.1.On any dataset, our desiderata for a successful interpolation are that intermediate points look realistic and provide a semantically meaningful morphing between its endpoints.

On this synthetic lines dataset, we can formalize these notions as specific evaluation metrics, which we describe in detail in appendix A.2.

To summarize, we propose two metrics: Mean Distance and Smoothness.

Mean Distance measures the average distance between interpolated points and "real" datapoints.

Smoothness measures whether the angles of the interpolated lines follow a linear trajectory between the angle of the start and endpoint.

Both of these metrics are simple to define due to our construction of a dataset where we exactly know the data distribution and manifold; we provide a full definition and justification in appendix A.2.

A perfect alignment would achieve 0 for both scores; larger values indicate a failure to generate realistic interpolated points or produce a smooth interpolation respectively.

By choosing a synthetic benchmark where we can explicitly measure the quality of an interpolation, we can confidently evaluate different autoencoders on their interpolation abilities.

To evaluate an autoencoder on the synthetic lines task, we randomly sample line images during training and compute our evaluation metrics on a separate randomly-sampled test set of images.

Note that we never train any autoencoder explicitly to produce an optimal interpolation; "good" interpolation is an emergent property which occurs only when the architecture, loss function, training procedure, etc. produce a suitable latent space.

In this section, we describe various common autoencoder structures and objectives and try them on the lines task.

Our goal is to quantitatively evaluate the extent to which standard autoencoders exhibit useful interpolation behavior.

Our results, which we describe below, are summarized in table 1.Base Model Perhaps the most basic autoencoder structure is one which simply maps input datapoints through a "bottleneck" layer whose dimensionality is smaller than the input.

In this setup, f ?? and g ?? are both neural networks which respectively map the input to a deterministic latent code z and then back to a reconstructed input.

Typically, f ?? and g ?? are trained simultaneously with respect to x ???x 2 .

We will use this framework as a baseline for experimentation for all of the autoencoder variants discussed below.

In particular, for our base model and all of the other autoencoders we will use the model architecture and training procedure described in appendix B. As a short summary, our encoder consists of a stack of convolutional and average pooling layers, whereas the decoder consists of convolutional and nearest-neighbor upsampling layers.

For experiments on the synthetic "lines" task, we use a latent dimensionality of 64.

Note that, because the data manifold is effectively onedimensional, we might expect autoencoders to be able to model this dataset using a one-dimensional latent code; however, using a larger latent code reflects the realistic scenario where the latent space is larger than necessary.

After training our baseline autoencoder, we achieved a Mean Distance score which was the worst (highest) of all of the autoencoders we studied, though the Smoothness was on par with various other approaches.

In general, we observed some reasonable interpolations when using the baseline model, but found that the intermediate points on the interpolation were typically not realistic as seen in the example interpolation in fig. 3a .Denoising Autoencoder An early modification to the standard autoencoder setup was proposed by BID44 , where instead of feeding x into the autoencoder, a corrupted versionx ??? q(x|x) is sampled from the conditional probability distribution q(x|x) and is fed into the autoencoder instead.

The autoencoder's goal remains to producex which minimizes x ???x 2 .

One justification of this approach is that the corrupted inputs should fall outside of the true data manifold, so the autoencoder must learn to map points from outside of the data manifold back onto it.

This provides an implicit way of defining and learning the data manifold via the coordinate system induced by the latent space.

While various corruption procedures q(x|x) have been used such as masking and salt-and-pepper noise, in this paper we consider the simple case of additive isotropic Gaussian noise wherex ??? N (x, ?? 2 I) and ?? is a hyperparameter.

After tuning ??, we found simply setting ?? = 1.0 to work best.

Interestingly, we found the denoising autoencoder often produced "data-space" interpolation (as seen in fig. 3b ) when interpolating in latent space.

This resulted in comparatively poor Mean Distance and Smoothness scores.

Variational Autoencoder The Variational Autoencoder (VAE) BID18 BID34 introduces the constraint that the latent code z is a random variable distributed according to a prior distribution p(z).

The encoder f ?? can then be considered an approximation to the posterior p(z|x).

Then, the decoder g ?? is taken to parametrize the likelihood p(x|z); in all of our experiments, we consider x to be Bernoulli distributed.

The latent distribution constraint is enforced by an additional loss term which measures the KL divergence between approximate posterior and prior.

VAEs then use log-likelihood for the reconstruction loss (cross-entropy in the case of Bernoulli-distributed data), which results in the following combined loss function: ???E[log g ?? (z)] + KL(f ?? (x)||p(z)) where the expectation is taken with respect to z ??? f ?? (x) and KL(??||??) is the KL divergence.

Minimizing this loss function can be considered maximizing a lower bound (the "ELBO") on the likelihood of the training set, producing a likelihood-based generative model which allows novel data points to be sampled by first sampling z ??? p(z) and then computing g ?? (z).

A common choice is to let q(z|x) be a diagonalcovariance Gaussian, in which case backpropagation through sampling from q(z|x) is feasible via the "reparametrization trick" which replaces z ??? N (??, ??I) with ??? N (0, I), z = ?? + ?? where ??, ?? ??? R dz are the predicted mean and standard deviation produced by f ?? .

Various modified objectives BID15 BID47 , improved prior distributions BID19 BID39 BID17 and improved model architectures BID37 BID8 BID13 have been proposed to better the VAE's performance on downstream tasks, but in this paper we solely consider the "vanilla" VAE objective and prior described above applied to our baseline autoencoder structure.

When trained on the lines benchmark, we found the VAE was able to effectively model the data distribution (see samples, fig. 5 in appendix C) and accurately reconstruct inputs.

In interpolations produced by the VAE, intermediate points tend to look realistic, but the angle of the lines do not follow a smooth or short path ( fig. 3c ).

This resulted in a very good Mean Distance score but a very poor Smoothness score.

Contrary to expectations, this suggests that desirable interpolation behavior may not follow from an effective generative model of the data distribution.

Adversarial Autoencoder The Adversarial Autoencoder (AAE) BID27 proposes an alternative way of enforcing structure on the latent code.

Instead of minimizing a KL divergence between the distribution of latent codes and a prior distribution, a critic network is trained in tandem with the autoencoder to predict whether a latent code comes from f ?? or from the prior p(z).

The autoencoder is simultaneously trained to reconstruct inputs (via a standard reconstruction loss) and to "fool" the critic.

The autoencoder is allowed to backpropagate gradients through the critic's loss function, but the autoencoder and critic parameters are optimized separately.

This effectively computes an "adversarial divergence" between the latent code distribution and the chosen prior.

This framework was later generalized and referred to as the "Wasserstein Autoencoder" BID38 One advantage of this approach is that it allows for an arbitrary prior (as opposed to those which have a tractable KL divergence).

The disadvantages are that the AAE no longer has a probabilistic interpretation and involves optimizing a minimax game, which can cause instabilities.

Using the AAE requires choosing a prior, a critic structure, and a training scheme for the critic.

For simplicity, we also used a spherical Gaussian prior for the AAE.

We experimented with various architectures for the critic, and found the best performance with a critic which consisted of two dense layers, each with 100 units and a leaky ReLU nonlinearity.

We found it satisfactory to simply use the same optimizer and learning rate for the critic as was used for the autoencoder.

On our lines benchmark, the AAE typically produced smooth interpolations, but exhibited degraded quality in the middle of interpolations ( fig. 3d ).

This behavior produced the best Smoothness score among existing autoencoders, but a relatively poor Mean Distance score.

The Vector Quantized Variational Autoencoder (VQ-VAE) was introduced by (van den Oord et al., 2017) as a way to train discrete-latent autoencoders using a learned codebook.

In the VQ-VAE, the encoder f ?? (x) produces a continuous hidden representation z ??? R d z which is then mapped to z q , its nearest neighbor in a "codebook" {e j ??? R dz , j ??? 1, . . .

, K}. z q is then passed to the decoder for reconstruction.

The encoder is trained to minimize the reconstruction loss using the straight-through gradient estimator BID2 , together with a commitment loss term ?? z ??? sg(z q ) (where ?? is a scalar hyperparameter) which encourages encoder outputs to move closer to their nearest codebook entry.

Here sg denotes the stop gradient operator, i.e. sg(x) = x in the forward pass, and sg(x) = 0 in the backward pass.

The codebook entries e j are updated as an exponential moving average (EMA) of the continuous latents z that map to them at each training iteration.

The VQ-VAE training procedure using this EMA update rule can be seen as performing the K-means or the hard Expectation Maximization (EM) algorithm on the latent codes BID36 .We perform interpolation in the VQ-VAE by interpolating continuous latents, mapping them to their nearest codebook entries, and decoding the result.

Assuming sufficiently large codebook, a semantically "smooth" interpolation may be possible.

On the lines task, we found that this procedure produced poor interpolations.

Ultimately, many entries of the codebook were mapped to unrealistic datapoints, and the interpolations resembled those of the baseline autoencoder.

Adversarially Constrained Autoencoder Interpolation Finally, we turn to evaluating our proposed adversarial regularizer for improving interpolations.

For simplicity, on the lines benchmark we found it sufficient to use a critic architecture which was equivalent to the encoder (as described in appendix B).

To produce a single scalar value from its output, we computed the mean of its final layer activations.

For the hyperparameters ?? and ?? we found values of 0.5 and 0.2 to achieve good results, though the performance was not very sensitive to their values.

We use these values for the coefficients for all of our experiments.

Finally, we trained the critic using the same optimizer and hyperparameters as the autoencoder.

We found dramatically improved performance on the lines benchmark when using ACAI -it achieved the best Mean Distance and Smoothness score among the autoencoders we considered.

When inspecting the resulting interpolations, we found it occasionally chose a longer path than necessary but typically produced "perfect" interpolation behavior as seen in fig. 3f .

This provides quantitative evidence ACAI is successful at encouraging realistic and smooth interpolations.

We have so far only discussed results on our synthetic lines benchmark.

We also provide example reconstructions and interpolations produced by each autoencoder for MNIST BID22 , SVHN BID31 , and CelebA BID24 in appendix D. For each dataset, we trained autoencoders with latent dimensionalities of 32 and 256.

Since we do not know the underlying data manifold for these datasets, no metrics are available to evaluate performance and we can only make qualitative judgments as to the reconstruction and interpolation quality.

We find that most autoencoders produce "blurrier" images with d z = 32 but generally give smooth interpolations regardless of the latent dimensionality.

The exception to this observation was the VQ-VAE which seems generally to work better with d z = 32 and occasionally even diverged for d z = 256 (see e.g. fig. 9e ).

This may be due to the nearest-neighbor discretization (and gradient estimator) failing in high dimensions.

Across datasets, we found the VAE and denoising autoencoder typically produced more blurry interpolations.

AAE and ACAI generally produced realistic interpolations, even between dissimilar datapoints (for example, in fig. 7 bottom) .

The baseline model often effectively interpolated in data space.

We have so far solely focused on measuring the interpolation abilities of different autoencoders.

Now, we turn to the question of whether improved interpolation is associated with improved performance on downstream tasks.

Specifically, we will evaluate whether using our proposed regularizer results in latent space representations which provide better performance in supervised learning and clustering.

Put another way, we seek to test whether improving interpolation results in a latent representation which has disentangled important factors of variation (such as class identity) in the dataset.

To answer this question, we ran classification and clustering experiments using the learned latent spaces of Table 3 : Clustering accuracy for using K-Means on the latent space of different autoencoders (left) and previously reported methods (right).

On the right, "Data" refers to performing K-Means directly on the data and DEC, RIM, and IMSAT are the methods proposed in BID46 BID20 BID16 respectively.

Results marked * are excerpted from BID16 and ** are from BID46 .

Single-Layer Classifier A common method for evaluating the quality of a learned representation (such as the latent space of an autoencoder) is to use it as a feature representation for a simple, one-layer classifier (i.e. logistic regression) trained on a supervised learning task .

The justification for this evaluation procedure is that a learned representation which has effectively disentangled class identity will allow the classifier to obtain reasonable performance despite its simplicity.

To test different autoencoders in this setting, we trained a separate single-layer classifier in tandem with the autoencoder using the latent representation as input.

We did not optimize autoencoder parameters with respect to the classifier's loss, which ensures that we are measuring unsupervised representation learning performance.

We repeated this procedure for latent dimensionalities of 32 and 256 (MNIST and SVHN) and 256 and 1024 (CIFAR-10).Our results are shown in table 2.

In all settings, using ACAI instead of the baseline autoencoder upon which it is based produced significant gains -most notably, on SVHN with a latent dimensionality of 256, the baseline achieved an accuracy of only 22.74% whereas ACAI achieved 85.14%.

In general, we found the denoising autoencoder, VAE, and ACAI obtained significantly higher performance compared to the remaining models.

On MNIST and SVHN, ACAI achieved the best accuracy by a significant margin; on CIFAR-10, the performance of ACAI and the denoising autoencoder was similar.

By way of comparison, we found a single-layer classifier applied directly to (flattened) image pixels achieved an accuracy of 92.31%, 23.48%, and 39.70% on MNIST, SVHN, and CIFAR-10 respectively, so classifying using the representation learned by ACAI provides a huge benefit.

Clustering If an autoencoder groups points with common salient characteristics close together in latent space without observing any labels, it arguably has uncovered some important structure in the data in an unsupervised fashion.

A more difficult test of an autoencoder is therefore clustering its latent space, i.e. separating the latent codes for a dataset into distinct groups without using any labels.

To test the clusterability of the latent spaces learned by different autoencoders, we simply apply K-Means clustering BID26 to the latent codes for a given dataset.

Since K-Means uses Euclidean distance, it is sensitive to each dimension's relative variance.

We therefore used PCA whitening on the latent space learned by each autoencoder to normalize the variance of its dimensions prior to clustering.

K-Means can exhibit highly variable results depending on how it is initialized, so for each autoencoder we ran K-Means 1,000 times from different random initializations and chose the clustering with the best objective value on the training set.

For evaluation, we adopt the methodology of BID46 BID16 : Given that the dataset in question has labels (which are not used for training the model, the clustering algorithm, or choice of random initialization), we can cluster the data into C distinct groups where C is the number of classes in the dataset.

We then compute the "clustering accuracy", which is simply the accuracy corresponding to the optimal one-to-one mapping of cluster IDs and classes BID46 .Our results are shown in table 3.

On both MNIST and SVHN, ACAI achieved the best or second-best performance for both d z = 32 and d z = 256.

We do not report results on CIFAR-10 because all of the autoencoders we studied achieved a near-random clustering accuracy.

Previous efforts to evaluate clustering performance on CIFAR-10 use learned feature representations from a convolutional network trained on ImageNet BID16 which we believe only indirectly measures unsupervised learning capabilities.

In this paper, we have provided an in-depth perspective on interpolation in autoencoders.

We proposed Adversarially Constrained Autoencoder Interpolation (ACAI), which uses a critic to encourage interpolated datapoints to be more realistic.

To make interpolation a quantifiable concept, we proposed a synthetic benchmark and showed that ACAI substantially outperformed common autoencoder models.

This task also yielded unexpected insights, such as that a VAE which has effectively learned the data distribution might not interpolate.

We also studied the effect of improved interpolation on downstream tasks, and showed that ACAI led to improved performance for feature learning and unsupervised clustering.

These findings confirm our intuition that improving the interpolation abilities of a baseline autoencoder can also produce a better learned representation for downstream tasks.

However, we emphasize that we do not claim that good interpolation always implies a good representation -for example, the AAE produced smooth and realistic interpolations but fared poorly in our representations learning experiments and the denoising autoencoder had low-quality interpolations but provided a useful representation.

In future work, we are interested in investigating whether our regularizer improves the performance of autoencoders other than the standard "vanilla" autoencoder we applied it to.

In this paper, we primarily focused on image datasets due to the ease of visualizing interpolations, but we are also interested in applying these ideas to non-image datasets.

A LINE BENCHMARK

Some example data and interpolations for our synthetic lines benchmark are shown in FIG2 .

Full discussion of this benchmark is available in section 3.1.

We define our Mean Distance and Smoothness metrics as follows: Let x 1 and x 2 be two input images we are interpolating between and DISPLAYFORM0 be the decoded point corresponding to mixing x 1 and x 2 's latent codes using coefficient ?? = n???1 /N???1.

The imagesx n , n ??? {1, . . .

, N } then comprise a length-N interpolation between x 1 and x 2 .

To produce our evaluation metrics, we first find the closest true datapoint (according to cosine distance) for each of the N intermediate images along the interpolation.

Finding the closest image among all possible line images is infeasible; instead we first generate a size-D collection of line images D with corresponding angles ?? q , q ??? {1, . . .

, D} spaced evenly between 0 and 2??.

Then, we match each image in the interpolation to a real datapoint by finding DISPLAYFORM1 for n ??? {1, . . .

, N }, where C n,q is the cosine distance betweenx n and the qth entry of D. To capture the notion of "intermediate points look realistic", we compute DISPLAYFORM2 We now define a perfectly smooth interpolation to be one which consists of lines with angles which linearly move from the angle of D q 1 to that of D q N .

Note that if, for example, the interpolated lines go from ?? q 1 = ?? /10 to ?? q N = 19?? /10 then the angles corresponding to the shortest path will have a discontinuity from 0 to 2??.

To avoid this, we first "unwrap" the angles {?? q 1 , . . .

, ?? q N } by removing discontinuities larger than ?? by adding multiples of ??2?? when the absolute difference between ?? q n???1 and ?? q n is greater than ?? to produce the angle sequence {?? q 1 , . . .

,?? q N }.

2 Then, we define a measure of smoothness as DISPLAYFORM3 In other words, we measure the how much larger the largest change in (normalized) angle is compared to the minimum possible value ( 1 /(N???1)).

B BASE MODEL ARCHITECTURE AND TRAINING PROCEDUREAll of the autoencoder models we studied in this paper used the following architecture and training procedure: The encoder consists of blocks of two consecutive 3 ?? 3 convolutional layers followed by 2 ?? 2 average pooling.

All convolutions (in the encoder and decoder) are zero-padded so that the input and output height and width are equal.

The number of channels is doubled before each average pooling layer.

Two more 3 ?? 3 convolutions are then performed, the last one without activation and the final output is used as the latent representation.

All convolutional layers except for the final use a leaky ReLU nonlinearity BID25 .

For experiments on the synthetic "lines" task, the convolution-average pool blocks are repeated 4 times until we reach a latent dimensionality of 64.For subsequent experiments on real datasets (section 4), we repeat the blocks 3 times, resulting in a latent dimensionality of 256.The decoder consists of blocks of two consecutive 3 ?? 3 convolutional layers with leaky ReLU nonlinearities followed by 2 ?? 2 nearest neighbor upsampling BID32 .

The number of channels is halved after each upsampling layer.

These blocks are repeated until we reach the target resolution (32 ?? 32 in all experiments).

Two more 3 ?? 3 convolutions are then performed, the last one without activation and with a number of channels equal to the number of desired colors.

All parameters are initialized as zero-mean Gaussian random variables with a standard deviation of 1 / ??? fan_in(1+0.

In fig. 5 , we show some samples from our VAE trained on the synthetic line benchmark.

The VAE generally produces realistic samples and seems to cover the data distribution well, despite the fact that it does not produce high-quality interpolations ( fig. 3c ).

In this section, we provide a series of figures (figs. 6 to 11) showing interpolation behavior for the different autoencoders we studied.

Further discussion of these results is available in section 3.

<|TLDR|>

@highlight

We propose a regularizer that improves interpolation and autoencoders and show that it also improves the learned representation for downstream tasks.