Bayesian inference is used extensively to infer and to quantify the uncertainty in a field of interest from a measurement of a related field when the two are linked by a mathematical model.

Despite its many applications, Bayesian inference faces challenges when inferring fields that have discrete representations of large dimension, and/or have prior distributions that are difficult to characterize mathematically.

In this work we demonstrate how the approximate distribution learned by a generative adversarial network (GAN) may be used as a prior in a Bayesian update to address both these challenges.

We demonstrate the efficacy of this approach by inferring and quantifying uncertainty in a physics-based inverse problem and an inverse problem arising in computer vision.

In this latter example, we also demonstrate how the knowledge of the spatial variation of uncertainty may be used to select an optimal strategy of placing the sensors (i.e. taking measurements), where information about the image is revealed one sub-region at a time.

Bayesian inference is a principled approach to quntify uncertainty in inverse problems that are constrained by mathematical model (Kaipio and Somersalo [2006] , Dashti and Stuart [2016] , Polpo et al. [2018] ).

It has found applications in diverse fields such as geophysics (Gouveia and Scales [1997] , Martin et al. [2012] , Isaac et al. [2015] ), climate modeling (Jackson et al. [2004] ), chemical kinetics ), heat conduction (Wang and Zabaras [2004] ), astrophysics (Loredo [1990] , Asensio Ramos et al. [2007] ), materials modeling (Sabin et al. [2000] ) and the detection and diagnosis of disease (Siltanen et al. [2003] , Kolehmainen et al. [2006] ).

The two critical ingredients of a Bayesian inference problem are -an informative prior representing the prior belief about the parameters and an efficient method for sampling from the posterior distribution.

In this manuscript we describe how a deep generative model (generative adversarial networks (GANs)) can be used in these roles.

In a typical inverse problem, we wish to infer a vector of parameters x ??? R N from the measurement of a related vector y ??? R P , where the two are related through a forward model y = f (x).

A noisy measurement of y is denoted by?? = f (x) + ??, where ?? ??? R P represents noise.

While the forward map is typically well-posed, its inverse is not, and hence to infer x from the measurement?? requires techniques that account for this ill-posedness.

Classical techniques based on regularization tackle this ill-posedness by using additional information about the sought parameter field explicitly or implicitly (Tarantola [2005] ).

Bayesian inference offers a different solution to this problem by modeling the unknown parameter and the measurements as random variables and allows for the characterization of the uncertainty in the inferred parameter field.

For additive noise, the posterior distribution of x, determined using Bayes' theorem after accounting for the observation?? is given by

where Z is the prior-predictive distribution of y, p prior X (x) is the prior distribution of x, and p l (y|x) is the likelihood, often determined by the distribution of the error in the model, denoted by p ?? .

Despite its numerous applications, Bayesian inference faces significant challenges.

These include constructing a reliable and informative prior distribution from a collection of prior measurements denoted by the S = {x

(1) , ?? ?? ?? , x (S) }, and efficiently sampling from the posterior distribution when the dimension of x is large.

In this work we consider the use of GANs (Goodfellow et al. [2014] ) in addressing these challenges.

These networks are useful in this role because of (a) they are able to generate samples of x from p gen X (x) while ensuring closeness (in an appropriate measure) between p gen X (x) and the true distribution, and (b) because they accomplish this by sampling from the much simpler distribution of the latent vector z, whose dimension is much smaller than that of x.

Related work and our contribution: The main idea in this work involves training a GAN using the sample set S, and then using the distribution learned by the GAN as the prior distribution in Bayesian inference.

This leads to a useful method for representing complex prior distributions and an efficient approach for sampling from the posterior distribution in terms of the latent vector z.

The solution of inverse problems using sample-based priors has a rich history (see Vauhkonen et al. [1997] , Calvetti and Somersalo [2005] for example).

As does the idea of dimension reduction in parameter space , Lieberman et al. [2010] ).

However, the use of GANs in these tasks is novel.

Recently, a number of authors have considered the use machine learning-based methods for solving inverse problems.

These include the use of convolutional neural networks (CNNs) to solve physics-driven inverse problems (Adler and ??ktem [2017] , Jin et al. [2017] , Patel et al. [2019] ), and GANs to solve problems in computer vision (Chang et al., Kupyn et al. [2018] , Yang et al. [2018] , Ledig et al., Anirudh et al. [2018] , Isola et al. [2016] , Zhu et al. [2017] , Kim et al. [2017] ).

There is also a growing body of work on using GANs to learn regularizers in inverse problems (Lunz et al. [2018] ) and in compressed sensing (Bora et al. [2017 (Bora et al. [ , 2018 , Kabkab et al. [2018] , Wu et al. [2019] , Shah and Hegde [2018] ).

However, these approaches differ from ours in that they solve the inverse problem as an optimization problem and do not quantify uncertainty in a Bayesian framework .

More recently, the approach described in (Adler and ??ktem [2018] ) utilizes GANs in a Bayesian setting; however the GAN is trained to approximate the posterior distribution, and training is done in a supervised fashion with paired samples of the measurement?? and the corresponding true solution x.

Let z ??? p Z (z) characterize the latent vector space and g(z) be the generator of a GAN trained using S. Then with infinite capacity and sufficient data, the generator learns the true distribution (Goodfellow et al. [2014] ).

That is, p

(2) Here p Z is the multivariate distribution of the latent vector whose components are iid and typically conform to a Gaussian or a uniform distribution.

Now consider a measurement?? for which we would like to infer the posterior distribution of x. For this we use (1) and set the prior distribution to be equal to the true distribution, that is p

Using this it is easy to show that for any l(x),

where E is the expectation operator, and

Note that the distribution p post Z is the analog of p post X in the latent vector space.

The measurement y updates the prior distribution for x to the posterior distribution; similarly, it updates the prior distribution for z, p Z , to the posterior distribution, p post Z .

Equation (4) implies that sampling from the posterior distribution for x is equivalent to sampling from the posterior distribution for z and transforming the sample through the generator g. That is,

Since the dimension of z is typically smaller than that of x, and since the operation of the generator is typically inexpensive, this represents an efficient approach to sampling from the posterior of x.

As mentioned in section 1, we wish to infer and characterize the uncertainty in the vector of parameters x from a noisy measurement?? , where f is a known map that connects x and y. We also have several prior measurements of x, contained in the set S. To solve this problem we train a GAN with a generator g(z) on S, and then sample x from p post X (x|y) given in (6).

Since GANs can be used to represent complex distributions efficiently, this algorithm provides a means of including complex priors that are defined by samples.

It also leads to an efficient approach to sampling from p post X (x|y) since the dimension of z is typically smaller (10 1 -10 2 ) than that of x (10 4 -10 7 ).

In Appendix A we describe approaches based on Monte-Carlo, Markov-Chain Monte-Carlo and MAP estimation for estimating population parameters of the posterior that make use of this observation.

A problem motivated by physics We apply our approach to the problem of determining the initial temperature distribution of a solid from a measurement of its current temperature.

The inferred field (x) is represented on a 32 2 grid on a square and the forward operator is defined by the solution of the time-dependent heat conduction problem with uniform conductivity.

This operator maps the initial temperature to the temperature at time t = 1, and its discrete version is generated by approximating the time-dependent linear heat conduction equation using central differences in space and backward difference in time.

It is assumed that the initial temperature is zero everywhere except in a rectangular region, and it is parameterized by the horizontal and vertical coordinates of two corners of the rectangular region and the value of the temperature field within it.

50,000 initial temperature fields sampled from this distribution are included in the sample set S used to train a Wasserstein GAN (WGAN-GP (Gulrajani et al. [2017] )) with an 8-dimensional latent space with batch size of 64 and learning rate of 0.0002.

The target field we wish to infer is shown in Figure 1a .

This field is passed through the forward map to generate the noise-free and the noisy versions (Gaussian with zero mean and unit variance) of the measured field shown in Figure 1b and 1c.

We apply the algorithms developed in the previous section to probe the posterior distribution.

We first use these to determine the MAP estimate for the posterior distribution of the latent vector (denoted by z map ).

The value of g(z map ) is shown in Figure 1d .

By comparing this with the true value of the inferred field, shown in Figure 1a , we observe that the MAP estimate is very close to the true value.

This agreement is remarkable if we recognize that the ratio of noise to signal is around 30%, and also compare the MAP estimates obtained using an H 1 or an L 2 prior (see Figures 1e and 1f) with the true value.

Figure 2: Iterative image recovery with very sparse measurements using uncertainty information: for each digit left most column represents true signal (x * ) and its noisy version.

The following columns represent the sparse measurement, the estimated MAP, and the estimated variance, respectively at each iteration.

The red window in the variance map is the sub-region with maximum variance.

Next, we consider the results obtained by sampling from the MCMC approximation to the posterior distribution of z defined in (5).

The MCMC approximation to the mean of the inferred field computed using (8) is shown in Figure 1g .

We observe that the edges and the corners of the temperature field are smeared out.

This indicates the uncertainty in recovering the values of the initial field along these locations, which can be attributed to the smoothing nature of the forward operator especially for the higher modes.

A more precise estimate of the uncertainty in the inferred field is provided by the variance of the inferred initial temperature at each spatial location.

In Figure 1h we have plotted the point-wise standard deviation (square-root of the diagonal of co-variance) of the inferred fieldour metric of quantified uncertainty.

We observe that it is largest along the edges and at the corners, where the forward operator has smoothed out the initial data, and thus introduced large levels of uncertainty in the location of these features.

Additional examples of this inverse heat conduction problem with different target fields is shown in Appendix B. A problem in computer vision: Next we consider a problem in computer vision that highlights the utility of estimating the uncertainty in an inference problem: one of determining the noise-free version of an image from a noisy version of a sub-region of the image.

In particular, we consider an iterative version of this problem, where one sub-region is revealed in each iteration, and the user is given the freedom to select this sub-region.

We use a strategy that is based on selecting a region where the variance is maximum, and conclude that we arrive at a very good guess for the image in very few iterations.

This task falls under active learning regime of machine learning and is useful when measurements are expensive.

We use 55,000 images from the MNIST data set to train a WGAN-GP and use it as a prior in Bayesian inference.

We select an image from the complementary set, add Gaussian noise with 0.8 variance, mask regions within this image, and use it to infer the original image.

We utilize a forward map that is zero in the masked region and identity everywhere else.

We begin by masking the entire image, and allow the user to select the sub-region (which is a square with edge length equal to 1/7th of the original image) in each iteration.

We report results when the user selects the sub-region with maximum variance as the sub-region to be revealed in the next iteration.

For computing the variance we utilize the algorithm developed in this work.

In Figure 2 we have shown the true image and results from several iterations for two different MNIST digits from test set.

For each iteration, we have shown the image that was used as measurement, the corresponding MAP and variance determined using our algorithms.

We observe that in the 0 th iteration, when nothing is revealed in the measurement, the variance is largest in the center of the image where most digits assume different intensities.

This leads to the user requesting a measurement in this region in the subsequent iteration.

Thereafter, the estimated variance reduces with each iteration, and we converge to an image which is very close to the true image in very few (2-3) iterations.

Additional results for MNIST and CelebA dataset are provided in Appendix B.

Here we provide additional examples of iterative image recovery scheme described in section 3 for MNIST (figure 4) and CelebA (figure 6) dataset.

We also compare the performance of this variance-driven iterative strategy to random sampling scheme, where the next sub-region is selected randomly (figure 5).

Figure 6: Estimate of the MAP (3rd row), mean (4th row) and variance (5th row) from the limited view of a noisy image (2nd row) using the proposed method.

The window to be revealed at a given iteration (shown in red box) is selected using a variance-driven strategy.

Top row indicates ground truth.

For all images additive Gaussian noise with variance=1 is used.

<|TLDR|>

@highlight

Using GANs as priors for efficient Bayesian inference of complex fields.