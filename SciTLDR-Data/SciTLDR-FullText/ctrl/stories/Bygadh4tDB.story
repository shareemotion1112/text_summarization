Stochastic neural networks with discrete random variables are an important class of models for their expressivity and interpretability.

Since direct differentiation and backpropagation is not possible, Monte Carlo gradient estimation techniques have been widely employed for training such models.

Efficient stochastic gradient estimators, such Straight-Through and Gumbel-Softmax, work well for shallow models with one or two stochastic layers.

Their performance, however, suffers with increasing model complexity.

In this work we focus on stochastic networks with multiple layers of Boolean latent variables.

To analyze such such networks, we employ the framework of harmonic analysis for Boolean functions.

We use it to derive an analytic formulation for the source of bias in the biased Straight-Through estimator.

Based on the analysis we propose \emph{FouST}, a simple gradient estimation algorithm that relies on three simple bias reduction steps.

Extensive experiments show that FouST performs favorably compared to state-of-the-art biased estimators, while being much faster than unbiased ones.

To the best of our knowledge FouST is the first gradient estimator to train up very deep stochastic neural networks, with up to 80 deterministic and 11 stochastic layers.

Stochastic neural networks with discrete latent variables have been an alluring class of models for their expressivity and interpretability, dating back to foundational work on Helmholtz machines (Dayan et al., 1995) and sigmoid belief nets (Neal, 1992) .

Since they are not directly differentiable, discrete random variables do not mesh well with the workhorse of modern Deep Learning, that is the backpropagation algorithm.

Monte Carlo gradient estimation is an effective solution where, instead of computing the true gradients, one can sample gradients from some distribution.

The sample estimates can be either biased or unbiased.

Unbiased gradient estimates like score function estimators (Williams, 1992) come typically at the cost of high variance leading to slow learning.

In contrast, biased gradient estimates such Straight-Through (Bengio et al., 2013) , while efficient, run the risk of convergence to poor minima and unstable training.

To this end several solutions have recently been proposed that either reduce variance in unbiased estimators (Mnih & Gregor, 2014; Gu et al., 2015; Tucker et al., 2017; Rezende et al., 2014; Grathwohl et al., 2017) or control bias in biased estimators (Jang et al., 2016; Maddison et al., 2016) .

These methods, however, have difficulty scaling up to complex neural networks with multiple stochastic layers: low-variance unbiased estimators are too expensive 1 , while the compounded bias from the continuous relaxations on multiple stochastic layers leads to poor minima.

In this work we focus on biased estimators.

Our goal in this paper is a gradient estimator for Boolean random variables that works for any complex -deep or wide-neural network architecture.

We resort to the term Boolean instead of binary to emphasize that we work directly on the Boolean space {???1, +1}, without any continuous relaxations or quantizations.

With this in mind we re-purpose the framework of harmonic analysis of Boolean functions, widely used in computational learning and computational complexity theory (O'Donnell, 2014; Linial et al., 1993; Mossel et al., 2003; Mansour, 1994) .

We cast stochastic neural networks as Boolean functions f (z) over Boolean latent variables z sampled from probability 1.

We introduce the framework of harmonic analysis of Boolean functions to analyze discrete stochastic neural networks and their REINFORCE and Straight-Through gradients.

We show that stochastic gradients compute Fourier coefficients.

2.

Based on the above harmonic analysis we present FouST -a low-bias gradient estimator for Boolean latent variables based on three bias reduction steps.

As a side contribution, we show that the gradient estimator employed with DARN (Gregor et al., 2013) , originally proposed for autoregressive models, is a strong baseline for gradient estimation in large and complex models with many stochastic layers.

3.

We show that FouST is amenable to complex stochastic neural networks with Boolean random variables.

To the best of our knowledge, FouST is the first gradient estimate algorithm that can train very deep stochastic neural networks with Boolean latent variables.

The practical outcome is a simple gradient estimate algorithm that can be plugged in complex stochastic neural networks with multiple layers of Boolean random variables.

We consider Boolean functions on the n-dimensional Boolean cube, f : {???1, +1} n ??? R. The setting of Harmonic Analysis for Boolean functions is the space of Boolean functions f with a product probability distribution on the Boolean input, that is p(z) = n i=1 p i (z).

We denote p i as the probability of the i-th dimension being one, i.e., p i := p i (z i = +1).

We denote the mean and variance of z i by ?? i and ?? i , respectively.

An example of a Boolean function in this setting is a generative neural network f : z ??? y with a factorized latent distribution, as commonly done in representation learning (Kingma & Welling, 2013; Higgins et al., 2017) .

In this example, z is the stochastic input -also known as the latent code in stochastic neural networks -taking only two possible values and y is the output, like cross entropy loss.

Often, the goal of a generative neural network is to learn or approximate the latent distribution given input data x, i.e., p(z|x), as we will also explore in the experiments.

We first introduce a few basic operations in the context of Harmonic Analysis of Boolean functions, which we shall use further on.

Further necessary details are in the Appendix A. For a more comprehensive introduction, however, we refer the reader to O'Donnell (2014) .

Inner product.

The inner product of two Boolean functions f and g is:

Fourier expansion.

Let S be any subset of dimensions of the n-dimensional Boolean cube, S ???

[n] = {1, ..., n}. Per S we define a basis function, ?? S (z) := i???S ?? i (z i ), where for the empty set ?? ??? (z) = 1 and ?? i is the z-score normalized dimension, i.e., ?? i := zi?????i ??i .

For example, under the uniform Bernoulli distribution for the i-th dimension,

The 2 n functions ?? S form an orthonormal basis for the space of Boolean functions f ,

, for i = j, where the expectations compute the inner product between two Boolean functions.

The last identity derives from the independence of any dimensions i = j.

We can then expand the Boolean function f on the set of 2 n orthonormal basis functions,

also known as the p-biased Fourier expansion of f .

Thef (p) (S) are the Fourier coefficients computed by the inverse Fourier expansion,

That is, the inverse expansion is computed by the inner product for Boolean functions defined above.

The cardinality of S is the degree of the coefficientf (p) (S).

For instance, we have only one degree-0 coefficient,f (p) (???), which equals to the expected value of f under the distribution p(z),

Further, we have n degree-1 coefficientsf (p) (i) = f, ?? i and so on.

We examine Straight-Through gradient estimates using the framework of Harmonic Analysis of Boolean functions.

For training the model parameters with a loss function L := E p(z) [f (z)]

we want to compute the gradient ??? pi L = ??? pi E p(z) [f (z)] for the i-th dimension.

As the random sampling operation in the expectation is not differentiable, Bengio et al. (2013) propose the Straight-Through estimator that approximates the gradient with

.

Clearly, the Straight-Through computes a biased approximation to the true gradient.

Next, we quantify the bias in the Straight-Through estimator using the harmonic analysis of f (x).

For the quantification of bias we first need the following lemma that connects the REINFORCE gradients with the degree-1 Fourier coefficients.

The lemma is an extension of Margulis-Russo (Margulis, 1974; Russo, 1982; O'Donnell, 2014) formula.

Lemma 1.

Let f be a Boolean function.

Then, the REINFORCE gradient estimates the degree 1 Fourier coefficientsf

Proof.

For compactness and clarity, we provide the proof in the Appendix B.1.

We introduce new notation.

In the following lemma bias

) denotes the bias of the i-th gradient estimate under the distribution p. Also, given the distribution p(z), p i???1/2 (z) is the distribution for which we set p(z i = +1) = p(z i = ???1) = 1/2 for a given dimension i. Lemma 2.

Let f be a Boolean function,

where c k are the Taylor coefficients for the i-th dimension on f (z), that is z i , around 0 and bias

Proof.

For compactness and clarity, we provide only a proof sketch here showing the basic steps.

These steps are also needed later in the description of the proposed gradient estimate algorithm.

For the detailed proof please refer to the Appendix B.2.

The proof sketch goes as follows.

First, we derive a relation between the Fourier coefficients under the unknown distribution p(z) and under the uniform Bernoulli distribution p i???1/2 (z).

Then, using this relation we derive the Taylor expansions for the true gradient as well as the Straight-Through gradient estimator.

Last, to prove the lemma we compare the two Taylor expansions.

Relation between Fourier coefficients under p(z) and p i???1/2 (z).

If we expand the function f in terms of its ?? S basis as in equation 2 and focus on the i-th dimension, by Lemma 1 we can show that the REINFORCE gradient is given by

Taylor expansions of the true and the Straight-Through gradients.

The Taylor expansion of

Let's first focus on the true gradient.

Since we work with Boolean ??1 values, we have that z k i = 1 for even k and z k i = z i for odd k. This will influence the even and the odd terms of the Taylor expansions.

Specifically, for the Taylor expansion of the true gradient we can show that

The expression in equation 7 implies that the true gradient with respect to the p i is the expected sum of the odd Taylor coefficients.

Here we note that although the final expression in equation 7 can also be derived by a finite difference method, it does not make explicit, as in equation 31, the dependence on z i and ?? i of the term inside the expectation.

Now, let's focus on the Straight-Through gradient.

Taking the derivative of the Taylor expansion w.r.t.

to z i , we have

The Straight-Through gradient is the expectation of equation 8 in the i-th dimension, that is

where

Comparing the Taylor expansions.

By comparing the expansion of the Straight-Through gradient in equation 10 and the expansion of the true gradient in equation 7 and given that bias

Taking the expectation in equation 9 under p i???1/2 causes the final term in equation 11 to vanish leaving bias

Combining this expression with equation 9 gives the final expression (equation 4) from the lemma.

Inspired by the Harmonic Analysis of the Straight-Through gradient estimates, we present a gradient estimate algorithm for deep Boolean latent models, FouST, for Fourier Straight-Through estimator.

The algorithm relies on three bias reduction steps on the Straight-Through, lines 2, 3, 5 in Algorithm 1.

As detailed earlier, the bias in the Straight-Through estimator is the sum of the bias under the uniform Bernoulli distribution plus extra bias due to non-zero expectation terms in higher-order harmonics when sampling from p(z).

Sampling from p i???1/2 instead of p(z) would decrease the total bias from the form in equation 11 by setting the final term to 0.

As a first bias reduction step, therefore, we do importance sampling.

Specifically, after getting samples from p(z) and computing the gradients ??? zi f (z) with the Straight-Through, we estimate the expectation under p i???1/2 as

Interestingly, Gregor et al. (2013) arrive at equation 12 in the context of unbiased control variates for quadratic functions.

Lemma 2 shows that part of the bias in the Straight-Through estimator is due to the presence of extra factors in the Taylor coefficients.

We can reduce the effect of these factors by taking advantage of the moments of the uniform distribution.

Recalling that

, we can attempt to correct the coefficients in equation 9, which for z k have the form (k + 1)c k , with the same extra factor of k + 1 that appears in the denominator of the kth moment.

This suggests that we can sample from an auxiliary variable u and then use the auxiliary variable u with f instead of z and exploit the higher moments of the uniform distribution to reduce bias.

For brevity, we illustrate the method with a case study of a two-dimensional z, and a bivariate f (z 1 , z 2 ).

As in Lemma 2, the partial true gradient of f (z 1 , z 2 ) w.r.t.

the first distribution parameter p 1 equals to

"Bernoulli splitting uniform" trick.

Assume an auxiliary variable u = (u 1 , u 2 ), which we choose as follows.

First, we sample z = (z 1 , z 2 ) from the uniform Bernoulli distribution p 1???1/2 with (i set to 1).

Then we take a uniform sample (u 1 , u 2 ) with u i sampled from either

The expectation of the gradient under such random sampling is

Further detail is in Appendix C.1.

We compare equation 14 with equation 13.

In equation 14 we observe that the pure terms in z 1 , namely terms with j = 0, always match those of the true gradient in equation 13.

For j > 0 we obtain mixed terms with coefficients that do not match those of the true gradient in equation 13.

Due to the 1 j+1 factor, for small j the mixed-degree terms are closer to the original ones in equation 13.

For functions with small mixed degree terms, this can lead to bias reduction, at the cost of an increased variance because of sampling an auxiliary variable.

In practice, to manage this bias-variance trade-off and to deal with functions that have greater dependence on mixed degree terms, we use smaller intervals for the random samples as in Algorithm 1.

To summarize, for appropriate functions, the "Bernoulli splitting uniform" relies on the continuous variable u conditioned on the binary sample to reduce the bias.

However, it is important to emphasize that u is only an auxiliary variable; the actual latent variable z is always binary.

Thus, the "Bernoulli splitting uniform" trick does not lead to a relaxation of the sort used by Gumbel-Softmax (Jang et al., 2016) , where there are no hard samples.

Lastly we note that for a univariate f the "Bernoulli splitting uniform" trick leads to an unbiased estimator with an increased variance.

The Fourier basis does not depend on the particular input representation and any two-valued set, say {???t, t} can be used as the Boolean representation.

The choice of a representation, however, does affect the bias as we show next.

As a concrete example, we let our input representation be

n , where p i = p(z i = +1/2).

While we can change the input representation like that, in general the Fourier coefficients in equation 3 will be different than for the {???1, +1} representation.

We give the final forms of the gradients here.

Details are given in Appendix C.2.

Under the p i???1/2 distribution the degree-1 Fourier coefficients are:

Note that compared to equation 7, in equation 15 we still get the odd terms c 1 , c 3 albeit decayed by inverse powers of 2.

Following the same process for the Straight-Through gradient as in equation 10, we have that

While this is still biased, compared to equation 7 the bias is reduced by damping higher order terms by inverse powers of 2.

The algorithm, described in algorithm 1, is a Straight-Through gradient estimator with the bias reduction steps described above, where a single sample is used to estimate the gradient.

We emphasize that the algorithm uses a single sample and a single evaluation of the decoder per example and latent vector sample.

Thus, the algorithm has the same complexity as that of the original Straight-Through estimator.

Monte Carlo gradient estimators for training models with stochastic variables can be biased or unbiased.

Perhaps the best known example of an unbiased gradient estimator is the REINFORCE algorithm (Williams, 1992) .

Unfortunately, REINFORCE gives gradients of high variance.

For continuous stochastic variables Kingma & Welling (2013) propose the reparameterization trick, which transforms the random variable into a function of deterministic ones perturbed by a fixed noise source, yielding much lower variance gradient estimates.

For discrete stochastic variables, REINFORCE is augmented with control variates for variance reduction.

A number of control variate schemes have been proposed: NVIL (Mnih & Gregor, 2014) subtracts two baselines (one constant and one input-dependent) from the objective to reduce variance.

MuProp (Gu et al., 2015) uses the first-order Taylor approximation of the function as a baseline.

REBAR (Tucker et al., 2017) uses the Gumbel-Softmax trick to form a control variate for unbiased gradient estimates.

RELAX (Grathwohl et al., 2017) generalizes REBAR to include an auxiliary network in the gradient expression and uses continuous relaxations and the reparameterization trick to give unbiased gradients.

Regarding biased estimators, a simple choice is the Straight-Through estimator (Bengio et al., 2013) which uses the gradient relative to the sample as that relative to the probability parameter.

Another recent approach is to use continuous relaxations of discrete random variables so that the reparameterization trick becomes applicable.

The most common example of this being the GumbelSoftmax estimator (Maddison et al., 2016; Jang et al., 2016) .

Although this is a continuous relaxation, it has been used to define the Gumbel Straight-Through estimator with hard samples.

This uses arg max in the forward pass and the Gumbel-Softmax gradient is used as an approximation during in the backward pass.

DARN (Gregor et al., 2013) , like MuProp, also uses the first-order Taylor expansion as a baseline but does not add the analytical expectation, making the estimator biased for non-quadratic functions.

In this work we focus on biased Straight-Through gradient estimators.

Specifically, we analyse how to reduce bias via Fourier expansions of Boolean functions.

The Fourier expansion itself is widely used in computational learning theory with applications to learning low-degree functions (Kushilevitz & Mansour, 1993) , decision trees (Mansour, 1994) , constant-depth circuits (Linial et al., 1993) and juntas (Mossel et al., 2003) .

To the best of our knowledge we are the first to explore Fourier expansions for bias reduction of biased stochastic gradient estimators.

Experimental Setup.

We first validate FouST on a toy setup, where we already know the analytic expression of f (z).

Next we validate FouST by training generative models using the variational autoencoder framework of Kingma & Welling (2013) .

We optimize the single sample variational lower bound (ELBO) of the log-likelihood.

We train variational autoencoders exclusively with Boolean latent variables on OMNIGLOT, CIFAR10, mini-ImageNet (Vinyals et al., 2016) and MNIST (Appendix D.1).

We train all models using a regular GPU with stochastic gradient descent with a momentum of 0.9 and a batch size of 128.

We compare against Straight-Through, GumbelSoftmax, and DARN, although on more complex models some estimators diverge.

The results were consistent over multiple runs.

Details regarding the architectures and hyperparameters used are in Appendix E. Upon acceptance we will open source all code, models, data and experiments.

, where t ??? (0, 1) is a continuous target value and z is a sample from the Bernoulli distribution p(z).

The optimum is obtained for p(z = +1) ??? {0, 1}. Figure 1 , shows a case with t = 0.45, where the minimizing solution is p(z = +1) = 0.

We observe that unlike the Straight-Through estimator, FouST converges to the minimizing deterministic solution (lower is better).

Training Stochastic MLPs.

We train MLPs with one and two stochastic layers on OMNIGLOT, following the non-linear architecture of Tucker et al. (2017) .

Each stochastic Boolean layer is preceded by two deterministic layers of 200 tanh units.

All hyperparameters remain fixed throughout the training.

All estimators use one sample per example and a single decoder evaluation.

We present results in Fig. 2 .

FouST outperforms other biased gradient estimators in both datasets and architectures.

FouST is clearly better than the StraightThrough estimator.

Despite the complicated nature of the optimized neural network function f (z) the bias reduction appears fruitful.

With one or two stochastic layers we can also use the unbiased REBAR.

REBAR is not directly comparable to the estimators we study, since it uses multiple decoder evaluations and for models with multiple stochastic layers, multiple passes through later layers.

Nevertheless, as shown in appendix D.1 for MNIST, with two stochastic layers REBAR reaches a worse test ELBO of -94.43 v. -91.94 for FouST.

A possibility of worse test than training ELBOs for REBAR was also suggested in the original work (Tucker et al., 2017) .

Training Stochastic ResNets.

We further validate FouST in a setting where the encoder and decoder are stochastic ResNets, S-ResNets, which are standard ResNets with stochastic layers inserted between ResNet blocks.

Similar to MLPs, FouST outperforms other biased estimators in this setting on CIFAR-10 (left in Figure 3 ).

Note that despite the hyperparameter sweep, we were unable to train S-ResNet's with Gumbel-Softmax.

So we compare against DARN and Straight-Through only.

With an S-ResNet with 12 ResNet blocks and 4 stochatic layers FouST yields a score of 5.08 bits per dimension (bpd).

This is comparable to the 5.14 bpd with the categorical VIMCO-trained model (Mnih & Rezende, 2016) In the plots, we observe sharp spikes or slower curving.

We hypothesize these are due, respectively, to stochasticity and bias, and are corrected to some degree along the way.

Efficiency.

We compare the efficiency of different estimators in Tab.

1.

Like other biased estimators, FouST requires a single sample for estimating the gradients and has similar wallclock times.

On MNIST, the unbiased REBAR is 15x and 40x slower than the biased estimators for two and five stochastic layer MLP's respectively.

From the above experiments we conclude that FouST allows for efficient and effective training of fully connected and convolutional neural networks with Boolean stochastic variables.

Last, we evaluate FouST on more complex neural networks with deeper and wider stochastic layers.

We perform experiments with convolutional architectures on the larger scale and more realistic mini-ImageNet (Vinyals et al., 2016) .

As the scope of this work is not architecture search, we present two architectures inspired from residual networks (He et al., 2016) of varying stochastic depth and width.

The first one is a wide S-ResNet, S-ResNet-40-2-800, and has 40 deterministic (with encoder and decoder combined), 2 stochastic layers, and 800 channels for the last stochastic layer.

The second, S-ResNet-80-11-256, is very deep with 80 deterministic and 11 stochastic layers, and a last stochastic layer with 256 channels.

Architecture details are given in Appendix E.2.

In this setup, training with existing unbiased estimators is intractable.

We present results in Fig. 3 .

We compare against DARN, since we were unable to train the models with Gumbel-Softmax.

Incomplete lines indicate failure.

We observe that FouST is able to achieve better training ELBO's in both cases.

We conclude that FouST allows for scaling up the complexity of stochastic neural networks in terms of stochastic depth and width.

For a Boolean function f the discrete derivative on the i-th latent dimension with a basis function ??i is defined as

The Fourier expansion of the discrete derivative equals

The Fourier expansion of the discrete derivative is derived by equation 2: (i) all bases that do not contain the i-th dimension are constant to ??i and thus set to zero, while (ii) for the rest of the terms ????? S d?? i = ?? S\i from the definition of basis functions ??S(z).

In the following we differentiate partial derivatives on continuous functions noted with ????? from discrete derivatives on Boolean functions noted with D??. The i-th discrete derivative is independent of zi both in the above definitions.

Proof.

We follow O'Donnell (2014, ??8.4) .

In this proof we work with two representations of the Boolean function f .

The first is the Fourier expansion of f under the uniform Bernoulli distribution.

This is also the representation obtained by expressing f as a polynomial in zi.

Since the domain of the function f is the Boolean cube, the polynomial representation is multilinear.

That is f (z) = S???[n]f (S) j???S zj.

To avoid confusion and to differentiate the representation from the Boolean function, we use f (u) (z) to denote this representation in the following.

Note that since this representation is a polynomial it is defined over any input in R n .

In particular,

The second representation we use is the Fourier expansion of the Boolean function f under p(x).

We denote this by f (p) .

The following relation follows from the fact that when working with the Fourier representation, f (z) is multilinear, E p(z) [zi] = ??i and the linearity of expectation.

As the partial derivative of f (u) w.r.t.

??i is equivalent to discrete derivative of

, and keeping in mind that ??i = (zi ??? ??i)/??i, we have that

We then note that the discrete derivative of f w.r.t.

zi, Dz i f (u) (??), from the left hand side of equation 19, is equivalent to the partial derivative of f w.r.t.

??i, ????? i f (u) (??).

We complete the proof by noting that the right hand side in equation 24 is 1 2 times the REINFORCE gradient.

The detailed proof of Lemma 2 is as follows.

Proof.

We first derive the Taylor expansions for the true gradient as well as the Straight-Through gradient estimator.

Then, we prove the lemma by comparing the two Taylor expansions.

By expanding the function f in terms of its ??S basis as in equation 2 and focusing on the i-th dimension, we have thatf

The first term,f (p) (i)

, is the term corresponding to {i} in the Fourier expansion of f under p i???1/2 (z).

That isf

This follows from the fact that when moving from p(z) to p i???1/2 (z), (i) we have that ??i = zi, and (ii) no other term under the p(z) expansion contributes to the zi term under the p i???1/2 (z) expansion.

As a consequence of Lemma 1 the REINFORCE gradient for the i-th dimension is given by

Next, we will derive the Taylor expansions of the true and the Straight-Through gradients.

The Taylor expansion of f (z) for zi around 0 is

where

are the Taylor coefficients.

All c k are a function of zj, j = i.

Let's first focus on the true gradient.

Since we work with binary ??1 values, we have that z k i = 1 for even k and z k i = zi for odd k. This will influence the even and the odd terms of the Taylor expansions.

Specifically, for the Taylor expansion of the true gradient we have from equation 27 and equation 3 that

The expression in equation 32 implies that the true gradient with respect to the pi is the expected sum of the odd Taylor coefficients.

Here we note that the although final expression in equation 32 can also be derived by a finite difference method, it does not make explicit, as in equation 31, the dependence on zi and ??i of the term inside the expectation.

By comparing the expansion of the Straight-Through gradient in equation 35 and the expansion of the true gradient in equation 32,

Taking the expectation in equation 34 under p i???1/2 causes the final term in equation 37 to vanish leaving

C LOW-BIAS GRADIENT ESTIMATES

We describe the case of a bivariate function in detail.

For brevity, we focus on a case study of a two-dimensional z, and a bivariate f (z1, z2) with the bivariate Taylor expansion f (z1, z2) = i,j ci,jz

As in Lemma 2, the partial true gradient of f (z1, z2) w.r.t.

the first distribution parameter p1 equals to

Further, the Taylor expansion of

"Bernoulli splitting uniform" trick.

Assume an auxiliary variable u = (u1, u2), which we choose as follows.

First, we sample z = (z1, z2) from the uniform Bernoulli distribution p 1???1/2 .

Then we take a uniform sample (u1, u2) with ui sampled from either [0, 1] for zi = +1 or from [???1, 0] if zi = ???1.

At this point it is important to note that the moments of the uniform distribution in [a, b]

, which simplifies to b/2, b 2 /3, b 3 /4, . . .

for a = 0, and where we think of b as a binary sample i.e., b ??? {???1, 1}. The expectation of the gradient under such random sampling is

We then compare equation 40 with equation 39.

In equation 40 we observe that the pure terms in z1, namely terms with j = 0, always match those of the true gradient in equation 39.

For j > 0 we obtain mixed terms with coefficients that do not match those of the true gradient in equation 39.

However, the partial gradient obtained with the auxiliary variables in equation 40 has coefficients following a decaying trend due to the 1 j+1

.

For small j, that is, the mixed-degree terms are closer to the original ones in equation 39.

For functions with smaller mixed degree terms this leads to bias reduction, at the cost of an increased variance due to additional sampling.

In practice many functions would have greater dependence on mixed degree terms.

For such functions and to manage the bias-variance trade-off we choose smaller intervals for the uniform samples, that is a ??? b.

The Fourier basis does not depend on the particular input representation and any two-valued set, say {???t, t} can be used as the Boolean representation.

The choice of a representation, however, does affect the bias as we show next.

As a concrete example, we let our input representation be zi ??? {???1/2, 1/2} n , where pi = p(zi = +1/2).

While we can change the input representation like that, in general the Fourier coefficients in equation 3 will be different than for the {???1, +1} representation.

Letting h(zi) = 2zi ??? {???1, 1}, the functions ??i are now given

Next, we write the Taylor series of f in terms of h(zi), f (z) = c0 + c1zi + c2z

Under the p i???1/2 distribution, we still have that E p???1/2 [h(zi)] = 0 and the degree-1 Fourier coefficients are:

Note that compared to equation 7, in equation 43 we still get the odd terms c1, c3 albeit decayed by inverse powers of 2.

Following the same process like for equation 10, we have that To further judge the effect of our proposed modifications to Straight-Through, we performed ablation experiments where we separately applied scaling and noise to the importance-corrected Straight-Through.

These experiments were performed on the single stochastic layer MNIST and OMNIGLOT models.

The results of the ablation experiments are shown in figure 5 .

From the figure it can be seen that scaling alone improves optimization in both cases and noise alone helps in the case of MNIST.

Noise alone results in a worse ELBO in the case of OMNIGLOT, but gives an improvement when combined with scaling.

From these results we conclude that the proposed modifications are effective.

The encoder and decoder networks in this case are MLP's with one or more stochastic layers.

Each stochastic layer is preceded by 2 deterministic layers with a tanh activation function.

We chose learning rates from {1 ?? 10 ???4 , 2 ?? 10 ???4 , 4 ?? 10 ???4 , 6 ?? 10 ???4 }, Gumbel-Softmax temperatures from {0.1, 0.5}, and noise interval length for FouST from {0.1, 0.2}.

For these dataset we use a stochastic variant or ResNets (He et al., 2016) .

Each network is composed of stacks of layers.

Each layer has (i) one regular residual block as in He et al. (2016) , (ii) followed by at most one stochastic layer, except for the CIFAR architecture B in figure 3 where we used two stochastic layers in the last layer.

The stacks are followed by a final stochastic layer in the encoder.

We do downsampling at most once per stack.

We used two layers per stack.

For CIFAR we downsample twice so that the last stochastic layer has feature maps of size 8x8.

We chose learning rate from {9 ?? 10 ???7 , 1 ?? 10 ???6 , 2 ?? 10 ???6 , 4 ?? 10 ???6 }, the FouST scaling parameter from {0.5, 0.8, 0.9}, and the uniform interval was scaled by a factor from {0.01, 0.05, 0.1}

For mini-ImageNet we downsample thrice.

We chose the learning rate from {2 ?? 10 ???7 , 3 ?? 10 ???7 , 4 ?? 10 ???7 , 5 ?? 10 ???7 }.

<|TLDR|>

@highlight

We present a low-bias estimator for Boolean stochastic variable models with many stochastic layers.