Building on the success of deep learning, two modern approaches to learn a probability model of the observed data are Generative Adversarial Networks (GANs) and Variational AutoEncoders (VAEs).

VAEs consider an explicit probability model for the data and compute a generative distribution by maximizing a variational lower-bound on the log-likelihood function.

GANs, however, compute a generative model by minimizing a distance between observed and generated probability distributions without considering an explicit model for the observed data.

The lack of having explicit probability models in GANs prohibits computation of sample likelihoods in their frameworks and limits their use in statistical inference problems.

In this work, we show that an optimal transport GAN with the entropy regularization can be viewed as a generative model that maximizes a lower-bound on average sample likelihoods, an approach that VAEs are based on.

In particular, our proof constructs an explicit probability model for GANs that can be used to compute likelihood statistics within GAN's framework.

Our numerical results on several datasets demonstrate consistent trends with the proposed theory.

Learning generative models is becoming an increasingly important problem in machine learning and statistics with a wide range of applications in self-driving cars BID26 , robotics BID10 , natural language processing BID14 , domain-transfer BID25 , computational biology BID6 , etc.

Two modern approaches to deal with this problem are Generative Adversarial Networks (GANs) BID7 and Variational AutoEncoders (VAEs) BID13 BID15 BID23 BID28 BID17 .VAEs BID13 ) compute a generative model by maximizing a variational lowerbound on average sample likelihoods using an explicit probability distribution for the data.

GANs, however, learn a generative model by minimizing a distance between observed and generated distributions without considering an explicit probability model for the data.

Empirically, GANs have been shown to produce higher-quality generative samples than that of VAEs BID12 .

However, since GANs do not consider an explicit probability model for the data, we are unable to compute sample likelihoods using their generative models.

Computations of sample likelihoods and posterior distributions of latent variables are critical in several statistical inference.

Inability to obtain such statistics within GAN's framework severely limits their applications in such statistical inference problems.

In this paper, we resolve these issues for a general formulation of GANs by providing a theoreticallyjustified approach to compute sample likelihoods using GAN's generative model.

Our results can open new directions to use GANs in massive-data applications such as model selection, sample selection, hypothesis-testing, etc (see more details in Section 5).

Now, we state our main results informally without going into technical conditions while precise statements of our results are presented in Section 2.

Let Y and?? ???= G(X) represent observed (i.e. real) and generative (i.e. fake or synthetic) variables, respectively.

X (i.e. the latent variable) is the randomness used as the input to the generator G(.).

Consider the following explicit probability By training a GAN model, we first compute optimal generator G * and optimal coupling between the observed variable Y and the latent variable X. The likelihood of a test sample y test can then be lower-bounded using a combination of three terms: (1) the expected distance of y test to the distribution learnt by the generative model, (2) the entropy of the coupled latent variable given y test and (3) the likelihood of the coupled latent variable with y test .model of the data given a latent sample X = x: f Y X=x (y) ??? exp(??? (y, G(x))), (1.1)where (., .) is a loss function.

f Y X=x (y) is the model that we are considering for the underlying data distribution.

This is a reasonable model for the data as the function G can be a complex function.

Similar data models have been used in VAEs.

Under this explicit probability model, we show that minimizing the objective of an optimal transport GAN (e.g. Wasserstein BID0 ) with the cost function (., .) and an entropy regularization BID2 BID27 ) maximizes a variational lower-bound on average sample likelihoods.

I.e.average sample likelihoods ??? ??? (entropic GAN objective) + constants.(1.2)If (y,??) = y ????? 2 , the optimal transport (OT) GAN simplifies to WGAN while if (y,??) = y ????? 2 2 , the OT GAN simplifies to the quadratic GAN (or, W2GAN) BID3 ).

The precise statement of this result can be found in Theorem 1.

This result provides a statistical justification for GAN's optimization and puts it in par with VAEs whose goal is to maximize a lower bound on sample likelihoods.

We note that the entropy regularization has been proposed primarily to improve computational aspects of GANs BID5 .

Our results provide an additional statistical justification for this regularization term.

Moreover, using GAN's training, we obtain a coupling between the observed variable Y and the latent variable X. This coupling provides the conditional distribution of the latent variable X given an observed sample Y = y. The explicit model of equation 1.1 acts similar to the decoder in the VAE framework, while the coupling computed using GANs acts as an encoder.

Connections between GANs and VAEs have been investigated in some of the recent works as well BID11 BID16 .

In BID11 , GANs are interpreted as methods performing variational inference on a generative model in the label space.

In their framework, observed data samples are treated as latent variables while the generative variable is the indicator of whether data is real or fake.

The method in BID16 , on the other hand, uses an auxiliary discriminator network to rephrase the maximum-likelihood objective of a VAE as a twoplayer game similar to the objective of a GAN.

Our method is different from both these approaches as we consider an explicit probability model for the data, and show that the entropic GAN objective maximizes a variational lower bound under this probability model, thus allowing sample likelihood computation in GANs similar to VAEs.

Of relevance to our work is BID30 , in which annealed importance sampling (AIS) is used to evaluate the approximate likelihood of decoder-based generative models.

More specifically, a Gaussian observation model with a fixed variance is used as the generative distribution for GANbased models on which the AIS is computed.

Gaussian observation models may not be proper specially in high-dimensional spaces.

Our approach, on the other hand, makes a connection between GANs and VAEs by constructing a theoretically-motivated model for the data distribution in GANs.

We then leverage this approach in computing sample likelihood estimates in GANs.

Another key question that we address here is how to estimate the likelihood of a new sample y test given the generative model trained using GANs.

For instance, if we train a GAN on stop-sign images, upon receiving a new image, one may wish to compute the likelihood of the new sample y test according to the trained generative model.

In standard GAN formulations, the support of the generative distribution lies on the range of the optimal generator function.

Thus, if the observed sample y test does not lie on that range (which is very likely in practice), there is no way to assign a sensible likelihood score to that sample.

Below, we show that using the explicit probability model of equation 1.1, we can lower-bound the likelihood of this sample y test .

This is similar to the variational lower-bound on sample likelihoods used in VAEs.

Our numerical results show that this lower-bound well-reflect the expected trends of the true sample likelihoods.

Let G * and P * Y,X be the optimal generator and the optimal coupling between real and latent variables, respectively.

The optimal coupling P * Y,X can be computed efficiently for entropic GANs as we explain in Section 3.

For other GAN architectures, one may approximate such couplings as we explain in Section 4.

The log likelihood of a new test sample y test can be lower-bounded as DISPLAYFORM0 distance to the generative model DISPLAYFORM1 We present the precise statement of this result in Corollary 2.

This result combines three components in order to approximate the likelihood of a sample given a trained generative model:??? The distance between y test to the generative model.

If this distance is large, the likelihood of observing y test from the generative model is small.??? The entropy of the coupled latent variable.

If the entropy term is large, the coupled latent variable has a large randomness.

This contributes positively to the sample likelihood.??? The likelihood of the coupled latent variable.

If latent samples have large likelihoods, the likelihood of the observed test sample will be large as well.

FIG2 provides a pictorial illustration of these components.

In what follows, we explain the technical ingredients of our main results.

In Section 3, we present computational methods for GANs and entropic GANs, while in Section 4, we provide numerical experiments on benchmark datasets.

Let Y ??? R d represent the real-data random variable with a probability density function f Y (y).GAN's goal is to find a generator function G ??? R r ??? R d such that?? ???= G(X) has a similar distribution to Y .

Let X be an r-dimensional random variable with a fixed probability density function f X (x).

Here, we assume f X (.) is the density of a normal distribution.

In practice, we observe m samples {y 1 , ..., y m } from Y and generate m ??? samples from?? , i.e., {?? 1 , ...,?? m ??? } wher?? DISPLAYFORM0 We represent these empirical distributions by P Y and P?? , respectively.

Note that the number of generative samples m ??? can be arbitrarily large.

GAN computes the optimal generator G * by minimizing a distance between the observed distribution P Y and the generative one P?? .

Common distance measures include optimal transport measures (e.g. Wasserstein GAN , WGAN+Gradient Penalty BID8 , GAN+Spectral Normalization BID18 , WGAN+Truncated Gradient Penalty BID21 , relaxed WGAN BID9 ), and divergence measures (e.g. the original GAN's formulation BID7 , f -GAN BID20 ), etc.

In this paper, we focus on GANs based on optimal transport (OT) distance BID29 defined for a general loss function (., .) as follows DISPLAYFORM1 P Y,?? is the joint distribution whose marginal distributions are equal to P Y and P?? , respectively.

If (y,??) = y ????? 2 , this distance is called the first-order Wasserstein distance and is referred to by W 1 (., .), while if (y,??) = y ????? 2 2 , this measure is referred to by W 2 (., .) where W 2 is the second-order Wasserstein distance BID29 .The optimal transport (OT) GAN is formulated using the following optimization BID29 : DISPLAYFORM2 where G is the set of generator functions.

Examples of the OT GAN are WGAN corresponding to the first-order Wasserstein distance W 1 (., .) 1 and the quadratic GAN (or, the W2GAN) BID3 corresponding to the second-order Wasserstein distance W 2 (., .).Note that optimization 2.2 is a min-min optimization.

The objective of this optimization is not smooth in G and it is often computationally expensive to obtain a solution BID24 .

One approach to improve computational aspects of this optimization is to add a regularization term to make its objective strongly convex BID2 BID27 .

The Shannon entropy function is defined as H(P Y,?? ) ???= ???E log P Y,?? .

The negative Shannon entropy is a common strongly-convex regularization term.

This leads to the following optimal transport GAN formulation with the entropy regularization, or for simplicity, the entropic GAN formulation: DISPLAYFORM3 where ?? is the regularization parameter.

There are two approaches to solve the optimization problem 2.3.

The first approach uses an iterative method to solve the min-min formulation BID4 .

Another approach is to solve an equivelent min-max formulation by writing the dual of the inner minimization BID27 BID24 .

The latter is often referred to as a GAN formulation since the min-max optimization is over a set of generator functions and a set of discriminator functions.

The details of this approach are further explained in Section 3.In the following, we present an explicit probability model for entropic GANs under which their objective can be viewed as maximizing a lower bound on average sample likelihoods.

Theorem 1 Let the loss function be shift invariant, i.e., (y,??) = h(y ?????).

Let DISPLAYFORM4 be an explicit probability model for Y given X = x for a well-defined normalization DISPLAYFORM5 Then, we have DISPLAYFORM6 ave.

sample likelihoods DISPLAYFORM7 In words, the entropic GAN maximizes a lower bound on sample likelihoods according to the explicit probability model of equation 2.4.The proof of this theorem is presented in Section A. This result has a similar flavor to that of VAEs BID15 BID23 BID28 BID17 ) where a generative model is computed by maximizing a lower bound on sample likelihoods.

Having a shift invariant loss function is critical for Theorem 1 as this makes the normalization term C independent from G and x (to see this, one can define y ??? ???= y ??? G(x) in equation 2.6).

The most standard OT GAN loss functions such as the L 2 for WGAN and the quadratic loss for W2GAN BID3 ) satisfy this property.

One can further simplify this result by considering specific loss functions.

For example, we have the following result for the entropic GAN with the quadratic loss function.

is equal to ??? log(m) ??? d log(2????) 2 ??? r 2 ??? log(2??) 2.

DISPLAYFORM8 Let G * and P * Y,X be optimal solutions of an entropic GAN optimization 2.3 (note that the optimal coupling can be computed efficiently using equation 3.7).

Let y test be a newly observed sample.

An important question is what the likelihood of this sample is given the trained generative model.

Using the explicit probability model of equation 2.4 and the result of Theorem 1, we can (approximately) compute sample likelihoods as explained in the following corollary.

Corollary 2 Let G * and P * Y,?? (or, alternatively P * Y,X ) be optimal solutions of the entropic GAN equation 2.3.

Let y test be a new observed sample.

We have DISPLAYFORM9 The inequality becomes tight iff DISPLAYFORM10

In this section, we discuss dual formulations for OT GAN (equation 2.2) and entropic GAN (equation 2.3) optimizations.

These dual formulations are min-max optimizations over two function classes, namely the generator and the discriminator.

Often local search methods such as alternating gradient descent (GD) are used to compute a solution for these min-max optimizations.

First, we discuss the dual formulation of OT GAN optimization 2.2.

Using the duality of the inner minimization, which is a linear program, we can re-write optimization 2.2 as follows BID29 : DISPLAYFORM0 where DISPLAYFORM1 The maximization is over two sets of functions D 1 and D 2 which are coupled using the loss function.

Using the Kantorovich duality BID29 , we can further simplify this optimization as follows: DISPLAYFORM2 where DISPLAYFORM3 and D is restricted to -convex functions BID29 .

The above optimization provides a general formulation for OT GANs.

If the loss function is .

2 , then the optimal transport distance is referred to as the first order Wasserstein distance.

In this case, the min-max optimization 3.2 simplifies to the following optimization : min DISPLAYFORM4 This is often referred to as Wasserstein GAN, or WGAN .

If the loss function is quadratic, then the OT GAN is referred to as the quadratic GAN (or, W2GAN) BID3 .Similarly, the dual formulation of the entropic GAN equation 2.3 can be written as the following optimization BID2 BID27 2 : DISPLAYFORM5 (3.5) Note that the hard constraint of optimization 3.1 is being replaced by a soft constraint in optimization 3.2.

In this case, optimal primal variables P * Y,?? can be computed according to the following lemma BID27 :Lemma 1 Let D * 1 and D * 2 be the optimal discriminator functions for a given generator function G according to optimization 3.4.

Let DISPLAYFORM6 This lemma is important for our results since it provides an efficient way to compute the optimal coupling between real and generative variables (i.e. P * Y,?? ) using the optimal generator (G * ) and discriminators (D * 1 and D * 2 ) of optimization 3.4.

It is worth noting that without the entropy regularization term, computing the optimal coupling using the optimal generator and discriminator functions is not straightforward in general (unless in some special cases such as W2GAN BID29 BID3 ).

This is another additional computational benefit of using entropic GAN.

In this section, we supplement our theoretical results with experimental validations.

One of the main objectives of our work is to provide a framework to compute sample likelihoods in GANs.

Such likelihood statistics can then be used in several statistical inference applications that we discuss in Section 5.

With a trained entropic WGAN, the likelihood of a test sample can be lower-bounded using Corollary 2.

Note that this likelihood estimate requires the discriminators D 1 and D 2 to be solved to optimality.

In our implementation, we use the algorithm presented in BID24 to train the Entropic GAN.

It has been proven BID24 ) that this algorithm leads to a good approximation of stationary solutions of Entropic GAN.To obtain the surrogate likelihood estimates using Corollary 2, we need to compute the density P * X Y =y test (x).

As shown in Lemma 1, WGAN with entropy regularization provides a closedform solution to the conditional density of the latent variable (equation 3.7).

When G * is injective, P * X Y =y test (x) can be obtained from equation 3.7 by change of variables.

In general case, P * X Y =y test (x) is not well defined as multiple x can produce the same y test .

In this case, DISPLAYFORM0 Also, from equation 3.7, we have DISPLAYFORM1 One solution (which may not be unique) that satisfies both equation 4.1 and 4.2 is DISPLAYFORM2 Ideally, we would like to choose P * X Y =y test (x) satisfying equation 4.1 and 4.2 that maximizes the lower bound of Corollary 2.

But finding such a solution can be difficult in general.

Instead we use equation 4.3 to evaluate the surrogate likelihoods of Corollary 2 (note that our results still hold in this case).

In order to compute our proposed surrogate likelihood, we need to draw samples from the distribution P * X Y =y test (x).

One approach is to use a Markov chain Monte Carlo (MCMC) method to sample from this distribution.

In our experiments, however, we found that MCMC demonstrates poor performance owing to the high dimensional nature of X. A similar issue with MCMC has been reported for VAEs in BID13 .

Thus, we use a different estimator to compute the likelihood surrogate which provides a better exploration of the latent space.

We present our sampling procedure in Alg.

1 of Appendix.

In the experiments of this section, we study how sample likelihoods vary during GAN's training.

An entropic WGAN is first trained on MNIST dataset.

Then, we randomly choose 1, 000 samples from MNIST test-set to compute the surrogate likelihoods using Algorithm 1 at different training iterations.

Surrogate likelihood computation requires solving D 1 and D 2 to optimality for a given G (refer to Lemma.

2), which might not be satisfied at the intermediate iterations of the training process.

Therefore, before computing the surrogate likelihoods, discriminators D 1 and D 2 are updated for 100 steps for a fixed G. We expect sample likelihoods to increase over training iterations as the quality of the generative model improves.

In this section, we perform experiments across different datasets.

An entropic WGAN is first trained on a subset of samples from the MNIST dataset containing digit 1 (which we call the MNIST-1 dataset).

With this trained model, likelihood estimates are computed for (1) samples from the entire MNIST dataset, and (2) samples from the Street View House Numbers (SVHN) dataset BID19 FIG2 .

In each experiment, the likelihood estimates are computed for 1000 samples.

We note that highest likelihood estimates are obtained for samples from MNIST-1 dataset, the same dataset on which the GAN was trained.

The likelihood distribution for the MNIST dataset is bimodal with one mode peaking inline with the MNIST-1 mode.

Samples from this mode correspond to digit 1 in the MNIST dataset.

The other mode, which is the dominant one, contains the rest of the digits and has relatively low likelihood estimates.

The SVHN dataset, on the other hand, has much smaller likelihoods as its distribution is significantly different than that of MNIST.

Furthermore, we observe that the likelihood distribution of SVHN samples has a large spread (variance).

This is because samples of the SVHN dataset is more diverse with varying backgrounds and styles than samples from MNIST.

We note that SVHN samples with high likelihood estimates correspond to images that are similar to MNIST digits, while samples with low scores are different than MNIST samples.

Details of this experiment are presented in Appendix E.

Most standard GAN architectures do not have the entropy regularization.

Likelihood lower bounds of Theorem 1 and Corollary 2 hold even for those GANs as long as we obtain the optimal coupling P * Y,?? in addition to the optimal generator G * from GAN's training.

Computation of optimal cou- pling P * Y,?? from the dual formulation of OT GAN can be done when the loss function is quadratic BID3 .

In this case, the gradient of the optimal discriminator provides the optimal coupling between Y and?? BID29 ) (see Lemma.

2 in Appendix C).For a general GAN architecture, however, the exact computation of optimal coupling P * Y,?? may be difficult.

One sensible approximation is to couple Y = y test with a single latent samplex (we are assuming the conditional distribution P * X Y =y test is an impulse function).

To computex corresponding to a y test , we sample k latent samples {x DISPLAYFORM0 ) is closest to y test .

This heuristic takes into account both the likelihood of the latent variable as well as the distance between y test and the model (similarly to equation 3.7).

We can then use Corollary 2 to approximate sample likelihoods for various GAN architectures.

We use this approach to compute likelihood estimates for CIFAR-10 (Krizhevsky, 2009) and LSUNBedrooms (Yu et al., 2015) datasets.

For CIFAR-10, we train DCGAN while for LSUN, we train WGAN (details of these experiments can be found in Appendix E).

FIG4 demonstrates sample likelihood estimates of different datasets using a GAN trained on CIFAR-10.

Likelihoods assigned to samples from MNIST and Office datasets are lower than that of the CIFAR dataset.

Samples from the Office dataset, however, are assigned to higher likelihood values than MNIST samples.

We note that the Office dataset is indeed more similar to the CIFAR dataset than MNIST.

A similar experiment has been repeated for LSUN-Bedrooms (Yu et al., 2015) dataset.

We observe similar performance trends in this experiment FIG4 .

In this paper, we have provided a statistical framework for a family of GANs.

Our main result shows that the entropic GAN optimization can be viewed as maximization of a variational lower-bound on average log-likelihoods, an approach that VAEs are based upon.

This result makes a connection between two most-popular generative models, namely GANs and VAEs.

More importantly, our result constructs an explicit probability model for GANs that can be used to compute a lower-bound on sample likelihoods.

Our experimental results on various datasets demonstrate that this likelihood surrogate can be a good approximation of the true likelihood function.

Although in this paper we mainly focus on understanding the behavior of the sample likelihood surrogate in different datasets, the proposed statistical framework of GANs can be used in various statistical inference applications.

For example, our proposed likelihood surrogate can be used as a quantitative measure to evaluate the performance of different GAN architectures, it can be used to quantify the domain shifts, it can be used to select a proper generator class by balancing the bias term vs. variance, it can be used to detect outlier samples, it can be used in statistical tests such as hypothesis testing, etc.

We leave exploring these directions for future work.

APPENDIX A PROOF OF THEOREM 1Using the Baye's rule, one can compute the log-likelihood of an observed sample y as follows: DISPLAYFORM0 where the second step follows from equation 2.4.Consider a joint density function P X,Y such that its marginal distributions match P X and P Y .

Note that the equation A.1 is true for every x. Thus, we can take the expectation of both sides with respect to a distribution P X Y =y .

This leads to the following equation: DISPLAYFORM1 where H(.) is the Shannon-entropy function.

Next we take the expectation of both sides with respect to P Y : DISPLAYFORM2 Here, we replaced the expectation over P X with the expectation over f X since one can generate an arbitrarily large number of samples from the generator.

Since the KL divergence is always nonnegative, we have DISPLAYFORM3 Moreover, using the data processing inequality, we have BID1 .

Thus, DISPLAYFORM4 DISPLAYFORM5 GAN objective with entropy regularizer DISPLAYFORM6 This inequality is true for every P X,Y satisfying the marginal conditions.

Thus, similar to VAEs, we can pick P X,Y to maximize the lower bound on average sample log-likelihoods.

This leads to the entropic GAN optimization 2.3.

In Theorem 1, we showed that the Entropic GAN objective maximizes a lower-bound on the average sample log-likelihoods.

This result is in the same flavor of variational lower bounds used in VAEs, thus providing a connection between these two areas.

One drawback of VAEs in general is about the lack of tightness analysis of the employed variational lower bounds.

In this section, we aim to understand the tightness of the entropic GAN lower bound for some generative models.

Corollary 2 shows that the entropic GAN lower bound is tight when KL P X Y =y f X Y =y approaches 0.

Quantifying this term can be useful for assessing the quality of the proposed likelihood surrogate function.

We refer to this term as the approximation gap.

Computing the approximation gap can be difficult in general as it requires evaluating f X Y =y .

Here we perform an experiment for linear generative models and a quadratic loss function (same setting of Corrolary 1).

Let the real data Y be generated from the following underlying model Figure 4 : A visualization of density functions of P X Y =y test and f X Y =y test for a random twodimensional y test .

Both distributions are very similar to one another making the approximation gap (i.e. KL P X Y =y test f X Y =y test ) very small.

Our other experimental results presented in TAB0 are consistent with this result.

DISPLAYFORM0 where X ??? N (0, I)Using the Bayes rule, we have DISPLAYFORM1 Since we have a closed-form for f X Y , KL P X Y =y f X Y =y can be computed efficiently.

The matrix G to generate Y is chosen randomly.

Then, an entropic GAN with a linear generator and non-linear discriminators are trained on this dataset.

P X Y =y is then computed using equation 4.3.

TAB0 reports the average surrogate log-likelihood values and the average approximation gaps computed over 100 samples drawn from the underlying data distribution.

We observe that the approximation gap is orders of magnitudes smaller than the log-likelihood values.

Additionally, in Figure 4 , we demonstrate the density functions of P X Y =y and f X Y =y for a random y and a two-dimensional case (r = 2) .

In this figure, one can observe that both distributions are very similar to one another making the approximation gap very small.

Architecture and hyper-parameter details: For the generator network, we used 3 linear layers without any non-linearities (2 ??? 128 ??? 128 ??? 2).

Thus, it is an over-parameterized linear system.

The discriminator architecture (both D 1 and D 2 ) is a 2-layer MLP with ReLU non-linearities (2 ??? 128 ??? 128 ??? 1).

?? = 0.1 was used in all the experiments.

Both generator and discriminator were trained using the Adam optimizer with a learning rate 10 ???6 and momentum 0.5.

The discriminators were trained for 10 steps per generator iteration.

Batch size of 512 was used.

DISPLAYFORM2 Optimal coupling P * Y,?? for the W2GAN (quadratic GAN BID3 ) can be computed using the gradient of the optimal discriminator BID29

In practice, it has been observed that a slightly modified version of the entropic GAN demonstrates improved computational properties BID4 BID24 .

We explain this modification in this section.

Let DISPLAYFORM0 where D KL (. .) is the KullbackLeibler divergence.

Note that the objective of this optimization differs from that of the entropic GAN optimization 2.3 by a constant term ??H(P Y ) + ??H(P?? ).

A sinkhorn distance function is then defined as BID4 : DISPLAYFORM1 W is called the Sinkhorn loss function.

Reference BID4 has shown that as ?? ??? 0, DISPLAYFORM2 For a general ??, we have the following upper and lower bounds:Lemma 3 For a given ?? > 0, we hav?? DISPLAYFORM3 (this can be seen by using an identity coupling as a feasible solution for optimization D.1) and similarly DISPLAYFORM4 Since H(P Y ) + H(P?? ) is constant in our setup, optimizing the GAN with the Sinkhorn loss is equivalent to optimizing the entropic GAN.

So, our likelihood estimation framework can be used with models trained using Sinkhorn loss as well.

This is particularly important from a practical standpoint as training models with Sinkhorn loss tends to be more stable in practice.

In this section, we discuss how WGANs with entropic regularization is trained.

As discussed in Section 3, the dual of the entropic GAN formulation can be written as DISPLAYFORM0 We can optimize this min-max problem using alternating optimization.

A better approach would be to take into account the smoothness introduced in the problem due to the entropic regularizer, and solve the generator problem to stationarity using first-order methods.

Please refer to BID24 for more details.

In all our experiments, we use Algorithm 1 of BID24 to train our GAN model.

MNIST dataset constains 28??28 grayscale images.

As a pre-processing step, all images were resized in the range [0, 1].

The Discriminator and the Generator architectures used in our experiments are given in Tables.

2,3.

Note that the dual formulation of GANs employ two discriminators -D 1 and D 2 , and we use the same architecture for both.

The hyperparameter details are given in TAB4 .

Some sample generations are shown in Fig. 5

We trained a DCGAN model on CIFAR dataset using the discriminator and generator architecture used in BID22 .

The hyperparamer details are mentioned in Table.

5.

Some sample generations are provided in Figure 7 We trained a WGAN model on LSUN-Bedrooms dataset with DCGAN architectures for generator and discriminator networks .

The hyperparameter details are given in Table.

6, and some sample generations are provided in Fig

@highlight

A statistical approach to compute sample likelihoods in Generative Adversarial Networks

@highlight

Show that WGAN with entropic regularization maximizes a lower bound on the likelihood of the observed data distribution.

@highlight

Authors claim it is possible to leverage the upper bound from an entropy regularized optimal transport to come up with a measure of 'sample likelihood'.