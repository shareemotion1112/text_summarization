This paper studies the undesired phenomena of over-sensitivity of representations learned by deep networks to semantically-irrelevant changes in data.

We identify a cause for this shortcoming in the classical Variational Auto-encoder (VAE) objective, the evidence lower bound (ELBO).

We show that the ELBO fails to control the behaviour of the encoder out of the support of the empirical data distribution and this behaviour of the VAE can lead to extreme errors in the learned representation.

This is a key hurdle in the effective use of representations for data-efficient learning and transfer.

To address this problem, we propose to augment the data with specifications that enforce insensitivity of the representation with respect to families of transformations.

To incorporate these specifications, we propose a regularization method that is based on a selection mechanism that creates a fictive data point by explicitly perturbing an observed true data point.

For certain choices of parameters, our formulation naturally leads to the minimization of the entropy regularized Wasserstein distance between representations.

We illustrate our approach on standard datasets and experimentally show that significant improvements in the downstream adversarial accuracy can be achieved by learning robust representations completely in an unsupervised manner, without a reference to a particular downstream task and without a costly supervised adversarial training procedure.

Representation learning is a fundamental problem in Machine learning and holds the promise to enable data-efficient learning and transfer to new tasks.

Researchers working in domains like Computer Vision (Krizhevsky et al., 2012) and Natural Language Processing (Devlin et al., 2018) have already demonstrated the effectiveness of representations and features computed by deep architectures for the solution of other tasks.

A case in point is the example of the FC7 features from the AlexNet image classification architecture that have been used for many other vision problems (Krizhevsky et al., 2012) .

The effectiveness of learned representations has given new impetus to research in representation learning, leading to a lot of work being done on the development of techniques for inducing representations from data having desirable properties like disentanglement and compactness (Burgess et al., 2018; Achille & Soatto, 2017; Bengio, 2013; Locatello et al., 2019) .

Many popular techniques for generating representation are based on the Variational AutoEncoders (VAE) model (Kingma & Welling, 2013; Rezende et al., 2014) .

The use of deep networks as universal function approximators has facilitated very rapid advancements which samples generated from these models often being indistinguishable from natural data.

While the quality of generated examples can provide significant convincing evidence that a generative model is flexible enough to capture the variability in the data distribution, it is far from a formal guarantee that the representation is fit for other purposes.

In fact, if the actual goal is learning good latent representations, evaluating generative models only based on reconstruction fidelity and subjective quality of typical samples is neither sufficient nor entirely necessary, and can be even misleading.

In this paper, we uncover the problematic failure mode where representations learned by VAEs exhibit over-sensitivity to semantically-irrelevant changes in data.

One example of such problematic behaviour can be seen in Figure 1 .

We identify a cause for this shortcoming in the classical Vari-ational Auto-encoder (VAE) objective, the evidence lower bound (ELBO) , that fails to control the behaviour of the encoder out of the support of the empirical data distribution.

We show this behaviour of the VAE can lead to extreme errors in the recovered representation by the encoder and is a key hurdle in the effective use of representations for data-efficient learning and transfer.

To address this problem, we propose to augment the data with properties that enforce insensitivity of the representation with respect to families of transformations.

To incorporate these specifications, we propose a regularization method that is based on a selection mechanism that creates a fictive data point by explicitly perturbing an observed true data point.

For certain choices of parameters, our formulation naturally leads to the minimization of the entropy regularized Wasserstein distance between representations.

We illustrate our approach on standard datasets and experimentally show that significant improvements in the downstream adversarial accuracy can be achieved by learning robust representations completely in an unsupervised manner, without a reference to a particular downstream task and without a costly supervised adversarial training procedure.

Figure 1: An illustration of the intrinsic fragility of VAE representations.

Outputs from a Variational Autoencoder with encoder f and decoder g parametrized by η and θ, respectively, trained on CelebA. Conditioned on the encoder input X a = x a the decoder output X = g(f (x a )) = (g • f )(x a ) is shown on the top row.

When the original example is perturbed with a carefully selected vector d such that X b = X a + d with d ≤ , the output X turns out to be perceptually very different.

Such examples suggest that either the representations Z a and Z b are very different (the encoder is not smooth), or the decoder is very sensitive to small changes in the representation (the decoder is not smooth), or both.

We identify the source of the problem primarily as the encoder and propose a practical solution.

It is clear that if learned representations are overly sensitive to irrelevant changes in the input (for example, small changes in the pixels of an image or video, or inaudible frequencies added to an audio signal), models that rely on these representations are naturally susceptible to make incorrect predictions when inputs are changed.

We argue that such specifications about the robustness properties of learned representations can be one of the tractable guiding features in the search for good representations.

Based on these observations, we make the following contributions:

1.

We introduce a method for learning robust latent representations by explicitly targeting a structured model that admits the original VAE model as a marginal.

We also show that in the case the target is chosen a pairwise conditional random field with attractive potentials, this choice leads naturally to the Wasserstein divergence between posterior distributions over the latent space.

This insight provides us a flexible class of robustness metrics for controlling representations learned by VAEs.

2.

We develop a modification to training algorithms for VAEs to improve robustness of learned representations, using an external selection mechanism for obtaining transformed examples and by enforcing the corresponding representations to be close.

As a particular selection mechanism, we adopt attacks in adversarial supervised learning (Madry et al., 2017) to attacks to the latent representation.

Using this novel unsupervised training procedure we learn encoders with adjustable robustness properties and show that these are effective at learning representations that perform well across a variety of downstream tasks.

3.

We show that alternative models proposed in the literature, in particular β-VAE model used for explicitly controlling the learned representations, or Wasserstein Generative Adversarial Networks (GANs) can also be interpreted in our framework as variational lower bound maximization.

4.

We show empirically using simulation studies on MNIST, color MNIST and CelebA datasets, that models trained using our method learn representations that provide a higher degree of adversarial robustness even without supervised adversarial training.

Modern generative models are samplers p(X|θ) for generating realizations from an ideal target distribution π(X), also known as the data distribution.

In practice π(X) is unknown in the sense that it is hard to formally specify.

Instead, we have a representative data set X , samples that are assumed to be conditionally independently drawn from the data distribution π(X) of interest.

We will refer to the empirical distribution asπ(X) =

, thereby also learning a generator.

The VAE corresponds to the latent variable model p(X|Z, θ)p(Z) with latent variable Z and observation X. The forward model p(X|Z = z, θ) (the decoder) is represented using a neural network g with parameters θ, usually the mean of a Gaussian N (X; g(z; θ), vI x ) where v is a scalar observation noise variance and I x is an identity matrix.

The prior is usually a standard Gaussian p(Z = z) = N (z; 0, I z ).

The exact posterior over latent variables p(Z|X = x, θ) is approximated by a probability model q(Z|X = x, η) with parameters η.

A popular choice here is a multivariate Gaussian N (Z; µ(x; η), Σ(x; η)), where the mapping f such that (µ, Σ) = f (x, η) is chosen to be a neural network (with parameters η to be learned from data).

We will refer to the pair f, g as an encoder-decoder pair.

Under the above assumptions, VAE's are trained by maximizing the following form of the ELBO using stochastic gradient descent (SGD),

The gradient of the Kullback-Leibler (KL) divergence term above (see A.1) is available in closed form.

An unbiased estimate of the gradient of the first term can be obtained via sampling z from q using the reparametrization trick Kingma & Welling (2013) , aided by automatic differentiation.

Under the i.i.d.

assumption, where each data point x (n) , for n = 1 . . .

N is independently drawn from the model an equivalent batch ELBO objective can be defined as

where the empirical distribution of observed data is denoted asπ (See E.1 for a derivation).

This form makes it more clear that the variational lower bound is only calculating the distance between the encoder and decoder under the support of the empirical distribution.

To see how this locality leads to a fragile representation, we construct a VAE with discrete latents and observations.

We let X ∈ {1, . . .

, N x } and Z ∈ {1, . . .

, N z } and define the following system of conditional distributions as the decoder and encoder models as:

where ω(u) = cos(2πu).

These distributions can be visualized by heatmaps of probability tables where i and j are row and column indicies, respectively Figure 2 .

This particular von-Mises like parametrization is chosen for avoiding boundary effects due to a finite latent and observable spaces.

The prior p(Z) is taken as uniform, and is not shown.

Note that this parametrization emulates a high capacity network that can model any functional relationship between latent states and observations, while being qualitatively similar to a standard VAE model with conditionally Gaussian decoder and encoder functions.

In reality, the true target density is not available but we would have a representative sample.

To simulate this scenario, we sample a 'dataset' from a discrete target distribution π(X): this is merely a draw from a multinomial distribution, yielding a multinomial vector s with entries s i that gives the count how many times we observe x = i.

The results of such an experiment are depicted in Figure 3 (a) (see caption for details).

This picture reveals several important properties of a VAE approximation.

Figure 3: (a) Result by optimizing the ELBO for a VAE that illustrates the fragility of the encoder.

Subfigure with the title 'Data' (π(X)) is a random sample from the true target 'Target' (π(X)) on the right.

The resulting encoder q(Z|X) and decoder p(X|Z) are shown as 'Q' and 'P', respectively.

The vertical and horizontal axes correspond to latents Z and observations X respectively.

Integrating over the decoder distribution using a uniform prior p(Z) over the latents, we obtain the model

The results obtained by a smooth encoder.

Both the decoder and the representation (encoder) are more smooth while essentially having a similar fitting quality.

1.

After training, we observe that when j and j are close, the corresponding conditionals p(X|Z = j) and p(X|Z = j ) are close (hence corresponding decoder mean parameters m j and m j are close, hence (see middle panel of Fig.3 (a) with the title P showing the decoder).

This smoothness is perhaps surprising at a first sight: in this example, we could arbitrarily permute columns of the decoder and still get the same marginal distribution.

Technically speaking, given a uniform prior p(Z), the marginal likelihood p(X|θ) is entirely invariant with respect to permutations of the latent state.

In fact if the encoder distribution wouldn't be constrained we could also permute the columns of the encoder to keep the ELBO invariant.

In the appendix E.2, we provide an argument why the choice of an unimodal encoder model and optimization of the variational objective leads naturally to smooth decoder functions.

2.

The encoders found by the VAE on the other hand are not smooth at all, despite the fact that the model shows a relatively good fit.

This behaviour alerts us about judging generative models only by the quality of the samples, by traversing the latent space and generating conditional samples from the decoder.

The quality of the decoder seems to be not a proxy for the robustness of the representation.

The fragility of representations is inherent from the ELBO objective.

For the entire dataset, a batch ELBO that involves the counts s i can be written as

The last expression is proportional to the negative KL divergence between two tabular distributions:

.

As such, whenever s i is zero, the contribution of row i of the encoder distribution vanishes and the corresponding parameters µ i and σ i are not effecting the lower bound.

In a sense, the objective does not enforce any structure on the encoder outside of the position of the data points in the training set.

This figure shows that the outof-sample behaviour (i.e., for i whereπ(X) = 0) the encoder is entirely initialization dependent, hence no learning takes place.

We would also expect that the resulting representations would be fragile, in the sense that a small perturbation of an observation can result in a large change in the encoder output.

In this section, we will adopt a strategy for training the encoder that is guaranteed not to change the original objective of the decoder when maximizing the lower bound while obtaining a smoother representation.

The key idea of our approach is that we assume an external selection mechanism that is able to provide new fictive data point x in the vicinity of each observation in our data set x. Here, "in the vicinity" means that we desire that the corresponding latent state of the original datapoint z = f (x; η) and the latent state of the fictitious point z = f (x ; η) should be close to each other in some sense.

Assuming the existence of such an external selection mechanism, we first define the following augmented distribution

where

This is a pairwise conditional Markov random field (CRF) model (Lafferty et al., 2001; Sutton & McCallum, 2012) , where we take c(Z a , Z b ) as a pairwise cost function.

A natural choice here would be, for example, the Euclidean square distance Z a − Z b 2 .

Moreover, we choose a nonnegative coupling parameter γ ≥ 0.

For any pairwise Q(Z a , Z b ) distribution, the ELBO has the following form

It may appear that the SE has to maintain a pairwise approximation distribution Q(Z a , Z b ).

However, this turns out to be not necessary.

Given the encoder, the marginals of

, so the only remaining terms that depend on the pair distribution are the final two terms in (5).

We note that this two terms are just the objective function of the entropy regularized optimal transport problem (Cuturi, 2013; Amari et al., 2017) .

If we view Q(Z a , Z b ) as a transport plan, the first term is maximal when the expected cost is minimal while the second term is maximal when the variational distribution is factorized as

In retrospection, this link is perhaps not that surprising as the Wasserstein distance, the solution of the optimal transport problem, is itself defined as the solution to a variational problem (Solomon, 2018) : Consider a set Γ of joint densities Q(Z a , Z b ) with the property that Q has fixed marginals

The Wasserstein divergence 1 , denoted by WD is defined as the solution of the optimization problem with respect to pairwise distribution Q

where c(Z a , Z b ) is a function that specifies the 'cost' of transferring a unit of probability mass from

It is important to note that with our choice of the particular form of the variational distribution Q(Z a , Z b ) we can ensure that we are still optimizing a lower bound of the original problem.

We can achieve this by simply integrating out the X , effectively ignoring the likelihood term for the fictive observations.

Our choice does not modify the original objective of the decoder due to the fact that the marginals are fixed given η.

To see this, take the exponent of (5) and integrate over the unobserved X log p(X = x|θ) = log dX p(X = x, X |θ)

we name this lower bound B SE as the Smooth Encoder ELBO (SE-ELBO).

The gradient of B SE with respect to the decoder parameters θ is identical to the gradient of the original VAE objective B. This is intuitive as x is an artificially generated sample, we should use only terms that depend on x and not on x .

Another advantage of this choice is that it is possible to optimize the decoder and encoder concurrently as in the standard VAE.

Only an additional term enters for the regularization of the encoder where the marginals obtained via amortized inference q(Z a |x a , η) and q(Z b |x b , η) are forced to be close in a regularized Wasserstein distance sense, with the coupling strength γ.

Effectively, we are doing data augmentation for smoothing the representations obtained by the encoder without changing the actual data distribution.

In the appendix E.3, we also provide an argument about the smoothness of the corresponding encoder mapping, justifying the name.

The resulting algorithm is actually a simple modification to the standard VAE and is summarized below:

Adversarial attacks are one of the most popular approaches for probing trained models in supervised tasks, where the goal of an adversarial attack is finding small perturbations to an input example that would maximally change the output, e.g., flip a classification decision, change significantly a prediction (Szegedy et al., 2013) .

The perturbed input is named as an adversarial example and these extra examples are used, along with the original data points, for training adversarially robust models (Madry et al., 2017; Kurakin et al., 2016) .

As extra samples are also included, such a training procedure is referred as data augmentation.

However, in unsupervised learning and density 1 We use the term divergence to distinguish the optimal transport cost from the corresponding metric.

This distinction is reminiscent to the distinction between Euclidian divergence · 2 and the Euclidian distance · estimation, data augmentation is not a valid approach as the underlying empirical distribution would be altered by the introducing new points.

However, as we let the encoder to target a different distribution than the actual decoder, we can actually use the extra, self generated samples to improve desirable properties of a given model.

Hence this approach could also be interpreted as a 'self-supervised' learning approach where we bias our search for a 'good encoder' and the data selection mechanism acts like a critique, carefully providing examples that should lead to similar representations.

In this paper we will restrict ourselves to Projected Gradient Descent (PGD) attacks popular in adversarial training Carlini & Wagner (2016) as a selection mechanism, where the goal of the attacker is finding a point that would introduce the maximum difference in the Wasserstein distance of the latent representation.

In other words, we implement our selection mechanism where the extra data point is found by approximately solving the following constrained optimization problem

This attack is assigned a certain iteration budget L for a given radius , that we refer as selection iteration budget and the selection radius, respectively.

We note a similar attack mechanism is proposed for generative models as described in (Kos et al., 2017) , where one of the proposed attacks is directly optimizing against differences in source and target latent representations.

Note that our method is not restricted to a particular selection mechanism; indeed two inputs that should give a similar latent representation could be used as candidates.

Goal and Protocol In our experiments, we have tested and compared the adversarial accuracy of representations learned using a VAE and our smooth encoder approach.

We adopt a two step experimental protocol, where we first train encoder-decoder pairs agnostic to any downstream task.

Then we fix the representation, that is we freeze the encoder parameters and only use the mean of the encoder as the representation, then train a simple linear classifier based on the fixed representation using standard techniques.

In this supervised stage, no adversarial training technique is employed.

Ideally, we hope that such an approach will provide a degree of adversarial robustness, without the need for a costly, task specific adversarial training procedure.

To evaluate the robustness of the resulting classifier, for each data point in the test set, we search for an adversarial example using an untargeted attack that tries to change the classification decision.

The adversarial accuracy is reported in terms of percentage of examples where the attack is not able to find an adversarial example.

The VAE and SE decoder and encoder are implemented using standard MLP and ConvNet architectures.

The selection procedure for SE training is implemented as a projected gradient descent optimization (a PGD attack) with selection iteration budget of L iterations to maximize the Wasserstein distance between q(Z|X = x) and q(Z|X = x + δ) with respect to the perturbation δ where δ ∞ < .

Further details about the experiment can be found in the appendix C.1.

We run simulations on ColorMNIST, MNIST and CelebA datasets.

The ColorMNIST is constructed from the MNIST dataset by coloring each digit artificially with all of the colors corresponding to the seven of the eight corners of the RGB cube (excluding black).

We present the results with the strongest attack we have experimented: a PGD attack with 100 iterations and 10 restarts.

We observe that for weaker attacks (such as 50 iterations with no restarts), the adversarial accuracy is typically much higher.

For the ColorMNIST dataset, the results are shown in Figure 4 where we test the adversarial accuracy of representations learned by our method and compare it to a VAE.

We observe that the adversarial accuracy of a VAE representation quickly drops towards zero while SE can maintain adversarial accuracy in both tasks.

In particular, we observe that for the arguably simpler color classification task, we are able to obtain close to perfect adversarial test accuracy using representations learned by the VAE and SE.

However, when the classifiers are attacked using PGD, the adversarial accuracy quickly drops with increasing radius size, while the accuracy degrades more gracefully in the SE case.

In Figure 5 , we show the robustness behaviour of the method for different architectures.

A ConvNet seems to perform relatively better than an MLP but these results show that the VAE representation is not robust, irrespective of the architecture.

We have also carried out controlled experiments with random selection instead of the more costly untargetted adversarial attacks (See appendix C.1 Figure 7(a) for further results).

We observe some limited improvements with SE using random selection in adversarial accuracy compared to VAE but training a SE with adversarial selection seems to be much more effective.

We note that the selection iteration budget was lower (L = 20 with no restarts) than the attack iteration budget (100 with 10 restarts) during evaluation.

It was not practical to train the encoder with more powerful selection attacks, thus it remains to be seen if the tendency of increased accuracy with increased iteration budgets would continue.

We also observe that essentially the same level of adversarial accuracy can be obtained with a small fraction of the available labels (See appendix C.1 Figure 8 for further results).

We have also repeated our experiments on the CelebA dataset, a large collection of high resolution face images labeled with 40 attribute labels per example.

We have used 17 of the attribute labels as the targets of 17 different downstream classification tasks.

The results are shown in Table.

2.

The results clearly illustrate that we can achieve much more robust representations than a VAE.

It is also informative to investigate specific adversarial examples to understand the failure modes.

In Figure 6 we show two illustrative examples from the CelebA. Here we observe that attacks to the SE representations are much more structured and semantically interpretable.

In our exploratory investigations, we qualitatively observe that the reconstructions corresponding to the adversarial In the VAE case, the attacker is able to find a perturbation to map the representation to that of a bearded person.

However, the perturbation here does not seem to have a particular visible structure.

(b) The SE representation is attacked by a perturbation that can clearly be identified as drawing a beard on the image.

In this case, the attack is able to fool the classifier and the generated image from the representation is that of a person with beard and mustache.

In the second example (c), the VAE representation seems to be attacked by exploiting the non-smooth nature of the encoder; the attacker is able to identify an adversarial example that maps the latent representation to the one in the vicinity of a clearly different person with the desired features, as can be seen from the corresponding reconstruction.

In contrast, in the SE case (d), the attack adds a much more structured perturbation, and in this example it was actually not successful in switching the decision.

Additionally, from the reconstruction it is evident that a latent feature is attacked that seems to control the receding hairline.

examples are almost always legitimate face images with clearly recognizable features.

This also seems to support our earlier observation that VAE decoders are typically smooth while the encoders are inferring non-robust features.

Our approach seems to be a step towards obtaining more robust representations.

Table 1 : Comparison of nominal (Nom) and adversarial (Adv) accuracy (in percentage) on 17 downstream tasks using a VAE and a SE trained with a selection radius of = 0.1.

The experiment is carried out using the experimental protocol described in the text (Section 4).

The adversarial evaluation on CelebA with Attack radius of 0.1 and attack iteration budget of 100 with 10 restarts.

The literature on deep generative models and representation learning is quite extensive and is rapidly expanding.

There is a plethora of models, but some approaches have been quite popular in recent years: Generative Adversarial Networks (GANs) and VAEs.

While the connection of our approach to VAE's is evident, there is also a connection to GANs.

In the appendix, we provide the details where we show that a GAN decoder can be viewed as an instance of a particular smooth encoder.

Our method is closely related to the β-VAE (Higgins et al., 2017) , used for controlling representations replaces the original variational objective (1) with another one for explicitly trading the data fidelity with that of prior fidelity.

In the appendix, we show that the method can be viewed as an instance of the smooth encoders.

Wasserstein distance minimization has been applied in generative models as an alternative objective for fitting the decoder.

Following the general framework sketched in , the terms of the variational decomposition of the marginal likelihood can be modified in order to change the observation model or the regulariser.

For example, Wasserstein AutoEncoders (WAE) , Zhang et al. (2019) or sliced Wasserstein Autoencoders Kolouri et al. (2018) propose to replace data fidelity and/or the KL terms with a Wasserstein distance.

Our approach is different from these approaches as we do not propose to replace the likelihood as a fundamental principle for data fitting.

In contrast, the Wasserstein distance formulation naturally emerges from the particular model choice and the corresponding variational approximation.

Our approach involves an adversarial selection step.

The word 'Adversarial' is an overloaded term in generative modelling so it is important to mention differences between our approach.

Adversarial Variational Bayes is a well known technique in the literature that aims to combine the empirical success of GANs with the probabilistic formulation of VAEs, where the limiting functional form of the variational distribution can be replaced by blackbox inference (Mescheder et al., 2017) .

This approach also does not modify the original VAE objective, however, the motivation here is different as the aim is developing a more richer family.

In our view, for learning useful representations, when the decoder is unknown, the advantage of having a more powerful approximating family is not clear yet; this can even make the task of learning a good representation harder.

Adversarial Autoencoders (Makhzani et al., 2015) , Adversarially Learned Inference (ALI) (Dumoulin et al., 2016) and BiGANs (Bidirectional GANs) (Donahue et al., 2016) are also techniques that combine ideas from GANs and VAEs for learning generative models.

The key idea is matching an encoder process q(z|x)p(x) and to the decoder process p(z)p(x|z) using an alternative objective, rather than by minimizing the KL divergence as done by the variational inference (see (??)).

In this formulation, p(x) is approximated by the empirical data distribution, and p(z) is the prior model of a VAE.

The encoder q(z|x) and decoder p(x|z) are modelled using deep networks.

This approach is similar to Wasserstein autoencoders that propose to replace the likelihood principle.

The idea of improving VAEs by capturing the correlation structure between data points using MRFs and graphical models has been also been recently proposed (Tang et al., 2019) under the name Correlated Variational Auto-Encoders (CVAEs).

Our approach is similar, however we introduce the correlation structure not between individual data points but only between true data points and artificially selected data points.

We believe that correctly selecting such a correlation structure of the individual data points can be quite hard in practice, but if such prior knowledge is available, CVAE can be indeed a much more powerful model than a VAE.

We note that a proposal for automatically learning such a correlation structure is also recently proposed by (Louizos et al., 2019) .

In this paper, we have introduced a method for improving robustness of latent representations learned by a VAE.

It must be stressed that our goal is not building the most powerful adversarially robust supervised classifier, but obtaining a method for learning generic representations that can be used for several tasks; the tasks can be even unknown at the time of learning the representations.

While the nominal accuracy of an unsupervised approach is expected to be inferior to a supervised training method that is informed by extra label information, we observe that significant improvements in adversarial robustness can be achieved by our approach that forces smooth representations.

The KL divergence between two Gaussian distributions translates to a well known divergence in the parameters (in the general case this is a Bregman divergence)

where P a = N (µ a , Σ a ) and P b = N (µ b , Σ b ) are Gaussian densities with mean µ · and covariance matrix Σ · , and | · | denotes the determinant for a matrix argument, and Tr denotes the trace.

The KL divergence consists of two terms, the first term is the scale invariant divergence between two covariance matrices also known as a Itakuro-Saito divergence and the second term is a Mahalonobis distance between the means.

The KL divergence is invariant to the choice of parametrization or the choice of the coordinate system.

Consider a set Γ of joint densities Q(Z a , Z b ) with the property that Q has fixed marginals Q a (Z a ) and

The Wasserstein divergence WD is defined as the solution of the optimization problem with respect to pairwise distribution Q

where c(z a , z b ) is a function that specifies the 'cost' of transferring a unit of probability mass from z a to z b .

The 2 -Wasserstein distance W 2 2 for two Gaussians has an interesting form.

The optimum transport plan, where the minimum of (11) is attained, is given

where

b .

It can be checked that this optimal Guassian density is degenerate in the sense that there exists a linear mapping between z a and z b :

where A 1/2 denotes the matrix square root, a symmetric matrix such that (A 1/2 ) 2 = A for a symmetric positive semidefinite matrix A. The 2 -Wasserstein distance is the value attained by the optimum transport plan

Entropy Regularized 2 -Wasserstein is the value attained by the minimizer of the following functional

where H is the entropy of the joint distribution Q. Using the form in (12) subject to the semidefinite constraint

The entropy of a Gaussian Q(z a , z b ) is given by the Schur formula

Here, D is the dimension of the vector (z a , z b ).

The entropy regularized problem has a solution where we need to minimizẽ

Taking the derivative and setting to zero

we obtain a particular Matrix Ricatti equation

that gives us a closed form formula for the specific entropy regularized Wasserstein distance

For the case of two univariate Gaussians, i.e., when the joint distribution has the form

the solution is given by the solution of the scalar quadratic equation.

We take the root that gives a feasible solution as the minimizer.

In the scalar case, this is the solution that satisfies

where we have defined

It can be easily checked that the other root is infeasible.

For the scalar ψ case we obtain

where D z is the dimension of the latent representation, and µ k and Σ k are the k'th component of the output of a neural network with parameters η.

Similarly, x i denotes the i'th component of the observation vector x of size D x .

For optimization, we need an unbiased estimate of the gradient of the SE-ELBO with respect to encoder parameters η and decoder parameters θ:

Given x, we first select a fictive sample x via a selection mechanism, in this case as an adversarial attack as explained in section 3.1.

Sample a latent representation and calculate the associated prediction

The terms of the SE-ELBO can be calculated as

We always train decoder-encoder pairs with identical architectures using both the standard VAE ELBO and the SE ELBO with a fixed γ.

Then, in each case by fixing the encoder (that is essentially using the same representation) and by only using the mean activations of the encoders, we train linear classifiers using standard training for solving several downstream tasks.

For both encoder and decoder networks we use a 4 layer multi layer perceptron (MLP) and a convolutional network (ConvNET) architectures with 200 units of ReLU activations at each layer.

We carried out experiments with latent space dimensions of 32, 64 and 128, corresponding to an output sizes of an encoder with 64, 128 and 256 units, with two units per dimensions to encode the mean and the log-variance parameters of a fully factorized Gaussian condition distribution.

The training is done using the Adam optimizer.

Each network (both the encoder and decoder) are randomly initialized and trained for 300K iterations.

GANs are presented as neural sampling models of observations x of form x = f (ζ; η) where f is typically a deep neural network with parameters η, and ζ is a realization from some simple distribution p(Z).

In the context of GANs, the function f is called a generator.

When the dimension of x is bigger than the dimension of ζ, the density p(x) induced by the transformation f is inevitably a degenerate distribution.

Since f is continuous, and it is concentrated on a subset of the data space X f ≡ {x : ∃ζ, x = f (ζ; η)}. Our use of letter f and parameters η is deliberate and we will illustrate in the sequel that the generator network of a GANs is actually analogous to a smooth encoder, where the roles of the latent variables and observations are switched, but we will first review GANs.

To fit a degenerate distribution to a dataset, the GAN approach adopts a strategy where the generator is co-trained with a second neural network d(x; w) with parameters w with the following objective

where D real (x) is the empirical data distribution.

This objective is (unfortunately) referred as an adversarial objective in the literature, not to be confused with adversarial attack mechanism in the context of supervised learning Madry et al. (2017) .

The function d is called a discriminator.

After replacing expectations with finite sample averages, this objective enforces that in a dataset that contains both synthetically generated (fake) and real examples, the classification function d should increase the correct classification rate by discriminating fakes from real examples while the generator f should decrease the detection rate of fake examples.

When 0 ≤ d(·) ≤ 1, which is the case for a classifier, one can also write the objective as

where l(x; w) = log d(x; w).

This form also highlights an interesting alternative formulation and an interpretation in terms of optimal transport.

In fact, not long after the seminal work of Goodfellow et al. (2014) , the mechanism beyond the GAN objective and its direct connection to the theory of optimal transport has been recognized by the seminal paper Arjovsky et al. (2017) where the problem is further framed as

with the constraint that |l(x; w) − l(x; w)| ≤ c(x,x) , i.e. l is a Lipschitz function for some L where c(x,x) ≤ L x −x .

Here, D fake (x; θ) is the fitted density ofx = f (ζ; η).

This is the dual formulation of the optimal transport problem, that can be understood as an economic transaction between a customer and a shipment company.

Here, the difference l(x; w) − l(x; w) can be interpreted as the profit made by the company for the shipment of one unit of mass from x and tox, and the Lipschitz condition ensures that it makes still sense for the customer to make use of the services of the company rather than simply doing the transport of her own Solomon (2018) .

The customer wants to pay less, so she should minimize the profit of the company.

This can be achieved by changing the desired delivery distribution D fake by adjusting θ, so that the transfer from the fixed source distribution D real is minimized.

Ideally, when D fake = D real , there is nothing to transfer and no cost is incurred.

This objective also minimizes the Wasserstein distance between the actual data distribution D real and the fake data distribution D fake as given by the generator.

Once the GAN objective can be viewed as minimizing a particular Wasserstein distance, it is rather straightforward to view it as a maximizer of a particular ELBO corresponding to a particular smooth encoder, albeit in one where the positions of the observations and the latents are exchanged and a very large coupling coefficient γ is chosen.

Moreover, the variational marginals have specific forms: One marginal Q a (X) is chosen as the empirical data distribution and the other marginal is chosen as having the form of a neural sampler

The artificial extended target becomes

It can be seen that the ELBO in this case becomes

Now, by taking the coupling γ sufficiently large, the coupling term dominates the lower bound and we obtain the Wasserstein minimization objective.

The random draws from p(Z) become the selection mechanism.

Moreover, the terms that depend on the artificial target p(Z|X, θ) become also irrelevant so in this regime the problem becomes just solving the optimal transport problem between Q a and Q b .

A link between entropic GANs and VAEs is also pointed at in the literature, albeit for calculating a likelihood for GANs Balaji et al. (2018) .

However, our motivations as well as the interpretation of the connection is quite different and we view the GAN decoder as an instance of the smooth encoder.

Targeting the encoder to an augmented distribution different than the decoder us the freedom to express some extensions of VAE in the same framework.

One of such extensions is the β-VAE, quite often used for controlling representations replaces the original variational objective (1) with the following objective

The justification in the original paper Higgins et al. (2017) is obtained from an implicit robustness criteria where D KL (q(Z|X a = x, η)||p(Z)) < and β appears in a Lagrangian formulation.

Hoffman & Johnson (2016) have also provided an alternative justification.

In our formulation, β can be simply interpreted as a dispersion term that is related to the number of points selected by the selection mechanism.

To see this, suppose the selection mechanism chooses β − 1 points x b,i where i = 1 . . .

β − 1 that are identical to the true observation x = x b,i = x i for i = 1 . . .

β − 1.

Now, instead of integrating out Z 1:β−1 , we choose a variational distribution with identical marginals of form

The variational lower bound becomes identical to the β-VAE objective as

where the last step follows due to the functional form of the variational distribution.

E TECHNICAL RESULTS

In section 2.2, we have defined a batch ELBO (2).

To see the connection to VAE ELBO (1)

we first define the empirical data distribution π(X) = 1 N N i=1 δ(X − x i ).

We can now write log p(X = x|θ) ≥ E {log p(X = x|Z, θ)} q(Z|X=x,η) − D KL (q(Z|X = x, η)||p(Z)) ≡ B x (η, θ)

E {log p(X = x i |Z, θ)} q(Z|X=xi,η)

E {log q(Z|X = x i , η)} q(Z|X=xi,η)

E {log p(Z)} q(Z|X=xi,η) = E {log p(X|Z, θ)} q(Z|X,η)π(X) − E {log q(Z|X, η)} q(Z|X,η)π(X)

+E {log p(Z)} q(Z|X,η)π(X) −E {log π(X)} q(Z|X,η)π(X) + E {log π(X)} π(X) = −D KL (q(Z|X, η)π(X)||p(X|Z, θ)p(Z)) + const

This result shows that the ELBO is minimizing the KL distance between one exact and one approximate factorization of the joint distribution p(X, Z) = p(X|Z, θ)p(Z) ≈ q(Z|X, η)π(X).

In the context of a VAE, the smoothness of the decoder is implicitly enforced by the highly constrained encoder distribution and the dynamics of an SGD based training.

In the sequel, we will illustrate that, if two latent coordinates are sufficiently close, the decoder mean mapping is forced to be bounded.

In a standard VAE, the encoder output for each data point is conditionally Gaussian as q(Z|X = x; η) = N (f µ (x; η), f Σ (x; η)).

The decoder is chosen as p(X|Z = z; η) = N (g(z; θ), vI).

The decoder parameters θ under the ELBO depend only on the data fidelity term x − g(z; θ) 2 /v.

For a moment, assume that the encoder is fixed and focus on a single data point x. During training, a set of latent state vectors z i for i = 1 . . .

T are sampled from the conditionally Gaussian encoder distribution.

When the dimension of the latent space D z is large, these samples z i will be with high probability on the typical set.

The typical set of a nondegenerate Gaussian distribution is approximately the surface of a Mahalanobis ball, a compact hyper-ellipsoid M (x) centered at f µ (x; η) with scaling matrix f Σ (x; η) 1/2 .

If we assume that the training procedure is able to reduce the error in the sense that x − g(z i ; θ) ≤ E for all z i where E is a bound on the error magnitude for z i sampled from the encoder, the decoder is forced to give the same output for each point approximately on M (x).

For a point z a drawn from q(Z|X = x; η) we have z a − f µ (x; η) K ≈ D z with high probability where K = f Σ (x; η) −1 and x K ≡ √ x Kx.

For a point z b independently drawn from q(Z|X = x; η), by the triangle inequality we have g(z a ; θ) − g(z b ; θ) ≤ 2E

where the Mahalanobis distance

where λ min is the smallest eigenvalue of the covariance matrix.

Hence the distance is also bounded when the variance is not degenerate and minimum distance will be on the order of z a − z b ≈ 2 √ D z λ min so we expect the ratio to be bounded

We see that the ELBO objective enforces the decoder to be invariant on the typical set of q(Z|X = x; η), where most of the probability mass is concentrated.

Now, for each data point x, the corresponding latent space hyper-ellipsoid M (x) are forced to be large in the sense of having a large determinant by the entropy term of the encoder that promotes large log-determinant.

The size of M (x) is also controlled by the prior fidelity term, avoiding blowing up.

Hence the union ∪ x∈X M (x), where X is the dataset, will approximately cover the latent space when the encoder has converged and on each hyper-ellipsoid M (x) the decoder will be enforced to be smooth.

In this section we show that the smooth encoder training forces a small Lipschitz constant for the encoder mean mapping.

To simplify the argument, we will assume that the variance mapping of the encoder would be a constant function that does not vary with x, i.e., f Σ (x; η) = Σ(η).

The latter assumption could be removed by considering a metric on the joint space of the means and covariance.

Using the adversarial selection mechanism, during training we solve the following problem using PGD: x * = arg max

x : x −x p ≤ WD(q(Z|X = x, η), q(Z|X = x , η))

Assuming that PGD finds the global maximum at the boundary of the -ball where x − x * p = , under constant variance assumption for the encoder we can see that the Wasserstein divergence simply becomes the square distance between mean mappings WD(q(Z|X = x, η), q(Z|X = x * , η)) = f µ (x; η) − f µ (x * ; η) 2 2

We know that the SE ELBO objective has to minimize this distance for any coupling term γ so the procedure actually tries to reduce the local Lipschitz constant L(x) around data point x L(x) = f µ (x; η) − f µ (x * ; η)

x − x * p ≤ E and promotes smoothness where E is an upper bound on the change in the representation f µ (x; η)− f µ (x * ; η) ≤ E.

<|TLDR|>

@highlight

We propose a method for computing adversarially robust representations in an entirely unsupervised way.