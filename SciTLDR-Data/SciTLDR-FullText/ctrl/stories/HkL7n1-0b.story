We propose the Wasserstein Auto-Encoder (WAE)---a new algorithm for building a generative model of the data distribution.

WAE minimizes a penalized form of the Wasserstein distance between the model distribution and the target distribution, which leads to a different regularizer than the one used by the Variational Auto-Encoder (VAE).

This regularizer encourages the encoded training distribution to match the prior.

We compare our algorithm with several other techniques and show that it is a generalization of adversarial auto-encoders (AAE).

Our experiments show that WAE shares many of the properties of VAEs (stable training, encoder-decoder architecture, nice latent manifold structure) while generating samples of better quality.

The field of representation learning was initially driven by supervised approaches, with impressive results using large labelled datasets.

Unsupervised generative modeling, in contrast, used to be a domain governed by probabilistic approaches focusing on low-dimensional data.

Recent years have seen a convergence of those two approaches.

In the new field that formed at the intersection, variational auto-encoders (VAEs) BID16 constitute one well-established approach, theoretically elegant yet with the drawback that they tend to generate blurry samples when applied to natural images.

In contrast, generative adversarial networks (GANs) BID9 turned out to be more impressive in terms of the visual quality of images sampled from the model, but come without an encoder, have been reported harder to train, and suffer from the "mode collapse" problem where the resulting model is unable to capture all the variability in the true data distribution.

There has been a flurry of activity in assaying numerous configurations of GANs as well as combinations of VAEs and GANs.

A unifying framework combining the best of GANs and VAEs in a principled way is yet to be discovered.

This work builds up on the theoretical analysis presented in BID3 .

Following ; BID3 , we approach generative modeling from the optimal transport (OT) point of view.

The OT cost (Villani, 2003) is a way to measure a distance between probability distributions and provides a much weaker topology than many others, including f -divergences associated with the original GAN algorithms BID25 .

This is particularly important in applications, where data is usually supported on low dimensional manifolds in the input space X .

As a result, stronger notions of distances (such as f -divergences, which capture the density ratio between distributions) often max out, providing no useful gradients for training.

In contrast, OT was claimed to have a nicer behaviour BID11 although it requires, in its GAN-like implementation, the addition of a constraint or a regularization term into the objective. : Both VAE and WAE minimize two terms: the reconstruction cost and the regularizer penalizing discrepancy between P Z and distribution induced by the encoder Q. VAE forces Q(Z|X = x) to match P Z for all the different input examples x drawn from P X .

This is illustrated on picture (a), where every single red ball is forced to match P Z depicted as the white shape.

Red balls start intersecting, which leads to problems with reconstruction.

In contrast, WAE forces the continuous mixture Q Z := Q(Z|X)dP X to match P Z , as depicted with the green ball in picture (b).

As a result latent codes of different examples get a chance to stay far away from each other, promoting a better reconstruction.

In this work we aim at minimizing OT W c (P X , P G ) between the true (but unknown) data distribution P X and a latent variable model P G specified by the prior distribution P Z of latent codes Z ??? Z and the generative model P G (X|Z) of the data points X ??? X given Z. Our main contributions are listed below (cf. also FIG0 ):??? A new family of regularized auto-encoders (Algorithms 1, 2 and Eq. 4), which we call Wasserstein Auto-Encoders (WAE), that minimize the optimal transport W c (P X , P G ) for any cost function c. Similarly to VAE, the objective of WAE is composed of two terms: the c-reconstruction cost and a regularizer D Z (P Z , Q Z ) penalizing a discrepancy between two distributions in Z: P Z and a distribution of encoded data points, i.e. DISPLAYFORM0 When c is the squared cost and D Z is the GAN objective, WAE coincides with adversarial auto-encoders of BID23 .???

Empirical evaluation of WAE on MNIST and CelebA datasets with squared cost c(x, y) = x ??? y 2 2 .

Our experiments show that WAE keeps the good properties of VAEs (stable training, encoder-decoder architecture, and a nice latent manifold structure) while generating samples of better quality, approaching those of GANs.??? We propose and examine two different regularizers D Z (P Z , Q Z ).

One is based on GANs and adversarial training in the latent space Z. The other uses the maximum mean discrepancy, which is known to perform well when matching high-dimensional standard normal distributions P Z BID10 .

Importantly, the second option leads to a fully adversary-free min-min optimization problem.??? Finally, the theoretical considerations presented in BID3 and used here to derive the WAE objective might be interesting in their own right.

In particular, Theorem 1 shows that in the case of generative models, the primal form of W c (P X , P G ) is equivalent to a problem involving the optimization of a probabilistic encoder Q(Z|X) .The paper is structured as follows.

In Section 2 we review a novel auto-encoder formulation for OT between P X and the latent variable model P G derived in BID3 .

Relaxing the resulting constrained optimization problem we arrive at an objective of Wasserstein auto-encoders.

We propose two different regularizers, leading to WAE-GAN and WAE-MMD algorithms.

Section 3 discusses the related work.

We present the experimental results in Section 4 and conclude by pointing out some promising directions for future work.

Our new method minimizes the optimal transport cost W c (P X , P G ) based on the novel auto-encoder formulation (see Theorem 1 below).

In the resulting optimization problem the decoder tries to accurately reconstruct the encoded training examples as measured by the cost function c. The encoder tries to simultaneously achieve two conflicting goals: it tries to match the encoded distribution of training examples Q Z := E P X [Q(Z|X)] to the prior P Z as measured by any specified divergence D Z (Q Z , P Z ), while making sure that the latent codes provided to the decoder are informative enough to reconstruct the encoded training examples.

This is schematically depicted on FIG0 .

We use calligraphic letters (i.e. X ) for sets, capital letters (i.e. X) for random variables, and lower case letters (i.e. x) for their values.

We denote probability distributions with capital letters (i.e. P (X)) and corresponding densities with lower case letters (i.e. p(x)).

In this work we will consider several measures of discrepancy between probability distributions P X and P G .

The class of f -divergences BID21 ) is defined by DISPLAYFORM0

A rich class of divergences between probability distributions is induced by the optimal transport (OT) problem (Villani, 2003) .

Kantorovich's formulation of the problem is given by DISPLAYFORM0 where c(x, y) : X ?? X ??? R + is any measurable cost function and P(X ??? P X , Y ??? P G ) is a set of all joint distributions of (X, Y ) with marginals P X and P G respectively.

A particularly interesting case is when (X , d) is a metric space and c(x, y) = d p (x, y) for p ??? 1.

In this case W p , the p-th root of W c , is called the p-Wasserstein distance.

When c(x, y) = d(x, y) the following Kantorovich-Rubinstein duality holds 1 : DISPLAYFORM1 where F L is the class of all bounded 1-Lipschitz functions on (X , d).

One way to look at modern generative models like VAEs and GANs is to postulate that they are trying to minimize certain discrepancy measures between the data distribution P X and the model P G .

Unfortunately, most of the standard divergences known in the literature, including those listed above, are hard or even impossible to compute, especially when P X is unknown and P G is parametrized by deep neural networks.

Previous research provides several tricks to address this issue.

In case of minimizing the KL-divergence D KL (P X , P G ), or equivalently maximizing the marginal log-likelihood E P X [log p G (X)], the famous variational lower bound provides a theoretically grounded framework successfully employed by VAEs BID16 BID24 .

More generally, if the goal is to minimize the f -divergence D f (P X , P G ) (with one example being D KL ), one can resort to its dual formulation and make use of f -GANs and the adversarial training BID25 .

Finally, OT cost W c (P X , P G ) is yet another option, which can be, thanks to the celebrated Kantorovich-Rubinstein duality (2), expressed as an adversarial objective as implemented by the Wasserstein-GAN .

We include an extended review of all these methods in Supplementary A.In this work we will focus on latent variable models P G defined by a two-step procedure, where first a code Z is sampled from a fixed distribution P Z on a latent space Z and then Z is mapped to the image X ??? X = R d with a (possibly random) transformation.

This results in a density of the form DISPLAYFORM0 assuming all involved densities are properly defined.

For simplicity we will focus on non-random decoders, i.e. generative models P G (X|Z) deterministically mapping Z to X = G(Z) for a given map G : Z ??? X .

Similar results for random decoders can be found in Supplementary B.1.It turns out that under this model, the OT cost takes a simpler form as the transportation plan factors through the map G: instead of finding a coupling ?? in (1) between two random variables living in the X space, one distributed according to P X and the other one according to P G , it is sufficient to find a conditional distribution DISPLAYFORM1 is identical to the prior distribution P Z .

This is the content of the theorem below proved in BID3 .

To make this paper self contained we repeat the proof in Supplementary B.Theorem 1 For P G as defined above with deterministic P G (X|Z) and any function G : DISPLAYFORM2 where Q Z is the marginal distribution of Z when X ??? P X and Z ??? Q(Z|X).This result allows us to optimize over random encoders Q(Z|X) instead of optimizing over all couplings between X and Y .

Of course, both problems are still constrained.

In order to implement a numerical solution we relax the constraints on Q Z by adding a penalty to the objective.

This finally leads us to the WAE objective: DISPLAYFORM3 where Q is any nonparametric set of probabilistic encoders, D Z is an arbitrary divergence between Q Z and P Z , and ?? > 0 is a hyperparameter.

Similarly to VAE, we propose to use deep neural networks to parametrize both encoders Q and decoders G. Note that as opposed to VAEs, the WAE formulation allows for non-random encoders deterministically mapping inputs to their latent codes.

We propose two different penalties D Z (Q Z , P Z ): DISPLAYFORM4 and use the adversarial training to estimate it.

Specifically, we introduce an adversary (discriminator) in the latent space Z trying to separate 2 "true" points sampled from P Z and "fake" ones sampled from Q Z BID9 .

This results in the WAE-GAN described in Algorithm 1.

Even though WAE-GAN falls back to the min-max problem, we move the adversary from the input (pixel) space X to the latent space Z. On top of that, P Z may have a nice shape with a single mode (for a Gaussian prior), in which case the task should be easier than matching an unknown, complex, and possibly multi-modal distributions as usually done in GANs.

This is also a reason for our second penalty: DISPLAYFORM5 For a positive-definite reproducing kernel k : Z ?? Z ??? R the following expression is called the maximum mean discrepancy (MMD): DISPLAYFORM6 where H k is the RKHS of real-valued functions mapping Z to R. If k is characteristic then MMD k defines a metric and can be used as a divergence measure.

We propose to use DISPLAYFORM7 Fortunately, MMD has an unbiased U-statistic estimator, which can be used in conjunction with stochastic gradient descent (SGD) methods.

This results in the WAE-MMD described in Algorithm 2.

It is well known that the maximum mean discrepancy performs well when matching high-dimensional standard normal distributions BID10 ) so we expect this penalty to work especially well working with the Gaussian prior P Z .

Require: Regularization coefficient ?? > 0.

Initialize the parameters of the encoder Q ?? , decoder G ?? , and latent discriminator D??.

while (??, ??) not converged do Sample {x1, . . .

, xn} from the training set Sample {z1, . . .

, zn} from the prior PZ Samplezi from Q ?? (Z|xi) for i = 1, . . .

, n Update D?? by ascending: DISPLAYFORM0 Update Q ?? and G ?? by descending: DISPLAYFORM1 end while ALGORITHM 2 Wasserstein Auto-Encoder with MMD-based penalty (WAE-MMD).Require: Regularization coefficient ?? > 0, characteristic positive-definite kernel k.

Initialize the parameters of the encoder Q ?? , decoder G ?? , and latent discriminator D??.

while (??, ??) not converged do Sample {x1, . . .

, xn} from the training set Sample {z1, . . .

, zn} from the prior PZ Samplezi from Q ?? (Z|xi) for i = 1, . . .

, n Update Q ?? and G ?? by descending: DISPLAYFORM2 end whileWe point out once again that the encoders Q ?? (Z|x) in Algorithms 1 and 2 can be non-random, i.e. deterministically mapping input points to the latent codes.

In this case Q ?? (Z|x) = ?? ?? ?? (x) for a function ?? ?? : X ??? Z and in order to samplez i from Q ?? (Z|x i ) we just need to return ?? ?? (x i ).

Literature on auto-encoders Classical unregularized auto-encoders minimize only the reconstruction cost.

This results in different training points being encoded into non-overlapping zones chaotically scattered all across the Z space with "holes" in between where the decoder mapping P G (X|Z) has never been trained.

Overall, the encoder Q(Z|X) trained in this way does not provide a useful representation and sampling from the latent space Z becomes hard BID1 .Variational auto-encoders BID16 ) minimize a variational bound on the KLdivergence D KL (P X , P G ) which is composed of the reconstruction cost plus the regularizer DISPLAYFORM0 The regularizer captures how distinct the image by the encoder of each training example is from the prior P Z , which is not guaranteeing that the overall encoded distribution E P X [Q(Z|X)] matches P Z like WAE does.

Also, VAEs require non-degenerate (i.e. nondeterministic) Gaussian encoders and random decoders for which the term log p G (x|z) can be computed and differentiated with respect to the parameters.

Later BID24 proposed a way to use VAE with non-Gaussian encoders.

WAE minimizes the optimal transport W c (P X , P G ) and allows both probabilistic and deterministic encoder-decoder pairs of any kind.

The VAE regularizer can be also equivalently written BID13 as a sum of D KL (Q Z , P Z ) and a mutual information I Q (X, Z) between the images X and latent codes Z jointly distributed according to P X ?? Q(Z|X).

This observation provides another intuitive way to explain a difference between our algorithm and VAEs: WAEs simply drop the mutual information term I Q (X, Z) in the VAE regularizer.

When used with c(x, y) = x ??? y 2 2 WAE-GAN is equivalent to adversarial auto-encoders (AAE) proposed by BID23 .

Theory of BID3 (and in particular Theorem 1) thus suggests that AAEs minimize the 2-Wasserstein distance between P X and P G .

This provides the first theoretical justification for AAEs known to the authors.

WAE generalizes AAE in two ways: first, it can use any cost function c in the input space X ; second, it can use any discrepancy measure D Z in the latent space Z (for instance MMD), not necessarily the adversarial one of WAE-GAN.Finally, Zhao et al. (2017b) independently proposed a regularized auto-encoder objective similar to BID3 and our (4) based on very different motivations and arguments.

Following VAEs their objective (called InfoVAE) defines the reconstruction cost in the image space implicitly through the negative log likelihood term ??? log p G (x|z), which should be properly normalized for all z ??? Z. In theory VAE and InfoVAE can both induce arbitrary cost functions, however in practice this may require an estimation of the normalizing constant (partition function) which can 3 be different for different values of z. WAEs specify the cost c(x, y) explicitly and don't constrain it in any way.

Literature on OT Genevay et al. (2016) address computing the OT cost in large scale using SGD and sampling.

They approach this task either through the dual formulation, or via a regularized version of the primal.

They do not discuss any implications for generative modeling.

Our approach is based on the primal form of OT, we arrive at regularizers which are very different, and our main focus is on generative modeling.

The WGAN minimizes the 1-Wasserstein distance W 1 (P X , P G ) for generative modeling.

The authors approach this task from the dual form.

Their algorithm comes without an encoder and can not be readily applied to any other cost W c , because the neat form of the Kantorovich-Rubinstein duality (2) holds only for W 1 .

WAE approaches the same problem from the primal form, can be applied for any cost function c, and comes naturally with an encoder.

In order to compute the values (1) or (2) of OT we need to handle non-trivial constraints, either on the coupling distribution ?? or on the function f being considered.

Various approaches have been proposed in the literature to circumvent this difficulty.

For W 1 tried to implement the constraint in the dual formulation (2) by clipping the weights of the neural network f .

Later BID11 proposed to relax the same constraint by penalizing the objective of (2) with a term ?? ?? E ( ???f (X) ??? 1) 2 which should not be greater than 1 if f ??? F L .

In a more general OT setting of W c BID5 proposed to penalize the objective of (1) with the KLdivergence ?? ?? D KL (??, P ??? Q) between the coupling distribution and the product of marginals.

BID8 showed that this entropic regularization drops the constraints on functions in the dual formulation as opposed to (2).

Finally, in the context of unbalanced optimal transport it has been proposed to relax the constraint in (1) by regularizing the objective with BID4 BID20 , where ?? X and ?? Y are marginals of ??. In this paper we propose to relax OT in a way similar to the unbalanced optimal transport, i.e. by adding additional divergences to the objective.

However, we show that in the particular context of generative modeling, only one extra divergence is necessary.

DISPLAYFORM1 Literature on GANs Many of the GAN variations (including f -GAN and WGAN) come without an encoder.

Often it may be desirable to reconstruct the latent codes and use the learned manifold, in which cases these models are not applicable.

There have been many other approaches trying to blend the adversarial training of GANs with autoencoder architectures (Zhao et al., 2017a; BID6 BID29 BID2 .

The approach proposed by BID29 is perhaps the most relevant to our work.

The authors use the discrepancy between Q Z and the distribution E Z ???P Z [Q Z|G(Z ) ] of auto-encoded noise vectors as the objective for the max-min game between the encoder and decoder respectively.

While the authors showed that the saddle points correspond to P X = P G , they admit that encoders and decoders trained in this way have no incentive to be reciprocal.

As a workaround they propose to include an additional reconstruction term to the objective.

WAE does not necessarily lead to a min-max game, uses a different penalty, and has a clear theoretical foundation.

Several works used reproducing kernels in context of GANs.

BID19 ; BID7 use MMD with a fixed kernel k to match P X and P G directly in the input space X .

These methods have been criticised to require larger mini-batches during training: estimating MMD k (P X , P G ) requires number of samples roughly proportional to the dimensionality of the input space X BID28 which is typically larger than 10 3 .

BID18 take a similar approach but further train k adversarially so as to arrive at a meaningful loss function.

WAE-MMD uses MMD to match Q Z to the prior P Z in the latent space Z. Typically Z has no more than 100 dimensions and P Z is Gaussian, which allows us to use regular mini-batch sizes to accurately estimate MMD.

Random samples

In this section we empirically evaluate 4 the proposed WAE model.

We would like to test if WAE can simultaneously achieve (i) accurate reconstructions of data points, (ii) reasonable geometry of the latent manifold, and (iii) random samples of good (visual) quality.

Importantly, the model should generalize well: requirements (i) and (ii) should be met on both training and test data.

We trained WAE-GAN and WAE-MMD (Algorithms 1 and 2) on two real-world datasets: MNIST (LeCun et al., 1998) consisting of 70k images and CelebA BID22 containing roughly 203k images.

Experimental setup In all reported experiments we used Euclidian latent spaces Z = R dz for various d z depending on the complexity of the dataset, isotropic Gaussian prior distributions P Z (Z) = N (Z; 0, ?? 2 z ?? I d ) over Z, and a squared cost function c(x, y) = x ??? y 2 2 for data points x, y ??? X = R dx .

We used deterministic encoder-decoder pairs, Adam BID15 ) with ?? 1 = 0.5, ?? 2 = 0.999, and convolutional deep neural network architectures for encoder mapping ?? ?? : X ??? Z and decoder mapping G ?? : Z ??? X similar to the DCGAN ones reported by BID27 with batch normalization BID14 .

We tried various values of ?? and noticed that ?? = 10 seems to work good across all datasets we considered.

Since we are using deterministic encoders, choosing d z larger than intrinsic dimensionality of the dataset would force the encoded distribution Q Z to live on a manifold in Z. This would make matching Q Z to P Z impossible if P Z is Gaussian and may lead to numerical instabilities.

We use d z = 8 for MNIST and d z = 64 for CelebA which seems to work reasonably well.

Random samples VAE WAE-MMD WAE-GAN Figure 3 : VAE (left column), WAE-MMD (middle column), and WAE-GAN (right column) trained on CelebA dataset.

In "test reconstructions" odd rows correspond to the real test points.

We also report results of VAEs.

VAEs used the same latent spaces as discussed above and standard Gaussian priors P Z = N (0, I d ).

We used Gaussian encoders Q(Z|X) = N Z; ?? ?? (X), ??(X) with mean ?? ?? and diagonal covariance ??. For both MNIST and CelebA we used Bernoulli decoders parametrized by G ?? .

Functions ?? ?? , ??, and G ?? were parametrized by deep nets of the same architectures as used in WAE.

In WAE-GAN we used discriminator D composed of several fully connected layers with ReLu.

We tried WAE-MMD with the RBF kernel but observed that it fails to penalize the outliers of Q Z because of the quick tail decay.

If the codesz = ?? ?? (x) for some of the training points x ??? X end up far away from the support of P Z (which may happen in the early stages of training) the corresponding terms in the U-statistic k(z,z) = e ??? z???z 2 2 /?? 2 k will quickly approach zero and provide no gradient for those outliers.

This could be avoided by choosing the kernel bandwidth ?? 2 k in a data-dependent manner, however in this case per-minibatch U-statistic would not provide an unbiased estimate for the gradient.

Instead, we used the inverse multiquadratics kernel k(x, y) = C/(C + x ??? y 2 2 ) which is also characteristic and has much heavier tails.

In all experiments we used C = 2d z ?? 2 z , which is the expected squared distance between two multivariate Gaussian vectors drawn from P Z .

This significantly improved the performance compared to the RBF kernel (even the one with ?? Random samples are generated by sampling P Z and decoding the resulting noise vectors z into G ?? (z).

As expected, in our experiments we observed that for both WAE-GAN and WAE-MMD the quality of samples strongly depends on how accurately Q Z matches P Z .

To see this, notice that during training the decoder function G ?? is presented only with encoded versions ?? ?? (X) of the data points X ??? P X .

Indeed, the decoder is trained on samples from Q Z and thus there is no reason to expect good results when feeding it with samples from P Z .

In our experiments we noticed that even slight differences between Q Z and P Z may affect the quality of samples.

In some cases WAE-GAN seems to lead to a better matching and generates better samples than WAE-MMD.

However, due to adversarial training WAE-GAN is highly unstable, while WAE-MMD has a very stable training much like VAE.

In order to quantitatively assess the quality of the generated images, we use the Fr??chet Inception Distance introduced by BID12 and report the results on CelebA in Table 1 .

These results confirm that the sampled images from WAE are of better quality than from VAE, and WAE-GAN gets a slightly better score than WAE-MMD, which correlates with visual inspection of the images.

Test reconstructions and interpolations.

We take random points x from the held out test set and report their auto-encoded versions G ?? (?? ?? (x)).

Next, pairs (x, y) of different data points are sampled randomly from the held out test set and encoded: z x = ?? ?? (x), z y = ?? ?? (y).

We linearly interpolate between z x and z y with equally-sized steps in the latent space and show decoded images.

Using the optimal transport cost, we have derived Wasserstein auto-encoders-a new family of algorithms for building generative models.

We discussed their relations to other probabilistic modeling techniques.

We conducted experiments using two particular implementations of the proposed method, showing that in comparison to VAEs, the images sampled from the trained WAE models are of better quality, without compromising the stability of training and the quality of reconstruction.

Future work will include further exploration of the criteria for matching the encoded distribution Q Z to the prior distribution P Z , assaying the possibility of adversarially training the cost function c in the input space X , and a theoretical analysis of the dual formulations for WAE-GAN and WAE-MMD.

Even though GANs and VAEs are quite different-both in terms of the conceptual frameworks and empirical performance-they share important features: (a) both can be trained by sampling from the model P G without knowing an analytical form of its density and (b) both can be scaled up with SGD.

As a result, it becomes possible to use highly flexible implicit models P G defined by a twostep procedure, where first a code Z is sampled from a fixed distribution P Z on a latent space Z and then Z is mapped to the image G(Z) ??? X = R d with a (possibly random) transformation G : Z ??? X .

This results in latent variable models P G of the form (3).These models are indeed easy to sample and, provided G can be differentiated analytically with respect to its parameters, P G can be trained with SGD.

The field is growing rapidly and numerous variations of VAEs and GANs are available in the literature.

Next we introduce and compare several of them.

The original generative adversarial network (GAN) BID9 approach minimizes DISPLAYFORM0 with respect to a deterministic decoder G : Z ??? X , where T is any non-parametric class of choice.

It is known that D GAN (P X , P G ) ??? 2 ?? D JS (P X , P G ) ??? log(4) and the inequality turns into identity in the nonparametric limit, that is when the class T becomes rich enough to represent all functions mapping X to (0, 1).

Hence, GANs are minimizing a lower bound on the JS-divergence.

However, GANs are not only linked to the JS-divergence: the f -GAN approach BID25 showed that a slight modification D f,GAN of the objective (5) allows to lower bound any desired f -divergence in a similar way.

In practice, both decoder G and discriminator T are trained in alternating SGD steps.

Stopping criteria as well as adequate evaluation of the trained GAN models remain open questions.

Recently, the authors of argued that the 1-Wasserstein distance W 1 , which is known to induce a much weaker topology than D JS , may be better suited for generative modeling.

When P X and P G are supported on largely disjoint low-dimensional manifolds (which may be the case in applications), D KL , D JS , and other strong distances between P X and P G max out and no longer provide useful gradients for P G .

This "vanishing gradient" problem necessitates complicated scheduling between the G/T updates.

In contrast, W 1 is still sensible in these cases and provides stable gradients.

The Wasserstein GAN (WGAN) minimizes DISPLAYFORM1 where W is any subset of 1-Lipschitz functions on X .

It follows from (2) that D WGAN (P X , P G ) ??? W 1 (P X , P G ) and thus WGAN is minimizing a lower bound on the 1-Wasserstein distance.

Variational auto-encoders (VAE) BID16 utilize models P G of the form FORMULA4 and minimize DISPLAYFORM2 with respect to a random decoder mapping P G (X|Z).

The conditional distribution P G (X|Z) is often parametrized by a deep net G and can have any form as long as its density p G (x|z) can be computed and differentiated with respect to the parameters of G. A typical choice is to use Gaussians P G (X|Z) = N (X; G(Z), ?? 2 ?? I).

If Q is the set of all conditional probability distributions Q(Z|X), the objective of VAE coincides with the negative marginal log-likelihood D VAE (P X , P G ) = ???E P X [log P G (X)].

However, in order to make the D KL term of (6) tractable in closed form, the original implementation of VAE uses a standard normal P Z and restricts Q to a class of Gaussian distributions Q(Z|X) = N Z; ??(X), ??(X) with mean ?? and diagonal covariance ?? parametrized by deep nets.

As a consequence, VAE is minimizing an upper bound on the negative log-likelihood or, equivalently, on the KL-divergence D KL (P X , P G ).One possible way to reduce the gap between the true negative log-likelihood and the upper bound provided by D VAE is to enlarge the class Q. Adversarial variational Bayes (AVB) BID24 follows this argument by employing the idea of GANs.

Given any point x ??? X , a noise ??? N (0, 1), and any fixed transformation e : X ?? R ??? Z, a random variable e(x, ) implicitly defines one particular conditional distribution Q e (Z|X = x).

AVB allows Q to contain all such distributions for different choices of e, replaces the intractable term D KL Q e (Z|X), P Z in BID31 by the adversarial approximation D f,GAN corresponding to the KL-divergence, and proposes to minimize DISPLAYFORM3 The D KL term in (6) may be viewed as a regularizer.

Indeed, VAE reduces to the classical unregularized auto-encoder if this term is dropped, minimizing the reconstruction cost of the encoder-decoder pair Q(Z|X), P G (X|Z).

This often results in different training points being encoded into nonoverlapping zones chaotically scattered all across the Z space with "holes" in between where the decoder mapping P G (X|Z) has never been trained.

Overall, the encoder Q(Z|X) trained in this way does not provide a useful representation and sampling from the latent space Z becomes hard BID1 .Adversarial auto-encoders (AAE) BID23 replace the D KL term in (6) with another regularizer: DISPLAYFORM4 where Q Z is the marginal distribution of Z when first X is sampled from P X and then Z is sampled from Q(Z|X), also known as the aggregated posterior BID23 .

Similarly to AVB, there is no clear link to log-likelihood, as D AAE ??? D AVB .

The authors of BID23 argue that matching Q Z to P Z in this way ensures that there are no "holes" left in the latent space Z and P G (X|Z) generates reasonable samples whenever Z ??? P Z .

They also report an equally good performance of different types of conditional distributions Q(Z|X), including Gaussians as used in VAEs, implicit models Q e as used in AVB, and deterministic encoder mappings, i.e. Q(Z|X) = ?? ??(X) with ?? : X ??? Z.

We will consider certain sets of joint probability distributions of three random variables (X, Y, Z) ??? X ?? X ?? Z. The reader may wish to think of X as true images, Y as images sampled from the model, and Z as latent codes.

We denote by P G,Z (Y, Z) a joint distribution of a variable pair (Y, Z), where Z is first sampled from P Z and next Y from P G (Y |Z).

Note that P G defined in (3) and used throughout this work is the marginal distribution of Y when (Y, Z) ??? P G,Z .In the optimal transport problem (1), we consider joint distributions ??(X, Y ) which are called couplings between values of X and Y .

Because of the marginal constraint, we can write ??(X, Y ) = ??(Y |X)P X (X) and we can consider ??(Y |X) as a non-deterministic mapping from X to Y .

Theorem 1. shows how to factor this mapping through Z, i.e., decompose it into an encoding distribution Q(Z|X) and the generating distribution P G (Y |Z).As in Section 2.2, P(X ??? P X , Y ??? P G ) denotes the set of all joint distributions of (X, Y ) with marginals P X , P G , and likewise for P(X ??? P X , Z ??? P Z ).

The set of all joint distributions of (X, Y, Z) such that X ??? P X , (Y, Z) ??? P G,Z , and (Y ??? ??? X)|Z will be denoted by P X,Y,Z .

Finally, we denote by P X,Y and P X,Z the sets of marginals on (X, Y ) and (X, Z) (respectively) induced by distributions in P X,Y,Z .

Note that P(P X , P G ), P X,Y,Z , and P X,Y depend on the choice of conditional distributions P G (Y |Z), while P X,Z does not.

In fact, it is easy to check that P X,Z = P(X ??? P X , Z ??? P Z ).

From the definitions it is clear that P X,Y ???

P(P X , P G ) and we immediately get the following upper bound: DISPLAYFORM0 If P G (Y |Z) are Dirac measures (i.e., Y = G(Z)), it turns out that P X,Y = P(P X , P G ):adversary in WAE-GAN to ?? = 5 ?? 10 ???4 .

After 30 epochs we decreased both by factor of 2, and after first 50 epochs further by factor of 5.Both encoder and decoder used fully convolutional architectures with 4x4 convolutional filters.

Encoder architecture: DISPLAYFORM1 Decoder architecture: DISPLAYFORM2 Adversary architecture for WAE-GAN: DISPLAYFORM3 Here Conv k stands for a convolution with k filters, FSConv k for the fractional strided convolution with k filters (first two of them were doubling the resolution, the third one kept it constant), BN for the batch normalization, ReLU for the rectified linear units, and FC k for the fully connected layer mapping to R k .

All the convolutions in the encoder used vertical and horizontal strides 2 and SAME padding.

Finally, we used two heuristics.

First, we always pretrained separately the encoder for several minibatch steps before the main training stage so that the sample mean and covariance of Q Z would try to match those of P Z .

Second, while training we were adding a pixel-wise Gaussian noise truncated at 0.01 to all the images before feeding them to the encoder, which was meant to make the encoders random.

We played with all possible ways of combining these two heuristics and noticed that together they result in slightly (almost negligibly) better results compared to using only one or none of them.

Our VAE model used cross-entropy loss (Bernoulli decoder) and otherwise same architectures and hyperparameters as listed above.

We pre-processed CelebA images by first taking a 140x140 center crops and then resizing to the 64x64 resolution.

We used mini-batches of size 100 and trained the models for various number of epochs (up to 250).

All reported WAE models were trained for 55 epochs and VAE for 68 epochs.

For WAE-MMD we used ?? = 100 and for WAE-GAN ?? = 1.

Both used ?? 2 z = 2.

For WAE-MMD the learning rate of Adam was initially set to ?? = 10 ???3 .

For WAE-GAN the learning rate of Adam for the encoder-decoder pair was initially set to ?? = 3 ?? 10 ???4 and for the adversary to 10 ???3 .

All learning rates were decreased by factor of 2 after 30 epochs, further by factor of 5 after 50 first epochs, and finally additional factor of 10 after 100 first epochs.

Both encoder and decoder used fully convolutional architectures with 5x5 convolutional filters.

Encoder architecture:x ??? R 64??64??3 ??? Conv 128 ??? BN ??? ReLU For WAE-GAN we used a heuristic proposed in Supplementary IV of BID24 .

Notice that the theoretically optimal discriminator would result in D * (z) = log p Z (z) ??? log q Z (z), where p Z and q Z are densities of P Z and Q Z respectively.

In our experiments we added the log prior log p Z (z) explicitly to the adversary output as we know it analytically.

This should hopefully make it easier for the adversary to learn the remaining Q Z density term.

Our VAE model used a cross-entropy reconstruction loss (Bernoulli decoder) and ?? = 10 ???4 as the initial Adam learning rate and the same decay schedule as explained above.

Otherwise all the architectures and hyperparameters were as explained above.

<|TLDR|>

@highlight

We propose a new auto-encoder based on the Wasserstein distance, which improves on the sampling properties of VAE.