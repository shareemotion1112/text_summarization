We study how, in generative adversarial networks, variance in the discriminator's output affects the generator's ability to learn the data distribution.

In particular, we contrast the results from various well-known techniques for training GANs when the discriminator is near-optimal and updated multiple times per update to the generator.

As an alternative, we propose an additional method to train GANs by explicitly modeling the discriminator's output as a bi-modal Gaussian distribution over the real/fake indicator variables.

In order to do this, we train the Gaussian classifier to match the target bi-modal distribution implicitly through meta-adversarial training.

We observe that our new method, when trained together with a strong discriminator, provides meaningful, non-vanishing gradients.

Generative adversarial networks BID7 are a framework for training a generator of some target (i.e., "real") distribution without explicitly defining a parametric generating distribution or a tractable likelihood function.

Training the generator relies on a learning signal from a discriminator or discriminator, which is optimized on a relatively simple objective to distinguish between (i.e., classify) generated (i.e., "fake") and real samples.

In order to match the true distribution, the generator parameters are optimized to maximize the loss as defined by the discriminator, which by analogy makes the generator and discriminator adversaries.

In recent years, GANs have attained strong recognition as being able to generate high-quality images with sharp edges in comparison to maximum-likelihood estimation-based methods BID4 BID10 BID22 BID2 BID26 .

Despite their recent successes, GANs can also be notoriously hard to train, suffering from collapse (i.e., mapping its noise to a very small set of singular outputs), missing modes of the real distribution BID3 , and vanishing and/or unstable gradients .

In practice, successful learning is highly reliant on hyperparameter-tuning and model choice, and finding architectures that work with adversarial learning objectives with any / all of the above problems can be challenging BID20 .Many methods have been proposed to address learning difficulties associated with learning instability.??? The use of autoencoders or posterior models in the generator or discriminator.

These have been shown help to alleviate mode collapse or stabilize learning BID12 BID3 BID5 .???

Regularizing the discriminator, such as with input noise , instance noise BID23 , and gradient norm regularization BID21 ).???

Alternate difference measures, such as integral probability metrics (IPMs, Sriperumbudur et al., 2009 ) metrics.

The most well-known of these use a dual formulation of the Wasserstein distance.

These are implemented via either weight clipping or gradient penalty BID8 .

These can yeild stable gradients, high-quality samples and work on a variety of architectures.

On this last point, it is unclear whether the metric or the associated regularization used to impose Lipschitz is important, as regularization techniques have also been shown to be effective with stabilizing learning in f -divergences BID21 .

In this paper, we study the integral role of variance in the discriminator's output in the regime of the generated distribution and how it ultimately affects learning.

In the following sections, we describe theoretical motivations, an empirical analysis from multiple variants of GANs, and finally propose a regularization scheme to combat vanishing gradients on part of the discriminator when it is well-trained.

The generative adversarial framework entails training a generating function G ?? : Z ??? X , where z ??? Z is noise sampled from a relatively simple prior (such as spherical Gaussian), p(z), and G ?? is modeled by a deep neural network with parameters, ??.

The objective is to find the optimal parameters such that the induced or generated distribution, Q ?? , matches a desired target distribution, P. In order to accomplish this, adversarial models define a discriminator function, D ??,d : X ??? R, modeled by a deep neural network with parameters, ??.

The objective of the discriminator is to estimate a measure of distance or divergence, d(P, Q ?? ), between the two distributions, P and Q ?? .

In practice, which divergence is best to estimate in this context is still a subject of debate, but it suffices to say that the most common distances come from either the family of f -divergences BID18 BID19 , such as the Jensen-Shannon divergence (JSD) or ??-squared divergence BID15 or Integral Probability Metrics (IPMs, Sriperumbudur et al., 2009; , such as the Wasserstein distance BID8 and maximum mean discrepancy (MMD, Sutherland et al., 2016; .

Generally, the estimator given by the family of functions defined by D ??,d represent a family of lower-bounds of the respective divergence: DISPLAYFORM0 In BID7 DISPLAYFORM1 where V JSD (D ??,d , G ?? , P) is a lower bound to JSD = 2 * JSD ???2 log 2.

Here, D = ?? ????? 1 is the composition of the sigmoid function, ??(y) = 1 1+exp(???y) , and a neural network, ??.

More generally, BID19 more formally define a GAN objective in terms of a family of f -divergences: DISPLAYFORM2 where f ??? is a convex conjugate to the f -divergence generator function and ?? f = g f ??? ?? is the composition of a nonlinearity and neural network, such that g f respects the domain of the domain of the corresponding f -divergence.

The above family of f -divergence estimators such as the JSD one above are powerful and can produce generators with high quality samples on complex datasets BID20 , but there is some evidence that they provide poor learning signals for the generator parameters, ??.

As shown in , the JS divergence will be flat everywhere important if P and Q ?? both lie on low-dimensional manifolds (as is likely the case with real data) and do not prefectly align.

In principle, one would think that a better divergence estimate (tighter bound) to a strong divergence metric would provide the most meaningful learning signal for the generator.

In practice, however, as we optimize the JSD discriminator, it will have zero gradients on X and hence provide almost no learning signal for the generator.

In addition, the proxy loss, E x???Q ?? [log D(x)], which is commonly used as an alternative to the GAN objective in training the discriminator can be highly unstable, and is generally plagued by missing modes and model collapse.

Finally, as we will show below, other f -divergences (e.g., BID15 suffer different types of instabilities that can make training difficult to impossible.

Some of the above observations motivate the use of a "weaker" metric that use weaker topologies than that of f -divergences, notably the earth-movers distance or Wasserstein distance .

Besides being a useful information metric for computing the distance between two distributions, the Wasserstein distance has the nice property of being differentiable nearly everywhere, and thus provides a better convergence properties on the generator.

The Kantorovich-Rubinstein dual problem of estimating the Wasserstein distance involves the GAN objective: DISPLAYFORM0 where D is constrained to be in the set of Lipschitz-1 functions 2 .

This GAN objective is a lowerbound for the Wasserstein, as we saw with f -GANs and the f -divergences, and adversarial models that follow this metric are called Wasserstein GANs ( WGAN Arjovsky et al., 2017) .

The Lipchitz constraint should ensure a smoothness or continuity or the gradients being given to the generator.

In principle then, under this constraint, one can train the discriminator to optimality without having to worry about any catastrophic effects on the generator.

The difficulty, however, is in constraining D to be Lipschitz, and it is conventional to use some sort of regularization BID8 .

However, questions remain regarding said regularization and whether the supremum of the resulting family of functions is close enough to the true Wasserstein distance.

In addition, there is strong evidence that regularization on f -GANs can improve training stability BID21 , indicating that having a looser bound over a constrained family of discriminator functions may in principle be a better strategy for training a GAN.

Taking the perspective provided in the previous section, our goal becomes to train a discriminator to near-optimality in the hopes that this will improve learning in the generator.

A well-known scenario is one in which the discriminator can perfectly discern samples from P from samples from Q ?? .

Moreover, show that the discriminator is likely to be constant in the regime of Q ?? , thus emitting zero gradients and no learning signal for the generator.

Consider such a discriminator; if x, y ??? Q ?? are samples, then as the generator attempts to minimize any discrepancy between P and Q ?? , the slope of the tangent hyperplane connecting (x, ??(x)) and (y, ??(y)) likely obeys DISPLAYFORM0 where d X (??, ??) is a metric defined on X .

This roughly corresponds to a flat discriminator over Q ?? which doesn't emit any gradients and impedes the generator from learning.

As a result, we find DISPLAYFORM1 Alternatively, a flat discriminator in the regime of Q ?? can be interpreted as having minimal variance (see Figure 1) .

That is, when the discriminator is mostly constant over P and Q ?? , the discriminator's unnormalized output distribution is peaked and tends to mimic a bi-modal Dirac distribution.

This is because, in a sense, the discriminator acts as a bi-modal mapping: real examples from P are mapped to, for example, 1 while generated examples from Q ?? (x) are mapped to 0.

In contrast, when the discriminator is non-constant over the data/generated manifolds and exhibits variance in its output over within P and/or Q ?? , its unnormalized output distribution is spread out and not peaked.

Following these observations, we propose a regularization technique targetting the discriminator via meta-adversarial learning.

We hypothesize that constraining the discriminator's output distribution Figure 1 : The red line represents the discriminator's output, the blue and green lines represent the real and fake distributions, respectively.

Top: When the discriminator is near-perfect, it's unnormalized output distribution is peaked.

Bottom: Variance in discriminator output within the data and/or generated distribution will result in variance in its unnormalized output distribution.

Note that the discriminator's output need not be in the range (0, 1), we provide just an example here.to follow a mixture of Gaussians will have the same effect as the use of different metrics and/or regularization as seen in works discussed above.

Let p(y|l = 0) and p(y|l = 1) be two univariate Gaussian distributions with unit variance and means ?? f and ?? r , respectively, and where the indicator variable as values, l = 0 that corresponds to "fake" or generated and l = 1 that corresponds to real.

Define the mixture: DISPLAYFORM0 Depending on the location of the means, this Gaussian classifier has an associated classification error that is directly related to their overlap.

In principle, the classifier defines a bound on a difference or distance between two distributions defined by the values of the indicator variable, l = 0 and l = 1.

As the discriminator is really a classifier that in principle is trying to minimize the Bayes risk, under the above constraint, the bound is maximized when the means, ?? f and ?? r , are as far apart as possible.

This is obviously undesirable as this can lead to the same problems mentioned in the previous sections.

Here we will take a different strategy: keep the means and variance fixed so that the associated classifier has a fixed error rate.

This will enforce a sort of fixed "weakness" on the discriminator, which corresponds to a fixed "looseness" on the lower bound defined by the classifier.

We ensure the discriminator's output follows p ?? (y) through adding two additional discriminators which are trained in a similar was as the original GAN formulation: DISPLAYFORM1 DISPLAYFORM2 where F and R are the are the fake and real output discriminators, respectively, each parametrized by a two-layer MLP.

F's goal is to discern a batch of real values {

y i } i drawn from a Gaussian with mean ?? f and unit variance from a batch taken from the discriminator's output over fake samples.

Likewise, R follows a similar protocol for real samples.

It is important to note that comparisons are performed batch-wise rather than per sample.

In essence, F and R are meta-discriminators which instill a regularizing effect on the discriminator.

Meanwhile, the original GAN formulation is only slightly perturbed, as the generator tries to match Q ?? to P while the discriminator attempts to differentiate real and fake samples.

The discriminator's loss objective combines equations 8 and 9: DISPLAYFORM3 The generator's loss function is: DISPLAYFORM4 standard of IPMs.

In practice, however, we find that driving the discriminator's output towards ?? r using a least squares loss also works well: DISPLAYFORM5 Algorithm 1 Variance Regularizing Adversarial Learning (VRAL) using meta-discriminators.

Default parameter values are ?? = 10 ???4 , ?? 1 = 0.5, ?? 2 = 0.999, = 3 ?? 10 ???6 , m = 64.

n batches is determined by the chosen dataset.

Input: n F ,R,?? , the number of updates to perform on F, R and ?? per update to G ?? .1: while ?? G not converged do

for i = 1 . . .

n batches do 3:for k = 1 . . .

n F ,R,?? do 4: DISPLAYFORM0 DISPLAYFORM1 10: DISPLAYFORM2 12: DISPLAYFORM3 13: DISPLAYFORM4 The training procedure for our proposed GAN with regularizing networks is similar to standard GAN training, however requiring each of the meta-discriminators to be updated prior to the discriminator update.

In practice, we update F, R and D the same n F ,R,D number of times per generator update.

The full training procedure is presented in algorithm 1.

at values ?? r and ?? f , respectively.

However, this can become problematic during optimization as the discriminator's goal is to maximize the difference between these two means.

To accommodate, we subtract the mean of the discriminator's output over a batch of fake samples.

This useful insight results in the modified objective in which the discriminator's output over fake examples is no longer fixed to a distribution centered at ?? f , but can move around and preserves unit variance.

Consequently, each occurrence of {??(x (i) )} i where x ??? Q ?? in equation 10 is replaced with {??( DISPLAYFORM0 .

As we will show in subsequent sections, this modification proves extremely helpful.

In future sections, we show our method which uses meta-discriminators to regularize the discriminator's output distribution to be successful.

However, this same technique can be applied to encourage GANs exhibit other desirable properties.

To this end, we demonstrate how we can enforce the discriminator to approximate a K-Lipschitz function.

Consider a similar setup to Variance Regularizing GANs in which only one meta discriminator M is used, and its objective is DISPLAYFORM0 where ?? * = K and ?? * can vary, but is typically close to 0 (e.g. 0.1).

The discriminator's objective is almost identical to equation 10, but its goal is to now maximize the second term in the above equation.

In essence, the single meta-discriminator M is distinguishing samples drawn from a Gaussian distribution fixed at ?? * versus the slope of the hyperplane between x and y, where x and y are real and generated examples, respectively.

Linear Discriminant Analysis (LDA) is a commonly used technique for dimensionality reduction and/or binary classification.

Consider two sets of classes, P and Q ?? , and a set of observations {??(x)}, where each x ??? X (i.e., ??(x) may be interpreted as the result of dimensionality reduction on x).

The objective is to preserve the discrepancy between classes while maintaining intra-class variance.

Let p ?? (??(x); x ??? P) and p ?? (??(x); x ??? Q ?? ) be the probability densities of each class, assumed to follow a Gaussian distribution with means ?? r and ?? f and variances ?? 2 r and ?? 2 f , respectively.

For the sake of simplicity, assume ?? 2 r = ?? 2 f = 1.

The LDA classification rule is then to predict ??(x) belonging to the class of samples from Q ?? if DISPLAYFORM0 otherwise P, and T is a chosen threshold.

Notice that when ??(x) is near ?? r , the LHS of equation 14 becomes small (assuming ??(x) is far from ?? f ) and vice-versa when ??(x) is close to ?? f .McGan employs mean and covariance feature matching between P and Q ?? .

To achieve this, a function ?? : X ??? R m , parametrized as a deep neural network, which maps x ??? X to an embedding space is learned.

The objective is to minimize the mean and covariance between ??(P) and ??(Q ?? ).

More recently, Fisher GANs constrain the second order raw moments of the discriminator such that DISPLAYFORM1 )]) = 1 using Lagrange multipliers.

This approach slightly differs from Variance-Regularizing GANs, which enforce that variance, the second order central moments of the discriminator, are equal to 1.

Nonetheless, Fisher GANs are similar to Variance Regularizing GANs as they aim to stabilize variance in the discriminator's output, although not explicitly defining a distribution which it should follow.

We run experiments on the CIFAR-10 ( BID11 and CelebA BID14 datasets for image generation.

We use a DCGAN BID20 model in all experiments and perform all parameter updates using the Adam optimizer BID9 .

For our proposed method, the meta-discriminators F and R are MLPs with 28 hidden units with a sigmoid output.

Having established theoretical motivations for a well-trained discriminator, we show how many GAN algorithms fail to preserve variance in the discriminator output when trained at large ratios, such as 50 discriminator updates per generator update.

We train the standard GAN, standard GAN using proxy loss (i.e., ??? log D loss), Least Squares GAN BID15 , Wasserstein GAN and our proposed method (both with and without subtracting means) on CIFAR-10 with a varying number of discriminator updates per generator update.

We find that variance in the discriminator's unnormalized output distribution directly correlates with the generator's ability to learn if the discriminator is well-trained.

Figure 2 shows samples from updating the discriminator 50 times per generator update (which we refer to as 50:1 training), while figure 3 demonstrates the discriminator's expected real-valued output over the training set.

Interestingly, the discriminators in standard and Least Squares GANs exhibit almost no output variance within P or Q ?? , suggesting the discriminator is constant over these regimes in image space and would emit zero gradients w.r.t.

any generated sample, hence hindering learning in the generator.

Notice that these methods fail to learn at a 50:1 ratio Furthermore, the generated samples also posit that some amount of variance is necessary; GANs using ??? log D and Wasserstein losses are able a) 1:1 training b) 10:1 training c) 25:1 training d) 50:1 training Figure 6 : Generated images on CelebA by forcing the discriminator to be 1-Lipschitz using a single meta-discriminator.to generate sharp samples while the discriminator in these instances is imperfect and exhibits some level of uncertainty amongst the generated samples.

Our hypothesis of (near) zero gradients is evidently supported as we contrast the discriminator's gradient norms for any particular GAN at varying update ratios.

Figure 4 indicates that as the discriminator achieves perfection by being updated more frequently than the generator, ||??? x???Q ?? L ?? || approaches zero.

However, the standard GAN with ??? log D loss and Wasserstein GAN are much more robust to large update ratios.

The discriminators of these methods don't map all samples, real or fake, to a single value, hence providing a meaningful learning signal for the generator and avoiding the vanishing gradient scenario.

Our proposed method follows this line of reasoning, as the meta-discriminators prevent the discriminator from achieving perfection, regardless of update ratio.

Adhering to our earlier discussion of using a single meta-discriminator to force the discriminator to approximate a K-Lipschitz function, we trained a meta-discriminator M to perform batch-wise discrimination between samples drawn from N (y; ?? * = 1, ?? * = 0.1) and |??(x)?????(y)| d X (x,y) where x ??? P, y ??? Q ?? .

See figure 6 for results.

More work is needed to robustly train GANs using meta-adversarial learning.

In this paper, we have demonstrated the importance of intra-class variance in the discriminator's output.

In particular, our results show that methods whose discriminators tend to map inputs of a class to single real values are unable to provide a reliable learning signal for the generator.

Furthermore, variance in the discriminator's output is essential to allow the generator to learn in the presence of a well-trained discriminator.

We proposed a technique, conceptually in line with LDA, which ensures the discriminator's output distribution follows a specified prior.

Taking a broader perspective, we also introduced a new regularization technique called meta-adversarial learning, which can be applied to ensure enforce various desirable properties in GANs.

<|TLDR|>

@highlight

We introduce meta-adversarial learning, a new technique to regularize GANs, and propose a training method by explicitly controlling the discriminator's output distribution.

@highlight

The paper proposes variance regularizing adversarial learning for training GANs to ensure that the gradient for the generator does not vanish