Generative adversarial networks (GANs) evolved into one of the most successful unsupervised techniques for generating realistic images.

Even though it has recently been shown that GAN training converges, GAN models often end up in local Nash equilibria that are associated with mode collapse or otherwise fail to model the target distribution.

We introduce Coulomb GANs, which pose the GAN learning problem as a potential field, where generated samples are attracted to training set samples but repel each other.

The discriminator learns a potential field while the generator decreases the energy by moving its samples along the vector (force) field determined by the gradient of the potential field.

Through decreasing the energy, the GAN model learns to generate samples according to the whole target distribution and does not only cover some of its modes.

We prove that Coulomb GANs possess only one Nash equilibrium which is optimal in the sense that the model distribution equals the target distribution.

We show the efficacy of Coulomb GANs on LSUN bedrooms, CelebA faces, CIFAR-10 and the Google Billion Word text generation.

Generative adversarial networks (GANs) excel at constructing realistic images BID28 BID24 BID3 and text BID18 .

In GAN learning, a discriminator network guides the learning of another, generative network.

This procedure can be considered as a game between the generator which constructs synthetic data and the discriminator which separates synthetic data from training set data BID16 .

The generator's goal is to construct data which the discriminator cannot tell apart from training set data.

GAN convergence points are local Nash equilibria.

At these local Nash equilibria neither the discriminator nor the generator can locally improve its objective.

Despite their recent successes, GANs have several problems.

First (I), until recently it was not clear if in general gradient-based GAN learning could converge to one of the local Nash equilibria BID38 BID15 .

It is even possible to construct counterexamples BID16 .

Second (II), GANs suffer from "mode collapsing", where the model generates samples only in certain regions which are called modes.

While these modes contain realistic samples, the variety is low and only a few prototypes are generated.

Mode collapsing is less likely if the generator is trained with batch normalization, since the network is bound to create a certain variance among its generated samples within one batch .

However batch normalization introduces fluctuations of normalizing constants which can be harmful BID16 .

To avoid mode collapsing without batch normalization, several methods have been proposed BID5 BID38 .

Third (III), GANs cannot assure that the density of training samples is correctly modeled by the generator.

The discriminator only tells the generator whether a region is more likely to contain samples from the training set or synthetic samples.

Therefore the discriminator can only distinguish the support of the model distribution from the support of the target distribution.

Beyond matching the support of distributions, GANs with proper objectives may learn to locally align model and target densities via averaging over many training examples.

On a global scale, however, GANs fail to equalize model and target densities.

The discriminator does not inform the generator globally where probability mass is missing.

Consequently, standard GANs are not assured to capture the global sample density and are prone to neglect large parts of the target distribution.

The next paragraph gives an example of this.

Fourth (IV), the discriminator of GANs may forget previous modeling errors of the generator which then may reappear, a property that leads to oscillatory behavior instead of convergence BID16 .Recently, problem (I) was solved by proving that GAN learning does indeed converge when discriminator and generator are learned using a two time-scale learning rule BID20 .

Convergence means that the expected SGD-gradient of both the discriminator objective and the generator objective are zero.

Thus, neither the generator nor the discriminator can locally improve, i.e., learning has reached a local Nash equilibrium.

However, convergence alone does not guarantee good generative performance.

It is possible to converge to sub-optimal solutions which are local Nash equilibria.

Mode collapse is a special case of a local Nash equilibrium associated with suboptimal generative performance.

For example, assume a two mode real world distribution where one mode contains too few and the other mode too many generator samples.

If no real world samples are between these two distinct modes, then the discriminator penalizes to move generated samples outside the modes.

Therefore the generated samples cannot be correctly distributed over the modes.

Thus, standard GANs cannot capture the global sample density such that the resulting generators are prone to neglect large parts of the real world distribution.

A more detailed example is listed in the Appendix in Section A.1.In this paper, we introduce a novel GAN model, the Coulomb GAN, which has only one Nash equilibrium.

We are later going to show that this Nash equilibrium is optimal, i.e., the model distribution matches the target distribution.

We propose Coulomb GANs to avoid the GAN shortcoming (II) to (IV) by using a potential field created by point charges analogously to the electric field in physics.

The next section will introduce the idea of learning in a potential field and prove that its only solution is optimal.

We will then show how learning the discriminator and generator works in a Coulomb GAN and discuss the assumptions needed for our optimality proof.

In Section 3 we will then see that the Coulomb GAN does indeed work well in practice and that the samples it produces have very large variability and appear to capture the original distribution very well.

Related Work.

Several GAN approaches have been suggested for bringing the target and model distributions in alignment using not just local discriminator information: Geometric GANs combine samples via a linear support vector machine which uses the discriminator outputs as samples, therefore they are much more robust to mode collapsing BID31 .

Energy-Based GANs BID41 and their later improvement BEGANs BID3 ) optimize an energy landscape based on auto-encoders.

McGANs match mean and covariance of synthetic and target data, therefore are more suited than standard GANs to approximate the target distribution BID34 .

In a similar fashion, Generative Moment Matching Networks BID30 and MMD nets BID12 directly optimize a generator network to match a training distribution by using a loss function based on the maximum mean discrepancy (MMD) criterion BID17 .

These approaches were later expanded to include an MMD criterion with learnable kernels and discriminators .

The MMD criterion that these later approaches optimize has a form similar to the energy function that Coulomb GANs optimize (cf.

Eq. (33)).

However, all MMD approaches end up using either Gaussian or Laplace kernels, which are not guaranteed to find the optimal solution where the model distribution matches the target distribution.

In contrast, the Plummer kernel which is employed in this work has been shown to lead to the optimal solution BID22 .

We show that even a simplified version of the Plummer kernel, the low-dimensional Plummer kernel, ensures that gradient descent convergences to the optimal solution as stated by Theorem 1.

Furthermore, most MMD GAN approaches use the MMD directly as loss function though the number of possible samples in a mini-batch is limited.

Therefore MMD approaches face a sampling problem in high-dimensional spaces.

The Coulomb GAN instead learns a discriminator network that gradually improves its approximation of the potential field via learning Figure 1 : The vector field of a Coulomb GAN.

The basic idea behind the Coulomb GAN: true samples (blue) and generated samples (red) create a potential field (scalar field).

Blue samples act as sinks that attract the red samples, which repel each other.

The superimposed vector field shows the forces acting on the generator samples to equalize potential differences, and the background color shows the potential at each position.

Best viewed in color.on many mini-batches.

The discriminator network also tracks the slowly changing generator distribution during learning.

Most importantly however, our approach is, to the best of our knowledge, the first one for which optimality, i.e., ability to perfectly learn a target distribution, can be proved.

The use of the Coulomb potential for learning is not new.

Coulomb Potential Learning was proposed to store arbitrary many patterns in a potential field with perfect recall and without spurious patterns BID35 .

Another related work is the Potential Support Vector Machine (PSVM), which minimizes Coulomb potential differences BID21 BID23 .

BID22 also used a potential function based on Plummer kernels for optimal unsupervised learning, on which we base our work on Coulomb GANs.

We assume data samples a ??? R m for a model density p x (.)

and a target density p y (.).

The goal of GAN learning is to modify the model in a way to obtain p x (.) = p y (.).

We define the difference of densities ??(a) = p y (a) ??? p x (a) which should be pushed toward zero for all a ??? R m during learning.

In the GAN setting, the discriminator D(a) is a function D : R m ??? R that learns to discriminate between generated and target samples and predicts how likely it is that a is sampled from the target distribution.

In conventional GANs, D(a) is usually optimized to approximate the probability of seeing a target sample, or ??(a) or some similar function.

The generator G(z) is a continuous function G : R n ??? R m which maps some n-dimensional random variable z into the space of target samples.

z is typically sampled from a multivariate Gaussian or Uniform distribution.

In order to improve the generator, a GAN uses the gradient of the discriminator ??? a D(a) with respect to the discriminator input a = G(z) for learning.

The objective of the generator is a scalar function D(G(z)), therefore the gradient of the objective function is just a scaled version of the gradient ??? a D(a) which would then propagate further to the parameters of G. This gradient ??? a D(a) tells the generator in which direction ??(a) becomes larger, i.e., in which direction the ratio of target examples increases.

The generator changes slightly so that z is now mapped to a new a = G (z), moving the sample generated by z a little bit towards the direction where ??(a) was larger, i.e., where target examples were more likely.

However, ??(a) and its derivative only take into account the local neighborhood of a, since regions of the sample space that are distant from a do not have much influence on ??(a).

Regions of data space that have strong support in p y but not in p x will not be noticed by the generator via discriminator gradients.

The restriction to local environments hampers GAN learning significantly .The theoretical analysis of GAN learning can be done at three different levels: (1) in the space of distributions p x and p y regardless of the fact that p x is realized by G and p z , (2) in the space of functions G and D regardless of the fact that G and D are typically realized by a parametric form, i.e., as neural networks, or (3) in the space of the parameters of G and D. use (1) to prove convergence of GAN learning in their Proposition 2 in a hypothetical scenario where the learning algorithm operates by making small, local moves in p x space.

In order to see that level (1) and (2) should both be understood as hypothetical scenarios, remember that in all practical implementations, p x can only be altered implicitly by making small changes to the generator function G, which in turn can only be changed implicitly by small steps in its parameters.

Even if we assume that the mapping from a distribution p x to the generator G that induced it exists and is unique, this mapping from p x to the space of G is not continuous.

To see this, consider changing a distribution p1 x to a new distribution p2 x by moving a small amount of its density to an isolated region in space where p1 x has no support.

Let's further assume this region has distance d to any other regions of support of p1 x .

By letting ??? 0, the distance between p1 x and p2 x becomes smaller, yet the distance between the inducing generator functions G 1 and G 2 (e.g. using the supremum norm on bounded functions) will not tend to zero because for at least one function input z we have: DISPLAYFORM0 Because of this, we need to go further than the distribution space when analyzing GAN learning.

In practice, when learning GANs, we are restricted to small steps in parameter space, which in turn lead to small steps in function space and finally to small steps in distribution space.

But not all small steps in distribution space can be realized this way as shown in the example above.

This causes local Nash equilibria in the function space, because even though in distribution space it would be easy to escape by making small steps, such a step would require very large changes in function space and is thus not realizable.

In this paper we show that Coulomb GANs do not exhibit any local Nash equilibria in the space of the functions G and D. To the best of our knowledge, this is the first formulation of GAN learning that can guarantee this property.

Of course, Coulomb GANs are learned as parametrized neural networks, and as we will discuss in Subsection 2.4.2, Coulomb GANs are not immune to the usual issues that arise from parameter learning, such as over-and underfitting, which can cause local Nash Equilibria due to a bad choice of parameters.

If the density p x (.)

or p y (.) approaches a Dirac delta-distribution, gradients vanish since the density approaches zero except for the exact location of data points.

Similarly, electric point charges are often represented by Dirac delta-distributions, however the electric potential created by a point charge has influence everywhere in the space, not just locally.

The electric potential (Coulomb potential) created by the point charge Q is ?? C = 1 4????0 Q r , where r is the distance to the location of Q and ?? 0 is the dielectric constant.

Motivated by this electric potential, we introduce a similar concept for GAN learning: Instead of the difference of densities ??(a), we rather consider a potential function ??(a) defined as DISPLAYFORM0 with some kernel k (a, b) which defines the influence of a point at b onto a point at a. The crucial advantage of potentials ??(a) is that each point can influence each other point in space if k is chosen properly.

If we minimize this potential ??(a)

we are at the same time minimizing the difference of densities ??(a): For all kernels k it holds that if ??(b) = 0 for all b then ??(a) = 0 for all a. We must still show that (i) ??(a) = 0 for all a then ??(b) = 0 for all b, and even more importantly, (ii) whether a gradient optimization of ??(a) leads to ??(a) = 0 for all a.

This is not the case for every kernel.

Indeed only for particular kernels k gradient optimization of ??(a) leads to BID22 ) (see also Theorem 1 below).

DISPLAYFORM1 An example for such a kernel k is the one leading to the Coulomb potential ?? C from above, where DISPLAYFORM2 As we will see in the following, the ability to have samples that influence each other over long distances, like charges in a Coulomb potential, will lead to GANs with a single, optimal Nash equilibrium.

For Coulomb GANs, the generator objective is derived from electrical field dynamics: real and generated samples generate a potential field, where samples of the same class (real vs. generated) repel each other, but attract samples of the opposite class.

However, real data points are fixed in space, so the only samples that can move are the generated ones.

In turn, the gradient of the potential with respect to the input samples creates a vector field in the space of samples.

The generator can move its samples along the forces generated by this field.

Such a field is depicted in Figure 1 .

The discriminator learns to predict the potential function, in order to approximate the current potential landscape of all samples, not just the ones in the current mini-batch.

Meanwhile, the generator learns to distribute its samples across the whole field in such a way that the energy is minimized, thus naturally avoids mode collapse and covering the whole region of support of the data.

The energy is minimal and equal to zero only if all potential differences are zero and the model distribution is equal to the target distribution.

Within an electrostatic field, the strength of the force on one particle depends on its distance to other particles and their charges.

If left to move freely, the particles will organize themselves into a constellation where all forces equal out and no potential differences are present.

For continuous charge distributions, the potential field is constant without potential differences if charges no longer move since forces are equaled out.

If the potential field is constant, then the difference of densities ?? is constant, too.

Otherwise the potential field would have local bumps.

The same behavior is modeled within our Coulomb GAN, except that real and generated samples replace the positive and negative particles, respectively, and that the real data points remain fixed.

Only the generated samples are allowed to move freely, in order to minimize ??.

The generated samples are attracted by real samples, so they move towards them.

At the same time, generated samples should repel each other, so they do not clump together, which would lead to mode collapsing.

Analogously to electrostatics, the potential ??(a) from Eq. (1) gives rise to a field E(a) = ?????? a ??(a).

and to an energy function DISPLAYFORM0 The field E(a) applies a force on charges at a which pushes the charges toward lower energy constellations.

Ultimately, the Coulomb GAN aims to make the potential ?? zero everywhere via the field E(a), which is the negative gradient of ??. For proper kernels k, it can be shown that (i) ?? can be pushed to zero via its negative gradient given by the field and (ii) that ??(a) = 0 for all a implies ??(a) = 0 for all a, therefore, p x (a) = p y (a) for all a BID22 ) (see also Theorem 1 below).

During learning we do not change ?? or ?? directly.

Instead, the location a = G(z) to which the random variable z is mapped changes to a new location a = G (z).

For the GAN optimization dynamics, we assume that generator samples a = G(z) can move freely, which is ensured by a sufficiently complex generator.

Importantly, generator samples originating from random variables z do neither disappear nor are they newly created but are conserved.

This conservation is expressed by the continuity equation BID39 ) that describes how the difference between distributions ??(a) changes as the particles are moving along the field, i.e., how moving samples during the learning process changes our densities:?? DISPLAYFORM0 for sample density difference ?? and unit charges that move with "velocity" v(a) = sign(??(a))E(a).

The continuity equation is crucial as it establishes the connection between moving samples and changing the generator density and thereby ??.

The sign function of the velocity indicates whether positive or negative charges are present at a. The divergence operator "?????" determines whether samples move toward or outward of a for a given field.

Basically, the continuity equation says that if the generator density increases, then generator samples must flow into the region and if the generator density decreases, they flow outwards.

We assume that differently charged particles cancel each other.

If generator samples are moved away from a location a then ??(a) is increasing while ??(a) is decreasing when generator samples are moved toward a. The continuity equation is also obtained as a first order ODE to move particles in a potential field BID11 , therefore describes the dynamics how the densities are changing.

We obtai??? DISPLAYFORM1 Published as a conference paper at ICLR 2018The density difference ??(a) indicates how many samples are locally available for being moved.

At each local minimum and local maximum a of ?? we obtain ??? a ??(a) = 0.

Using the product rule for the divergence operator, at points a that are minima or maxima, Eq. (3) reduces t?? DISPLAYFORM2 In order to ensure that ?? converges to zero, it is necessary and sufficient that sign(??? ?? E(a)) = sign(??(a)), where ??a??(a) = 0, as this condition ensures the uniform decrease of the maximal absolute density differences |??(a max )|.

As discussed before, the choice of kernel is crucial for Coulomb GANs.

The m-dimensional Coulomb kernel and the m-dimensional Plummer kernel lead to (i) ?? that is pushed to zero via the field it creates and (ii) that ??(a) = 0 for all a implies ??(a) = 0 for all a, therefore, p x (a) = p y (a) for all a BID22 .

Thus, gradient learning with these kernels has been proved to converge to an optimal solution.

However, both the m-dimensional Coulomb and the mdimensional Plummer kernel lead to numerical instabilities if m is large.

Therefore the Coulomb potential ??(a) for the Coulomb GAN was constructed by a low-dimensional Plummer kernel k with parameters d m ??? 2 and : DISPLAYFORM0 The original Plummer kernel is obtained with d = m ??? 2.

The resulting field and potential energy is DISPLAYFORM1 DISPLAYFORM2 The next theorem states that for freely moving generated samples, ?? converges to zero, that is, p x (.) = p y (.), when using this potential function ??(a).

Theorem 1 (Convergence with low-dimensional Plummer kernel).

For a, b ??? R m , d m ??? 2, and > 0 the densities p x (.) and p y (.) equalize over time when minimizing energy F with the low-dimensional Plummer kernel by gradient descent.

The convergence is faster for larger d.

Proof.

See Section A.2.

The Coulomb GAN minimizes the electric potential energy from Eq. (6) using a stochastic gradient descent based approach using mini-batches.

Appendix Section A.4 contains the equations for the Coulomb potential, field, and energy in this case.

Generator samples are obtained by drawing N x random numbers z i and transforming them into outputs x i = G(z i ).

Each mini-batch also includes N y real world samples y i .

This gives rise to a mini-batch specific potential, where in Eq. FORMULA8 we use ??(a) = p y (a) ??? p x (a) and replace the expectations by empirical means using the drawn samples: DISPLAYFORM0 It is tempting to have a generator network that directly minimizes this potential?? between generated and training set points.

In fact, we show that?? is an unbiased estimate for ?? in Appendix Section A.4.

However, the estimate has very high variance: for example, if a mini-batch fails to sample training data from an existing mode, the field would drive all generated samples that have been generated at this mode to move elsewhere.

The high variance has to be counteracted by extremely low learning rates, which makes learning infeasible in practice, as confirmed by initial experiments.

Our solution to this problem is to have a network that generalizes over the mini-batch specific potentials: each mini-batch contains different generator samples X = x i for i = 1, . . .

, N x and real world samples Y = y i for i = 1, . . .

, N y , they create a batch-specific potential??. The goal of the discriminator is to learn E X ,Y (??(a)) = ??(a), i.e., the potential averaged over many mini-batches.

Thus the discriminator function D fulfills a similar role as other typical GAN discriminator functions, i.e., it discriminates between real and generated data such that for any point in space a, D(a) should be greater than zero if the p y (a) > p x (a) and smaller than zero otherwise.

In particular D(a) also indicates, via its gradient and its potential properties, directions toward regions where training set samples are predominant and where generator samples are predominant.

The generator in turn tries to move all of its samples according to the vector field into areas where generator samples are missing and training set samples are predominant.

The generator minimizes the approximated energy F as predicted by the discriminator.

The loss L D for the discriminator and L G for the generator are given by: DISPLAYFORM1 Where p(a) = 1/2 N (a; G(z), I)p z (z)dz + 1/2 N (a; y, I)p y (y)dy, i.e., a distribution where each point of support both of the generator and the real world distribution is surrounded with a Gaussian ball of width I similar to BID4 , in order to overcome the problem that the generator distribution is only a sub-manifold of R m .

These loss functions cause the approximated potential values D(a) that are negative are pushed toward zero.

Finally, the Coulomb GAN, like all other GANs, consists of two parts: a generator to generate model samples, and a discriminator that provides its learning signal.

Without a discriminator, our would be very similar to GMMNs BID30 , as can be seen in Eq. (33), but with an optimal Kernel specifically tailored to the problem of estimating differences between probability distributions.

We use each mini-batch only for one update of the discriminator and the generator.

It is important to note that the discriminator uses each sample in the mini batch twice: once as a point to generate the mini-batch specific potential??, and once as a point in space for the evaluation of the potential?? and its approximation D. Using each sample twice is done for performance reasons, but not strictly necessary: the discriminator could learn the potential field by sampling points that lie between generator and real samples as in BID18 , but we are mainly interested in correct predictions in the vicinity of generator samples.

Pseudocode for the learning algorithm is detailed in Algorithm 1 in the appendix.

Convergence of the GAN learning process was proved for a two time-scales update rule by BID20 .

A local Nash equilibrium is a pair of generator and discriminator (D * , G * ) that fulfills the two conditions DISPLAYFORM0 for some neighborhoods U (D * ) and U (G * ).

We show in the following Theorem 2 that for Coulomb GANs every local Nash equilibrium necessarily is identical to the unique global Nash equilibrium.

In other words, any equilibrium point of the Coulomb GAN that is found to be local optimal has to be the one global Nash equilibrium as the minimization of the energy F (??) in Eq. (33) leads to a single, global optimum at p y = p x .

Theorem 2 (Optimal Solution).

If the pair (D * , G * ) is a local Nash equilibrium for the Coulomb GAN objectives, then it is the global Nash equilibrium, and no other local Nash equilibria exist, and G * has output distribution p x = p y .Proof.

See Appendix Section A.3.

To implement GANs in practice, we need learnable models for G and D.

We assume that our models for G and D are continuously differentiable with respect to their parameters and inputs.

Toward this end, GANs are typically implemented as neural networks optimized by (some variant of) gradient descent.

Thus we may not find the optimal G * or D * , since neural networks may suffer from capacity or optimization issues.

Recent research indicates that the effect of local minima in deep learning vanishes with increasing depth BID10 BID8 BID25 , such that this limitation becomes less restrictive as our ability to train deep networks grows thanks to hardware and optimization improvements.

The main problem with learning Coulomb GANs is to approximate the potential function ??, which is a complex function in a high-dimensional space, since the potential can be very non-linear and non-smooth.

When learning the discriminator, we must ensure that enough data is sampled and averaged over.

We already lessened the non-linear function problem by using a low-dimensional Plummer kernel.

But still, this kernel can introduce large non-linearities if samples are close to each other.

It is crucial that the discriminator learns slow enough to accurately estimate the potential function which is induced by the current generator.

The generator, in turn, must be even slower since it must be tracked by the discriminator.

These approximation problems are supposed to be tackled by the research community in near future, which would enable optimal GAN learning.

The formulation of GAN learning as a potential field naturally solves the mode collapsing issue: the example described in Section A.1, where a normal GAN cannot get out of a local Nash equilibria is not a converged solution for the Coulomb GAN: If all probability mass of the generator lies in one of the modes, then both attracting forces from real-world samples located at the other mode as well as repelling forces from the over-represented generator mode will act upon the generator until it generates samples at the other mode as well.

In all of our experiments, we used a low-dimensional Plummer Kernel of dimensionality d = 3.

This kernel both gave best computational performance and has low risk of running into numerical issues.

We used a batch size of 128.

To evaluate the quality of a GAN, the FID metric as proposed by BID20 was calculated by using 50k samples drawn from the generator, while the training set statistics were calculated using the whole training set.

We compare to BEGAN BID3 , DCGAN and WGAN-GP BID18 both in their original version as well as when using the two-timescale update-rule (TTUR), using the settings from BID20 .

We additionally compare to MMD-GAN , which is conceptually very similar to the Coulomb GAN, but uses a Gaussian Kernel instead of the Plummer Kernel.

We use the dataset-specific settings recommended in and report the highest FID score over the course of training.

All images shown in this paper were produced with a random seed and not cherry picked.

The implementation used for these experiments is available online 1 .

The appendix Section A.5 contains an additional toy example demonstrating that Coulomb GANs do not suffer from mode collapse when fitting a simple Gaussian Mixture of 25 components.

To demonstrate the ability of the Coulomb GAN to learn distributions in high dimensional spaces, we trained a Coulomb GAN on several popular image data sets: The cropped and centered images of celebrities from the Large-scale CelebFaces Attributes ("CelebA") data set BID32 , the LSUN bedrooms data set consists of over 3 million 64x64 pixel images of the bedrooms category of the large scale image database LSUN BID40 as well as the CIFAR-10 data set.

For these experiments, we used the DCGAN architecture ) with a few modifications: our convolutional kernels all have a kernel size of 5x5, our random seed that serves as input to the generator has fewer dimensions: 32 for CelebA and LSUN bedrooms, and 16 for CIFAR-10.

Furthermore, the discriminator uses twice as many feature channels in each layer as in the DCGAN architecture.

For the Plummer kernel, was set to 1.

We used the Adam optimizer with a learning rate of 10 ???4 for the generator and 5 ?? 10 ???5 for the discriminator.

To improve convergence performance, we used the tanh output activation function BID27 .

For regularization we used an L2 weight decay term with a weighting factor of 10 ???7 .

Learning was stopped by monitoring the FID metric BID20 .

Once learning plateaus, we scaled the learning rate down by a factor of 10 and let it continue once more until the FID plateaus.

The results are reported in Table 1b , and generated images can be seen in Figure 2 and in the Appendix in Section A.7.

Coulomb GANs tend to outperform standard GAN approaches like BEGAN and DCGAN, but are outperformed by the Improved Wasserstein GAN.

However it is important to note that the Improved Wasserstein GAN used a more advanced network architecture based on ResNet blocks BID18 , which we could not replicate due to runtime constraints.

Overall, the low FID of Coulomb GANs stem from the fact that the images show a wide variety of different samples.

E.g. on CelebA, Coulomb GAN exhibit a very wide variety of faces, backgrounds, eye colors and orientations.

To further investigate how The most similar pairs found in batches of 1024 generated faces sampled from the Coulomb GAN, and the nearest neighbor from the training data shown as third image.

Distances were calculated as Euclidean distances on pixel level.

much variation the samples generated by the Coulomb GAN contains, we followed the advice of Arora and Zhang BID2 to estimate the support size of the generator's distribution by checking how large a sample from the generator must be before we start generating duplicates.

We were able to generate duplicates with a probability of around 50 % when using samples of size 1024, which indicates that the support size learned by the Coulomb GAN would be around 1M.

This is a strong indication that the Coulomb GAN was able to spread out its samples over the whole target distribution.

A depiction is included in Figure 3 , which also shows the nearest neighbor in the training set of the generated images, confirming that the Coulomb GAN does not just memorize training images.

We repeated the experiments from BID18 , where Improved Wasserstein GANs (WGAN-GP) were trained to produce text samples after being trained on the Google Billion Word data set BID6 , using the same network architecture as in the original publication.

We use the Jensen-Shannon-divergence on 4-grams and 6-grams as an evaluation criterion.

The results are summarized in

The Coulomb GAN is a generative adversarial network with strong theoretical guarantees.

Our theoretical results show that the Coulomb GAN will be able to approximate the real distribution perfectly if the networks have sufficient capacity and training does not get stuck in local minima.

Our results show that the potential field used by the Coulomb GAN far outperforms MMD based approaches due to its low-dimensional Plummer kernel, which is better suited for modeling probability density functions, and is very effective at eliminating the mode collapse problem in GANs.

This is because our loss function forces the generated samples to occupy different regions of the learned distribution.

In practice, we have found that Coulomb GANs are able to produce a wide range of different samples.

However, in our experience, this sometimes leads to a small number of generated samples that are non-sensical interpolations of existing data modes.

While these are sometimes also present in other GAN models , we found that our model produces such images at a slightly higher rate.

This issue might be solved by finding better ways of learning the discriminator, as learning the correct potential field is crucial for the Coulomb GAN's performance.

We also observed that increasing the capacity of the discriminator seems to always increase the generative performance.

We thus hypothesize that the largest issue in learning Coulomb GANs is that the discriminator needs to approximate the potential field ?? very well in a high-dimensional space.

In summary, instead of directly optimizing a criterion based on local differences of densities which can exhibit many local minima, Coulomb GANs are based on a potential field that has no local minima.

The potential field is created by point charges in an analogy to electric field in physics.

We have proved that if learning converges then it converges to the optimal solution if the samples can be moved freely.

We showed that Coulomb GANs avoid mode collapsing, model the target distribution more truthfully than standard GANs, and do not overlook high probability regions of the target distribution.

A APPENDIX

As an example of how a GAN can converge to a Nash Equilibrium that exhibits mode collapse, consider a target distribution that consists of two distinct/non-overlapping regions of support C 1 and C 2 that are distant from each other, i.e., the target probability is zero outside of C 1 and C 2 .

Further assume that 50 % of the probability mass is in C 1 and 50 % in C 2 .

Assume that the the generator has mode-collapsed onto C 1 , which contains 100 % of the generator's probability mass.

In this situation, the optimal discriminator classifies all points from C 2 as "real" (pertaining to the target distribution) by supplying an output of 1 for them (1 is the target for real samples and 0 the target for generated samples).

Within C 1 , the other region, the discriminator sees twice as many generated data points as real ones, as 100 % of the probability mass of the generator's distribution is in C 1 , but only 50 % of the probability mass of the real data distribution.

So one third of the points seen by the discriminator in C 1 are real, the other 2 thirds are generated.

Thus, to minimize its prediction error for a proper objective (squared or cross entropy), the discriminator has to output 1/3 for every point from C 1 .

The optimal output is even independent of the exact form of the real distribution in C 1 .

The generator will match the shape of the target distribution locally.

If the shape is not matched, local gradients of the discriminator with respect to its input would be present and the generator would improve locally.

If local improvements of the generator are no longer possible, the shape of the target distribution is matched and the discriminator output is locally constant.

In this situation, the expected gradient of the discriminator is the zero vector, because it has reached an optimum.

Since the discriminator output is constant in C 1 (and C 2 ), the generator's expected gradient is the zero vector, too.

The situation is also stable even though we still have random fluctuations from the ongoing stochastic gradient (SGD) learning: whenever the generator produces data outside of (but close to) C 1 , the discriminator can easily detect this and push the generator's samples back.

Inside C 1 , small deviations of the generator from the shape of the real distribution are detected by the discriminator as well, by deviating slightly from 1/3.

Subsequently, the generator is pushed back to the original shape.

If the discriminator deviates from its optimum, it will also be forced back to its optimum.

So overall, the GAN learning reached a local Nash equilibrium and has converged in the sense that the parameters fluctuate around the attractor point (fluctuations depend on learning rate, sample size, etc.).

To achieve true mathematical convergence, BID20 assume decaying learning rates to anneal the random fluctuations, similar to BID37 original convergence proof for SGD.

We first recall Theorem 1:Theorem (Convergence with low-dimensional Plummer kernel).

For a, b ??? R m , d m ??? 2, and > 0 the densities p x (.) and p y (.) equalize over time when minimizing energy F with the lowdimensional Plummer kernel by gradient descent.

The convergence is faster for larger d.

In a first step, we prove that for local maxima or local minima a of ??, the expression sign(??? ?? E(a)) = sign(??(a)) holds for small enough.

For proving this equation, we apply the Laplace operator for spherical coordinates to the low-dimensional Plummer kernel.

Using the result, we see that the integral ??? ?? E(a) = ??? ??(b)??? 2 a k (a, b) db is dominated by large negative values of ??? 2 a k around a. These negative values can even be decreased by decreasing .

Therefore we can ensure by a small enough that at each local minimum and local maximum a of ?? sign(??(a)) = ???sign(??(a)).

Thus, the maximal and minimal points of ?? move toward zero.

In a second step, we show that new maxima or minima cannot appear and that the movement of ?? toward zero stops at zero and not earlier.

Since ?? is continuously differentiable, all points in environments of maxima and minima move toward zero.

Therefore the largest |??(a)| moves toward zero.

We have to ensure that moving toward zero does not converge to a point apart from zero.

We derive that the movement toward zero is lower bounded by??(a) = ???sign(??(a))???? 2 (a).

Thus, the movement slows down at ??(a) = 0.

Solving the differential equation and applying it to the maximum of the absolute value of ?? gives |??| max (t) = 1/(??t + (|??| max (0)) ???1 ).

Thus, ?? converges to zero over time.

DISPLAYFORM0 , where the theorem has already been proved for small enough BID22 .At each local minimum and local maximum a of ?? we have ??? a ??(a) = 0.

Using the product rule for the divergence operator, Eq. (3) reduces t?? DISPLAYFORM1 The term ??? ?? E(a) can be expressed as DISPLAYFORM2 We next consider ??? 2 a k (a, b) for the low-dimensional Plummer kernel.

We define the spherical Laplace operator in (m ??? 1) dimensions as ??? 2 S m???1 , then the Laplace operator in spherical coordinates is (Proposition 2.5 in Frye & Efthimiou BID13 ): DISPLAYFORM3 Note that ??? 2 S m???1 only has second order derivatives with respect to the angles of the spherical coordinates.

With r = a ??? b we obtain for the Laplace operator applied to the low-dimensional Plummer kernel: DISPLAYFORM4 and in particular DISPLAYFORM5 For d m ??? 2 we have (2 + d ??? m) 0, and obtain DISPLAYFORM6 and DISPLAYFORM7 Therefore, ??? 2 k(a, b) is negative with minimum ???md ???(d+2) at r = 0 and increasing with r and increasing with for d m ??? 4.

For d = m ??? 3 we have to restrict in the following the sphere S ?? (a) to ?? < ??? m and ensure increase of ??? 2 k(a, b) with .If ??(b) = 0, then we define a sphere S ?? (a) with radius ?? around a for which holds sign(??(b)) = sign(??(a)) for each b ??? S ?? (a).

Note that ??? 2 k(a, b) is continuous differentiable.

We have DISPLAYFORM8 Using ?? , we now bound T \S?? (a) ??(b) ??? 2 a k (a, b) db independently from , since ?? is a difference of distributions.

For small enough we can ensure DISPLAYFORM9 Therefore we have sign(??? ?? E(a)) = sign(??(a)) .Therefore we have at each local minimum and local maximum a of ?? sign(??(a)) = ??? sign(??(a)) .(23) Therefore the maximal and minimal points of ?? move toward zero.

Since ?? is continuously differentiable as is the field, also the points in an environment of the maximal and minimal points move toward zero.

Points that are not in an environment of the maximal or minimal points cannot become maximal points in an infinitesimal time step.

Since the contribution of a environment S ?? (a) dominates the integral Eq. (19), for small enough there exists a positive 0 < ?? globally for all minima and maxima as well as for all time steps for which holds: DISPLAYFORM10 The factor ?? depends on k and on the initial ??.

?? is proportional to d. Larger d lead to larger |??? ?? E(a)| since the maximum or minimum ??(a) is upweighted.

There might exist initial conditions ?? for which ?? ??? 0, e.g. for infinite many maxima and minima, but they are impossible in our applications.

Therefore maximal or minimal points approach zero faster or equal than given b??? Consequently, ?? converges to the zero function over time, that is, p x (.) becomes equal to p y (.).

DISPLAYFORM11

We first recall Theorem 2: Theorem (Optimal Solution).

If the pair (D * , G * ) is a local Nash equilibrium for the Coulomb GAN objectives, then it is the global Nash equilibrium, and no other local Nash equilibria exist, and G * has output distribution p x = p y .Proof. (D * , G * ) being in a local Nash equilibrium means that (D * , G * ) fulfills the two conditions DISPLAYFORM0 for some neighborhoods U (D * ) and U (G * ).

For Coulomb GANs that means, D * has learned the potential ?? induced by G * perfectly, because L D is convex in D, thus if D * is optimal within an neighborhood U (D * ), it must be the global optimum.

This means that G * is directly minimizing (G(z)) ).

The Coulomb potential energy is according to Eq. (7) DISPLAYFORM1 DISPLAYFORM2 Only the samples from p x stem from the generator, where p x (a) = ??(a ??? G(z))p z (z)dz.

Here ?? is the ??-distribution centered at zero.

The part of the energy which depends on the generator is DISPLAYFORM3 Theorem 1 guarantees that there are no other local minima except the global one when minimizing F .

F has one minimum, F = 0, which implies ??(a) = 0 and ??(a) = 0 for all a, therefore also p y = p x according to Theorem 1.

Each ??(a) = 0 would mean there exist potential differences which in turn would cause forces on generator samples that allow to further minimize the energy.

Since we assumed that the generator can reach the minimum p y = p x for any p y , it will be reached by local (stepwise) optimization of ??? 1 2 E pz (??(G(z))) with respect to G. Since the pair (D * , G * ) is optimal within their neighborhood, the generator has reached this minimum as there is not other local minimum than the global one.

Therefore G * has model density p x with p y = p x .

The convergence point is a global Nash equilibrium, because there is no approximation error and zero energy F = 0 is a global minimum for discriminator and generator, respectively.

Theorem 1 ensures that other local Nash equilibria are not possible.

GANs are sample-based, that is, samples are drawn from the model for learning BID22 BID19 .

Typically this is done in mini-batches, where each mini-batch consists of two sets of samples, the target samples Y = {y i |i = 1 . . .

N y }, and the model samples X = {x i |i = 1 . . .

N x }.For such finite samples, i.e. point charges, we have to use delta distributions to obtain unbiased estimates of the the model distribution p x (.) and the target distribution p y (.): DISPLAYFORM0 where ?? is the Dirac ??-distribution centered at zero.

These are unbiased estimates of the underlying distribution, as can be seen by: DISPLAYFORM1 In the rest of the paper, we will drop the explicit parameterization with X and Y for all estimates to unclutter notation, and instead just use the hat sign to denote estimates.

In the same fashion as for the distributions, when we use fixed samples X and Y, we obtain the following unbiased estimates for the potential, energy and field given by Eq. (5), Eq. (6), and Eq. FORMULA10 : DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 These are again unbiased, e.g.: DISPLAYFORM5 If we draw samples of infinite size, all these expressions for a fixed sample size lead to the equivalent statements for densities.

The sample-based formulation, that is, point charges in physical terms, can only have local energy minima or maxima at locations of samples BID11 .

Furthermore the field lines originate and end at samples, therefore the field guides model samples x toward real world samples y, as depicted in Figure 1 .

The factors N y and N x in the last equations arise from the fact that ?????? a F gives the force which is applied to a sample with charge.

A sample y i is positively charged with 1/N y and follows ?????? yi F while a sample x i is negatively charged with ???1/N x and therefore follows ?????? xi F , too.

Thus, following the force induced on a sample by the field is equivalent to gradient descent of the energy F with respect to samples y i and x i .

We use the synthetic data set introduced by BID31 BID31 , the Coulomb GAN used a discriminator network with 2 hidden layers of 128 units, however we avoided batch normalization by using the ELU activation function BID9 .

We used the Plummer kernel in 3 dimensions (d = 3) with an epsilon of 3 ( = 3) and a learning rate of 0.01, both of which were exponentially decayed during the 1M update steps of the Adam optimizer.

As can be seen in FIG2 , samples from the learned Coulomb GAN very well approximate the target distribution.

All components of the original distribution are present at the model distribution at approximately the correct ratio, as shown in FIG3 .

Moreover, the generated samples are distributed approximately according to the same spread for each component of the real world distribution.

Coulomb GANs outperform other compared methods, which either fail to learn the distribution completely, ignore some of the modes, or do not capture the within-mode spread of a Gaussian.

The Coulomb GAN is the only GAN approach that manages to avoid a within-cluster collapse leading to insufficient variance within a cluster.

Gaussians.

For constructing the histogram, 10k samples were drawn from the target and the model distribution.

The Coulomb GAN captures the underlying distribution well, does not miss any modes, and places almost all probability mass on the modes.

Only the Coulomb GAN captured the withinmode spread of the Gaussians.

The following gives the pseudo code for training GANs.

Note that when calculating the derivative of??(a i ; X , Y), it is important to only derive with respect to a, and not wrt.

X , Y, even if it can happen that e.g. a ??? X .

In frameworks that offer automatic differentiation such as Tensorflow or Theano, this means stopping the possible gradient back-propagation through those parameters.

Algorithm 1 Minibatch stochastic gradient descent training of Coulomb GANs for updating the the discriminator weights w and the generator weights ??.while Stopping criterion not met do ??? Sample minibatch of N x training samples {x 1 , . . .

, x Nx } from training set ??? Sample minibatch of N y generator samples {y 1 , . . .

, y Ny } from the generator ??? Calculate the gradient for the discriminator weights: DISPLAYFORM0 ??? Calculate the gradient for the generator weights: DISPLAYFORM1 ??? Update weights according to optimizer rule (e.g. Adam): DISPLAYFORM2 end while

Images from a Coulomb GAN after training on the LSUN bedroom data set.

Images from a Coulomb GAN after training on the CIFAR 10 data set.

<|TLDR|>

@highlight

Coulomb GANs can optimally learn a distribution by posing the distribution learning problem as optimizing a potential field