For many applications, in particular in natural science, the task is to determine hidden system parameters from a set of measurements.

Often, the forward process from parameter- to measurement-space is well-defined, whereas the inverse problem is ambiguous: multiple parameter sets can result in the same measurement.

To fully characterize this ambiguity, the full posterior parameter distribution, conditioned on an observed measurement, has to be determined.

We argue that a particular class of neural networks is well suited for this task – so-called Invertible Neural Networks (INNs).

Unlike classical neural networks, which attempt to solve the ambiguous inverse problem directly, INNs focus on learning the forward process, using additional latent output variables to capture the information otherwise lost.

Due to invertibility, a model of the corresponding inverse process is learned implicitly.

Given a specific measurement and the distribution of the latent variables, the inverse pass of the INN provides the full posterior over parameter space.

We prove theoretically and verify experimentally, on artificial data and real-world problems from medicine and astrophysics, that INNs are a powerful analysis tool to find multi-modalities in parameter space, uncover parameter correlations, and identify unrecoverable parameters.

When analyzing complex physical systems, a common problem is that the system parameters of interest cannot be measured directly.

For many of these systems, scientists have developed sophisticated theories on how measurable quantities y arise from the hidden parameters x. We will call such mappings the forward process.

However, the inverse process is required to infer the hidden states of a system from measurements.

Unfortunately, the inverse is often both intractable and ill-posed, since crucial information is lost in the forward process.

To fully assess the diversity of possible inverse solutions for a given measurement, an inverse solver should be able to estimate the complete posterior of the parameters, conditioned on an observation.

This makes it possible to quantify uncertainty, reveal multi-modal distributions, and identify degenerate and unrecoverable parameters -all highly relevant for applications in natural science.

In this paper, we ask if invertible neural networks (INNs) are a suitable model class for this task.

INNs are characterized by three properties:(i) The mapping from inputs to outputs is bijective, i.e. its inverse exists, (ii) both forward and inverse mapping are efficiently computable, and (iii) both mappings have a tractable Jacobian, which allows explicit computation of posterior probabilities.

Networks that are invertible by construction offer a unique opportunity: We can train them on the well-understood forward process x → y and get the inverse y → x for free by The standard direct approach requires a discriminative, supervised loss (SL) term between predicted and true x, causing problems when y → x is ambiguous.

Our network uses a supervised loss only for the well-defined forward process x →

y. Generated x are required to follow the prior p(x) by an unsupervised loss (USL), while the latent variables z are made to follow a Gaussian distribution, also by an unsupervised loss.

See details in Section 3.3.running them backwards at prediction time.

To counteract the inherent information loss of the forward process, we introduce additional latent output variables z, which capture the information about x that is not contained in y. Thus, our INN learns to associate hidden parameter values x with unique pairs [y, z] of measurements and latent variables.

Forward training optimizes the mapping [y, z] = f (x) and implicitly determines its inverse x = f −1 (y, z) = g(y, z).

Additionally, we make sure that the density p(z) of the latent variables is shaped as a Gaussian distribution.

Thus, the INN represents the desired posterior p(x | y) by a deterministic function x = g(y, z) that transforms ("pushes") the known distribution p(z) to x-space, conditional on y.

Compared to standard approaches (see FIG0 , left), INNs circumvent a fundamental difficulty of learning inverse problems: Defining a sensible supervised loss for direct posterior learning is problematic since it requires prior knowledge about that posterior's behavior, constituting a kind of hen-end-egg problem.

If the loss does not match the possibly complicated (e.g. multimodal) shape of the posterior, learning will converge to incorrect or misleading solutions.

Since the forward process is usually much simpler and better understood, forward training diminishes this difficulty.

Specifically, we make the following contributions:• We show that the full posterior of an inverse problem can be estimated with invertible networks, both theoretically in the asymptotic limit of zero loss, and practically on synthetic and real-world data from astrophysics and medicine.• The architectural restrictions imposed by invertibility do not seem to have detrimental effects on our network's representational power.• While forward training is sufficient in the asymptotic limit, we find that a combination with unsupervised backward training improves results on finite training sets.• In our applications, our approach to learning the posterior compares favourably to approximate Bayesian computation (ABC) and conditional VAEs.

This enables identifying unrecoverable parameters, parameter correlations and multimodalities.

Modeling the conditional posterior of an inverse process is a classical statistical task that can in principle be solved by Bayesian methods.

Unfortunately, exact Bayesian treatment of real-world problems is usually intractable.

The most common (but expensive) solution is to resort to sampling, typically by a variant of Markov Chain Monte Carlo (Robert and Casella, 2004; BID11 .

If a model y = s(x) for the forward process is available, approximate Bayesian computation (ABC) is often preferred, which embeds the forward model in a rejection sampling scheme for the posterior p(x|y) (Sunnåker et al., 2013; BID27 Wilkinson, 2013) .Variational methods offer a more efficient alternative, approximating the posterior by an optimally chosen member of a tractable distribution family BID2 .

Neural networks can be trained to predict accurate sufficient statistics for parametric posteriors BID31 Siddharth et al., 2017) , or can be designed to learn a mean-field distribution for the network's weights via dropout variational inference BID10 BID24 .

Both ideas can be combined BID21 to differentiate between data-related and model-related uncertainty.

However, the restriction to limited distribution families fails if the true distribution is too complex (e.g. when it requires multiple modes to represent ambiguous or degenerate solutions) and essentially counters the ability of neural networks to act as universal approximators.

Conditional GANs (cGANs; BID29 BID19 overcome this restriction in principle, but often lack satisfactory diversity in practice (Zhu et al., 2017b) .

For our tasks, conditional variational autoencoders (cVAEs; Sohn et al., 2015) perform better than cGANs, and are also conceptually closer to our approach (see appendix Sec. 2), and hence serve as a baseline in our experiments.

Generative modeling via learning of a non-linear transformation between the data distribution and a simple prior distribution BID5 BID18 has the potential to solve these problems.

Today, this approach is often formulated as a normalizing flow (Tabak et al., 2010; Tabak and Turner, 2013) , which gradually transforms a normal density into the desired data density and relies on bijectivity to ensure the mapping's validity.

These ideas were applied to neural networks by BID5 ; Rippel and Adams (2013); Rezende and Mohamed (2015) and refined by Tomczak and Welling (2016); BID1 ; Trippe and Turner (2018) .

Today, the most common realizations use auto-regressive flows, where the density is decomposed according to the Bayesian chain rule BID25 BID17 BID12 BID32 BID30 BID26 Salimans et al., 2017; Uria et al., 2016) .

These networks successfully learned unconditional generative distributions for artificial data and standard image sets (e.g. MNIST, CelebA, LSUN bedrooms), and some encouraging results for conditional modeling exist as well BID30 Salimans et al., 2017; BID32 Uria et al., 2016) .These normalizing flows possess property (i) of an INN, and are usually designed to fulfill requirement (iii) as well.

In other words, flow-based networks are invertible in principle, but the actual computation of their inverse is too costly to be practical, i.e. INN property (ii) is not fulfilled.

This precludes the possibility of bi-directional or cyclic training, which has been shown to be very beneficial in generative adversarial nets and auto-encoders (Zhu et al., 2017a; BID9 BID8 Teng et al., 2018) .

In fact, optimization for cycle consistency forces such models to converge to invertible architectures, making fully invertible networks a natural choice.

True INNs can be built using coupling layers, as introduced in the NICE BID6 and RealNVP BID7 architectures.

Despite their simple design and training, these networks were rarely studied: BID13 used a NICE-like design as a memory-efficient alternative to residual networks, BID20 demonstrated that the lack of information reduction from input to representation does not cause overfitting, and Schirrmeister et al. FORMULA1 trained such a network as an adverserial autoencoder.

BID4 showed that minimization of an adversarial loss is superior to maximum likelihood training in RealNVPs, whereas the Flow-GAN of BID15 performs even better using bidirectional training, a combination of maximum likelihood and adverserial loss.

The Glow architecture by BID23 incorporates invertible 1x1 convolutions into RealNVPs to achieve impressive image manipulations.

This line of research inspired us to extend RealNVPs for the task of computing posteriors in real-world inverse problems from natural and life sciences.

We consider a common scenario in natural and life sciences: Researchers are interested in a set of variables x ∈ R D describing some phenomenon of interest, but only variables y ∈ R M can actually be observed, for which the theory of the respective research field provides a model y = s(x) for the forward process.

Since the transformation from x to y incurs an information loss, the intrinsic dimension m of y is in general smaller than D, even if the nominal dimensions satisfy M > D. Hence we want to express the inverse model as a conditional probability p(x | y), but its mathematical derivation from the forward model is intractable in the applications we are going to address.

We aim at approximating p(x | y) by a tractable model q(x | y), taking advantage of the possibility to create an arbitrary amount of training data {( DISPLAYFORM0 from the known forward model s(x) and a suitable prior p(x).

While this would allow for training of a standard regression model, we want to approximate the full posterior probability.

To this end, we introduce a latent random variable z ∈ R K drawn from a multi-variate standard normal distribution and reparametrize q(x | y) in terms of a deterministic function g of y and z, represented by a neural network with parameters θ: DISPLAYFORM1 Note that we distinguish between hidden parameters x representing unobservable real-world properties and latent variables z carrying information intrinsic to our model.

Choosing a Gaussian prior for z poses no additional limitation, as proven by the theory of non-linear independent component analysis BID18 .In contrast to standard methodology, we propose to learn the model g(y, z; θ) of the inverse process jointly with a model f (x; θ) approximating the known forward process s(x): DISPLAYFORM2 Functions f and g share the same parameters θ and are implemented by a single invertible neural network.

Our experiments show that joint bi-directional training of f and g avoids many complications arising in e.g. cVAEs or Bayesian neural networks, which have to learn the forward process implicitly.

The relation f = g −1 is enforced by the invertible network architecture, provided that the nominal and intrinsic dimensions of both sides match.

When m ≤ M denotes the intrinsic dimension of y, the latent variable z must have dimension K = D − m, assuming that the intrinsic dimension of x equals its nominal dimension D. If the resulting nominal output dimension M + K exceeds D, we augment the input with a vector x 0 ∈ R M +K−D of zeros and replace x with the concatenation [x, x 0 ] everywhere.

Combining these definitions, our network expresses q(x | y) as DISPLAYFORM3 with Jacobian determinant J x .

When using coupling layers, according to BID7 , computation of J x is simple, as each transformation has a triangular Jacobian matrix.

To create a fully invertible neural network, we follow the architecture proposed by BID7 : The basic unit of this network is a reversible block consisting of two complementary affine coupling layers.

Hereby, the block's input vector u is split into two halves, u 1 and u 2 , which are transformed by an affine function with coefficients exp(s i ) and t i (i ∈ {1, 2}), using element-wise multiplication ( ) and addition: DISPLAYFORM0 Given the output v = [v 1 , v 2 ], these expressions are trivially invertible: DISPLAYFORM1 Importantly, the mappings s i and t i can be arbitrarily complicated functions of v 1 and u 2 and need not themselves be invertible.

In our implementation, they are realized by a succession of several fully connected layers with leaky ReLU activations.

A deep invertible network is composed of a sequence of these reversible blocks.

To increase model capacity, we apply a few simple extensions to this basic architecture.

Firstly, if the dimension D is small, but a complex transformation has to be learned, we find it advantageous to pad both the in-and output of the network with an equal number of zeros.

This does not change the intrinsic dimensions of in-and output, but enables the network's interior layers to embed the data into a larger representation space in a more flexible manner.

Secondly, we insert permutation layers between reversible blocks, which shuffle the elements of the subsequent layer's input in a randomized, but fixed, way.

This causes the splits u = [u 1 , u 2 ] to vary between layers and enhances interaction among the individual variables.

Kingma and Dhariwal (2018) use a similar architecture with learned permutations.

Invertible networks offer the opportunity to simultaneously optimize for losses on both the inand output domains BID15 , which allows for more effective training.

Hereby, we perform forward and backward iterations in an alternating fashion, accumulating gradients from both directions before performing a parameter update.

For the forward iteration, we penalize deviations between simulation outcomes y i = s(x i ) and network predictions f y (x i ) with a loss L y y i , f y (x i ) .

Depending on the problem, L y can be any supervised loss, e.g. squared loss for regression or cross-entropy for classification.

The loss for latent variables penalizes the mismatch between the joint distribution of network DISPLAYFORM0 We block the gradients of L z with respect to y to ensure the resulting updates only affect the predictions of z and do not worsen the predictions of y. Thus, L z enforces two things: firstly, the generated z must follow the desired normal distribution p(z); secondly, y and z must be independent upon convergence (i.e. p(z | y) = p(z)), and not encode the same information twice.

As L z is implemented by Maximum Mean Discrepancy D (Sec. 3.4), which only requires samples from the distributions to be compared, the Jacobian determinants J yz and J s do not have to be known explicitly.

In appendix Sec. 1, we prove the following theorem: DISPLAYFORM1 is trained as proposed, and both the supervised loss DISPLAYFORM2 and the unsupervised loss L z = D q(y, z), p(y) p(z) reach zero, sampling according to Eq. 1 with g = f −1 returns the true posterior p(x | y * ) for any measurement y * .Although L y and L z are sufficient asymptotically, a small amount of residual dependency between y and z remains after a finite amount of training.

This causes q(x | y) to deviate from the true posterior p(x | y).

To speed up convergence, we also define a loss L x on the input side, implemented again by MMD.

It matches the distribution of backward predictions DISPLAYFORM3 In the appendix, Sec. 1, we prove that L x is guaranteed to be zero when the forward losses L y and L z have converged to zero.

Thus, incorporating L x does not alter the optimum, but improves convergence in practice.

Finally, if we use padding on either network side, loss terms are needed to ensure no information is encoded in the additional dimensions.

We a) use a squared loss to keep those values close to zero and b) in an additional inverse training pass, overwrite the padding dimensions with noise of the same amplitude and minimize a reconstruction loss, which forces these dimensions to be ignored.

Maximum Mean Discrepancy (MMD) is a kernel-based method for comparison of two probability distributions that are only accessible through samples BID14 .

While a trainable discriminator loss is often preferred for this task in high-dimensional problems, especially in GAN-based image generation, MMD also works well, is easier to use and much cheaper, and leads to more stable training (Tolstikhin et al., 2017) .

The method requires a kernel function as a design parameter, and we found that kernels with heavier tails than Gaussian are needed to get meaningful gradients for outliers.

We achieved best results with the Inverse Multiquadratic k( DISPLAYFORM0 Figure 2: Viability of INN for a basic inverse problem.

The task is to produce the correct (multi-modal) distribution of 2D points x, given only the color label y * .

When trained with all loss terms from Sec. 3.3, the INN output matches ground truth almost exactly (2nd image).

The ablations (3rd and 4th image) show that we need L y and L z to learn the conditioning correctly, whereas L x helps us remain faithful to the prior.suggestion from Tolstikhin et al. (2017) .

Since the magnitude of the MMD depends on the kernel choice, the relative weights of the losses L x , L y , L z are adjusted as hyperparameters, such that their effect is about equal.

We first demonstrate the capabilities of INNs on two well-behaved synthetic problems and then show results for two real-world applications from the fields of medicine and astrophysics.

Additional details on the datasets and network architectures are provided in the appendix.

Gaussian mixture model: To test basic viability of INNs for inverse problems, we train them on a standard 8-component Gaussian mixture model p(x).

The forward process is very simple: The first four mixture components (clockwise) are assigned label y = red, the next two get label y = blue, and the final two are labeled y = green and y = purple (Fig. 2) .

The true inverse posteriors p(x | y * ) consist of the mixture components corresponding to the given one-hot-encoded label y * .

We train the INN to directly regress one-hot vectors y using a squared loss L y , so that we can provide plain one-hot vectors y * to the inverse network when sampling p(x | y * ).

We observe the following: (i) The INN learns very accurate approximations of the posteriors and does not suffer from mode collapse. (ii) The coupling block architecture does not reduce the network's representational power -results are similar to standard networks of comparable size (see appendix Sec. 2). (iii) Bidirectional training works best, whereas forward training alone (using only L y and L z ) captures the conditional relationships properly, but places too much mass in unpopulated regions of x-space.

Conversely, pure inverse training (just L x ) learns the correct x-distribution, but loses all conditioning information.

Inverse kinematics: For a task with a more complex and continuous forward process, we simulate a simple inverse kinematics problem in 2D space: An articulated arm moves vertically along a rail and rotates at three joints.

These four degrees of freedom constitute the parameters x. Their priors are given by a normal distribution, which favors a pose with 180• angles and centered origin.

The forward process is to calculate the coordinates of the end point y, given a configuration x.

The inverse problem asks for the posterior distribution over all possible inputs x that place the arm's end point at a given y position.

An example for a fixed y * is shown in FIG1 , where we compare our INN to a conditional VAE (see appendix FIG5 for conceptual comparison of architectures).

Adding Inverse Autoregressive Flow (IAF, Kingma et al., 2016) does not improve cVAE performance in this case (see appendix, TAB2 ).

The y * chosen in FIG1 is a hard example, as it is unlikely under the prior p(x) FIG1 and has a strongly bi-modal posterior p(x | y * ).In this case, due to the computationally cheap forward process, we can use approximate Bayesian computation (ABC, see appendix Sec. 7) to sample from the ground truth posterior.

Compared to ground truth, we find that both INN and cVAE recover the two symmetric .

The prior (right) is shown for reference.

The actual end point of each sample may deviate slightly from the target y * ; contour lines enclose the regions containing 97% of these end points.

We emphasize the articulated arm with the highest estimated likelihood for illustrative purposes.

modes well.

However, the true end points of x-samples produced by the cVAE tend to miss the target y * by a wider margin.

This is because the forward process x → y is only learned implicitly during cVAE training.

See appendix for quantitative analysis and details.

After demonstrating the viability on synthetic data, we apply our method to two real world problems from medicine and astronomy.

While we focus on the medical task in the following, the astronomy application is shown in Fig. 5 .In medical science, the functional state of biological tissue is of interest for many applications.

Tumors, for example, are expected to show changes in oxygen saturation s O2 BID16 .

Such changes cannot be measured directly, but influence the reflectance of the tissue, which can be measured by multispectral cameras BID28 .

Since ground truth data can not be obtained from living tissue, we create training data by simulating observed spectra y from a tissue model x involving s O2 , blood volume fraction v hb , scattering magnitude a mie , anisotropy g and tissue layer thickness d (Wirkert et al., 2016) .

This model constitutes the forward process, and traditional methods to learn point estimates of the inverse (Wirkert et al., 2016; BID3 are already sufficiently reliable to be used in clinical trials.

However, these methods can not adequately express uncertainty and ambiguity, which may be vital for an accurate diagnosis.

Competitors.

We train an INN for this problem, along with two ablations (as in Fig. 2) , as well as a cVAE with and without IAF BID25 and a network using the method of BID21 , with dropout sampling and additional aleatoric error terms for each parameter.

The latter also provides a point-estimate baseline (classical NN) when used without dropout and error terms, which matches the current state-of-the-art results in Wirkert et al. (2017) .

Finally, we compare to ABC, approximating p(x | y * ) with the 256 samples closest to y * .

Note that with enough samples ABC would produce the true posterior.

We performed 50 000 simulations to generate samples for ABC at test time, taking one week on a GPU, but still measure inconsistencies in the posteriors.

The learning-based methods are trained within minutes, on a training set of 15 000 samples generated offline.

Error measures.

We are interested in both the accuracy (point estimates), and the shape of the posterior distributions.

For point estimatesx, i.e. MAP estimates, we compute the deviation from ground-truth values x * in terms of the RMSE over test set observations y * , RMSE = E y * [ x − x * 2 ].

The scores are reported both for the main parameter of interest s O2 , and the parameter subspace of s O2 , v hb , a mie , which we found to be the only recoverable parameters.

Furthermore, we check the re-simulation error: We apply the simulation s(x) to the point estimate, and compare the simulation outcome to the conditioning y * .

To evaluate the shape of the posteriors, we compute the calibration error for the sampling-based methods, based on the fraction of ground truth inliers α inl.

for corresponding α-confidence-region of Quantitative results.

Evaluation results for all methods are presented in TAB0 .

The INN matches or outperforms other methods in terms of point estimate error.

Its accuracy deteriorates slightly when trained without L x , and entirely when trained without the conditioning losses L y and L z , just as in Fig. 2 .

For our purpose, the calibration error is the most important metric, as it summarizes the correctness of the whole posterior distribution in one number (see appendix FIG0 .

Here, the INN has a big lead over cVAE(-IAF) and Dropout, and even over ABC due to the low ABC sample count.

Qualitative results.

FIG2 shows generated parameter distributions for one fixed measurement y * , comparing the INN to cVAE-IAF, Dropout sampling and ABC.

The three former methods use a sample count of 160 000 to produce smooth curves.

Due to the sparse posteri- Figure 5: Astrophysics application.

Properties x of star clusters in interstellar gas clouds are inferred from multispectral measurements y.

We train an INN on simulated data, and show the sampled posterior of 5 parameters for one y * (colors as in FIG2 .

The peculiar shape of the prior is due to the dynamic nature of these simulations.

We include this application as a real-world example for the INN's ability to recover multiple posterior modes, and strong correlations in p(x | y * ), see details in appendix, Sec. 5.ors of 256 samples in the case of ABC, kernel density estimation was applied to its results, with a bandwidth of σ = 0.1.

The results produced by the INN provide relevant insights: First, we find that the posteriors for layer thickness d and anisotropy g match the shape of their priors, i.e. y * holds no information about these parameters -they are unrecoverable.

This finding is supported by the ABC results, whereas the other two methods misleadingly suggest a roughly Gaussian posterior.

Second, we find that the sampled distributions for the blood volume fraction v hb and scattering amplitude a mie are strongly correlated (rightmost plot).

This phenomenon is not an analysis artifact, but has a sound physical explanation: As blood volume fraction increases, more light is absorbed inside the tissue.

For the sensor to record the same intensities y * as before, scattering must be increased accordingly.

In FIG0 in the appendix, we show how the INN is applied to real multispectral images.

We have shown that the full posterior of an inverse problem can be estimated with invertible networks, both theoretically and practically on problems from medicine and astrophysics.

We share the excitement of the application experts to develop INNs as a generic tool, helping them to better interpret their data and models, and to improve experimental setups.

As a side effect, our results confirm the findings of others that the restriction to coupling layers does not noticeably reduce the expressive power of the network.

In summary, we see the following fundamental advantages of our INN-based method compared to alternative approaches: Firstly, one can learn the forward process and obtain the (more complicated) inverse process 'for free', as opposed to e.g. cGANs, which focus on the inverse and learn the forward process only implicitly.

Secondly, the learned posteriors are not restricted to a particular parametric form, in contrast to classical variational methods.

Lastly, in comparison to ABC and related Bayesian methods, the generation of the INN posteriors is computationally very cheap.

In future work, we plan to systematically analyze the properties of different invertible architectures, as well as more flexible models utilizing cycle losses, in the context of representative inverse problem.

We are also interested in how our method can be scaled up to higher dimensionalities, where MMD becomes less effective.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros.

Unpaired image-to-image translation using cycle-consistent adversarial networks.

In CVPR, pages 2223-2232, 2017a.

Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A Efros, Oliver Wang, and Eli Shechtman.

Toward multimodal image-to-image translation.

In Advances in Neural Information Processing Systems, pages 465-476, 2017b.

DISPLAYFORM0 In Eq. 9, the Jacobians cancel out due to the inverse function theorem, i.e. the Jacobian DISPLAYFORM1 is trained as proposed, and both the supervised loss DISPLAYFORM2 and the unsupervised loss L z = D q(y, z), p(y) p(z) reach zero, sampling according to Eq. 1 with g = f −1 returns the true posterior p(x | y * ) for any measurement y * .Proof: We denote the chosen latent distribution as p Z (z), the distribution of observations as p Y (y), and the joint distribution of network outputs as q(y, z).

As shown by BID14 , if the MMD loss converges to 0, the network outputs follow the prescribed distribution: DISPLAYFORM3 Suppose we take a posterior conditioned on a fixed y * , i.e. p(x | y * ), and transform it using the forward pass of our perfectly converged INN.

From this we obtain an output distribution q * (y, z).

Because L y = 0, we know that the output distribution of y (marginalized over z) must be q * (y) = δ(y − y * ).

Also, because of the independence between z and y in the output, the distribution of z-outputs is still q * (z) = p Z (z).

So the joint distribution of outputs is DISPLAYFORM4 When we invert the network, and repeatedly input y * while sampling z ∼ p Z (z), this is the same as sampling [y, z] from the q * (y, z) above.

Using the Lemma from above, we know that the inverted network will output samples from p(x | y * ).Corollary: If the conditions of the theorem above are fulfilled, the unsupervised reverse loss L x = D q(x), p X (x) between the marginalized outputs of the inverted network, q(x), and the prior data distribution, p X (x), will also be 0.

This justifies using the loss on the prior to speed up convergence, without altering the final results.

Proof: Due to the theorem, the estimated posteriors generated by the INN are correct, i.e. q(x | y * ) = p(x | y * ).

If they are marginalized over observations y * from the training data, then q(x) will be equal to p X (x) by definition.

As shown by BID14 , this is equivalent to L x = 0.

In Sec. 4.1, we demonstrate that the proposed INN can approximate the true posteriors very well and is not hindered by the required coupling block architecture.

Here we show how some existing methods do on the same task, using neural networks of similar size as the INN.

INN, all losses cVAE cVAE-IAF cGAN Larger cGAN Generator + MMD Dropout sampling Figure 6 : Results of several existing methods for the Gaussian mixture toy example.cGAN Training a conditional GAN of network size comparable to the INN (counting only the generator) and only two noise dimensions turned out to be challenging.

Even with additional pre-training to avoid mode collapse, the individual modes belonging to one label are reduced to nearly one-dimensional structures.

Larger cGAN In order to match the results of the INN, we trained a more complex cGAN with 2M parameters instead of the previous 10K, and a latent dimension of 128, instead of 2.To prevent mode collapse, we introduced an additional regularization: an extra loss term forces the variance of generator outputs to match the variance of the training data prior.

With these changes, the cGAN can be seen to recover the posteriors reasonably well.

Generator + MMD Another option is to keep the cGAN generator the same size as our INN, but replace the discriminator with an MMD loss (cf.

Sec. 3.4) .

This loss receives a concatenation of the generator output x and the label y it was supplied with, and compares these batch-wise with the concatenation of ground truth (x, y)-pairs.

Note that in contrast to this, the corresponding MMD loss of the INN only receives x, and no information about y. For this small toy problem, we find that the hand-crafted MMD loss dramatically improves results compared to the smaller learned discriminator.cVAE We also compare to a conditional Variational Autoencoder of same total size as the INN.

There is some similarity between the training setup of our method FIG5 and For the standard cVAE, the IAF component is omitted.that of cVAE FIG5 , as the forward and inverse pass of an INN can also be seen as an encoder-decoder pair.

The main differences are that the cVAE learns the relationship x → y only indirectly, since there is no explicit loss for it, and that the INN requires no reconstruction loss, since it is bijective by construction.cVAE-IAF We adapt the cVAE to use Inverse Autoregressive Flow BID25 between the encoder and decoder.

On the Gaussian mixture toy problem, the trained cVAE-IAF generates correct posteriors on par with our INN (see Fig. 6 ).Dropout sampling The method of dropout sampling with learned error terms is by construction not able to produce multi-modal outputs, and therefore fails on this task.

To analyze how the latent space of our INN is structured for this task, we choose a fixed label y * and sample z from a dense grid.

For each z, we compute x through our inverse network and colorize this point in latent (z) space according to the distance from the closest mode in x-space.

We can see that our network learns to shape the latent space such that each mode receives the expected fraction of samples (Fig. 8 ).Figure 8: Layout of INN latent space for one fixed label y * , colored by mode closest to x = g(y * , z).

For each latent position z, the hue encodes which mode the corresponding x belongs to and the luminosity encodes how close x is to this mode.

Note that colors used here do not relate to those in Fig. 2 , and encode the position x instead of the label y.

The first three columns correspond to labels green, blue and red Fig. 2 .

White circles mark areas that contain 50% and 90% of the probability mass of latent prior p(z).

A short video demonstrating the structure of our INN's latent space can be found under https://gfycat.com/SoggyCleanHog, for a slightly different arm setup.

The dataset is constucted using gaussian priors x i ∼ N (0, σ i ), with σ 1 = 0.25 and σ 2 = σ 3 = σ 4 = 0.5 ∧ = 28.65• .

The forward process is given by DISPLAYFORM0 with the arm lenghts l 1 = 0.5, l 2 = 0.5, l 3 = 1.0.To judge the quality of posteriors, we quantify both the re-simulation error and the calibration error over the test set, as in Sec. 4.2 of the paper.

Because of the cheap simulation, we average the re-simulation error over the whole posterior, and not only the MAP estimate.

In TAB2 , we find that the INN has a clear advantage in both metrics, confirming the observations from FIG1 .

Figure 9 : Posteriors generated for less challenging observations y * than in FIG1 .

The following figure shows the results when the INN trained in Sec. 4.2 is applied pixel-wise to multispectral endoscopic footage.

In addition to estimating the oxygenation s O2 , we measure the uncertainty in the form of the 68% confidence interval.

The clips (arrows) on the connecting tissue cause lower oxygenation (blue) in the small intestine.

Uncertainty is low in crucial areas and high only at some edges and specularities.

Star clusters are born from a large reservoir of gas and dust that permeates the Galaxy, the interstellar medium (ISM).

The densest parts of the ISM are called molecular clouds, and star formation occurs in regions that become unstable under their own weight.

The process is governed by the complex interplay of competing physical agents such as gravity, turbulence, magnetic fields, and radiation; with stellar feedback playing a decisive regulatory role (S. Klessen and C. O. Glover, 2016) .

To characterize the impact of the energy and momentum input from young star clusters on the dynamical evolution of the ISM, astronomers frequently study emission lines from chemical elements such as hydrogen or oxygen.

These lines are produced when gas is ionized by stellar radiation, and their relative intensities depend on the ionization potential of the chemical species, the spectrum of the ionizing radiation, the gas density as well as the 3D geometry of the cloud, and the absolute intensity of the radiation (Pellegrini et al., 2011) .

Key diagnostic tools are the so-called BPT diagrams (after BID0 emission of ionized hydrogen, H+ , to normalize the recombination lines of O++ , O+ and S+ (see also BID22 .

We investigate the dynamical feedback of young star clusters on their parental cloud using the WARPFIELD 1D model developed by Rahner et al. (2017) .

It follows the entire temporal evolution of the system until the cloud is destroyed, which could take several stellar populations to happen.

At each timestep we employ radiative transfer calculations BID36 to generate synthetic emission line maps which we use to train the neural network.

Similar to the medical application from Section 4.2, the mapping from simulated observations to underlying physical parameters (such as cloud and cluster mass, and total age of the system) is highly degenerate and ill-posed.

As an intermediary step, we therefore train our forward model to predict the observable quantities y (emission line ratios) from composite simulation outputs x (such as ionizing luminosity and emission rate, cloud density, expansion velocity, and age of the youngest cluster in the system, which in the case of multiple stellar populations could be considerably smaller than the total age).

Using the inverse of our trained model for a given set of observations y * , we can obtain a distribution over the unobservable properties x of the system.

Results for one specific y are shown in Fig. 5 .

Note that our network recovers a decidedly multimodal distribution of x that visibly deviates from the prior p(x).

Note also the strong correlations in the system.

For example, the measurements y * investigated may correspond to a young cluster with large expansion velocity, or to an older system that expands slowly.

Finding these ambiguities in p(x | y * ) and identifying degeneracies in the underlying model are pivotal aspects of astrophysical research, and a method to effectively approximate full posterior distributions has the potential to lead to a major breakthrough in this field.

In Sec. 4.2, we report the median calibration error for each method.

The following figure plots the calibration error, q inliers − q, against the level of confidence q. Negative values mean that a model is overconfident, while positive values say the opposite.

While there is a whole field of research concerned with ABC approaches and their efficiencyaccuracy tradeoffs, our use of the method here is limited to the essential principle of rejection sampling.

When we require N samples of x from the posterior p(x | y * ) conditioned on some y * , there are two basic ways to obtain them:Threshold: We set an acceptance threshold , repeatedly draw x-samples from the prior, compute the corresponding y-values (via simulation) and keep those where dist(y, y * ) < , until we have accepted N samples.

The smaller we want , the more simulations have to be run, which is why we use this approach only for the experiment in Sec. 4.1, where we can afford to run the forward process millions or even billions of times.

Quantile: Alternatively, we choose what quantile q of samples shall be accepted, and then run exactly N/q simulations.

All sampled pairs (x, y) are sorted by dist(y, y * ) and the N closest to y * form the posterior.

This allows for a more predictable runtime when the simulations are costly, as in the medical application in Sec. 4.2 where q = 0.005.

TAB3 summarizes the datasets used throughout the paper.

The architecture details are given in the following.

<|TLDR|>

@highlight

To analyze inverse problems with Invertible Neural Networks

@highlight

The author proposes to use invertible networks to solve ambiguous inverse problems and suggest to not only train the forward model, but also the inverse model with an MMD critic.

@highlight

The research paper proposes an invertible network with observations for posterior probability of complex input distributions with a theoretical valid bidirectional training scheme.
