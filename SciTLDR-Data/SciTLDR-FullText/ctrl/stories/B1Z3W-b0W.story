Inference models, which replace an optimization-based inference procedure with a learned model, have been fundamental in advancing Bayesian deep learning, the most notable example being variational auto-encoders (VAEs).

In this paper, we propose iterative inference models, which learn how to optimize a variational lower bound through repeatedly encoding gradients.

Our approach generalizes VAEs under certain conditions, and by viewing VAEs in the context of iterative inference, we provide further insight into several recent empirical findings.

We demonstrate the inference optimization capabilities of iterative inference models, explore unique aspects of these models, and show that they outperform standard inference models on typical benchmark data sets.

Generative models present the possibility of learning structure from data in unsupervised or semisupervised settings, thereby facilitating more flexible systems to learn and perform tasks in computer vision, robotics, and other application domains with limited human involvement.

Latent variable models, a class of generative models, are particularly well-suited to learning hidden structure.

They frame the process of data generation as a mapping from a set of latent variables underlying the data.

When this mapping is parameterized by a deep neural network, the model can learn complex, non-linear relationships, such as object identities (Higgins et al. (2016) ) and dynamics (Xue et al. (2016) ; Karl et al. (2017) ).

However, performing exact posterior inference in these models is computationally intractable, necessitating the use of approximate inference methods.

Variational inference (Hinton & Van Camp (1993) ; Jordan et al. (1998) ) is a scalable approximate inference method, transforming inference into a non-convex optimization problem.

Using a set of approximate posterior distributions, e.g. Gaussians, variational inference attempts to find the distribution that most closely matches the true posterior.

This matching is accomplished by maximizing a lower bound on the marginal log-likelihood, or model evidence, which can also be used to learn the model parameters.

The ensuing expectation-maximization procedure alternates between optimizing the approximate posteriors and model parameters (Dempster et al. (1977) ; Neal & Hinton (1998) ; Hoffman et al. (2013) ).

Amortized inference (Gershman & Goodman (2014) ) avoids exactly computing optimized approximate posterior distributions for each data example, instead learning a separate inference model to perform this task.

Taking the data example as input, this model outputs an estimate of the corresponding approximate posterior.

When the generative and inference models are parameterized with neural networks, the resulting set-up is referred to as a variational auto-encoder (VAE) (Kingma & Welling (2014) ; Rezende et al. (2014) ).We introduce a new class of inference models, referred to as iterative inference models, inspired by recent work in learning to learn (Andrychowicz et al. (2016) ).

Rather than directly mapping the data to the approximate posterior, these models learn how to iteratively estimate the approximate posterior by repeatedly encoding the corresponding gradients, i.e. learning to infer.

With inference computation distributed over multiple iterations, we conjecture that this model set-up should provide improved inference estimates over standard inference models given sufficient model capacity.

Our work is presented as follows: Section 2 contains background on latent variable models, variational inference, and inference models; Section 3 motivates and introduces iterative inference models; Section 4 presents this approach for latent Gaussian models, showing that a particular form of iterative inference models reduces to standard inference models under mild assumptions; Section 5 contains empirical results; and Section 6 concludes our work.

Latent variable models are generative probabilistic models that use local (per data example) latent variables, z, to model observations, x, using global (across data examples) parameters, θ.

A model is defined by the joint distribution p θ (x, z) = p θ (x|z)p θ (z), which is composed of the conditional likelihood and the prior.

Learning the model parameters and inferring the posterior p(z|x) are intractable for all but the simplest models, as they require evaluating the marginal likelihood, p θ (x) = p θ (x, z)dz, which involves integrating the model over z. For this reason, we often turn to approximate inference methods.

Variational inference reformulates this intractable integration as an optimization problem by introducing an approximate posterior 1 q(z|x), typically chosen from some tractable family of distributions, and minimizing the KL-divergence from the true posterior, D KL (q(z|x)||p(z|x)).

This quantity cannot be minimized directly, as it contains the true posterior.

Instead, the KL-divergence can be decomposed into DISPLAYFORM0 where L is the evidence lower bound (ELBO), which is defined as:L ≡ E z∼q(z|x) [log p θ (x, z) − log q(z|x)] (2) = E z∼q(z|x) [log p θ (x|z)] − D KL (q(z|x)||p θ (z)).Briefly, the first term in eq. 3 can be considered as a reconstruction term, as it expresses how well the output fits the data example.

The second term can be considered as a regularization term, as it quantifies the dissimilarity between the latent representation and the prior.

Because log p θ (x) is not a function of q(z|x), in eq. 1 we can minimize D KL (q(z|x)||p(z|x)), thereby performing approximate inference, by maximizing L w.r.t.

q(z|x).

Likewise, because D KL (q(z|x)||p(z|x)) is non-negative, L is a lower bound on log p θ (x), meaning that if we have inferred an optimal q(z|x), learning corresponds to maximizing L w.r.t.

θ.

The optimization procedures involved in inference and learning, when implemented using conventional gradient ascent techniques, are respectively the expectation and maximization steps of the variational EM algorithm (Dempster et al. (1977); Neal & Hinton (1998); Hoffman et al. (2013) ), which alternate until convergence.

When q(z|x) takes a parametric form, the expectation step for data example x (i) involves finding a set of distribution parameters, λ (i) , that are optimal.

With a factorized Gaussian density over continuous variables, i.e. DISPLAYFORM0 q ), this entails repeatedly estimating the stochastic gradients DISPLAYFORM1 .

This direct optimization procedure, which is repeated for each example, is not only computationally costly for expressive generative models and large data sets, but also sensitive to step sizes and initial conditions.

Amortized inference (Gershman & Goodman (2014) ) replaces the optimization of each set of local approximate posterior parameters, λ (i) , with the optimization of a set of global parameters, φ, contained within an inference model.

Taking x (i) as input, this model directly outputs estimates of λ (i) .

Sharing the inference model across data examples allows for an efficient algorithm, in which φ and θ can be updated jointly.

The canonical example, the variational auto-encoder (VAE) (Kingma & Welling (2014); Rezende et al. (2014) ), employs the reparameterization trick to propagate stochastic gradients from the generative model to the inference model, both of which are parameterized by neural networks.

The formulation has an intuitive interpretation: the inference model encodes x into q(z|x), and the generative model decodes samples from q(z|x) into p(x|z).

Throughout the rest of this paper, we refer to inference models of this form as standard inference models.

Optimization surface of L (in nats) for a 2-D latent Gaussian model and a particular MNIST data example.

Shown on the plot are the MAP (optimal estimate), the output of a standard inference model (VAE), and an expectation step trajectory of variational EM using stochastic gradient ascent.

The plot on the right shows the estimates of each inference scheme near the optimum.

The expectation step arrives at a better final inference estimate than the standard inference model.

In Section 3.2, we introduce our contribution, iterative inference models.

We first motivate our approach in Section 3.1 by interpreting standard inference models in VAEs as optimization models, i.e. models that learn to perform optimization.

Using insights from other optimization models, this interpretation extends and improves upon standard inference models.

As described in Section 2.1, variational inference transforms inference into the maximization of L w.r.t.

the parameters of q(z|x), constituting the expectation step of the variational EM algorithm.

In general, this is a non-convex optimization problem, making it somewhat surprising that an inference model can learn to output reasonable estimates of q(z|x) across data examples.

Of course, directly comparing inference schemes is complicated by the fact that generative models adapt to accommodate their approximate posteriors.

Nevertheless, inference models attempt to replace traditional optimization techniques with a learned mapping from x to q(z|x).We demonstrate this point in FIG0 by visualizing the optimization surface of L defined by a trained 2-D latent Gaussian model and a particular data example, in this case, a binarized MNIST digit.

To visualize the surface, we use a 2-D point estimate as the approximate posterior, q(z|x) = δ(z = µ q ), where µ q = (µ 1 , µ 2 ) ∈ R 2 and δ is the Dirac delta function.

See Appendix C.1 for further details.

Shown on the plot are the MAP (i.e. optimal) estimate, the estimate from a trained inference model, and an expectation step trajectory using stochastic gradient ascent on µ q .

The expectation step arrives at a better final estimate, but it requires many iterations and is dependent on the step size and initial estimate.

The inference model outputs a near-optimal estimate in one forward pass without hand tuning (other than the architecture), but it is restricted to this single estimate.

Note that the inference model does not attain the optimal estimate, resulting in an "amortization gap" (Cremer et al. (2017) ).This example illustrates how inference models differ from conventional optimization techniques.

Despite having no convergence guarantees on inference optimization, inference models have been shown to work well empirically.

However, by learning a direct mapping from x to q(z|x), standard inference models are restricted to only single-step estimation procedures, which may yield worse inference estimates.

The resulting large amortization gap then limits the quality of the accompanying generative model.

To improve upon this paradigm, we take inspiration from the area of learning to learn, where Andrychowicz et al. (2016) showed that an optimizer model, instantiated as a recurrent neural network, can learn to optimize the parameters of an optimizee model, another neural network, .

θ refers to the generative model (decoder) parameters.

∇ λ L denotes the gradients of the ELBO w.r.t.

the distribution parameters, λ, of the approximate posterior, q(z|x).

Iterative inference models learn to perform approximate inference optimization by using these gradients and a set of inference model (encoder) parameters, φ.

See FIG6 for a similar set of diagrams with unrolled computational graphs.for various tasks.

The optimizer model receives the optimizee's parameter gradients and outputs updates to these parameters to improve the optimizee's loss.

Because the computational graph is differentiable, the optimizer itself can also be learned.

Optimization models can learn to adaptively adjust update step sizes, potentially speeding up and improving optimization.

While Andrychowicz et al. (2016) focus primarily on parameter optimization (i.e. learning), we apply an analogous approach to inference optimization in latent variable models.

We refer to this class of optimization models as iterative inference models, as they are inference models that iteratively update their approximate posterior estimates.

Our work differs from that of Andrychowicz et al. (2016) in three distinct ways: (1) variational inference is a qualitatively different optimization problem, involving amortization across data examples rather than learning tasks; (2) we utilize nonrecurrent optimization models, providing a more computationally efficient model that breaks the assumption that previous gradient information is essential for learned optimization; and (3) we provide a novel model formulation that approximates gradient steps using locally computed errors on latent and observed variables (see Section 4.1).

We formalize our approach in the following section.

We present iterative inference models starting from the context of standard inference models.

For a standard inference model f with parameters φ, the estimate of the approximate posterior distribution parameters λ (i) for data example x (i) is of the form: DISPLAYFORM0 We propose to instead use an iterative inference model, also denoted as f with parameters φ.

With DISPLAYFORM1 t ; θ) as the ELBO for data example x (i) at inference iteration t, the model uses DISPLAYFORM2 t , to output updated estimates of λ (i) : DISPLAYFORM3 where DISPLAYFORM4 t is the estimate of λ (i) at inference iteration t. We use f t to highlight that the form of f at iteration t may depend on hidden states within the iterative inference model, such as those found within recurrent neural networks.

See Figures 2 and 8 for schematic comparisons of iterative inference models with variational EM and standard inference models.

As with standard inference models, the parameters of an iterative inference model can be updated using stochastic estimates of ∇ φ L, obtained through the reparameterization trick or other methods.

Model parameter updating is typically performed using standard optimization techniques.

Note that eq. 5 is in a general form and contains, as a special case, the residual updating scheme used in Andrychowicz et al. (2016) .

We now describe an example of iterative inference models for latent Gaussian generative models, deriving the gradients to understand the source of the approximate posterior updates.

Latent Gaussian models are latent variable models with Gaussian prior distributions over latent variables: DISPLAYFORM0 This class of models is often used in VAEs and is a common choice for representing continuous-valued latent variables.

While the approximate posterior can be any probability density, it is typically also chosen as Gaussian: q(z|x) = N (z; µ q , diag σ 2 q ).

With this choice, λ (i) corresponds to {µ DISPLAYFORM1 q } for example x (i) .

Dropping the superscript (i) to simplify notation, we can express eq. 5 for this model as: DISPLAYFORM2 DISPLAYFORM3 where f µq t and f σ 2 q t are the iterative inference models for updating µ q and σ 2 q respectively.

For continuous observations, we can use a Gaussian output density: DISPLAYFORM4 is a non-linear function of z, and σ 2 x is a global parameter, a common assumption in these models.

The approximate posterior parameter gradients for this model are (see Appendix A): DISPLAYFORM5 where ∼ N (0, I) is the auxiliary noise variable from the reparameterization trick, denotes element-wise multiplication, and all division is performed element-wise.

In Appendix A, we also derive the corresponding gradients for a Bernoulli output distribution, which take a similar form.

Although we only derive gradients for these two output distributions, note that iterative inference models can be used with any distribution form.

We now briefly discuss the terms in eqs. 8 and 9.

Re-expressing the reparameterized latent variable as z = µ q + σ q , the gradients have two shared terms, (x − µ x )/σ 2 x and (z − µ p )/σ 2 p , the precision-weighted errors at the observed ("bottom-up") and latent ("top-down") levels respectively.

The terms ∂µx ∂µq and ∂µx ∂σ 2 q are the Jacobian matrices of µ x w.r.t.

the approximate posterior parameters, which effectively invert the output model.

Understanding the significance of each term, in the following section we provide an alternative formulation of iterative inference models for latent Gaussian generative models.

The approximate posterior gradients are inherently stochastic, arising from the fact that evaluating L involves approximating expectations (eq. 2) using Monte Carlo samples of z ∼ q(z|x).

As these estimates always contain some degree of noise, a close approximation to these gradients should also suffice for updating the approximate posterior parameters.

The motivations for this are two-fold: (1) approximate gradients may be easier to compute, especially in an online setting, and (2) by encoding more general terms, the inference model may be able to approximate higher-order approximate posterior derivatives, allowing for faster convergence.

We now provide an alternative formulation of iterative inference models for latent Gaussian models that approximates gradient information.

With the exception of ∂µx ∂µq and ∂µx ∂σ 2 q , all terms in eqs. 8 and 9 can be easily computed using x and the distribution parameters of p(x|z), p(z), and q(z|x).

Likewise, higher-order approximate posterior derivatives consist of these common terms as well as higher-order derivatives of the output model.

As the output model derivatives are themselves functions, by encoding only the common terms, we can offload these (approximate) derivative calculations onto the iterative inference model.

Again dropping the superscript (i), one possible set-up is formulated as follows: DISPLAYFORM0 DISPLAYFORM1 where, in the case of a Gaussian output density, the stochastic error terms are defined as DISPLAYFORM2 .

This encoding scheme resembles the approach taken in DRAW (Gregor et al. FORMULA0 ), where reconstruction errors, x − µ t,x , are iteratively encoded.

However, DRAW and later variants (Gregor et al. FORMULA0 ) do not explicitly account for latent errors, ε z,t , or approximate posterior estimates.

If possible, these terms must instead be implicitly handled by the inference model's hidden states.

In Section 5.2, we demonstrate that iterative inference models of this form do indeed learn to infer.

Unlike gradient encoding iterative inference models, these error encoding models do not require gradients at test time and they empirically perform well even with few inference iterations.

Under a certain set of assumptions, single-iteration iterative inference models of the derivative approximating form proposed in Section 4.1 are equivalent to standard inference models, as used in conventional VAEs.

Specifically, assuming:1.

the initial approximate posterior estimate is a global constant: DISPLAYFORM0 we are in the limit of infinite samples of the initial auxiliary variable 0 , then the initial approximate posterior estimate (µ q,0 , σ 2 q,0 ) and initial latent error (ε z,0 ) are constants and the initial observation error (ε x,0 ) is a constant affine transformation of the observation (x).

When the inference model is a neural network, then encoding x or an affine transformation of x is equivalent (assuming the inputs are properly normalized).

Therefore, eqs. 10 and 11 simplify to that of a standard inference model, eq. 4.

From this perspective, standard inference models can be interpreted as single-step optimization models that learn to approximate derivatives at a single latent point.

In the following section, we consider the case in which the second assumption is violated; iterative inference models naturally handle this case, whereas standard inference models do not.

Hierarchical latent variable models contain higher level latent variables that provide empirical priors on lower level variables; p θ (z) is thus observation-dependent (see Figure 7 in Appendix A.6).

The approximate posterior gradients for an intermediate level in a hierarchical latent Gaussian model (see Appendix A.6) take a similar form as eqs. 8 and 9, comprising bottom-up errors from lower variables and top-down errors from higher variables.

Iterative inference models encode both of these errors, either directly or through the gradient.

However, standard inference models, which map x and lower latent variables to each level of latent variables, can only approximate bottom-up information.

Lacking top-down prior information, these models must either use a less expressive prior or output poor approximate posterior estimates.

Sønderby et al. (2016) identified this phenomenon, proposing a "top-down inference" technique.

Iterative inference models formalize and extend this technique.

We performed experiments using latent Gaussian models trained on MNIST, Omniglot (Lake et al. FORMULA14 ).

MNIST and Omniglot were dynamically binarized and modeled with Bernoulli output distributions, and Street View House Numbers and CIFAR-10 were modeled with Gaussian output distributions, using the procedure from Gregor et al. (2016) .

All experiments presented here use fully-connected neural networks.

Reported values of L were estimated using 1 sample (Figures 3, 5, 6) , and reported values of − log p(x) were estimated using 5,000 importance weighted samples TAB0 .

Additional experiment details, including model architectures and optimizers, can be found in Appendix C. We present additional experiments on text data in Appendix D. Source code will be released online.

To confirm the ability of iterative inference models to optimize the approximate posterior, we tested these models in the simplified setting of a 2D latent Gaussian model, trained on MNIST, with a point estimate approximate posterior.

The generative model architecture and approximate posterior form are identical to those used in Section 3.1 (see Appendix C.1).

Here we show a result from encoding x and ∇ µq L through a feedforward neural network.

In Figure 3 , we visualize an optimization trajectory taken by this model for a particular test example.

Despite lacking convergence guarantees, the model learns to adaptively adjust inference update step sizes to navigate the optimization surface, arriving and remaining at a near-optimal approximate posterior estimate for this example.

Approximate inference optimization can also be visualized through data reconstructions.

In eq. 3, the reconstruction term encourages q(z|x) to represent outputs that closely match the data examples.

As this is typically the dominant term in L, during inference optimization, the output reconstructions should improve in terms of visual quality, more closely matching x. We demonstrate this phenomenon with iterative inference models for several data sets in Figure 4 (see Appendix C.2 for additional reconstructions.).

Reconstruction quality noticeably improves during inference.

We highlight two unique aspects of iterative inference models: direct improvement with additional samples and inference iterations.

These aspects provide two advantageous qualitative differences over standard inference models.

Additional approximate posterior samples provide more precise gradient estimates, potentially allowing an iterative inference model to output more precise updates.

To verify this, we trained standard and iterative inference models on MNIST using 1, 5, 10, and 20 approximate posterior samples.

Iterative inference models were trained by encoding the data (x) and approximate posterior gradients (∇ λ L) for 5 iterations.

The results are shown in FIG4 , where we observe that the iterative inference model improves by more than 1 nat with additional samples, while the standard inference model improves by roughly 0.5 nats.

We investigated the effect of training with additional inference iterations while encoding approximate posterior gradients (∇ λ L) or errors (ε x , ε z ), with or without the data (x).

Section 4 and Appendix A define these terms.

Note that the encoded terms affect the number of input parameters to the inference model.

Here, the iterative inference model that only encodes ∇ λ L has fewer input parameters than a standard inference model, whereas the models that encode errors or data have strictly more input parameters.

Experiments were performed on MNIST, with results for 2, 5, 10, and 16 inference iterations in FIG4 .

All encoding schemes outperformed standard inference models with the same architecture, which we found to be consistent over a range of architectures.

Encoding the data was beneficial, allowing the inference model to trade off between learning a direct and iterative mapping.

Encoding errors allows the iterative inference model to approximate higher order derivatives (Section 4.1), which we observe helps when training with fewer inference iterations.

However, it appears that these approximations are less helpful with additional iterations, where derivative approximation errors likely limit performance.

TAB0 contains the estimated marginal log-likelihood on MNIST and CIFAR-10 for standard and iterative inference models, including hierarchical inference models.

Iterative inference models were trained by encoding the data and errors for 5 inference iterations.

With the same architecture, iterative inference models outperform their standard counterparts.

See Appendix C.5 for details and discussion.

We also compared the inference optimization performance of iterative inference models with variational EM expectation steps using various optimizers.

In Figure 6 , we observe that the iterative inference model empirically converges substantially faster to better estimates, even with only local gradient information.

See Appendix C.6 for details and discussion.

To summarize, iterative inference models outperform standard inference models in terms of inference capabilities, yet are far more computationally efficient than variational EM.

Consider a latent variable model, p θ (x, z) = p θ (x|z)p θ (z), where the prior on z is a factorized Gaussian density, p θ (z) = N (z; µ p , diag σ 2 x ), and the conditional likelihood, p θ (x|z), is Bernoulli for binary observations or Gaussian for continuous observations.

We introduce an approximate posterior distribution, q(z|x), which can be any parametric probability density defined over real values.

Here, we assume that q also takes the form of a factorized Gaussian density, q(z|x) = N (z; µ q , diag σ 2 q ).

The objective during variational inference is to maximize L w.r.t.

the parameters of q(z|x), i.e. µ q and σ DISPLAYFORM0 To solve this optimization problem, we will inspect the gradients ∇ µq L and ∇ σ 2 q L, which we now derive.

The objective can be written as: DISPLAYFORM1 Plugging in p θ (z) and q(z|x): DISPLAYFORM2 Since expectation and differentiation are linear operators, we can take the expectation and derivative of each term individually.

We can write the log-prior as: DISPLAYFORM0 where n z is the dimensionality of z. We want to evaluate the following terms: DISPLAYFORM1 and DISPLAYFORM2 To take these derivatives, we will use the reparameterization trick to re-express z = µ q + σ q , where ∼ N (0, I) is an auxiliary standard Gaussian variable, and denotes the element-wise product.

We can now perform the expectations over , allowing us to bring the gradient operators inside the expectation brackets.

The first term in eqs. 17 and 18 does not depend on µ q or σ 2 q , so we can write: DISPLAYFORM3 and DISPLAYFORM4 To simplify notation, we define the following term: DISPLAYFORM5 allowing us to rewrite eqs. 19 and 20 as: DISPLAYFORM6 and DISPLAYFORM7 We must now find ∂ξ ∂µq and DISPLAYFORM8 and DISPLAYFORM9 where division is performed element-wise.

Plugging eqs. 24 and 25 back into eqs. 22 and 23, we get: DISPLAYFORM10 and DISPLAYFORM11 Putting everything together, we can express the gradients as: DISPLAYFORM12 and DISPLAYFORM13 A.3 GRADIENT OF THE LOG-APPROXIMATE POSTERIOR We can write the log-approximate posterior as: DISPLAYFORM14 where n z is the dimensionality of z. Again, we will use the reparameterization trick to re-express the gradients.

However, notice what happens when plugging the reparameterized z = µ q + σ q into the second term of eq. 30: DISPLAYFORM15 This term does not depend on µ q or σ 2 q .

Also notice that the first term in eq. 30 depends only on σ 2 q .

Therefore, the gradient of the entire term w.r.t.

µ q is zero: DISPLAYFORM16 The gradient w.r.t.

σ 2 q is DISPLAYFORM17 (33) Note that the expectation has been dropped, as the term does not depend on the value of the sampled z. Thus, the gradient of the entire term w.r.t.

σ 2 q is: DISPLAYFORM18

The form of the conditional likelihood will depend on the data, e.g. binary, discrete, continuous, etc.

Here, we derive the gradient for Bernoulli (binary) and Gaussian (continuous) conditional likelihoods.

Bernoulli Output Distribution The log of a Bernoulli output distribution takes the form: DISPLAYFORM0 where µ x = µ x (z, θ) is the mean of the output distribution.

We drop the explicit dependence on z and θ to simplify notation.

We want to compute the gradients DISPLAYFORM1 and DISPLAYFORM2 Again, we use the reparameterization trick to re-express the expectations, allowing us to bring the gradient operators inside the brackets.

Using z = µ q + σ q , eqs. 36 and 37 become: DISPLAYFORM3 and DISPLAYFORM4 where µ x is re-expressed as function of µ q , σ 2 q , , and θ.

Distributing the gradient operators yields: DISPLAYFORM5 and DISPLAYFORM6 Taking the partial derivatives and combining terms gives: DISPLAYFORM7 and DISPLAYFORM8 Gaussian Output Density The log of a Gaussian output density takes the form: DISPLAYFORM9 where µ x = µ x (z, θ) is the mean of the output distribution and σ 2 x = σ 2 x (θ) is the variance.

We assume σ 2 x is not a function of z to simplify the derivation, however, using σ 2 x = σ 2 x (z, θ) is possible and would simply result in additional gradient terms in ∇ µq L and ∇ σ 2 q L. We want to compute the gradients DISPLAYFORM10 and DISPLAYFORM11 The first term in eqs. 45 and 46 is zero, since σ 2 x does not depend on µ q or σ 2 q .

To take the gradients, we will again use the reparameterization trick to re-express z = µ q + σ q .

We now implicitly express µ x as µ x (µ q , σ 2 q , θ).

We can then write: DISPLAYFORM12 and DISPLAYFORM13 To simplify notation, we define the following term: DISPLAYFORM14 allowing us to rewrite eqs. 47 and 48 as DISPLAYFORM15 and DISPLAYFORM16 We must now find ∂ξ ∂µq and DISPLAYFORM17 and DISPLAYFORM18 Plugging these expressions back into eqs. 50 and 51 gives DISPLAYFORM19 and DISPLAYFORM20 Despite having different distribution forms, Bernoulli and Gaussian output distributions result in approximate posterior gradients of a similar form: the Jacobian of the output model multiplied by a weighted error term.

A.5 SUMMARY Putting the gradient terms from log p θ (x|z), log p θ (z), and log q(z|x) together, we arrive at Bernoulli Output Distribution: DISPLAYFORM21 Gaussian Output Distribution: Figure 7 : Plate notation for a hierarchical latent variable model consisting of L levels of latent variables.

Variables at higher levels provide empirical priors on variables at lower levels.

With data-dependent priors, the model has more flexibility in representing the intricacies of each data example.

DISPLAYFORM22

Hierarchical latent variable models factorize the latent variables over multiple levels, z = {z 1 , z 2 , . . .

, z L }.

Latent variables at higher levels provide empirical priors on latent variables at lower levels.

For an intermediate latent level, we use the notation DISPLAYFORM0 DISPLAYFORM1 .

FORMULA0 Notice that these gradients take a similar form to those of a one-level latent variable model.

The first terms inside each expectation can be interpreted as a "bottom-up" gradient coming from reconstruction errors at the level below.

The second terms inside the expectations can be interpreted as "top-down" errors coming from priors generated by the level above.

The last term in the variance gradient expresses a form of regularization.

Standard hierarchical inference models only contain bottom-up information, and therefore have no way of estimating the second term in each of these gradients.

Equation 5 provides a general form for an iterative inference model.

Here, we provide specific implementation details for these models.

Code for reproducing the experiments will be released online.

As mentioned in Andrychowicz et al. (2016) , gradients can be on vastly different scales, which is undesirable for training neural networks.

To handle this issue, we adopt the technique they proposed: replacing ∇ λ L with the concatenation of [α log(|∇ λ L| + ), sign(∇ λ L)], where α is a scaling constant and is a small constant for numerical stability.

This is performed for both parameters in λ = {µ q , log σ 2 q }.

When encoding the errors, we instead input the concatenation of [ε x , ε z ] (see section 4.1 for definitions of these terms).

As we use global variances on the output and prior densities, we drop σ 2 x and σ 2 p from these expressions because they are constant across all examples.

We also found it beneficial to encode the current estimates of µ q and log σ 2 q .

We end by again noting that encoding gradients or errors over successive iterations can be difficult, as the distributions of these inputs change quickly during both learning and inference.

Work remains to be done in developing iterative encoding architectures that handle this aspect more thoroughly, perhaps through some form of input normalization or saturation.

For the output form of these models, we use a gated updating scheme, sometimes referred to as a "highway" connection (Srivastava et al. (2015) ).

Specifically, approximate posterior parameters are updated according to DISPLAYFORM0 where represents element-wise multiplication and DISPLAYFORM1 is the gating function for λ at time t, which we combine with the iterative inference model f t .

We found that this yielded improved performance and stability over the residual updating scheme used in Andrychowicz et al. (2016) .

In our experiments with latent Gaussian models, we found that means tend to receive updates over many iterations, whereas variances (or log variances) tend to receive far fewer updates, often just a single large update.

Further work could perhaps be done in developing schemes that update these two sets of parameters differently.

We parameterize iterative inference models as neural networks.

Although Andrychowicz et al. (2016) exclusively use recurrent neural networks, we note that optimization models can also be instantiated with feed-forward networks.

Note that even with a feed-forward network, because the entire model is run over multiple iterations, the model is technically a recurrent network, though quite different from the standard RNN formulation.

RNN iterative inference models, through hidden or memory states, are able to account for non-local curvature information, analogous to momentum or other moment terms in conventional optimization techniques.

Feed-forward networks are unable to capture and utilize this information, but purely local curvature information is still sufficient to update the output estimate, e.g. vanilla stochastic gradient descent.

Andrychowicz et al. (2016) propagate optimizer parameter gradients (∇ φ L) from the optimizee's loss at each optimization step, giving each step equal weight.

We take the same approach; we found it aids in training recurrent iterative inference models and is essential for training feed-forward iterative inference models.

With a recurrent model, ∇ φ L is calculated using stochastic backpropagation through time.

With a feedforward model, we accumulate ∇ φ L at each step using stochastic backpropagation, then average over the total number of steps.

The advantage of using a feed-forward iterative inference model is that it maintains a constant memory footprint, as we do not need to keep track of gradients across iterations.

However, as mentioned above, this limits the iterative inference model to only local optimization information.

Overall, we found iterative inference models were not difficult to train.

Almost immediately, these models started learning to improve their estimates.

As noted by Andrychowicz et al. (2016) , some care must be taken to ensure that the input gradients stay within a reasonable range.

We found their log transformation trick to work well in accomplishing this.

We also observed that the level of stochasticity in the gradients played a larger role in inference performance for iterative inference models.

For instance, in the Gaussian case, we noticed a sizable difference in performance between approximating the KL-divergence and evaluating it analytically.

This difference was much less noticeable for standard inference models.

In all experiments, inference model and generative model parameters were learned jointly using the AdaM optimizer (Kingma & Ba FORMULA0 ).

The learning rate was set to 0.0002 for both sets of parameters and all other optimizer parameters were set to their default values.

Learning rates were decayed exponentially by a factor of 0.999 at every epoch.

All models utilized exponential linear unit (ELU) activation functions (Clevert et al. (2015) ), although we found other non-linearities to work as well.

Unless otherwise stated, all inference models were symmetric to their corresponding generative models, with the addition of "highway" connections (Srivastava et al. (2015) ) between hidden layers.

Though not essential, we found that these connections improved stability and performance.

Iterative inference models for all experiments were implemented as feed-forward networks to make comparison with standard inference models easier.

See appendix B for further details.

To visualize the optimization surface and trajectories of latent Gaussian models, we trained models with 2 latent dimensions and a point estimate approximate posterior.

That is, q(z|x) = δ(z = µ q ) is a Dirac delta function at the point µ q = (µ 1 , µ 2 ).

We used a 2D point estimate approximate posterior instead of a 1D Gaussian density because it results in more variety in the optimization surface, making it easier to visualize the optimization.

We trained these models on binarized MNIST due to the data set's relatively low complexity, meaning that 2 latent dimensions can reasonably capture the relevant information specific to a data example.

The generative models consisted of a neural network with 2 hidden layers, each with 512 units.

The output of the generative model was the mean of a Bernoulli distribution, and log p θ (x|z) was evaluated using binary cross-entropy.

KL-divergences were estimated using 1 sample of z ∼ q(z|x).

The optimization surface of each model was evaluated on a grid with range [-5, 5] in increments of 0.05 for each latent variable.

To approximate the MAP estimate, we up-sampled the optimization surface using a cubic interpolation scheme.

FIG0 visualizes the ELBO optimization surface after training for 80 epochs.

Figure 3 visualizes the ELBO optimization surface after training (by encoding x, ε x , and ε z ) for 50 epochs.

For the qualitative results shown in figure 4 , we trained iterative inference models on MNIST, Omniglot, and Street View House Numbers by encoding approximate posterior gradients (∇ λ L) for 16 inference iterations.

For CIFAR-10, we had difficulty in obtaining sharp reconstructions in a reasonable number of inference iterations, so we trained an iterative inference model by encoding errors for 10 inference iterations.

For binarized MNIST and Omniglot, we used a generative model architecture with 2 hidden layers, each with 512 units, a latent space of size 64, and a symmetric iterative inference model, with the addition of highway connections at each layer.

For Street View House Numbers and CIFAR-10, we used 3 hidden layers in the iterative inference and 1 in the generative model, with 2048 units at each hidden layer and a latent space of size 1024.

We used the same architecture of 2 hidden layers, each with 512 units, for the output model and inference models.

The latent variables consisted of 64 dimensions.

Each model was trained by drawing the corresponding number of samples from the approximate posterior distribution using the reparameterization trick, yielding lower variance ELBO estimates and gradients.

Iterative inference models were trained by encoding the data (x) and the approximate posterior gradients (∇ λ L) for 5 inference iterations.

All models were trained for 1,500 epochs.

The model architecture for all encoding schemes was identical to that used in the previous section.

All models were trained by evaluating the ELBO with a single approximate posterior sample.

We trained all models for 1,500 epochs.

We were unable to run multiple trials for each experimental set-up, but on a subset of runs for standard and iterative inference models, we observed that final performance had a standard deviation less than 0.1 nats, below the difference in performance between models trained with different numbers of inference iterations.

Directly comparing inference optimization performance between inference techniques is difficult; inference estimates affect learning, resulting in models that are better suited to the inference scheme.

Instead, to quantitatively compare the performance between standard and iterative inference models, we trained models with the same architecture using each inference model form.

We trained both one-level and hierarchical models on MNIST and one-level models on CIFAR-10.

In each case, iterative inference models were trained by encoding the data and errors for 5 inference iterations.

We estimated marginal log-likelihoods for each model using 5,000 importance weighted samples per data example.

C.5.1 MNIST For MNIST, one-level models consisted of a latent variable of size 64, and the inference and generative networks both consisted of 2 hidden layers, each with 512 units.

Hierarchical models consisted of 2 levels with latent variables of size 64 and 32 in hierarchically ascending order.

At each level, the inference and generative networks consisted of 2 hidden layers, with 512 units at the first level and 256 units at the second level.

At the first level of latent variables, we also used a set of deterministic units, also of size 64, in both the inference and generative networks.

Hierarchical models included batch normalization layers at each hidden layer of the inference and generative networks; we found this beneficial for training both standard and iterative inference models.

Both encoder and decoder networks in the hierarchical model utilized highway skip connections at each layer at both levels.

For CIFAR-10, models consisted of a latent variable of size 1024, an encoder network with 3 hidden layers of 2048 units with highway connections, and a decoder network with 1 hidden layer with 2048 units.

The variance of the output Gaussian distribution was a global variable for this model.

We note that the results reported in table 1 are significantly worse than those typically reported in the literature, however these results are for relatively small fully-connected networks rather than larger convolutional networks.

We also experimented with hierarchical iterative inference models on CIFAR-10, but found these models more difficult to train without running into numerical instabilities.

C.6 COMPARISON WITH VARIATIONAL EM Variational EM is not typically used in practice, as it does not scale well with large models or large data sets.

However, because iterative inference models iteratively optimize the approximate posterior parameters, we felt it would be beneficial to provide a comparison of inference optimization performance between iterative inference models and expectation steps from variational EM.

We used one-level latent Gaussian models trained with iterative inference models on MNIST for 16 iterations.

We compared against vanilla SGD, SGD with momentum, RMSProp, and AdaM, trying learning rates in {0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001}. In all comparisons, we found that iterative inference models outperformed conventional optimization techniques by large margins.

Figure 6 shows the optimization performance on the test set for all optimizers and an iterative inference model trained by encoding the approximate posterior gradients.

The iterative inference model quickly arrives at a stable approximate posterior estimate, outperforming all optimizers.

It is important to note that the iterative inference model here actually has less derivative information than the adaptive optimizers; it only has access to the local gradient.

Also, despite only being trained using 16 iterations, the iterative inference remains stable for hundreds of iterations.

We also compared the optimization techniques on the basis of wall clock time: FIG0 reproduces the results from figure 6.

We observe that, despite requiring more time per inference iteration, the iterative inference model still outperforms the conventional optimization techniques.

Concurrent with our work, Krishnan et al. (2017) propose closing the amortization gap by performing inference optimization steps after initially encoding the data with a standard inference model, reporting substantial gains on sparse, high-dimensional data, such as text and ratings.

We observe similar findings and present a confirmatory experimental result on the RCV1 data set (Lewis et al. (2004) ), which consists of 10,000 dimensions containing word counts.

We follow the same processing procedure as Krishnan et al. (2017) , encoding data using normalized TF-IDF features and modeling the data using a multinomial distribution.

For encoder and decoder, we use 2-layer networks, each with 512 units and ELU non-linearities.

We use a latent variable of size 512 as well.

The iterative inference model was trained by encoding gradients for 16 steps.

We evaluate the models by reporting (an upper bound on) perplexity on the test set TAB1 .

Perplexity, P , is defined as DISPLAYFORM0 where N is the number of examples and N i is the total number of word counts in example i.

We evaluate perplexity by estimating each log p(x i ) with 5,000 importance weighted samples.

We observe that iterative inference models outperform standard inference models on this data set by a similar margin reported by Krishnan et al. (2017) .

Note, however, that iterative inference models here have substantially fewer input parameters than standard inference models (2,048 vs. 10,000).

We also run a single optimization procedure for an order of magnitude fewer steps than that of Krishnan et al. (2017) .In FIG0 , we further illustrate the optimization capabilities of the iterative inference model used here.

Plotting the average gradient magnitude of the approximate posterior for inference iterations in FIG0 , we see that over successive iterations, the magnitude decreases.

This implies that the model is capable of arriving at near-optimal estimates, where the gradient is close to zero.

In FIG0 , we plot the average relative improvement in the ELBO over inference iterations.

We see that the model is quickly able to improve its inference estimates, eventually reaching a relative improvement of roughly 25%.

<|TLDR|>

@highlight

We propose a new class of inference models that iteratively encode gradients to estimate approximate posterior distributions.