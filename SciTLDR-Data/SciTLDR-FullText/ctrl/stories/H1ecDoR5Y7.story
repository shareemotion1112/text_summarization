Wasserstein GAN(WGAN) is a model that minimizes the Wasserstein distance between a data distribution and sample distribution.

Recent studies have proposed stabilizing the training process for the WGAN and implementing the Lipschitz constraint.

In this study, we prove the local stability of optimizing the simple gradient penalty $\mu$-WGAN(SGP $\mu$-WGAN) under suitable assumptions regarding the equilibrium and penalty measure $\mu$. The measure valued differentiation concept is employed to deal with the derivative of the penalty terms, which is helpful for handling abstract singular measures with lower dimensional support.

Based on this analysis, we claim that penalizing the data manifold or sample manifold is the key to regularizing the original WGAN with a gradient penalty.

Experimental results obtained with unintuitive penalty measures that satisfy our assumptions are also provided to support our theoretical results.

Deep generative models reached a turning point after generative adversarial networks (GANs) were proposed by BID2 .

GANs are capable of modeling data with complex structures.

For example, DCGAN can sample realistic images using a convolutional neural network (CNN) structure BID12 .

GANs have been implemented in many applications in the field of computer vision with good results, such as super-resolution, image translation, and text-to-image generation BID7 BID6 Zhang et al., 2017; BID13 .However, despite these successes, GANs are affected by training instability and mode collapse problems.

GANs often fail to converge, which can result in unrealistic fake samples.

Furthermore, even if GANs successfully synthesize realistic data, the fake samples exhibit little variability.

A common solution to this instability problem is injecting an instance noise and finding different divergences.

The injection of instance noise into real and fake samples during the training procedure was proposed by Sønderby et al. (2017) , where its positive impact on the low dimensional support for the data distribution was shown to be a regularizing factor based on the Wasserstein distance, as demonstrated analytically by .

In f -GAN, f -divergence between the target and generator distributions was suggested which generalizes the divergence between two distributions BID11 .

In addition, a gradient penalty term which is related with Sobolev IPM(Integral Probability Metric) between data distribution and sample distribution was suggested by BID9 .The Wasserstein GAN (WGAN) is known to resolve the problems of generic GANs by selecting the Wasserstein distance as the divergence .

However, WGAN often fails with simple examples because the Lipschitz constraint on discriminator is rarely achieved during the optimization process and weight clipping.

Thus, mimicking the Lipschitz constraint on the discriminator by using a gradient penalty was proposed by BID3 .Noise injection and regularizing with a gradient penalty appear to be equivalent.

The addition of instance noise in f -GAN can be approximated to adding a zero centered gradient penalty BID14 .

Thus, regularizing GAN with a simple gradient penalty term was suggested by BID8 who provided a proof of its stability.

Based on a theoretical analysis of the dynamic system, BID10 proved the local exponential stability of the gradient-based optimization dynamics in GANs by treating the simultaneous gradient descent algorithm with a dynamic system approach.

These previous studies were useful because they showed that the local behavior of GANs can be explained using dynamic system tools and the related Jacobian's eigenvalues.

In this study, we aim to prove the convergence property of the simple gradient penalty µ-Wasserstein GAN(SGP µ-WGAN) dynamic system under general gradient penalty measures µ. To the best of our knowledge, our study is the first theoretical approach to GAN stability analysis which deals with abstract singular penalty measure.

In addition, measure valued differentiation BID4 ) is applied to take the derivative on the integral with a parametric measure, which is helpful for handling an abstract measure and its integral in our proof.

The main contributions of this study are as follows.• We prove the regularized effect and local stability of the dynamic system for a general penalty measure under suitable assumptions.

The assumptions are written as both a tractable strong version and intractable weak version.

To prove the main theorem, we also introduce the measure valued differentiation concept to handle the parametric measure.• Based on the proof of the stability, we explain the reason for the success of previous penalty measures.

We claim that the support of a penalty measure will be strongly related to the stability, where the weight on the limiting penalty measure might affect the speed of convergence.• We experimentally examined the general convergence results by applying two test penalty measures to several examples.

The proposed test measures are unintuitive but they still satisfy the assumptions and similar convergence results were obtained in the experiment.

First, we introduce our notations and basic measure-theoretic concepts.

Second, we define our SGP µ-WGAN optimization problem and treat this problem as a continuous dynamic system.

Preliminary measure theoretic concepts are required to justify that the dynamic system changes in a sufficiently smooth manner as the parameter changes, so it is possible to use linearization theorem.

They are also important for dealing with the parametric measure and its derivative.

The problem setting with a simple gradient term is also discussed.

The squared gradient size and simple gradient penalty term are used to build a differentiable dynamic system and to apply soft regularization as a resolving constraint, respectively.

The continuous dynamic system approach, which is a so-called ODE method, is used to analyze the GAN optimization problem with the simultaneous gradient descent algorithm, as described by Nagarajan & Kolter (2017).

D(x; ψ) : X → R is a discriminator function with its parameter ψ and G(z; θ) : Z → X is a generator function with its parameter θ.

p d is the distribution of real data and p g = p θ is the distribution of the generated samples in X , which is induced from the generator function G(z; θ) and a known initial distribution p latent (z) in the latent space Z. · denotes the L 2 Euclidean norm if no special subscript is present.

The concept of weak convergence for finite measures is used to ensure the continuity of the integral term over the measure in the dynamic system, which must be checked before applying the theorems related to stability.

Throughout this study, we assume that the measures in the sample space are all finite and bounded.

Definition 1.

For a set of finite measures {µ i } i∈I in (R n , d) with euclidean distance d, {µ i } i∈I is referred to as bounded if there exists some M > 0 such that for all i ∈ I, DISPLAYFORM0 For instance, M can be set as 1 if {µ i } are probability measures on R n .

Assuming that the penalty measures are bounded, Portmanteau theorem offers the equivalent definition of the weak conver-gence for finite measures.

This definition is important for ensuring that the integrals over p θ and µ in the dynamic system change continuously.

Definition 2. (Portmanteau Theorem) For a bounded sequence of finite measures {µ n } n∈N on the Euclidean space R n with a σ-field of Borel subsets B(R n ), µ n converges weakly to µ if and only if for every continuous bounded function φ on R n , its integrals with respect to µ n converge to φdµ, i.e., DISPLAYFORM1 The most challenging problem in our analysis with the general penalty measure is taking the derivative of the integral, where the measure depends on the variable that we want to differentiate.

If our penalty measure is either absolutely continuous or discrete, then it is easy to deal with the integral.

However, in the case of singular penalty measure, dealing with the integral term is not an easy task.

Therefore, we introduce the concept of a weak derivative of a probability measure in the following BID4 DISPLAYFORM2 , makes the dynamic system differentiable and we define the WGAN problem with the square of the gradient's norm as a simple gradient penalty.

This simple gradient penalty can be treated as soft regularization based on the size of the discriminator's gradient, especially in case where µ is the probability measure BID14 .

It is convenient to determine whether the system is stable by observing the spectrum of the Jacobian matrix.

In the following, (D(x; ψ), p d , p θ , µ) is defined as an SGP µ-WGAN optimization problem (SGP-form) with a simple gradient penalty term on the penalty measure µ.Definition 4.

The WGAN optimization problem with a simple gradient penalty term ∇ x D 2 , penalty measure µ, and penalty weight hyperparameter ρ > 0 is given as follows, where the penalty term is only introduced to update the discriminator.

DISPLAYFORM3 According to Nagarajan & Kolter (2017) and many other optimization problem studies, the simultaneous gradient descent algorithm for GAN updating can be viewed as an autonomous dynamic system of discriminator parameters and generator parameters, which we denote as ψ and θ.

As a result, the related dynamic system is given as follows.

DISPLAYFORM4

We investigate two examples considered in previous studies by BID8 and BID10 .

We then generalize the results to a finite measure case.

The first example is the univariate Dirac GAN, which was introduced by BID8 .

DISPLAYFORM0 The Dirac GAN with a gradient penalty with an arbitrary probability measure is known to be globally convergent BID8 .

We argue that this result can be generalized to a finite penalty measure case.

Lemma 1.

Consider the Dirac GAN problem with SGP form (D(x; ψ) = ψx, δ 0 , δ θ , µ ψ,θ ).

Suppose that some small η > 0 exists such that its finite penalty measure µ ψ,θ with mass M (ψ, θ) = 1dµ ψ,θ ≥ 0 satisfies either DISPLAYFORM1 Then, the SGP µ-WGAN optimization dynamics with (D(x; ψ) = ψx, δ 0 , δ θ , µ ψ,θ ) are locally stable at the origin and the basin of attraction B = B R ( (0, 0) ) is open ball with radius R. Its radius is given as follows.

DISPLAYFORM2 Motivated by this example, we can extend this idea to the other toy example given by BID10 , where WGAN fails to converge to the equilibrium points (ψ, θ) = (0, ±1).1 In this study, we prefer to use the expectation notation on the finite measure, which can be understood as follows.

Suppose that µ ψ,θ = M (ψ, θ)μ ψ,θ whereμ ψ,θ is normalized to the probability measure.

Then, DISPLAYFORM3 Lemma 2.

Consider the toy example (D(x; ψ) = ψx 2 , U (−1, 1), U (−|θ|, |θ|), µ θ ) where U (0, 0) = δ 0 and the ideal equilibrium points are given by (ψ * , θ * ) = (0, ±1).

For a finite measure µ = µ θ on R which is independent of ψ, suppose that µ θ → µ * with µ * = Cδ 0 for C ≥ 0.

The dynamic system is locally stable near the desired equilibrium (0, ±1), where the spectrum of the DISPLAYFORM4

We propose the convergence property of WGAN with a simple gradient penalty on an arbitrary penalty measure µ for a realizable case: θ = θ * with p d = p θ * exists.

In subsection 4.1, we provide the necessary assumptions, which comprise our main convergence theorem.

In subsection 4.2, we give the main convergence theorem with a sketch of the proof.

A more rigorous analysis is given in the Appendix.

The first assumption is made regarding the equilibrium condition for GANs, where we state the ideal conditions for the discriminator parameter and generator parameter.

As the parameters converge to the ideal equilibrium, the sample distribution(p θ ) converges to the real data distribution(p d ) and the discriminator cannot distinguish the generated sample and the real data.

DISPLAYFORM0 The second assumption ensures that the higher order terms cannot affect the stability of the SGP µ-WGAN.

In the Appendix, we consider the case where the WGAN fails to converge when Assumption 2 is not satisfied.

Compared with the previous study by BID10 , the conditions for the discriminator parameter are slightly modified.

Assumption 2.

DISPLAYFORM1 are locally constant along the nullspace of the Hessian matrix.

The third assumption allows us to extend our results to discrete probability distribution cases, as described by BID8 .

DISPLAYFORM2 The fourth assumption indicates that there are no other "bad" equilibrium points near (ψ * , θ * ), which justifies the projection along the axis perpendicular to the null space.

Assumption 4.

A bad equilibrium does not exist near the desired equilibrium point.

Thus, (ψ DISPLAYFORM3 The last assumption is related to the necessary conditions for the penalty measure.

A calculation of the gradient penalty based on samples from the data manifold and generator manifold or the interpolation of both was introduced in recent studies BID3 BID14 BID8 .

First, we propose strong conditions for the penalty measure.

Assumption 5.

The finite penalty measure µ = µ θ satisfies the followings: a µ θ → µ θ * = µ * and µ θ is independent of the discriminator parameter ψ.

DISPLAYFORM4 The assumption given above means that the support of the penalty measure µ θ should approach the data manifolds smoothly as θ → θ * .

However, the penalty measure from WGAN-GP with a simple gradient penalty still reaches equilibrium without satisfying Assumption 5c.

Therefore, we suggest Assumption 6, which is a weak version of Assumption 5.

Assumption 6a 2 is technically required to take the derivative of the integral DISPLAYFORM5 2 ] with respect to ψ.

Assumption 6. (Weak version of Assumption 5) The finite penalty measure µ = µ ψ,θ satisfies the following.a µ ψ,θ → µ ψ * ,θ * = µ * , where supp(µ ψ,θ ) only depends on θ.

Near the equilibrium, µ ψ,θ can be weakly differentiated twice with respect to ψ.

In addition, its mass M (ψ, θ) = 1dµ ψ,θ is a twice-differentiable function of ψ and bounded near the equilibrium.

DISPLAYFORM6 The assumption above implies the following situations; The penalty measure's support approaches to data manifold and its weight changes smoothly with respect to ψ and θ.

At the equilibrium, penalty measure's support contains data manifold.

Also, ideal discriminator will remain flat on the penalty area.

In summary, the gradient penalty regularization term with any penalty measure where the support approaches B(supp(p d )) in a smooth manner works well and this main result can explain the regularization effect of previously proposed penalty measures such as µ GP , p d , p θ , and their mixtures.

According to the modified assumptions given above, we prove that the related dynamic system is locally stable near the equilibrium.

The tools used for analyzing stability are mainly based on those described by Nagarajan & Kolter (2017).

Our main contributions comprise proposing the necessary conditions for the penalty measure and proving the local stability for all penalty measures that satisfy Assumption 6.

Theorem 1.

Suppose that our SGP µ-WGAN optimization problem (D, p d , p θ , µ) with equilibrium point (ψ * , θ * ) satisfies the assumptions given above.

Then, the related dynamic system is locally stable at the equilibrium.

A detailed proof of the main convergence theorem is given in the Appendix.

A sketch of the proof is given in three steps.

First, the undesired terms in the Jacobian matrix of the system at the equilibrium are cancelled out.

Next, the Jacobian matrix at equilibrium is given by DISPLAYFORM0 The system is locally stable when both Q and R T R are positive definite.

We can complete the proof by dealing with zero eigenvalues by showing that N (Q T ) ⊂ N (R T ) and the projected system's stability implies the original system's stability.

Our analysis mainly focuses on WGAN, which is the simplest case of general GAN minimax optimization max DISPLAYFORM1 with f (x) = x. Similar approach is still valid for general GANs with concave function f with f (x) < 0 and f (0) = 0.

We claim that every penalty measure that satisfies the assumptions can regularize the WGAN and generate similar results to the recently proposed gradient penalty methods.

Several penalty measures were tested based on two-dimensional problems (mixture of 8 Gaussians, mixture of 25 Gaussians, and swissroll), MNIST and CIFAR-10 datasets using a simple gradient penalty term.

In the comparisons with WGAN, the recently proposed penalty measures and our test penalty measures used the same network settings and hyperparameters.

The penalty measures and its detailed sampling methods are listed in TAB1 , where DISPLAYFORM0 , and α ∼ U (0, 1).

A indicates fixed anchor point in X .

BID3 , which penalizes the WGAN with non-zero centered gradient penalty terms, whereas µ GP represents the simple method.

In our experiment, no additional weights are applied on 5 penalty measures and they are all probability distributions.

Penalty term Penalty measure, sampling method DISPLAYFORM0 By setting the previously proposed WGAN with weight-clipping and WGAN-GP BID3 as the baseline models, SGP µ-WGAN was examined with various penalty measures comprising three recently proposed measures and two artificially generated measures.

p θ and p d were suggested by BID8 and µ GP was introduced from the WGAN-GP.

We analyzed the artificial penalty measures µ mid and µ g,anc as the test penalty measures.

The experiments were conducted based on the implementation of the BID3 .

The hyperparameters, generator/discriminator structures, and related TensorFlow implementations can be found at https://github.com/igul222/improved_wgan_training BID3 .

Only the loss function was modified slightly from a non-zero centered gradient penalty to a simple penalty.

For the CIFAR-10 image generation tasks, the inception score BID15 and FID BID5 were used as benchmark scores to evaluate the generated images.

We checked the convergence of p θ for the 2D examples (8 Gaussians, swissroll data, and 25 Gaussians) and MNIST digit generation for the SGP-WGANs with five penalty measures.

MNIST and 25 Gaussians were trained over 200K iterations, the 8 Gaussians were trained for 30K iterations, and the Swiss Roll data were trained for 100K iterations.

The anchor A for µ g,anc was set as (2, −1) for the 2D examples and 784 gray pixels for MNIST.

We only present the results obtained for the MNIST dataset with the penalty measures comprising µ mid and µ g,anc in Figure 1 .

The others are presented in the Appendix.

Figure 1: MNIST example.

Images generated with µ mid (left) and µ g,anc (right).

DCGAN and ResNet architectures were tested on the CIFAR-10 dataset.

The generators were trained for 200K iterations.

The anchor A for µ g,anc during CIFAR-10 generation was set as fixed random pixels.

The WGAN, WGAN-GP, and five penalty measures were evaluated based on the inception score and FID, as shown in TAB2 , which are useful tools for scoring the quality of generated images.

The images generated from µ mid and µ g,anc with ResNet are shown in FIG0 .

The others are presented in the Appendix.

In this study, we proved the local stability of simple gradient penalty µ-WGAN optimization for a general class of finite measure µ.

This proof provides insight into the success of regularization with previously proposed penalty measures.

We explored previously proposed analyses based on various gradient penalty methods.

Furthermore, our theoretical approach was supported by experiments using unintuitive penalty measures.

In future research, our works can be extended to alternative gradient descent algorithm and its related optimal hyperparameters.

Stability at non-realizable equilibrium points is one of the important topics on stability of GANs.

Optimal penalty measure for achieving the best convergence speed can be also investigated using a spectral theory, which provides the mathematical analysis on stability of GAN with a precise information on the convergence theory.

Proof of Lemma 1.

The related dynamic system of (D(x; ψ) = ψx, δ 0 , δ θ , µ ψ,θ ) can be written as follows.ψ DISPLAYFORM0 First, the only equilibrium point is given by (ψ * , θ * ) = (0, 0) from DISPLAYFORM1 The corresponding Jacobian matrix for the dynamic system is written as: DISPLAYFORM2 ∇ ψ D(x; ψ) = ψ does not depend on x, so this can be rewritten as: DISPLAYFORM3 Therefore, if M (0, 0) > 0, then the given system is locally stable because the eigenvalues of its linearized system have negative real parts.

If M (0, 0) = 0, then the stability of the system cannot be proved by the linearization theorem.

In this case, we consider the following Lyapunov function.

DISPLAYFORM4 By differentiating with t, we obtaiṅ DISPLAYFORM5 Clearly, L(ψ, θ) ≥ 0 and the equality holds iff 0) ) for all τ ≥ 0 because the Lyapunov function (square of the distance between the origin and (ψ(τ ), θ(τ ))) always decreases as τ → ∞. Therefore, the given system is stable according to the Lyapunov stability theorem.

DISPLAYFORM6 Again, we can check that if µ ψ,θ is a probability measure, then the system is globally stable, as shown by BID8 .

The basin of attraction is given by the whole R 2 plane since DISPLAYFORM7 Proof of Lemma 2.

From the general setup of the SGP µ-WGAN optimization problem, the dynamic system corresponding to the simple-GAN in Definition 6 can be written as follows.

DISPLAYFORM8 If we let E µ * [x 2 ] = A 2 , then the Jacobian matrix at the equilibrium (0, ±1) is given by J = −4ρA 2 ∓ 2 3 ± 2 3 0 .

Therefore, the given system is locally stable when A = 0.

Lemma 3.

Consider the Dirac-GAN setup and SGP µ-WGAN optimization system with a slightly changed discriminator function D 2 (x; ψ) = ψx 2 .

The system (D 2 , δ 0 , δ θ , µ GP ) does not converge to (0, 0) but for any point (a, 0) with a < 0, the system has equilibrium points on the whole ψ-axis and it violates Assumption 2.Proof of Lemma 3.

For the SGP µ-WGAN optimization problem (D 2 , δ 0 , δ θ , µ GP ), the dynamic system can be written as follows.ψ = −θ 2 − 4 3 ρψθ

Proof.

Let us consider the Jacobian matrix DISPLAYFORM0 First, Assumption 1 implies that DISPLAYFORM1 is locally zero near the equilibrium θ * , which implies that DISPLAYFORM2 We still need to evaluate DISPLAYFORM3 According to Assumption 6a, finite signed measures µ ψ,θ and µ ψ,θ exist 5 , so they are the first and second weak derivatives of µ ψ,θ with respect to the parameter ψ at (ψ * , θ * ).

Therefore, the expectations given above can be rewritten as below.

DISPLAYFORM4 where DISPLAYFORM5 From Assumption 6c and the fact that the weak derivative of µ ψ,θ vanishes outside of supp(µ ψ,θ ), ∇ x D(x; ψ * ) = 0 on supp(µ ψ,θ ) ⊂ V for all θ with |θ − θ * | < µ and µ ψ,θ = µ ψ,θ = 0 on the outside of supp(µ ψ,θ ), which leads to the desired results: DISPLAYFORM6 After cancelling the undesired terms, the Jacobian matrix at the equilibrium (ψ * , θ * ) is given as: DISPLAYFORM7 In standard notation, ∇ ψ g is the dim(range of g) × dim(ψ) matrix.

For a real-valued function f , we consider the first derivative as the column vector instead of the row vector.

∇ ψ f is considered to be the dim(ψ) × 1 matrix(column vector) of the total derivative.

For the second derivative, ∇ ψθ f = (∇ ψ )(∇ θ f ) is the dim(θ) × dim(ψ) matrix.

The transpose notation is used in a similar manner to the matrix.

5 µ ψ,θ and µ ψ,θ will be considered as row vector(1 × dim(ψ) matrix) and dim(ψ) × dim(ψ) matrix of finite signed measures respectively.

DISPLAYFORM8 that for a negative definite matrix A and full column rank matrix B, the block matrix A B −B T 0 is Hurwitz, i.e., all eigenvalues of the matrix have a negative real part.

Therefore, if Q is positive definite and R is full column rank, the proof is complete.

We consider the complementary case.

Suppose that Q or R T R have some zero eigenvalues.

DISPLAYFORM9 , where T D and T G are the eigenvectors of Q and R T R that correspond to non-zero eigenvalues.

First, we assume that T D and T G are not empty.

We can show that (ψ * + ξv, θ * + νw) is also an equilibrium point for a sufficiently small ξ, ν and v ∈ N (Q), w ∈ N (R T R) by using the techniques given by BID10 .

If the system does not update at the equilibrium point (ψ * , θ * ) and its small neighborhood (ψ * + ξv, θ * + νw) is perturbed along N (Q) and N (R T R), then it is reasonable to project the system orthogonal to N (Q) and N (R T R).First, we assume that v ∈ N (Q).

By Assumption 2, h(ψ * + ξv) = h(ψ * ) = 0 for |ξ| < ξ d , which implies that ∇ x D(x; ψ * + ξv) = 0 for x ∈ supp(µ ψ * +ξv,θ * ) = supp(µ * ) and |ξ| < ξ d .

Thus, we obtain Therefore, the point (ψ * + ξv, θ * ) with |ξ| < ξ d is an equilibrium point.

According to Assumption 4, D(x; ψ * + ξv) is an equilibrium discriminator for |ξ| < δ d , and thus D(x; ψ * + ξv) is already an optimal discriminator for |ξ| < min(ξ d , δ d ).Suppose that w ∈ N (R T R).

By Assumption 2, g(θ * ) = g(θ * + νw) = 0 for |ν| < ν g , and thus E p d [∇ ψ D(x; ψ * )] − E p θ * +νw [∇ ψ D(x; ψ * )] = 0 for |ν| < ν g .

Furthermore, Assumption 3 gives E p θ * +νw [D(x; ψ * )] = 0 for a sufficiently close |ν| < g , which implies thaṫ DISPLAYFORM10

<|TLDR|>

@highlight

This paper deals with stability of simple gradient penalty $\mu$-WGAN optimization by introducing a concept of measure valued differentiation.

@highlight

WGAN with a squared zero centered gradient penalty term w.r.t. to a general measure is studied.

@highlight

Characterizes the convergence of gradient penalized Wasserstein GAN.