Generative Adversarial Networks (GANs) have been proposed as an approach to learning generative models.

While GANs have demonstrated promising performance on multiple vision tasks, their learning dynamics are not yet well understood, neither in theory nor in practice.

In particular, the work in this domain has been focused so far only on understanding the properties of the stationary solutions that this dynamics might converge to, and of the behavior of that dynamics in this solutions’ immediate neighborhood.



To address this issue, in this work we take a first step towards a principled study of the GAN dynamics itself.

To this end, we propose a model that, on one hand, exhibits several of the common problematic convergence behaviors (e.g., vanishing gradient, mode collapse, diverging or oscillatory behavior), but on the other hand, is sufficiently simple to enable rigorous convergence analysis.



This methodology enables us to exhibit an interesting phenomena: a GAN with an optimal discriminator provably converges, while guiding the GAN training using only a first order approximation of the discriminator leads to unstable GAN dynamics and mode collapse.

This suggests that such usage of the first order approximation of the discriminator, which is a de-facto standard in all the existing GAN dynamics, might be one of the factors that makes GAN training so challenging in practice.

Additionally, our convergence result constitutes the first rigorous analysis of a dynamics of a concrete parametric GAN.

Generative modeling is a fundamental learning task of growing importance.

As we apply machine learning to increasingly sophisticated problems, we often aim to learn functions with an output domain that is significantly more complex than simple class labels.

Common examples include image "translation" BID12 , speech synthesis BID16 , and robot trajectory prediction BID6 .

Due to progress in deep learning, we now have access to powerful architectures that can represent generative models over such complex domains.

However, training these generative models is a key challenge.

Simpler learning problems such as classification have a clear notion of "right" and "wrong," and the approaches based on minimizing the corresponding loss functions have been tremendously successful.

In contrast, training a generative model is far more nuanced because it is often unclear how "good" a sample from the model is.

Generative Adversarial Networks (GANs) have recently been proposed to address this issue BID8 .

In a nutshell, the key idea of GANs is to learn both the generative model and the loss function at the same time.

The resulting training dynamics are usually described as a game between a generator (the generative model) and a discriminator (the loss function).

The goal of the generator is to produce realistic samples that fool the discriminator, while the discriminator is trained to distinguish between the true training data and samples from the generator.

GANs have shown promising results on a variety of tasks, and there is now a large body of work that explores the power of this framework BID9 .Unfortunately, reliably training GANs is a challenging problem that often hinders further research in this area.

Practitioners have encountered a variety of obstacles such as vanishing gradients, mode collapse, and diverging or oscillatory behavior BID9 .

At the same time, the theoretical underpinnings of GAN dynamics are not yet well understood.

To date, there were no convergence proofs for GAN models, even in very simple settings.

As a result, the root cause of frequent failures of GAN dynamics in practice remains unclear.

In this paper, we take a first step towards a principled understanding of GAN dynamics.

Our general methodology is to propose and examine a problem setup that exhibits all common failure cases of GAN dynamics while remaining sufficiently simple to allow for a rigorous analysis.

Concretely, we introduce and study the GMM-GAN: a variant of GAN dynamics that captures learning a mixture of two univariate Gaussians.

We first show experimentally that standard gradient dynamics of the GMM-GAN often fail to converge due to mode collapse or oscillatory behavior.

Interestingly, this also holds for techniques that were recently proposed to improve GAN training such as unrolled GANs (Metz et al., 2017 ).

In contrast, we then show that GAN dynamics with an optimal discriminator do converge, both experimentally and provably.

To the best of our knowledge, our theoretical analysis of the GMM-GAN is the first global convergence proof for parametric and non-trivial GAN dynamics.

Our results show a clear dichotomy between the dynamics arising from applying simultaneous gradient descent and the one that is able to use an optimal discriminator.

The GAN with optimal discriminator provably converges from (essentially) any starting point.

On the other hand, the simultaneous gradient GAN empirically often fails to converge, even when the discriminator is allowed many more gradient steps than the generator.

These findings go against the common wisdom that first order methods are sufficiently strong for all deep learning applications.

By carefully inspecting our models, we are able to pinpoint some of the causes of this, and we highlight a phenomena we call discriminator collapse which often causes first order methods to fail in our setting.

Generative adversarial networks are commonly described as a two player game BID8 .

Given a true distribution P , a set of generators G = {G u , u ∈ U}, a set of discriminators D = {D v , v ∈ V}, and a monotone measuring function m : R → R, the objective of GAN training is to find a generator u in arg min DISPLAYFORM0 In other words, the game is between two players called the generator and discriminator, respectively.

The goal of the discriminator is to distinguish between samples from the generator and the true distribution.

The goal of the generator is to fool the discriminator by generating samples that are similar to the data distribution.

By varying the choice of the measuring function and the set of discriminators, one can capture a wide variety of loss functions.

Typical choices that have been previously studied include the KL divergence and the Wasserstein distance BID8 BID1 .

This formulation can also encode other common objectives: most notably, as we will show, the total variation distance.

To optimize the objective (1), the most common approaches are variants of simultaneous gradient descent on the generator u and the discriminator v. But despite its attractive theoretical grounding, GAN training is plagued by a variety of issues in practice.

Two major problems are mode collapse and vanishing gradients.

Mode collapse corresponds to situations in which the generator only learns a subset (a few modes) of the true distribution P BID9 .

For instance, a GAN trained on an image modeling task would only produce variations of a small number of images.

Vanishing gradients BID1 BID1 are, on the other hand, a failure case where the generator updates become vanishingly small, thus making the GAN dynamics not converge to a satisfying solution.

Despite many proposed explanations and approaches to solve the vanishing gradient problem, it is still often observed in practice BID9 .

GANs provide a powerful framework for generative modeling.

However, there is a large gap between the theory and practice of GANs.

Specifically, to the best of the authors' knowledge, all theoretical studies of GAN dynamics for parametric models simply consider global optima and stationary points of the dynamics, and there has been no rigorous study of the actual GAN dynamics.

In practice, GANs are always optimized using first order methods, and the current theory of GANs cannot tell us whether or not these methods converge to a meaningful solution.

This raises a natural question, also posed as an open problem in BID9 :Our theoretical understanding of GANs is still fairly poor.

In particular, to the best of the authors' knowledge, all existing analyzes of GAN dynamics for parametric models simply consider global optima and stationary points of the dynamics.

There has been no rigorous study of the actual GAN dynamics, except studying it in the immediate neighborhood of such stationary points BID15 .

This raises a natural question:Can we understand the convergence behavior of GANs?This question is difficult to tackle for many reasons.

One of them is the non-convexity of the GAN objective/loss function, and of the generator and discriminator sets.

Another one is that, in practice, GANs are always optimized using first order methods.

That is, instead of following the "ideal" dynamics that has both the generator and discriminator always perform the optimal update, we just approximate such updates by a sequence of gradient steps.

This is motivated by the fact that computing such optimal updates is, in general, algorithmically intractable, and adds an additional layer of complexity to the problem.

In this paper, we want to change this state of affairs and initiate the study of GAN dynamics from an algorithmic perspective.

Specifically, we pursue the following question:What is the impact of using first order approximation on the convergence of GAN dynamics?Concretely, we focus on analyzing the difference between two GAN dynamics: a "first order" dynamics, in which both the generator and discriminator use first order updates; and an "optimal discriminator" dynamics, in which only the generator uses first order updates but the discriminator always makes an optimal update.

Even the latter, simpler dynamics has proven to be challenging to understand.

Even the question of whether using the optimal discriminator updates is the right approach has already received considerable attention.

In particular, BID1 present theoretical evidence that using the optimal discriminator at each step may not be desirable in certain settings (although these settings are very different to the one we consider in this paper).We approach our goal by defining a simple GAN model whose dynamics, on one hand, captures many of the difficulties of real-world GANs but, on the other hand, is still simple enough to make analysis possible.

We then rigorously study our questions in the context of this model.

Our intention is to make the resulting understanding be the first step towards crystallizing a more general picture.

Perhaps a tempting starting place for coming up with a simple but meaningful set of GAN dynamics is to consider the generators being univariate Gaussians with fixed variance.

Indeed, in the supplementary material we give a short proof that simple GAN dynamics always converge for this class of generators.

However, it seems that this class of distributions is insufficiently expressive to exhibit many of the phenomena such as mode collapse mentioned above.

In particular, the distributions in this class are all unimodal, and it is unclear what mode collapse would even mean in this context.

The above considerations motivate us to make our model slightly more complicated.

We assume that the true distribution and the generator distributions are all mixtures of two univariate Gaussians with unit variance, and uniform mixing weights.

Formally, our generator set is G, where DISPLAYFORM0 For any µ ∈ R 2 , we let G µ (x) denote the distribution in G with means at µ 1 and µ 2 .

While this is a simple change compared to a single Gaussian case, it makes a large difference in the behavior of the dynamics.

In particular, many of the pathologies present in real-world GAN training begin to appear.

Loss function.

While GANs are usually viewed as a generative framework, they can also be viewed as a general method for density estimation.

We want to set up learning an unknown generator G µ * ∈ G as a generative adversarial dynamics.

To this end, we must first define the loss function for the density estimation problem.

A well-studied goal in this setting is to recover G µ * (x) in total variation (also known as L 1 or statistical) distance, where the total variation distance between two distributions P, Q is defined as DISPLAYFORM1 where the maximum is taken over all measurable events A.Such finding the best-fit distribution in total variation distance can indeed be naturally phrased as generative adversarial dynamics.

Unfortunately, for arbitrary distributions, this is algorithmically problematic, simply because the set of discriminators one would need is intractable to optimize over.

However, for distributions that are structurally simple, like mixtures of Gaussians, it turns out we can consider a much simpler set of discriminators.

In Appendix B.1 in the supplementary material, motivated by connections to VC theory, we show that for two generators DISPLAYFORM2 where the maxima is taken over two disjoint intervals I 1 , I 2 ⊆ R. In other words, instead of considering the difference of measure between the two generators G µ1 , G µ2 on arbitrary events, we may restrict our attention to unions of two disjoint intervals in R. This is a special case of a well-studied distance measure known as the A k -distance, for k = 2 BID5 BID4 .

Moreover, this class of subsets has a simple parametric description.

Discriminators.

Now, the above discussion motivates our definition of discriminators to be DISPLAYFORM3 In other words, the set of discriminators is taken to be the set of indicator functions of sets which can be expressed as a union of at most two disjoint intervals.

With this definition, finding the best fit in total variation distance to some unknown G µ * ∈ G is equivalent to finding µ minimizing DISPLAYFORM4 is a smooth function of all three parameters (see the supplementary material for details).Dynamics.

The objective in (6) is easily amenable to optimization at parameter level.

A natural approach for optimizing this function would be to define G( µ) = max ,r L( µ, , r), and to perform (stochastic) gradient descent on this function.

This corresponds to, at each step, finding the the optimal discriminator, and updating the current µ in that direction.

We call these dynamics the optimal discriminator dynamics.

Formally, given µ (0) and a stepsize η g , and a true distribution G µ * ∈ G, the optimal discriminator dynamics for G µ * , G, D starting at µ (0) are given iteratively as DISPLAYFORM5 where the maximum is taken over , r which induce two disjoint intervals.

For more complicated generators and discriminators such as neural networks, these dynamics are computationally difficult to perform.

Therefore, instead of the updates as in FORMULA6 , one resorts to simultaneous gradient iterations on the generator and discriminator.

These dynamics are called the first order dynamics.

Formally, given µ (0) , (0) , r (0) and a stepsize η g , η d , and a true distribution G µ * ∈ G, the first order dynamics for G µ * , G, D starting at µ (0) are specified as DISPLAYFORM6 DISPLAYFORM7 Even for our relatively simple setting, the first order dynamics can exhibit a variety of behaviors, depending on the starting conditions of the generators and discriminators.

In particular, in FIG0 , we see that depending on the initialization, the dynamics can either converge to optimality, exhibit a primitive form of mode collapse, where the two generators collapse into a single generator, or converge to the wrong value, because the gradients vanish.

This provides empirical justification for our model, and shows that these dynamics are complicated enough to model the complex behaviors which real-world GANs exhibit.

Moreover, as we show in Section 5 below, these behaviors are not just due to very specific pathological initial conditions: indeed, when given random initial conditions, the first order dynamics still more often than not fail to converge.

Parametrization We note here that there could be several potential GAN dynamics to consider here.

Each one resulting from slightly different parametrization of the total variation distance.

For instance, a completely equivalent way to define the total variation distance is DISPLAYFORM8 which does not change the value of the variational distance, but does change the induced dynamics.

We do not focus on these induced dynamics in this paper since they do not exactly fit within the traditional GAN framework, i.e. it is not of the form (1) (see Appendix C).

Nevertheless, it is an interesting set of dynamics and it is a natural question whether similar phenomena occur in these dynamics.

In Appendix C, we show the the optimal discriminator dynamics are unchanged, and the induced first order dynamics have qualitatively similar behavior to the ones we consider in this paper.

This also suggests that the phenomena we exhibit might be more fundamental.

Observe that the optimal discriminator dynamics converge, and then the discriminator varies wildly, because the objective function is not differentiable at optimality.

Despite this it remains roughly at optimality from step to step.

We now describe our results in more detail.

We first consider the dynamics induced by the optimal discriminator.

Our main theoretical result is 1 :Theorem 4.1.

Fix δ > 0 sufficiently small and C > 0.

Let µ * ∈ R 2 so that |µ * i | ≤ C, and DISPLAYFORM0 In other words, if the µ * are bounded by a constant, and not too close together, then in time which is polynomial in the inverse of the desired accuracy δ and e −C 2 , where C is a bound on how far apart the µ * and µ are, the optimal discriminator dynamics converge to the ground truth in total variation distance.

Note that the dependence on e −C 2 is necessary, as if the µ and µ * are initially very far apart, then the initial gradients for the µ will necessarily be of this scale as well.

On the other hand, we provide simulation results that demonstrate that first order updates, or more complicated heuristics such as unrolling, all fail to consistently converge to the true distribution, even under the same sorts of conditions as in Theorem 4.1.

In FIG0 , we gave some specific examples where the first order dynamics fail to converge.

In Section 5 we show that this sort of divergence is common, even with random initializations for the discriminators.

In particular, the probability of convergence is generally much lower than 1, for both the regular GAN dynamics, and unrolling.

In general, we believe that this phenomena should occur for any natural first order dynamics for the generator.

In particular, one barrier we observed for any such dynamics is something we call discriminator collapse, that we describe in Appendix A.

We provide now a high level overview of the proof of Theorem 4.1.

The key element we will need in our proof is the ability to quantify the progress our updates make on converging towards the optimal solution.

This is particularly challenging as our objective function is neither convex nor smooth.

The following lemma is our main tool for achieving that.

Roughly stated, it says that for any Lipschitz function, even if it is non-convex and non-smooth, as long as the change in its derivative is smaller in magnitude than the value of the derivative, gradient descent makes progress on the function value.

Note that this condition is much weaker than typical assumptions used to analyze gradient descent.

Lemma 4.2.

Let g : R k → R be a Lipschitz function that is differentiable at some fixed x ∈ R k .

For some η > 0, let x = x − η∇f (x).

Suppose there exists c < 1 so that almost all v ∈ L(x, x ), where L(x, y) denotes the line between x and y, g is differentiable, and moreover, we have DISPLAYFORM0 Here, we will use the convention that µ * 1 ≤ µ * 2 , and during the analysis, we will always assume for simplicity of notation that DISPLAYFORM1 be the objective function and the difference of the PDFs between the true distribution and the generator, respectively.

For any δ > 0, define the sets DISPLAYFORM2 to be the set of parameter values which have at least one parameter which is not too far from optimality, and the set of parameter values so that all parameter values are close.

We also let B(C) denote the box of sidelength C around the origin, and we let Sep(γ) = {v ∈ R 2 : |v 1 − v 2 | > γ} be the set of parameter vectors which are not too close together.

Our main work lies within a set of lemmas which allow us to instantiate the bounds in Lemma 4.2.

We first show a pair of lemmas which show that, explicitly excluding bad cases such as mode collapse, our dynamics satisfy the conditions of Lemma 4.2.

We do so by establishing a strong (in fact, nearly constant) lower bound on the gradient when we are fairly away from optimality (Lemma 4.3).

Then, we show a relatively weak bound on the smoothness of the function (Lemma 4.4), but which is sufficiently strong in combination with Lemma 4.3 to satisfy Lemma 4.2.

Finally, we rule discriminator dynamics after adding an arbitrarily small amount of Gaussian noise.

It is clear that by taking this noise to be sufficiently small (say exponentially small) then we avoid this pathological set with probability 1, and moreover the noise does not otherwise affect the convergence analysis at all.

For simplicity, we will ignore this issue for the rest of the paper.

out the pathological cases we explicitly excluded earlier, such as mode collapse or divergent behavior (Lemmas 4.5 and 4.6).

Putting all these together appropriately yields the desired statement.

Our first lemma is a lower bound on the gradient value: Lemma 4.3.

Fix C ≥ 1 ≥ γ ≥ δ > 0.

Suppose µ ∈ Rect(0), and suppose µ * , µ ∈ B(C) and DISPLAYFORM3 The above lemma statement is slightly surprising at first glance.

It says that the gradient is never 0, which would suggest there are no local optima at all.

To reconcile this, one should note that the gradient is not continuous (defined) everywhere.

The second lemma states a bound on the smoothness of the function: DISPLAYFORM4 O(1) be the K for which Lemma 4.3 holds with those parameters.

If we have DISPLAYFORM5 for appropriate choices of constants on the RHS, we get DISPLAYFORM6 These two lemmas almost suffice to prove progress as in Lemma 4.2, however, there is a major caveat.

Specifically, Lemma 4.4 needs to assume that µ and µ are sufficiently well-separated, and that they are bounded.

While the µ i start out separated and bounded, it is not clear that it does not mode collapse or diverge off to infinity.

However, we are able to rule these sorts of behaviors out.

Formally: Lemma 4.5 (No mode collapse).

Fix γ > 0, and let δ be sufficiently small.

Let η ≤ δ/C for some C large.

Suppose µ * ∈ Sep(γ).

Then, if µ ∈ Sep(δ), and DISPLAYFORM7 Lemma 4.6 (No diverging to infinity).

Let C > 0 be sufficiently large, and let η > 0 be sufficiently small.

Suppose µ * ∈ B(C), and µ ∈ B(2C).

Then, if we let DISPLAYFORM8 Together, these four lemmas together suffice to prove Theorem 4.1 by setting parameters appropriately.

We refer the reader to Appendix D for more details including the proofs.

To illustrate more conclusively that the phenomena demonstrated in FIG0 are not particularly rare, and that first order dynamics do often fail to converge, we also conducted the following heatmap experiments.

We set µ * = (−0.5, 0.5) as in FIG0 .

We then set a grid for the µ, so that each coordinate is allowed to vary from −1 to 1.

For each of these grid points, we randomly chose a set of initial discriminator intervals, and ran the first order dynamics for 3000 iterations, with constant stepsize 0.3.

We then repeated this 120 times for each grid point, and plotted the probability that the generator converged to the truth, where we say the generator converged to the truth if the TV distance between the generator and optimality is < 0.1.

The choice of these parameters was somewhat arbitrary, however, we did not observe any qualitative difference in the results by varying these numbers, and so we only report results for these parameters.

We also did the same thing for the optimal discriminator dynamics, and for unrolled discriminator dynamics with 5 unrolling steps, as described in (Metz et al., 2017) , which attempt to match the optimal discriminator dynamics.

The results of the experiment are given in Figure 2 .

We see that all three methods fail when we initialize the two generator means to be the same.

This makes sense, since in that regime, the generator starts out mode collapsed and it is impossible for it to un-"mode collapse", so it cannot fit the true distribution well.

Ignoring this pathology, we see that the optimal discriminator otherwise always converges to the ground truth, as our theory predicts.

On the other hand, both regular first order dynamics and unrolled dynamics often times fail, although unrolled dynamics do succeed more often than regular first order dynamics.

This suggests that the pathologies in FIG0 are not so rare, and that these first order methods are quite often unable to emulate optimal discriminator dynamics.

GANs have received a tremendous amount of attention over the past two years BID9 .

Hence we only compare our results to the most closely related papers here.

Figure 2: Heatmap of success probability for random discriminator initialization for regular GAN training, unrolled GAN training, and optimal discriminator dynamics.

The recent paper ) studies generalization aspects of GANs and the existence of equilibria in the two-player game.

In contrast, our paper focuses on the dynamics of GAN training.

We provide the first rigorous proof of global convergence and show that a GAN with an optimal discriminator always converges to an approximate equilibrium.

One recently proposed method for improving the convergence of GAN dynamics is the unrolled GAN (Metz et al., 2017) .

The paper proposes to "unroll" multiple discriminator gradient steps in the generator loss function.

The authors argue that this improves the GAN dynamics by bringing the discriminator closer to an optimal discriminator response.

Our experiments show that this is not a perfect approximation: the unrolled GAN still fails to converge in multiple initial configurations (however, it does converge more often than a "vanilla" one-step discriminator).The authors of BID1 ) also take a theoretical view on GANs.

They identify two important properties of GAN dynamics: (i) Absolute continuity of the population distribution, and (ii) overlapping support between the population and generator distribution.

If these conditions do not hold, they show that the GAN dynamics fail to converge in some settings.

However, they do not prove that the GAN dynamics do converge under such assumptions.

We take a complementary view: we give a convergence proof for a concrete GAN dynamics.

Moreover, our model shows that absolute continuity and support overlap are not the only important aspects in GAN dynamics: although our distributions clearly satisfy both of their conditions, the first-order dynamics still fail to converge.

The paper BID15 studies the stability of equilibria in GAN training.

In contrast to our work, the results focus on local stability while we establish global convergence results.

Moreover, their theorems rely on fairly strong assumptions.

While the authors give a concrete model for which these assumptions are satisfied (the linear quadratic Gaussian GAN), the corresponding target and generator distributions are unimodal.

Hence this model cannot exhibit mode collapse.

We propose the GMM-GAN specifically because it is rich enough to exhibit mode collapse.

The recent work BID10 views GAN training through the lens of online learning.

The paper gives results for the game-theoretic minimax formulation based on results from online learning.

The authors give results that go beyond the convex-concave setting, but do not address generalization questions.

Moreover, their algorithm is not based on gradient descent (in contrast to essentially all practical GAN training) and relies on an oracle for minimizing the highly non-convex generator loss.

This viewpoint is complementary to our approach.

We establish results for learning the unknown distribution and analyze the commonly used gradient descent approach for learning GANs.

We haven taken a step towards a principled understanding of GAN dynamics.

We define a simple yet rich model of GAN training and prove convergence of the corresponding dynamics.

To the best of our knowledge, our work is the first to establish global convergence guarantees for a parametric GAN.

We find an interesting dichotomy: If we take optimal discriminator steps, the training dynamics provably converge.

In contrast, we show experimentally that the dynamics often fail if we take first order discriminator steps.

We believe that our results provide new insights into GAN training and point towards a rich algorithmic landscape to be explored in order to further understand GAN dynamics.

As discussed above, our simple GAN dynamics are able to capture the same undesired behaviors that more sophisticated GANs exhibit.

In addition to these behaviors, our dynamics enables us to discern another degenerate behavior which does not seem to have previously been observed in the literature.

We call this behavior discriminator collapse.

We first explain this phenomenon using language specific to our GMM-GAN dynamics.

In our dynamics, discriminator collapse occurs when a discriminator interval which originally had finite width is forced by the dynamics to have its width converge to 0.

This happens whenever this interval lies entirely in a region where the generator PDF is much larger than the discriminator PDF.

We will shortly argue why this is undesirable.

In FIG2 , we show an example of discriminator collapse in our dynamics.

Each plot in the figure shows the true PDF minus the PDF of the generators, where the regions covered by the discriminator are shaded.

Plot (a) shows the initial configuration of our example.

Notice that the leftmost discriminator interval lies entirely in a region for which the true PDF minus the generators' PDF is negative.

Since the discriminator is incentivized to only have mass on regions where the difference is positive, the first order dynamics will cause the discriminator interval to collapse to have length zero if it is in a negative region.

We see in Plot (c) that this discriminator collapses if we run many discriminator steps for this fixed generator.

In particular, these steps do not converge to the globally optimal discriminator shown in Plot (b).This collapse also occurs when we run the dynamics.

In Plots (d) and (e), we see that after running the first order dynamics -or even unrolled dynamics -for many iterations, eventually both discriminators collapse.

When a discriminator interval has length zero, it can never uncollapse, and moreover, its contribution to the gradient of the generator is zero.

Thus these dynamics will never converge to the ground truth.

For general GANs, we view discriminator collapse as a situation when the local optimization landscape around the current discriminator encourages it to make updates which decrease its representational power.

For instance, this could happen because the first order updates are unable to wholly follow the evolution of the optimal discriminator due to attraction of local maxima, and thus only capture part of the optimal discriminator's structure.

We view understanding the exact nature of discriminator collapse in more general settings and interesting research problem to explore further.

Here we formally prove (4).

In fact, we will prove a slight generalization of this fact which will be useful later on.

We require the following theorem from Hummel and Gidas: Theorem B.1 ( BID11 ).

Let f be any analytic function with at most n zeros.

Then f • N (0, σ 2 ) has at most n zeros.

Theorem B.2.

Any linear combination F (x) of the probability density functions of k Gaussians with the same variance has at most k − 1 zeros, provided at least two of the Gaussians have different means.

In particular, for any µ = ν, the function F (x) = D µ (x) − D ν (x) has at most 3 zeroes.

Proof.

If we have more than 1 Gaussian with the same mean, we can replace all Gaussians having that mean with an appropriate factor times a single Gaussian with that mean.

Thus, we assume without loss of generality that all Gaussians have distinct means.

We may also assume without loss of generality that all Gaussians have a nonzero coefficient in the definition of F .Suppose the minimum distance between the means of any of the Gaussians is δ.

We first prove the statement when δ is sufficiently large compared to everything else.

Consider any pair of Gaussians with consecutive means ν, µ. WLOG assume that µ > ν = 0.

Suppose our pair of Gaussians has the same sign in the definition of F .

In particular they are both strictly positive.

For sufficiently large δ, we can make the contribution of the other Gaussians to F an arbitrarily small fraction of the whichever Gaussian in our pair is largest for all points on [ν, µ].

Thus, for δ sufficiently large, that there are no zeros on this interval.

Now suppose our pair of Gaussians have different signs in the definition of F .

Without loss of generality, assume the sign of the Gaussian with mean ν is positive and the sign of the Gaussian with mean µ is negative.

Then the PDF of the first Gaussian is strictly decreasing on (ν, µ] and the PDF of the negation of the second Gaussian is decreasing on [ν, µ).

Thus, their sum is strictly decreasing on this interval.

Similarly to before, by making δ sufficiently large, the magnitude of the contributions of the other Gaussians to the derivative in this region can be made an arbitrarily small fraction of the magnitude of whichever Gaussian in our pair contributes the most at each point in the interval.

Thus, in this case, there is exactly one zero in the intervale [µ, ν].Also, note that there can be no zeros of F outside of the convex hull of their means.

This follows by essentially the same argument as the two positive Gaussians case above.

The general case (without assuming δ sufficiently large) follows by considering sufficiently skinny (nonzero variance) Gaussians with the same means as the Gaussians in the definition of F , rescaling the domain so that they are sufficiently far apart, applying this argument to this new function, unscaling the domain (which doesn't change the number of zeros), then convolving the function with an appropriate (very fat) Gaussian to obtain the real F , and invoking Theorem B.1 to say that the number of zeros does not increase from this convolution.

In this section, we derive the form of L. By definition, we have DISPLAYFORM0 It is not to hard to see from the Fundamental theorem of calculus that L is indeed a smooth function of all parameters.

Our focus in this paper is on the dynamics induced by, since it arises naturally from the form of the total variation distance (3) and follows the canonical form of GAN dynamics (1).

However, one could consider other equivalent definitions of total variation distance too.

And these definitions could, in principle, induce qualitatively different behavior of the first order dynamics.

As mentioned in Section 3, an alternative dynamics could be induced by the definition of total variation distance given in (10).

The corresponding loss function would be DISPLAYFORM0 i.e. the same as in FORMULA63 but with absolute values on the outside of the expression.

Observe that this loss function does not actually fit the form of the general GAN dynamics presented in (1).

However, it still constitutes a valid and fairly natural dynamics.

Thus one could wonder whether similar behavior to the one we observe for the dynamics we actually study occurs also in this case.

To answer this question, we first observe that by the chain rule, the (sub)-gradient of L with respect to µ, , r are given by DISPLAYFORM1 that is, they are the same as for L except modulated by the sign of L.We now show that the optimal discriminator dynamics is identical to the one that we analyze in the paper FORMULA6 , and hence still provably converge.

This requires some thought; indeed a priori it is not even clear that the optimal discriminator dynamics are well-defined, since the optimal discriminator is no longer unique.

This is because for any µ * , µ, the sets A 1 = {x : G µ * (x) ≥ G µ (x)} and A 2 = {x : G µ (x) ≥ G µ * (x)} both achieve the maxima in (10), since DISPLAYFORM2 However, we show that the optimal discriminator dynamics are still well-formed.

WLOG assume that A1 G µ (x) − G µ * (x)dx ≥ 0, so that A 1 is also the optimal discriminator for the dynamics we consider in the paper.

If we let (i) , r (i) be the left and right endpoints of the intervals in A i for i = 1, 2, we have that the update to µ induced by ((1) , r (1) ) is given by DISPLAYFORM3 so the update induced by ( (1) , r (1) ) is the same as the one induced by the optimal discriminator dynamics in the paper.

Moreover, the update to µ induced by ( (2) , r (2) ) is given by DISPLAYFORM4 where (a) follows from the assumption that A1 G µ (x) − G µ * (x)dx ≥ 0 and from (13), so it is also equal to the the one induced by the optimal discriminator dynamics in the paper.

Hence the optimal discriminator dynamics are well-formed and unchanged from the optimal discriminator dynamics described in the paper.

Thus the question is whether the first order approximation of this dynamics and/or the unrolled first order dynamics exhibit the same qualitative behavior too.

To evaluate the effectiveness, we performed for these dynamics experiments analogous to the ones summarized in Figure 2 in the case of the dynamics we actually analyzed.

The results of these experiments are presented in FIG3 .

Although the probability of success for these dynamics is higher, they still often do not converge.

We can thus see that a similar dichotomy occurs here as in the context of the dynamics we actually study.

In particular, we still observe the discriminator collapse phenomena in these first order dynamics.

It might be somewhat surprising that even with absolute values discriminator collapse occurs.

Originally the discriminator collapse occurred because if an interval was stuck in a negative region, it always subtracts from the value of the loss function, and so the discriminator is incentivized to make it disappear.

Now, since the value of the loss is always nonnegative, it is not so clear that this still happens.

Despite this, we still observe discriminator collapse with these dynamics.

Here we describe one simple scenario in which discriminator collapse still occurs.

Suppose the discriminator intervals have left and right endpoints , r and L(µ, , r) > 0.

Then, if it is the case that DISPLAYFORM5 .

that is, on one of the discriminator intervals the value of the loss is negative, then the discriminator is still incentivized locally to reduce this interval to zero, as doing so increases both L(µ, , r) and hence L (µ, , r).

Symmetrically if L(µ, , r) < 0 and there is a discriminator interval on which the loss is positive, the discriminator is incentivized locally to reduce this interval to zero, since that increases L (µ, , r).

This causes the discriminator collapse and subsequently causes the training to fail to converge.

This appendix is dedicated to a proof of Theorem 4.1.

We start with some remarks on the proof techniques for these main lemmas.

At a high level, Lemmas 4.3, 4.5, 4.6 all follow from involved case analyses.

Specifically, we are able to deduce structure about the possible discriminator intervals by reasoning about the structure of the current mean estimate µ and the true means.

From there we are able to derive bounds on how these discriminator intervals affect the derivatives and hence the update functions.

To prove Lemma 4.4, we carefully study the evolution of the optimal discriminator as we make small changes to the generator.

The key idea is to show that when the generator means are far from the true means, then the zero crossings of F ( µ, x) cannot evolve too unpredictably as we change µ.

We do so by showing that locally, in this setting F can be approximated by a low degree polynomial with large coefficients, via bounding the condition number of a certain Hermite Vandermonde matrix.

This gives us sufficient control over the local behavior of zeros to deduce the desired claim.

By being sufficiently careful with the bounds, we are then able to go from this to the full generality of the lemma.

We defer further details to Appendix D.

By inspection on the form of FORMULA0 , we see that the gradient of the function f µ * ( µ) if it is defined must be given by DISPLAYFORM0 .

DISPLAYFORM1 are the intervals which achieve the supremum in (4).

While these intervals may not be unique, it is not hard to show that this value is well-defined, as long as µ = µ * , that is, when the optimal discriminator intervals are unique as sets.

DISPLAYFORM2

Before we begin, we require the following facts.

We first need that the Gaussian, and any fixed number of derivatives of the Gaussian, are Lipschitz functions.

Fact D.1.

For any constant i, there exists a constant B such that for all x, µ ∈ R, DISPLAYFORM0 Proof.

Note that every derivative of the Gaussian PDF (including the 0th) is a bounded function.

Furthermore, all these derivatives eventually tend to 0 whenever the input goes towards ∞ or −∞. Thus, any particular derivative is bounded by a constant for all R. Furthermore, shifting the mean of the Gaussian does not change the set of values the derivatives of its derivative takes (only their locations).We also need the following bound on the TV distance between two Gaussians, which is folklore, and is easily proven via Pinsker's inequality.

Fact D.2 (folklore).

If two univariate Gaussians with unit variance have means within distance at most ∆ then their TV distance is at most O(1) · ∆.This immediately implies the following, which states that f µ * is Lipschitz.

Corollary D.3.

There is some absolute constant C so that for any µ, ν, we have |f DISPLAYFORM1 We also need the following basic analysis fact: DISPLAYFORM2 This implies that f µ * is indeed differentiable except on a set of measure zero.

As mentioned previously, we will always assume that we never land within this set during our analysis.

D.3 PROOF OF THEOREM 4.1 GIVEN LEMMATA Before we prove the various lemmata described in the body, we show how Theorem 4.1 follows from them.

Proof of Theorem 4.1.

Set δ be a sufficiently small constant multiple of δ.

Provided we make the nonzero constant factor on the step size sufficiently small (compared to δ /δ), and the exponent on δ in the magnitude step size at least one, the magnitude of our step size will be at most δ .

Thus, in any step where µ ∈ Opt(δ ), we end the step outside of this set but still in Opt(2δ ).

By Lemma D.2, for a sufficiently small choice of constant in the definition of δ , the TV-distance at the end of such a step will be at most δ.

Contrapositively, in any step where the TV-distance at the start is more than δ, we will have at the start that µ ∈ Opt(δ ).

Then, it suffices to prove that the step decreases the total variation distance additively by at least 1/ poly(C, e C 2 , 1/δ) in this case.

For appropriate choices of constants in expression for the step size (sufficiently small multiplicative and sufficiently large in the exponent), this is immediately implied by Lemma 4.4 and Lemma 4.2 provided that µ * , µ, µ ∈ B(2C) and | µ 1 − µ 2 | ≥ δ at the beginning of each step.

The condition that we always are within B(2C) at the start of each step is proven in Lemma 4.6 and the condition that the means are separated (ie., that we don't have mode collapse) is proven in Lemma 4.5.It is interesting that a critical component of the above proof involves proving explicitly that mode collapse does not occur.

This suggests the possibility that understanding mode collapse may be helpful in understanding convergence of Generative Adversarial Models and Networks.

In this section we prove Lemma 4.3.

We first require the following fact: DISPLAYFORM0 We also have the following, elementary lemma: DISPLAYFORM1 Then there is some x > µ 2 so that F µ * ( µ, x) < 0.We are now ready to prove Lemma 4.3Proof of Lemma 4.3.

We proceed by case analysis on the arrangement of the µ and µ * .Case 1: µ * 1 < µ 1 and µ * 2 < µ 2 In this case we have F µ * ( µ, x) ≤ 0 for all x ≥ µ 2 .

Hence the optimal discriminators are both to the left of µ 2 .

Moreover, by a symmetric logic we have F µ * ( µ, x) ≥ 0 for all x ≤ µ * 1 , so the optimal discriminator has an interval of the form I 1 = [−∞, r 1 ] and possibly I 2 = [l 2 , r 2 ] where r 1 < l 2 < r 2 < µ 2 .

Then it is easy to see that DISPLAYFORM2 Case 2: µ 1 < µ * 1 and µ 2 < µ * 2 This case is symmetric to Case (1).Case 3: µ 1 < µ * 1 < µ * 2 < µ 2 By Lemma D.6, we know that F µ * ( µ, x) < 0 for some x ≥ µ 2 , and similarly DISPLAYFORM3 , by Theorem B.2 and continuity, the optimal discriminator has one interval.

Denote it by I = [ , r], so that we have ≤ µ * 1 and r ≥ µ * DISPLAYFORM4 We get the symmetric bound on DISPLAYFORM5 The final case is if < µ 1 < µ 2 < r. Consider the auxiliary function DISPLAYFORM6 On the domain [ , r], this function is monotone decreasing.

Moreover, for any µ ∈ [ , r], we have DISPLAYFORM7 In particular, this implies that H( µ 1 ) < H( µ 2 ) − γ 2 e −γ 2 /8 /2, so at least one of H( µ 2 ) or H( µ 1 ) must be γ 2 e −γ 2 /8 /4 in absolute value.

Since DISPLAYFORM8 , this completes the proof in this case.

Case 4: µ * 1 < µ 1 < µ 2 < µ * 2 By a symmetric argument to Case 3, we know that the optimal discriminator intervals are of the form (−∞, r] and [ , ∞) for some r < µ 1 < µ 2 < .

The form of the derivative is then exactly the same as in the last sub-case in Case 3 with signs reversed, so the same bound holds here.

D.5 PROOF OF LEMMA 4.4We now seek to prove Lemma 4.4.

Before we do so, we need to get lower bounds on derivatives of finite sums of Gaussians with the same variance.

In particular, we first show: Lemma D.7.

Fix γ ≥ δ > 0 and C ≥ 1.

Suppose we have µ * , µ ∈ B(C), µ * , µ ∈ Sep(γ), with µ ∈ Rect(δ), where all these vectors have constant length k. Then, for any DISPLAYFORM9 Proof.

Observe that the value of the ith derivative of F µ * ( µ, x) for any x is given by DISPLAYFORM10 where w j ∈ {−1/k, 1/k}, the z j is either x − µ * j or x − µ j , and H i (z) is the ith (probabilist's) Hermite polynomial.

Note that the (−1) i H i are orthogonal with respect to the Gaussian measure over R, and are orthonormal after some finite scaling that depends only on i and is therefore constant.

Hence, if we form the matrix DISPLAYFORM11 w j e −(x−zj ) 2 /2 .

By assumption, we have DISPLAYFORM12 2 /2 ).

Thus, to show that some u i cannot be too small, it suffices to show a lower bound on the smallest singular value of M .

To do so, we leverage the following fact, implicit in the arguments of BID7 : BID7 ).

Let p r (z) be family of orthonormal polynomials with respect to a positive measure dσ on the real line for r = 1, 2, . . .

, t and let z 1 , . . . , z t be arbitrary real numbers with z i = z j for i = j. Define the matrix V given by V ij = p i (z j ).

Then, the smallest singular value of V , denoted σ t (V ), is at least DISPLAYFORM13 DISPLAYFORM14 , where r (y) = s =r y−zs zr−zs is the Langrange interpolating polynomial for the z r .Set p r = H r−1 t = 2k, and σ as the Gaussian measure; then apply the theorem.

Observe that for any i, j, we have |z i − z j | ≥ min(δ, γ) ≥ δ and |z i | ≤ C. Hence the largest coefficient of any Lagrange interpolating polynomial through the z i is at most ( C δ ) 2k−1 with degree 2k − 1.

So, the square of any such polynomial has degree at most 2(2k − 1) and max coefficient at most 2k( DISPLAYFORM15 .Hence by Theorem D.8 we have that DISPLAYFORM16 2 /2 , which immediately implies the desired statement.

We next show that the above Lemma can be slightly generalized, so that we can replace the condition µ ∈ Rect(δ) with µ ∈ Opt(δ).

DISPLAYFORM17 Proof.

Let Ξ be of the form Ω(1) · (δe DISPLAYFORM18 , where we will pick its precise value later.

Lemma D.7 with δ in that Lemma set to Ξ and k = 2 proves the special case when µ ∈ Rect(Ξ).

Thus, the only remaining case is when µ i is close to µ * i for some i and far away for the other i. Without loss of generality, we assume the first entries are the close pair.

Then we have DISPLAYFORM19 There are four terms in the expression for DISPLAYFORM20 Lemma D.7 with δ = Ξ and k = 1 implies that the contribution of the µ 2 and µ * 2 terms to at least one of the 0th through 3rd derivatives has magnitude at least Ω(1) · (δe DISPLAYFORM21 2 /2 such that the magnitude of the contribution of these second two terms is less than half that of the first two, which gives a lower bound on the magnitude of the sum of all the terms of Ω FORMULA0 DISPLAYFORM22 We now show that any function which always has at least one large enough derivative-including its 0th derivaive-on some large enough interval must have a nontrivial amount of mass on the interval.

Lemma D.10.

Let 0 < ξ < 1 and t ∈ N. Let F (x) : R → R be a (t + 1)-times differentiable function such that at every point x on some interval I of length |I| ≥ ξ, F (x) ≥ 0 and there exists an DISPLAYFORM23 Proof.

Let 0 < a < 1 be a non-constant whose value we will choose later.

If I has length more than aξ, truncate it to have this length.

Let z denote the midpoint of I. By assumption, we know that there is some i ∈ 0, . . .

, t such that DISPLAYFORM24 Thus, by Taylor's theorem, we have that DISPLAYFORM25 t+1 for some degree t polynomial p that has some coefficient of magnitude at least B /t!.Thus, if we let G(y) = y z p(x)dx, then G(y) is a degree t + 1 polynomial with some coefficient which is at least B /(t! · t).

By the nonnegativity of F on I, we have that G is nonnegative on [−aξ/2, aξ/2].

By this and the contrapositive of Fact D.5 (invoked with α set to a sufficiently small nonzero constant multiple of B), we have for some such y and some constant B > 0 that G(y) = |G(y)| ≥ B (|I|/2) t+1 B /(t! · t).

Therefore, at this point, we have DISPLAYFORM26 If B B ≤ B, we set a = B B /B ≤ 1 which gives DISPLAYFORM27 Otherwise, B B ≥ B and we perform this substitution along with a = 1 which gives the similar bound DISPLAYFORM28 Together, these bounds imply that we always have DISPLAYFORM29 This allows us to prove the following lemma, which lower bounds how much mass F can put on any interval which is moderately large.

Formally: DISPLAYFORM30 1) be the K for which Lemma 4.3 is always true with those parameters.

Let µ * , µ be so that µ ∈ Opt(δ), µ, µ * ∈ Sep(γ), DISPLAYFORM31 2 ) O(1) such that for any interval I of length |I| ≥ ξ which satisfies I ∩ [−C − 2 log(100/K), C + 2 log(100/K)] = ∅ and on which F ( µ, x) is nonnegative, we have DISPLAYFORM32 Proof.

By Lemma D.9 with C in that lemma set to C + 2 log(100/K), we get a lower bound of DISPLAYFORM33 on the magnitude of at least one of the 0th through 3rd derivatives of F ( µ, x) with respect to x. Set ξ equal to a sufficiently small (nonzero) constant times this value.

By Fact D.1 there exists a constant B such that the magnitude of the fifth derivative of F ( µ, x) with respect to x-which is a linear combination of four fifth derivatives of Gaussians with constant coefficients-is at most B.By Lemma D.10 applied to F ( µ, x) as a function of x, we have I F ( µ, x)dx ≥ Ω(1) · ξ 6 .

Now we can prove Lemma 4.4.Proof of Lemma 4.4.

Let A = [C − 2 log(100/K), C + 2 log(100/K)] where DISPLAYFORM34 is the K for which Lemma 4.3 is always true with those parameters.

Let Z ± denote the set of all x ∈ A for which F ( µ , x) and F ( µ, x) have different nonzero signs.

Let Z + denote the subset of Z ± where F ( µ , x) > 0 > F ( µ, x) and Z − denote the subset where DISPLAYFORM35 Note that Z + can be obtained by making cuts in the real line at the zeros of F ( µ , x), F ( µ, x), and F ( µ , x) − F ( µ, x), then taking the union of some subset of the open intervals induced by these cuts.

By Theorem B.2, the total number of such intervals is O(1).

Thus, Z + is the union of a constant number of open intervals.

By similar arguments, Z − is also the union of a constant number of open intervals.

We now prove that vol( DISPLAYFORM36 .

Note also that by Lemma D.2, each of these intervals has mass under F ( µ , x) at most DISPLAYFORM37 .

By Lemma D.11 and Lemma 4.3, each of these intervals has length at most DISPLAYFORM38 Since there are at most a constant number of such intervals, this is also a bound on vol(Z + ) (and vol(Z − )).Let Y denote the set of x ∈ A on which both F ( µ, x) and F ( µ , x) are nonnegative.

Let X, X denote the x ∈ A for which F ( µ, x) and F ( µ , x) are respectively positive.

Let W, W denote, respectively, the sets of endpoints of the union of the optimal discriminators for µ, µ .

Then the union of the optimal discriminators for µ, µ are respectively Y ∪ Z − ∪ X ∪ W and Y ∪ Z + ∪ X ∪ W .

Furthermore, each of these two unions is given by some constant number of closed intervals and more specifically, that X, X each contain at most two intervals by Lemma B.2.

Thus, we have for any i that This bound also upper bounds ∇f µ * ( µ ) − ∇f µ * ( µ) 2 up to a constant factor.

Thus, if we choose our step to have magnitude µ − µ 2 ≤ Ω(1) · (δe −C 2 /C) O(1) with appropriate choices of constants, we get ∇f µ * ( µ ) − ∇f µ * ( µ) 2 ≤ K/2 ≤ ∇f µ * ( µ) 2 /2 , as claimed D.6 PROOF OF LEMMA 4.5We now prove Lemma 3.4, which forbids mode collapse.

Proof of Lemma 4.5.

Since η ≤ δ, if | µ 1 − µ 2 | > 2δ then clearly µ ∈ Sep(δ), since the gradient is at most a constant since the function is Lipschitz.

Thus assume WLOG that | µ 1 − µ 2 | ≤ 2δ ≤ γ/50.

There are now six cases:Case 1: µ 1 ≤ µ * 1 ≤ µ * 2 ≤ µ 2 This case cannot happen since we assume | µ 1 − µ 2 | ≤ 2δ ≤ γ/50.Case 2: µ * 1 ≤ µ 1 ≤ µ 2 ≤ µ * 2 In this case, by Lemma D.6, we know F is negative at −∞ and at +∞. Since clearly F ≥ 0 when x ∈ [µ * 1 , µ * 2 ], by Theorem B.2 and continuity, the discriminator intervals must be of the form (−∞, r], [ , ∞) for some r ≤ µ 1 ≤ µ 2 ≤ .

Thus, the update to µ i is (up to a constant factor of 2 /2 is monotone on x ∈ [r, ], and thus µ i must actually move away from each other in this scenario.

Case 3: µ * 1 ≤ µ 1 ≤ µ * 2 ≤ µ 2 In this case we must have |µ * 2 − µ 1 | ≤ 2δ and similarly |µ * 2 − µ 2 | ≤ 2δ.

We claim that in this case, the discriminator must be an infinitely long interval (−∞, m] for some m ≤ µ 1 .

This is equivalent to showing that the function F ( µ, x) has only one zero, and this zero occurs at some m ≤ µ 1 .

This implies the lemma in this case since then the update to µ 1 and µ 2 are then in the same direction, and moreover, the magnitude of the update to µ 1 is larger, by inspection.

where the first line follows since moving µ * 2 to µ 1 only increases the value of the function on this interval, and the final line is negative as long as x > (µ * 1 + µ 2 )/2, which is clearly satisfied by our choice of parameters.

By a similar logic (moving µ 2 to µ * 2 ), we get that on the interval [µ * 1 , α − 10δ], the function is strictly positive.

Thus all zeros of F must occur in the interval [α − 10δ, α + 10δ].We now claim that in this interval, the function F is strictly decreasing, and thus has exactly one zero (it has at least one zero because the function changes sign).

The derivative of F with respect to x is given by .

@highlight

To understand GAN training, we define simple GAN dynamics, and show quantitative differences between optimal and first order updates in this model.

@highlight

The authors study the impact of GANs in settings where at each iteration, the discriminator trains to convergence and the generator updates with gradient steps, or where a few gradient steps are done for the disciminator and generator.

@highlight

This paper studies the dynamics of adversarial training of GANs on a Gaussian mixture model