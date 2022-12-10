Generative adversarial networks (GANs) form a generative modeling approach known for producing appealing samples, but they are notably difficult to train.

One common way to tackle this issue has been to propose new formulations of the GAN objective.

Yet, surprisingly few studies have looked at optimization methods designed for this adversarial training.

In this work, we cast GAN optimization problems in the general variational inequality framework.

Tapping into the mathematical programming literature, we counter some common misconceptions about the difficulties of saddle point optimization and propose to extend methods designed for variational inequalities to the training of GANs.

We apply averaging, extrapolation and a computationally cheaper variant that we call extrapolation from the past to the stochastic gradient method (SGD) and Adam.

Generative adversarial networks (GANs) BID12 ) form a generative modeling approach known for producing realistic natural images (Karras et al., 2018) as well as high quality super-resolution (Ledig et al., 2017) and style transfer (Zhu et al., 2017) .

Nevertheless, GANs are also known to be difficult to train, often displaying an unstable behavior BID11 .

Much recent work has tried to tackle these training difficulties, usually by proposing new formulations of the GAN objective (Nowozin et al., 2016; .

Each of these formulations can be understood as a two-player game, in the sense of game theory (Von Neumann and Morgenstern, 1944) , and can be addressed as a variational inequality problem (VIP) BID15 , a framework that encompasses traditional saddle point optimization algorithms (Korpelevich, 1976) .Solving such GAN games is traditionally approached by running variants of stochastic gradient descent (SGD) initially developed for optimizing supervised neural network objectives.

Yet it is known that for some games (Goodfellow, 2016, §8.

2) SGD exhibits oscillatory behavior and fails to converge.

This oscillatory behavior, which does not arise from stochasticity, highlights a fundamental problem: while a direct application of basic gradient descent is an appropriate method for regular minimization problems, it is not a sound optimization algorithm for the kind of two-player games of GANs.

This constitutes a fundamental issue for GAN training, and calls for the use of more principled methods with more reassuring convergence guarantees.

Contributions.

We point out that multi-player games can be cast as variational inequality problems (VIPs) and consequently the same applies to any GAN formulation posed as a minimax or non-zerosum game.

We present two techniques from this literature, namely averaging and extrapolation, widely used to solve VIPs but which have not been explored in the context of GANs before.

1 We extend standard GAN training methods such as SGD or Adam into variants that incorporate these techniques (Alg.

4 is new).

We also explain that the oscillations of basic SGD for GAN training previously noticed BID11 can be explained by standard variational inequality optimization results and we illustrate how averaging and extrapolation can fix this issue.

We introduce a technique, called extrapolation from the past, that only requires one gradient computation per update compared to extrapolation which requires to compute the gradient twice, rediscovering, with a VIP perspective, a particular case of optimistic mirror descent (Rakhlin and Sridharan, 2013) .

We prove its convergence for strongly monotone operators and in the stochastic VIP setting.

Finally, we test these techniques in the context of GAN training.

We observe a 4-6% improvement over Miyato et al. (2018) on the inception score and the Fréchet inception distance on the CIFAR-10 dataset using a WGAN-GP BID14 ) and a ResNet generator.

Outline.

§2 presents the background on GAN and optimization, and shows how to cast this optimization as a VIP.

§3 presents standard techniques and extrapolation from the past to optimize variational inequalities in a batch setting.

§4 considers these methods in the stochastic setting, yielding three corresponding variants of SGD, and provides their respective convergence rates.

§5 develops how to combine these techniques with already existing algorithms.

§6 discusses the related work and §7 presents experimental results.

The purpose of generative modeling is to generate samples from a distribution q θ that matches best the true distribution p of the data.

The generative adversarial network training strategy can be understood as a game between two players called generator and discriminator.

The former produces a sample that the latter has to classify between real or fake data.

The final goal is to build a generator able to produce sufficiently realistic samples to fool the discriminator.

In the original GAN paper BID12 , the GAN objective is formulated as a zero-sum game where the cost function of the discriminator D ϕ is given by the negative log-likelihood of the binary classification task between real or fake data generated from q θ by the generator, However BID12 recommends to use in practice a second formulation, called non-saturating GAN.

This formulation is a non-zero-sum game where the aim is to jointly minimize: DISPLAYFORM0 The dynamics of this formulation has the same stationary points as the zero-sum one (1) but is claimed to provide "much stronger gradients early in learning" BID12 .

The minimax formulation (1) is theoretically convenient because a large literature on games studies this problem and provides guarantees on the existence of equilibria.

Nevertheless, practical considerations lead the GAN literature to consider a different objective for each player as formulated in (2).

In that case, the two-player game problem (Von Neumann and Morgenstern, 1944) consists in finding the following Nash equilibrium: DISPLAYFORM0 Only when L G = −L D is the game called a zero-sum game and (3) can be formulated as a minimax problem.

One important point to notice is that the two optimization problems in (3) are coupled and have to be considered jointly from an optimization point of view.

Standard GAN objectives are non-convex (i.e. each cost function is non-convex), and thus such (pure) equilibria may not exist.

As far as we know, not much is known about the existence of these equilibria for non-convex losses (see BID17 and references therein for some results).

In our theoretical analysis in §4, our assumptions (monotonicity (24) of the operator and convexity of the constraint set) imply the existence of an equilibrium.

In this paper, we focus on ways to optimize these games, assuming that an equilibrium exists.

As is often standard in non-convex optimization, we also focus on finding points satisfying the necessary stationary conditions.

As we mentioned previously, one difficulty that emerges in the optimization of such games is that the two different cost functions of (3) have to be minimized jointly in θ and ϕ. Fortunately, the optimization literature has for a long time studied so-called variational inequality problems, which generalize the stationary conditions for two-player game problems.

We first consider the local necessary conditions that characterize the solution of the smooth two-player game (3), defining stationary points, which will motivate the definition of a variational inequality.

In the unconstrained setting, a stationary point is a couple (θ * , ϕ * ) with zero gradient: DISPLAYFORM0 When constraints are present, 3 a stationary point (θ * , ϕ * ) is such that the directional derivative of each cost function is non-negative in any feasible direction (i.e. there is no feasible descent direction): DISPLAYFORM1 Defining ω def = (θ, ϕ), ω * def = (θ * , ϕ * ), Ω def = Θ × Φ, Eq. (5) can be compactly formulated as: DISPLAYFORM2 These stationary conditions can be generalized to any continuous vector field: let Ω ⊂ R d and F : Ω → R d be a continuous mapping.

The variational inequality problem BID15 ) (depending on F and Ω) is: DISPLAYFORM3 We call optimal set the set Ω * of ω ∈ Ω verifying (VIP).

The intuition behind it is that any ω * ∈ Ω * is a fixed point of the constrained dynamic of F (constrained to Ω).We have thus showed that both saddle point optimization and non-zero sum game optimization, which encompass the large majority of GAN variants proposed in the literature, can be cast as VIPs.

In the next section, we turn to suitable optimization techniques for such problems.

Let us begin by looking at techniques that were developed in the optimization literature to solve VIPs.

We present the intuitions behind them as well as their performance on a simple bilinear problem (see FIG1 ).

Our goal is to provide mathematical insights on averaging ( §3.1) and extrapolation ( §3.2) and propose a novel variant of the extrapolation technique that we called extrapolation from the past ( §3.3).

We consider the batch setting, i.e., the operator F (ω) defined in Eq. 6 yields an exact full gradient.

We present extensions of these techniques to the stochastic setting later in §4.The two standard methods studied in the VIP literature are the gradient method BID3 and the extragradient method (Korpelevich, 1976) .

The iterates of the basic gradient method are given by DISPLAYFORM0 is the projection onto the constraint set (if constraints are present) associated to (VIP).

These iterates are known to converge linearly under an additional assumption on the operator 4 BID4 , but oscillate for a bilinear operator as shown in FIG1 .

On the other hand, the uniform average of these iterates converge for any bounded monotone operator with a O(1/ √ t) rate (Nedić and Ozdaglar, 2009) , motivating the presentation of averaging in §3.1.

By contrast, the extragradient method (extrapolated gradient) does not require any averaging to converge for monotone operators (in the batch setting), and can even converge at the faster O(1/t) rate (Nesterov, 2007) .

The idea of this method is to compute a lookahead step (see intuition on extrapolation in §3.2) in order to compute a more stable direction to follow.

More generally, we consider a weighted averaging scheme with weights ρ t ≥ 0.

This weighted averaging scheme have been proposed for the first time for (batch) VIP by BID3 , DISPLAYFORM0 Averaging schemes can be efficiently implemented in an online fashion noticing that, DISPLAYFORM1 For instance, settingρ T = 1 T yields uniform averaging (ρ t = 1) andρ t = 1 − β < 1 yields geometric averaging, also known as exponential moving averaging (ρ t = β T −t , 1 ≤ t ≤ T ).

Averaging is experimentally compared with the other techniques presented in this section in FIG1 .In order to illustrate how averaging tackles the oscillatory behavior in game optimization, we consider a toy example where the discriminator and the generator are linear: D ϕ (x) = ϕ T x and G θ (z) = θz (implicitly defining q θ ).

By substituting these expressions in the WGAN objective, 5 we get the following bilinear objective: min DISPLAYFORM2 A similar task was presented by Nagarajan and Kolter (2017) where they consider a quadratic discriminator instead of a linear one, and show that gradient descent is not necessarily asymptotically stable.

The bilinear objective has been extensively used BID11 Mescheder et al., 2018; Yadav et al., 2018) to highlight the difficulties of gradient descent for saddle point optimization.

Yet, ways to cope with this issue have been proposed decades ago in the context of mathematical programming.

For illustrating the properties of the methods of interest, we will study their behavior in the rest of §3 on a simple unconstrained unidimensional version of Eq. 9 (this behavior can be generalized to general multidimensional bilinear examples, see §B.3): min DISPLAYFORM3 The operator associated with this minimax game is F (θ, φ) = (φ, −θ).

There are several ways to compute the discrete updates of this dynamics.

The two most common ones are the simultaneous and the alternating gradient update rules, Simultaneous update: DISPLAYFORM4 Interestingly, these two choices give rise to completely different behaviors.

The norm of the simultaneous updates diverges geometrically, whereas the alternating iterates are bounded but do not converge to the equilibrium.

As a consequence, their respective uniform average have a different behavior, as highlighted in the following proposition (proof in §B.1 and generalization in §B.3): Proposition 1.

The simultaneous iterates diverge geometrically and the alternating iterates defined in (11) are bounded but do not converge to 0 as DISPLAYFORM5 The uniform average (θ t ,φ t ) def = 1 t t−1 s=0 (θ s , φ s ) of the simultaneous updates (resp.

the alternating updates) diverges (resp.

converges to 0) as, DISPLAYFORM6 This sublinear convergence result, proved in §B, underlines the benefits of averaging when the sequence of iterates is bounded (i.e. for alternating update rule).

When the sequence of iterates is not bounded (i.e. for simultaneous updates) averaging fails to ensure convergence.

This theorem also shows how alternating updates may have better convergence properties than simultaneous updates.

Another technique used in the variational inequality literature to prevent oscillations is extrapolation.

This concept is anterior to the extragradient method since Korpelevich (1976) mentions that the idea of extrapolated "prices" to give "stability" had been already formulated by Polyak (1963, Chap.

II) .

The idea behind this technique is to compute the gradient at an (extrapolated) point different from the current point from which the update is performed, stabilizing the dynamics: DISPLAYFORM0 Perform update step: DISPLAYFORM1 Note that, even in the unconstrained case, this method is intrinsically different from Nesterov's momentum 6 (Nesterov, 1983, Eq. 2.2.9) because of this lookahead step for the gradient computation: DISPLAYFORM2 Nesterov's method does not converge when trying to optimize (10).

One intuition of why extrapolation has better convergence properties than the standard gradient method comes from Euler's integration framework.

Indeed, to first order, we have ω t+1/2 ≈ ω t+1 + o(η) and consequently, the update step (15) can be interpreted as a first order approximation to an implicit method step: DISPLAYFORM3 Implicit methods are known to be more stable and to benefit from better convergence properties BID1 than explicit methods, e.g., in §B.2 we show that (17) on (10) converges for any η.

Though, they are usually not practical since they require to solve a potentially non-linear system at each step.

Going back to the simplified WGAN toy example (10) from §3.1, we get the following update rules:Implicit: DISPLAYFORM4 In the following proposition, we see that for η < 1, the respective convergence rates of the implicit method and extrapolation are highly similar.

Keeping in mind that the latter has the major advantage of being more practical, this proposition clearly underlines the benefits of extrapolation.

Note that Prop.

1 and 2 generalize to general unconstrained bilinear game (more details and proof in §B.3), Proposition 2.

The squared norm of the iterates DISPLAYFORM5 t , where the update rule of θ t and φ t are defined in (18), decreases geometrically for any η < 1 as, DISPLAYFORM6

One issue with extrapolation is that the algorithm "wastes" a gradient (14).

Indeed we need to compute the gradient at two different positions for every single update of the parameters.

We thus propose a technique that we call extrapolation from the past that only requires a single gradient computation per update.

The idea is to store and re-use the extrapolated gradient for the extrapolation:Extrapolation from the past: DISPLAYFORM0 Perform update step: DISPLAYFORM1 The same update scheme was proposed by Chiang et al. (2012, Alg.

1) in the context of online convex optimization and generalized by Rakhlin and Sridharan (2013) for general online learning.

Without projection, FORMULA1 and FORMULA0 reduce to the optimistic mirror descent described by BID7 : DISPLAYFORM2 We rediscovered this technique from a different perspective: it was motivated by VIP and inspired from the extragradient method.

Using the VIP point of view, we are able to prove a linear convergence rate for extrapolation from the past (see details and proof of Theorem 1 in §B.4).

We also provide results for a stochastic version in §4.

In comparison to the results from BID7 that Adam) with the techniques presented in §3 on the optimization of (9).

Only the algorithms advocated in this paper (Averaging, Extrapolation and Extrapolation from the past) converge quickly to the solution.

Each marker represents 20 iterations.

We compare these algorithms on a non-convex objective in §G.1.

DISPLAYFORM3 Figure 2: Three variants of SGD computing T updates, using the techniques introduced in §3.hold only for a bilinear objective, we provide a faster convergence rate (linear vs sublinear) on the last iterate for a general (strongly monotone) operator F and any projection on a convex Ω. One thing to notice is that the operator of a bilinear objective is not strongly monotone, but in that case one can use the standard extrapolation method (14) which converges linearly for a (constrained or not) bilinear game (Tseng, 1995, Cor.

3.3) .

Theorem 1 (Linear convergence of extrapolation from the past).

If F is µ-strongly monotone (see §A for the definition of strong monotonicity) and L-Lipschitz, then the updates (20) and (21) with η = 1 4L provide linearly converging iterates, DISPLAYFORM4

In this section, we consider extensions of the techniques presented in §3 to the context of a stochastic operator, i.e., we no longer have access to the exact gradient F (ω) but to an unbiased stochastic estimate of it, F (ω, ξ), where ξ ∼ P and DISPLAYFORM0 It is motivated by GAN training where we only have access to a finite sample estimate of the expected gradient, computed on a mini-batch.

For GANs, ξ is a mini-batch of points coming from the true data distribution p and the generator distribution q θ .For our analysis, we require at least one of the two following assumptions on the stochastic operator: DISPLAYFORM1 Assumption 2.

Bounded expected squared norm by DISPLAYFORM2 Assump.

1 is standard in stochastic variational analysis, while Assump.

2 is a stronger assumption sometimes made in stochastic convex optimization.

To illustrate how strong Assump.

2 is, note that it does not hold for an unconstrained bilinear objective like in our example FORMULA0 We now present and analyze three algorithms that are variants of SGD that are appropriate to solve (VIP).

The first one Alg.

1 (AvgSGD) is the stochastic extension of the gradient method for solving (VIP); Alg.

2 (AvgExtraSGD) uses extrapolation and Alg.

3 (AvgPastExtraSGD) uses extrapolation from the past.

A fourth variant that re-use the mini-batch for the extrapolation step (ReExtraSGD, Alg.

5) is described in §D. These four algorithms return an average of the iterates (typical in stochastic setting).

The proofs of the theorems presented in this section are in §F.To handle constraints such as parameter clipping , we gave a projected version of these algorithms, where P Ω [ω ] denotes the projection of ω onto Ω (see §A).

Note that when Ω = R d , the projection is the identity mapping (unconstrained setting).

In order to prove the convergence of these four algorithms, we will assume that F is monotone: DISPLAYFORM3 (24) If F can be written as (6), it implies that the cost functions are convex.7 Note however that general GANs parametrized with neural networks lead to non-monotone VIPs.

Assumption 3.

F is monotone and Ω is a compact convex set, such that max ω,ω ∈Ω ω−ω 2 ≤ R 2 .In that setting the quantity g(ω * ) := max ω∈Ω F (ω) (ω * − ω) is well defined and is equal to 0 if and only if ω * is a solution of (VIP).

Moreover, if we are optimizing a zero-sum game, DISPLAYFORM4 is well defined and equal to 0 if and only if (θ * , ϕ * ) is a Nash equilibrium of the game.

The two functions g and h are called merit functions (more details on the concept of merit functions in §C).

In the following, we call, DISPLAYFORM5 Averaging.

Alg.

1 (AvgSGD) presents the stochastic gradient method with averaging, which reduces to the standard (simultaneous) SGD updates for the two-player games used in the GAN literature, but returning an average of the iterates.

Theorem 2.

Under Assump.

1, 2 and 3, SGD with averaging (Alg.

1) with a constant step-size gives, FORMULA1 is called the variance term.

This type of bound is standard in stochastic optimization.

We also provide in §F a similarÕ(1/ √ t) rate with an extra log factor when η t = η √ t DISPLAYFORM6

.

We show that this variance term is smaller than the one of SGD with prediction method (Yadav et al., 2018) in §E.Extrapolations.

Alg.

2 (AvgExtraSGD) adds an extrapolation step compared to Alg.

1 in order to reduce the oscillations due to the game between the two players.

A theoretical consequence is that it has a smaller variance term than (26).

As discussed previously, Assump.

2 made in Thm.

2 for the convergence of Alg.

1 is very strong in the unbounded setting.

One advantage of SGD with extrapolation is that Thm.

3 does not require this assumption.

gives, DISPLAYFORM0 Since in practice σ M , the variance term in FORMULA1 is significantly smaller than the one in (26).

To summarize, SGD with extrapolation provides better convergence guarantees but requires two gradient computations and samples per iteration.

This motivates our new method, Alg.

3 (AvgPastExtraSGD) which uses extrapolation from the past and achieves the best of both worlds (in theory). , gives that the averaged iterates converge as, DISPLAYFORM1 The bound is similar to the one provided in Thm.

3 but each iteration of Alg.

3 is computationally half the cost of an iteration of Alg.

2.

In the previous sections, we presented several techniques that converge for stochastic monotone operators.

These techniques can be combined in practice with existing algorithms.

We propose to combine them to two standard algorithms used for training deep neural networks: the Adam optimizer (Kingma and Ba, 2015) and the SGD optimizer (Robbins and Monro, 1951) .

For the Adam optimizer, there are several possible choices on how to update the moments.

This choice can lead to different algorithms in practice: for example, even in the unconstrained case, our proposed Adam with extrapolation from the past (Alg.

4) is different from Optimistic Adam BID7 (the moments are updated differently).

Note that in the case of a two-player game (3), the previous convergence results can be generalized to gradient updates with a different step-size for each player by simply rescaling the objectives L G and L D by a different scaling factor.

A detailed pseudo-code for Adam with extrapolation step (Extra-Adam) is given in Algorithm 4.

Note that our interest regarding this algorithm is practical and that we do not provide any convergence proof.

Algorithm 4 Extra-Adam: proposed Adam with extrapolation step.input: step-size η, decay rates for moment estimates β 1 , β 2 , access to the stochastic gradients ∇ t (·) and to the projection DISPLAYFORM0 Sample new mini-batch and compute stochastic gradient: g t ← ∇ t (ω t ) Option 2: Extrapolation from the past Load previously saved stochastic gradient: DISPLAYFORM1 Correct the bias for the moments: DISPLAYFORM2 Sample new mini-batch and compute stochastic gradient: g t+1/2 ← ∇ t+1/2 (ω t+1/2 ) Update estimate of first moment: DISPLAYFORM3 Update estimate of second moment: DISPLAYFORM4 Compute bias corrected for first and second moment: DISPLAYFORM5 2 ) Perform update step from the iterate at time t: DISPLAYFORM6 The extragradient method is a standard algorithm to optimize variational inequalities.

This algorithm has been originally introduced by Korpelevich (1976) and extended by Nesterov FORMULA1 and Nemirovski (2004) .

Stochastic versions of the extragradient have been recently analyzed (Juditsky et al., 2011; Yousefian et al., 2014; BID18 for stochastic variational inequalities with bounded constraints.

A linearly convergent variance reduced version of the stochastic gradient method has been proposed by Palaniappan and Bach (2016) for strongly monotone variational inequalities.

Extrapolation can also be related to optimistic methods BID5 Rakhlin and Sridharan, 2013) proposed in the online learning literature (see more details in §3.3).

Interesting non-convex results were proved, for a new notion of regret minimization, by BID16 and in the context of online learning for GANs by BID13 .Several methods to stabilize GANs consist in transforming a zero-sum formulation into a more general game that can no longer be cast as a saddle point problem.

This is the case of the non-saturating formulation of GANs BID12 BID8 , the DCGANs (Radford et al., 2016) , the gradient penalty 8 for WGANs BID14 .

Yadav et al. (2018) propose an optimization method for GANs based on AltSGD using an additional momentum-based step on the generator.

BID7 proposed a method inspired from game theory.

Li et al. (2017) suggest to dualize the GAN objective to reformulate it as a maximization problem and Mescheder et al. (2017) propose to add the norm of the gradient in the objective to get a better signal.

BID10 analyzed a generalization of the bilinear example (9) with a focus put on the effect of momentum on this problem.

They do not consider extrapolation (see §B.3 for more details).

Unrolling steps (Metz et al., 2017) can be confused with extrapolation but is fundamentally different: the perspective is to try to approximate the "true generator objective function" unrolling for K steps the updates of the discriminator and then updating the generator.

Regarding the averaging technique, some recent work appear to have already successfully used geometric averaging (7) for GANs in practice, but only briefly mention it (Karras et al., 2018; Mescheder et al., 2018) .

By contrast, the present work formally motivates and justifies the use of averaging for GANs by relating them to the VIP perspective, and sheds light on its underlying intuitions in §3.1.

Subsequent to our first preprint, Yazıcı et al. (2019) explored averaging empirically in more depth, while Mertikopoulos et al. FORMULA0 also investigated extrapolation, providing asymptotic convergence results (i.e. without any rate of convergence) in the context of coherent saddle point.

The coherence assumption is slightly weaker than monotonicity.

Our goal in this experimental section is not to provide new state-of-the art results with architectural improvements or a new GAN formulation, but to show that using the techniques (with theoretical guarantees in the monotone case) that we introduced earlier allows us to optimize standard GANs in a better way.

These techniques, which are orthogonal to the design of new formulations of GAN optimization objectives, and to architectural choices, can potentially be used for the training of any type of GAN.

We will compare the following optimization algorithms: baselines are SGD and Adam using either simultaneous updates on the generator and on the discriminator (denoted SimAdam and SimSGD) or k updates on the discriminator alternating with 1 update on the generator (denoted AltSGD{k} and AltAdam{k}).9 Variants that use extrapolation are denoted ExtraSGD (Alg.

2) and ExtraAdam (Alg.

4).

Variants using extrapolation from the past are PastExtraSGD (Alg.

3) and PastExtraAdam (Alg.

4).

We also present results using as output the averaged iterates, adding Avg as a prefix of the algorithm name when we use (uniform) averaging.

We first test the various stochastic algorithms on a simple (n = 10 3 , d = 10 3 ) finite sum bilinear objective (a monotone operator) constrained to [−1, 1] d :

solved by (θ DISPLAYFORM0 Results are shown in FIG2 .

We can see that AvgAltSGD1 and AvgPastExtraSGD perform the best on this task.

We evaluate the proposed techniques in the context of GAN training, which is a challenging stochastic optimization problem where the objectives of both players are non-convex.

We propose to evaluate the Adam variants of the different optimization algorithms (see Alg.

4 for Adam with extrapolation) by training two different architectures on the CIFAR10 dataset (Krizhevsky and Hinton, 2009 Right: WGAN-GP trained on CIFAR10: mean and standard deviation of the inception score computed over 5 runs for each method using the best performing learning rates; all experiments were run on a NVIDIA Quadro GP100 GPU.

We see that ExtraAdam converges faster than the Adam baselines.2016) with the WGAN objective and weight clipping as proposed by .

Then, we compare the different methods on a state-of-the-art architecture by training a ResNet with the WGAN-GP objective similar to BID14 .

Models are evaluated using the inception score (IS) (Salimans et al., 2016) computed on 50,000 samples.

We also provide the FID BID17 and the details on the ResNet architecture in §G.3.For each algorithm, we did an extensive search over the hyperparameters of Adam.

We fixed β 1 = 0.5 and β 2 = 0.9 for all methods as they seemed to perform well.

We note that as proposed by BID17 , it is quite important to set different learning rates for the generator and discriminator.

Experiments were run with 5 random seeds for 500,000 updates of the generator.

Tab.

1 reports the best IS achieved on these problems by each considered method.

We see that the techniques of extrapolation and averaging consistently enable improvements over the baselines (see §G.5 for more experiments on averaging).

FIG3 shows training curves for each method (for their best performing learning rate), as well as samples from a ResNet generator trained with ExtraAdam on a WGAN-GP objective.

For both tasks, using an extrapolation step and averaging with Adam (ExtraAdam) outperformed all other methods.

Combining ExtraAdam with averaging yields results that improve significantly over the previous state-of-the-art IS (8.2) and FID (21.7) on CIFAR10 as reported by Miyato et al. FORMULA0 (see Tab.

5 for FID).

We also observed that methods based on extrapolation are less sensitive to learning rate tuning and can be used with higher learning rates with less degradation; see §G.4 for more details.

We newly addressed GAN objectives in the framework of variational inequality.

We tapped into the optimization literature to provide more principled techniques to optimize such games.

We leveraged these techniques to develop practical optimization algorithms suitable for a wide range of GAN training objectives (including non-zero sum games and projections onto constraints).

We experimentally verified that this could yield better trained models, improving the previous state of the art.

The presented techniques address a fundamental problem in GAN training in a principled way, and are orthogonal to the design of new GAN architectures and objectives.

They are thus likely to be widely applicable, and benefit future development of GANs.

In this section, we recall usual definitions and lemmas from convex analysis.

We start with the definitions and lemmas regarding the projection mapping.

A.1 PROJECTION MAPPING Definition 1.

The projection P Ω onto Ω is defined as, DISPLAYFORM0 When Ω is a convex set, this projection is unique.

This is a consequence of the following lemma that we will use in the following sections: the non-expansiveness of the projection onto a convex set.

Lemma 1.

Let Ω a convex set, the projection mapping DISPLAYFORM1 This is standard convex analysis result which can be found for instance in BID2 .

The following lemma is also standard in convex analysis and its proof uses similar arguments as the proof of Lemma 1.Lemma 2.

Let ω ∈ Ω and ω DISPLAYFORM2 , then for all ω ∈ Ω we have, DISPLAYFORM3 Proof of Lemma 2.

We start by simply developing, DISPLAYFORM4 Then since ω + is the projection onto the convex set Ω of ω + u, we have that DISPLAYFORM5 leading to the result of the Lemma.

Another important property used is the Lipschitzness of an operator.

DISPLAYFORM0 In this paper, we also use the notion of strong monotonicity, which is a generalization for operators of the notion of strong convexity.

Let us first recall the definition of the latter, DISPLAYFORM1 If a function f (resp.

L) is strongly convex (resp.

strongly convex-concave), its gradient ∇f (resp.

DISPLAYFORM2 Definition 5.

For µ > 0, an operator F : Ω → R d is said to be µ-strongly monotone if DISPLAYFORM3

In this section, we will prove the results provided in §3, namely Proposition 1, Proposition 2 and Theorem 1.

For Proposition 1 and 2, let us recall the context.

We wanted to derive properties of some gradient methods on the following simple illustrative example DISPLAYFORM0 B.1 PROOF OF PROPOSITION 1Let us first recall the proposition:Proposition' 1.

The simultaneous iterates diverge geometrically and the alternating iterates defined in (11) are bounded but do not converge to 0 as DISPLAYFORM1 The uniform average (θ t ,φ t ) def = 1 t t−1 s=0 (θ s , φ s ) of the simultaneous updates (resp.

the alternating updates) diverges (resp.

converges to 0) as, DISPLAYFORM2 Proof.

Let us start with the simultaneous update rule: DISPLAYFORM3 Then we have, DISPLAYFORM4 The update rule (39) also gives us, DISPLAYFORM5 Summing FORMULA1 for 0 ≤ t ≤ T − 1 to get telescoping sums, we get DISPLAYFORM6 Let us continue with the alternating update rule DISPLAYFORM7 Then we have, DISPLAYFORM8 By simple linear algebra, for η < 2, the matrix M def = 1 −η η 1 − η 2 has two complex conjugate eigenvalues which are DISPLAYFORM9 and their squared magnitude is equal to det(M ) = 1 − η 2 + η 2 = 1.

We can diagonalize M meaning that there exists P an invertible matrix such that M = P −1 diag(λ + , λ − )P .

Then, we have DISPLAYFORM10 and consequently, DISPLAYFORM11 where · C is the norm in C 2 and P := max u∈C 2 P u C u C is the induced matrix norm.

The same way we have, DISPLAYFORM12 Hence, if θ 2 0 + φ 2 0 > 0, the sequence (θ t , φ t ) is bounded but do not converge to 0.

Moreover the update rule gives us, DISPLAYFORM13 Consequently, since θ DISPLAYFORM14

In this section, we will prove a slightly more precise proposition than Proposition 2, Proposition' 2.

The squared norm of the iterates N 2 t def = θ 2 t + φ 2 t , where the update rule of θ t and φ t is defined in (18), decrease geometrically for any 0 < η < 1 as, DISPLAYFORM0 Proof.

Let us recall the update rule for the implicit method DISPLAYFORM1 Then, DISPLAYFORM2 implying that DISPLAYFORM3 which is valid for any η.

For the extrapolation method, we have the update rule DISPLAYFORM4 Implying that, DISPLAYFORM5 DISPLAYFORM6

In this section, we will show how to simply extend the study of the algorithm of interest provided in §3 on the general unconstrained bilinear example, DISPLAYFORM0 where, A ∈ R d×p , b ∈ R d and c ∈ R p .

The only assumption we will make is that this problem is feasible which is equivalent to say that there exists a solution (θ * , ϕ * ) to the system DISPLAYFORM1 In this case, we can re-write (63) as DISPLAYFORM2 where c := −θ * Aϕ * is a constant that does not depend on θ and ϕ.First, let us show that we can reduce the study of simultaneous, alternating, extrapolation and implicit updates rules for (63) to the study of the respective unidimensional updates (11) and (18).This reduction has already been proposed by BID10 .

For completeness, we reproduce here similar arguments.

The following lemma is a bit more general than the result provided by BID10 .

It states that the study of a wide class of unconstrained first order method on (63) can be reduced to the study of the method on (36), with potentially rescaled step-sizes.

Before explicitly stating the lemma, we need to introduce a bit of notation to encompass easily our several methods in a unified way.

First, we let ω t := (θ t , ϕ t ), where the index t here is a more general index which can vary more often than the one in §3.

For example, for the extrapolation method, we could consider ω 1 = ω 0+1/2 and ω 2 = ω 1 , where ω was the sequence defined for the extragradient.

For the alternated updates, we can consider ω 1 = (θ 1 , ϕ 0 ) and ω 2 = (θ 1 , ϕ 1 ) (this also defines θ 2 = θ 1 ), where θ and ϕ were the sequences originally defined for alternated updates.

We are thus ready to state the lemma.

Lemma 3.

Let us consider the following very general class of first order methods on (63), i.e., DISPLAYFORM3 where ω t := (θ t , ϕ t ) and F θ (ω t ) := Aϕ t − b, F ϕ (ω t ) = A θ t − c. Then, we have DISPLAYFORM4 where A = U DV (SVD decomposition) and the couples Proof.

Our general class of first order methods can be written with the following update rules: DISPLAYFORM5 where λ it , µ it ∈ R , 0 ≤ i ≤ t + 1.

We allow the dependence on t for the algorithm coefficients λ and µ (for example, the alternating rule would zero out some of the coefficients depending on whether we are updating θ or ϕ at the current iteration).

Notice also that if both λ (t+1)t and µ (t+1)t are non-zero, we have an implicit scheme.

Thus, using the SVD of A = U DV , we get DISPLAYFORM6 which is equivalent to DISPLAYFORM7 where D is a rectangular matrix with zeros except on a diagonal block of size r. Thus, each coordinate ofθ t+1 andφ t+1 are updated independently, reducing the initial problem to r unidimensional problems, DISPLAYFORM8 where σ 1 ≥ . . .

≥ σ r > 0 are the positive diagonal coefficients of D. Note that the only additional restriction is that the coefficients (λ st ) and (σ st ) (that are the same for 1 ≤ i ≤ r) are rescaled by the singular values of A. In practice, for our methods of interest with a step-size η, it corresponds to the study of r unidimensional problem with a respective step-size DISPLAYFORM9 From this lemma, an extension of Proposition 1 and 2 directly follows to the general unconstrained bilinear objective (63).

We note DISPLAYFORM10 where (Θ * , Φ * ) is the set of solutions of (63).

The following corollary is divided in two points, the first point is a result from BID10 (note that the result on the average is a straightforward extension of the one provided in Proposition 1 and was not provided by BID10 ), the second result is new.

Very similar asymptotic upper bounds regarding extrapolation and implicit methods can be derived by Tseng (1995) computing the exact values of the constant τ 1 and τ 2 (and noticing that τ 3 = ∞) introduced in (Tseng, 1995, Eq. 3 & 4) for the unconstrained bilinear case.

However, since Tseng (1995) works in a very general setting, the bound are not as tight as ours and his proof technique is a bit more technical.

Our reduction above provides here a simple proof for our simple setting.

• Gidel et al. FORMULA0 : The simultaneous iterates diverge geometrically and the alternating iterates are bounded but do not converge to 0 as, Simultaneous: φ s ) of the simultaneous updates (resp.

the alternating updates) diverges (resp.

converges to 0) as, DISPLAYFORM0 DISPLAYFORM1 • Extrapolation and Implicit method: The iterates respectively generated by the update rules FORMULA0 and FORMULA0 on a bilinear unconstrained problem (63) do converge linearly for any 0 < η < 1 σmax(A) at a rate, DISPLAYFORM2 Particularly, for η = 1 2σmax(A) we get for the extrapolation method, DISPLAYFORM3 where κ :=

2 σmin(A) 2 is the condition number of A A.

Let us recall what we call projected extrapolation form the past, where we used the notation ω t = ω t+1/2 for compactness, Extrapolation from the past: DISPLAYFORM0 Perform update step: DISPLAYFORM1 where P Ω [·] is the projection onto the constraint set Ω. An operator F : Ω → R d is said to be µ-strongly monotone if DISPLAYFORM2 If F is strongly monotone, we can prove the following theorem:Theorem' 1.

If F is µ-strongly monotone (see §A for the definition of strong monotonicity) and L-Lipschitz, then the updates FORMULA1 and FORMULA0 with η = 1 4L provide linearly converging iterates, DISPLAYFORM3 Proof.

In order to prove this theorem, we will prove a slightly more general result, DISPLAYFORM4 with the convention that ω 0 = ω −1 = ω −2 .

It implies that DISPLAYFORM5 Let us first proof three technical lemmas.11 As before, the inequality (73) for the implicit scheme is actually valid for any step-size.

Lemma 4.

If F is µ-strongly monotone, we have DISPLAYFORM6 Proof.

By strong monotonicity and optimality of ω * , DISPLAYFORM7 and then we use the inequality 2 ω t − ω * 2 2 ≥ ω t − ω * 2 2 − 2 ω t − ω t 2 2 to get the result claimed.

DISPLAYFORM8 and DISPLAYFORM9 (85) Summing FORMULA3 and FORMULA4 we get, DISPLAYFORM10 (87) DISPLAYFORM11 Then, we can use the Young's inequality 2a DISPLAYFORM12 Lemma 6.

For all t ≥ 0, if we set ω −2 = ω −1 = ω 0 we have DISPLAYFORM13 Proof.

We start with a + b 2 2 ≤ 2 a 2 + 2 b 2 .

DISPLAYFORM14 Moreover, since the projection is contractive we have that DISPLAYFORM15 Combining FORMULA1 and FORMULA4 we get, DISPLAYFORM16 (97) DISPLAYFORM17 Proof of Theorem 1.

Let ω * ∈ Ω * be an optimal point of (VIP).

Combining Lemma 4 and Lemma 5 we get, DISPLAYFORM18 leading to, DISPLAYFORM19 Now using Lemma 6 we get, DISPLAYFORM20 Now with η t = 1 4L ≤ 1 4µ we get, DISPLAYFORM21 Hence, using the fact that DISPLAYFORM22 we get, DISPLAYFORM23 In this section, we will present how to handle an unbounded constraint set Ω with a more refined merit function than (25) used in the main paper.

Let F be the continuous operator and Ω be the constraint set associated with the VIP, DISPLAYFORM24 When the operator F is monotone, we have that DISPLAYFORM25 Hence, in this case (VIP) implies a stronger formulation sometimes called Minty variational inequality BID6 : DISPLAYFORM26 This formulation is stronger in the sense that if (MVI) holds for some ω * ∈ Ω, then (VIP) holds too.

A merit function useful for our analysis can be derived from this formulation.

Roughly, a merit function is a convergence measure.

More formally, a function g : Ω → R is called a merit function if g is non-negative such that g(ω) = 0 ⇔ ω ∈ Ω * (Larsson and Patriksson, 1994) .

A way to derive a merit function from (MVI) would be to use g(ω * ) = sup ω∈Ω F (ω) (ω * − ω) which is zero if and only if (MVI) holds for ω * .

To deal with unbounded constraint sets (leading to a potentially infinite valued function outside of the optimal set), we use the restricted merit function (Nesterov, 2007) : = Ω ∩ {ω : ω − ω 0 < R}. Then for any pointω ∈ Ω R , we have: DISPLAYFORM27 DISPLAYFORM28 The reference point ω 0 is arbitrary, but in practice it is usually the initialization point of the algorithm.

R has to be big enough to ensure that Ω R contains a solution.

Err R measures how much (MVI) is violated on the restriction Ω R .

Such merit function is standard in the variational inequality literature.

A similar one is used in (Nemirovski, 2004; Juditsky et al., 2011) .

When F is derived from the gradients (5) of a zero-sum game, we can define a more interpretable merit function.

One has to be careful though when extending properties from the minimization setting to the saddle point setting (e.g. the merit function used by Yadav et al. FORMULA0 is vacuous for a bilinear game as explained in App C.2).In the appendix, we adopt a set of assumptions a little more general than the one in the main paper: DISPLAYFORM29 • F is monotone and Ω is convex and closed.• R is set big enough such that R > ω 0 − ω * and F is a monotone operator.

Contrary to Assumption 3, in Assumption 4 the constraint set in no longer assumed to be bounded.

Assumption 4 is implied by Assumption 3 by setting R to the diameter of Ω, and is thus more general.

In this appendix, we will note Err (VI) R the restricted merit function defined in (102).

Let us recall its definition, Err DISPLAYFORM0 When the objective is a saddle point problem i.e., DISPLAYFORM1 and L is convex-concave (see Definition 4 in §A), we can use another merit function than FORMULA0 on Ω R that is more interpretable and more directly related to the cost function of the minimax formulation: DISPLAYFORM2 In particular, if the equilibrium (θ * , ϕ * ) ∈ Ω * ∩ Ω R and we have that L(·, ϕ * ) and −L(θ * , ·) are µ-strongly convex (see §A), then the merit function for saddle points upper bounds the distance for (θ, ϕ) ∈ Ω R to the equilibrium as: DISPLAYFORM3 In the appendix, we provide our convergence results with the merit functions (104) and (106), depending on the setup: DISPLAYFORM4 (108)

In this section, we illustrate the fact that one has to be careful when extending results and properties from the minimization setting to the minimax setting (and consequently to the variational inequality setting).

Another candidate as a merit function for saddle point optimization would be to naturally extend the suboptimality f (ω) − f (ω * ) used in standard minimization (i.e. find ω * the minimizer of f ) to the gap DISPLAYFORM0 In a previous analysis of a modification of the stochastic gradient descent (SGD) method for GANs, Yadav et al. (2018) gave their convergence rate on P that they called the "primal-dual" gap.

Unfortunately, if we do not assume that the function L is strongly convex-concave (a stronger assumption defined in §A and which fails for bilinear objective e.g.), P may not be a merit function.

It can be 0 for a non optimal point, see for instance the discussion on the differences between (106) and P in (Gidel et al., 2017, Section 3) .

In particular, for the simple 2D bilinear example L(θ, ϕ) = θ · ϕ, we have that θ * = ϕ * = 0 and thus P (θ, ϕ) = 0 ∀θ, ϕ .

When the cost functions defined in (3) are non-convex, the operator F is no longer monotone.

Nevertheless, (VIP) and (MVI) can still be defined, though a solution to (MVI) is less likely to exist.

We note that (VIP) is a local condition for F (as only evaluating F at the points ω * ).

On the other hand, an appealing property of (MVI) is that it is a global condition.

In the context of minimization of a function f for example (where F = ∇f ), if ω * solves (MVI) then ω * is a global minimum of f (and not just a stationary point for the solution of (MVI); see Proposition 2.2 from BID6 ).A less restrictive way to consider variational inequalities in the non-monotone setting is to use a local version of (MVI).

If the cost functions are locally convex around the optimal couple (θ * , ϕ * ) and if our iterates eventually fall and stay into that neighborhood, then we can consider our restricted merit function Err R (·) with a well suited constant R and apply our convergence results for monotone operators.

We now introduce another way to combine extrapolation and SGD.

This extension is very similar to AvgExtraSGD Alg.

2, the only difference is that it re-uses the mini-batch sample of the extrapolation step for the update of the current point.

The intuition is that it correlates the estimator of the gradient of the extrapolation step and the one of the update step leading to a better correction of the oscillations which are also due to the stochasticity.

One emerging issue (for the analysis) of this method is that since ω t depend on ξ t , the quantity F (ω t , ξ t ) is a biased estimator of F (ω t ).Algorithm 5 Re-used mini-batches for stochastic extrapolation (ReExtraSGD) DISPLAYFORM0 Sample ξ t ∼ P 4: DISPLAYFORM1 Extrapolation step 5: DISPLAYFORM2 Update step with the same sample 6: end for has the following convergence properties: DISPLAYFORM3 DISPLAYFORM4 The assumption that the sequence of the iterates provided by the algorithm is bounded is strong, but has also been made for instance in (Yadav et al., 2018) .

The proof of this result is provided in §F.

To compare the variance term of AvgSGD in FORMULA1 with the one of the SGD with prediction method (Yadav et al., 2018), we need to have the same convergence certificate.

Fortunately, their proof can be adapted to our convergence criterion (using Lemma 7 in §F), revealing an extra σ 2 /2 in the variance term from their paper.

The resulting variance can be summarized with our notation as DISPLAYFORM0 where the L is the Lipschitz constant of the operator F .

Since M σ, their variance term is then 1 + L time larger than the one provided by the AvgSGD method.

This section is dedicated on the proof of the theorems provided in this paper in a slightly more general form working with the merit function defined in (108).

First we prove an additional lemma necessary to the proof of our theorems.

Lemma 7.

Let F be a monotone operator and let (ω t ), (ω t ), (z t ), (∆ t ), (ξ t ) and (ζ t ) be six random sequences such that, for all t ≥ 0 2η t F (ω t ) (ω t − ω) ≤ N t − N t+1 + η where N t = N (ω t , ω t−1 , ω t−2 ) ≥ 0 and we extend (ω t ) with ω −2 = ω −1 = ω 0 .

Let also assume that with DISPLAYFORM0 Proof of Lemma 7.

We sum (7) for 0 ≤ t ≤ T − 1 to get, DISPLAYFORM1 We will then upper bound each sum in the right-hand side, DISPLAYFORM2 where u t+1 DISPLAYFORM3 Then noticing that z 0 def = ω 0 , back to (110) we get a telescoping sum, DISPLAYFORM4 (112) If F is the operator of a convex-concave saddle point (5), we get, with ω t = (θ t , ϕ t ) DISPLAYFORM5 then by convexity of L(·, ϕ) and concavity of L(θ, ·), we have that, DISPLAYFORM6 In both cases, we can now maximize the left hand side respect to ω (since the RHS does not depend on ω) to get, DISPLAYFORM7 Then taking the expectation, since E[∆ t |z t , DISPLAYFORM8 Published as a conference paper at ICLR 2019 DISPLAYFORM9 First let us state Theorem 2 in its general form, Theorem' 2.

Under Assumption 1, 2 and 4, Alg.

1 with constant step-size η has the following convergence rate for all T ≥ 1, DISPLAYFORM10 DISPLAYFORM11 Proof of Theorem 2.

Let any ω ∈ Ω such that ω 0 − ω 2 ≤ R, DISPLAYFORM12 (projections are non-contractive, Lemma 1) DISPLAYFORM13 Then we can make appear the quantity F (ω t ) (ω t − ω) on the left-hand side, DISPLAYFORM14 we can sum (118) for 0 ≤ t ≤ T − 1 to get, DISPLAYFORM15 where we noted DISPLAYFORM16 DISPLAYFORM17 We will then upper bound each sum in the right hand side, DISPLAYFORM18 where u t+1 def = P Ω (u t − η t ∆ t ) and u 0 = ω 0 .

Then, DISPLAYFORM19 Then noticing that u 0 def = ω 0 , back to (120) we get a telescoping sum, DISPLAYFORM20 Then the right hand side does not depends on ω, we can maximize over ω to get, DISPLAYFORM21 Noticing that E[∆ t |ω t , u t ] = 0 (the estimates of F are unbiased), by Assumption 2 DISPLAYFORM22 particularly for η t = η and η t = η √ t+1we respectively get, DISPLAYFORM23 and DISPLAYFORM24 Theorem' 3.

Under Assumption 1 and 4, if DISPLAYFORM25 has the following convergence rate for any T ≥ 1, DISPLAYFORM26 DISPLAYFORM27 Proof of Thm.

3. Let any ω ∈ Ω such that ω 0 − ω 2 ≤ R. Then, the update rules become ω t+1 = P Ω (ω t − η t F (ω t , ζ t )) and ω t = P Ω (ω t − ηF (ω t , ξ t )).

We start by applying Lemma 2 for (ω, u, ω , ω + ) = (ω t , −ηF (ω t , ζ t ), ω, ω t+1 ) and (ω, u, ω , ω DISPLAYFORM28 Then, summing them we get DISPLAYFORM29 Using the inequality DISPLAYFORM30 Then we can use the L-Lipschitzness of F to get, DISPLAYFORM31 As we restricted the step-size to η t ≤ 1 √ 3Lwe get, DISPLAYFORM32 We get a particular case of (7) so we can use Lemma 7 where DISPLAYFORM33 By Assumption 1, M 1 = M 2 = 3σ 2 and by the fact that DISPLAYFORM34 the hypothesis of Lemma 7 hold and we get, DISPLAYFORM35 has the following convergence rate for any T ≥ 1, DISPLAYFORM36 DISPLAYFORM37 First let us recall the update rule DISPLAYFORM38 Lemma 8.

We have for any ω ∈ Ω, DISPLAYFORM39 Proof.

Applying Lemma 2 for (ω, u, ω DISPLAYFORM40 and DISPLAYFORM41 Summing FORMULA0 and FORMULA0 we get, DISPLAYFORM42 DISPLAYFORM43 (135) DISPLAYFORM44 Then, we can use the inequality of arithmetic and geometric means 2a DISPLAYFORM45 Using the inequality a DISPLAYFORM46 where we used the L-Lipschitzness of F for the last inequality.

Combining FORMULA0 with FORMULA0 we get, DISPLAYFORM47 Lemma 9.

For all t ≥ 0, if we set ω −2 = ω −1 = ω 0 we have DISPLAYFORM48 Proof.

We start with a + b 2 2 ≤ 2 a 2 + 2 b 2 .

DISPLAYFORM49 Moreover, since the projection is contractive we have that DISPLAYFORM50 where in the last line we used the same inequality as in (142).

Combining FORMULA0 and FORMULA0 we get, DISPLAYFORM51 Proof of Theorem 4.

Combining Lemma 9 and Lemma 8 we get, DISPLAYFORM52 Then for DISPLAYFORM53 we have 36η DISPLAYFORM54 We can then use Lemma 7 where DISPLAYFORM55

We now consider a task similar to (Mescheder et al., 2018) where the discriminator is linear D ϕ (ω) = ϕ T ω, the generator is a Dirac distribution at θ, q θ = δ θ and the distribution we try to match is also a Dirac at ω * , p = δ ω * .

The minimax formulation from BID12 gives: min DISPLAYFORM0 Note that as observed by Nagarajan and Kolter (2017), this objective is concave-concave, making it hard to optimize.

We compare the methods on this objective where we take ω * = −2, thus the position of the equilibrium is shifted towards the position (θ, ϕ) = (−2, 0).

The convergence and the gradient vector field are shown in FIG7 .

We observe that depending on the initialization, some methods can fail to converge but extrapolation (18) seems to perform better than the other methods.

In addition to the results presented in section §7.2, we also trained the DCGAN architecture with the WGAN-GP objective.

The results are shown in Table 3 .

The best results are achieved with uniform averaging of AltAdam5.

However, its iterations require to update the discriminator 5 times for every generator update.

With a small drop in best final score, ExtraAdam can train WGAN-GP significantly faster (see Fig. 6 right) as the discriminator and generator are updated only twice.

In addition to the inception scores, we also computed the FID scores BID17 ) using 50,000 samples for the ResNet architecture with the WGAN-GP objective; the results are presented in TAB7 .

We see that the results and conclusions are similar to the one obtained from the inception scores, adding an extrapolation step as well as using Exponential Moving Average (EMA) consistently improves the FID scores.

However, contrary to the results from the inception score, we observe that uniform averaging does not necessarily improve the performance of the methods.

This could be due to the fact that the samples produced using uniform averaging are more blurry and FID is more sensitive to blurriness; see §G.3 for more details about the effects of uniform averaging.

Figure 6: DCGAN architecture with WGAN-GP trained on CIFAR10: mean and standard deviation of the inception score computed over 5 runs for each method using the best performing learning rate plotted over number of generator updates (Left) and wall-clock time (Right); all experiments were run on a NVIDIA Quadro GP100 GPU.

We see that ExtraAdam converges faster than the Adam baselines.

Number of generator updates Figure 7: Inception score on CIFAR10 for WGAN-GP (DCGAN) over number of generator updates for different learning rates.

We can see that AvgExtraAdam is less sensitive to the choice of learning rate.

In this section, we compare how the methods presented in §7 perform with the same step-size.

We follow the same protocol as in the experimental section §7, we consider the DCGAN architecture with WGAN-GP experiment described in App §G.2.

In Figure 7 we plot the inception score provided by each training method as a function of the number of generator updates.

Note that these plots advantage AltAdam5 a bit because each iteration of this algorithm is a bit more costly (since it perform 5 discriminator updates for each generator update).

Nevertheless, the goal of this experiment is not to show that AltAdam5 is faster but to show that ExtraAdam is less sensitive to the choice of learning rate and can be used with higher learning rates with less degradation.

In FIG11 , we compare the sample quality on the DCGAN architecture with the WGAN-GP objective of AltAdam5 and AvgExtraAdam for different step-sizes.

We notice that for AvgExtraAdam, the sample quality does not significantly change whereas the sample quality of AltAdam5 seems to be really sensitive to step-size tunning.

We think that robustness to step-size tuning is a key property for an optimization algorithm in order to save as much time as possible to tune other hyperparameters of the learning procedure such as regularization.

In this section, we compare how uniform averaging affect the performance of the methods presented in §7.

We follow the same protocol as in the experimental section §7, we consider the DCGAN architecture with the WGAN and weight clipping objective as well as the WGAN-GP objective.

In Figure 9 and 10, we plot the inception score provided by each training method as a function of the number of generator updates with and without uniform averaging.

We notice that uniform averaging seems to improve the inception score, nevertheless it looks like the sample are a bit more blurry (see FIG1 ).

This is confirmed by our result FIG1 Number of generator updates Number of generator updates Figure 12: The Fréchet Inception Distance (FID) from BID17 computed using 50,000 samples, on the WGAN experiments.

ReExtraAdam refers to Alg.

5 introduced in §D. We can see that averaging performs worse than when comparing with the Inception Score.

We observed that the samples generated by using averaging are a little more blurry and that the FID is more sensitive to blurriness, thus providing an explanation for this observation.

@highlight

We cast GANs in the variational inequality framework and import techniques from this literature to optimize GANs better; we give algorithmic extensions and empirically test their performance for training GANs.