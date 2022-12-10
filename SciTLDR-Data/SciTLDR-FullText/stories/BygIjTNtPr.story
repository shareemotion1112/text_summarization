Despite remarkable empirical success, the training dynamics of generative adversarial networks (GAN), which involves solving a minimax game using stochastic gradients, is still poorly understood.

In this work, we analyze last-iterate convergence of simultaneous gradient descent (simGD) and its variants under the assumption of convex-concavity, guided by a continuous-time analysis with differential equations.

First, we show that simGD, as is, converges with stochastic sub-gradients under strict convexity in the primal variable.

Second, we generalize optimistic simGD to accommodate an optimism rate separate from the learning rate and show its convergence with full gradients.

Finally, we present anchored simGD, a new method, and show convergence with stochastic subgradients.

Training of generative adversarial networks (GAN) (Goodfellow et al., 2014) , solving a minimax game using stochastic gradients, is known to be difficult.

Despite the remarkable empirical success of GANs, further understanding the global training dynamics empirically and theoretically is considered a major open problem (Goodfellow, 2016; Radford et al., 2016; Metz et al., 2017; Mescheder et al., 2018; Odena, 2019) .

The local training dynamics of GANs is understood reasonably well.

Several works have analyzed convergence assuming the loss functions have linear gradients and assuming the training uses full (deterministic) gradients.

Although the linear gradient assumption is reasonable for local analysis (even though the loss functions may not be continuously differentiable due to ReLU activation functions) such results say very little about global convergence.

Although the full gradient assumption is reasonable when the learning rate is small, such results say very little about how the randomness affects the training.

This work investigates global convergence of simultaneous gradient descent (simGD) and its variants for zero-sum games with a convex-concave cost using using stochastic subgradients.

We specifically study convergence of the last iterates as opposed to the averaged iterates.

Organization.

Section 2 presents convergence of simGD with stochastic subgradients under strict convexity in the primal variable.

The goal is to establish a minimal sufficient condition of global convergence for simGD without modifications.

Section 3 presents a generalization of optimistic simGD , which allows an optimism rate separate from the learning rate.

We prove the generalized optimistic simGD using full gradients converges, and experimentally demonstrate that the optimism rate must be tuned separately from the learning rate when using stochastic gradients.

However, it is unclear whether optimistic simGD is theoretically compatible with stochastic gradients.

Section 4 presents anchored simGD, a new method, and presents its convergence with stochastic subgradients.

Anchoring represents what we consider to be the strongest contribution of this work.

The presentation and analyses of Sections 2, 3, and 4 are guided by continuous-time firstorder ordinary differential equations (ODE).

In particular, we interpret optimism and anchoring as discretizations of certain regularized dynamics.

Section 5 experimentally demonstrates the benefit of optimism and anchoring for training GANs in some setups.

Prior work.

There are several independent directions for improving the training of GANs such as designing better architectures, choosing good loss functions, or adding appropriate regularizers (Radford et al., 2016; Sønderby et al., 2017; Gulrajani et al., 2017; Wei et al., 2018; Roth et al., 2017; Mescheder et al., 2018; 2017; Miyato et al., 2018) .

In this work, we accept these factors as a given and focus on how to train (optimize) the model effectively.

Optimism is a simple modification to remedy the cycling behavior of simGD, which can occur even under the bilinear convex-concave setup Daskalakis & Panageas, 2018; Mertikopoulos et al., 2019; Gidel et al., 2019a; Liang & Stokes, 2019; Mokhtari et al., 2019; Peng et al., 2019) .

These prior work assume the gradients are linear and use full gradients.

Although the recent name 'optimism' originates from its use in online optimization (Chiang et al., 2012; Rakhlin & Sridharan, 2013a; b; Syrgkanis et al., 2015) , the idea dates back to Popov's work in the 1980s (Popov, 1980) and has been studied independently in the mathematical programming community (Malitsky & Semenov, 2014; Malitsky, 2015; Malitsky & Tam, 2018; Malitsky, 2019; Csetnek et al., 2019) .

We note that there are other mechanisms similar to optimism and anchoring such as "prediction" (Yadav et al., 2018) , "negative momentum" (Gidel et al., 2019b) , and "extragradient" (Korpelevich, 1976; Tseng, 2000; Chavdarova et al., 2019) .

In this work, we focus on optimism and anchoring.

Classical literature analyze convergence of the Polyak-averaged iterates (which assigns less weight to newer iterates) when solving convex-concave saddle point problems using stochastic subgradients (Bruck, 1977; Nemirovski & Yudin, 1978; Nemirovski et al., 2009; Juditsky et al., 2011; Gidel et al., 2019a) .

For GANs, however, last iterates or exponentially averaged iterates (Yazıcı et al., 2019) (which assigns more weight to newer iterates) are used in practice.

Therefore, the classical work with Polyak averaging do not fully explain the empirical success of GANs.

We point out that we are not the first to utilize classical techniques for analyzing the training of GANs.

In particular, the stochastic approximation technique (Heusel et al., 2017; Duchi & Ruan, 2018) , control theoretic techniques (Heusel et al., 2017; Nagarajan & Kolter, 2017) , ideas from variational inequalities and monotone operator theory (Gemp & Mahadevan, 2018; Gidel et al., 2019a) , and continuous-time ODE analysis (Heusel et al., 2017; Csetnek et al., 2019) have been utilized for analyzing GANs.

Consider the cost function L : R m × R n → R and the minimax game min x max u L(x, u).

We say

We assume L is convex-concave and has a saddle point. (A0) By convex-concave, we mean L(x, u) is a convex function in x for fixed u and a concave function in u for fixed x. Define

where ∂ x and ∂ u respectively denote the subdifferential with respect to x and u. For simplicity, write z = (x, u) ∈ R m+n and G(z) = G(x, u).

Note that 0 ∈ G(z) if and only if z is a saddle point.

Since L is convex-concave, the operator G is monotone (Rockafellar, 1970) :

Let g(z; ω) be a stochastic subgradient oracle, i.e., E ω g(z; ω) ∈ G(z) for all z ∈ R m+n , where ω is a random variable.

Consider Simultaneous Stochastic Sub-Gradient Descent

for k = 0, 1, . . . , where z 0 ∈ R m+n is a starting point, α 0 , α 1 , . . .

are positive learning rates, and ω 0 , ω 1 , . . .

are IID random variables.

(We read SSSGD as "triple-SGD".)

In this section, we provide convergence of SSSGD when L(x, u) is strictly convex in x.

T (z − z ) = 0 so z(t) − z does not decrease and z(t) forms a cycle. (Right) L(x, u) = 0.2x 2 + xu.

The dashed line denotes where G(z)

T (z − z ) = 0, but it is visually clear that z = 0 is the only cluster point.

To understand the asymptotic dynamics of the stochastic discrete-time system, we consider a corresponding deterministic continuous-time system.

For simplicity, assume G is single-valued and smooth.

Considerż

with an initial value z(0) = z 0 . (We introduce g(t) for notational simplicity.) Let z be a saddle point, i.e., G(z ) = 0.

Then z(t) does not move away from z :

where we used (1).

However, there is no mechanism forcing z(t) to converge to a solution.

where x ∈ R and u ∈ R and ρ > 0.

Note that L 0 is the canonical counter example that also arises as the Dirac-GAN (Mescheder et al., 2018) .

See Figure 1 .

The classical LaSalle-Krasnovskii invariance principle (Krasovskii, 1959; LaSalle, 1960) states (paraphrased) if z ∞ is a cluster point of z(t), then the dynamics starting at z ∞ will have a constant distance to z .

On the left of Figure 1 , we can see z(t) − z 2 is constant as

for all t.

On the right of Figure 1 , we can see that although d dt 1 2 z(t) − z 2 = 0 when z(t) = (0, u) for u = 0 (the dotted line) this 0 derivative is temporary as z(t) will soon move past the dotted line.

Therefore, z(t) can maintain a constant constant distance to z only if it starts at 0, and 0 is the only cluster point of z(t).

Consider the further assumptions

where ω 1 and ω 2 are independent random variables and R 1 ≥ 0 and R 2 ≥ 0.

These assumptions are standard in the sense that analogous assumptions are used in convex minimization to establish almost sure convergence of stochastic gradient descent.

Theorem 1.

Assume (A0), (A1), and (A2).

Furthermore, assume L(x, u) is strictly convex in x for all u. Then SSSGD converges in the sense of z k a.s.

→

z where z is a saddle point of L.

We can alternatively assume L(x, u) is strictly concave in u for all x and obtain the same result.

The proof uses the stochastic approximation technique of (Duchi & Ruan, 2018) .

We show that the discrete-time process converges (in an appropriate topology) to a continuous-time trajectory satisfying a differential inclusion and use the LaSalle-Krasnovskii invariance principle to argue that cluster points are solutions.

Related prior work.

Theorem 3.1 of (Mertikopoulos et al., 2019) considers the more general mirror descent setup and proves convergence under the assumption of "strict coherence", which is analogous to the stronger assumption of strict convex-concavity in both x and u.

Consider the setup where L is continuously differentiable and we access full (deterministic) gradients

Consider Optimistic Simultaneous Gradient Descent

for k ≥ 0, where z 0 ∈ R m+n is a starting point, z −1 = z 0 , α > 0 is learning rate, and β > 0 is the optimism rate.

Optimism is a modification to simGD that remedies the cycling behavior; for the bilinear example L 0 of (2), simGD (case β = 0) diverges while SimGD-O with appropriate β > 0 converges.

In this section, we provide a continuous-time interpretation of SimGD-O as a regularized dynamics and provide convergence for the deterministic setup.

Consider the continuous-time dynamicṡ

We discuss how this system arises as a certain regularized dynamics and derive the convergence rate

Regularized gradient mapping.

The Moreau-Yosida (Moreau, 1965; Yosida, 1948) regularization of G with parameter β > 0 is

To clarify, I : R m+n → R m+n is the identity mapping and (I + βG) −1 is the inverse (as a function) of I + βG, which is well-defined by Minty's theorem (Minty, 1962) .

It is straightforward to verify that G β (z) = 0 if and only if G(z) = 0, i.e., G β and G share the same equilibrium points.

For small β, we can think of G β as an approximation G that is better-behaved.

Specifically, G is merely monotone (satisfies (1)), but G β is furthermore β-cocoercive, i.e.,

Regularized dynamics.

Consider the regularized dynamicṡ

Reparameterize the dynamicsζ(t) = −αG β (ζ(t)) with z(t) = (I + βG) −1 (ζ(t)) and g(t) = G(z(t)) to get ζ(t) = z(t) + βg(t) anḋ

This gives usż(t) = −αg(t) − βġ(t).

We use L 0 of (2) and Gaussian random noise.

The shaded region denotes ± standard error.

For simGD-OS, we see that neither q = 0 nor q = p leads to convergence.

Rather, q must satisfy 0 < q < p so that the learning rate diminishes faster than the optimism rate.

Training in machine learning usually relies on stochastic gradients, rather than full gradients.

We can consider a stochastic variation of SimGD-O:

with learning rate α k and optimism rate β k .

Figure 2 presents experiments of SimGD-OS on a simple bilinear problem.

The choice β k = α k where α k → 0 does not lead to convergence.

Discretizingż(t) = −αg(t)−βġ(t) with a diminishing step h k leads to the choice α k = αh k and β k = β, but this choice does not lead to convergence either.

Rather, it is necessary to tune α k and β k separately as in Theorem 2 to obtain convergence and dynamics appear to be sensitive to the choice of α k and β k .

In particular, both α k and β k must diminish and α k must diminish faster than β k .

One explanation of this difficulty is that the finite difference approximation α

is unreliable when using stochastic gradients.

Whether the observed convergence holds generally in the nonlinear convex-concave setup and whether optimism is compatible with subgradients is unclear.

This motivates anchoring of the following section which is provably compatible with stochastic subgradients.

Related prior work.

Gidel et al. (2019a) show averaged iterates of SimGD-OS converge if iterates are projected onto a compact set.

Mertikopoulos et al. (2019) show almost sure convergence of SimGD-OS under strict convex-concavity (and more generally under "strict coherence").

However, such analyses do not provide a compelling reason to use optimism since SimGD without optimism already converges under these setups.

Consider setup of Section 3.

We propose Anchored Simultaneous Gradient Descent

for k ≥ 0, where z 0 ∈ R m+n is a starting point, p ∈ (1/2, 1), and γ > 0 is the anchor rate.

In this section, we provide a continuous-time illustration of SimGD-A and provide convergence for both the deterministic and stochastic setups.

Consider the continuous-time dynamicṡ

for t ≥ 0, where γ ≥ 1 and z(0) = z 0 .

We will derive the convergence rate

Discretizing the continuous-time ODE with diminishing steps (1 − p)/(k + 1) p leads to SimGD-A.

Rate of convergence.

First note

Using this, we have d dt

2 and integrating both sides gives us

Reorganizing, we get

Using γ ≥ 1, the monotonicity inequality, and Young's inequality, we get

Interestingly, anchoring leads to a faster rate O(1/t 2 ) compared to the rate O(1/t) of optimism in continuous time.

The discretized method, however, is not faster than O(1/k).

We further discuss this difference in Section A.

Related prior work.

Anchoring was inspired by Halpern's method (Halpern, 1967; Wittmann, 1992; Lieder, 2017) and James-Stein estimator (Stein, 1956; James & Stein, 1961) ; these methods pull/shrink the iterates/estimator towards a specified point z 0 .

We now present convergence results with anchoring.

In Theorem 3, we use deterministic gradients, and in Theorem 4, we use stochastic subgradients.

Theorem 3.

Assume (A0) and (A3).

If p ∈ (1/2, 1) and γ ≥ 2, then SimGD-A converges in the sense of

The proof can be considered a discretization of the continuous-time analysis.

Consider the setup of Section 2.

We propose Anchored Simultaneous Stochastic SubGradient Descent

(The small ε > 0 is introduced for the proof of Theorem 4.

See Section A for further discussion.) (To clarify, we do not assume L is differentiable.)

Main contribution.

To the best of our knowledge, Theorem 4 is the first result establishing lastiterate convergence for convex-concave cost functions using stochastic subgradients without assuming strict convexity or analogous assumptions.

In this section, we experimentally demonstrate the effectiveness of optimism and anchoring for training GANs.

We train Wasserstein-GANs with gradient penalty (Gulrajani et al., 2017) on the MNIST and CIFAR-10 dataset and plot the Fréchet Inception Distance (FID) (Heusel et al., 2017; Lucic et al., 2018) .

The experiments were implemented in PyTorch (Paszke et al., 2017) .

We combine Adam with optimism and anchoring (described precisely in Appendix G) and compare it against the baseline Adam optimizer (Kingma & Ba, 2015) .

The generator and discriminator architectures and the hyperparameters are described in Appendix G. For optimistic and anchored Adam, we roughly tune the optimism and anchor rates and show the curve corresponding to the best parameter choice.

Figure 4 shows that the MNIST setup benefits from anchoring but not from optimism, while the CIFAR-10 setup benefits from optimism but not from anchoring.

We leave comparing the effects of optimism and anchoring in practical GAN training (where the cost function is not convex-concave) as a topic of future work.

In this work, we analyzed the convergence of SSSGD, Optimistic simGD, and Anchored SSSGD.

Under the assumption that the cost L is convex-concave, Anchored SSSGD provably converges under the most general setup.

Through experiments, we showed that the practical GAN training benefits from optimism and anchoring in some (but not all) setups.

Generalizing these results to accommodate projections and proximal operators, analogous to projected and proximal gradient methods, is an interesting direction of future work.

Weight clipping and spectral normalization (Miyato et al., 2018) A FURTHER DISCUSSION ON THE CONVERGENCE RESULTS Theorems 1, 2, 3, and 4 use related but different notions of convergence.

Theorems 1 and 4 are asymptotic (has no rate) while Theorems 2 and 3 are non-asymptotic (has a rate).

Theorems 1 and 3 respectively show almost sure and L 2 convergence of the iterates.

Theorems 2 and 3 show convergence of the squared gradient norm for the best and last iterates, respectively.

We did not make these choices.

The choices were dictated by what we can prove based on the analysis.

The discrete-time analysis of SimGD-O of Theorem 2 bounds the squared gradient norm of the best iterate, while the continuous-time analysis bounds the squared gradient norm of the "last iterate" (at terminal time).

The discrepancy comes from the fact that while we have monotonic decrease of g(t) in continuous-time, we have no analogous monotonicity condition on g k in discrete-time.

To the best of our knowledge, there is no result establishing a O(1/k) rate on the squared gradient norm of the last iterate for SimGD-O or the related "extragradient method" Korpelevich (1976) .

Theorem 3 is the first result showing a rate close to O(1/k) on the last literate.

For SimGD-O and Corollary 1, the parameter choices are almost optimal.

The optimal choices that minimize the bound of Theorem 2 are α = 0.124897/R and β = 1.94431α; they provide a factor of 135.771, a very small improvement over the factor 136 of Corollary 1.

For SimGD-A and Theorem 3, there is a discrepancy in the rate between the continuous time analysis O(1/t 2 ) and the discrete time rate O(1/k 2−2p ) for p ∈ (1/2, 1), which is slightly slower than O(1/k).

In discretizing the continuous-time calculations to obtain a discrete proof, errors accumulate and prevent the rate from being better than O(1/k).

This is not an artifact of the proof.

Simple tests on bilinear examples show divergence when p < 1/2.

SSSGD-A and Theorem 4 involves the parameter ε.

While the proof requires ε > 0, we believe this is an artifact of the proof.

In particular, we conjecture that Lemma 17 holds with o(s/τ ) rather than O(s/τ ), and, if so, it is possible to establish convergence with ε = 0.

In Figure 2 , it seems that that the choice ε = 0 and p = 2/3 is optimal for SSSGD-A. While we do not have a theoretical explanation for this, we point out that this is not surprising as p = 2/3 is known to be optimal in stochastic convex minimization (Moulines & Bach, 2011; Taylor & Bach, 2019) .

Theorems 2, 3, and 4 extend to monotone operators (Ryu & Boyd, 2016; Bauschke & Combettes, 2017) without any modification to their proofs.

In infinite dimensional setups (which is of interest in the field of monotone operators) Theorem 4 establishes strong convergence, while many convergence results (including Theorems 2 and 3) establish weak convergence.

However, Theorem 1 does not extend to monotone operators, as the use of the LaSalle-Krasnovskii principle is particular to convex-concave saddle functions.

Write R + to denote the set of nonnegative real numbers and ·, · to denote inner product, i.e.,

We say A is a point-to-set mapping on R d if A maps points of R d to subsets of R d .

For notational simplicity, we write

Using this notation, we define monotonicity of A with

where the inequality requires every member of the set to be nonnegative.

We say a monotone operator A is maximal if there is no other monotone operator B such that the containment

is maximal monotone (Rockafellar, 1970

(In other words, the graph of G is closed.)

Define Zer(G) = {z ∈ R d | 0 ∈ G(z)}, which is the set of saddle-points or equilibrium points.

When G is maximal monotone, Zer(G) is a closed convex set.

Write

for the projection onto Zer(G).

In other words, we consider the topology of uniform convergence on compact sets.

We rely on the following inequalities, which hold for any a, b ∈ R m+n any ε > 0.

Both inequalities are called Young's inequality.

(Note, (6) follows from (5) with ε = 1.)

Lemma 1 (Theorem 5.3.33 of Dembo (2019)).

Let {F k } k∈N+ be an increasing sequence of σ-algebras.

Let (m k , F k ) be a martingale such that

then m k converges almost surely to a limit.

& Siegmund (1971) ).

Let {F k } k∈N+ be an increasing sequence of σ-algebras.

Let {V k } k∈N+ , {S k } k∈N+ , {U k } k∈N+ , and {β k } k∈N+ be nonnegative F k -measurable random sequences satisfying

almost surely, where V ∞ is a random limit.

Note that 0 =G(z ) is possible even if 0 ∈ G(z ) when L is not continuously differentiable.

Lemma 3.

Under Assumptions (A0) and (A2), we have

for some R 3 > 0 and R 4 > 0.

Proof.

Let z be a saddle point, which exists by Assumption (A0).

Let ω and ω be independent and identically distributed.

Then

where we use the fact that g(z ; ω ) −G(z ) is a zero-mean random variable, Assumption (A2), and (6).

The stated result holds with R C ANALYSIS OF THEOREM 1

For convenience, we restate the update, assumptions, and the theorem:

L is convex-concave and has a saddle point (A0) → z where z is a saddle point of L.

Differential inclusion technique.

We use the differential inclusion technique of Duchi & Ruan (2018) , also recently used in Davis et al. (2019) .

The high-level summary of the technique is very simple and elegant: (i) show the discrete-time process converges to a continuous-time trajectory satisfying a differential inclusion, (ii) show any solution of the differential inclusion has a desirable property, and (iii) translate the conclusion in continuous-time to discrete-time.

However, the actual execution of this technique does require careful and technical considerations.

Proof outline.

For step (i), we adapt the LaSalle-Krasnovskii principle to show that a solution of the continuous-time differential inclusion converges to a saddle point.

(Lemma 5.)

Then we carry out step (ii) showing the time-shifted interpolated discrete time process converges to a solution of the differential inclusion.

(Lemma 6.) Finally, step (iii), the "Continuous convergence to discrete convergence", combines these two pieces to conclude that the discrete time process converges to a saddle point.

The contribution and novelty of our proof is in our steps (i) and (iii).

Preliminary definitions and results.

Consider the differential inclusioṅ

with the initial condition z(0) = z 0 .

We say z : [0, ∞) → R m+n satisfies (7) if there is a Lebesgue integrable ζ : [0, ∞) →

R m+n such that

Write z(t) = φ t (z 0 ) and call φ t : R m+n → R m+n the time evolution operator.

In other words, φ t maps the initial condition of the differential inclusion to the point at time t, which is well defined by the following result.

Lemma 4 (Theorem 5.2.1 of Aubin & Cellina (1984) ).

If G is maximal monotone, the solution to (7) exists and is unique.

Furthermore, φ t : R m+n → R m+n is 1-Lipschitz continuous for all t ≥ 0.

Lemma 5 and its proof can be considered an adaptation of the LaSalle-Krasnovskii invariance principle (Krasovskii, 1959; LaSalle, 1960) to the setup of differential inclusions.

The standard result applies to differential equations.

Lemma 5 (LaSalle-Krasnovskii).

Assume (A0).

Assume L(x, u) is strictly convex in x for all u. If z(·) satisfies (7), then z(t) → z ∞ as t → ∞ and z ∞ ∈ Zer(G).

Proof.

Consider any z ∈ Zer(G), which exists by Assumption (A0).

Since z(t) is absolutely continuous, so is z(t) − z 2 , and we have

for almost all t > 0, where ζ(·) is as defined in (8) and the inequality follows from (1), monotonicity of G. Therefore, z(t) − z 2 is a nonincreasing function of t, and lim t→∞ z(t) − z = χ for some limit χ ≥ 0.

Since z(t) is a bounded sequence, it has at least one cluster point.

Since φ t (·) (with fixed t) is continuous by Lemma 4, we have

for all s ≥ 0.

This means φ s (z ∞ ) is also a cluster point of z(·) and

for almost all s ≥ 0.

by strict convexity, and, in light of (9), we conclude φ x s (z ) = x for almost all s ≥ 0.

Then for almost all s ≥ 0, we have

where the first inequality follows from concavity of L(x, u) in u and the second inequality follows from the fact that u is a maximizer when x is fixed.

Therefore, we have equality throughout, and

is a continuous function of s for all s ≥ 0.

Therefore, that φ x s (z ∞ ) = x and that φ u s (z ∞ ) maximizes L(x , ·) for almost all s ≥ 0 imply that the conditions hold for s = 0.

In other words, x ∞ = x and u ∞ maximizes L(x , ·), and therefore z ∞ ∈ ZerG.

Finally, since z ∞ is a solution, z(t)−z ∞ converges to a limit as t → ∞. Since z(t k )

−z ∞ → 0, we conclude that z(t) − z ∞ → 0 as t → ∞.

The following lemma is the crux of the differential inclusion technique.

It makes precise in what sense the discrete-time process converges to a solution of the continuous-time differential inclusion.

Lemma 6 (Theorem 3.7 of Duchi & Ruan (2018) ).

Consider the update

Define the time-shifted process z

Let the following conditions hold:

(i) The iterates are bounded, i.e., sup k z k < ∞ and sup k ζ k < ∞.

(ii) The stepsizes α k satisfy Assumption (A1).

(iii) The weighted noise sequence converges:

(iv) For any increasing sequence n k such that z n k → z ∞ , we have

Then for any sequence {τ k } ∞ k=1 ⊂ R + , the sequence of functions {z

We verify the conditions of Lemma 6 and make the argument that the noisy discrete time process is close to the noiseless continuous time process and the two processes converge to the same limit.

Verifying conditions of Lemma 6.

Condition (i).

Let z ∈ Zer(G).

Write F k for the σ-field generated by ω 0 , . . .

, ω k−1 .

Writẽ

where we used Assumption (A2) and Lemma 3.

Since ∞ k=0 α 2 k < ∞ by Assumption (A1), this inequality and Lemma 2 tells us z k − z 2 → limit for some limit, which implies z k is a bounded sequence.

Since z k is bounded, so isG(z k ) since

by Lemma 3.

This condition is assumed.

< ∞ almost surely, where the first inequality is the second moment upper bounding the variance, the second inequality is Lemma 3, and the third inequality is (6) and condition (i).

Finally, we have (iii) by Lemma 1.

As discussed in Section B, G is maximal monotone, which implies G is upper semicontinuous, i.e., (

, and G(z ∞ ) is a closed convex set.

Therefore, dist(ζ n k , G(z ∞ )) → 0 as otherwise we can find a further subsequence such that converging to ζ ∞ such that dist(ζ ∞ , G(z ∞ )) > 0. (Here we use the fact that ζ k is bounded due to

In the main proof, we show that cluster points of z interp (·) are solutions.

We need the following lemma to conclude that these cluster points are also cluster points of the original discrete time process z k .

Lemma 7.

Under the conditions of Lemma 6, z interp (·) and z k share the same cluster points.

Proof.

If z ∞ is a cluster point of z k , then it is a cluster point of z interp (·) by definition.

Assume z ∞ is a cluster point of z interp (·), i.e., assume there is a sequence

where we use the assumption (i) which states that ζ k is bounded and assumption (iii) which states that α k ξ k → 0.

We conclude z kj → z ∞ .

Continuous convergence to discrete convergence.

Let k j → ∞ be a subsequence such that z kj → z ∞ .

Let k j → ∞ be a further subsequence such that

for all T ≥ 0, which exists by Lemma 6.

(The time-shifted interpolated process converges to a solution of the differential inclusion.)

By Lemma 5,

where φ t z ∞ → φ ∞ z ∞ as t → ∞ and φ ∞ z ∞ is a saddle point.

(The solution to the differential inclusion converges to a solution.)

These facts together imply that for any ε > 0, there exists k j and τ j large enough that

.

Therefore, φ ∞ z ∞ is a cluster point of z interp (·), and, by Lemma 7, φ ∞ z ∞ is a cluster point of z k .

Since z k −φ ∞ z ∞ converges to a limit and converges to 0 on this further subsequence, we conclude

For convenience, we restate the update, assumptions, and the theorem:

L is convex-concave and has a saddle point (A0) L is differentiable and ∇L is R-Lipschitz continuous (A3)

Theorem 2.

Assume (A0) and (A3).

If 0 < α < 2β(1 − 2βR), then SimGD-O converges in the sense of

Furthermore, z k → z , where z is a saddle point of L.

Throughout this section, write g k = G(z k ) for k ≥ −1.

Since we can defineG = αG andβ = β/α and write the iteration as

we assume α = 1 without loss of generality.

Then

where the inequality follows from (1), monotonicity of G, and

We can bound

where the first inequality follows from (5), Young's inequality, with ε = R and the second inequality follows from Assumption (A3), R-Lipschitz continuity of G. Putting these together we get

Since β > 1/2 and R < (2β − 1)/(4β 2 ) is assumed for Theorem 2, we have

The proof follows from a basic application of the inequality

and p ∈ (0, 1).

Lemma 9.

For p ∈ (0, 1) and k ≥ 1,

The proof follows from integrating the decreasing function p/x 1−p from k to k + 1.

Lemma 10.

For p ∈ (0, 1) and k ≥ 1,

The proof follows from Lemma 8.

Lemma 11.

Given any V 0 , V 1 , . . .

∈ R, we have

The proof follows from basic calculations.

This result can be thought of as the discrete analog of

Lemma 12.

Let z 0 , z 1 , . . .

∈ R m+n be an arbitrary sequence.

Then for any k = 0, 1, . . .

,

The proof follows from basic calculations.

This result can be thought of as the discrete analog of

In the proofs of Theorems 3 and 4, we establish certain descent inequalities.

The following lemmas state that these inequalities imply boundedness or convergence.

Lemma 13.

Let {V k } k∈N+ and {U k } k∈N+ be nonnegative (deterministic) sequences satisfying

Proof.

For any δ ∈ (0, C 1 ), there is a large enough K ≥ 0 such that for all k ≥ K,

With a standard recursion argument (e.g. Lemma 3 of (Polyak, 1987)) we conclude max {0, V k − ν} → 0.

Since this holds for any δ > 0, we conclude lim sup k→∞ V k ≤ C 2 2 /C 2 1 .

Lemma 14.

Let ε ∈ (0, 1).

Let {V k } k∈N+ and {U k } k∈N+ be nonnegative (deterministic) sequences satisfying

Proof.

For any δ > 0, there is a large enough K ≥ 0 such that

Since this holds for all δ > 0, we conclude V k → 0.

For convenience, we restate the update, assumptions, and the theorem:

L is convex-concave and has a saddle point (A0) L is differentiable and ∇L is R-Lipschitz continuous

where we use the assumption that γ ≥ 2.

Reorganizing again, we get

, where the second inequality follows from (1), the monotonicity inequality, and the third inequality follows from (5), Young's inquality, with ε = γ/(k + 1) 1−p .

Finally, we have

F ANALYSIS OF THEOREM 4

For convenience, we restate the update, assumptions, and the theorem:

L is convex-concave and has a saddle point (A0) E ω1,ω2 g(z 1 ; ω 1 ) − g(z 2 ; ω 2 ) 2 ≤ R To clarify, we do not assume L is differentiable for Theorem 4.

Proof outline.

The key insight is to define ζ k to be something like a "fixed point" of the k-th iteration of SSSGD-A and then to show z k shrinks towards to ζ k in the following sense z k+1 − ζ k+1 2 ≤ (1 − something) z k − ζ k 2 + (something small).

Lemma 17 states that ζ k slowly (stably) converges to a solution.

Using the fact that z k shrinks towards ζ k and the fact that ζ k is a slowly moving target converging to a solution, we conclude z k converges to a solution.

Preliminary definition and result.

More precisely, we define ζ k to satisfy

(However, ζ k is not actually a fixed point, since SSSGD-A has noise and since G is a multi-valued operator.)

We equivalently write Optimism rate ρ = 1 Anchor rate γ = 1 Anchor refresh period T = 10000 Table 2 : Hyperparameters for the MNIST experiment Generator latent space 128 (Gaussian noise) dense 4 × 4 × 512 batchnorm ReLU 4 × 4 conv.

T stride=2 256 batchnorm ReLU 4 × 4 conv.

T stride=2 128 batchnorm ReLU 4 × 4 conv.

T stride=2 64 batchnorm ReLU 4 × 4 conv.

T stride=1 3 weightnorm tanh Discriminator Input Image 32 × 32 × 3 3 × 3 conv.

stride=1 64 lReLU 3 × 3 conv.

stride=2 128 lReLU 3 conv.

stride=1 128 lReLU 3 conv.

stride=2 256 lReLU 3 conv.

stride=1 256 lReLU 3 conv.

stride=2 512 lReLU 3 conv.

stride=1 512 lReLU dense 1 Table 3 : Generator and discriminator architectures for the CIFAR-10 experiment batch size = 64 Adam learning rate = 0.0001

Adam β 1 = 0.0 Adam β 2 = 0.9 max iteration = 100000 GAN objctive = "WGAN-GP" Gradient penalty parameter λ = 1 n dis = 1 Optimizer = "Adam", "Optimistic Adam", or "Anchored Adam"

Optimism rate ρ = 1 Anchor rate γ = 1 Anchor refresh period T = 10000

@highlight

Convergence proof of stochastic sub-gradients method and variations on convex-concave minimax problems

@highlight

An anaysis of simultaneous stochastic subgradient, simultaneous gradient with optimism, and simultaneous gradient with anchoring in the context of minmax convex concave games.

@highlight

This paper analyzes the dynamics of stochastic gradient descent when applied to convex-concave games, as well as GD with optimism and a new anchored GD algorithm that converges under weaker assumptions than SGD or SGD with optimism.