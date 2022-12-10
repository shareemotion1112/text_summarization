Min-max formulations have attracted great attention in the ML community due to the rise of deep generative models and adversarial methods, and understanding the dynamics of (stochastic) gradient algorithms for solving such formulations has been a grand challenge.

As a first step, we restrict to bilinear zero-sum games and give a systematic analysis of popular gradient updates, for both simultaneous and alternating versions.

We provide exact conditions for their convergence and find the optimal parameter setup and convergence rates.

In particular, our results offer formal evidence that alternating updates converge "better" than simultaneous ones.

Min-max optimization has received significant attention recently due to the popularity of generative adversarial networks (GANs) (Goodfellow et al., 2014) and adversarial training (Madry et al., 2018) , just to name some examples.

Formally, given a bivariate function f (x, y), we aim to find a saddle point (x * , y * ) such that f (x * , y) ≤ f (x * , y * ) ≤ f (x, y * ), ∀x ∈ R n , ∀y ∈ R n .

(1.1)

Since the beginning of game theory, various algorithms have been proposed for finding saddle points (Arrow et al., 1958; Dem'yanov & Pevnyi, 1972; Gol'shtein, 1972; Korpelevich, 1976; Rockafellar, 1976; Bruck, 1977; Lions, 1978; Nemirovski & Yudin, 1983; Freund & Schapire, 1999) .

Due to its recent resurgence in ML, new algorithms specifically designed for training GANs were proposed (Daskalakis et al., 2018; Kingma & Ba, 2015; Gidel et al., 2019b; Mescheder et al., 2017) .

However, due to the inherent non-convexity in deep learning formulations, our current understanding of the convergence behaviour of new and classic gradient algorithms is still quite limited, and existing analysis mostly focused on bilinear games or strongly-convex-strongly-concave games (Tseng, 1995; Daskalakis et al., 2018; Gidel et al., 2019b; Liang & Stokes, 2019; Mokhtari et al., 2019b) .

Nonzero-sum bilinear games, on the other hand, are known to be PPAD-complete (Chen et al., 2009 ) (for finding approximate Nash equilibria, see e.g. Deligkas et al. (2017) ).

In this work, we study bilinear zero-sum games as a first step towards understanding general min-max optimization, although our results apply to some simple GAN settings (Gidel et al., 2019a) .

It is well-known that certain gradient algorithms converge linearly on bilinear zero-sum games (Liang & Stokes, 2019; Mokhtari et al., 2019b; Rockafellar, 1976; Korpelevich, 1976) .

These iterative algorithms usually come with two versions: Jacobi style updates or Gauss-Seidel (GS) style.

In a Jacobi style, we update the two sets of parameters (i.e., x and y) simultaneously whereas in a GS style we update them alternatingly (i.e., one after the other).

Thus, Jacobi style updates are naturally amenable to parallelization while GS style updates have to be sequential, although the latter is usually found to converge faster (and more stable).

In numerical linear algebra, the celebrated Stein-Rosenberg theorem (Stein & Rosenberg, 1948) formally proves that in solving certain linear systems, GS updates converge strictly faster than their Jacobi counterparts, and often with a larger set of convergent instances.

However, this result does not readily apply to bilinear zero-sum games.

Our main goal here is to answer the following questions about solving bilinear zero-sum games:

• When exactly does a gradient-type algorithm converge?

• What is the optimal convergence rate by tuning the step size or other parameters?

• Can we prove something similar to the Stein-Rosenberg theorem for Jacobi and GS updates?

Table 2 : Optimal convergence rates.

In the second column, β * denotes a specific parameter that depends on σ 1 and σ n (see equation 4.2).

In the third column, the linear rates are for large κ.

The optimal parameters for both Jacobi and Gauss-Seidel EG algorithms are the same.

α denotes the step size (α 1 = α 2 = α), and β 1 and β 2 are hyper-parameters for EG and OGD, as given in §2.

Algorithm α β 1 β 2 rate exponent Comment

Jacobi and Gauss-Seidel Jacobi OGD 2β 1 β * β 1 ∼ 1 − 1/(6κ 2 ) β 1 = β 2 = α/2 GS OGD √ 2/σ 1 √ 2σ 1 /(σ Contributions We summarize our main results from §3 and §4 in Table 1 and 2 respectively, with supporting experiments given in §5.

We use σ 1 and σ n to denote the largest and the smallest singular values of matrix E (see equation 2.1), and κ := σ 1 /σ n denotes the condition number.

The algorithms will be introduced in §2.

Note that we generalize gradient-type algorithms but retain the same names.

Table 1 shows that in most cases that we study, whenever Jacobi updates converge, the corresponding GS updates converge as well (usually with a faster rate), but the converse is not true ( §3).

This extends the well-known Stein-Rosenberg theorem to bilinear games.

Furthermore, Table 2 tells us that by generalizing existing gradient algorithms, we can obtain faster convergence rates.

In the study of GAN training, bilinear games are often regarded as an important simple example for theoretically analyzing and understanding new algorithms and techniques (e.g. Daskalakis et al., 2018; Gidel et al., 2019a; b; Liang & Stokes, 2019) .

It captures the difficulty in GAN training and can represent some simple GAN formulations (Arjovsky et al., 2017; Daskalakis et al., 2018; Gidel et al., 2019a; Mescheder et al., 2018) .

Mathematically, bilinear zero-sum games can be formulated as the following min-max problem:

min x∈R n max y∈R n x Ey + b x + c y.

The set of all saddle points (see definition in eq. (1.1)) is:

Throughout, for simplicity we assume E to be invertible, whereas the seemingly general case with non-invertible E is treated in Appendix G.

The linear terms are not essential in our analysis and we take b = c = 0 throughout the paper 1 .

In this case, the only saddle point is (0, 0).

For bilinear games, it is well-known that simultaneous gradient descent ascent does not converge (Nemirovski & Yudin, 1983 ) and other gradient-based algorithms tailored for min-max optimization have been proposed (Korpelevich, 1976; Daskalakis et al., 2018; Gidel et al., 2019a; Mescheder et al., 2017) .

These iterative algorithms all belong to the class of general linear dynamical systems (LDS, a.k.a.

matrix iterative processes).

Using state augmentation z (t) := (x (t) , y (t) ) we define a general k-step LDS as follows:

where the matrices A i and vector d depend on the gradient algorithm (examples can be found in Appendix C.1).

Define the characteristic polynomial, with A 0 = −I:

The following well-known result decides when such a k-step LDS converges for any initialization: Theorem 2.1 (e.g. Gohberg et al. (1982) ).

The LDS in eq. (2.3) converges for any initialization (z (0) , . . .

, z (k−1) ) iff the spectral radius r := max{|λ| : p(λ) = 0} < 1, in which case {z (t) } converges linearly with an (asymptotic) exponent r.

Therefore, understanding the bilinear game dynamics reduces to spectral analysis.

The (sufficient and necessary) convergence condition reduces to that all roots of p(λ) lie in the (open) unit disk, which can be conveniently analyzed through the celebrated Schur's theorem (Schur, 1917) Schur (1917) ).

The roots of a real polynomial p(λ) = a 0 λ n + a 1 λ n−1 + · · · + a n are within the (open) unit disk of the complex plane iff ∀k ∈ {1, 2, . . .

, n}, det(

In the theorem above, we denoted 1 S as the indicator function of the event S, i.e. 1 S = 1 if S holds and 1 S = 0 otherwise.

For a nice summary of related stability tests, see Mansour (2011) .

We therefore define Schur stable polynomials to be those polynomials whose roots all lie within the (open) unit disk of the complex plane.

Schur's theorem has the following corollary (proof included in Appendix B.2 for the sake of completeness): Corollary 2.1 (e.g. Mansour (2011)) .

A real quadratic polynomial λ 2 + aλ + b is Schur stable iff b < 1, |a| < 1 + b; A real cubic polynomial λ 3 + aλ 2 + bλ + c is Schur stable iff |c| < 1,

Let us formally define Jacobi and GS updates: Jacobi updates take the form

while Gauss-Seidel updates replace x (t−i) with the more recent x (t−i+1) in operator T 2 , where T 1 , T 2 : R nk × R nk → R n can be any update functions.

For LDS updates in eq. (2.3) we find a nice relation between the characteristic polynomials of Jacobi and GS updates in Theorem 2.3 (proof in Appendix B.1), which turns out to greatly simplify our subsequent analyses:

and L i is strictly lower block triangular.

Then, the characteristic polynomial of Jacobi updates is p(λ, 1) while that of Gauss-Seidel updates is p(λ, λ).

Compared to the Jacobi update, in some sense the Gauss-Seidel update amounts to shifting the strictly lower block triangular matrices L i one step to the left, as p(λ, λ) can be rewritten as det

This observation will significantly simplify our comparison between Jacobi and Gauss-Seidel updates.

Next, we define some popular gradient algorithms for finding saddle points in the min-max problem

We present the algorithms for a general (bivariate) function f although our main results will specialize f to the bilinear case in eq. (2.1).

Note that we introduced more "step sizes" for our refined analysis, as we find that the enlarged parameter space often contains choices for faster linear convergence (see §4).

We only define the Jacobi updates, while the GS counterparts can be easily inferred.

We always use α 1 and α 2 to define step sizes (or learning rates) which are positive.

The generalized GD update has the following form:

When α 1 = α 2 , the convergence of averaged iterates (a.k.a.

Cesari convergence) for convex-concave games is analyzed in (Bruck, 1977; Nemirovski & Yudin, 1978; Nedić & Ozdaglar, 2009 ).

We study a generalized version of EG, defined as follows:

EG was first proposed in Korpelevich (1976) with the restriction α 1 = α 2 = γ 1 = γ 2 , under which linear convergence was proved for bilinear games.

A slightly more generalized version was analyzed in Liang & Stokes (2019) where α 1 = α 2 , γ 1 = γ 2 , again with linear convergence proved.

For later convenience we define β 1 = α 2 γ 1 and β 2 = α 1 γ 2 .

Optimistic gradient descent (OGD) We study a generalized version of OGD, defined as follows:

10) The original version of OGD was given in Daskalakis et al. (2018) with α 1 = α 2 = 2β 1 = 2β 2 , and its linear convergence for bilinear games was proved in Liang & Stokes (2019) .

A slightly more generalized version with α 1 = α 2 and β 1 = β 2 was analyzed in Mokhtari et al. (2019b) , again with linear convergence proved.

Momentum method Generalized heavy ball method was analyzed in Gidel et al. (2019b) :

11)

(2.12) This is a modification of Polyak's heavy ball (HB) (Polyak, 1964) , which also motivated Nesterov's accelerated gradient algorithm (NAG) (Nesterov, 1983) .

Note that for both x-update and the y-update, we add a scale multiple of the successive difference (e.g. proxy of the momentum).

For this algorithm our result below improves those obtained in Gidel et al. (2019b) , as will be discussed in §3.

EG and OGD as approximations of proximal point algorithm It has been observed recently in Mokhtari et al. (2019b) that for convex-concave games, EG (α 1 = α 2 = γ 1 = γ 2 = η) and OGD (α 1 /2 = α 2 /2 = β 1 = β 2 = η) can be treated as approximations of the proximal point algorithm (Martinet, 1970; Rockafellar, 1976) when η is small.

With this result, one can show that EG and OGD converge to saddle points sublinearly for smooth convex-concave games (Mokhtari et al., 2019a) .

We give a brief introduction of the proximal point algorithm in Appendix A (including a linear convergence result for the slightly generalized version).

The above algorithms, when specialized to a bilinear function f (see eq. (2.1)), can be rewritten as a 1-step or 2-step LDS (see. eq. (2.3)).

See Appendix C.1 for details.

With tools from §2, we formulate necessary and sufficient conditions under which a gradient-based algorithm converges for bilinear games.

We sometimes use "J" as a shorthand for Jacobi style updates and "GS" for Gauss-Seidel style updates.

For each algorithm, we first write down the characteristic polynomials (see derivation in Appendix C.1) for both Jacobi and GS updates, and present the exact conditions for convergence.

Specifically, we show that in many cases the GS convergence regions strictly include the Jacobi convergence regions.

The proofs for Theorem 3.1, 3.2, 3.3 and 3.4 can be found in Appendix C.2, C.3, C.4, and C.5, respectively.

GD The characteristic equations can be computed as:

Scaling symmetry From section 3 we obtain a scaling symmetry (α 1 , α 2 ) → (tα 1 , α 2 /t), with t > 0.

With this symmetry we can always fix α 1 = α 2 = α.

This symmetry also holds for EG and momentum.

For OGD, the scaling symmetry is slightly different with (α 1 , β 1 , α 2 , β 2 ) → (tα 1 , tβ 1 , α 2 /t, β 2 /t), but we can still use this symmetry to fix α 1 = α 2 = α.

Theorem 3.1 (GD).

Jacobi GD and Gauss-Seidel GD do not converge.

However, Gauss-Seidel GD can have a limit cycle while Jacobi GD always diverges.

When α 1 = α 2 , this theorem was proved by Gidel et al. (2019a) .

EG The characteristic equations can be computed as:

Theorem 3.2 (EG).

For generalized EG with α 1 = α 2 = α and γ i = β i /α, Jacobi and Gauss-Seidel updates achieve linear convergence iff for any singular value σ of E, we have:

, the convergence region of GS updates strictly include that of Jacobi updates.

OGD The characteristic equations can be computed as:

Theorem 3.3 (OGD).

For generalized OGD with α 1 = α 2 = α, Jacobi and Gauss-Seidel updates achieve linear convergence iff for any singular value σ of E, we have:

The convergence region of GS updates strictly include that of Jacobi updates.

Momentum The characteristic equations can be computed as:

Theorem 3.4 (momentum).

For the generalized momentum method with α 1 = α 2 = α, the Jacobi updates never converge, while the GS updates converge iff for any singular value σ of E, we have:

This condition implies that at least one of β 1 , β 2 is negative.

Prior to our work, only sufficient conditions for linear convergence were given for the usual EG and OGD; see §2 above.

For the momentum method, our result improves upon Gidel et al. (2019b) where they only considered specific cases of parameters.

For example, they only considered β 1 = β 2 ≥ −1/16 for Jacobi momentum (but with explicit rate of divergence), and β 1 = −1/2, β 2 = 0 for GS momentum (with convergence rate).

Our Theorem 3.4 gives a more complete picture and formally justifies the necessity of negative momentum.

In the theorems above, we used the term "convergence region" to denote a subset of the parameter space (with parameters α, β or γ) where the algorithm converges.

Our result shares similarity with the celebrated Stein-Rosenberg theorem (Stein & Rosenberg, 1948) , which only applies to solving linear systems with non-negative matrices (if one were to apply it to our case, the matrix S in eq. (F.1) in Appendix F needs to have non-zero diagonal entries, which is not possible).

In this sense, our results extend the Stein-Rosenberg theorem to cover nontrivial bilinear games.

In this section we study the optimal convergence rates of EG and OGD.

We define the exponent of linear convergence as r = lim t→∞ ||z (t) ||/||z (t−1) || which is the same as the spectral radius.

For ease of presentation we fix α 1 = α 2 = α > 0 (using scaling symmetry) and we use r * to denote the optimal exponent of linear convergence (achieved by tuning the parameters α, β, γ).

Our results show that by generalizing gradient algorithms one can obtain better convergence rates.

Theorem 4.1 (EG optimal).

Both Jacobi and GS EG achieve the optimal exponent of linear conver-

Note that we defined β i = γ i α in Section 2.

In other words, we are taking very large extra-gradient steps (γ i → ∞) and very small gradient steps (α → 0).

.

For Jacobi OGD with β 1 = β 2 = β, to achieve the optimal exponent of linear convergence, we must have α ≤ 2β.

For the original OGD with α = 2β, the optimal exponent of linear convergence r * satisfies

.

For GS OGD with β 2 = 0, the optimal exponent of convergence is r * = (κ 2 − 1)/(κ 2 + 1), at α = √ 2/σ 1 and

Remark The original OGD (Daskalakis et al., 2018) with α = 2β may not always be optimal.

For example, take one-dimensional bilinear game and σ = 1, and denote the spectral radius given α, β as r(α, β).

If we fix α = 1/2, by numerically solving section 3 we have

i.e, α = 1/2, β = 1/3 is a better choice than α = 2β = 1/2.

Numerical method We provide a numerical method for finding the optimal exponent of linear convergence, by realizing that the unit disk in Theorem 2.2 is not special.

Let us call a polynomial to be r-Schur stable if all of its roots lie within an (open) disk of radius r in the complex plane.

We can scale the polynomial with the following lemma:

With the lemma above, one can rescale the Schur conditions and find the convergence region where the exponent of linear convergence is at most r (r < 1).

A simple binary search would allow one to find a better and better convergence region.

See details in Appendix D.3.

Bilinear game We run experiments on a simple bilinear game and choose the optimal parameters as suggested in Theorem 4.1 and 4.2.

The results are shown in the left panel of Figure 1 , which confirms the predicted linear rates.

Figure 2: Heat maps of the spectral radii of different algorithms.

We take σ = 1 for convenience.

The horizontal axis is α and the vertical axis is β.

Top row: Jacobi updates; Bottom row: Gauss-Seidel updates.

Columns (left to right): EG; OGD; momentum.

If the spectral radius is strictly less than one, it means that our algorithm converges.

In each column, the Jacobi convergence region is contained in the GS convergence region (for EG we need an additional assumption, see Theorem 3.2).

Density plots We show the density plots (heat maps) of the spectral radii in Figure 2 .

We make plots for EG, OGD and momentum with both Jacobi and GS updates.

These plots are made when β 1 = β 2 = β and they agree with our theorems in §3.

Wasserstein GAN As in Daskalakis et al. (2018) , we consider a WGAN (Arjovsky et al., 2017) that learns the mean of a Gaussian:

where s(x) is the sigmoid function.

It can be shown that near the saddle point (θ * , φ * ) = (0, v) the min-max optimization can be treated as a bilinear game (Appendix E.1).

With GS updates, we find that Adam diverges, SGD goes around a limit cycle, and EG converges, as shown in the middle panel of Figure 1 .

We can see that Adam does not behave well even in this simple task of learning a single two-dimensional Gaussian with GAN.

Our next experiment shows that generalized algorithms may have an advantage over traditional ones.

Inspired by Theorem 4.1, we compare the convergence of two EGs with the same parameter β = αγ, and find that with scaling, EG has better convergence, as shown in the right panel of Figure 1 .

Finally, we compare Jacobi updates with GS updates.

In Figure 3 , we can see that GS updates converge even if the corresponding Jacobi updates do not.

Mixtures of Gaussians (GMMs) Our last experiment is on learning GMMs with a vanilla GAN (Goodfellow et al., 2014 ) that does not directly fall into our analysis.

We choose a 3-hidden layer ReLU network for both the generator and the discriminator, and each hidden layer has 256 units.

We find that for GD and OGD, Jacobi style updates converge more slowly than GS updates, and whenever Jacobi updates converge, the corresponding GS updates converges as well.

These comparisons can be found in Figure 4 and 5, which implies the possibility of extending our results to non-bilinear games.

Interestingly, we observe that even Jacobi GD converges on this example.

We provide additional comparison between the Jacobi and GS updates of Adam (Kingma & Ba, 2015) in Appendix E.2.

In this work we focus on the convergence behaviour of gradient-based algorithms for solving bilinear games.

By drawing a connection to discrete linear dynamical systems ( §2) and using Schur's theorem, we provide necessary and sufficient conditions for a variety of gradient algorithms, for both simultaneous (Jacobi) and alternating (Gauss-Seidel) updates.

Our results show that Gauss-Seidel updates converge more easily than Jacobi updates.

Furthermore, we find the optimal exponents of linear convergence for EG and OGD, and provide a numerical method for searching that exponent.

We performed a number of experiments to validate our theoretical findings and suggest further analysis.

There are many future directions to explore.

For example, our preliminary experiments on GANs suggest that similar (local) results might be obtained for more general games.

Indeed, the local convergence behaviour of min-max nonlinear optimization can be studied through analyzing the spectrum of the Jacobian matrix of the update operator (see, e.g., Nagarajan & Kolter (2017); Gidel et al. (2019b) ).

We believe our framework that draws the connection to linear discrete dynamic systems and Schur's theorem is a powerful machinery that can be applied in such problems and beyond.

It would be interesting to generalize our results to the constrained case (even for bilinear games), initiated in the recent work of Daskalakis & Panageas (2019) .

Extending our results to account for stochastic noise (as empirically tested in our experiments) is another interesting direction, with some initial results in Gidel et al. (2019a A PROXIMAL POINT (PP) ALGORITHM PP was originally proposed by Martinet (1970) with α 1 = α 2 and then carefully studied by Rockafellar (1976) .

The linear convergence for bilinear games was also proved in the same reference.

Note that we do not consider Gauss-Seidel PP since we do not get a meaningful solution after a shift of steps 2 .

where x (t+1) and y (t+1) are given implicitly by solving the equations above.

For bilinear games, one can derive that:

We can compute the exact form of the inverse matrix, but perhaps an easier way is just to compute the spectrum of the original matrix (the same as Jacobi GD except that we flip the signs of α i ) and perform λ → 1/λ.

Using the fact that the eigenvalues of a matrix are reciprocals of the eigenvalues of its inverse, the characteristic equation is:

With the scaling symmetry (α 1 , α 2 ) → (tα 1 , α 2 /t), we can take α 1 = α 2 = α > 0.

With the notations in Corollary 2.1, we have a = −2/(1 + α 2 σ 2 ) and b = 1/(1 + α 2 σ 2 ), and it is easy to check |a| < 1 + b and b < 1 are always satisfied, which means linear convergence is always guaranteed.

Hence, we have the following theorem: Theorem A.1.

For bilinear games, the proximal point algorithm always converges linearly.

Although the proximal point algorithm behaves well, it is rarely used in practice since it is an implicit method, i.e., one needs to solve (x (t+1) , y In this section we apply Theorem 2.1 to prove Theorem 2.3, an interesting connection between Jacobi and Gauss-Seidel updates:

and L i is strictly lower block triangular.

Then, the characteristic polynomial of Jacobi updates is p(λ, 1) while that of Gauss-Seidel updates is p(λ, λ).

Let us first consider the block linear iterative process in the sense of Jacobi (i.e., all blocks are updated simultaneously):

. . .

. . .

where A i,j is the j-th column block of A i .

For each matrix A i , we decompose it into the sum

where L i is the strictly lower block triangular part and U i is the upper (including diagonal) block triangular part.

Theorem 2.1 indicates that the convergence behaviour of equation B.1 is governed by the largest modulus of the roots of the characteristic polynomial:

Alternatively, we can also consider the updates in the sense of Gauss-Seidel (i.e., blocks are updated sequentially):

We can rewrite the Gauss-Seidel update elegantly 3 as:

i.e.,

where L k+1 := 0.

Applying Theorem 2.1 again we know the convergence behaviour of the GaussSeidel update is governed by the largest modulus of roots of the characteristic polynomial:

Note that A 0 = −I and the factor det(I − L 1 ) −1 can be discarded since multiplying a characteristic polynomial by a non-zero constant factor does not change its roots.

B.2 PROOF OF COROLLARY 2.1 Corollary 2.1 (e.g. Mansour (2011)) .

A real quadratic polynomial λ 2 + aλ + b is Schur stable iff b < 1, |a| < 1 + b; A real cubic polynomial λ 3 + aλ 2 + bλ + c is Schur stable iff |c| < 1,

Proof.

It suffices to prove the result for quartic polynomials.

We write down the matrices:

We require det(

2 and thus |c − ad| < 1 − d 2 due to the first condition.

δ 4 > 0 simplifies to:

14)

which yields |a + c| < |b + d + 1|.

Finally, δ 3 > 0 reduces to:

Denote p(λ) := λ 4 + aλ 3 + bλ 2 + cλ + d, we must have p(1) > 0 and p(−1) > 0, as otherwise there is a real root λ 0 with |λ 0 | ≥ 1.

Hence we obtain b + d + 1 > |a + c| > 0.

Also, from |c − ad| < 1 − d 2 , we know that:

So, the second factor in B.15 is negative and the positivity of the first factor reduces to:

To obtain the Schur condition for cubic polynomials, we take d = 0, and the quartic Schur condition becomes:

To obtain the Schur condition for quadratic polynomials, we take c = 0 in the above and write:

The proof is now complete.

Some of the following proofs in Appendix C.4 and C.5 rely on Mathematica code (mostly with the built-in function Reduce) but in principle the code can be verified manually using cylindrical algebraic decomposition.

In this appendix, we derive the exact forms of LDSs (eq. (2.3)) and the characteristic polynomials for all gradient-based methods introduced in §2, with eq. (2.4).

The following lemma is well-known and easy to verify using Schur's complement:

Gradient descent From equation 2.6 the update equation of Jacobi GD can be derived as:

and with Lemma C.1, we compute the characteristic polynomial as in eq. (2.4):

With spectral decomposition we obtain equation 3.1.

Taking α 2 → λα 2 and with Theorem 2.3 we obtain the corresponding GS updates.

Therefore, the characteristic polynomials for GD are:

Extra-gradient From eq. (2.7) and eq. (2.8), the update of Jacobi EG is:

the characteristic polynomial is:

Since we assumed α 2 > 0, we can left multiply the second row by β 2 E/α 2 and add it to the first row.

Hence, we obtain:

With Lemma C.1 the equation above becomes:

which simplifies to equation 3.2 with spectral decomposition.

Note that to obtain the GS polynomial, we simply take α 2 → λα 2 in the Jacobi polynomial as shown in Theorem 2.3.

For the ease of reading we copy the characteristic equations for generalized EG:

Optimistic gradient descent We can compute the LDS for OGD with eq. (2.9) and eq. (2.10):

With eq. (2.4), the characteristic polynomial for Jacobi OGD is

Taking the determinant and with Lemma C.1 we obtain equation 3.6.

The characteristic polynomial for GS updates in equation 3.7 can be subsequently derived with Theorem 2.3, by taking (α 2 , β 2 ) → (λα 2 , λβ 2 ).

For the ease of reading we copy the characteristic polynomials from the main text as:

Momentum method With eq. (2.11) and eq. (2.12), the LDS for the momentum method is:

From eq. (2.4), the characteristic polynomial for Jacobi momentum is

Taking the determinant and with Lemma C.1 we obtain equation 3.10, while equation 3.11 can be derived with Theorem 2.3, by taking α 2 → λα 2 .

For the ease of reading we copy the characteristic polynomials from the main text as:

C.2 PROOF OF THEOREM 3.1: SCHUR CONDITIONS OF GD Theorem 3.1 (GD).

Jacobi GD and Gauss-Seidel GD do not converge.

However, Gauss-Seidel GD can have a limit cycle while Jacobi GD always diverges.

Proof.

With the notations in Corollary 2.1, for Jacobi GD, b = 1 + α 2 σ 2 > 1.

For Gauss-Seidel GD, b = 1.

The Schur conditions are violated.

For generalized EG with α 1 = α 2 = α and γ i = β i /α, Jacobi and Gauss-Seidel updates achieve linear convergence iff for any singular value σ of E, we have:

If β 1 + β 2 + α 2 < 2/σ 2 1 , the convergence region of GS updates strictly include that of Jacobi updates.

Both characteristic polynomials can be written as a quadratic polynomial λ 2 + aλ + b, where:

Compared to Jacobi EG, the only difference between Gauss-Seidel and Jacobi updates is that the α 2 σ 2 in b is now in a, which agrees with Theorem 2.3.

Using Corollary 2.1, we can derive the Schur conditions equation 3.4 and equation 3.5.

More can be said if β 1 + β 2 is small.

For instance, if β 1 + β 2 + α 2 < 2/σ More precisely, to show that the GS convergence region strictly contains that of the Jacobi convergence region, simply take β 1 = β 2 = β.

The Schur condition for Jacobi EG and Gauss-Seidel EG are separately:

It can be shown that if β = α 2 /3 and α → 0, equation C.21 is always violated whereas equation C.22 is always satisfied.

Conversely, we give an example when Jacobi EG converges while GS EG does not.

Let β 1 σ 2 = β 2 σ 2 ≡ 3 2 , then Jacobi EG converges iff α 2 σ 2 < 3 4 while GS EG converges iff α 2 σ 2 < 1 4 .

In this subsection, we fill in the details of the proof of Theorem 3.3, by first deriving the Schur conditions of OGD, and then studying the relation between Jacobi OGD and GS OGD.

Theorem 3.3 (OGD).

For generalized OGD with α 1 = α 2 = α, Jacobi and Gauss-Seidel updates achieve linear convergence iff for any singular value σ of E, we have:

The convergence region of GS updates strictly include that of Jacobi updates.

The Jacobi characteristic polynomial is now quartic in the form λ 4 + aλ 3 + bλ 2 + cλ + d, with

Comparably, the GS polynomial equation 3.7 can be reduced to a cubic one λ 3 + aλ 2 + bλ + c with

First we derive the Schur conditions equation 3.8 and equation 3.9.

Note that other than Corollary 2.1, an equivalent Schur condition can be read from Cheng & Chiou (2007, Theorem 1) as:

Theorem C.1 (Cheng & Chiou (2007)

With equation C.23 and Theorem C.1, it is straightforward to derive equation 3.8.

With equation C.24 and Corollary 2.1, we can derive equation 3.9 without much effort.

Now, let us study the relation between the convergence region of Jacobi OGD and GS OGD, as given in equation 3.8 and equation 3.9.

Namely, we want to prove the last sentence of Theorem 3.3.

The outline of our proof is as follows.

We first show that each region of (α, β 1 , β 2 ) described in equation 3.8 (the Jacobi region) is contained in the region described in equation 3.9 (the GS region).

Since we are only studying one singular value, we slightly abuse the notations and rewrite β i σ as β i (i = 1, 2) and ασ as α.

From equation 3.6 and equation 3.7, β 1 and β 2 can switch.

WLOG, we assume β 1 ≥ β 2 .

There are four cases to consider:

The third Jacobi condition in equation 3.8 now is redundant, and we have α > β 1 or α < β 2 for both methods.

Solving the quadratic feasibility condition for α gives:

where u = (β 1 β 2 + 1)(

.

On the other hand, assume α > β 1 , the first and third GS conditions are automatic.

Solving the second gives:

2 /2 and g(β 2 ) := (β 2 + 4 + 5β 2 2 )/(2(1 + β 2 2 )), and one can show that

(C.28)

Furthermore, it can also be shown that given 0 < β 2 < 1 and β 2 ≤ β 1 < g(β 2 ), we have

• β 1 ≥ β 2 = 0.

The Schur condition for Jacobi and Gauss-Seidel updates reduces to:

One can show that given β 1 ∈ (0, 1), we have

Reducing the first, second and fourth conditions of equation 3.8 yields:

This region contains the Jacobi region.

It can be similarly proved that even within this larger region, GS Schur condition equation 3.9 is always satisfied.

• β 2 ≤ β 1 < 0.

We have u < 0, tv < 0 and thus α < (u + √ u 2 + tv)/t < 0.

This contradicts our assumption that α > 0.

Combining the four cases above, we know that the Jacobi region is contained in the GS region.

To show the strict inclusion, take β 1 = β 2 = α/5 and α → 0.

One can show that as long as α is small enough, all the Jacobi regions do not contain this point, each of which is described with a singular value in equation 3.8.

However, all the GS regions described in equation 3.9 contain this point.

The proof above is still missing some details.

We provide the proofs of equation C.26, equation C.28, equation C.29 and equation C.32 in the sub-sub-sections below, with the help of Mathematica, although one can also verify these claims manually.

Moreover, a one line proof of the inclusion can be given with Mathematica code, as shown in Section C.4.5.

The fourth condition of equation 3.8 can be rewritten as:

where we used |β 1 β 2 | < 1 in both cases.

So, equation C.33 becomes:

Combining with α > β 1 or α < β 2 obtained from the second condition, we have:

The first case is not possible, with the following code: u = (b1 b2 + 1) (b1 + b2); v = b1 b2 (b1 b2 + 1) (b1 b2 -3); t = (b1^2 + 1) (b2^2 + 1); Reduce[b2 t > u -Sqrt[u^2 + t v] && b1 >= b2 > 0 && Abs[b1 b2] < 1], and we have:

Therefore, the only possible case is β 1 < α < (u + √ u 2 + tv)/t.

Where the feasibility region can be solved with:

What we get is: 1 + b2^2) ), {b2, b1}], we can remove the first constraint and get: 0 < b2 < 1 && b2 <= b1 < b2/(2 (1 + b2^2)) + 1/2 Sqrt[(4 + 5 b2^2)/(1 + b2^2)^2].

The second Jacobi condition simplifies to α > β 1 and the fourth simplifies to equation C.34.

Combining with the first Jacobi condition:

we have:

This can be further simplified to achieve equation C.32.

In fact, there is another very simple proof:

Reduce[ForAll[{b1, b2, a}, (a -b1) (a -b2) > 0 && (a + b1) (a + b2) > -4 && Abs[b1 b2] < 1 && a^2 (b1^2 + 1) (b2^2 + 1) < (b1 b2 + 1) (2 a (b1 + b2) + b1 b2 (b1 b2 -3)), (a -b1) (a -b2) > 0 && (a + b1) (a + b2) < 4 && (a b1 + 1) (a b2 + 1) > (1 + b1 b2)^2], {b2, b1, a}] True.

However, this proof does not tell us much information about the range of our variables.

Theorem 3.4 (momentum).

For the generalized momentum method with α 1 = α 2 = α, the Jacobi updates never converge, while the GS updates converge iff for any singular value σ of E, we have:

This condition implies that at least one of β 1 , β 2 is negative.

Jacobi condition We first rename ασ as al and β 1 , β 2 as b1, b2.

With Theorem C.1:

We obtain:

{Abs[b1 b2]

< 1, Abs[2 + b1 + b2]

< 3 + b1 b2, al^2 > 0, al^2 + 4 (1 + b1) (1 + b2) > 0, al^2 (-1 + b1 b2)^2 < 0}.

The last condition is never satisfied and thus Jacobi momentum never converges.

Gauss-Seidel condition With Theorem C.1, we compute:

The result is:

{Abs[b1 b2]

< 1, Abs[2 -al^2 + b1 + b2]

< 3 + b1 b2, al^2 > 0, 4 (1 + b1) (1 + b2) > al^2, al^2 (b1 + b2 + (-2 + al^2 -b1) b1 b2 + b1 (-1 + 2 b1) b2^2) < 0}, which can be further simplified to equation ??.

With Theorem 3.4, we can actually show that in general at least one of β 1 and β 2 must be negative.

There are three cases to consider, and in each case we simplify equation ??

:

1. β 1 β 2 = 0.

WLOG, let β 2 = 0, and we obtain −1 < β 1 < 0 and α 2 σ 2 < 4(1 + β 1 ).

(C.36) 2.

β 1 β 2 > 0.

We have

3.

β 1 β 2 < 0.

WLOG, we assume β 1 ≥ β 2 .

We obtain:

The constraints for α are α > 0 and:

These conditions can be further simplified by analyzing all singular values.

They only depend on σ 1 and σ n , the largest and the smallest singular values.

Now, let us derive equation C.37, equation C.38 and equation C.39 more carefully.

Note that we use a for ασ.

Reduce[Abs[b1 b2] < 1 && Abs[-a^2 + b1 + b2 + 2] < b1 b2 + 3 && 4 (b1 + 1) (b2 + 1) > a^2 && a^2 b1 b2 < (1 -b1 b2) (2 b1 b2 -b1 -b2) && b1 b2 > 0 && a > 0, {b2, b1, a}] -1 < b2 < 0 && -1 < b1 < 0 && 0 < a < Sqrt[4 + 4 b1 + 4 b2 + 4 b1 b2] C.5.4 PROOF OF EQUATIONS C.38 AND C.39

Reduce[Abs[b1 b2] < 1 && Abs[-a^2 + b1 + b2 + 2] < b1 b2 + 3 && 4 (b1 + 1) (b2 + 1) > a^2 && a^2 b1 b2 < (1 -b1 b2) (2 b1 b2 -b1 -b2) && b1 b2 < 0 && b1 >= b2 && a > 0, {b2, b1, a}] (-1 < b2 <= -(1/3) && ((0 < b1 <= b2/(-1 + 2 b2) && 0 < a < Sqrt[4 + 4 b1 + 4 b2 + 4 b1 b2]) || (b2/(-1 + 2 b2) < b1 < -(1/(3 b2)) && Sqrt[(-b1 -b2 + 2 b1 b2 + b1^2 b2 + b1 b2^2 -2 b1^2 b2^2)/( b1 b2)]

< a < Sqrt[4 + 4 b1 + 4 b2 + 4 b1 b2]))) || (-(1/3) < b2 < 0 && ((0 < b1 <= b2/(-1 + 2 b2) && 0 < a < Sqrt[4 + 4 b1 + 4 b2 + 4 b1 b2]) || (b2/(-1 + 2 b2) < b1 < -(b2/(1 + 2 b2)) && Sqrt[(-b1 -b2 + 2 b1 b2 + b1^2 b2 + b1 b2^2 -2 b1^2 b2^2)/( b1 b2)]

< a < Sqrt[4 + 4 b1 + 4 b2 + 4 b1 b2]))) Some further simplication yields equation C.38 and equation C.39.

For bilinear games and gradient-based methods, a Schur condition defines the region of convergence in the parameter space, as we have seen in Section 3.

However, it is unknown which setting of parameters has the best convergence rate in a Schur stable region.

We explore this problem now.

Due to Theorem 3.1, we do not need to study GD.

The remaining cases are EG, OGD and GS momentum (Jacobi momentum does not converge due to Theorem 3.4).

Analytically (Section D.1 and D.2), we study the optimal linear rates for EG and special cases of generalized OGD (Jacobi OGD with β 1 = β 2 and Gauss-Seidel OGD with β 2 = 0).

The special cases include the original form of OGD.

We also provide details for the numerical method described at the end of Section 4.

The optimal spectral radius is obtained by solving another min-max optimization problem:

where θ denotes the collection of all hyper-parameters, and r(θ, σ) is defined as the spectral radius function that relies on the choice of parameters and the singular value σ.

We also use Sv(E) to denote the set of singular values of E.

In general, the function r(θ, σ) is non-convex and thus difficult to analyze.

However, in the special case of quadratic characteristic polynomials, it is possible to solve equation D.1.

This is how we will analyze EG and special cases of OGD, as r(θ, σ) can be expressed using root functions of quadratic polynomials.

For cubic and quartic polynomials, it is in principle also doable as we have analytic formulas for the roots.

However, these formulas are extremely complicated and difficult to optimize and we leave it for future work.

For EG and OGD, we will show that the optimal linear rates depend only on the conditional number κ := σ 1 /σ n .

For simplicity, we always fix α 1 = α 2 = α > 0 using the scaling symmetry studied in Section 3.

D.1 PROOF OF THEOREM 4.1: OPTIMAL CONVERGENCE RATE OF EG Theorem 4.1 (EG optimal).

Both Jacobi and GS EG achieve the optimal exponent of linear convergence r * = (κ 2 − 1)/(κ 2 + 1) at α → 0 and β 1 = β 2 = 2/(σ 2 1 + σ 2 n ).

As κ → ∞, r * → 1 − 2/κ 2 .

For Jacobi updates, if β 1 = β 2 = β, by solving the roots of equation 3.2, the min-max problem is:

If σ 1 = σ n = σ, we can simply take α → 0 and β = 1/σ 2 to obtain a super-linear convergence rate.

Otherwise, let us assume σ 1 > σ n .

We obtain a lower bound by taking α → 0 and equation D.2 reduces to:

The optimal solution is given at 1 − βσ 2 n = βσ From general β 1 , β 2 , it can be verified that the optimal radius is achieved at β 1 = β 2 and the problem reduces to the previous case.

The optimization problem is:

In the first case, a lower bound is obtained at α 2 = (β 1 − β 2 ) 2 σ 2 /4 and thus the objective only depends on β 1 + β 2 .

In the second case, the lower bound is obtained at α → 0 and β 1 → β 2 .

Therefore, the function is optimized at β 1 = β 2 and α → 0.

Our analysis above does not mean that α → 0 and β 1 = β 2 = 2/(σ 2 1 + σ 2 n ) is the only optimal choice.

For example, when σ 1 = σ n = 1, we can take β 1 = 1 + α and β 2 = 1 − α to obtain a super-linear convergence rate.

For Gauss-Seidel updates and β 1 = β 2 = β, we do the following optimization:

where by solving equation 3.3:

r(σ, β, σ 2 ) is quasi-convex in σ 2 , so we just need to minimize over α, β at both end points.

Hence, equation D.5 reduces to: min α,β max{r(α, β, σ 1 ), r(α, β, σ n )}.

By arguing over three cases:

n , we find that the minimum (κ 2 − 1)/(κ 2 + 1) can be achieved at α → 0 and β = 2/(σ 2 1 + σ 2 n ), the same as Jacobi EG.

This is because α → 0 decouples x and y and it does not matter whether the update is Jacobi or GS.

For general β 1 , β 2 , it can be verified that the optimal radius is achieved at β 1 = β 2 .

We do the following transformation:

β i → ξ i − α 2 /2, so that the characteristic polynomial becomes:

Denote ξ 1 + ξ 2 = φ, and (ξ 1 − α 2 /2)(ξ 2 − α 2 /2) = ν, we have:

The discriminant is ∆ := σ 2 (σ 2 (φ 2 − 4ν) − 4α 2 ).

We discuss two cases:

1.

φ 2 − 4ν < 0.

We are minimizing:

with a ∨ b := max{a, b} a shorthand.

A minimizer is at α → 0 and ν → φ 2 /4 (since φ 2 < 4ν), where β 1 = β 2 = 2/(σ 2 1 + σ 2 n ) and α → 0.

2.

φ 2 − 4ν ≥ 0.

A lower bound is:

.

This is only possible if α → 0 and φ 2 → 4ν, which yields β 1 = β 2 = 2/(σ

From what has been discussed, the optimal radius is (κ 2 − 1)/(κ 2 + 1) which can be achieved at β 1 = β 2 = 2/(σ 2 1 + σ 2 n ) and α → 0.

Again, this might not be the only choice.

For instance, take σ 1 = σ 2 n = 1, from equation 3.3, a super-linear convergence rate can be achieved at β 1 = 1 and

D.2 PROOF OF THEOREM 4.2: OPTIMAL CONVERGENCE RATE OF OGD Theorem 4.2 (OGD optimal).

For Jacobi OGD with β 1 = β 2 = β, to achieve the optimal linear rate, we must have α ≤ 2β.

For the original OGD with α = 2β, the optimal linear rate r * satisfies

.

For Gauss-Seidel OGD with β 2 = 0, the optimal linear rate is r * = (κ 2 − 1)/(κ 2 + 1), at α = √ 2/σ 1 and

For OGD, the characteristic polynomials equation 3.6 and equation 3.7 are quartic and cubic separately, and thus optimizing the spectral radii for generalized OGD is difficult.

However, we can study two special cases: for Jacobi OGD, we take β 1 = β 2 ; for Gauss-Seidel OGD, we take β 2 = 0.

In both cases, the spectral radius functions can be obtained by solving quadratic polynomials.

We assume β 1 = β 2 = β in this subsection.

The characteristic polynomial for Jacobi OGD equation 3.6 can be written as:

(D.10) Factorizing it gives two equations which are conjugate to each other:

11) The roots of one equation are the conjugates of the other equation.

WLOG, we solve λ(λ − 1) + i(λα − β)σ = 0 which gives (1/2)(u ± v), where

(D.12)

, v can be expressed as:

therefore, the spectral radius r(α, β, σ) satisfies:

(D.14) and the minimum is achieved at α = 2β.

From now on, we assume α ≤ 2β, and thus v = a + ib.

We write: E SUPPLEMENTARY MATERIAL FOR SECTIONS 5 AND 6

We provide supplementary material for Sections 5 and 6.

We first prove that when learning the mean of a Gaussian, WGAN is locally a bilinear game in Appendix E.1.

For mixtures of Gaussians, we provide supplementary experiments about Adam in Appendix E.2.

This result implies that in some cases, Jacobi updates are better than GS updates.

We further verify this claim in Appendix E.3 by showing an example of OGD on bilinear games.

Optimizing the spectral radius given a certain singular value is possible numerically, as in Appendix E.4.

Inspired by Daskalakis et al. (2018) , we consider the following WGAN (Arjovsky et al., 2017) :

with s(x) := 1/(1 + e −x ) the sigmoid function.

We study the local behavior near the saddle point (v, 0), which depends on the Hessian:

with E v a shorthand for E x∼N (v,σ 2 I) and E φ for E z∼N (φ,σ 2 I) .

At the saddle point, the Hessian is simplified as: Therefore, this WGAN is locally a bilinear game.

Given the same parameter settings as in Section 5, we train the vanilla GAN using Adam, with the step size α = 0.0002, and β 1 = 0.9, β 2 = 0.999.

As shown in Figure 6 , Jacobi updates converge faster than the corresponding GS updates.

: Contour plot of spectral radius equal to 0.8.

The red curve is for the Jacobi polynomial and the blue curve is for the GS polynomial.

The GS region is larger but for some parameter settings, Jacobi OGD achieves a faster convergence rate.

Take α = 0.9625, β 1 = β 2 = β = 0.5722, and σ = 1, the Jacobi and GS OGD radii are separately 0.790283 and 0.816572 (by solving equation 3.6 and equation 3.7), which means that Jacobi OGD has better performance for this setting of parameters.

A more intuitive picture is given as Figure 7 , where we take β 1 = β 2 = β.

We minimize r(θ, σ) for a given singular value numerically.

WLOG, we take σ = 1, since we can rescale parameters to obtain other values of σ.

We implement grid search for all the parameters within the range [−2, 2] and step size 0.05.

For the step size α, we take it to be positive.

We use {a, b, s} as a shorthand for {a, a + s, a + 2s, . . .

, b}.

• We first numerically solve the characteristic polynomial for Jacobi OGD equation 3.6, fixing α 1 = α 2 = α with scaling symmetry.

With α ∈ {0, 2, 0.05}, β i ∈ {−2, 2, 0.05}, the best parameter setting is α = 0.7, β 1 = 0.1 and β 2 = 0.6.

β 1 and β 2 can be switched.

The optimal radius is 0.6.

• We also numerically solve the characteristic polynomial for Gauss-Seidel OGD equation 3.7, fixing α 1 = α 2 = α with scaling symmetry.

With α ∈ {0, 2, 0.05}, β i ∈ {−2, 2, 0.05}, the best parameter setting is α = 1.4, β 1 = 0.7 and β 2 = 0.

β 1 and β 2 can be switched.

The optimal rate is 1/(5 √ 2).

This rate can be further improved to be zero where α = √ 2, β 1 = 1/ √ 2 and β 2 = 0.

• Finally, we numerically solve the polynomial for Gauss-Seidel momentum equation 3.11, with the same grid.

The optimal parameter choice is α = 1.8, β 1 = −0.1 and β 2 = −0.05.

β 1 and β 2 can be switched.

The optimal rate is 0.5.

In this appendix, we interpret the gradient-based algorithms (except PP) we have studied in this paper as splitting methods (Saad, 2003) , for both Jacobi and Gauss-Seidel updates.

By doing this, one can understand our algorithms better in the context of numerical linear algebra and compare our results in Section 3 with the Stein-Rosenberg theorem.

For EG, we need to compute an inverse:

Given det(α 1 α 2 I + β 1 β 2 EE ) = 0, the inverse always exists.

The splitting method can also work for second-step methods, such as OGD and momentum.

We split S = M − N − P and solve:

For OGD, we have: For EG, we need to compute an inverse:

The splitting method can also work for second-step methods, such as OGD and momentum.

We split S = M − N − P and solve: z t+1 = M −1 N z t + M

In this paper we considered the bilinear game when E is a non-singular square matrix for simplicity.

Now let us study the general case where E ∈ R m×n .

As stated in Section 2, saddle points exist iff b ∈ R(E), c ∈ R(E ).

The set of saddle points is:

{(x, y)|y ∈ N (E), x ∈ N (E )}. (G.3)

<|TLDR|>

@highlight

We systematically analyze the convergence behaviour of popular gradient algorithms for solving bilinear games, with both simultaneous and alternating updates.