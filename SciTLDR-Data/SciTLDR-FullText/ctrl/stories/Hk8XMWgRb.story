We propose a principled method for kernel learning, which relies on a Fourier-analytic characterization of translation-invariant or rotation-invariant kernels.

Our method produces a sequence of feature maps, iteratively refining the SVM margin.

We provide rigorous guarantees for optimality and generalization, interpreting our algorithm as online equilibrium-finding dynamics in a certain two-player min-max game.

Evaluations on synthetic and real-world datasets demonstrate scalability and consistent improvements over related random features-based methods.

Choosing the right kernel is a classic question that has riddled machine learning practitioners and theorists alike.

Conventional wisdom instructs the user to select a kernel which captures the structure and geometric invariances in the data.

Efforts to formulate this principle have inspired vibrant areas of study, going by names from feature selection to multiple kernel learning (MKL).We present a new, principled approach for selecting a translation-invariant or rotation-invariant kernel to maximize the SVM classification margin.

We first describe a kernel-alignment subroutine, which finds a peak in the Fourier transform of an adversarially chosen data-dependent measure.

Then, we define an iterative procedure that produces a sequence of feature maps, progressively improving the margin.

The resulting algorithm is strikingly simple and scalable.

Intriguingly, our analysis interprets the main algorithm as no-regret learning dynamics in a zero-sum min-max game, whose value is the classification margin.

Thus, we are able to quantify convergence guarantees towards the largest margin realizable by a kernel with the assumed invariance.

Finally, we exhibit experiments on synthetic and benchmark datasets, demonstrating consistent improvements over related random features-based kernel methods.

There is a vast literature on MKL, from which we use the key concept of kernel alignment BID7 .

Otherwise, our work bears few similarities to traditional MKL; this and much related work (e.g. BID5 ; Gönen & Alpaydın (2011); Lanckriet et al. (2004) ) are concerned with selecting a kernel by combining a collection of base kernels, chosen beforehand.

Our method allows for greater expressivity and even better generalization guarantees.

Instead, we take inspiration from the method of random features BID12 .

In this pioneering work, originally motivated by scalability, feature maps are sampled according to the Fourier transform of a chosen kernel.

The idea of optimizing a kernel in random feature space was studied by BID17 .

In this work, which is most similar to ours, kernel alignment is optimized via importance sampling on a fixed, finitely supported proposal measure.

However, the proposal can fail to contain informative features, especially in high dimension; indeed, they highlight efficiency, rather than showing performance improvements over RBF features.

Learning a kernel in the Fourier domain (without the primal feature maps) has also been considered previously: BID11 and BID22 model the Fourier spectrum parametrically, which limits expressivity; the former also require complicated posterior inference procedures.

BID2 study learning a kernel in the Fourier domain jointly with regression parameters.

They show experimentally that this locates informative frequencies in the data, without theoretical guarantees.

Our visualizations suggest this approach can get stuck in poor local minima, even in 2 dimensions.

BID6 also use boosting to build a kernel sequentially; however, they only consider a basis of linear feature maps, and require costly generalized eigenvector computations.

From a statistical view, Fukumizu et al. (2009) bound the SVM margin in terms of maximum mean discrepancy, which is equivalent to (unweighted) kernel alignment.

Notably, their bound can be loose if the number of support vectors is small; in such situations, our theory provides a tighter characterization.

Moreover, our attention to the margin goes beyond the usual objective of kernel alignment.

We present an algorithm that outputs a sequence of Fourier features, converging to the maximum realizable SVM classification margin on a labeled dataset.

At each iteration, a pair of features is produced, which maximizes kernel alignment with a changing, adversarially chosen measure.

As this measure changes slowly, the algorithm builds a diverse and informative feature representation.

Our main theorem can be seen as a case of von Neumann's min-max theorem for a zero-sum concave-linear game; indeed, our method bears a deep connection to boosting (Freund & Schapire, 1996; BID15 .

In particular, both the theory and empirical evidence suggest that the generalization error of our method decreases as the number of random features increases.

In traditional MKL methods, generalization bounds worsen as base kernels are added.

Other methods in the framework of Fourier random features take the approach of approximating a kernel by sampling feature maps from a continuous distribution.

In contrast, our method constructs a measure with small finite support, and realizes the kernel exactly by enumerating the associated finite-dimensional feature map; there is no randomness in the features.

We focus on two natural families of kernels k(x, x ): translation-invariant kernels on X = R d , which depend only on x − x , and rotation-invariant kernels on the hypersphere X = S d−1 , which depend only on x, x .

These invariance assumptions subsume most widely-used classes of kernels; notably, the Gaussian (RBF) kernel satisfies both.

For the former invariance, Bochner's theorem provides a Fourier-analytic characterization: Theorem 2.1 (e.g. Eq. (1), Sec. 1.4.3 in Rudin (2011) DISPLAYFORM0 is the Fourier transform of a symmetric non-negative measure (where the Fourier domain is Ω = R d ).

That is, DISPLAYFORM1 DISPLAYFORM2 A similar characterization is available for rotation-invariant kernels, where the Fourier basis functions are the spherical harmonics, a countably infinite family of complex polynomials which form an orthonormal basis for square-integrable functions X → C. To unify notation, let Ω ⊂ N × Z be the set of valid index pairs ω = ( , m), and let ω → −ω denote a certain involution on Ω; we supply details and references in Appendix B.1.

Theorem 2.2.

A rotation-invariant continuous function k : DISPLAYFORM3 DISPLAYFORM4 for some λ ∈ L 1 (Ω), with λ(ω) ≥ 0 and λ(ω) = λ(−ω) for all valid index pairs ω = ( , m) ∈ Ω.In each of these cases, we call this Fourier transform λ k (ω) the dual measure of k. This measure decomposes k into a non-negative combination of Fourier basis kernels.

Furthermore, this decomposition gives us a feature map φ : X → L 2 (Ω, λ k ) whose image realizes the kernel under the codomain's inner product; 2 that is, for all x, x ∈ X , DISPLAYFORM5 Respectively, these feature maps are φ x (ω) = e i ω,x and φ x ( , m) = Y d ,m (x).

Although they are complex-valued, symmetry of λ k allows us to apply the transformation {φ x (ω), φ x (−ω)} → {Re φ x (ω), Im φ x (ω)} to yield real features, preserving the inner product.

The analogous result holds for spherical harmonics.

In a binary classification task with n training samples (x i ∈ X , y i ∈ {±1}), a widely-used quantity for measuring the quality of a kernel k : X × X → R is its alignment BID7 BID5 , 3 defined by DISPLAYFORM0 where y is the vector of labels and G k is the Gram matrix.

Here, we let P = i:yi=1 δ xi and Q = i:yi=−1 δ xi denote the (unnormalized) empirical measures of each class, where δ x is the Dirac measure at x. When P, Q are arbitrary measures on X , this definition generalizes to DISPLAYFORM1 In terms of the dual measure λ k (ω), kernel alignment takes a useful alternate form, noted by BID18 .

Let µ denote the signed measure P − Q. Then, when k is translationinvariant, we have DISPLAYFORM2 Analogously, when k is rotation-invariant, we have DISPLAYFORM3 It can also be verified that λ k L1(Ω) = k(x, x), which is of course the same for all x ∈ X .

In each case, the alignment is linear in λ k .

We call v(ω) the Fourier potential, which is the squared magnitude of the Fourier transform of the signed measure µ = P − Q. This function is clearly bounded pointwise by (P(X ) + Q(X )) 2 .

First, we consider the problem of finding a kernel k (subject to either invariance) that maximizes alignment γ k (P, Q); we optimize the dual measure λ k (ω).

Aside from the non-negativity and symmetry constraints from Theorems 2.1 and 2.2, we constrain λ k 1 = 1, as this quantity appears as a normalization constant in our generalization bounds (see Theorem 4.3).

Maximizing γ k (P, Q) in this constraint set, which we call L ⊂ L 1 (Ω), takes the form of a linear program on an infinitedimensional simplex.

Noting that v(ω) = v(−ω) ≥ 0, γ k is maximized by placing a Dirac mass at any pair of opposite modes ±ω * ∈ argmax ω v(ω).At first, P and Q will be the empirical distributions of the classes, specified in Section 2.2.

However, as Algorithm 2 proceeds, it will reweight each data point x i in the measures by α(i).

Explicitly, the reweighted Fourier potential takes the form DISPLAYFORM0 Due to its non-convexity, maximizing v α (ω), which can be interpreted as finding a global Fourier peak in the data, is theoretically challenging.

However, we find that it is easy to find such peaks in our experiments, even in hundreds of dimensions.

This arises from the empirical phenomenon that realistic data tend to be band-limited, a cornerstone hypothesis in data compression.

An 2 constraint (or equivalently, 2 regularization; see Kakade et al. (2009) ) can be explicitly enforced to promote band-limitedness; we find that this is not necessary in practice.

When a gradient is available (in the translation-invariant case), we use Langevin dynamics (Algorithm 1) to find the peaks of v(ω), which enjoys mild theoretical hitting-time guarantees (see Theorem 4.5).

See Appendix A.3 for a discussion of the (discrete) rotation-invariant case.

Algorithm 1 Langevin dynamics for kernel alignment DISPLAYFORM1 , weights α ∈ R n .

2: Parameters: time horizon τ , diffusion rate ζ, temperature ξ.

Update DISPLAYFORM2 It is useful in practice to use parallel initialization, running m concurrent copies of the diffusion process and returning the best single ω encountered.

This admits a very efficient GPU implementation: the multi-point evaluations v α (ω 1..m ) and ∇v α (ω 1..m ) can be computed from an (m, d) by (d, n) matrix product and pointwise trigonometric functions.

We find that Algorithm 1 typically finds a reasonable peak within ∼100 steps.

Support vector machines (SVMs) are perhaps the most ubiquitous use case of kernels in practice.

To this end, we propose a method that boosts Algorithm 1, building a kernel that maximizes the classification margin.

Let k be a kernel with dual λ k , and Y def = diag(y).

Write the dual l 1 -SVM objective, 5 parameterizing the kernel by λ k : DISPLAYFORM0 Thus, for a fixed α, F is equivalent to kernel alignment, and can be minimized by Algorithm 1.

However, the support vector weights α are of course not fixed; given a kernel k, α is chosen to maximize F , giving the (reciprocal) SVM margin.

In all, to find a kernel k which maximizes the margin under an adversarial choice of α, one must consider a two-player zero-sum game: DISPLAYFORM1 4 Whenever i is an index, we will denote the imaginary unit by ι.

5 Of course, our method applies to l2 SVMs, and has even stronger theoretical guarantees; see Section 4.1.where L is the same constraint set as in Section 3.1, and K is the usual dual feasible set {0 α C, y T α = 0} with box parameter C.In this view, we make some key observations.

First, Algorithm 1 allows the min-player to play a pure-strategy best response to the max-player.

Furthermore, a mixed strategyλ for the min-player is simply a translation-or rotation-invariant kernel, realized by the feature map corresponding to its support.

Finally, since the objective is linear in λ and concave in α, there exists a Nash equilibrium (λ * , α * ) for this game, from which λ * gives the margin-maximizing kernel.

We can use no-regret learning dynamics to approximate this equilibrium.

Algorithm 2 runs Algorithm 1 for the min-player, and online gradient ascent BID25 for the max-player.

Intuitively (and as is visualized in our synthetic experiments), this process slowly morphs the landscape of v α to emphasize the margin, causing Algorithm 1 to find progressively more informative features.

At the end, we simply concatenate these features; contingent on the success of the kernel alignment steps, we have approximated the Nash equilibrium.

Algorithm 2 No-regret learning dynamics for SVM margin maximization DISPLAYFORM2 .

2: Parameters: box constraint C, # steps T , step sizes {η t }; parameters for Algorithm 1.

DISPLAYFORM3 Use Algorithm 1 (or other v α maximizer) on S with weights α t , returning ω t .

Append two features {Re φ x (ω t ), Im φ x (−ω t )} to each x i 's representation Φ(x i ).

DISPLAYFORM0 We provide a theoretical analysis in Section 4.1, and detailed discussion on heuristics, hyperparameters, and implementation details in depth in Appendix A. One important note is that the online gradient g t = 1−2Y Re( Φ t , Yα t Φ t ) is computed very efficiently, where DISPLAYFORM1 is the vector of the most recently appended features.

Langevin dynamics, easily implemented on a GPU, comprise the primary time bottleneck.

We first state the main theoretical result, which quantifies the convergence properties of Algorithm 2.

Theorem 4.1 (Main).

Assume that at each step t, Algorithm 1 returns an ε t -approximate global maximizer ω t (i.e., v αt (ω t ) ≥ sup ω∈Ω v αt (ω) − ε t ).

Then, with a certain choice of step sizes η t , Algorithm 2 produces a dual measureλ ∈ L which satisfies DISPLAYFORM0 (Alternate form.)

Suppose instead that v αt (ω t ) ≥ ρ at each time t.

Then,λ satisfies DISPLAYFORM1 We prove Theorem 4.1 in Appendix C. For convenience, we state the first version as an explicit margin bound (in terms of competitive ratio M/M * ): Corollary 4.2.

Let M be the margin obtained by training an 1 linear SVM with the same C as in Algorithm 2, on the transformed samples {(Φ(x i ), y i )}.

Then, M is (1 − δ)-competitive with M * , the maximally achievable margin by a kernel with the assumed invariance, with DISPLAYFORM2 This bound arises from the regret analysis of online gradient ascent BID25 ; our analysis is similar to the approach of Freund & Schapire (1996) , where they present a boosting perspective.

When using an 2 -SVM, the final term can be improved to O( log T T ) (Hazan et al., 2007) .

For a general overview of results in the field, refer to Hazan (2016).

Finally, we state two (rather distinct) generalization guarantees.

Both depend mildly on a bandwidth assumption ω 2 ≤ R ω and the norm of the data R x def = max i x i .

First, we state a margindependent SVM generalization bound, due to Koltchinskii & Panchenko (2002) .

Notice the appearance of λ k 1 = R λ , justifying our choice of normalization constant for Algorithm 1.

Intriguingly, the end-to-end generalization error of our method decreases with an increasing number of random features, since the margin bound is being refined during Algorithm 2.Theorem 4.3 (Generalization via margin).

For any SVM decision function f : X → R with a kernel k λ constrained by λ 1 ≤ R λ trained on samples S drawn i.i.d.

from distribution D, the generalization error is bounded by DISPLAYFORM0 The proof can be found in Appendix D. Note that this improves on the generic result for MKL, from Theorem 2 in BID4 , which has a √ log T dependence on the number of base kernels T .

This improvement stems from the rank-one property of each component kernel.

Next, we address another concern entirely: the sample size required for v(ω) to approximate the ideal Fourier potential v ideal (ω), the squared magnitude of the Fourier transform of the signed measure P − Q arising from the true distribution.

For the shift-invariant case: .

We have that for all ω : ω ≤ R ω , with probability at least 1 − δ, DISPLAYFORM1 The O(·) suppresses factors polynomial in R x and R ω .The full statement and proof are standard, and deferred to Appendix E. In particular, this result allows for a mild guarantee of polynomial hitting-time on the locus of approximate local maxima of v ideal (as opposed to the empirical v).

Adapting the main result from BID24 :Theorem 4.5 (Langevin hitting time).

Let ω τ be the output of Algorithm 1 on v α (ω), after τ steps.

Algorithm 1 finds an approximate local maximum of v ideal in polynomial time.

That is, with U being the set of ε-approximate local maxima of v ideal (ω), some ω t satisfies DISPLAYFORM2 , with probability at least 1 − δ.

Of course, one should not expect polynomial hitting time on approximate global maxima; BID13 give asymptotic mixing guarantees.

In this section, we highlight the most important and illustrative parts of our experimental results.

For further details, we provide an extended addendum to the experimental section in Appendix A. The code can be found at github.com/yz-ignescent/Not-So-Random-Features.

First, we exhibit two simple binary classification tasks, one in R 2 and the other on S 2 , to demonstrate the power of our kernel selection method.

As depicted in FIG6 , we create datasets with sharp boundaries, which are difficult for the standard RBF and arccosine BID3 ) kernel.

In both cases, n train = 2000 and n test = 50000.6 Used on a 1 -SVM classifier, these baseline kernels saturate at 92.1% and 95.1% test accuracy, respectively; they are not expressive enough.

On the R 2 "windmill" task, Algorithm 2 chooses random features that progressively refine the decision boundary at the margin.

By T = 1000, it exhibits almost perfect classification (99.7% training, 99.3% test).

Similarly, on the S 2 "checkerboard" task, Algorithm 2 (with some adaptations described in Appendix A.3) reaches almost perfect classification (99.7% training, 99.1% test) at T = 100, supported on only 29 spherical harmonics as features.

We provide some illuminating visualizations.

FIG6 show the evolution of the dual weights, random features, and classifier.

As the theory suggests, the objective evolves to assign higher weight to points near the margin, and successive features improve the classifier's decisiveness in challenging regions (1a, bottom).

FIG6 visualizes some features from the S 2 experiment.

Next, we evaluate our kernel on standard benchmark binary classification tasks.

Challenging label pairs are chosen from the MNIST (LeCun et al., 1998) and CIFAR-10 (Krizhevsky, 2009) datasets; each task consists of ∼10000 training and ∼2000 test examples; this is considered to be large-scale for kernel methods.

Following the standard protocol from Yu et al. FORMULA1 , 512-dimensional HoG features BID9 are used for the CIFAR-10 tasks instead of raw images.

Of course, our intent is not to show state-of-the-art results on these tasks, on which deep neural networks easily dominate.

Instead, the aim is to demonstrate viability as a scalable, principled kernel method.

We compare our results to baseline random features-based kernel machines: the standard RBF random features (RBF-RF for short), and the method of Sinha & Duchi (2016) (LKRF), 7 using the same 1 -SVM throughout.

As shown by Table 1 , our method reliably outperforms these baselines, most significantly in the regime of few features.

Intuitively, this lines up with the expectation that isotropic random sampling becomes exponentially unlikely to hit a good peak in high dimension; our method searches for these peaks.

Furthermore, our theory predicts that the margin keeps improving, regardless of the dimensionality of the feature maps.

Indeed, as our classifier saturates on the training data, test accuracy continues increasing, without overfitting.

This decoupling of generalization from model complexity is characteristic of boosting methods.

7 Traditional MKL methods are not tested here, as they are noticeably (> 100 times) slower.

In practice, our method is robust with respect to hyperparameter settings.

As well, to outperform both RBF-RF and LKRF with 5000 features, our method only needs ∼ 100 features.

Our GPU implementation reaches this point in ∼30 seconds.

See Appendix A.1 for more tuning guidelines.

We have presented an efficient kernel learning method that uses tools from Fourier analysis and online learning to optimize over two natural infinite families of kernels.

With this method, we show meaningful improvements on benchmark tasks, compared to related random features-based methods.

Many theoretical questions remain, such as accelerating the search for Fourier peaks (e.g. Hassanieh et al. FORMULA1 ; Kapralov FORMULA1 ).

These, in addition to applying our learned kernels to state-of-the-art methods (e.g. convolutional kernel networks (Mairal et al., 2014; BID21 Mairal, 2016) ), prove to be exciting directions for future work.

Algorithm 2 gives a high-level outline of the essential components of our method.

However, it conceals several hyperparameter choices and algorithmic heuristics, which are pertinent when applying our method in practice.

We discuss a few more details in this section.

Throughout all experiments presented, we use hinge-loss SVM classifiers with C = 1.

Note that the convergence of Algorithm 2 depends quadratically on C.With regard to Langevin diffusion (Algorithm 1), we observe that the best samples arise from using high temperatures and Gaussian parallel initialization.

For the latter, a rule-of-thumb is to initialize 500 parallel copies of Langevin dynamics, drawing the initial position {ω 0 } from a centered isotropic Gaussian with 1.5× the variance of the optimal RBF random features. (In turn, a common rule-of-thumb for this bandwidth is the median of pairwise Euclidean distances between data points.)The step size in Algorithm 1 is tuned based on the magnitude of the gradient on v(ω), which can be significantly smaller than the upper bound derived in Section E. As is standard practice in Langevin Monte Carlo methods, the temperature is chosen so that the pertubation is roughly at the same magnitude as the gradient step.

Empirically, running Langevin dynamics for ∼100 steps suffices to locate a reasonably good peak.

To further improve efficiency, one can modify Algorithm 1 to pick the top k ≈ 10 samples, a k-fold speedup which does not degrade the features much.

The step size of online gradient ascent is set to balance between being conservative and promoting diverse samples; these steps should not saturate (thereby solving the dual SVM problem), in order to have the strongest regret bound.

In our experiments, we find that the step size achieving the standard regret guarantee (scaling as 1/ √ T ) tends to be a little too conservative.

On the other hand, it never hurts (and seems important in practice) to saturate the peak-finding routine (Algorithm 1), since this contributes an additive improvement to the margin bound.

Noting that the objective is very smooth (the k-th derivative scales as R k x ), it may be beneficial to refine the samples using a few steps of gradient descent with a very small learning rate, or an accelerated algorithm for finding approximate local minima of smooth non-convex functions; see, e.g. BID0 .

A quick note on projection onto the feasible set K = {0 α C, y T α = 0} of the SVM dual convex program: it typically suffices in practice to use alternating projection.

This feasible set is the intersection of a hyperplane and a hypercube; both of which admit a simple projection step.

The alternation projection onto the intersection of two non-empty convex sets was originally proposed by von BID20 .

The convergence rate can be shown to be linear.

To obtain the dual variables in our experiments, we use 10 such alternating projections.

This results in a dual feasible solution up to hardware precision, and is a negligible component of the total running time (for which the parallel gradient computations are the bottleneck).

Input: α ∈ R n Parameters: box constraint C, label vector y. repeat Project onto the box: α = clip(α, 0, C) Project onto the hyperplane: DISPLAYFORM0

As we note in Section 3.1, it is unclear how to define gradient Langevin dynamics on v(l, m) in the inner-product case, since no topology is available on the indices (l, m) of the spherical harmonics.

One option is to emulate Langevin dynamics, by constructing a discrete Markov chain which mixes to λ(l, m) ∝ e βv(l,m) .However, we find in our experiments that it suffices to compute λ(l, m) by examining all values of v(l, m) with j no more than some threshold J. One should view this as approximating the kerneltarget alignment objective function via Fourier truncation.

This is highly parallelizable: it involves approximately N (m, d) degree-J polynomial evaluations on the same sample data, which can be expressed using matrix multiplication.

In our experiments, it sufficed to examine the first 1000 coefficients; we remark that it is unnatural in any real-world datasets to expect that v(ω) only has large values outside the threshold J.Under Fourier truncation, the domain of λ becomes a finite-dimensional simplex.

In the gametheoretic view of 4.1, an approximate Nash equilibrium becomes concretely achievable via Nesterov's excessive gap technique (Nesterov, 2005; BID10 , given that the kernel player's actions are restricted to a mixed strategy over a finite set of basis kernels.

Finally, we note a significant advantage to this setting, where we have a discrete set of Fourier coefficients: the same feature might be found multiple times.

When a duplicate feature is found, it need not be concatenated to the representation; instead, the existing feature is scaled appropriately.

This accounts for the drastically smaller support of features required to achieve near-perfect classification accuracy.

In this section, we go into more detail about the spherical harmonics in d dimensions.

Although all of this material is standard in harmonic analysis, we provide this section for convenience, isolating only the relevant facts.

First, we provide a proof sketch for Theorem 2.2.

We rely on the following theorem, an analogue of Bochner's theorem on the sphere, which characterizes rotation-invariant kernels: DISPLAYFORM0 is positive semi-definite if and only if its expansion into Gegenbauer polynomials P d i has only non-negative coefficients, i.e. DISPLAYFORM1 with λ m ≥ 0, ∀m ∈ N + .The Gegenbauer polynomials P DISPLAYFORM2 where |S

We now specify the involution ω → −ω on indices of spherical harmonics, which gives a pairing that takes the role of opposite Fourier coefficients in the X = R d case.

In particular, the Fourier transform λ of a real-valued function k : R n → R satisfies λ(ω) = λ(−ω), so that the dual measure can be constrained to be symmetric.

Now, consider the X = S d−1 case, where the Fourier coefficients are on the set Ω of valid indices of spherical harmonics.

We would like a permutation on the indices σ so that σ(σ(ω)) = ω, and λ(ω) = λ(σ(ω)) whenever λ is the spherical harmonic expansion of a real kernel.

DISPLAYFORM0 where the functions in the product come from a certain family of associated Legendre functions in sin θ.

For a detailed treatment, we adopt the construction and conventions from Higuchi (1987).In the scope of this paper, the relevant fact is that the desired involution is well-defined in all dimensions.

Namely, consider the permutation σ that sends a spherical harmonic indexed by ω DISPLAYFORM1 The symmetry condition in Theorem 2.2 follows straightforwardly.

By orthonormality, we know that every square-integrable function f : R n → C has a unique decomposition into spherical harmonics, with coefficients λ ω = f, Y ω , so that λ −ω = f, Y ω = λ ω .

When f is real-valued, we conclude that λ −ω = −λ ω , as claimed.

In this section, we prove the main theorem, which quantifies convergence of Algorithm 2 to the Nash equilibrium.

We restate it here: Theorem 4.1.

Assume that during each timestep t, the call to Algorithm 1 returns an ε t -approximate global maximizer ω t (i.e.v αt (ω t ) ≥ max ω∈Kvαt (ω) − ε t ).

Then, Algorithm 2 returns a dual measureλ, which satisfies DISPLAYFORM0 Alternatively with the assumption that at each timestep t,v αt (ω t ) ≥ ρ,λ, satisfies DISPLAYFORM1 If Algorithm 2 is used on a l 2 -SVM, the regret bound can be improved to be O( DISPLAYFORM2 Proof.

We will make use of the regret bound of online gradient ascent (see, e.g., (Hazan, 2016) ).Here we only prove the theorem in the case for l 1 -SVM with box constraint C, under the assumption of ε t -approximate optimality of Algorithm 1.

Extending the proof to other cases is straightfoward.

Lemma C.1 (Regret bound for online gradient ascent).

Let D be the diameter of the constraint set K, and G a Lipschitz constant for an arbitrary sequence of concave functions f t (α) DISPLAYFORM3 , guarantees the following for all T ≥ 1: DISPLAYFORM4 Here, D ≤ C √ n by the box constraint, and we have DISPLAYFORM5 Thus, our regret bound is DISPLAYFORM6 ; that is, for all T ≥ 1, DISPLAYFORM7 Since at each timestep t,v αt (ω t ) ≥ max ω∈Kvαt (ω) − ε t , we have by assumption DISPLAYFORM8 and by concavity of F in α, we know that, for any fixed sequence α 1 , α 2 , . . .

, α T , To complete the proof, note that for a given α ∈ K,

T t=1 F (α, δ ωt + δ −ωt ) = F (α,λ) by linearity of F in the dual measure; here,λ = 1 2T T t=1 δ ωt +δ −ωt is the approximate Nash equilibrium found by the no-regret learning dynamics.

D PROOF OF THEOREM 4.3 In this section, we compute the Rademacher complexity of the composition of the learned kernel and the classifier, proving Theorem 4.3.

DISPLAYFORM0 Putting everything together, we finally obtain that DISPLAYFORM1 from which we apply the result from Koltchinskii & Panchenko (2002) to conclude the theorem.

In this section, we prove the following theorem about concentration of the Fourier potential.

It suffices to disregard the reweighting vector α; to recover a guarantee in this case, simply replace ε with ε/C 2 .

Note that we only argue this in the translation-invariant case.

we have that with probability at least 1 − δ, 1 n 2 v (n) (ω) − v ideal (ω) ≤ ε for all ω ∈ Ω such that ω ≤ R ω .Let P (n) , Q (n) denote the empirical measures from a sample S of size n, arising from i.i.d.

samples from the true distribution, whose classes have measures P, Q, adopting the convention P(X ) + Q(X ) = 1.

Then, in expectation over the sampling, and adopting the same normalization conventions as in the paper, we have E[P (n) /n] = P and E[Q (n) /n] = Q for every n.

Let v (n) (ω) denote the empirical Fourier potential, computed from P (n) , Q (n) , so that we have E[v (n) (ω)/n 2 ] = v ideal (ω).

The result follows from a concentration bound on v (n) (ω)/n 2 .

We first show that it is Lipschitz: is 2n 2 R x -Lipschitz with respect to ω.

Proof.

We have Thus, the Lipschitz constant of v (n) (ω)/n 2 scales linearly with the norm of the data, a safe assumption.

Next, we show Lipschitzness with respect to a single data point: DISPLAYFORM0 Lemma E.2 (Lipschitzness of v (n) in x i ).

v (n) (ω) is 2nR ω -Lipschitz with respect to any x i .

<|TLDR|>

@highlight

A simple and practical algorithm for learning a margin-maximizing translation-invariant or spherically symmetric kernel from training data, using tools from Fourier analysis and regret minimization.

@highlight

The paper proposes to learn a custom translation or rotation invariant kernal in the Fourier representation to maximize the margin of SVM.

@highlight

The authors propose an interesting algorithm for learning the l1-SVM and the Fourier represented kernel together

@highlight

The authors consider learning directly Fourier representations of shift/translation invariant kernels for machine learning applications with the alignment of the kernel to data as the objective function to optimize.