Owing to their connection with generative adversarial networks (GANs), saddle-point problems have recently attracted considerable interest in machine learning and beyond.

By necessity, most theoretical guarantees revolve around convex-concave (or even linear) problems; however, making theoretical inroads towards efficient GAN training depends crucially on moving beyond this classic framework.

To make piecemeal progress along these lines, we analyze the behavior of mirror descent (MD) in a class of non-monotone problems whose solutions coincide with those of a naturally associated variational inequality – a property which we call coherence.

We first show that ordinary, “vanilla” MD converges under a strict version of this condition, but not otherwise; in particular, it may fail to converge even in bilinear models with a unique solution.

We then show that this deficiency is mitigated by optimism: by taking an “extra-gradient” step, optimistic mirror descent (OMD) converges in all coherent problems.

Our analysis generalizes and extends the results of Daskalakis et al. [2018] for optimistic gradient descent (OGD) in bilinear problems, and makes concrete headway for provable convergence beyond convex-concave games.

We also provide stochastic analogues of these results, and we validate our analysis by numerical experiments in a wide array of GAN models (including Gaussian mixture models, and the CelebA and CIFAR-10 datasets).

The surge of recent breakthroughs in deep learning has sparked significant interest in solving optimization problems that are universally considered hard.

Accordingly, the need for an effective theory has two different sides: first, a deeper understanding would help demystify the reasons behind the success and/or failures of different training algorithms; second, theoretical advances can inspire effective algorithmic tweaks leading to concrete performance gains.

For instance, using tools from the theory of dynamical systems, BID28 BID29 and Panageas & Piliouras [2017] showed that a wide variety of first-order methods (including gradient descent and mirror descent) almost always avoid saddle points.

More generally, the optimization and machine learning communities alike have dedicated significant effort in understanding non-convex landscapes by searching for properties which could be leveraged for efficient training.

As an example, the "strict saddle" property was shown to hold in a wide range of salient objective functions ranging from low-rank matrix factorization BID8 BID20 and dictionary learning [Sun et al., 2017a,b] , to principal component analysis BID19 , and many other models.

On the other hand, adversarial deep learning is nowhere near as well understood, especially in the case of generative adversarial networks (GANs) BID22 .

Despite an immense amount of recent scrutiny, our theoretical understanding cannot boast similar breakthroughs as in "single-agent" deep learning.

Because of this, a considerable corpus of work has been devoted to exploring and enhancing the stability of GANs, including techniques as diverse as the use of Wasserstein metrics , critic gradient penalties BID23 , feature matching, minibatch discrimination, etc. [Radford et al., 2016; Salimans et al., 2016] .Even before the advent of GANs, work on adaptive dynamics in general bilinear zero-sum games (e.g. Rock-Paper-Scissors) established that they lead to persistent, chaotic, recurrent (i.e. cycle-like) behavior [Sato et al., 2002; Piliouras & Shamma, 2014; Piliouras et al., 2014] .

Recently, simple specific instances of cycle-like behavior in bilinear games have been revisited mainly through the lens of GANs BID15 Mescheder et al., 2018; Papadimitriou & Piliouras, 2018] .

Two important recent results have established unified pictures about the behavior of continuous and discrete-time first order methods in bilinear games: First, established that continuous-time descent methods in zero-sum games (e.g., gradient descent, follow-the-regularized-leader and the like) are Poincaré recurrent, returning arbitrarily closely to their initial conditions infinitely many times.

Second, BID4 examined the discrete-time analogues (gradient descent, multiplicative weights and follow-the-regularized-leader) showing that orbits spiral slowly outwards.

These recurrent systems have formal connections to Hamiltonian dynamics and do not behave in a gradient-like fashion BID6 ; BID5 .

This is a critical failure of descent methods, but one which BID15 showed can be overcome through "optimism", interpreted in this context as an "extra-gradient" step that pushes the training process further along the incumbent gradient -as a result, optimistic gradient descent (OGD) succeeds in cases where vanilla gradient descent (GD) fails (specifically, unconstrained bilinear saddle-point problems).A common theme in the above is that, to obtain a principled methodology for training GANs, it is beneficial to first establish improvements in a more restricted setting, and then test whether these gains carry over to more demanding learning environments.

Following these theoretical breadcrumbs, we focus on a class of non-monotone problems whose solutions are related to those of a naturally associated variational inequality, a property which we call coherence.

Then, hoping to overcome the shortcomings of ordinary descent methods by exploiting the problem's geometry, we examine the convergence of MD in coherent problems.

On the positive side, we show that if a problem is strictly coherent (a condition satisfied by all strictly convex-concave problems), MD converges almost surely, even in stochastic problems (Theorem 3.1).

However, under null coherence (the "saturated" opposite to strict coherence), MD spirals outwards from the problem's solutions and may cycle in perpetuity.

The null coherence property covers all bilinear models, so this result encompasses fully the analysis of BID4 for GD and follow-the-regularized-leader (FTRL) in general bilinear zero-sum games within our coherence framework.

Thus, in and by themselves, gradient/mirror descent methods do not suffice for training convoluted, adversarial deep learning models.

To mitigate this deficiency, we consider the addition of an extra-gradient step which looks ahead and takes an additional step along a "future" gradient.

This technique was first introduced by BID27 and subsequently gained great popularity as the basis of the mirror-prox algorithm of Nemirovski [2004] which achieves an optimal O(1/n) convergence rate in Lipschitz monotone variational inequalities (see also Nesterov, 2007 , for a primal-dual variant of the method and BID25 , for an extension to stochastic variational inequalities and saddle-point problems).In the learning literature, the extra-gradient technique (or, sometimes, a variant thereof) is often referred to as optimistic mirror descent (OMD) [Rakhlin & Sridharan, 2013] and its effectiveness in GAN training was recently examined by BID15 and Yadav et al. [2018] (the latter involving a damping mechanism for only one of the players).

More recently, BID21 considered a variant method which incorporates a mechanism that "extrapolates from the past" in order to circumvent the need for a second oracle call in the extra-gradient step.

Specifically, BID21 showed that the extra-gradient algorithm with gradient reuse converges a) geometrically in strongly monotone, deterministic variational inequalities; and b) ergodically in general stochastic variational inequalities, achieving in that case an oracle complexity bound that is √ 13/7/2 ≈ 68% of a bound previously established by BID25 for the mirror-prox algorithm.

However, beyond convex-concave problems, averaging offers no tangible benefits because there is no way to relate the value of the ergodic average to the value of the iterates.

As a result, moving closer to GAN training requires changing both the algorithm's output as well as the accompanying analysis.

With this as our guiding principle, we first show that the last iterate of OMD converges in all coherent problems, including null-coherent ones.

As a special case, this generalizes and extends the results of Noor et al. [2011] for OGD in pseudo-monotone problems, and also settles in the affirmative an open question of BID15 concerning the convergence of the last iterate of OGD in nonlinear problems.

Going beyond deterministic problems, we also show that OMD converges with probability 1 even in stochastic saddle-point problems that are strictly coherent.

These results complement the existing literature on the topic by showing that a cheap extra-gradient add-on can lead to significant performance gains when applied to state-of-the-art methods (such as Adam).

We validate this prediction for a wide array of standard GAN models in Section 5.

Saddle-point problems.

Consider a saddle-point problem of the general form min DISPLAYFORM0 where each feasible region X i , i = 1, 2, is a compact convex subset of a finite-dimensional normed space V i ≡ d i , and f : X ≡ X 1 × X 2 → denotes the problem's value function.

From a gametheoretic standpoint, (SP) can be seen as a zero-sum game between two optimizing agents (or players): Player 1 (the minimizer) seeks to incur the least possible loss, while Player 2 (the maximizer) seeks to obtain the highest possible reward -both determined by f (x 1 , x 2 ).To solve (SP), we will focus on incremental processes that exploit the individual loss/reward gradients of f (assumed throughout to be at least C 1 -smooth).

Since the individual gradients of f will play a key role in our analysis, we will encode them in a single vector as DISPLAYFORM1 and, following standard conventions, we will treat g(x) as an element of Y ≡ V * , the dual of the ambient space V ≡ V 1 × V 2 , assumed to be endowed with the product norm x 2 = x 1 2 + x 2 2 .Variational inequalities and coherence.

Most of the literature on saddle-point problems has focused on the monotone case, i.e., when f is convex-concave.

In such problems, it is well known that solutions of (SP) can be characterized equivalently as solutions of the Stampacchia variational inequality g(x * ), x − x * ≥ 0 for all x ∈ X (SVI) or, in Minty form: DISPLAYFORM2 The equivalence between solutions of (SP), (SVI) and (MVI) extends well beyond the realm of monotone problems: it trivially includes all bilinear problems ( f (x 1 , x 2 ) = x 1 Mx 2 ), pseudo-monotone objectives as in Noor et al. [2011] , etc.

For a concrete example which is not even pseudo-monotone, consider the problem min DISPLAYFORM3 The only saddle-point of f is x * = (0, 0): it is easy to check that x * is also the unique solution of the corresponding variational inequality (VI) problems, despite the fact that f is not even pseudomonotone.1 This shows that this equivalence encompasses a wide range of phenomena that are innately incompatible with convexity/monotonicity, even in the lowest possible dimension; for an in-depth discussion we refer the reader to BID16 .Motivated by all this, we introduce below the following notion of coherence:Definition 2.1.

We say that (SP) is coherent if:1.

Every solution of (SVI) also solves (SP).2.

There exists a solution p of (SP) that satisfies (MVI).

* of (SP) satisfies (MVI) locally, i.e., for all x sufficiently close to x * .In the above, if (MVI) holds as a strict inequality whenever x is not a solution thereof, (SP) will be called strictly coherent; by contrast, if (MVI) holds as an equality for all x ∈ X , we will say that (SP) is null-coherent.

The notion of coherence will play a central part in our considerations, so a few remarks are in order.

To the best of our knowledge, its first antecedent is a gradient condition examined by BID9 in the context of nonlinear programming; we borrow the term "coherence" from the more recent paper of Zhou et al. [2017b] who used the term "variational coherence" for a stronger variant of the above definition.

We should also note here that the set of solutions of a coherent problem does not need to be convex: for instance, if Player 1 controls (x, y), and the objective function is f (x, y) = x 2 y 2 (i.e., Player 2 has no impact in the game), the set of solutions is the non-convex set X * = {(x, y) : x = 0 or y = 0}. Moreover, regarding the distinction between coherence and strict coherence, we show in Appendix A that (SP) is strictly coherent when f is strictly convex-concave.

At the other end of the spectrum, typical examples of problems that are null-coherent are bilinear objectives with an interior solution: for instance, f (x 1 , x 2 ) = x 1 x 2 with x 1 , x 2 ∈ [−1, 1] has g(x), x = x 1 x 2 − x 2 x 1 = 0 for all x 1 , x 2 ∈ [−1, 1], so it is null-coherent.

Finally, neither strict, nor null coherence imply a unique solution to (SP), a property which is particularly relevant for GANs (the first example above is strictly coherent, but does not admit a unique solution).

The method.

Motivated by its prolific success in convex programming, our starting point will be the well-known mirror descent (MD) method of Nemirovski & Yudin [1983] , suitably adapted to our saddle-point context.

Several variants of the method exist, ranging from dual averaging [Nesterov, 2009 ] to follow-the-regularized-leader; for a survey, we refer the reader to BID12 .The basic idea of mirror descent is to generate a new state variable x + from some starting state x by taking a "mirror step" along a gradient-like vector y. To do this, let h : X → be a continuous and K-strongly convex distance-generating function (DGF) on X , i.e., DISPLAYFORM0 for all x, x ∈ X and all t ∈ [0, 1].

In terms of smoothness (and in a slight abuse of notation), we also assume that the subdifferential of h admits a continuous selection, i.e., a continuous function ∇h : dom ∂h → Y such that ∇h(x) ∈ ∂h(x) for all x ∈ dom ∂h.

2 Then, following BID11 , h generates a pseudo-distance on X via the relation DISPLAYFORM1 This pseudo-distance is known as the Bregman divergence.

As we show in Appendix B, we have DISPLAYFORM2 , so the convergence of a sequence X n to some target point p can be verified by showing that D(p, X n ) → 0.

On the other hand, D(p, x) typically fais to be symmetric and/or satisfy Require: K-strongly convex regularizer h : X → , step-size sequence γ n > 0 1: choose X ∈ dom ∂h # initialization 2: for n = 1, 2, . . .

do 3:oracle query at X returns g # gradient feedback 4: set X ← P X (−γ n g) # new state 5: end for 6: return X the triangle inequality, so it is not a true distance function per se.

Moreover, the level sets of D(p, x) may fail to form a neighborhood basis of p, so the convergence of X n to p does not necessarily imply that D(p, X n ) → 0; we provide an example of this behavior in Appendix B. For technical reasons, it will be convenient to assume that such phenomena do not occur, i.e., that D(p, X n ) → 0 whenever X n →

p.

This mild regularity condition is known in the literature as "Bregman reciprocity" BID13 BID0 BID30 BID10 , and it will be our standing assumption in what follows (note also that it holds trivially for both Examples 3.1 and 3.2 below).

Now, as with standard Euclidean distances, the Bregman divergence generates an associated proxmapping defined as P x (y) = arg min DISPLAYFORM3 In analogy with the Euclidean case (discussed below), the prox-mapping (3.3) produces a feasible point x + = P x (y) by starting from x ∈ dom ∂h and taking a step along a dual (gradient-like) vector y ∈ Y. In this way, we obtain the mirror descent (MD) algorithm DISPLAYFORM4 where γ n is a variable step-size sequence andĝ n is the calculated value of the gradient vector g(X n ) at the n-th stage of the algorithm (for a pseudocode implementation, see Section 3).For concreteness, two widely used examples of prox-mappings are as follows:Example 3.1 (Euclidean projections).

When X is endowed with the L 2 norm · 2 , the archetypal prox-function is the (square of the) norm itself, i.e., h(x) = and the induced prox-mapping is P x (y) = Π(x + y), (3.4) with Π(x) = arg min x ∈X x − x 2 denoting the ordinary Euclidean projection onto X .Example 3.2 (Entropic regularization).

When X is a d-dimensional simplex, a widely used DGF is the (negative) Gibbs-Shannon entropy h(x) = d j=1 x j log x j .

This function is 1-strongly convex with respect to the L 1 norm [Shalev-Shwartz, 2011 ] and the associated pseudo-distance is the Kullback-Leibler divergence D KL (p, x) = d j=1 p j log(p j /x j ); in turn, this yields the prox-mapping DISPLAYFORM5 The update rule x ← P x (y) is known in the literature as the multiplicative weights (MW) algorithm BID2 , and is one of the centerpieces for learning in games BID18 BID17 BID14 , adversarial bandits BID3 , etc.

Regarding the gradient input sequenceĝ n of (MD), we assume that it is obtained by querying a first-order oracle which outputs an estimate of g(X n ) when called at X n .

This oracle could be either perfect, returningĝ n = g(X n ) for all n, or imperfect, providing noisy gradient estimations.3 By that token, we will make the following standard assumptions for the gradient feedback sequenceĝ n [

Nesterov, 2007; Nemirovski et al., 2009; BID25 : a) Unbiasedness: DISPLAYFORM6 In the above, y * ≡ sup{ y, x : x ∈ V, x ≤ 1} denotes the dual norm on Y while F n represents the history (natural filtration) of the generating sequence X n up to stage n (inclusive).

Sinceĝ n is generated randomly from X n at stage n, it is obviously not F n -measurable, i.e.,ĝ n = g(X n ) + U n+1 , where U n is an adapted martingale difference sequence with ¾[ U n+1 2 * | F n ]

≤ σ 2 for some finite σ ≥ 0.

Clearly, when σ = 0, we recover the exact gradient feedback frameworkĝ n = g(X n ).Convergence analysis.

When (SP) is convex-concave, it is customary to take as the output of (MD) the so-called ergodic averageX DISPLAYFORM7 or some other average of the sequence X n where the objective is sampled.

The reason for this is that convexity guarantees -via Jensen's inequality and gradient monotonicity -that a regret-based analysis of (MD) can lead to explicit rates for the convergence ofX n to the solution set of (SP) [Nemirovski, 2004; Nesterov, 2007] .

However, when the problem is not convex-concave, the standard proof techniques for establishing convergence of the method's ergodic average no longer apply; instead, we need to examine the convergence properties of the generating sequence X n of (MD) directly.

With all this in mind, our main result for (MD) may be stated is as follows: Theorem 3.1.

Suppose that (MD) is run with a gradient oracle satisfying (3.6) and a variable step-size sequence γ n such that DISPLAYFORM8 This result establishes an important dichotomy between strict and null coherence: in strictly coherent problems, X n is attracted to the solution set of (SP); in null-coherent problems, X n drifts away and cycles without converging.

In particular, this dichotomy leads to the following immediate corollaries: Corollary 3.2.

Suppose that f is strictly convex-concave.

Then, with assumptions as above, X n converges (a.s.) to the (necessarily unique) solution of (SP).

Corollary 3.3.

Suppose that f is bilinear and admits an interior saddle-point x * ∈ X • .

If X 1 x * and (MD) is run with exact gradient input (σ = 0), we have lim n→∞ D(x * , X n ) > 0.Since bilinear models include all finite two-player, zero-sum games, Corollary 3.3 also encapsulates the non-convergence results of BID15 and Bailey & Piliouras [2018] for gradient descent and FTRL respectively (for a more comprehensive formulation, see Proposition C.3 in Appendix C).

The failure of (MD) to converge in this case is due to the fact that, witout a mitigating mechanism in place, a "blind" first-order step could overshoot and spiral outwards, even with a vanishing step-size.

This becomes even more pronounced in GANs where it can lead to mode collapse and/or cycles between different modes; the next two sections address precisely these issues.4 Extra-gradient analysisThe method.

In convex-concave problems, taking an average of the algorithm's generated samples as in (3.7) may resolve cycling phenomena by inducing an auxiliary sequence that gravitates towards the "center of mass" of the driving sequence X n (which orbits interior solutions).

However, this technique cannot be employed in problems that are not convex-concave because the structure of f cannot be leveraged to establish convergence of the ergodic average of the process.

In view of this, we replace averaging with an optimistic "extra-gradient" step which uses the obtained information to amortize the next prox step (possibly by exiting the convex hull of generated states).The seed of this "extra-gradient" idea dates back to BID27 and Nemirovski [2004] , and has since found wide applications in optimization theory and beyond -for a survey, see BID12 and references therein.

In a nutshell, given a state x, the extra-gradient method first generates an intermediate, "waiting" statê x = P x (−γg(x)) by taking a prox step as usual.

However, instead of continuing fromx, the method samples g(x) and goes back to the original state x in order to generate a new state x + = P x (−γg(x)).

Based on this heuristic, we obtain the optimistic mirror descent (OMD) algorithm DISPLAYFORM9 Algorithm 2: optimistic mirror descent (OMD) for saddle-point problemsRequire: K-strongly convex regularizer h : X → , step-size sequence γ n > 0 1: choose X ∈ dom ∂h # initialization 2: for n = 1, 2, . . .

do 3:oracle query at X returns g # gradient feedback 4:set X + ← P X (−γ n g) # waiting state 5:oracle query at X + returns g + # gradient feedback 6: set X ← P X (−γ n g + ) # new state 7: end for 8: return X where, in obvious notation,ĝ n andĝ n+1/2 represent gradient oracle queries at the incumbent and intermediate states X n and X n+1/2 respectively.

For a pseudocode implementation, see Algorithm 2; see also Rakhlin & Sridharan [2013] and BID15 for a variant of the method with a "momentum" step, and BID21 for a gradient reuse mechanism that replaces a second oracle call with a past gradient.

Convergence analysis.

In his original analysis, Nemirovski [2004] considered the ergodic average (3.7) of the algorithm's iterates and established an O(1/n) convergence rate in monotone problems.

However, as we explained above, even though this kind of averaging is helpful in convex-concave problems, it does not provide any tangible benefits beyond this class: in more general problems, X n appears to be the most natural solution candidate.

Our first result below justifies this choice in the class of coherent problems: Theorem 4.1.

Suppose that (SP) is coherent and g is L-Lipschitz continuous.

If (OMD) is run with exact gradient input (σ = 0) and γ n such that 0 < inf n γ n ≤ sup n γ n < K/L, the sequence X n converges monotonically to a solution x * of (SP), i.e., D(x * , X n ) decreases monotonically to 0.Corollary 4.2.

Suppose that f is bilinear.

If (OMD) is run with assumptions as above, the sequence X n converges monotonically to a solution of (SP). [2011] for pseudo-monotone problems.

Importantly, Theorem 4.1 shows that the extra-gradient step plays a crucial role in stabilizing (MD): not only does (OMD) converge in problems where (MD) provably fails, but this convergence is, in fact, monotonic.

In other words, at each iteration, (OMD) comes closer to a solution of (SP), whereas (MD) may spiral outwards, ultimately converging to a limit cycle.

This phenomenon is seen clearly in Fig. 1 , and also in the detailed analysis of Appendix C.Of course, except for very special cases, the monotonic convergence of X n cannot hold when the gradient input to (OMD) is imperfect: a single "bad" sample ofĝ n would suffice to throw X n off-track.

In this case, we have: Theorem 4.3.

Suppose that (SP) is strictly coherent and (OMD) is run with a gradient oracle satisfying (3.6) and a variable step-size sequence γ n such that DISPLAYFORM10 Then, with probability 1, X n converges to a solution of (SP).It is worth noting here that the step-size policy in Theorem 4.3 is different than that of Theorem 4.1.

This is due to a) the lack of randomness (which obviates the summability requirement ∞ n=1 γ 2 n < ∞ in Theorem 4.1); and b) the lack of Lipschitz continuity assumption (which, in the case of Theorem 4.1 guarantees monotonic decrease at each step, provided the step-size is not too big).

Importantly, the maximum allowable step-size is also controlled by the strong convexity modulus of h, suggesting that the choice of distance-generating function can be fine-tuned further to allow for more aggressive step-size policies -a key benefit of mirror descent methods.

Gaussian mixture models.

For the experimental validation of our theoretical results, we began by evaluating the extra-gradient add-on in a highly multi-modal mixture of 16 Gaussians arranged in a 4 × 4 grid as in Metz et al. [2017] .

The generator and discriminator have 6 fully connected layers with 384 neurons and Relu activations (plus an additional layer for data space projection), and the generator generates 2-dimensional vectors.

The output after {4000, 8000, 12000, 16000, 20000} iterations is shown in FIG3 .

The networks were trained with RMSprop [Tieleman & Hinton, 2012] and Adam BID26 , and the results are compared to the corresponding extra-gradient variant (for an explicit pseudocode representation in the case of Adam, see BID15 and Appendix E).

Learning rates and hyperparameters were chosen by an inspection of grid search results so as to enable a fair comparison between each method and its look-ahead version.

Overall, the different optimization strategies without look-ahead exhibit mode collapse or oscillations throughout the training period (we ran all models for at least 20000 iterations in order to evaluate the hopping behavior of the generator).

In all cases, the extra-gradient add-on performs consistently better in learning the multi-modal distribution and greatly reduces occurrences of oscillatory behavior.

Experiments with standard datasets.

In our experiments with Gaussian mixture models (GMMs), the most promising training method was Adam with an extra-gradient step (a concrete pseudocode implementation is provided in Appendix E).

Motivated by this, we trained a Wasserstein-GAN on the CelebA and CIFAR-10 datasets using Adam, both with and without an extra-gradient step.

The architecture employed was a standard DCGAN; hyperparameters and network architecture details may be found in Appendix E. Subsequently, to quantify the gains of the extra-gradient step, we employed the widely used inception score and Fréchet distance metrics, for which we report the results in FIG4 .

Under both metrics, the extra-gradient add-on provides consistently higher scores after an initial warm-up period (and is considerably more stable).

For visualization purposes, we also present in FIG5 an ensemble of samples generated at the end of the training period.

Overall, the generated samples provide accurate feature representation and low distortion (especially in CelebA).

Our results suggest that the implementation of an optimistic, extra-gradient step is a flexible add-on that can be easily attached to a wide variety of GAN training methods (RMSProp, Adam, SGA, etc.) , and provides noticeable gains in performance and stability.

From a theoretical standpoint, the dichotomy between strict and null coherence provides a justification of why this is so: optimism eliminates cycles and, in so doing, stabilizes the method.

We find this property particularly appealing because it paves the way to a local analysis with provable convergence guarantees in multi-modal settings, and beyond zero-sum games; we intend to examine this question in future work.

We begin our discussion with some basic results on coherence:Proposition A.1.

If f is convex-concave, (SP) is coherent.

In addition, if f is strictly convex-concave, (SP) is strictly coherent.

Proof.

Let x * be a solution point of (SP).

Since f is convex-concave, first-order optimality gives DISPLAYFORM0 and DISPLAYFORM1 Combining the two, we readily obtain the (Stampacchia) variational inequality DISPLAYFORM2 In addition to the above, the fact that f is convex-concave also implies that g(x) is monotone in the sense that DISPLAYFORM3 for all x, x ∈ X [Bauschke & Combettes, 2017].

Thus, setting x ← x * in (A.3) and invoking (SVI), we get DISPLAYFORM4 i.e., (MVI) is satisfied.

To establish the converse implication, focus for concreteness on the minimizer, and note that (MVI) implies that DISPLAYFORM5 Now, if we fix some x 1 ∈ X 1 and consider the function φ(t) = f (x * 1 + t(x 1 − x * 1 ), x * 2 ), the inequality (A.5) yields DISPLAYFORM6 2 ).

The maximizing component follows similarly, showing that x * is a solution of (SP) and, in turn, establishing that (SP) is coherent.

For the strict part of the claim, the same line of reasoning shows that if g(x), x − x * = 0 for some x that is not a saddle-point of f , the function φ(t) defined above must be constant on [0, 1], indicating in turn that f cannot be strictly convex-concave, a contradiction.

We proceed to show that the solution set of a coherent saddle-point problem is closed (we will need this regularity result in the convergence analysis of Appendix C):Lemma A.2.

Let X * denote the solution set of (SP).

If (SP) is coherent, X * is closed.

Proof.

Let x * n , n = 1, 2, . . .

, be a sequence of solutions of (SP) converging to some limit point x * ∈ X .

To show that X * is closed, it suffices to show that x * ∈ X .Indeed, given that (SP) is coherent, every solution thereof satisfies (MVI), so we have g(x), x−x * n ≥ 0 for all x ∈ X .

With x * n → x * as n → ∞, it follows that DISPLAYFORM7 i.e., x * satisfies (MVI).

By coherence, this implies that x * is a solution of (SP), as claimed.

In this appendix, we provide some auxiliary results and estimates that are used throughout the convergence analysis of Appendix C. Some of the results we present here (or close variants thereof) are not new [see e.g., Nemirovski et al., 2009; BID25 .

However, the hypotheses used to obtain them vary wildly in the literature, so we provide all the necessary details for completeness.

To begin, recall that the Bregman divergence associated to a K-strongly convex distance-generating function h : X → is defined as DISPLAYFORM0 with ∇h(x) denoting a continuous selection of ∂h(x).

The induced prox-mapping is then given by DISPLAYFORM1 and is defined for all x ∈ dom ∂h, y ∈ Y (recall here that Y ≡ V * denotes the dual of the ambient vector space V).

In what follows, we will also make frequent use of the convex conjugate h DISPLAYFORM2 By standard results in convex analysis [Rockafellar, 1970, Chap.

26] , h * is differentiable on Y and its gradient satisfies the identity ∇h * (y) = arg max DISPLAYFORM3 For notational convenience, we will also write Q(y) = ∇h * (y) (B.5) and we will refer to Q : Y → X as the mirror map generated by h. All these notions are related as follows: Lemma B.1.

Let h be a distance-generating function on X .

Then, for all x ∈ dom ∂h, y ∈ Y, we have: DISPLAYFORM4 Finally, if x = Q(y) and p ∈ X , we have DISPLAYFORM5 Remark.

By (B.6b), we have ∂h(x + ) ∅, i.e., x + ∈ dom ∂h.

As a result, the update rule x ← P x (y) is well-posed, i.e., it can be iterated in perpetuity.

For (B.7), by a simple continuity argument, it suffices to show that the inequality holds for interior p ∈ X • .

To establish this, let DISPLAYFORM6 Since h is strongly convex and y ∈ ∂h(x) by (B.6a), it follows that φ(t) ≥ 0 with equality if and only if t = 0.

Since ψ(t) = ∇h(x + t(p − x)) − y, p − x is a continuous selection of subgradients of φ and both φ and ψ are continuous on [0, 1], it follows that φ is continuously differentiable with φ = ψ on [0, 1].

Hence, with φ convex and φ(t) ≥ 0 = φ(0) for all t ∈ [0, 1], we conclude that φ (0) = ∇h(x) − y, p − x ≥ 0, which proves our assertion.

We continue with some basic bounds on the Bregman divergence before and after a prox step.

The basic ingredient for these bounds is a generalization of the (Euclidean) law of cosines which is known in the literature as the "three-point identity" BID13 :Lemma B.2.

Let h be a distance-generating function on X .

Then, for all p ∈ X and all x, x ∈ dom ∂h, we have DISPLAYFORM7 Proof.

By definition, we have: DISPLAYFORM8 (B.10)Our claim then follows by adding the last two lines and subtracting the first.

With this identity at hand, we have the following series of upper and lower bounds: Proposition B.3.

Let h be a K-strongly convex distance-generating function on X , fix some p ∈ X , and let x + = P x (y) for x ∈ dom ∂h, y ∈ Y. We then have: DISPLAYFORM9 Proof of (B.11a).

By the strong convexity of h, we get DISPLAYFORM10

Proof of (B.11b) and (B.11c).

By the three-point identity (B.9), we readily obtain DISPLAYFORM0 In turn, this gives DISPLAYFORM1 where, in the last step, we used (B.7) and the fact that x + = P x (y), so ∇h(x) + y ∈ ∂h(x + ).

The above is just (B.11b), so the first part of our proof is complete.

For (B.11c), the bound (B.14) gives DISPLAYFORM2 Therefore, by Young's inequality [Rockafellar, 1970] , we get DISPLAYFORM3 and hence (B.17) with the last step following from Lemma B.1 applied to x in place of p. DISPLAYFORM4 The first part of Proposition B.3 shows that X n converges to p if D(p, X n ) → 0.

However, as we mentioned in the main body of the paper, the converse may fail: in particular, we could have lim inf n→∞ D(p, X n ) > 0 even if X n →

p.

To see this, let X be the L 2 ball of d and take h(x) = − 1 − x 2 2 .

Then, a straightforward calculation gives DISPLAYFORM5 (B.19) which admits p as a solution for all c ≥ 0 (so p belongs to the closure of L c (p) even though D(p, p) = 0 by definition).

As a result, under this distance-generating function, it is possible to have X n → p even when lim inf n→∞ D(p, X n ) > 0 (simply take a sequence X n that converges to p while remaining on the same level set of D).

As we discussed in the main body of the paper, such pathologies are discarded by the Bregman reciprocity condition D(p, X n ) →

0 whenever X n →

p. (B.20) This condition comes into play at the very last part of the proofs of Theorems 3.1 and 4.1; other than that, we will not need it in the rest of our analysis.

Finally, for the analysis of the OMD algorithm, we will need to relate prox steps taken along different directions: Proposition B.4.

Let h be a K-strongly convex distance-generating function on X and fix some p ∈ X , x ∈ dom ∂h.

Then: a) For all y 1 , y 2 ∈ Y, we have: DISPLAYFORM6 i.e., P x is (1/K)-Lipschitz.b) In addition, letting x + 1 = P x (y 1 ) and x + 2 = P x (y 2 ), we have: DISPLAYFORM7 Proof.

We begin with the proof of the Lipschitz property of P x .

Indeed, for all p ∈ X , (B.7) gives ∇h(x DISPLAYFORM8 ) and rearranging, we obtain ∇h(x DISPLAYFORM9 (B.24) By the strong convexity of h, we also have K x DISPLAYFORM10 (B.25) Hence, combining (B.24) and (B.25), we get K x DISPLAYFORM11 26) and our assertion follows.

For the second part of our claim, the bound (B.11b) of Proposition B.3 applied to x + 2 = P x (y 2 ) readily gives DISPLAYFORM12 (B.27) thus proving (B.22a).

To complete our proof, note that (B.11b) with DISPLAYFORM13 where we used Young's inequality and (B.11a) in the second inequality.

The bound (B.22b) then follows by substituting (B.30) in (B.27).

We begin by recalling the definition of the mirror descent algorithm.

With notation as in the previous section, the algorithm is defined via the recursive scheme DISPLAYFORM0 where γ n is a variable step-size sequence andĝ n is the calculated value of the gradient vector g(X n ) at the n-th stage of the algorithm.

As we discussed in the main body of the paper, the gradient input sequenceĝ n of (MD) is assumed to satisfy the standard oracle assumptions a) Unbiasedness: DISPLAYFORM1 where F n represents the history (natural filtration) of the generating sequence X n up to stage n (inclusive).With this preliminaries at hand, our convergence proof for (MD) under strict coherence will hinge on the following results: Proposition C.1.

Suppose that (SP) is coherent and (MD) is run with a gradient oracle satisfying (3.6) and a variable step-size γ n such that DISPLAYFORM2 Proposition C.2.

Suppose that (SP) is strictly coherent and (MD) is run with a gradient oracle satisfying (3.6) and a step-size γ n such that DISPLAYFORM3 n < ∞. Then, with probability 1, there exists a (possibly random) solution x * of (SP) such that lim inf n→∞ D(x * , X n ) = 0.Proposition C.1 can be seen as a "dichotomy" result: it shows that the Bregman divergence is an asymptotic constant of motion, so (MD) either converges to a saddle-point x * (if D(x * ) = 0) or to some nonzero level set of the Bregman divergence (with respect to x * ).

In this way, Proposition C.1 rules out more complicated chaotic or aperiodic behaviors that may arise in general -for instance, as in the analysis of Palaiopanos et al. [2017] for the long-run behavior of the multiplicative weights algorithm in two-player games.

However, unless this limit value can be somehow predicted (or estimated) in advance, this result cannot be easily applied.

This is the main role of Proposition C.2: it shows that (MD) admits a subsequence converging to a solution of (SP) so, by (B.20), the limit of D(x * , X n ) must be zero.

Our first step is to prove Proposition C.2.

To do this, we first recall the following law of large numbers for L 2 martingales:Theorem (Hall & Heyde, 1980, Theorem 2.18) .

Let Y n = n k=1 ζ k be a martingale and T n a nondecreasing sequence such that lim n→∞ τ n = ∞. Then, DISPLAYFORM4 on the set DISPLAYFORM5

Proof of Proposition C.2.

We begin with the technical observation that the solution set X * of (SP) is closed -and hence, compact (cf.

Lemma A.2 in Appendix A).

Clearly, if X * = X , there is nothing to show; hence, without loss of generality, we may assume in what follows that X * X .Assume now ad absurdum that, with positive probability, the sequence X n generated by (MD) admits no limit points in X * .

Conditioning on this event, and given that X * is compact, there exists a (nonempty) compact set C ⊂ X such that C ∩ X * = ∅ and X n ∈ C for all sufficiently large n. Moreover, letting p be as in Definition 2.1, we have g(x), x − p >

0 whenever x ∈ C. Therefore, by the continuity of g and the compactness of X * and C, there exists some a > 0 such that DISPLAYFORM0 To proceed, let D n = D(p, X n ).

Then, by Proposition B.3, we have DISPLAYFORM1 where, in the last line, we set U n+1 =ĝ n − g(X n ), ξ n+1 = − U n+1 , X n − p , and we invoked the assumption that (SP) is coherent.

Hence, telescoping (C.3) yields the estimate DISPLAYFORM2 Subsequently, letting τ n = n k=1 γ k and using (C.2), we obtain DISPLAYFORM3 By the unbiasedness hypothesis of (3.6) for U n , we have DISPLAYFORM4 Moreover, since U n is bounded in L 2 and γ n is 2 summable (by assumption), it follows that DISPLAYFORM5 Therefore, by the law of large numbers for L 2 martingales stated above [Hall & Heyde, 1980, Theorem 2 .18], we conclude that τ −1 n n k=1 γ k ξ k+1 converges to 0 with probability 1.

Finally, for the last term of (C.4), let DISPLAYFORM6 i.e., S n is a submartingale with respect to F n .

Furthermore, by the law of total expectation, we also have DISPLAYFORM7 Hence, by Doob's submartingale convergence theorem [Hall & Heyde, 1980, Theorem 2 .5], we conclude that S n converges to some (almost surely finite) random variable S ∞ with ¾[S ∞ ]

< ∞, implying in turn that lim n→∞ S n+1 /τ n = 0 (a.s.).Applying all of the above, the estimate (C.4) gives D n+1 ≤ D 1 − aτ n /2 for sufficiently large n, so D(p, X n ) → −∞, a contradiction.

Going back to our original assumption, this shows that, at least one of the limit points of X n must lie in X * (a.s.), as claimed.

We now turn to the proof of Proposition C.1:Proof of Proposition C.1.

Let x * ∈ X * be a limit point of X n , as guaranteed by Proposition C.2, and let D n = D(x * , X n ).

Then, by Proposition B.3, we have DISPLAYFORM8 and hence, for large enough n: DISPLAYFORM9 where we used the ansatz that g(X n ), X n − x * ≤ 0 for sufficiently large n (to be proved below), and, as in the proof of Proposition C.2, we set U n+1 =ĝ n − g(X n ), ξ n+1 = − U n+1 , X n − x * .

Thus, conditioning on F n and taking expectations, we get DISPLAYFORM10 where we used the oracle assumptions (3.6) and the fact that X n is F n -measurable (by definition).

DISPLAYFORM11 i.e., R n is an F n -adapted supermartingale.

Since DISPLAYFORM12 i.e., R n is uniformly bounded in L 1 .

Thus, by Doob's convergence theorem for supermartingales [Hall & Heyde, 1980, Theorem 2.5] , it follows that R n converges (a.s.) to some finite random variable R ∞ with ¾[R ∞ ] < ∞. In turn, by inverting the definition of R n , this shows that D n converges (a.s.) to some random variable D(x * ) with ¾[D(x * )] < ∞, as claimed.

It remains to be shown that g(X n ), X n − x * ≥ 0 for sufficiently large n. By Definition 2.1, this amounts to showing that, for all large enough n, X n lies in a neighborhood U of x * such that (MVI) holds.

Since x * has been chosen so that lim inf D(x * , X n ) = 0, it follows that, for all ε > 0, there exists some n 0 such that ∞ n=n 0 γ 2 n < ε and X n 0 ∈ U. Hence, arguing in the same way as in the proof of Theorem 5.2 of Zhou et al.[2017a], we conclude that (X n ∈ U for all n ≥ n 0 ) = 1, implying in turn that g(X n ), X n − x * ≥ 0 for all n ≥ n 0 .

This proves our last claim and concludes our proof.

With all this at hand, we are finally in a position to prove our main result for (MD):Proof of Theorem 3.1(a).

Proposition C.2 shows that, with probability 1, there exists a (possibly random) solution x * of (SP) such that lim inf n→∞ X n − x * = 0 and, hence, lim inf n→∞ D(x * , X n ) = 0 (by Bregman reciprocity).

Since lim n→∞ D(x * , X n ) exists with probability 1 (by Proposition C.1), it follows that lim n→∞ D(x * , X n ) = lim inf n→∞ D(x * , X n ) = 0, i.e., X n converges to x * .We proceed with the negative result hinted at in the main body of the paper, namely the failure of (MD) to converge under null coherence:Proof of Theorem 3.1 (b) .

The evolution of the Bregman divergence under (MD) satisfies the identity DISPLAYFORM13 where, in the last line, we used the null coherence assumption g(x), x − x * = 0 for all x ∈ X .

Since D(X n , X n+1 ) ≥ 0, taking expecations above shows that D(x * , X n ) is nondecreasing, as claimed.

With Theorem 3.1 at hand, the proof of Corollary 3.2 is an immediate consequence of the fact that strictly convex-concave problems satisfy strict coherence (Proposition A.1).

As for Corollary 3.3, we provide below a more general result for two-player, zero-sum finite games.

To state it, let A i = {1, . . .

, A i }, i = 1, 2, be two finite sets of pure strategies, and let X i = ∆(A i ) denote the set of mixed strategies of player i. A finite, two-player zero-sum game is then defined by a matrix M ∈ A 1 ×A 2 so that the loss of Player 1 and the reward of Player 2 in the mixed strategy profile x = (x 1 , x 2 ) ∈ X are concurrently given by DISPLAYFORM14 Then, writing Γ ≡ Γ(A 1 , A 2 , M) for the resulting game, we have: Proposition C.3.

Let Γ be a two-player zero-sum game with an interior Nash equilibrium x * .

If X 1 x * and (MD) is run with exact gradient input (σ DISPLAYFORM15 DISPLAYFORM16 Remark.

Note that non-convergence does not require any summability assumptions on γ n .In words, Proposition C.3 states that (MD) does not converge in finite zero-sum games with a unique interior equilibrium and exact gradient input: instead, X n cycles at positive Bregman distance from the game's Nash equilibrium.

Heuristically, the reason for this behavior is that, for small γ → 0, the incremental step V γ (x) = P x (−γg(x)) − x of (MD) is essentially tangent to the level set of D(x * , ·) that passes through x.4 For finite γ > 0, things are even worse because V γ (x) points noticeably away from x, i.e., towards higher level sets of D. As a result, the "best-case scenario" for (MD) is to orbit x * (when γ → 0); in practice, for finite γ, the algorithm takes small outward steps throughout its runtime, eventually converging to some limit cycle farther away from x * .We make this intuition precise below (for a schematic illustration, see also Fig. 1

Proof of Proposition C.3.

Write v 1 (x) = −Mx 2 and v 2 (x) = x 1 M for the players' payoff vectors under the mixed strategy profile x = (x 1 , x 2 ).

By construction, we have g(x) = −(v 1 (x), v 2 (x)).

Furthermore, since x * is an interior equilibrium of f , elementary game-theoretic considerations show that v 1 (x * ) and v 2 (x * ) are both proportional to the constant vector of ones.

We thus get g(x), x − x * = v 1 (x), x 1 −

x * 1 + v 2 (x), x 2 − x * 2 = −x 1 Mx 2 + (x * 1 ) Mx 2 + x 1 Mx 2 − x 1 Mx * 2 = 0, (C.16) where, in the last line, we used the fact that x * is interior.

This shows that f satisfies null coherence, so our claim follows from Theorem 3.1(b).For our second claim, arguing as above and using (B.11c), we get D(x * , X n+1 ) ≤ D(x * , X n ) + γ n g(X n ), X n − x * + γ where we used the fact that g is L-Lipschitz and that p is a solution of (SP) such that (MVI) holds for all x ∈ X .We are now finally in a position to prove Theorem 4.1 (reproduced below for convenience): Theorem.

Suppose that (SP) is coherent and g is L-Lipschitz continuous.

If (OMD) is run with exact gradient input and a step-size sequence γ n such that 0 < lim n→∞ γ n ≤ sup n γ n < K/L, (D.3) the sequence X n converges monotonically to a solution x * of (SP), i.e., D(x * , X n ) is non-increasing and converges to 0.Proof.

Let p be a solution of (SP) such that (MVI) holds for all x ∈ X (that such a solution exists is a consequence of Definition 2.1).

Then, by the stated assumptions for γ n , Lemma D.1 yields DISPLAYFORM0 where α ∈ (0, 1) is such that γ 2 n < αK/L for all n (that such an α exists is a consequence of the assumption that sup n γ n < K/L).

Now, telescoping (D.1), we obtain 5) and hence: DISPLAYFORM1 DISPLAYFORM2 With sup n γ n < K/L, the above estimate readily yields ∞ n=1 X n+1/2 − X n 2 < ∞, which in turn implies that X n+1/2 − X n → 0 as n → ∞.By the compactness of X , we further infer that X n admits an accumulation point x In this section we present the results of our image experiments using OMD training techniques.

Inception and FID scores obtained by our model during training were reported in FIG4 : as can be seen there, the extra-gradient add-on improves the performance of GAN training and efficiently stabilizes the model; without the extra-gradient step, performance tends to drop noticeably after approximately 100k steps.

For ease of comparison, we provide below a collection of samples generated by Adam and optimistic Adam in the CelebA and CIFAR-10 datasets.

Especially in the case of CelebA, the generated samples are consistently more representative and faithful to the target data distribution.

For the reproducibility of our experiments, we provide TAB2 the network architectures and the hyperparameters of the GANs that we used.

The architecture employed is a standard DCGAN architecture with a 5-layer generator with batchnorm, and an 8-layer discriminator.

The generated samples were 32×32×3 RGB images.

<|TLDR|>

@highlight

We show how the inclusion of an extra-gradient step in first-order GAN training methods can improve stability and lead to improved convergence results.