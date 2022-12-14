Policy gradients methods often achieve better performance when the change in policy is limited to a small Kullback-Leibler divergence.

We derive policy gradients where the change in policy is limited to a small Wasserstein distance (or trust region).

This is done in the discrete and continuous multi-armed bandit settings with entropy regularisation.

We show that in the small steps limit with respect to the Wasserstein distance $W_2$, policy dynamics are governed by the heat equation, following the Jordan-Kinderlehrer-Otto result.

This means that policies undergo diffusion and advection, concentrating near actions with high reward.

This helps elucidate the nature of convergence in the probability matching setup, and provides justification for empirical practices such as Gaussian policy priors and additive gradient noise.

Deep reinforcement learning algorithms have enjoyed tremendous practical success at scale BID17 BID0 .

Separately, theoretical and practical success through smoothing has also been achieved by generative adversarial networks with the introduction of Wasserstein GANs BID2 .

In both instances, a smooth relaxation of the original problem has been key to further theoretical understanding.

In this work, we take the view of policy gradients iteration through the lens of converging towards a function of the rewards field r(s, a) for a given state s.

This view uses optimal transport metrized by the second Wasserstein distance rather than the standard KullbackLeibler divergence.

Simultaneously, gradient flows relax and generalize to continuous time the notion of gradient steps.

An important mathematical result due to BID14 shows that in that setting, continuous control policy transport is smooth; this achieved by the heat flow following the Fokker-Planck equation, which also admits a stochastic diffusion representation, and sheds light on qualitative convergence towards the optimal policy.

This is to our knowledge the first time that the connection between variational optimal transport and reinforcement learning is made.

Policy gradient methods BID28 ; BID18 look to directly maximize the functional of expected reward under a certain policy ??.

??(a|s) is the probability of taking action a in state s under policy ??.

A policy can hence be identified to a probability measure ?? ??? P, the space of all policies.

In what follows, functionals are applications from P ??? R. Out of a desire for simplification, we focus on formal derivations, and skip over regularity and integrability questions.

We investigate policy gradients with entropy regularisation in the following setting:??? Bandits, or reinforcement learning with 1-step returns??? Continuous action space

It is already known BID22 that entropic regularization of policy gradients leads to a limit energy-based policy that probabilistically matches the rewards distribution.

We investigate the dynamics of that convergence.

Our contributions are as follows:1.

We interpret the mathematical concept of gradient flow as a continuous-time version of policy iteration.2.

We show that the choice of a Wasserstein-2 trust region in such a setting leads to solving the Fokker-Planck equation in (infinite dimensional) policy space, leading to the concept of policy transport.

This shows optimal policies are arrived at via diffusion and advection.

This also justifies empirical practices such as adding Gaussian noise to gradients.

Let r(a) be the reward obtained by taking action a. The expected reward with respect to a policy ?? is: DISPLAYFORM0 Shannon entropy is often added as a regularization term to improve exploration and avoid early convergence to suboptimal policies.

This gives us the entropy-regularised reward, which is a free energy functional, named by analogy with a similar quantity in statistical mechanics 1 : DISPLAYFORM1

We are interested in the process of policy iteration, that is, finding a sequence of policies (?? n ) converging towards the optimal policy ?? * .

In this section we follow closely the exposition by BID23 .

Policy iteration is often implemented using gradient ascent according to DISPLAYFORM0 Rearranging gives the explict Euler method DISPLAYFORM1 In this article we are more interested in the implicit Euler method which simply replaces ???J(?? DISPLAYFORM2 This is a policy iteration method.

If integrated and interpreted as an L 2 regularized iterative problem, it is strictly equivalent to finding a solution to the proximal problem: DISPLAYFORM3 Rather than just the L 2 distance between policies for constraining and regularization, one can envision the more general case of any policy distance d: DISPLAYFORM4 DISPLAYFORM5 A gradient flow is obtained in the small step limit ?? ??? 0.

This proximal mapping is an example of Moreau envelope, and remains in a convex optimization setting when J is convex.

The Wasserstein distance W 2 is defined for pairs of measures (??, ??) seen as marginals of a coupling ?? by: DISPLAYFORM6 The optimal coupling ?? * in the infimum above is also called the optimal transport plan.

Also note that later, we will reformulate this Monge- Kantorovich problem Kantorovich. (1942) as an equivalent linear problem of inner product minimization in L 2 (P 2 ): DISPLAYFORM7 DISPLAYFORM8 ??(0) = ?? 0 (12) which describes a gradient flow.

Just like gradient flows are the continuous-time analogue of discrete gradient descent steps, the Euler continuity equation is the continuous-time analogue of the discrete Euler methods seen earlier.

A classic result in Wasserstein space analysis is that because optimal transport acts on probability measures, it must satisfy conservation of mass.

Hence all absolutely continuous curves, or flows of measures, (?? t ) in W 2 (P) are solutions of the Euler continuity equation.

The Euler continuity equation can be seen as the formal continuous-time limit of the Euler implicit method described above DISPLAYFORM9 where v t is a suitable vector velocity field (????? is the divergence operator).

In the case we are looking at: DISPLAYFORM10 where ??J ???? (??) is the first variation density defined via the Gateaux derivative as ??J ???? DISPLAYFORM11 for every perturbation ?? = ?? ??? ??.

Substituting v t into 13 gives us a partial differential equation necessarily of the form: DISPLAYFORM12 This is proven rigorously in BID14 .

It remains to compute the first variation density ??J ???? (??) for entropy regularised policy gradients.

First, the linear part K r , or potential energy has first variation given naturally by DISPLAYFORM0 Second, the entropy part H is a special case H = U t log t of the general internal energy density functional: DISPLAYFORM1 In the case of entropy, f (t) = t log t, f (t) = 1 + log t, and therefore ??H ???? DISPLAYFORM2 Finally we require the gradient of this first variation density, given by: DISPLAYFORM3 The gradients ??/???? are functional, whereas the gradients ??? are action-gradients ??? a with respect to actions a ??? A.Plugging this into 16 gives us the partial differential equation associated with steepest descent within W 2 for entropy-regularized rewards: DISPLAYFORM4

The entropy-regularized rewards J is convex in the policy ??, which means there is a single optimal policy.

The optimal policy will be achieved as long as each step improves the policy and this will be the case as long as the steps taken as are small enough.

Given that we converge to the optimal policy, we know that at optimality ??? t ?? = 0.

Using equation 16 then gives us a necessary condition for the optimal policy ?? * : DISPLAYFORM0 By setting ??J ???? (??) = 0 and substituting in 20 we get DISPLAYFORM1 which gives us the optimal policy DISPLAYFORM2 This is the Gibbs measure of the rewards density per action -also seen as an energy-based policy, in line with the static result of BID22 .

The unregularized case ?? = 0 appears degenerate.

The gradient flow associated with the pure entropy functional ??H is the heat equation DISPLAYFORM0 Here the Laplacian ??? is in action space.

This is one of the key messages of the derivations we have done in this part: the Wasserstein gradient flow turns the entropy into the Laplacian operator 2 .For the full, entropy-regularized reward J(??), there is an extra transport term generated by the rewards, and the PDE is therefore the Fokker-Planck equation.

This means that taking policy gradient ascent steps in W 2 (according to equation 8), is equivalent to solving the Fokker-Planck equation for the policy ?? with potential field equal to the gradient of rewards r. Alternately, we can say, the policy undergoes diffusion and advection -it concentrates around actions with high reward.

A partial differential equation for diffuse measures also admits a particle interpretation.

The result above can also be written, through Ito's lemma BID21 , as the stochastic diffusion version of the Fokker-Planck equation BID14 : DISPLAYFORM0 with ?? t a finite dimensional discretization of policy ?? t , and B t a Brownian motion of same dimensionality.

In that case, the density of solutions verifies equation 21 -formally, one replaces increments of the Brownian motion dB t by its infinitesimal generator, the Laplacian 1 2 ???. This stochastic differential equation can also be seen as a Cauchy problem of on-policy rewards maximization, this time with added isotropic Gaussian policy noise ??? 2??dB t .Discretizing again -but in the time variable rather than the action space -one writes, with an explicit method: DISPLAYFORM1 This is just noisy stochastic action-gradients ascent on the rewards field.

This shows a link between entropy-regularization and noisy gradients.

It also suggests a possible technique for implementation.

The key issue to overcome is to generate gradient noise in parameter space that is equivalent to isotropic Gaussian policy noise.

We note the Kullback-Leibler and Wasserstein-2 problems are related by the Talagrand pinequalities BID13 , which for p = 2 and under some conditions, ensure that for some constant C and given a suitable reference measure ??, ????? ??? P(X), W 2 2 (??, ??) ??? C ?? D KL (??, ??).

This justifies the square root in d = ??? D KL in the proximal mappings studied earlier (equation 8), but more would be beyond the scope of this article.

Since the first variation process DISPLAYFORM0 is known, and we derive it explicitly earlier, we can use a variational argument specific to W 2 (and invalid in W 1 !).

We admit BID23 ) that the solution of the minimization problem has to satisfy: DISPLAYFORM1 with ?? W2 a Kantorovich potential function for transport cost c(x, y) = 1 2 |x ??? y| 2 .

One useful way to think of the Kantorovich potential is that it is a function, whose gradient field generates the difference between the optimal transport plan T and the identity, according to the equation T (x) = x ??? ?????(x).It is well known BID21 that the Gibbs distribution is the invariant measure of the stochastic differential equation above.

Therefore we expect it to play a role of primary importance.

In fact, the solution of the W 2 gradient flow with discrete steps is known explicitly Santambrogio.

FORMULA0 if we know the successive Kantorovich transport potentials associated: DISPLAYFORM2 In practice, deriving the W 2 optimal transport as well as its cost at each gradient flow step is numerically instable and computationally expensive.

Furthermore, numerical estimators for the gradients of Wasserstein distances have been found to be biased; alternatives such as the Cramer distance behave better in practice but not in theory BID4 . (To our knowledge, the gradient flow of the Cramer distance is not known, and no results exist that relate it to the entropy and Fisher information functionals).

In appendix, we use fast approximate algorithms in their small parameter regime.

We show that another Gibbs measure, the two-dimensional kernel e ???c/ , where c is the ground transport cost, and an auxiliary regularization strength, arises naturally in this context, and leads to taking successive Kullback-Leibler steepest descent steps in the coupling, in a spirit close to trust region policy optimization, but using optimal transport and the Sinkhorn algorithm.

Optimal transport, and the study of Wasserstein distances, is a very active research field.

Foundational developments are found in Villani's reference opus BID27 .

The gradient flows perspective is presented in Ambrosio's book BID1 for a complete theoretical treatment, and in BID23 for a more applied view including a presentation of the Jordan-Kinderlehrer-Otto result.

A classic reference for connecting Brownian motions and partial differential equations is BID21 .

Efficient algorithms for regularized optimal transport were first explored by BID10 , and then BID20 who showed the equivalence to steepest descent of KL with respect to the smoothed Gibbs ground cost, and its formulation as a convex problem.

BID7 gives proofs of ??-convergence of W 2, to W 2 .

BID16 In the context of neural networks, partial differential equations and convex analysis methods are covered by BID8 .

The Monge-Kantorovich duality in the W 1 case, and Wasserstein representation gradients, are applied to generative adversarial networks by Arjovsky BID2 .

The W 2 connection with generative models is studied by BID5 .

Similarly, Genevay et al BID12 define Minimum Kantorovich Estimators in order to formulate a wide array of machine learning problems in a Wasserstein framework.

We have used tools of quadratic optimal transport in order to provide a theoretical framework for entropy-regularized reinforcement learning, under the strongly restrictive assumption of maximising one-step returns.

There, we equate policy gradient ascent in Wasserstein trust regions with the heat equation using the JKO result.

We show advection and diffusion of policies towards the optimal policy.

This optimal policy is the Gibbs measure of rewards, and is also the stationary distribution of the heat PDE.

Recast as a stochastic Brownian diffusion, this helps explain recent methods used empirically by practitioners -in particular it sheds some light on the success of noisy gradient methods.

It also provides a speculative mechanism besides the central limit theorem for why Gaussian distributions seem to arise in practice in distributional reinforcement learning BID3 .Our contribution largely consists in highlighting the connection between the functional of reinforcement learning and these mathematical methods inspired by statistical thermodynamics, in particular the Jordan-Kinderlehrer-Otto result.

While we have aimed to keep proofs in this paper as simple and intuitive as possible, an extension to the n-step returns (multi-step) case is the most urgent and obvious line of further research.

Finally, exploring efficient numerical methods for heat equation flows compatible with function approximation, are directions that will also be considered in future research.

Here we show that while taking discrete proximal steps according to distance d = W 2 is ill-advised, we can perform optimal transport with respect to the entropic regularization of the Wasserstein distance itself.

This is a second layer of entropic regularization.

If we let be a small positive real number, then we define W 2 2, as a regularization of the W 2 minimization in equation 9: DISPLAYFORM0 with the entropy H(??) now extended to two-dimensional coupling space ??(??, ??) as, in the discretized case, with cardA = q, DISPLAYFORM1 and by an analogue of continuity named ??-convergence Carlier et al. FORMULA0 , W DISPLAYFORM2 This change from a linear to a convex problem makes the solution set better conditioned numerically; the solution does not have to lie on a vertex of a convex polytope (by analogy with the simplex algorithm, see BID19 ) anymore, and therefore, is more robust to initial conditions.

In practice, cannot be chosen too small or these stability properties are lost.

A certain amount of smoothing is to be tolerated, which is acceptable in the reinforcement learning context, due to the inherent uncertainty on the rewards distribution.

Returning to our optimization problem, moving the inner product bracket inside the H part turns the expression into a single KL divergence.

This yields the equivalent problem, as detailed in Peyr?? (2015) DISPLAYFORM3 At this stage, the link with earlier sections becomes intuitively very clear, since the reference measure .

It is therefore not surprising the evolution gradient flows considered earlier were linked to the heat equation.

Performing JKO stepping from from DISPLAYFORM4 instead of equation 8.

Combining both definitions therefore gives the problem DISPLAYFORM5 to be solved in 2-d coupling space.

With this entropic smoothing, we can now re-cast the optimal transport problem as a Kullback-Leibler problem, trading a single optimal transport proximal step for several, 'fast' KL steps.

This is done next section using iterative convex projection algorithms.

This algorithm is used in convex optimization for iterative projections.

The method generalizes the computation of the projection on the intersection of convex sets.

Assume we give ourselves a convex function ?? and that we consider the associated Bregman divergence D ?? Amari.

FORMULA0 defined by DISPLAYFORM0 We look to minimize this Bregman divergence D ?? on the intersection of convex sets C = ??? C i .In our case of interest there are two such sets C 1 and C 2 .

For y a given point, or function, we solve for DISPLAYFORM1 or equivalently with ?? 1 and ?? 2 (??) playing the role of indicator barrier functions DISPLAYFORM2 In the case where ?? = H, we get ????? = log, ????? * = exp through Legendre transform gradient bijection, and DISPLAYFORM3 The Bregman algorithm Bregman. (1967) simply consists in solving the problem 36 by iteratively performing projection on each of the sets C i in a cyclical manner, therefore building the sequence DISPLAYFORM4 with [n] the modulo operator ensuring cyclicality of the projections.

Therefore, any problem that can be cast under the convex Bregman form 36 can be solved by taking many steepest descent steps.

We now proceed to explicit the P D?? C [n] operators, which in our case are P KL C [n] KL proximal steps, and integrate them into an efficient practical algorithm.

We start with the need to minimize the convex form 1 ij p ij log p ij + p ij c ij , subject to marginal constraints that the discretized measure ?? is transported by p onto ??.

From a matrix perspective this translates into the two following constraints: that the sum in column of P being equal to vector ??, and the sum across lines of P equal to ??: DISPLAYFORM0 The matrix set U (??, ??) is convex.

We can form the Lagrangian of this optimization problem in p ij using vector Lagrangian multipliers ??, ??, DISPLAYFORM1 A necessary condition for optimality is then DISPLAYFORM2 Hence we have shown that the optimal coupling is a diagonal scaling of the ground cost's Gibbs kernel.

With the two positive vectors u, v defined as diag(u) = e DISPLAYFORM3 as min DISPLAYFORM4 then this reads as a two-dimensional KL projection algorithm of point?? on set of marginal constraints C 1,?? enforcing u (Kv) = ??, and DISPLAYFORM5 and ultimately yields the Sinkhorn balancing algorithm: DISPLAYFORM6 these two updates being merged in Algorithm 1.

The Sinkhorn algorithm is a fast, iterative algorithm for optimal transport; it mostly involves matrix multiplications and vector operations such as term-by-term division, and as such scales extremely well on GPU platforms.

This makes it possible to use the Sinkhorn algorithm to numerically approximate optimal couplings.

We give its outline below, the interested reader can find more details can be found in Cuturi's original article Cuturi.

FORMULA0 ; BID26 , or in Frogner's version applied to deep learning BID11 .We will want to compute the optimal coupling, transport cost, and gradient pertaining to distance W 2 2, .

First we remember the regularized transport problem as per equation 31.

The 2-dimensional, relaxed coupling ?? can be discretized to a 2-dimensional matrix P with entries (p i,j ).

We show (see Appendix) that necessarily DISPLAYFORM0 where the matrix exponential of the ground cost is taken term-by-term.

Recalling the equality constraints on the row and column sums given by ?? and ?? in 39, we find that we have to solve a matrix balancing problem, using the terminology of linear algebra.

Once we have formed K, and have policies ?? and ?? as inputs, we can run through iterations of the Sinkhorn algorithm till convergence to a fixed point.

This is done below and runs a one-line while loop on vector (x) i , the component-wise inverse of (u) i .

Dotted operations are taken component-wise:Algorithm 1 Computation of policy transport W c (??, ??) by Sinkhorn iteration.

Input C, , ??, ??.

I = (?? > 0); ??=??(I); c = c(I, :) ; K=exp(-C/ ) x=ones(length(??),size(??,2))/length(??); while x has not converged do x=diag(1./??)*K*(??.*(1./(K'*(1./x)))) end while u=1./x; v=??.*(1./(K'*u)) W c (??,??)=sum(u.*((K.*C)*v)) ???W c (??,??) ????? = log u (up to a constant parallel shift)The Sinkhorn algorithm converges linearly.

Its theoretical justification is that it can be seen as an instance of the iterative convex projections explained previous section.

It is critical to notice that the distance W 2, is differentiable in the policy, unlike W 2 .

The vector u above is not unique; but one suitable gradient of the Wasserstein distance with respect to the first variable policy is known, and given by the formula in the algorithm above, simply proportional to the element-wise log of scaling vector u. This closed form differentiation allows us to perform gradient descent in Sinkhorn layers embedded in neural network systems.

In general, this gradient, just like vector u, is defined up to a constant shift only; the normalizing shift generally found in the literature is DISPLAYFORM1 that makes u tangent to the simplex BID11 .

Under this form, the algorithm is compatible with function approximation, where policy ?? is a function of a parameter and reads ?? ?? .

We note that another possibility to create this compatibility would be to unroll a fixed number of iterations of the algorithm, as they are effectively matrix and vector operations, as has already been done with generative adversarial networks and deep Q-networks.

We hypothesize that learning with a Wasserstein loss, in a continuous action state setting, will help agents pick actions that are semantically close to the optimum action, therefore increasing policy quality, and reducing the 'unnaturalness' of policy mistakes.

It is our hope that a Wasserstein loss, by implying relevant semantic directions in action space, will speed up convergence and training of reinforcement learning agents.

In practice, we are still limited by our fundamental assumption that the MDP and the statewise rewards density a ??? r(s, a) are known.

Possibilities such as bootstrapping the rewards density distribution exist, and will be explored practically in further work.

@highlight

Linking Wasserstein-trust region entropic policy gradients, and the heat equation.

@highlight

The paper explores the connections between reinforcement learning and the theory of quadratic optimal transport

@highlight

The authors studied policy gradient with change of policies limited by a trust region of Wasserstein distance in the multi-armed bandit setting, showing that in the small steps limit, the policy dynamics are governed by the heat equation (Fokker-Planck equation).