Since their invention, generative adversarial networks (GANs) have become a popular approach for learning to model a distribution of real (unlabeled) data.

Convergence problems during training are overcome by Wasserstein GANs which minimize the distance between the model and the empirical distribution in terms of a different metric, but thereby introduce a Lipschitz constraint into the optimization problem.

A simple way to enforce the Lipschitz constraint on the class of functions, which can be modeled by the neural network, is weight clipping.

Augmenting the loss by a regularization term that penalizes the deviation of the gradient norm of the critic (as a function of the network's input) from one, was proposed as an alternative that improves training.

We present theoretical arguments why using a weaker regularization term enforcing the Lipschitz constraint is preferable.

These arguments are supported by experimental results on several data sets.

General adversarial networks (GANs) BID8 are a class of generative models that have recently gained a lot of attention.

They are based on the idea of defining a game between two competing neural networks (NNs): a generator and a classifier (or discriminator).

While the classifier aims at distinguishing generated from real data, the generator tries to generate samples which the classifier can not distinguish from the ones from the empirical distribution.

Realizing the potential behind this new approach to generative models, more recent contributions focused on the stabilization of training, including ensemble methods BID18 , improved network structure BID14 BID15 and theoretical improvements BID13 BID15 that helped to successfully model complex distributions using GANs.

It was proposed by to train generator and discriminator networks by minimizing the Wasserstein-1 distance, a distance with properties superior to the Jensen-Shannon distance (used in the original GAN) in terms of convergence.

Accordingly, this version of GAN was called Wasserstein GAN (WGAN) .

The change of metric introduces a new minimization problem, which requires the discriminator function to lie in the space of 1-Lipschitz functions.

In the same paper, the Lipschitz constraint was guaranteed by performing weight clipping, i.e., by constraining the parameters of the discriminator NN to be smaller than a given value in magnitude.

An improved training strategy was proposed by BID9 based on results from optimal transport theory (see BID19 .

Here, instead of clipping weights, the loss gets augmented by a regularization term that penalizes any deviation of the norm of the gradient of the critic function (with respect to its input) from one.

We review these results and present both theoretical considerations and empirical results, leading to the proposal of a less restrictive regularization term for WGANs.1 More precisely, our contributions are as follows:• We review the arguments that the regularization technique proposed by BID9 is based on and make the following two observations: (i) The regularization strategy requires training samples and generated samples to be drawn from a certain joint distribution.

In practice, however, samples are drawn independently from their marginals.(ii) The arguments further assume the discriminator to be differentiable.

We explain why both can be harmful for training.• We propose a less restrictive regularization term and present empirical results strongly supporting our theoretical considerations.

We will require the notion of a coupling of two probability distributions.

Although a coupling can be defined more generally, we state the definition in the setting of our interest, i.e., we consider all spaces involved to equal R n .Definition 1. Let µ and ν be two probability distributions on R n .

A coupling π of µ and ν is a probability distribution on R n × R n such that π(A, R n ) = µ(A) and π(R n , A) = ν(A) for all measurable sets A ⊆ R n .

The set of all couplings of µ and ν is denoted by Π(µ, ν).The following theorem plays a central role in the theory of optimal transport (OT) and is known as the Kantorovich duality.

Note, that the presented theorem is a less general, but to our needs adapted version of Theorem 5.10 from BID19 .

2 A proof of how to derive our version from the referenced one can be found in Appendix C.1.

We will denote by Lip 1 the set of all 1-Lipschitz functions, i.e., the set of all functions f such that f (y) − f (x) ≤ ||x − y|| 2 for all x, y. Theorem 1 (Kantorovich).

Let µ and ν be two probability distributions on R n such that R n ||x|| 2 dµ(x) < ∞ and R n ||x|| 2 dν(x) < ∞. Then (i) min π∈Π(µ,ν) R n ×R n ||x − y|| 2 dπ(x, y) = max DISPLAYFORM0 In particular, both minimum and maximum exist.(ii) The following two statements are equivalent:(a) π * is an optimal coupling (minimizing the value on the left hand side of (1)).(b) Any optimal function f * ∈ Lip 1 (at which the maximum is attained for the right hand side of (1)) satisfies that for all (x, y) in the support of π * : f * (x) − f * (y) = ||x − y|| 2 .The field of OT offers several approaches to the computation of optimal couplings.

To speed up the computation of an optimal coupling, BID5 introduced a regularized version of the primal problem in which an entropic term E(π) is added leading to the minimization of R n ×R n ||x − y|| 2 dπ(x, y) + E(π), with regularization parameter .

Regularized OT was generalized by BID6 to a more general class of regularization terms Ω(π).

As we will discuss in Section 5, the learning algorithm we propose in this paper has connections to the approach using Ω(π) = dπ(x,y) dµ(x) dν(y) 2 dµ(x) dν(y).

By BID3 , this leads to the dual problem given by DISPLAYFORM1 3 WASSERSTEIN GANS Formally, given an empirical distribution µ, a class of generative distributions ν over some space X , and a class of discriminators d : X → [0, 1], GAN training BID8 aims at solving the optimization problem given by DISPLAYFORM2 3 In practice, the parameters of the generator and the discriminator networks are updated in an alternating fashion based on (several steps) of stochastic gradient descent.

The discriminator thereby tries to assign a value close to zero to generated data points and values close to one to real data points.

As an opposing agent, the generator aims to produce data where the discriminator expects to see real data.

Theorem 1 by BID8 shows that, if the optimal discriminator is found in each iteration, minimization of the resulting loss function of the generator leads to minimization of the Jensen-Shannon (JS) divergence.

Instead of minimizing the JS divergence, proposed to minimize the Wasserstein-1 distance, also known as Earth-Mover (EM) distance, which is defined for any Polish space (M, c) and probability distributions µ and ν on M by DISPLAYFORM3 From the Kantorovich duality (see Theorem 1, (i)) it follows that, in the special case we are considering, the infimum is attained and, instead of computing this minimum in Equation (3), the Wasserstein-1 distance can also be computed as DISPLAYFORM4 where the maximum is taken over the set of all 1-Lipschitz functions Lip 1 .Thus, the WGAN objective is to solve DISPLAYFORM5 which can be achieved by alternating gradient descent updates for the generating network ν and the 1-Lipschitz function f (also modeled by a NN), just as in the case of the original GAN.

The objective of the generator is still to generate real-looking data points and is led by function values of f that plays the role of an appraiser (or critic).

The appraiser's goal is to assign a value of confidence to each data point, which is as low as possible on generated data points and as high as possible on real data.

The confidence value it can assign is bounded by a constraint of similarity, where similarity is measured by the distance of data points.

This can be motivated by the idea that similar points should have similar values of confidence for being real.

The new role of the critic helps to solve convergence problems, but the interpretation of its value as classifying real (close to 1) and fake data (close to 0) is lost.

We refer to Appendix A for a detailed discussion.

Modeling the WGAN critic function by a NN raises the question on how to enforce the 1-Lipschitz constraint of the objective in Equation (5).

As proposed by a simple way to restrict the class of functions f that can be modeled by the NN to α-Lipschitz continuous functions (for some α) is to perform weight clipping, i.e. to enforce the parameters of the network not to exceed a certain value c max > 0 in absolute value.

As the authors note, this is not a good but simple choice.

We further demonstrate this in Appendix B by proving (for a standard NN architecture) that, using weight clipping, the optimal function is in general not contained in the class of functions modeled by the network.

Recently, an alternative to weight clipping was proposed by BID9 .

The basic idea is to augment the WGAN loss by a regularization term that penalizes the deviation of the gradient norm of the critic with respect to its input from one (leading to a variant referred to as WGAN-GP, where GP stands for gradient penalty).

More precisely, the loss of the critic to be minimized is then given by DISPLAYFORM0 where τ is the distribution ofx = tx + (1 − t)y for t ∼ U [0, 1] and x ∼ µ, y ∼ ν being a real and a generated sample, respectively.

The regularization term is derived based on the following result.

Proposition 1.

Let µ and ν be two probability distributions on R n .

Let f * be an optical critic, leading to the maximum max f ∈Lip1 R n f (x) dµ(x) − R n f (x) dν(x) , and let π * be an optimal coupling with respect to min π∈Π(µ,ν) R n ×R n ||x − y|| 2 dπ(x, y) .

If f * is differentiable and x t = tx + (1 − t)y for 0 ≤ t ≤ 1, it holds that P (x,y)∼π * (∇f * (x t ) = y−xt ||y−xt|| ) = 1 .

This in particular implies, that the norms of the gradients are one π * -almost surely on such points x t .For the convenience of the reader, we provide a simple argument for obtaining this result in Appendix C.2.Note, that Proposition 1 holds only when f * is differentiable and x and y are sampled from the optimal coupling π * .

However, sampling independently from the marginal distributions µ and ν very likely results in points (x, y) that lie outside the support of π * .

Furthermore, the optimal cost function f * does not need not to be differentiable everywhere.

These two points will be discussed in more detail in the following subsections.

Observation 1.

Suppose f * ∈ Lip 1 is an optimal critic function and π * the optimal coupling determined by the Kantorovich duality in Theorem 1.

Then |f DISPLAYFORM0 * , but not necessarily on the lines connecting an arbitrary pair of a real and a generated data point, i.e. arbitrary x ∼ µ and y ∼ ν.

Consider the examples in FIG0 , where every X denotes a sample from the generator and every O a real data sample.

Optimal couplings π * are indicated in red, and values of an optimal critic function are indicated in blue (optimality is shown in Appendix A.1).

In the one-dimensional example on the left, the left-most X and the right-most O satisfy f DISPLAYFORM1 7 |O − X| = |O − X|, illustrating that the basis for the derivation of the condition, that the norm of the gradient equals one between generated and real points, only holds for points sampled from the optimal coupling.

Note, while here the gradient is still of norm 1 almost everywhere, this does not necessarily hold in higher dimensions, where not all points lie on a line between some pair of points sampled from π * .

This is exemplified for two dimensions on the right side of FIG0 , where blue numbers with a ∈ R denote the values of an optimal critic function at these points (the values at these points is all that matters).

Without loss of generality we can assume the value at position (1, 2) to be zero, taking into account that an optimal critic function remains optimal under addition of an arbitrary constant.

Since the Lipschitz constraint of f * must be satisfied, we get 1 − a ≤ √ 2 and a + 1 ≤ √ 2.

Therefore a ∈ [1 − √ 2, √ 2 − 1] and one of the inequalities of the Lipschitz constraint must be strict.

Observation 2.

The assumption of differentiability of the optimal critic is not valid at points of interest.

Consider the example of two discrete probability distributions and its optimal critic function f * shown on the left in FIG1 .

We can see that the indicated function f * (x) = 1 − |x| ∈ Lip 1 is optimal as it leads to an equality in the equation of the Kantorovich dual.

(Also, it is the only continuous function, up to a constant, that realizes f * (x) − f * (y) = |y − x| for coupled points (x, y).)

However, it is not differentiable at 0.

The counterexample can be made continuous by considering the points as the center points of Gaussians, as illustrated on the right in FIG1 .

This is formalized by the following proposition showing that the critic indicated in blue is indeed optimal for the depicted gray Gaussian of real data and the green mixture of two Gaussians of generated data.

Proposition 2.

Let µ = N (0, 1) be a normal distribution centered around zero and ν = ν −1 + ν 1 be a mixture of the two normal distributions ν −1 = 1 2 N (−1, 1) and ν 1 = 1 2 N (1, 1) over the real line.

If µ describes the distribution of real data and ν describes the distribution of the generative model, then an optimal critic function is given by φ * (x) = −|x|.The proof can be found in Appendix C.3.The issue with non-differentiability can be generalized to higher-dimensional spaces based on the observation that an optimal coupling is in general not deterministic.

Deterministic couplings are particularly nice in the sense that they allow a transport plan assigning each point x from one distribution deterministically to a point y of the other distribution, without having to split any masses (the search for deterministic optimal couplings is called the Monge problem).

However, in a lot of settings no deterministic coupling exists.

The notion of a deterministic coupling is formalized in the following definition.

Definition 2.

Let (X, µ) and (Y, ν) be two probability spaces.

A coupling π ∈ Π(µ, ν) is called deterministic if there is a measurable function ρ : DISPLAYFORM0 We can now formulate the following observation.

Observation 3.

Suppose π * is a non-deterministic optimal coupling between two probability distributions over R n so that there exist points (x, y) and (x, y ) in supp(π * ).

Suppose further that there is no λ > 0 with (y − x) = λ · (y − x) (in particular this implies y = y ).

Then any optimal critic function f * is not differentiable at x.

The arguments can be found in Appendix C.5.In practice, where the optimal critic is approximated by a NN, the situation is slightly different: A function modeled by an NN is (almost) everywhere differentiable (depending on the activation functions).

By the Stone-Weierstrass theorem, on compact sets, we can approximate any (Lipschitz-)continuous function by differentiable functions uniformly.

Nevertheless, it seems to be a strong constraint on an approximating function to have a gradient of norm one in the neighborhood of a non-differentiability (cf.

FIG1 (a)).

Therefore, we argue -in contrast to the argumentation of BID9 -that the gradient should not be assumed to equal one for arbitrary points on the line between an arbitrary real point x and a generated point y.

In the following, we will discuss how the regularization of WGANs can be improved.

Penalizing the violation of the Lipschitz constraint.

For the critic function, we have nothing more at hand than the inequality of the Lipschitz-constraint.

Moreover (as shown in Lemma 1 in the Appendix) the exhaustion of the Lipschitz constant is automatic by maximizing the objective function.

Therefore, a natural choice of regularization is to penalize the given constraint directly, i.e., sample two points x ∼ µ and y ∼ ν from the empirical and the generated distribution respectively and add the regularization term DISPLAYFORM0 to the cost function.

(We square to penalize larger deviations more than smaller ones.)

Note the similarity of the regularization term to the squared Hinge loss, which is also used to turn a hard constraint into a soft one in the optimization problem connected to support vector machines.

Alternatively, since the NN generates (almost everywhere) differentiable functions, we can penalize whenever gradient norms are strictly larger than one, an option referred to as "one-sided penalty" and shortly discussed as an alternative to penalizing any deviation from one by BID9 4 .

Note that enforcing the gradient to be smaller than one in norm has the advantage that we penalize when the partial derivative has norm > 1 into the direction of steepest descent.

Hence, all partial derivatives are implicitly enforced to be bounded in norm by one, too.

At the same time, enforcing ≤ 1 for the gradient of smooth approximating functions is not an unreasonable constraint even at points of non-differentiability.

For these reasons we suggest to add the regularization term max {0, ||∇f (x)|| − 1} 2 to the cost function.

Different ways of sampling the pointx are analyzed in Appendix D.4.

Thus, our proposed method (WGAN-LP, where LP stands for Lipschitz penalty) alternates between updating the discriminator to minimize DISPLAYFORM1 (where τ depends on the concrete sampling strategy chosen) and updating the generator network modeling ν to minimize −E y∼ν [f (y)] using gradient descent.

The connection to regularized optimal transport.

Consider Equation (2) of regularized OT.

For a hard constraint f (x) − g(y) ≤ ||x − y|| 2 , one can attain the supremum over DISPLAYFORM2 and subsequently maximize over one function only.

Taking the advantage of dealing with a single function as a motivation, one may similarly replace f = g in Equation 2, which uses a soft constraint (even though this can now only approximate the supremum).

This leads to an objective of minimizing DISPLAYFORM3 that, similarly to Equation FORMULA10 , softly penalizes whenever f (x) − f (y) > ||x − y|| 2 for a real sample x and a generated sample y.

It is noteworthy that to justify the replacement f = g one would require a high regularization parameter λ = 4 of the dual problem, which corresponds to a low regularization of the primal problem.

Dependence on the regularization hyperparameter λ.

Let L GP λ and L LP λ denote the infimums of the regularized losses over a class of (differentiable) critic functions f from Equation (6) (WGAN-GP) and Equation (8) (WGAN-LP) respectively.

For the comparison of these optimal losses we have the following result (proof in Appendix C.4).

DISPLAYFORM4 In particular, for small λ the optimal scores approximately agree.

On the other hand, increasing λ strengthens the soft constraints, which means that the theoretical observations from Section 4 become more pertinent with growing λ.

Our experiments show exactly the behavior that WGAN-LP and WGAN-GP perform very similarly for small λ, while WGAN-LP performs much better for larger values of λ and its performance is much less dependent on the choice of hyperparameter λ.

A more general view.

The Kantorovich duality theorem holds in a quite general setting.

For example, a different metric can be substituted for the Euclidean distance ||·|| 2 .

Taking ||·|| p 2 for a different natural number p for example leads to the minimization of the Wasserstein distance of order p (i.e., the Wasserstein-p distance).

Based on the dual problem to the computation of the Wasserstein distance of order p (as given by the Kantorovich duality theorem) we still need to maximize Equation (5) with the only difference that 1-Lipschitz-continuity is now measured with respect to ||·|| p 2 .

For our training method this entails that the only modification to make is to use the regularization term given by FORMULA10 , where the Euclidean distance is replaced by the metric of interest.

We provide experimental results for the Wasserstein-2 distance in Appendix D.5.Recently, by BID2 , the Wasserstein distance was replaced by the energy distance 5 .

For the training of Cramer GANs, the authors apply the GP-penalty term proposed by BID9 .

We expect that using the LP-penalty term instead is also beneficial for Cramer GANs.

We perform several experiments on three toy data sets, 8Gaussians, 25Gaussians, and Swiss Roll 6 , to compare the effect of different regularization terms.

More specifically, we compare the performance of WGAN-GP and WGAN-LP as described in Equations FORMULA6 and FORMULA11 respectively, where the penalty was applied to points randomly sampled on the line between the training sample x and the generated sample y. Other sampling methods are discussed in Appendix D.4.Both, the generator network and the critic network, are simple feed-forward NNs with three hidden Leaky ReLU layers, each containing 512 neurons, and one linear output layer.

The dimensionality of the latent variables of the generator network was set to two.

During training, 10 critic updates are performed for every generator update, except for the first 25 generator updates, where the critic is updated 100 times for each generator update in order to get closer to the optimal critic in the beginning of training.

Both networks were trained using RMSprop BID17 with learning rate 5 · 10 −5 and a batch size of 256.To see whether our findings on toy data sets can be transferred to real world settings, we trained bigger WGAN-GPs and WGAN-LPs on CIFAR-10 as it is described below.

Code for the reproduction of our results is available under https://github.com/lukovnikov/improved_wgan_ training .Level sets of the critic.

A qualitative way to evaluate the learned critic function for a twodimensional data set is by displaying its level sets, as it was done by BID9 and BID10 .

The level sets after 10, 50, 100 and 1000 training iterations of a WGAN trained with the GP and LP penalty on the Swiss Roll data set are shown in FIG2 .

Similar experimental results for the 8Gaussians and 25Gaussian data sets can be found in Appendix D.1.It becomes clear that with a penalty weight of λ = 10, which corresponds to the hyperparameter value suggested by BID9 , the WGAN-GP does neither learn a good critic function nor a good model of the data generating distribution.

With a smaller regularization parameter, λ = 1, learning is stabilized.

However, with the LP-penalty a good critic is learned even with a high penalty weight in only a few iterations and the level sets show higher regularity.

Training a WGAN-LP with lower penalty weight led to equivalent observations (results not shown).

We also experimented with much higher values for λ, which led to almost the same results as for λ = 10, which emphasizes that LP-penalty based training is less sensitive to the choice of λ.

Evolution of the critic loss.

To yield a fair comparison of methods applying different regularization terms, we display values of the critic's loss functions without the regularization term throughout training.

Results for WGAN-GPs and WGAN-LPs are shown in FIG3 .The optimization of the critic with the GP-penalty and λ = 5 is very unstable: the loss is oscillating heavily around 0.

When we use the LP-penalty instead, the critic's loss smoothly reduces to zero, which is what we expect when the generative distribution ν steadily converges to the empirical distribution µ. Also note that we would expect the negative of the critic's loss to be slightly positive, as a good critic function assigns higher values to real data points x ∼ µ and lower values to generated points y ∼ ν.

This is exactly what we observe when using the LP-penalty Interestingly, when using the LP-penalty in combination with a very high penalty weight, like λ = 100, we obtain the same results, indicating that the constraint is always fulfilled for λ = 10 already.

Using λ = 1 in combination with the GP-penalty on the other hand stabilized training but still results in fluctuations in the beginning of the training (results shown in Appendix D.2).

Estimating the Wasserstein distance.

In order to estimate how the actual Wasserstein distance between the real and generated distribution evolves during training, we compute the cost of minimum assignment based on Euclidean distance between sets of samples from the real and generated distributions, using the Kuhn-Munkres algorithm BID11 .

We use a sample set size of 500 to maintain reasonable computation time and estimate the distance every 10th iteration over the course of 500 iterations.

All experiments were repeated 10 times for different random seeds.

From the re- sults for WGAN-GP and WGAN-LP with λ = 5 shown in FIG4 , we conclude that the proposed LP-penalty leads to smaller estimated Wasserstein distance and less fluctuations during training.

When training WGAN-GPs with a regularization parameter of λ = 1, training is stabilized as well (see Appendix D.3), indicating that the effect of using a GP-penalty is highly dependent on the right choice of λ.

Sample quality on CIFAR-10.

We trained WGANs with the same ResNet generator and discriminator and the same hyperparameters as BID9 and computed the Inception score BID15 throughout training (plots can be found in Appendix D.6).

The maximal scores reached in 100000 training iterations with different regularization parameters are reported in Table 1 .

WGAN-LP reaches the similar or slightly better Inception score as WGAN-GP with small penalty weight (λ ≤ 10), while being more stable to other choices of this hyperparameter.

This is especially interesting in the light of a recent large scale study, which also reported a strong dependence of sample quality on λ for WGAN-GP (see, FIG6 and 9 in BID12 .

Another interesting observation can be made by monitoring the value of the regularization term during training, as in Figure 6 ), where contributions to the penalty from ||∇f (x)|| > 1 are shown in the upper and contributions ||∇f (x)|| < 1 (only existing for WGAN-GP) are shown in the lower half plane.

While the values of the one-sided regularization of WGAN-LP are only slightly larger for larger λ (100 compared to 5) the regularization of WGAN-GP shows a strong dependence on the choice of the regularization parameter.

For λ = 5 the penalty contributions from gradient norms smaller than one almost vanished (we found this getting even more severe for even smaller regularization parameters).

That is, in a setting where WGAN-GP is performing fine it actually acts similar to WGAN-LP.Related penalties We tested the effects of using the regularization terms given by Equation FORMULA10 and Equation FORMULA13 instead of the the proposed regularization given in Equation (8).

Both lead to good performance on toy data but to considerably worse results on CIFAR-10, where training was very unstable.

Results are shown in Appendix D.7 Figure 6 : Comparison of the magnitude of the gradient penalty during training on CIFAR, showing < 1 and > 1 contributions (i.e. min(0, ||∇f (x)|| − 1) 2 resp.

max(0, ||∇f (x)|| − 1) 2 ).

Left: for regularization parameter λ = 5.

Right: for regularization parameter λ = 100.

The (one-sided) gradient penalty of WGAN-LP is depicted in blue (solid), the gradient penalty of WGAN-GP in red (dashed).

All the values for every iteration (one mini-batch) are shown in light blue and red.

Dark blue and red lines show the mean over a sliding window of size 500.

The figure shows that the part of the gradient penalty of WGAN-GP penalizing a gradient ≤ 1 almost vanishes for a small regularization parameter, bringing it close to WGAN-LP.

For larger values of the regularization parameter, the total penalty of WGAN-GP and its contributing parts are larger than the penalty of WGAN-LP, however, the performance of WGAN-GP suffers more.

For stable training of Wasserstein GANs, we propose to use the following penalty term to enforce the Lipschitz constraint that appears in the objective function: DISPLAYFORM0 We presented theoretical and empirical evidence that this gradient penalty performs better than the previously considered approaches of clipping weights and of applying the stronger gradient penalty given by Ex ∼τ [(||∇f (x)|| 2 − 1) 2 ].

In addition to more stable learning behavior, the proposed regularization term leads to lower sensitivity to the value of the penalty weight λ (demonstrating smooth convergence and well-behaved critic scores throughout the whole training process for different values of λ).

The authors thank the anonymous reviewers for their valuable suggestions.

An issue of the original GAN discriminator was that it outputs zero every time it is certain to see generated data, independent on how far away a generated data point lies from the real distribution.

As a consequence, locally, there is no incentive for the generator to rather generate a value closer to (but still off) the real data; GAN critic's optimal value is zero in either case.

The WGAN's optimal critic function measures this distance which helps for the generated distribution to converge, but the interpretation of the absolute value as indicating real (close to 1) and fake data (close to 0) is lost.

And worse, there is even no guarantee that the relative values of the optimal critic function help to decide what is real and what is fake.

Although this does not seem to cause major problems for the iterative training procedure in practice, we still consider it worthwhile to give a specific example justifying the following observation.

Observation 4.

The WGAN generator could learn wrong things, basing its decision on the values of the optimal critic function, i.e., if it generates at locations of high critic function values.

Consider the following setting, where the X's represent generated and the O's represent real data points.

DISPLAYFORM0 an optimal f* Figure 7 : Values of the WGAN critic function for some generated data points can be higher than the critic's values for some real data points.

Thus, fake and real points can not be distinguished based on the critics values alone.

Real data points are represented by O, generated by X.An optimal coupling in this example is quite obvious: We connect the left-most O with the X on the left, and then extend by an arbitrary matching of the other O's with the other X's.

It is then not hard to verify that the indicated critic function with slope 1 or −1 almost everywhere leads to an equality in the Kantorovich duality and hence is optimal.

The value of the critic function at the left-most X is higher than the value at the right-most O, suggesting to generate images at the wrong position.

This issue might be fixed by the alternating updates of generator and critic at a later stage of training when less X's are generated so far on the right side of the O's.

The critic function will then flatten the peak, eventually assigning a lower value to an X on the left than to any of the O's.

Remark 1.

The same holds (with only a slight change of the critic function f ) if the X's and O's denote the centers of Gaussians.

This can be shown with similar arguments as those in the proofs in Appendix C.3.

We show here that the coupling and critic function indicated in FIG0 are indeed optimal.

In the one-dimensional example on the left, DISPLAYFORM0 and thus π * and f * are indeed optimal.

In the two-dimensional example on the right, the coupling indicated in red and the critic function (described by its function values in blue) are optimal, since with this choice we have DISPLAYFORM1 Equality of the left hand side and right hand side of the equation proves optimality on both sides.

The critic function of WGAN is given by a neural network, which raises the question on how to enforce the 1-Lipschitz constraint in the maximization problem of the objective in Equation (5).

As point out, it does not matter whether to maximize over 1-Lipschitz or α-Lipschitz continuous functions, since we can equivalently optimize α · W (µ, ν) instead of W (µ, ν).

An easy consideration leads to the following lemma.

Lemma 1.

The optimal critic function f * (leading to the maximum in Eq. (4)) exhausts the Lipschitz constraint for given α in the sense that there is a pair of points (x, y) such that f DISPLAYFORM0 * generates a contradiction to the optimality of f * . (Alternatively, in the case α = 1, it follows directly from Theorem 1, (ii), that the transport is optimal if and only if the Lipschitz constraint of one is exhausted for any two points of the coupling.)Observation 5.

Weight clipping is not a good strategy to enforce the Lipschitz constraint for the critic function.

First note that by clipping the weights we enforce a common Lipschitz constraint, where the common Lipschitz constantᾱ is defined as the minimal α ∈ R such that f (x)−f (y) ≤ α||x − y|| 2 for all x, y and all functions f that can be generated by the network under weight clipping.

The actual value ofᾱ does not follow directly from the weight clipping constant c max but can be computed from the structure of the network.

From Lemma 1 we know that an optimal f * exhausts the Lipschitz constraint.

We will now show exemplarily for deep NN with rectified linear unit (ReLU) activation functions that there is an extremely limited number of functions generated by the NN using weight clipping that do exhaust the implicitly given common Lipschitz constraintᾱ. It follows that, in almost all cases, the optimal f * is not in the class of functions that can be generated by the network under the weight clipping constraint.

Proof.

We need to determine every function f * generated by the neural network, such that we can find points x * = y * with f * (y DISPLAYFORM1 Recall thatᾱ is defined as the minimal α satisfying f (y) − f (x) ≤ α||x − y|| 2 for all functions f generated by the neural network and all points x, y.

In the following, we will denote by α(f ) the Lipschitz constant of f , i.e., the smallest α ∈ R such that f (x) − f (y) ≤ α||x − y|| 2 for all x, y.

Every function generated by the neural net is a composition of functions DISPLAYFORM2 with linear functions f i and relu denoting a layer of activation functions with rectifier linear units.

Since each linear function f i is Lipschitz continuous with Lipschitz constant α(f i ) and relu is Lipschitz continuous with α(relu) = 1, it follows that f is Lipschitz continuous with α(f ) ≤ i α(f i ).

Moreover, equality holds if there is a pair of points (x, y) such that the consecutive images witness the maximal Lipschitz constant α(f i ) and α(relu) for each of the individual functions making up the composition of f .

More formally, equality holds if and only if there is a tuple of pairs of points ( DISPLAYFORM3 ) and f i (y (i) ) are larger or equal to zero.

This is equivalent to the condition that DISPLAYFORM4 It follows that to determine f * we need to maximize α(f i ) for the linear layers with weight constraint c max and find a sequence of points (x (i) , y (i) ) that satisfy (i)-(iv).

The existence of the sequence of points shows that α(f DISPLAYFORM5 Since, as we will show, the conditions in (a) and (b) maximize the Lipschitz constraint of each layer individually, the existence of suitable (x (i) , y (i) ) proves the if-direction of the proposition.

For the only-if direction, we will see that the ability to find the sequence of points gives restrictions on how to maximize α(f i ) of an individual layer, leading to the more restrictive condition of (b) for all but the first layer (cf.

(a)).So let us first maximize the Lipschitz constraint of each linear layer and then make sure that we can find the corresponding points.

We write the linear layer as a matrix multiplication DISPLAYFORM6 and our goal can be reformulated to finding the matrix A (i) maximizing α(f i ).For any fixed z, ||A (i) z|| 2 is maximized exactly when each vector entry is maximized in absolute value.

Now, with DISPLAYFORM7 j,k ) j,k and sgn(·) denoting the sign function, DISPLAYFORM8 and equality holds if and only if A (i) or −A (i) consists of columns of constant entry with the value DISPLAYFORM9 with equality if and only if DISPLAYFORM10 for all k.

Hence, for the first linear layer we need to choose a matrix A (1) satisfying (a) of the statement of the proposition.

Now, we find a pair (x (1) , y (1) ) with DISPLAYFORM11 This is the only possibility to ensure (iii) and (iv) of the conditions above.

Note that also (i) holds for (x (1) , y (1) ), and (ii) (together with (iv)) determines (x (2) , y (2) ) uniquely from (x (1) , y (1) ) as DISPLAYFORM12 We may assume that ||x (1) || 1 > ||y (1) || 1 .

(Otherwise, switch the roles of x and y. In the case of equality, we need to choose a different pair for (x (1) , y (1) ) not to violate (i) for (x (2) , y (2) ).)

Then we have that DISPLAYFORM13 Using the same arguments as above, it follows that for such (x (2) , y (2) ), to maximize the Lipschitz constant of f 2 (and to guarantee that the maximum is reached at (x (2) , y (2) )), we need to have Aequal to a matrix with c max at each position.

Now (i)-(iv) also hold for the second layer and one may now proceed by induction to show that for i ≥ 2, A (i) contains only c max for each of its entries.

This is the only way to maximize the Lipschitz constraint for functions generated by the neural net, and it does indeed hold ||f DISPLAYFORM14 Proof.

We provide the arguments how to derive our version from Theorem 5.10 of BID19 .With c(x, y) = ||x − y|| 2 , our assumptions imply (with c X = c Y = ||·|| 2 ) that all conclusions of Theorem 5.10 (i) − (iii) hold.

Moreover, 5.4 of BID19 shows that in this case ψ = ψ c (in the notation of Villani FORMULA1 ) and c-convexity is the same as 1-Lipschitz continuity.

This leads to our formulation in (i) and the existence of an optimal coupling π * and an optimal critic function f * by part (iii).

then it follows from the proof of Theorem 5.10 that the set Γ in part 5.10 (iii) is given by Γ = f * ∈Lip1 optimal Γ f * , where f * being optimal means that it leads to a maximum on the RHS of equation FORMULA0 .To prove our part (ii) from 5.10, let π * be optimal.

Then, by 5.10 (iii), π * (Γ) = 1.

Hence, in particular, π * (Γ f * ) = 1 for all optimal f * ∈ Lip 1 .

This shows that (a) implies (b).

For the other direction, we use that if π * (Γ f * ) = 1 for all optimal f * , then π * (Γ) = 1, which by Theorem 5.10 (iii) is equivalent to π * being optimal.

Proof.

It follows from Theorem 1 (ii) that for all (x, y) in the support of π * we have |f * (y) − f * (x)| = ||x − y|| 2 .

Considering the line between x and y, the 1-Lipschitz constraint implies that the values of f * have to follow a linear function (since assuming that the slope was smaller than one at some point would imply that the differentiable function must have a slope larger than one somewhere else between x and y, which contradicts the 1-Lipschitz constraint).

It follows that at each point on the line, the partial derivative has norm equal to one into the direction pointing from the real data point x to the generated one y (which are coupled by the corresponding optimal coupling).

Since, by the 1-Lipschitz constraint, the maximal norm of a partial derivative at any point into any direction is one, the given direction is the direction of maximal descent, i.e. equals the gradient.

To prove Proposition 2, we first prove that φ * (x) = −|x| is the optimal critic function for certain distributions with non-overlapping support, and then reduce the example with Gaussian functions to this simplified setting.

Proposition 5.

Let f and g be two continuous functions on the real line that satisfy the following conditions:• f and g are symmetric with respect to the y-axis.• f (x) ≥ 0 and g(x) ≥ 0 for all x.• If supp DISPLAYFORM0 • f has connected support (this implies that f is centered around 0 because of the symmetry).• DISPLAYFORM1 Then the maximum of DISPLAYFORM2 Proof.

Before going into the technical details, we wish to point out the simple idea of the proof, which is to transport the left/right half of the distribution given by g to the left/right half of the distribution given by f respectively.

We first multiply both f and g by a constant number c such that DISPLAYFORM3 Then c · f and c · g define probability density functions.

A function φ ∈ Lip 1 maximizes DISPLAYFORM4 )dx if and only if it maximizes R φ(x)(f (x) − g(x))dx.

We therefore may assume from now on that DISPLAYFORM5 Now it suffices to find a coupling π of the probability distributions defined by f and g (that is itself defined by a probability density function π : R × R → R) such that for φ(x) = −|x| we get DISPLAYFORM6 The proof then follows from the Kantorovich duality theorem 1, because the right hand side is always smaller or equal to the left hand side for arbitrary coupling π and function φ ∈ Lip 1 and is consequently maximized when equality holds.

By the assumption of symmetry, we may write g = g 1 + g 2 where the support supp(g 1 ) ⊆ {x | x < 0} and g 2 (x) = g 1 (−x) for all x. The area under g 1 (x) equals half the area under f (x), or put differently, DISPLAYFORM7 We now consider the probability density function π 1 : R × R → R given by DISPLAYFORM8 which defines a coupling between the two distributions given by the probability density functions 2g 1 and 2f · δ (−∞,0] .

For later use we note that DISPLAYFORM9 We define π 2 (x, y) = π 1 (−x, −y) for y = 0 and π 2 (x, 0) = 0.

Further, we let π = 1 2 π 1 + 1 2 π 2 .

Then π defines a coupling between g and f as can be seen by computing DISPLAYFORM10 and y∈R π(x, y)dy = 1 2 y∈R π 1 (x, y)dy + 1 2 y∈R π 2 (x, y)dy DISPLAYFORM11 We have established the existence of some coupling between f and g and we will now compute its transport costs.

We will subsequently show that this equals DISPLAYFORM12 )dx, hence both π and φ are optimal by realizing the Kantorovich duality.

We aim at showing DISPLAYFORM13 The latter equation holds because for (x, y) in the support of π 1 we have x ≤ y. (To see this, note that support of π 1 is a subset of the support of DISPLAYFORM14 , and DISPLAYFORM15

Published as a conference paper at ICLR 2018 DISPLAYFORM0 We are now able to proof Proposition 2Proof to Proposition 2.

Let f denote the probability density function of N (0, 1) and g = 1 2 g −1 + 1 2 g 1 denote the sum of half the probability density functions g −1 of N (−1, 1) and g 1 of N (1, 1).

DISPLAYFORM1 i.e.f andg are the positive and the negative part of (f − g).

Thenf andg satisfy the hypothesis of Proposition 5 and the maximum DISPLAYFORM2 is obtained for φ * (x) = −|x|.

Proof.

For any fixed function f and λ > 0, the two regularized losses of the critic function f are of the form DISPLAYFORM0 for some real number c , a function h with with h(z) ≥ 0 for all z and a probability distribution τ .

Since for any real number 0 ≤ a we have that DISPLAYFORM1 Therefore the inequalities also hold for the infimum over a class of functions, hence DISPLAYFORM2

For the coupled pairs (x, y) and (x, y ) we have that the partial derivatives at x into the directions of y and y respectively have an absolute value of one.

If there are two such directions and f * is differentiable, then the norm of its gradient must be larger than one, contradicting the 1-Lipschitz constraint.

Indeed, recall that, considering f as a function on the line {x + λ · v | λ ∈ R} with v of unit length, the slope of f at x is given by ∇f ( DISPLAYFORM0 with θ v being the angle between the vector ∇f (x) and the unit vector v. Equation FORMULA0

We analyzed the effect of the GP-and the LP-penalty using different sampling procedures.

In particular, we compared the sampling procedure proposed by BID9 with variants, which generate the samples used for the regularization term by adding random noise either onto training points or onto both training and generated samples.

We refer to this as "local perturbation" in the following.

7 The evolution of the critics loss when using this local perturbation can be seen in FIG0 .

Results are qualitatively similar to those when using the sampling procedure proposed by BID9 .

Interestingly, WGAN-GP training is stabilized at a later stage if one only adds noise to training examples and not to generated examples.

This indicates that enforcing the GP-penalty close to the data manifold is less harmful.

However, the critic's loss is still much more fluctuating than when training a WGAN-LP.The evolution of the approximated EM distance when using local perturbation (by adding noise to the training examples only) is shown in FIG0 .

Training with the GP-penalty leads to larger fluctuations of the approximated Wasserstein-1 distance than training with the LP-penalty.

However, fluctuations are less severe compared to the setting when the GP-penalty is used in combination with the sampling procedure proposed by BID9 .

and penalty weight λ = 10.

Results for the evolution of the critics loss and the approximated EM distance during training on the Swiss Roll data set are shown in FIG0 .

Both critic loss and EM reduce smoothly, which makes the Wasserstein-2 distance (in combination with its theoretical properties) an interesting candidate to further investigations.

Inception score.

The inception score was proposed by BID15 to evaluate the quality of images x sampled from a generative model ν based on the Inception model.

Let p(y|x) be the conditional probability of label y for image x under the Inception model and p(y) = p(y|x)ν(x)dx the marginal probability of labels y with respect to samples generated from ν.

Then the Inception score is given by exp (E x∼ν [KL(p(y|x) , p(y)]) .Intuitively, a good generative model should produce samples for which the conditional label distribution has low entropy, while the variability over samples and thus the entropy of the marginal label distribution should be high.

Therefore, a higher Inception score indicates a better performance of the generative model.

The maximal Inception scores reported in Table 1 are representative for the general evolution of the scores for WGAN-LP and WGAN-GP during training.

As an example we show the evolution of the Inception score for penalty weights of λ = 5 and λ = 100 in FIG0 .

It becomes clear that WGAN-GP performs similar to WGAN-LP for small values of the regularization parameter but much worse for larger values (this was consistently observed in all experiments).

In FIG0 we compare the performance of WGAN-LP and WGAN-GP in terms of the critics loss on a separate validation set, which again demonstrates a more stable behavior for WGAN-LP with respect to the choice of lambda.

We also trained WGAN-GP and WGAN-LP with a conditional model (making use of the label information of CIFAR10) with λ = 10 and found a similar performance for both, i.e. 8.537 ± 0.133 and 8.462 ± 0.115 for WGAN-GP and WGAN-LP, respectively.

Level sets for WGANs trained with the regularization terms given by Equation FORMULA10 and FORMULA13 and penalty weight 10 are shown in FIG0 .

As the evolution of the level sets and the sampled points indicate, training properly converges.

However, on CIFAR-10, the same penalties did not lead to good results.

As shown in FIG0 , using (7) for regularization initially lead to improving Inception scores but then quickly started to diverge, while using (9) lead to even greater instability.

Figure 17: Level sets of the critic f of WGANs during training, after 10, 50, 100, 500, and 1000 iterations.

Yellow corresponds to high, purple to low values of f .

Training samples are indicated in red, generated samples in blue.

Top: With the regularization term given in Equation FORMULA10 and λ = 10.

Bottom: With the regularization term given in Equation FORMULA13 and λ = 10.Figure 18: Inception scores for regularization Equation (7) for penalty weights 100 (red) and 5 (blue), shown on the left, and Inception scores for training with the regularization Equation (9) for penalty weights 100 (red) and 5 (blue), shown on the right.

@highlight

A new regularization term can improve your training of wasserstein gans

@highlight

The paper proposes a regularization scheme for Wasserstein GAN based on relaxation of the constraints on the Lipschitz constant of 1.

@highlight

The article deals with regularization/penalization in the fitting of GANs, when based on a L_1 Wasserstein metric.