Generative adversarial networks (GANs) are a widely used framework for learning generative models.

Wasserstein GANs (WGANs), one of the most successful variants of GANs, require solving a minmax problem to global optimality, but in practice, are successfully trained with stochastic gradient descent-ascent.

In this paper, we show that, when the generator is a one-layer network, stochastic gradient descent-ascent converges to a global solution in polynomial time and sample complexity.

Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) are a prominent framework for learning generative models of complex, real-world distributions given samples from these distributions.

GANs and their variants have been successfully applied to numerous datasets and tasks, including image-to-image translation (Isola et al., 2017) , image super-resolution (Ledig et al., 2017) , domain adaptation (Tzeng et al., 2017) , probabilistic inference (Dumoulin et al., 2016) , compressed sensing (Bora et al., 2017) and many more.

These advances owe in part to the success of Wasserstein GANs (WGANs) Gulrajani et al., 2017) , leveraging the neural net induced integral probability metric to better measure the difference between a target and a generated distribution.

Along with the afore-described empirical successes, there have been theoretical studies of the statistical properties of GANs-see e.g. (Zhang et al., 2018; Arora et al., 2017; Bai et al., 2018; Dumoulin et al., 2016) and their references.

These works have shown that, with an appropriate design of the generator and discriminator, the global optimum of the WGAN objective identifies the target distribution with low sample complexity.

On the algorithmic front, prior work has focused on the stability and convergence properties of gradient descent-ascent (GDA) and its variants in GAN training and more general min-max optimization problems; see e.g. (Nagarajan & Kolter, 2017; Heusel et al., 2017; Mescheder et al., 2017; Daskalakis et al., 2017; Daskalakis & Panageas, 2018a; b; Gidel et al., 2019; Liang & Stokes, 2019; Mokhtari et al., 2019; Jin et al., 2019; Lin et al., 2019) and their references.

It is known that, even in min-max optimization problems with convex-concave objectives, GDA may fail to compute the min-max solution and may even exhibit divergent behavior.

Hence, these works have studied conditions under which GDA converges to a globally optimal solution under a convex-concave objective, or different types of locally optimal solutions under nonconvex-concave or nonconvex-nonconcave objectives.

They have also identified variants of GDA with better stability properties in both theory and practice, most notably those using negative momentum.

In the context of GAN training, Feizi et al. (2017) show that for WGANs with a linear generator and quadratic discriminator GDA succeeds in learning a Gaussian using polynomially many samples in the dimension.

In the same vein, we are the first to our knowledge to study the global convergence properties of stochastic GDA in the GAN setting, and establishing such guarantees for non-linear generators.

In particular, we study the WGAN formulation for learning a single-layer generative model with some reasonable choices of activations including tanh, sigmoid and leaky ReLU.

Our contributions.

For WGAN with a one-layer generator network using an activation from a large family of functions and a quadratic discriminator, we show that stochastic gradient descent-ascent learns a target distribution using polynomial time and samples, under the assumption that the target distribution is realizable in the architecture of the generator.

This is achieved by a) analysis of the dynamics of stochastic gradient-descent to show it attains a global optimum of the minmax problem, and b) appropriate design of the discriminator to ensure a parametric O( 1 √ n ) statistical rate (Zhang et al., 2018; Bai et al., 2018) .

Related Work.

We briefly review relevant results in GAN training and learning generative models: -Optimization viewpoint.

For standard GANs and WGANs with appropriate regularization, Nagarajan & Kolter (2017) , Mescheder et al. (2017) and Heusel et al. (2017) establish sufficient conditions to achieve local convergence and stability properties for GAN training.

At the equilibrium point, if the Jacobian of the associated gradient vector field has only eigenvalues with negative real-part at the equilibrium point, GAN training is verified to converge locally for small enough learning rates.

A follow-up paper by (Mescheder et al., 2018) shows the necessity of these conditions by identifying a prototypical counterexample that is not always locally convergent with gradient descent based GAN optimization.

However, the lack of global convergence prevents the analysis to provide any guarantees of learning the real distribution.

The work of (Feizi et al., 2017) described above has similar goals as our paper, namely understanding the convergence properties of basic dynamics in simple WGAN formulations.

However, they only consider linear generators, which restrict the WGAN model to learning a Gaussian.

Our work goes a step further, considering WGANs whose generators are one-layer neural networks with a broad selection of activations.

We show that with a proper gradient-based algorithm, we can still recover the ground truth parameters of the underlying distribution.

More broadly, WGANs typically result in nonconvex-nonconcave min-max optimization problems.

In these problems, a global min-max solution may not exist, and there are various notions of local min-max solutions, namely local min-local max solutions Daskalakis & Panageas (2018b) , and local min solutions of the max objective Jin et al. (2019) , the latter being guaranteed to exist under mild conditions.

In fact, Lin et al. (2019) show that GDA is able to find stationary points of the max objective in nonconvex-concave objectives.

Given that GDA may not even converge for convexconcave objectives, another line of work has studied variants of GDA that exhibit global convergence to the min-max solution Daskalakis et al. (2017) ; Daskalakis & Panageas (2018a); Gidel et al. (2019); Liang & Stokes (2019) ; Mokhtari et al. (2019) , which is established for GDA variants that add negative momentum to the dynamics.

While the convergence of GDA with negative momentum is shown in convex-concave settings, there is experimental evidence supporting that it improves GAN training (Daskalakis et al., 2017; Gidel et al., 2019 ).

-Statistical viewpoint.

Several works have studied the issue of mode collapse.

One might doubt the ability of GANs to actually learn the distribution vs just memorize the training data (Arora et al., 2017; Dumoulin et al., 2016) .

Some corresponding cures have been proposed.

For instance, Zhang et al. (2018) ; Bai et al. (2018) show for specific generators combined with appropriate parametric discriminator design, WGANs can attain parametric statistical rates, avoiding the exponential in dimension sample complexity (Liang, 2018; Bai et al., 2018; Feizi et al., 2017) .

Recent work of Wu et al. (2019) provides an algorithm to learn the distribution of a single-layer ReLU generator network.

While our conclusion appears similar, our focus is very different.

Our paper targets understanding when a WGAN formulation trained with stochastic GDA can learn in polynomial time and sample complexity.

Their work instead relies on a specifically tailored algorithm for learning truncated normal distributions Daskalakis et al. (2018) .

We consider GAN formulations for learning a generator

, where A is a d × k parameter matrix and φ some activation function.

We consider discriminators

that are linear or quadratic forms respectively for the different purposes of learning the marginals or the joint distribution.

We assume latent variables z are sampled from the normal N (0, I k×k ), where I k×k denotes the identity matrix of size k. The real/target distribution outputs samples x ∼ D = G A * (N (0, I k0×k0 )), for some ground truth parameters A * , where A * is d × k 0 , and we take k ≥ k 0 for enough expressivity, taking k = d when k 0 is unknown.

The Wasserstain GAN under our choice of generator and discriminator is naturally formulated as:

We will replace v by V ∈ R d×d when necessary.

We use a i to denote the i-th row vector of A. We sometimes omit the 2 subscript, using x to denote the 2-norm of vector x, and X to denote the spectral norm of matrix X. S n ⊂ R n×n represents all the symmetric matrices of dimension n × n.

We use Df (X 0 ) [B] to denote the directional derivative of function f at point X 0 with direction B:

3 WARM-UP: LEARNING THE MARGINAL DISTRIBUTIONS As a warm-up, we ask whether a simple linear discriminator is sufficient for the purposes of learning the marginal distributions of all coordinates of D. Notice that in our setting, the i-th output of the generator is φ(x) where x ∼ N (0, a i 2 ), and is thus solely determined by a i 2 .

With a linear discriminator D v (x) = v x, our minimax game becomes:

Notice that when the activation φ is an odd function, such as the tanh activation, the symmetric property of the Gaussian distribution ensures that E x∼D [v x] = 0, hence the linear discriminator in f 1 reveals no information about A * .

Therefore specifically for odd activations (or odd plus a constant activations), we instead use an adjusted rectified linear discriminator

to enforce some bias, where C = 1 2 (φ(x) + φ(−x)) for all x, and R denotes the ReLU activation.

Formally, we slightly modify our loss function as:

We will show that we can learn each marginal of D if the activation function φ satisfies the following.

Assumption 1.

The activation function φ satisfies either one of the following: 1.

φ is an odd function plus constant, and φ is monotone increasing; 2.

The even component of φ, i.e.

To bound the capacity of the discriminator, similar to the Lipschitz constraint in WGAN, we regularize the discriminator.

For the regularized formulation we have: Theorem 1.

In the same setting as Lemma 1, alternating gradient descent-ascent with proper learning rates on

All the proofs of the paper can be found in the appendix.

We show that all local min-max points in the sense of (Jin et al., 2019) of the original problem are global min-max points and recover the correct norm of a * i , ∀i.

Notice for the source data distribution

2 )) and is determined by a * i .

Therefore we have learned the marginal distribution for each entry i.

It remains to learn the joint distribution.

In the previous section, we utilize a (rectified) linear discriminator, such that each coordinate v i interacts with the i-th random variable.

With the (rectified) linear discriminator, WGAN learns the correct a i , for all i. However, since there's no interaction between different coordinates of the random vector, we do not expect to learn the joint distribution with a linear discriminator.

To proceed, a natural idea is to use a quadratic discriminator D V (x) := x V x = xx , V to enforce component interactions.

Similar to the previous section, we study the regularized version:

Under review as a conference paper at ICLR 2020 where

By adding a regularizer on V and explicitly maximizing over V :

In the next subsection, we first focus on analyzing the second-order stationary points of g, then we establish that gradient descent ascent converges to second-order stationary points of g .

We first assume that both A and A * have unit row vectors, and then extend to general case since we already know how to learn the row norms from Section 3.

To explicitly compute g(A), we rely on the property of Hermite polynomials.

Since normalized Hermite polynomials {h i } ∞ i=0 forms an orthonomal basis in the functional space, we rewrite the activation function as φ(x) = ∞ i=0 σ i h i , where σ i is the i-th Hermite coefficient.

We use the following claim:

2 /2 ), and let its Hermite expansion be φ =

Therefore we could compute the value of f 2 explicitly using the Hermite polynomial expansion:

Here X •i is the Hadamard power operation where (X •i ) jk = (X jk ) i .

Therefore we have:

We reparametrize with Z = AA and defineg(Z) = g(A) with individual component functions

Assumption 2.

The activation function φ is an odd function plus constant.

In other words, its

Additionally we assume σ 1 = 0.

Remark 2.

Common activations like tanh and sigmoid satisfy Assumption 2.

Lemma 2.

For activations including leaky ReLU and functions satisfying Assumption 2,g(Z) has a unique stationary point where

Noticeg(Z) = jkg jk (z jk ) is separable across z jk , where eachg jk is a polynomial scalar function.

Lemma 2 comes from the fact that the only zero point forg jk is z jk = z * jk , for odd activation φ and leaky ReLU.

Then we migrate this good property to the original problem we want to solve: Problem 1.

We optimize over function g when a * i = 1, ∀i:

Existing work Journée et al. (2008) connectsg(Z) to the optimization over factorized version for g(A) (g(A) ≡g(AA )).

Specifically, when k = d, all second-order stationary points for g(A) are first-order stationary points forg(Z).

Thoughg is not convex, we are able to show that its first-order stationary points are global optima when the generator is sufficiently expressive, i.e., k ≥ k 0 .

In reality we won't know the latent dimension k 0 , therefore we just choose k = d for simplicity.

We make the following conclusion: Theorem 2.

For activations including leaky ReLU and functions satisfying Assumption 2, when k = d, all second-order KKT points for problem 1 are its global minimum.

Therefore alternating projected gradient descent-ascent on Eqn.

(3) converges to A : AA = A * (A * ) .

The extension for non-unit vectors is straightforward, and we defer the analysis to the Appendix.

Algorithm 1 Online stochastic gradient descent ascent on WGAN 1: Input: n training samples:

, learning rate for generating parameters η, number of iterations T .

2: Random initialize generating matrix

Generate m latent variables z

Gradient ascent on V with optimal step-size η V = 1:

Sample noise e uniformly from unit sphere 7:

Projected Gradient Descent on A, with constraints C = {A|(AA ) ii = (A * A * ) ii } :

8: end for 9: Output:

In this section, we consider analyzing Algorithm 1, i.e., gradient descent ascent on the following:

Notice in each iteration, gradient ascent with step-size 1 finds the optimal solution for V .

By Danskin's theorem (Danskin, 2012), our min-max optimization is essentially gradient descent over g

F with a batch of samples {z

Therefore to bound the difference between f n (A) and the population risk g (A) , we analyze the sample complexity required on the observation side (x i ∼ D, i ∈ [n]) and the mini-batch size required on the learning part (φ(Az j ), z j ∼ N (0, I k×k ), j ∈ [m]).

We will show that with large enough n, m, the algorithm specified in Algorithm 1 that optimizes over the empirical risk will yield the ground truth covariance matrix with high probability.

Our proof sketch is roughly as follows:

1.

With high probability, projected stochastic gradient descent finds a second order stationary pointÂ of f n (·) as shown in Theorem 31 of (Ge et al., 2015) .

2.

For sufficiently large m, our empirical objective, though a biased estimator of the population risk g(·), achieves good -approximation to the population risk on both the gradient and Hessian (Lemmas 4&5).

ThereforeÂ is also an O( )-approximate second order stationary point (SOSP) for the population risk g(A).

3.

We show that any -SOSPÂ for g(A) yields an O( )-first order stationary point (FOSP)Ẑ ≡ÂÂ for the semi-definite programming ong(Z) (Lemma 6).

4.

We show that any O( )-FOSP of functiong(Z) induces at most O( ) absolute error compared to the ground truth covariance matrix Z * = A * (A * ) (Lemma 7).

For simplicity, we assume the activation and its gradient satisfy Lipschitz continuous, and let the Lipschitz constants be 1 w.l.o.g.: Assumption 3.

Assume the activation is 1-Lipschitz and 1-smooth.

To estimate observation sample complexity, we will bound the gradient and Hessian for the population risk and empirical risk on the observation samples:

, and

Lemma 3.

Suppose the activation satisfies Assumption 3.

Lemma 4.

Suppose the activation satisfies Assumption 2&3.

With samples

) with probability 1 − δ.

Normally for empirical risk for supervised learning, the mini-batch size can be arbitrarily small since the estimator of the gradient is unbiased.

However in the WGAN setting, notice for each iteration, we randomly sample a batch of random variables {z i } i∈ [m] , and obtain a gradient of

, in Algorithm 1.

However, the finite sum is inside the Frobenius norm and the gradient on each mini-batch may no longer be an unbiased estimator for our target g n (A) =

In other words, we conduct stochastic gradient descent over the function f (A) ≡ E zgm,n (A).

Therefore we just need to analyze the gradient error between this f (A) and g n (A) (i.e.g m,n is almost an unbiased estimator of g n ).

Finally with the concentration bound derived in last section, we get the error bound between f (A) and g(A).

Lemma 5.

The empirical riskg m,n is almost an unbiased estimator of g n .

Specifically, the expected function f (A) = E zi∼N (0,I k×k ),i∈ [m] [g m,n ] satisfies:

2Θ hides log factors of d for simplicity.

For arbitrary direction matrix B,

In summary, we conduct concentration bound over the observation samples and mini-batch sizes, and show the gradient of f (A) that Algorithm 1 is optimizing over has close gradient and Hessian with the population risk g(A).

Therefore a second-order stationary point (SOSP) for f (A) (that our algorithm is guaranteed to achieve) is also an approximated SOSP for g(A).

Next we show such a point also yield an approximated first-order stationary point of the reparametrized functioñ g(Z) ≡ g(A), ∀Z = AA .

In this section, we establish the relationship betweeng and g. We present the general form of our target Problem 1:

s.t.

Tr(A X i A) = y i , X i ∈ S, y i ∈ R, i = 1, · · · , n. Similar to the previous section, the stationary property might not be obvious on the original problem.

Instead, we could look at the re-parametrized version as:

d×k is called an -approximate second-order stationary point ( -SOSP) of Eqn.

(5) if there exists a vector λ such that:

.

Specifically, when = 0 the above definition is exactly the second-order KKT condition for optimizing (5).

Next we present the approximate first-order KKT condition for (6): Definition 2.

A symmetric matrix Z ∈ S n is an -approximate first order stationary point of function (6) ( -FOSP) if and only if there exist a vector σ ∈ R m and a symmetric matrix S ∈ S such that the following holds: (5) with A and λ, it infers an -FOSP of function (6) with Z, σ and S that satisfies: Z = AA , σ = λ and S = ∇ Zg (AA ) − i λ i X i .

Now it remains to show an -FOSP ofg(Z) indeed yields a good approximation for the ground truth parameter matrix.

is the optimal solution for function (6).

Together with the previous arguments, we finally achieve our main theorem on connecting the recovery guarantees with the sample complexity and batch size 3 : Theorem 3.

For arbitrary δ < 1, , given small enough learning rate η < 1/poly(d, 1/ , log(1/δ)), let sample size n ≥Θ(d

In this section, we provide simple experimental results to validate the performance of stochastic gradient descent ascent and provide experimental support for our theory.

We focus on Algorithm 1 that targets to recover the parameter matrix.

We conduct a thorough empirical studies on three joint factors that might affect the performance: the number of observed samples m (we set n = m as in general GAN training algorithms), the different choices of activation function φ, and the output dimension d. In Figure 1 we plot the relative error for parameter estimation decrease over the increasing sample complexity.

We fix the hidden dimension k = 2, and vary the output dimension over {3, 5, 7} and sample complexity over {500, 1000, 2000, 5000, 10000}. Reported values are averaged from 20 runs and we show the standard deviation with the corresponding colored shadow.

Clearly the recovery error decreases with higher sample complexity and smaller output dimension.

To visually demonstrate the learning process, we also include a simple comparison for different φ: i.e. leaky ReLU and tanh activations, when k = 1 and d = 2.

We set the ground truth covariance matrix to be Notice that by proposing a rectified linear discriminator, we have essentially modified the activation function asφ := R(φ − C), where C = 1 2 (φ(x) + φ(−x)) is the constant bias term of φ.

Observe that we can rewrite the objectivef 1 for this case as follows:

Moreover, notice thatφ is positive and increasing on its support which is [0, +∞).

Now let us consider the other case in our statement where φ has a positive and monotone increasing even component in [0, +∞).

In this case, let us take:

Because of the symmetry of the Gaussian distribution, we can rewrite the objective function for this case as follows:

Moreover, notice thatφ is positive and increasing on its support which is [0, +∞).

To conclude, in both cases, the optimization objective can be written as follows, whereφ satisfies Assumption 1.2 and is only non-zero on [0, +∞).

The stationary points of the above objective satisfy:

We focus on the gradient over v. To achieve ∇ v f 1 (A, v) = 0, the stationary point satisfies:

To recap, for activations φ that follow Assumption 1, in both cases we have written the necessary condition on stationary point to be Eqn.

(7), whereφ is defined differently for odd or non-odd activations, but in both cases it is positive and monotone increasing on its support [0, ∞).

We then argue the only solution for Eqn.

(7) satisfies a j = a * j , ∀j.

This follows directly from the following claim:

Claim 3.

The function h(α) := E x∼N (0,α 2 ) f (x), α > 0 is a monotone increasing function if f is positive and monotone increasing on its support [0, ∞).

We could see from Claim 3 that the LHS and RHS of Eqn.

(7) is simply h( a j ) and h( a * j ) for each j. Now that h is an monotone increasing function, the unique solution for h( a j ) = h( a * j ) is to match the norm: a j = a * j , ∀j.

h

Since f , f , and α > 0, and we only care about the support of f where x is also positive, therefore h is always positive and h is monotone increasing.

To sum up, at stationary point where ∇f 1 (A, v) = 0, we have ∀i, a * i = a i .

Proof of Theorem 1.

We will take optimal gradient ascent steps with learning rate 1 on the discriminator side v, hence the function we will actually be optimizing over becomes (using the notation for φ from section A.1):

We just want to verify that there's no spurious local minimum for h(A).

Notice there's no interaction between each row vector of A. Therefore we instead look at each

Due to the symmetry of the Gaussian, we take a i = ae 1 , where a = a i .

It is easy to see that checking whether E z∼N (0,I k×k ) zφ (a i z) = 0 is equivalent to checking whether E z1∼N (0,1) z 1φ (az 1 ) = 0.

Recall thatφ is supported on [0, +∞) and it is monotonically increasing on its support.

Hence,

Therefore all stationary points of h(A) are global minima where E z∼N (0,I k 0 ×k 0 )φ (A * z) = E z∼N (0,I k×k )φ (Az) and according to Lemma 1, this only happens when

Proof of Lemma 2.

To study the stationary point forg(Z) = jkg jk (z jk ), we look at individual

Notice for odd-plus-constant activations, σ i is zero for even i > 0.

Recall our assumption in Lemma 2 also requires that σ 1 = 0.

Since the analysis is invariance to the position of the matrix Z, we simplify the notation here and essentially want to study the stationary point for

for some constant b and σ i , where

Notice now f (a) = 0 ⇔ a = b. This is because the polynomial f (a) is factorized to a − b and two factors I and II that are always positive.

Notice here we use

always nonnegative.

This is simply because a i − b i always shares the same sign as a − b when i is odd.

Therefore I=σ

Meanwhile, since a i−1 is always nonnegative for each odd i, we have II= σ 2 1 + i≥3 odd iσ 2 i a i−1 is also always positive for any a.

Next, for activation like ReLU, lossg (Daniely et al., 2016) .

Therefore h (−1) = 0 for any z * jk .

This fact prevents us from getting the same conclusion for ReLU.

However, for leaky ReLU with coefficient of leakage α ∈ (0, 1), φ(x) = max{x, αx} = (1 − α)ReLU(x) + αx.

We have

To sum up, for odd activations and leaky ReLU, since eachg jk (z) only has stationary point of z = z * jk , the stationary point Z ofg(Z) = jkg jk also satisfy

Proof of Theorem 2.

Instead of directly looking at the second-order stationary point of Problem 1, we look at the following problem on its reparametrized version:

Here Z * = A * (A * ) and satisfies z * ii = 1, ∀i.

Compared to function g in the original problem 1, it satisfies thatg(AA ) ≡ g(A).

A matrix Z satisfies the first-order stationary point for Problem 2 if there exists a vector σ such that:

Therefore for a stationary point Z, since Z * = A * (A * ) 0, and S 0, we have S, Z * − Z = S, Z * ≥ 0.

Meanwhile,

(Refer to proof of Lemma 2 for the value of g )

(P is always positive)

Therefore S, Z * − Z = 0, and this only happens when Z = Z * .

Finally, from Journée et al. (2008) we know that any first-order stationary point for Problem 2 is a second-order stationary point for our original problem 1 5 .

Therefore we conclude that all second-order stationary point for Problem 1 are global minimum A: AA = A * (A * ) .

In the previous argument, we simply assume that the norm of each generating vectors a i to be 1.

This practice simplifies the computation but is not practical.

Since we are able to estimate a i for all i first, we could analyze the landscape of our loss function for general matrix A.

The main tool is to use the multiplication theorem of Hermite functions:

For the ease of notation, we denote the coefficient as η

We extend the calculations for Hermite inner product for non-standard distributions.

Here l = min{m, n}.

Proof.

Denote the normalized variablesx = x/α,ŷ = y/β.

Let l = min{m, n}.

To simplify the notation, for a specific i, j pair, we writex = a i z/α, α = a i andŷ = a j z/β, where β = a j .

Namely we have

Therefore we could write out explicitly the coefficient for each term ρ k , k odd, as:

Now suppose σ i to have the same sign, and

We just want to show Z := AA , σ := λ, and S := S A satisfies the conditions for -FOSP of Eqn.

(6).

Therefore, by going over the conditions, its easy to tell that all other conditions automatically apply and it remains to show S A − I.

(from Lemma 5 of Journée et al. (2008)

(From Eqn.

(9) we have Tr(B XiA) = 0)

Notice that A ∈ R d×k and we have chosen k = d for simplicity.

We first argue when A is rankdeficient, i.e. rank(A) < k. There exists some vector v ∈ R k such that Av = 0.

Now for any vector b ∈ R d , let B = bv .

Therefore AB = Avb = 0.

From (10) we further have:

Therefore from the last three rows we have b S A b ≥ − /2 b 2 for any b, i.e. S A − /2I d×d .

On the other hand, when A is full rank, the column space of A is the entire R d vector space, and therefore S A −

I d×d directly follows from the second line of the -SOSP definition.

Recall the population risk

Write the empirical risk on observations as:

where X = E x∼D [xx ] , and

Proof.

Now write S(A) = φ(Az)φ(Az) .

And

Claim 5.

For arbitrary matrix B, the directional derivative of ∇g(A) − ∇g n (A) with direction B is:

6 .

Without loss of generality we assumed a j = 1, ∀j,

Then by matrix concentration inequality ((Vershynin, 2010) Corollary 5.52), we have with probability 1−δ:

Proof of Lemma 4.

On the other hand, our target function is:

Therefore E Sgm,n (A) − g n (A) =

(from the definition of -FOSP)

On the other hand, from the definition ofg, we have:

Here polynomial

is always positive for z = z * and k to be odd.

Therefore by comparing (12) and (13) we have Z − Z *

Proof of Theorem 3.

From Theorem 31 from Ge et al. (2015) , we know for small enough learning rate η, and arbitrary small , there exists large enough T , such that Algorithm 1 generates an output A (T ) that is sufficiently close to the second order stationary point for f .

Or formally we have, .

Then the second line is a sufficient condition for the following: Finally with Lemma 7, we get Z − Z * F ≤ O( ).

@highlight

We show that stochastic gradient descent ascent converges to a global optimum for WGAN with one-layer generator network.

@highlight

Attempts to prove that the Stochastic Gradient Decent-Ascent could converge to a global solution for the min-max problem of WGAN.