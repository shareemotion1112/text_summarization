This paper presents a novel two-step approach for the fundamental problem of learning an optimal map from one distribution to another.

First, we learn an optimal transport (OT) plan, which can be thought as a one-to-many map between the two distributions.

To that end, we propose a stochastic dual approach of regularized OT, and show empirically that it scales better than a recent related approach when the amount of samples is very large.

Second, we estimate a Monge map as a deep neural network learned by approximating the barycentric projection of the previously-obtained OT plan.

This parameterization allows generalization of the mapping outside the support of the input measure.

We prove two theoretical stability results of regularized OT which show that our estimations converge to the OT and Monge map between the underlying continuous measures.

We showcase our proposed approach on two applications: domain adaptation and generative modeling.

Mapping one distribution to another Given two random variables X and Y taking values in X and Y respectively, the problem of finding a map f such that f (X) and Y have the same distribution, denoted f (X) ∼ Y henceforth, finds applications in many areas.

For instance, in domain adaptation, given a source dataset and a target dataset with different distributions, the use of a mapping to align the source and target distributions is a natural formulation BID22 since theory has shown that generalization depends on the similarity between the two distributions BID2 .

Current state-of-the-art methods for computing generative models such as generative adversarial networks BID21 , generative moments matching networks BID26 or variational auto encoders BID24 ) also rely on finding f such that f (X) ∼ Y .

In this setting, the latent variable X is often chosen as a continuous random variable, such as a Gaussian distribution, and Y is a discrete distribution of real data, e.g. the ImageNet dataset.

By learning a map f , sampling from the generative model boils down to simply drawing a sample from X and then applying f to that sample.

Mapping with optimality Among the potentially many maps f verifying f (X) ∼ Y , it may be of interest to find a map which satisfies some optimality criterion.

Given a cost of moving mass from one point to another, one would naturally look for a map which minimizes the total cost of transporting the mass from X to Y .

This is the original formulation of Monge (1781) , which initiated the development of the optimal transport (OT) theory.

Such optimal maps can be useful in numerous applications such as color transfer BID17 , shape matching BID46 , data assimilation BID37 , or Bayesian inference BID31 .

In small dimension and for some specific costs, multi-scale approaches BID28 or dynamic formulations BID16 BID3 BID44 can be used to compute optimal maps, but these approaches become intractable in higher dimension as they are based on space discretization.

Furthermore, maps veryfiying f (X) ∼ Y might not exist, for instance when X is a constant but not Y .

Still, one would like to find optimal maps between distributions at least approximately.

The modern approach to OT relaxes the Monge problem by optimizing over plans, i.e. distributions over the product space X × Y, rather than maps, casting the OT problem as a linear program which is always feasible and easier to solve.

However, even with specialized algorithms such as the network simplex, solving that linear program takes O(n 3 log n) time, where n is the size of the discrete distribution (measure) support.

Large-scale OT Recently, BID14 showed that introducing entropic regularization into the OT problem turns its dual into an easier optimization problem which can be solved using the Sinkhorn algorithm.

However, the Sinkhorn algorithm does not scale well to measures supported on a large number of samples, since each of its iterations has an O(n 2 ) complexity.

In addition, the Sinkhorn algorithm cannot handle continuous probability measures.

To address these issues, two recent works proposed to optimize variations of the dual OT problem through stochastic gradient methods.

BID20 proposed to optimize a "semi-dual" objective function.

However, their approach still requires O(n) operations per iteration and hence only scales moderately w.r.t.

the size of the input measures.

BID1 proposed a formulation that is specific to the so-called 1-Wasserstein distance (unregularized OT using the Euclidean distance as a cost function).

This formulation has a simpler dual form with a single variable which can be parameterized as a neural network.

This approach scales better to very large datasets and handles continuous measures, enabling the use of OT as a loss for learning a generative model.

However, a drawback of that formulation is that the dual variable has to satisfy the non-trivial constraint of being a Lipschitz function.

As a workaround, BID1 proposed to use weight clipping between updates of the neural network parameters.

However, this makes unclear whether the learned generative model is truly optimized in an OT sense.

Besides these limitations, these works only focus on the computation of the OT objective and do not address the problem of finding an optimal map between two distributions.

We present a novel two-step approach for learning an optimal map f that satisfies f (X) ∼ Y .

First, we compute an optimal transport plan, which can be thought as a one-to-many map between the two distributions.

To that end, we propose a new simple dual stochastic gradient algorithm for solving regularized OT which scales well with the size of the input measures.

We provide numerical evidence that our approach converges faster than semi-dual approaches considered in BID20 .

Second, we learn an optimal map (also referred to as a Monge map) as a neural network by approximating the barycentric projection of the OT plan obtained in the first step.

Parameterization of this map with a neural network allows efficient learning and provides generalization outside the support of the input measure.

FIG0 example showing the computed map between a Gaussian measure and a discrete measure and the resulting density estimation.

On the theoretical side, we prove the convergence of regularized optimal plans (resp.

barycentric projections of regularized optimal plans) to the optimal plan (resp.

Monge map) between the underlying continuous measures from which data are sampled.

We demonstrate our approach on domain adaptation and generative modeling.

We denote X and Y some complete metric spaces.

In most applications, these are Euclidean spaces.

We denote random variables such as X or Y as capital letters.

We use X ∼ Y to say that X and Y have the same distribution, and also X ∼ µ to say that X is distributed according to the probability measure µ. Supp(µ) refers to the support of µ, a subset of X , which is also the set of values which X ∼ µ can take.

Given X ∼ µ and a map f defined on Supp(µ), f #µ is the probability distribution of f (X).

We say that a measure is continuous when it admits a density w.r.t.

the Lebesgues measure.

We denote id the identity map.

The Monge Problem Consider a cost function c : (x, y) ∈ X × Y → c(x, y) ∈ R + , and two random variables X ∼ µ and Y ∼ ν taking values in X and Y respectively.

The Monge problem (Monge, 1781) consists in finding a map f : X → Y which transports the mass from µ to ν while minimizing the mass transportation cost, DISPLAYFORM0 Monge originally considered the cost c(x, y) = ||x − y|| 2 , but in the present article we refer to the Monge problem as Problem (1) for any cost c. When µ is a discrete measure, a map f satisfying the constraint may not exist: if µ is supported on a single point, no such map exists as soon as ν is not supported on a single point.

In that case, the Monge problem is not feasible.

However, when X = Y = R d , µ admits a density and c is the squared Euclidean distance, an important result by BID8 states that the Monge problem is feasible and that the infinum of Problem FORMULA0 is attained.

The existence and uniqueness of Monge maps, also referred to as optimal maps, was later generalized to more general costs (e.g. strictly convex and super-linear) by several authors.

With the notable exception of the Gaussian to Gaussian case which has a close form affine solution, computation of Monge maps remains an open problem for measures supported on high-dimensional spaces.

Kantorovich Relaxation In order to make Problem (1) always feasible, BID23 relaxed the Monge problem by casting Problem (1) into a minimization over couplings (X, Y ) ∼ π rather than the set of maps, where π should have marginals equals to µ and ν, DISPLAYFORM1 Concretely, this relaxation allows mass at a given point x ∈ Supp(µ) to be transported to several locations y ∈ Supp(ν), while the Monge problem would send the whole mass at x to a unique location f (x).

This relaxed formulation is a linear program, which can be solved by specialized algorithms such as the network simplex when considering discrete measures.

However, current implementations of this algorithm have a super-cubic complexity in the size of the support of µ and ν, preventing wider use of OT in large-scale settings.

Regularized OT OT regularization was introduced by BID14 in order to speed up the computation of OT.

Regularization is achieved by adding a negative-entropy penalty R (defined in Eq. (5)) to the primal variable π of Problem (2), DISPLAYFORM2 Besides efficient computation through the Sinkhorn algorithm, regularization also makes the OT distance differentiable everywhere w.r.t.

the weights of the input measures , whereas OT is differentiable only almost everywhere.

We also consider the L 2 regularization introduced by BID15 , whose computation is found to be more stable since there is no exponential term causing overflow.

As highlighted by , adding an entropy or squared L 2 norm regularization term to the primal problem (3) makes the dual problem an unconstrained maximization problem.

We use this dual formulation in the next section to propose an efficient stochastic gradient algorithm.

By considering the dual of the regularized OT problem, we first show that stochastic gradient ascent can be used to maximize the resulting concave objective.

A close form for the primal solution π of Problem (3) can then be obtained by using first-order optimality conditions.

OT dual Let X ∼ µ and Y ∼ ν.

The Kantorovich duality provides the following dual of the OT problem (2), DISPLAYFORM0 This dual formulation suggests that stochastic gradient methods can be used to maximize the objective of Problem (4) by sampling batches from the independant coupling µ × ν.

However there is no easy way to fulfill the constraint on u and v along gradient iterations.

This motivates considering regularized optimal transport.

Regularized OT dual The hard constraint in Eq. (4) can be relaxed by regularizing the primal problem (2) with a strictly convex regularizer R as detailed in .

In the present paper, we consider both entropy regularization R e used in BID14 BID20 and DISPLAYFORM1 (5) where dπ(x,y) dµ(x)dν(y) is the density, i.e. the Radon-Nikodym derivative, of π w.r.t.

µ × ν.

When µ and ν are discrete, and so is π, the integrals are replaced by sums.

The dual of the regularized OT problems can be obtained through the Fenchel-Rockafellar's duality theorem, DISPLAYFORM2 where DISPLAYFORM3 Compared to Problem (4), the constraint u(x) + v(y) c(x, y) has been relaxed and is now enforced smoothly through a penalty term F ε (u(x), v(y)) which is concave w.r.t. (u, v).

Although we derive formula and perform experiments w.r.t.

entropy and L 2 regularizations, any strictly convex regularizer which is decomposable, i.e. which can be written R(π) = ij R ij (π ij ) (in the discrete case), gives rise to a dual problem of the form Eq. (6), and the proposed algorithms can be adapted.

In order to recover the solution π ε of the regularized primal problem (3), we can use the first-order optimality conditions of the Fenchel-Rockafellar's duality theorem, DISPLAYFORM0 Algorithm The relaxed dual (6) is an unconstrained concave problem which can be maximized through stochastic gradient methods by sampling batches from µ × ν.

When µ is discrete, i.e. µ = n i=1 a i δ xi , the dual variable u is a n-dimensional vector over which we carry the optimization, where u(x i ) def.= u i .

When µ has a density, u is a function on X which has to be parameterized in order to carry optimization.

We thus consider deep neural networks for their ability to approximate DISPLAYFORM1 sample a batch (y 1 , · · · , y p ) from ν 7: DISPLAYFORM2 end while general functions.

BID20 used the same stochastic dual maximization approach to compute the regularized OT objective in the continuous-continuous setting.

The difference lies in their pamaterization of the dual variables as kernel expansions, while we decide to use deep neural networks.

Using a neural network for parameterizing a continuous dual variable was done also by BID1 .

The same discussion also stands for the second dual variable v. Our stochastic gradient algorithm is detailed in Alg.

1.Convergence rates and computational cost comparison.

We first discuss convergence rates in the discrete-discrete setting (i.e. both measures are discrete), where the problem is convex, while parameterization of dual variables as neural networks in the semi-discrete or continuous-continuous settings make the problem non-convex.

Because the dual (6) is not strongly convex, full-gradient descent converges at a rate of O(1/k), where k is the iteration number.

SGD with a decreasing step size converges at the inferior rate of O(1/ √ k) BID32 ), but with a O(1) cost per iteration.

The two rates can be interpolated when using mini-batches, at the cost of O(p 2 ) per iteration, where p is the mini-batch size.

In contrast, BID20 considered a semi-dual objective of the form E X∼µ [u(X) + G ε (u(X))], with a cost per iteration which is now O(n) due to the computation of the gradient of G ε .

Because that objective is not strongly convex either, SGD converges at the same O(1/ √ k) rate, up to problem-specific constants.

As noted by BID20 , this rate can be improved to O(1/k) while maintaining the same iteration cost, by using stochastic average gradient (SAG) method BID42 .

However, SAG requires to store past stochastic gradients, which can be problematic in a large-scale setting.

In the semi-discrete setting (i.e. one measure is discrete and the other is continuous), SGD on the semi-dual objective proposed by BID20 also converges at a rate of O(1/ √ k), whereas we only know that Alg.

1 converges to a stationary point in this non-convex case.

In the continuous-continuous setting (i.e. both measures are continuous), BID20 proposed to represent the dual variables as kernel expansions.

A disadvantage of their approach, however, is the O(k 2 ) cost per iteration.

In contrast, our approach represents dual variables as neural networks.

While non-convex, our approach preserves a O(p 2 ) cost per iteration.

This parameterization with neural networks was also used by BID1 who maximized the 1-Wasserstein dual-objective function DISPLAYFORM3 .

Their algorithm is hence very similar to ours, with the same complexity O(p 2 ) per iteration.

The main difference is that they had to constrain u to be a Lipschitz function and hence relied of weight clipping in-between gradient updates.

The proposed algorithm is capable of computing the regularized OT objective and optimal plans between empirical measures supported on arbitrary large numbers of samples.

In statistical machine learning, one aims at estimating the underlying continuous distribution from which empirical observations have been sampled.

In the context of optimal transport, one would like to approximate the true (non-regularized) optimal plan between the underlying measures.

The next section states theoretical guarantees regarding this problem.

Consider discrete probability measures µ n = n i=1 a i δ xi ∈ P (X ) and ν n = n j=1 b j δ yj ∈ P (Y).

Analysis of entropy-regularized linear programs BID10 shows that the solution π ε n of the entropy-regularized problem (3) converges exponentially fast to a solution π n of the non-regularized OT problem (2).

Also, a result about stability of optimal transport BID47 [Theorem 5.20] states that, if µ n → µ and ν n → ν weakly, then a sequence (π n ) of optimal transport plans between µ n and ν n converges weakly to a solution π of the OT problem between µ and ν.

We can thus write, lim DISPLAYFORM0 A more refined result consists in establishing the weak convergence of π ε n to π when (n, ε) jointly converge to (∞, 0).

This is the result of the following theorem which states a stability property of entropy-regularized plans (proof in the Appendix).

Theorem 1.

Let µ ∈ P (X ) and ν ∈ P (Y) where X and Y are complete metric spaces.

Let µ n = n i=1 a i δ xi and ν n = n j=1 b j δ yj be discrete probability measures which converge weakly to µ and ν respectively, and let (ε n ) a sequence of non-negative real numbers converging to 0 sufficiently fast.

Assume the cost c is continuous on X × Y and finite.

Let π εn n the solution of the entropy-regularized OT problem (3) between µ n and ν n .

Then, up to extraction of a subsequence, (π εn n ) converges weakly to the solution π of the OT problem (2) between µ and ν, π εn n → π weakly.

Keeping the analogy with statistical machine learning, this result is an analog to the universal consistency property of a learning method.

In most applications, we consider empirical measures and n is fixed, so that regularization, besides enabling dual stochastic approach, may also help learn the optimal plan between the underlying continuous measures.

So far, we have derived an algorithm for computing the regularized OT objective and regularized optimal plans regardless of µ and ν being discrete or continuous.

The OT objective has been used successfully as a loss in machine learning BID30 BID19 BID40 BID1 BID12 , whereas the use of optimal plans has straightforward applications in logistics, as well as economy BID23 Carlier, 2012) or computer graphics BID7 .

In numerous applications however, we often need mappings rather than joint distributions.

This is all the more motivated since BID8 proved that when the source measure is continuous, the optimal transport plan is actually induced by a map.

Assuming that available data samples are sampled from some underlying continuous distributions, finding the Monge map between these continuous measures rather than a discrete optimal plan between discrete measures is essential in machine learning applications.

Hence in the next section, we investigate how to recover an optimal map, i.e. find an approximate solution to the Monge problem (1), from regularized optimal plans.

A map can be obtained from a solution to the OT problem (2) or regularized OT problem (3) through the computation of its barycentric projection.

Indeed, a solution π of Problem (2) or (3) between a source measure µ and a target measure ν is, identifying the plan π with its density w.r.t.

a reference measure, a function π : (x, y) ∈ X × Y → R + which can be seen as a weighted one-to-many map, i.e. π sends x to each location y ∈ Supp(ν) where π(x, y) > 0.

A map can then be obtained by simply averaging over these y according to the weights π(x, y).

Definition 1. (Barycentric projection) Let π be a solution of the OT problem (2) or regularized OT problem (3).

The barycentric projectionπ w.r.t.

a convex cost d : Y × Y → R + is defined as, DISPLAYFORM0 In the special case d(x, y) = ||x − y|| 2 2 , Eq. (11) has the close form solutionπ(x) = E Y ∼π(·|x) [Y ] , which is equal toπ = πy t a in a discrete setting with y = (y 1 , · · · , y n ) and a the weights of µ. Moreover, for the specific squared Euclidean cost c(x, y) = ||x − y|| 2 2 , the barycentric projectionπ is an optimal map BID0 [Theorem 12.4 .4], i.e.π is a solution to the Monge problem (1) between the source measure µ and the target measureπ#µ. Hence the barycentric projection w.r.t.

the squared Euclidean cost is often used as a simple way to recover optimal maps from optimal transport plans BID38 BID17 BID43 .

Inputs: input measures µ, ν; cost function c; dual optimal variables u and v; map f θ parameterized as a deep NN; batch size n; learning rate γ.

while not converged do sample a batch ( DISPLAYFORM0 Formula (11) provides a pointwise value of the barycentric projection.

When µ is discrete, this means that we only have mapping estimations for a finite number of points.

In order to define a map which is defined everywhere, we parameterize the barycentric projection as a deep neural network.

We show in the next paragraph how to efficiently learn its parameters.

Optimal map learning An estimation f of the barycentric projection of a regularized plan π ε which generalizes outside the support of µ can be obtained by learning a deep neural network which minimizes the following objective w.r.t.

the parameters θ, DISPLAYFORM1 When d(x, y) = ||x − y|| 2 , the last term in Eq. FORMULA0 is simply a weighted sum of squared errors, with possibly an infinite number of terms whenever µ or ν are continuous.

We propose to minimize the objective (12) by stochastic gradient descent, which provides the simple Algorithm 2.

The OT problem being symmetric, we can also compute the opposite barycentric projection g w.r.t.

a cost DISPLAYFORM2 However, unless the plan π is induced by a map, the averaging process results in having the image of the source measure byπ only approximately equal to the target measure ν.

Still, when the size of discrete measure is large and the regularization is small, we show in the next paragraph that 1) the barycentric projection of a regularized OT plan is close to the Monge map between the underlying continuous measures (Theorem 2) and 2) the image of the source measure by this barycentric projection should be close to the target measure ν (Corollary 1).Theoretical guarantees As stated earlier, when X = Y and c(x, y) = ||x − y|| 2 2 , Brenier (1991) proved that when the source measure µ is continuous, there exists a solution to the Monge problem (1).

This result was generalized to more general cost functions, see BID47 [Corollary 9 .3] for details.

In that case, the plan π between µ and ν is written as (id, f )#µ where f is the Monge map.

Now considering discrete measures µ n and ν n which converge to µ (continuous) and ν respectively, we have proved in Theorem 1 that π ε n converges weakly to π = (id, f )#µ when (n, ε) → (∞, 0).

The next theorem, proved in the Appendix, shows that the barycentric projectionπ ε n also converges weakly to the true Monge map between µ and ν, justifying our approach.

Theorem 2.

Let µ be a continuous probability measure on R d , and ν an arbitrary probability measure on R d and c a cost function satisfying BID47 [Corollary 9.3] .

Let µ n = 1 n n i=1 δ xi and ν n = 1 n n j=1 δ yj converging weakly to µ and ν respectively.

Assume that the OT solution π n of Problem (2) between µ n and ν n is unique for all n. Let (ε n ) a sequence of non-negative real numbers converging sufficiently fast to 0 andπ This theorem shows that our estimated barycentric projection is close to an optimal map between the underlying continuous measures for n big and ε small.

The following corollary confirms the intuition that the image of the source measure by this map converges to the underlying target measure.

Corollary 1.

With the same assumptions as above,π εn n #µ n → ν weakly.

In terms of random variables, the last equation states that if X n ∼ µ n and Y ∼ ν, thenπ εn n (X n ) converges in distribution to Y .

BID20 : we use SGD instead of SAG), for several entropy-regularization values.

Learning rates are {5., 20., 20.} and batch sizes {1024, 500, 100} respectively and are taken the same for the dual and semi-dual methods.

These theoretical results show that our estimated Monge map can thus be used to perform domain adaptation by mapping a source dataset to a target dataset, as well as perform generative modeling by mapping a continuous measure to a target discrete dataset.

We demontrate this in the following section.

We start by evaluating the training time of our dual stochastic algorithm 1 against a stochastic semidual approach similar to BID20 .

In the semi-dual approach, one of the dual variable is eliminated and is computed in close form.

However, this computation has O(n) complexity where n is the size of the target measure ν.

We compute the regularized OT objective with both methods on a spectral transfer problem, which is related to the color transfer problem BID39 BID36 , but where images are multispectral, i.e. they share a finer sampling of the light wavelength.

We take two 500 × 500 images from the CAVE dataset BID49 that have 31 spectral bands.

As such, the optimal transport problem is computed on two empirical distributions of 250000 samples in R 31 on which we consider the squared Euclidean ground cost c. The timing evolution of train losses are reported in FIG2 for three different regularization values ε = {0.025, 0.1, 1.}. In the three cases, one can observe that convergence of our proposed dual algorithm is much faster.

We apply here our computation framework on an unsupervised domain adaptation (DA) task, for which optimal transport has shown to perform well on small scale datasets BID13 BID35 BID11 .

This restriction is mainly due to the fact that those works only consider the primal formulation of the OT problem.

Our goal here is not to compete with the state-of-the-art methods in domain adaptation but to assess that our formulation allows to scale optimal transport based domain adaptation (OTDA) to large datasets.

OTDA is illustrated in FIG3 and follows two steps: 1) learn an optimal map between the source and target distribution, 2) map the source samples and train a classifier on them in the target domain.

Our formulation also allows to use any differentiable ground cost c while BID13 ) was limited to the squared Euclidean distance.

Datasets We consider the three cross-domain digit image datasets MNIST BID25 , USPS, and SVHN BID33 , which have 10 classes each.

For the adaptation between MNIST and USPS, we use 60000 samples in the MNIST domain and 9298 samples in USPS domain.

MNIST images are resized to the same resolution as USPS ones (16 × 16).

For the adaptation between SVHN and MNIST, we use 73212 samples in the SVHN domain and 60000 samples in the MNIST domain.

MNIST images are zero-padded to reach the same resolution as SVHN (32 × 32) and extended to three channels to match SVHN image sizes.

The labels in the target domain are BID13 .

Source samples are mapped to the target set through the barycentric projectionπ ε .

A classifier is then learned on the mapped source samples.

withheld during the adaptation.

In the experiment, we consider the adaptation in three directions: MNIST → USPS, USPS → MNIST, and SVHN → MNIST.

Our goal is to demonstrate the potential of the proposed method in large-scale settings.

Adaptation performance is evaluated using a 1-nearest neighbor (1-NN) classifier, since it has the advantage of being parameter free and allows better assessment of the quality of the adapted representation, as discussed in BID13 .

In all experiments, we consider the 1-NN classification as a baseline, where labeled neighbors are searched in the source domain and the accuracy is computed on target data.

We compare our approach to previous OTDA methods where an optimal map is obtained through the discrete barycentric projection of either an optimal plan (computed with the network simplex algorithm 1 ) or an entropy-regularized optimal plan (computed with the Sinkhorn algorithm BID14 ), whenever their computation is tractable.

Note that these methods do not provide out-of-sample mapping.

In all experiments, the ground cost c is the squared Euclidean distance and the barycentric projection is computed w.r.t.

that cost.

We learn the Monge map of our proposed approach with either entropy or L2 regularizations.

Regarding the adaptation between SVHN and MNIST, we extract deep features by learning a modified LeNet architecture on the source data and extracting the 100-dimensional features output by the top hidden layer.

Adaptation is performed on those features.

We report for all the methods the best accuracy over the hyperparameters on the target dataset.

While this setting is unrealistic in a practical DA application, it is widely used in the DA community BID27 and our goal is here to investigate the relative performances of large-scale OTDA in a fair setting.

Hyper-parameters and learning rate The value for the regularization parameter is set in {5, 2, 0.9, 0.5, 0.1, 0.05, 0.01}. Adam optimizer with batch size 1000 is used to optimize the network.

The learning rate is varied in {2, 0.9, 0.1, 0.01, 0.001, 0.0001}.

The learned Monge map f in Alg.

2 is parameterized as a neural network with two fully-connected hidden layers (d → 200 → 500 → d) and ReLU activations, and the weights are optimized using the Adam optimizer with learning rate equal to 10 −4 and batch size equal to 1000.

For the Sinkhorn algorithm, regularization value is chosen from {0.01, 0.1, 0.5, 0.9, 2.0, 5.0, 10.0}. Results Results are reported in TAB1 .

In all cases, our proposed approach outperforms previous OTDA algorithms.

On MNIST→USPS, previous OTDA methods perform worse than using directly source labels, whereas our method leads to successful adaptation results with 20% and 10% accuracy points over OT and regularized OT methods respectively.

On USPS→MNIST, all three algorithms lead to successful adaptation results, but our method achieves the highest adaptation results.

Finally, on the challenging large-scale adaptation task SVHN→MNIST, only our method is able to handle the whole datasets, and outperforms the source only results.

Comparing the results between the barycentric projection and estimated Monge map illustrates that learning a parametric mapping provides some kind of regularization, and improves the performance.

Approach Corollary 1 shows that when the support of the discrete measures µ and ν is large and the regularization ε is small, then we have approximatelyπ ε #µ = ν.

This observation motivates the use of our Monge map estimation as a generator between an arbitrary continuous measure µ and a discrete measure ν representing the discrete distribution of some dataset.

We can thus obtain a generative model by first computing regularized OT through Alg.

1 between a Gaussian measure µ and a discrete dataset ν and then compute our generator with Alg.

2.

This requires to have a cost function between the latent variable X ∼ µ and the discrete variable Y ∼ ν.

The property we gain compared to other generative models is that our generator is, at least approximately, an optimal map w.r.t.

this cost.

In our case, the Gaussian is taken with the same dimensionality as the discrete data and the squared Euclidean distance is used as ground cost c.

Permutation-invariant MNIST We preprocess MNIST data by rescaling grayscale values in [−1, 1].

We run Alg.

1 and Alg.

2 where µ is a Gaussian whose mean and covariance are taken equal to the empirical mean and covariance of the preprocessed MNIST dataset; we have observed that this makes the learning easier.

The target discrete measure ν is the preprocessed MNIST dataset.

Permutation invariance means that we consider each grayscale 28 × 28 images as a 784-dimensional vector and do not rely on convolutional architectures.

In Alg.

1 the dual potential u is parameterized as a (d → 1024 → 1024 → 1) fully-connected NN with ReLU activations for each hidden layer, and the L 2 regularization is considered as it produced experimentally less blurring.

The barycentric projection f of Alg.

2 is parameterized as a (d → 1024 → 1024 → d) fully-connected NN with ReLU activation for each hidden layer and a tanh activation on the output layer.

We display some generated samples in FIG4

We proposed two original algorithms that allow for i) large-scale computation of regularized optimal transport ii) learning an optimal map that moves one probability distribution onto another (the so-called Monge map).

To our knowledge, our approach introduces the first tractable algorithms for computing both the regularized OT objective and optimal maps in large-scale or continuous settings.

We believe that these two contributions enable a wider use of optimal transport strategies in machine learning applications.

Notably, we have shown how it can be used in an unsupervised domain adaptation setting, or in generative modeling, where a Monge map acts directly as a generator.

Our consistency results show that our approach is theoretically well-grounded.

An interesting direction for future work is to investigate the corresponding convergence rates of the empirical regularized optimal plans.

We believe this is a very complex problem since technical proofs regarding convergence rates of the empirical OT objective used e.g. in BID45 BID6 BID18 do not extend simply to the optimal transport plans.that we have π n = (id, T n )#µ n .

This also impliesπ n = T n so that (id,π n )#µ n = (id, T n )#µ n .

Hence, the second term in the right-hand side of (18) converges to 0 as a result of the stability of optimal transport BID47 [Theorem 5.20] .

Now, we show that the first term also converges to 0 for ε n converging sufficiently fast to 0.

By definition of the pushforward operator, DISPLAYFORM0 g(x, T n (x))dµ n (x) (19) and we can bound, DISPLAYFORM1 where Y n = (y 1 , · · · , y n ) t and K g is the Lipschitz constant of g. The first inequality follows from g being Lipschitz.

The next equality follows from the discrete close form of the barycentric projection.

The last inequality is obtained through Cauchy-Schwartz.

We can now use the same arguments as in the previous proof.

A convergence result by BID10 shows that there exists positive constants (w.r.t.

ε n ) M cn,µn,νn and λ cn,µn,νn such that, where c n = (c(x 1 , y 1 ), · · · , c(x n , y n )).

The subscript indices indicate the dependences of each constant.

Hence, we see that choosing any (ε n ) such that (21) tends to 0 provides the results.

In particular, we can take ε n = λ cn,µn,νn ln(n 2 ||Y n || 1/2 R n×d ,2 M cn,µn,νn )which suffices to have the convergence of (15) to 0 for Lipschitz function g ∈ C l (R d × R d ).

This proves the weak convergence of (id,π εn n )#µ n to (id, f )#µ.

Proof.

Let h ∈ C b (R d ) a bounded continuous function.

Let g ∈ C b (R d × R d ) defined as g : (x, y) → h(y).

We have, DISPLAYFORM0 which converges to 0 by Theorem (2).

Since f #µ = ν, this proves the corollary.

<|TLDR|>

@highlight

Learning optimal mapping with deepNN between distributions along with theoretical guarantees.