We build a theoretical framework for understanding practical meta-learning methods that enables the integration of sophisticated formalizations of task-similarity with the extensive literature on online convex optimization and sequential prediction algorithms in order to provide within-task performance guarantees.

Our approach improves upon recent analyses of parameter-transfer by enabling the task-similarity to be learned adaptively and by improving transfer-risk bounds in the setting of statistical learning-to-learn.

It also leads to straightforward derivations of average-case regret bounds for efficient algorithms in settings where the task-environment changes dynamically or the tasks share a certain geometric structure.

Meta-learning, or learning-to-learn (LTL) BID26 , has recently re-emerged as an important direction for developing algorithms capable of performing well in multitask learning, changing environments, and federated settings.

By using the data of numerous training tasks, meta-learning algorithms seek to perform well on new, potentially related test tasks without using many samples from them.

Successful modern approaches have also focused on exploiting the capacity of deep neural networks, whether by learning multi-task data representations passed to simple classifiers BID25 or by neural control of the optimization algorithms themselves BID23 .Because of its simplicity and flexibility, a common approach is that of parameter-transfer, in which all tasks use the same class of Θ-parameterized functions f θ : X → Y; usually a shared global model φ ∈ Θ is learned that can then be used to train task-specific parameters.

In gradient-based meta-learning (GBML) BID11 , φ is a metainitialization such that a few stochastic gradient steps on a Preliminary work.

Under review by the International Conference on Machine Learning (ICML).

Do not distribute.

few samples from a new task suffice to learn a good taskspecific model.

GBML is now used in a variety of LTL domains such as vision BID18 BID21 BID17 , federated learning BID7 , and robotics BID0 .

However, its simplicity also raises many practical and theoretical questions concerning what task-relationships it is able to exploit and in which settings it may be expected to succeed.

While theoretical LTL has a long history BID4 BID19 BID22 , there has recently been an effort to understand GBML in particular.

This has naturally lead to online convex optimization (OCO) (Zinkevich, 2003) , either directly BID12 BID16 or via online-to-batch conversion to statistical LTL BID16 BID9 .

These efforts all consider learning a shared initialization of a descent method; BID12 then prove learnability of a metalearning algorithm while BID16 and BID9 give meta-test-time performance guarantees.

However, this line of work has so far considered at most a very restricted, if natural, notion of task-similarity -closeness to a single fixed point in the parameter space.

We introduce a new theoretical framework, Averaged-Regret Upper-Bound Analysis (ARUBA), that enables the derivation of meta-learning algorithms that can provably take advantage of much more sophisticated task-structure.

Expanding significantly upon the work of BID16 , ARUBA treats meta-learning as the online learning of a sequence of losses that each upper bound the regret on a single task.

These bounds frequently have convenient functional forms that are (a) nice enough for us to easily draw on the existing OCO literature and (b) strongly dependent on both the task-data and the meta-initialization, thus encoding task-similarity in a mathematically accessible way.

Using ARUBA we provide new or dramatically improved meta-learning algorithms in the following settings:• Adaptive Meta-Learning: A major drawback of previous work is the reliance on knowing the task-similarity beforehand to set the learning rate BID12 or regularization BID9 , or the use of a suboptimal guess-and-tune approach based on the doubling trick BID16 .

ARUBA yields a simple and efficient gradient-based algorithm that eliminates the need to guess the task-similarity by learning it on-the-fly.• Statistical LTL: ARUBA allows us to leverage powerful results in online-to-batch conversion BID27 BID15 to derive new upper-bounds on the transfer risk when using GBML for statistical LTL BID4 , including fast rates in the number of tasks when the task-similarity is known and fully highprobability guarantees for a class of losses that includes linear regression.

These results improve directly upon the guarantees of BID16 and BID9 for similar or identical GBML algorithms.• LTL in Dynamic Environments: Many practical applications of GBML include settings where the optimal initialization may change over time due to a changing taskenvironment BID0 .

However, current theoretical work on GBML has only considered learning a fixed initialization BID12 BID9 .

ARUBA reduces the problem of meta-learning in changing environments to a dynamic regret-minimization problem, for which there exists a vast array of online algorithms with provable guarantees.• Meta-Learning the Task Geometry:

A recurring theme in parameter-transfer LTL is the idea that certain model weights, such as those encoding a shared representation, are common to all tasks, whereas others, such as those performing a task-specific classification, need to be updated on each one.

However, by simply using a fixed initialization we are forced to re-learn this structure on every task.

Using ARUBA we provide an algorithm that can learn and take advantage of such structure by adaptively determining which directions in parameter-space need to be updated.

We further provide a fully adaptive, per-coordinate variant that may be viewed as an analog for Reptile BID21 of the Meta-SGD modification of MAML BID11 BID18 , which learns a per-coordinate learning rate; in addition to its provable guarantees, our version is more efficient and can be applied to a variety of GBML methods.

In the current paper we provide in Section 2 an introduction to ARUBA and use it to show guarantees for adaptive and statistical LTL.

We defer our theory for meta-learning in dynamic environments and of different task-geometries, as well as our empirical results, to the full version of the paper.

Theoretical Learning-to-Learn: The statistical analysis of LTL as learning over a task-distribution was formalized by BID4 and expanded upon by BID19 .

Recently, several works have built upon this theory to understand modern LTL, either from a PAC-Bayesian perspective BID2 or in the ridge regression setting with a learned kernel BID8 .

However, due to the nature of the data, tasks, and algorithms involved, much effort has been devoted to the online setting, often through the framework of lifelong learning BID22 BID3 BID1 .

The latter work considers a many-task notion of regret similar to our own in order to learn a shared data representations, although our algorithms are significantly more practical.

Very recently, BID6 also developed a more efficient online approach to learning a linear embedding of the data.

However, such work is related to popular shared-representation methods such as ProtoNets BID25 , whereas we consider the parameter-transfer setting of GBML.Gradient-Based Meta-Learning: GBML developed from the model-agnostic meta-learning (MAML) algorithm of BID11 and has been widely used in practice BID18 BID0 BID21 BID14 ).

An expressivity result was shown for MAML by BID10 , proving that the metalearner could approximate any permutation-invariant learning algorithm given enough data and a specific neural network architecture.

Under strong-convexity and smoothness assumptions and using a fixed learning rate, BID12 show that the MAML meta-initialization is learnable, albeit via a somewhat impractical Follow-the-Leader (FTL) method.

In contrast to these efforts, BID16 and BID9 focus on providing finite-sample meta-test-time performance guarantees in the convex setting, the former for the SGD-based Reptile algorithm of BID21 and the latter for a more strongly-regularized variant.

Our work improves upon these analyses by considering the case when the learning rate, a proxy for the task-similarity, is not known beforehand as in BID12 and BID9 but must be learned online; BID16 do consider an unknown task-similarity but use a rough doubling-trick-based approach that considers the absolute deviation of the task-parameters from the meta-initialization and is thus average-case suboptimal and sensitive to outliers.

Furthermore, ARUBA can handle more sophisticated and dynamic notions of task-similarity and in certain settings can provide better statistical guarantees than those of BID16 and BID9 .

Following the setup of BID1 , we consider a sequence of tasks t = 1, . . .

, T ; each task has rounds i = 1, . . .

, m, on each of which we see a loss function DISPLAYFORM0 In the online setting, our goal will be to design algorithms taking actions θ t,i ∈ Θ that result in small task-averaged regret (TAR) BID16 , which averages the within-task regret over t ∈ [T ]: DISPLAYFORM1 This quantity measures within-task performance by dynamically comparing to the best action on individual tasks.

A common approach in this setting is to run an online algorithm, such as online gradient descent (OGD) with learning rate η t > 0 and initialization φ t ∈ Θ, on each task t: DISPLAYFORM2 The meta-learning problem is then reduced to determining which learning rate and initialization to use on each task t. Specific cases of this setup include the Reptile method of BID21 and the algorithms in several recent theoretical analyses BID1 BID16 BID9 .

The observation that enables the results in the current paper is the fact that the online algorithms of interest in few-shot learning and meta-learning often have existing regret guarantees that depend strongly on both the parameters and the data; for example, the withintask regret of OGD for G-Lipschitz convex losses is DISPLAYFORM3 for θ * t the optimal parameter in hindsight.

Whereas more sophisticated adaptive methods for online learning attempt to reduce this dependence on initialization, in our setting each task does not have enough data to do so.

Instead we can observe that if the upper boundR t (φ t , η t ) ≥ R t on the task-t regret is low on average over t ∈ [T ] then the TAR of the actions θ t,i due to running OGD initialized at φ t with learning rate η t at each task t will also be low, i.e. DISPLAYFORM4 Often this upper-boundR t will have a nice functional form; for example, the OGD bound above is jointly convex in the learning rate η t and the initialization φ t .

Then standard OCO results can be applied directly.

While this approach was taken implicitly by BID16 , and indeed is related to earlier work on adaptive bound optimization for online learning BID20 , in this work we make explicit this framework, which we call Averaged-Regret Upper-Bound Analysis (ARUBA), and showcase its usefulness in deriving a variety of new results in both online and batch LTL.

Specifically, our approach will reduce LTL to the online learning of a sequence of regret upper-boundsR 1 (x), . . .

,R T (x), where x parameterizes the within-task algorithms.

The resulting guarantees will then have the generic form DISPLAYFORM5 Thus as T → ∞ the algorithm competes with the best parameterization x, which encodes the task-relatedness through the task-data-dependence ofR t .Algorithm 1: General form of meta-learning algorithm we study.

TASK η,φ corresponds to online mirror descent (OMD) or follow-the-regularized-leader (FTRL) with initialization φ ∈ Θ, learning rate η > 0, and regularization R : Θ → R. META (1) is follow-the-leader (FTL).

META (2) is some OCO algorithm.

Set meta-initialization φ 1 ∈ Θ and learning rate η 1 > 0.

DISPLAYFORM6

Our first result is an adaptive algorithm for a simple notion of task-similarity that serves also to demonstrate how our framework may be applied.

We consider tasks t = 1, . . .

, T whose optimal actions θ * t are close to some unknown global φ * ∈ Θ according to some metric.

For 2 -distance this assumption was made, explicitly or implicitly, by BID12 and BID9 ; BID16 also consider the case of a Bregman divergence B R (θ * t ||φ * ) for 1-strongly-convex R : Θ → R BID5 , with R(·) = B R (θ * t ||φ * ) of the task-parameters; for OCO methods V is proportional to the learning rate or the inverse of the regularization coefficient, which were fixed by BID12 and BID9 .

BID16 instead used the doubling trick to learn the maximum deviation max t B R (θ * t ||φ * ) ≥ V , which is suboptimal and extremely sensitive to outliers.

We first formalize the setting we consider, extensions of which will also be used for later results:Setting 2.1.

Each task t ∈ [T ] has m convex loss functions t,i Θ → R that are G-Lipschitz on average.

Let θ * t ∈ arg min θ∈Θ mt i=1 t,i (θ) be the minimum-norm optimal fixed action for task t.

We will consider variants of Algorithm 1, in which a parameterized OCO method TASK η,φ is run within-task and two OCO methods, META(1) and META (2) , are run in the outer loop to determine the learning rate η > 0 and initialization φ ∈ Θ. We provide the following guarantee: Theorem 2.1.

In Setting 2.1 Algorithm 1 achieves TAR DISPLAYFORM0 where D 2 = max t B R (θ * t ||φ t ) and R T is the regret of META (2) on a sequence f 1 , . . . , f T of functions of form DISPLAYFORM1 Proof Sketch.

The proof follows from the well-known re- x + x. This is nontrivial, as while the functions are convex they are non-Lipschitz near 0.

However, using strongly-convex coupling once more one can show that using the actions of FTL on the modified loss functionsf t (x) = by proving the exp-concavity off t and using the Exponentially-Weighted Online Optimization (EWOO) algorithm of BID13 , which can be implemented efficiently in this single-dimensional case, instead of FTL.

We thus have the following corollary: DISPLAYFORM2 DISPLAYFORM3 e. the mean and squared average deviation of the optimal task parameters, we have an asymptotic per-task regret of V G √ m, which is much better than the minimax-optimal single-task guarantee DG √ m when V D, i.e. when the tasks are on-average close in parameter space.

As in BID16 and assuming a quadratic growth condition on each task, in the full version we extend this result to the case when θ * t is not known and either the last or average within-task iterate is used to perform the meta-updates.

An important motivation for studying LTL via online learning has been to provide batch-setting bounds on the transfer risk BID1 BID9 .

While BID16 provide an in-expectation bound on the expected transfer risk of any low-TAR algorithm, their result cannot exploit the many stronger results in the online-tobatch conversion literature.

Following the classical distribution over task-distributions setup of BID4 , ARUBA yields strong bounds on the expected transfer risk in the general case of convexR, as well as fast rates in the stronglyconvex case using BID15 and high probability bounds for linear regression using BID27 .

Theorem 2.2.

Let convex losses t,i : Θ → [0, 1] be sampled i.i.d.

P t ∼ Q, { t,i } i ∼ P m t for some distribution Q over task distributions P t .

If the losses are given to an algorithm with averaged regret upper-boundR T that on each task runs an algorithm with regret upper-boundR t (s t ) a convex, nonnegative, and B √ m-bounded function of the state s t of the algorithm at the beginning of time t then we have the following bound on the expected transfer risk: 2 , 6αB √ m} αmT log 8 log T δ If the losses satisfy a certain self-bounding property then we have a high probability bound on the transfer risk itself: DISPLAYFORM0 DISPLAYFORM1 T m log 2 δ + 3ρ + 2 m log 2 δ w.p.

1 − δ for some ρ > 0.In the case of a known task-similarity, when we know the expected task-parameter deviation V and can fix the learning rate in Algorithm 1 accordingly, the above result yields DISPLAYFORM2 This can be compared to results of BID9 , where the last term only decreases as T .

277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 Adaptive Gradient-Based Meta-Learning Methods Zinkevich, M. Online convex programming and generalized infinitesimal gradient ascent.

In Proceedings of the 20th International Conference on Machine Learning, 2003.

<|TLDR|>

@highlight

Practical adaptive algorithms for gradient-based meta-learning with provable guarantees.