Hyperparameter tuning is one of the most time-consuming workloads in deep learning.

State-of-the-art optimizers, such as AdaGrad, RMSProp and Adam, reduce this labor by adaptively tuning an individual learning rate for each variable.

Recently researchers have shown renewed interest in simpler methods like momentum SGD as they may yield better results.

Motivated by this trend, we ask: can simple adaptive methods, based on SGD perform as well or better?

We revisit the momentum SGD algorithm and show that hand-tuning a single learning rate and momentum makes it competitive with Adam.

We then analyze its robustness to learning rate misspecification and objective curvature variation.

Based on these insights, we design YellowFin, an automatic tuner for momentum and learning rate in SGD.

YellowFin optionally uses a negative-feedback loop to compensate for the momentum dynamics in asynchronous settings on the fly.

We empirically show YellowFin can converge in fewer iterations than Adam on ResNets and LSTMs for image recognition, language modeling and constituency parsing, with a speedup of up to $3.28$x in synchronous and up to $2.69$x in asynchronous settings.

Accelerated forms of stochastic gradient descent (SGD), pioneered by BID0 and BID1 , are the de-facto training algorithms for deep learning.

Their use requires a sane choice for their hyperparameters: typically a learning rate and momentum parameter BID2 .

However, tuning hyperparameters is arguably the most time-consuming part of deep learning, with many papers outlining best tuning practices written BID4 BID6 .

Deep learning researchers have proposed a number of methods to deal with hyperparameter optimization, ranging from grid-search and smart black-box methods BID7 BID8 to adaptive optimizers.

Adaptive optimizers aim to eliminate hyperparameter search by tuning on the fly for a single training run: algorithms like AdaGrad BID9 , RMSProp BID10 and Adam BID11 use the magnitude of gradient elements to tune learning rates individually for each variable and have been largely successful in relieving practitioners of tuning the learning rate.

Recently some researchers have started favoring simple momentum SGD over the previously mentioned adaptive methods BID12 BID13 , often reporting better test scores BID14 .

Motivated by this trend, we ask the question: can simpler adaptive methods, based on momentum SGD perform as well or better?

We empirically show that, with hand-tuned learning rate, Polyak's momentum SGD achieves faster convergence than Adam for a large class of models.

We then formulate the optimization update as a dynamical system and study certain robustness properties of the momentum operator.

Building on our analysis, we design YELLOWFIN, an automatic hyperparameter tuner for momentum SGD.

YELLOWFIN simultaneously tunes the learning rate and momentum on the fly, and can handle the complex dynamics of asynchronous execution.

Specifically:• In Section 2, we show that momentum presents convergence robust to learning rate misspecification and curvature variation in a class of non-convex objectives; this robustness is desirable for deep learning.

They stem from a known but obscure fact: the momentum operator's spectral radius is constant in a large subset of the hyperparameter space.• In Section 3, we use these robustness insights and a simple quadratic model analysis to design YELLOWFIN, an automatic tuner for momentum SGD.

YELLOWFIN uses on-the-fly measurements from the gradients to tune both a single learning rate and momentum.• In Section 3.3, we discuss common stability concerns related to the phenomenon of exploding gradients .

We present a natural extension to our basic tuner, using adaptive gradient clipping, to stabilize training for objectives with exploding gradients.• In Section 4 we present closed-loop YELLOWFIN, suited for asynchronous training.

It uses a novel component for measuring the total momentum in a running system, including any asynchrony-induced momentum, a phenomenon described in BID16 .

This measurement is used in a negative feedback loop to control the value of algorithmic momentum.

We provide a thorough evaluation of the performance and stability of our tuner.

In Section 5, we demonstrate empirically that on ResNets and LSTMs YELLOWFIN can converge in fewer iterations compared to: (i) hand-tuned momentum SGD (up to 1.75x speedup); and (ii) default Adam (0.8x to 3.3x speedup).

Under asynchrony, the closed-loop control architecture speeds up YELLOWFIN, making it up to 2.69x faster than Adam.

Our experiments include runs on 7 different models, randomized over at least 5 different random seeds.

YELLOWFIN is stable and achieves consistent performance: the normalized sample standard deviation of test metrics varies from 0.05% to 0.6%.

We released PyTorch and TensorFlow implementations, that can be used as drop-in replacements for any optimizer.

YELLOWFIN has also been implemented in other various packages.

Its large-scale deployment in industry has taught us important lessons about stability; we discuss those challenges and our solution in Section 3.3.

We conclude with related work and discussion in Section 6 and 7.

In this section we identify the main technical insights guiding the design of YELLOWFIN.

We show that momentum presents convergence robust to learning rate misspecification and curvature variation for a class of non-convex objectives; these robustness properties are desirable for deep learning.

We aim to minimize some objective f (x).

In machine learning, x is referred to as the model and the objective is some loss function.

A low loss implies a well-fit model.

Gradient descent-based procedures use the gradient of the objective function, rf (x), to update the model iteratively.

Polyak's momentum gradient descent update BID0 ) is given by DISPLAYFORM0 where ↵ denotes the learning rate and µ the value of momentum used.

Momentum's main appeal is its established ability to accelerate convergence BID0 .

On a strongly convex smooth function with condition number , the optimal convergence rate of gradient descent (µ = 0) is O(  1 +1 ) BID17 .

On the other hand, for certain classes of strongly convex and smooth functions, like quadratics, the optimal momentum value, DISPLAYFORM1 yields the optimal accelerated rate O( DISPLAYFORM2 ).

1 This is the smallest value of momentum that ensures the same rate of convergence along all directions.

This fact is often hidden away in proofs.

We shed light on some of its previously unknown implications in Section 2.2.

In this section we analyze the dynamics of momentum on a simple class of one dimensional, nonconvex objectives.

We first introduce the notion of generalized curvature and use it to describe the momentum operator.

Then we discuss some properties of the momentum operator.

Definition 1 (Generalized curvature).

The derivative of f (x) : R !

R, can be written as DISPLAYFORM0 is the global minimum of f (x).

We call h(x) the generalized curvature.

The generalized curvature describes, in some sense, curvature with respect to the optimum, x ⇤ .

For quadratic objectives, it coincides with the standard definition of curvature, and is the sole quantity related to the objective that influences the dynamics of gradient descent.

For example, the contraction of a gradient descent step is 1 ↵h(x t ).

Let A t denote the momentum operator at time t. Using a state-space augmentation, we can express the momentum update as DISPLAYFORM1 Lemma 2 (Robustness of the momentum operator).

As proven in Appendix A, if the generalized curvature, h, and hyperparameters ↵, µ are in the robust region, that is: DISPLAYFORM2 then the spectral radius of the momentum operator only depends on momentum: ⇢(A t ) = p µ.We explain Lemma 2 as robust convergence with respect to learning rate and to variations in curvature.

Momentum is robust to learning rate misspecification For a one dimensional strongly convex quadratic objective, we get h(x) = h for all x and Lemma 2 suggests that ⇢(A t ) = p µ as long as DISPLAYFORM3 In FIG1 , we plot ⇢(A t ) for different ↵ and µ. As we increase the value of momentum, the optimal rate of convergence p µ is achieved by an ever-widening range of learning rates.

Furthermore, for objectives with large condition number, higher values of momentum are both faster and more robust.

This property influences the design of our tuner: as long as the learning rate satisfies (6), we are in the robust region and expect the same asymptotics, e.g. a convergence rate of p µ for quadratics, independent of the learning rate.

Having established that, we can just focus on optimally tuning momentum.

Momentum is robust to curvature variation As discussed in Section 2.1, the intuition hidden in classic results is that for strongly convex smooth objectives, the momentum value in (2) guarantees the same rate of convergence along all directions.

We extend this intuition to certain non-convex functions.

Lemma 2 guarantees a constant, time-homogeneous spectral radius for the momentum operators (A t ) t if (5) is satisfied at every step.

This motivates an extension of the condition number.

Definition 3 (Generalized condition number).

We define the generalized condition number (GCN) of a scalar function, f (x) : R !

R, to be the dynamic range of its generalized curvature, h(x): DISPLAYFORM4 The GCN captures variations in generalized curvature along a scalar slice.

From Lemma 2 we get DISPLAYFORM5 as the optimal hyperparameters.

Specifically, µ ⇤ is the smallest momentum value that guarantees a homogeneous spectral radius of p µ ⇤ for all (A t ) t .

The spectral radius of an operator describes its asymptotic convergence behavior.

However, the product of a sequence of operators A t · · · A 1 all with spectral radius p µ does not always follow the asymptotics of p µ t .

In other words, we do not provide a convergence rate guarantee.

Instead, we provide evidence in support of this intuition.

For example, the non-convex objective in FIG2 (a), composed of two quadratics with curvatures 1 and 1000, has a GCN of 1000.

Using the tuning rule of FORMULA8 , and running the momentum algorithm FIG2 ) yields a practically constant rate of convergence throughout.

In FIG2 we demonstrate that for an LSTM, the majority of variables follow a p µ convergence rate.

This property influences the design of our tuner: in the next section we use the tuning rules of (8) in YELLOWFIN, generalized appropriately to handle SGD noise.

In this section we describe YELLOWFIN, our tuner for momentum SGD.

We introduce a noisy quadratic model and work on a local quadratic approximation of f (x) to apply the tuning rule of FORMULA8 to SGD on an arbitrary objective.

YELLOWFIN is our implementation of that rule.

Noisy quadratic model We consider minimizing the one-dimensional quadratic DISPLAYFORM0 The objective is defined as the average of n component functions, f i .

This is a common model for SGD, where we use only a single data point (or a mini-batch) drawn uniformly at random, DISPLAYFORM1 to compute a noisy gradient, rf St (x), for step t. Here, C = 1 2n P i hc 2 i denotes the gradient variance.

This scalar model is sufficient to study an arbitrary local quadratic approximation: optimization on quadratics decomposes trivially into scalar problems along the principal eigenvectors of the Hessian.

Next we get an exact expression for the mean square error after running momentum SGD on a scalar quadratic for t steps.

Lemma 4.

Let f (x) be defined as in (9), x 1 = x 0 and x t follow the momentum update (1) with stochastic gradients rf St (x t 1 ) for t 2.

Let e 1 = [1, 0] T , the expectation of squared distance to the optimum x DISPLAYFORM2 where the first and second term correspond to squared bias and variance, and their corresponding momentum dynamics are captured by operators DISPLAYFORM3 Even though it is possible to numerically work on (10) directly, we use a scalar, asymptotic surrogate based in (12) on the spectral radii of operators to simplify analysis and expose insights.

This decision is supported by our findings in Section 2: the spectral radii can capture empirical convergence speed.

DISPLAYFORM4 One of our design decisions for YELLOWFIN is to always work in the robust region of Lemma 2.

We know that this implies a spectral radius p µ of the momentum operator, A, for the bias.

Lemma 5shows that under the exact same condition, the variance operator B has spectral radius µ. Lemma 5.

The spectral radius of the variance operator, DISPLAYFORM5 As a result, the surrogate objective of FORMULA0 , takes the following form in the robust region.

DISPLAYFORM6 We use this surrogate objective to extract a noisy tuning rule for YELLOWFIN.

Based on the surrogate in (13), we present YELLOWFIN (Algorithm 1).

Let D denote an estimate of the current model's distance to a local quadratic approximation's minimum, and C denote an estimate for gradient variance.

Also, let h max and h min denote estimates for the largest and smallest generalized curvature respectively.

The extremal curvatures h min and h max are meant to capture both curvature variation along different directions (like the classic condition number) and also variation that occurs as the landscape evolves.

At each iteration, we solve the following SINGLESTEP problem.(SINGLESTEP) DISPLAYFORM0 SINGLESTEP minimizes the surrogate for the expected squared distance from the optimum of a local quadratic approximation (13) after a single step (t = 1), while keeping all directions in the robust region (5).

This is the SGD version of the noiseless tuning rule in (8).

It can be solved in closed form; we refer to Appendix D for discussion on the closed form solution.

YELLOWFIN uses functions CURVATURERANGE, VARIANCE and DISTANCE to measure quantities h max , h min , C and D respectively.

These measurement functions can be designed in different ways.

We present the implementations we used for our experiments, based completely on gradients, in Section 3.2.

This section describes our implementation of the measurement oracles used by YELLOWFIN: CUR-VATURERANGE, VARIANCE, and DISTANCE.

We design the measurement functions with the assumption of a negative log-probability objective; this is in line with typical losses in machine learning, e.g. cross-entropy for neural nets and maximum likelihood estimation in general.

Under this assumption, the Fisher information matrix-i.e.

the expected outer product of noisy gradients-equals the Hessian of the objective BID19 .

This allows for measurements purely from minibatch gradients with overhead linear to model dimensionality.

These implementations are not guaranteed to give accurate measurements.

Nonetheless, their use in our experiments in Section 5 shows that they are sufficient for YELLOWFIN to outperform the state of the art on a variety of objectives.

We also refer to Appendix E for implementation details on zero-debias BID11 , slow start (Schaul et al., 2013) and smoothing for curvature range estimation.

Algorithm 4 Distance to opt.

DISPLAYFORM0 Curvature range Let g t be a noisy gradient, we estimate the range of curvatures in Algorithm 2.

We notice that the outer product of g t and g T t has an eigenvalue h t = kg t k 2 with eigenvector g t .

Thus under our negative log-likelihood assumption, we use h t to approximate the curvature of Hessian along gradient direction g t .

Specifically, we maintain h min and h max as running averages of extreme curvature h min,t and h max,t , from a sliding window of width 20.

Table 1 : German-English translation validation performance using convolutional seq-to-seq learning.

Distance to optimum In Algorithm 4, we estimate the distance to the optimum of the local quadratic approximation.

Inspired by the fact that krf (x )k  kH kkx x ?

k for a quadratic f (x) with Hessian H and minimizer x ⇤ , we first maintain h and kgk as running averages of curvature h t and gradient norm kg t k. Then the distance is approximated using kgk/h.

Neural network objectives can involve arbitrary non-linearities, and large Lipschitz constants (Szegedy et al., 2013) .

Furthermore, the process of training them is inherently non-stationary, with the landscape abruptly switching from flat to steep areas.

In particular, the objective functions of RNNs with hidden units can exhibit occasional but very steep slopes .

To deal with this issue, we use adaptive gradient clipping heuristics as a very natural addition to our basic tuner.

It is discussed with extensive details in Appendix F. In Figure 4 , we present an example of an LSTM that exhibits the 'exploding gradient' issue.

The proposed adaptive clipping can stabilize the training process using YELLOWFIN and prevent large catastrophic loss spikes.

We validate the proposed adaptive clipping on the convolutional sequence to sequence learning model BID13 for IWSLT 2014 German-English translation.

The default optimizer BID13 uses learning rate 0.25 and Nesterov's momentum 0.99, diverging to loss overflow due to 'exploding gradient'.

It requires, as in BID13 , strict manually set gradient norm threshold 0.1 to stabilize.

In Table 1 , we can see YellowFin, with adaptive clipping, outperforms the default optimizer using manually set clipping, with 0.84 higher validation BLEU4 after 120 epochs.

Asynchrony is a parallelization technique that avoids synchronization barriers (Niu et al., 2011) .

It yields better hardware efficiency, i.e. faster steps, but can increase the number of iterations to a given metric, i.e. statistical efficiency, as a tradeoff (Zhang and Ré, 2014).

BID16 interpret asynchrony as added momentum dynamics.

We design closed-loop YELLOWFIN, a variant of YELLOWFIN to automatically control algorithmic momentum, compensate for asynchrony and accelerate convergence.

We use the formula in (14) to model the dynamics in the system, where the total momentum, µ T , includes both asynchrony-induced and algorithmic momentum, µ, in (1).

DISPLAYFORM0 We first use FORMULA0 to design an robust estimatorμ T for the value of total momentum at every iteration.

Then we use a simple negative feedback control loop to adjust the value of algorithmic momentum so thatμ T matches the target momentum decided by YELLOWFIN in Algorithm 1.

In Figure 5 , we demonstrate momentum dynamics in an asynchronous training system.

As directly using the target value as algorithmic momentum, YELLOWFIN (middle) presents total momentumμ T strictly larger than the target momentum, due to asynchrony-induced momentum.

Closed-loop YELLOWFIN (right) automatically brings down algorithmic momentum, match measured total momentumμ T to target value and, as we will see, significantly speeds up convergence comparing to YELLOWFIN.

We refer to Appendix G for details on estimatorμ T and Closed-loop YELLOWFIN in Algorithm 5.

In this section, we empirically validate the importance of momentum tuning and evaluate YELLOWFIN in both synchronous (single-node) and asynchronous settings.

In synchronous settings, we first demonstrate that, with hand-tuning, momentum SGD is competitive with Adam, a state-of-the-art et al., 1993) , and constituency parsing on the Wall Street Journal (WSJ) dataset (Choe and Charniak).

We refer to TAB5 in Appendix H for model specifications.

To eliminate influences of a specific random seed, in our synchronous and asynchronous experiments, the training loss and validation metrics are averaged from 3 runs using different random seeds.

We tune Adam and momentum SGD on learning rate grids with prescribed momentum 0.9 for SGD.

We fix the parameters of Algorithm 1 in all experiments, i.e. YELLOWFIN runs without any hand tuning.

We provide full specifications, including the learning rate (grid) and the number of iterations we train on each model in Appendix I. For visualization purposes, we smooth training losses with a uniform window of width 1000.

For Adam and momentum SGD on each model, we pick the configuration achieving the lowest averaged smoothed loss.

To compare two algorithms, we record the lowest smoothed loss achieved by both.

Then the speedup is reported as the ratio of iterations to achieve this loss.

We use this setup to validate our claims.

Momentum SGD is competitive with adaptive methods In TAB2 , we compare tuned momentum SGD and tuned Adam on ResNets with training losses shown in Figure 9 in Appendix J. We can observe that momentum SGD achieves 1.71x and 1.87x speedup to tuned Adam on CIFAR10 and CIFAR100 respectively.

In FIG4 and TAB2 , with the exception of PTB LSTM, momentum SGD also produces better training loss, as well as better validation perplexity in language modeling and validation F1 in parsing.

For the parsing task, we also compare with tuned Vanilla SGD and AdaGrad, which are used in the NLP community.

FIG4 (right) shows that fixed momentum 0.9 can already speedup Vanilla SGD by 2.73x, achieving observably better validation F1.

We refer to Appendix J.2 for further discussion on the importance of momentum adaptivity in YELLOWFIN.YELLOWFIN can match hand-tuned momentum SGD and can outperform hand-tuned Adam In our experiments, YELLOWFIN, without any hand-tuning, yields training loss matching hand-tuned momentum SGD for all the ResNet and LSTM models in FIG4 and 9.

When comparing to tuned Adam in TAB2 , except being slightly slower on PTB LSTM, YELLOWFIN achieves 1.38x to 3.28x speedups in training losses on the other four models.

More importantly, YELLOWFIN consistently shows better validation metrics than tuned Adam in FIG4 .

It demonstrates that YELLOWFIN can match tuned momentum SGD and outperform tuned state-of-the-art adaptive optimizers.

In Appendix J.4, we show YELLOWFIN further speeding up with finer-grain manual learning rate tuning.

In this section, we evaluate closed-loop YELLOWFIN with focus on the number of iterations to reach a certain solution.

To that end, we run 16 asynchronous workers on a single machine and force them to update the model in a round-robin fashion, i.e. the gradient is delayed for 15 iterations.

FIG6 presents training losses on the CIFAR100 ResNet, using YELLOWFIN in Algorithm 1, closedloop YELLOWFIN in Algorithm 5 and Adam with the learning rate achieving the best smoothed loss in Section 5.1.

We can observe closed-loop YELLOWFIN achieves 20.1x speedup to YELLOWFIN, and consequently a 2.69x speedup to Adam.

This demonstrates that (1) closed-loop YELLOWFIN accelerates by reducing algorithmic momentum to compensate for asynchrony and (2) can converge in less iterations than Adam in asynchronous-parallel training.

Many techniques have been proposed on tuning hyperparameters for optimizers.

General hyperparameter tuning approaches, such as random search BID7 and Bayesian approaches BID8 Hutter et al., 2011) , directly applies to optimizer tuning.

As another trend, adaptive methods, including AdaGrad BID9 , RMSProp BID10 and Adam (Chilimbi et al., 2014) , uses per-dimension learning rate.

Schaul et al. (2013) use a noisy quadratic model similar to ours to extract learning rate tuning rule in Vanilla SGD.

However they do not use momentum which is essential in training modern neural networks.

Existing adaptive momentum approach either consider the deterministic setting (Graepel and Schraudolph, 2002; Rehman and Nawi, 2011; Hameed et al., 2016; Swanston et al., 1994; Ampazis and Perantonis, 2000; Qiu et al., 1992) or only analyze stochasticity with O(1/t) learning rate (Leen and Orr, 1994) .

In the contrast, we aim at practical momentum adaptivity for stochastically training neural networks.

We presented YELLOWFIN, the first optimization method that automatically tunes momentum as well as the learning rate of momentum SGD.

YELLOWFIN outperforms the state-of-the-art adaptive optimizers on a large class of models both in synchronous and asynchronous settings.

It estimates statistics purely from the gradients of a running system, and then tunes the hyperparameters of momentum SGD based on noisy, local quadratic approximations.

As future work, we believe that more accurate curvature estimation methods, like the bbprop method (Martens et al., 2012) can further improve YELLOWFIN.

We also believe that our closed-loop momentum control mechanism in Section 4 could accelerate convergence for other adaptive methods in asynchronous-parallel settings.

A PROOF OF LEMMA 2To prove Lemma 2, we first prove a more generalized version in Lemma 6.

By restricting f to be a one dimensional quadratics function, the generalized curvature h t itself is the only eigenvalue.

We can prove Lemma 2 as a straight-forward corollary.

Lemma 6 also implies, in the multiple dimensional correspondence of (4), the spectral radius ⇢(A t ) = p µ if the curvature on all eigenvector directions (eigenvalue) satisfies (5).

Lemma 6.

Let the gradients of a function f be described by DISPLAYFORM0 with H (x t ) 2 R n 7 !

R n⇥n .

Then the momentum update can be expressed as a linear operator: DISPLAYFORM1 where DISPLAYFORM2 .

Now, assume that the following condition holds for all eigenvalues (H ( DISPLAYFORM3 then the spectral radius of A t is controlled by momentum with ⇢(A t ) = p µ.

with M t = (I ↵H t + µI ).

In other words, t satisfied that 2 t t (M t ) + µ = 0 with (M t ) being one eigenvalue of M t .

I.e. DISPLAYFORM0 On the other hand, (17) guarantees that (1 ↵ (HB PROOF OF LEMMA 4We first prove Lemma 7 and Lemma 8 as preparation for the proof of Lemma 4.

After the proof for one dimensional case, we discuss the trivial generalization to multiple dimensional case.

Lemma 7.

Let the h be the curvature of a one dimensional quadratic function f and x t = Ex t .

We assume, without loss of generality, the optimum point of f is x ? = 0.

Then we have the following recurrence DISPLAYFORM1 Proof.

From the recurrence of momentum SGD, we have DISPLAYFORM2 By putting the equation in to matrix form, FORMULA1 is a straight-forward result from unrolling the recurrence for t times.

Note as we set x 1 = x 0 with no uncertainty in momentum SGD, we have DISPLAYFORM3 x t 1 ) with x t being the expectation of x t .

For quadratic function f (x) with curvature h 2 R, We have the following recurrence U DISPLAYFORM4 where DISPLAYFORM5 and DISPLAYFORM6 ) 2 is the variance of gradient on minibatch S t .Proof.

We prove by first deriving the recurrence for U t and V t respectively and combining them in to a matrix form.

For U t , we have DISPLAYFORM7 Again, the term involving rf (x t ) rf St (x t ) cancels in the third equality as a results of FORMULA1 and FORMULA1 can be jointly expressed in the following matrix form DISPLAYFORM8 DISPLAYFORM9 (25) Note the second term in the second equality is zero because x 0 and x 1 are deterministic.

Thus DISPLAYFORM10 According to Lemma 7 and 8, we have E(x DISPLAYFORM11 1 where e 1 2 R n has all zero entries but the first dimension.

Combining these two terms, we prove Lemma 4.

Though the proof here is for one dimensional quadratics, it trivially generalizes to multiple dimensional quadratics.

Specifically, we can decompose the quadratics along the eigenvector directions, and then apply Lemma 4 to each eigenvector direction using the corresponding curvature h (eigenvalue).

By summing quantities in (10) for all eigenvector directions, we can achieve the multiple dimensional correspondence of (10).

Again we first present a proof of a multiple dimensional generalized version of Lemma 5.

The proof of Lemma 5 is a one dimensional special case of Lemma 9.

Lemma 9 also implies that for multiple dimension quadratics, the corresponding spectral radius ⇢(B) = µ if DISPLAYFORM0 on all the eigenvector directions with h being the eigenvalue (curvature).

Lemma 9.

Let H 2 R n⇥n be a symmetric matrix and ⇢(B) be the spectral radius of matrix DISPLAYFORM1 We have ⇢(B) = µ if all eigenvalues (H ) of H satisfies DISPLAYFORM2 Proof.

Let be an eigenvalue of matrix B, it gives det (B I ) = 0 which can be alternatively expressed as DISPLAYFORM3 assuming F is invertible, i.e. + µ 6 = 0, where the blocks in DISPLAYFORM4 with M = I ↵H + µI .

FORMULA1 can be transformed using straight-forward algebra as DISPLAYFORM5 Using similar simplification technique as in (28), we can further simplify into DISPLAYFORM6 being an eigenvalue of symmetric M .

The analytic solution to the equation can be explicitly expressed as DISPLAYFORM7 When the condition in (27) holds, we have (M ) 2 = (1 ↵ (H ) + µ) 2  4µ. One can verify that DISPLAYFORM8 Thus the roots in (31) are conjugate with | | = µ. In conclusion, the condition in (27) can guarantee all the eigenvalues of B has magnitude µ. Thus the spectral radius of B is controlled by µ.

The problem in (14) does not need iterative solver but has an analytical solution.

Substituting only the second constraint, the objective becomes p(x) = x 2 D 2 + (1 x) 4 /h 2 min C with x = p µ 2 [0, 1).

By setting the gradient of p(x) to 0, we can get a cubic equation whose root x = p µ p can be computed in closed form using Vieta's substitution.

As p(x) is uni-modal in [0, 1), the optimizer for FORMULA0 is exactly the maximum of µ p and ( DISPLAYFORM0 , the right hand-side of the first constraint in (14).

In Section 3.2, we discuss estimators for learning rate and momentum tuning in YELLOWFIN.

In our experiment practice, we have identified a few practical implementation details which are important for improving estimators.

Zero-debias is proposed by BID11 , which accelerates the process where exponential average adapts to the level of original quantity in the beginning.

We applied zero-debias to all the exponential average quantities involved in our estimators.

In some LSTM models, we observe that our estimated curvature may decrease quickly along the optimization process.

In order to better estimate extremal curvature h max and h min with fast decreasing trend, we apply zero-debias exponential average on the logarithmic of h max,t and h min,t , instead of directly on h max,t and h min,t .

Except from the above two techniques, we also implemented the slow start heuristic proposed by (Schaul et al., 2013) .

More specifically, we use ↵ = min{↵ t , t · ↵ t /(10 · w)} as our learning rate with w as the size of our sliding window in h max and h min estimation.

It discount the learning rate in the first 10 · w steps and helps to keep the learning rate small in the beginning when the exponential averaged quantities are not accurate enough.

Gradient clipping has been established in literature as a standard-almost necessary-tool for training such objectives Goodfellow et al., 2016; BID13 .

However, the classic tradeoff between adaptivity and stability applies: setting a clipping threshold that is too low can hurt performance; setting it to be high, can compromise stability.

YELLOWFIN, keeps running estimates of extremal gradient magnitude squares, h max and h min in order to estimate a generalized condition number.

We posit that p h max is an ideal gradient norm threshold for adaptive clipping.

In order to ensure robustness to extreme gradient spikes, like the ones in Figure 4 , we also limit the growth rate of the envelope h max in Algorithm 2 as follows: DISPLAYFORM0 Our heuristics follows along the lines of classic recipes like .

However, instead of using the average gradient norm to clip, it uses a running estimate of the maximum norm h max .

In Section 3.3, we saw that adaptive clipping stabilizes the training on objectives that exhibit exploding gradients.

In FIG7 , we demonstrate that the adaptive clipping does not hurt performance on models that do not exhibit instabilities without clipping.

Specifically, for both PTB LSTM and CIFAR10 ResNet, the difference between YELLOWFIN with and without adaptive clipping diminishes quickly.

In Section 4, we briefly discuss the closed-loop momentum control mechanism in closed-loop YELLOWFIN.

In this section, after presenting more preliminaries on asynchrony, we show with details on the mechanism: it measures the dynamics on a running system and controls momentum with a negative feedback loop.

• CIFAR10 ResNet -40k iterations (⇠114 epochs) -Momentum SGD learning rates {0.001, 0.01(best), 0.1, 1.0}, momentum 0.9 -Adam learning rates {0.0001, 0.001(best), 0.01, 0.1}• CIFAR100 ResNet -120k iterations (⇠341 epochs) -Momentum SGD learning rates {0.001, 0.01(best), 0.1, 1.0}, momentum 0.9 -Adam learning rates {0.00001, 0.0001(best), 0.001, 0.01}• PTB LSTM -30k iterations (⇠13 epochs) -Momentum SGD learning rates {0.01, 0.1, 1.0(best), 10.0}, momentum 0.9 -Adam learning rates {0.0001, 0.001(best), 0.01, 0.1}• TS LSTM -⇠21k iterations (50 epochs) -Momentum SGD learning rates {0.05, 0.1, 0.5, 1.0(best), 5.0}, momentum 0.9 -Adam learning rates {0.0005, 0.001, 0.005(best), 0.01, 0.05} -Decrease learning rate by factor 0.97 every epoch for all optimizers, following the design by Karpathy et al. (2015) .•

WSJ LSTM -⇠120k iterations (50 epochs) -Momentum SGD learning rates {0.05, 0.1, 0.5(best), 1.0, 5.0}, momentum 0.9 -Adam learning rates {0.0001, 0.0005, 0.001(best), 0.005, 0.01} -Vanilla SGD learning rates {0.05, 0.1, 0.5, 1.0(best), 5.0} -Adagrad learning rates {0.05, 0.1, 0.5(best), 1.0, 5.0} -Decrease learning rate by factor 0.9 every epochs after 14 epochs for all optimizers, following the design by Choe and Charniak.

J ADDITIONAL EXPERIMENT RESULTS

In Figure 9 , we demonstrate the training loss on CIFAR10 ResNet and CIFAR100 ResNet.

Specifically, YELLOWFIN can match the performance of hand-tuned momentum SGD, and achieves 1.93x and 1.38x speedup comparing to hand-tuned Adam respectively on CIFAR10 and CIFAR100 ResNet.

To further emphasize the importance of momentum adaptivity in YELLOWFIN, we run YF on CIFAR100 ResNet and TS LSTM.

In the experiments, YELLOWFIN tunes the learning rate.

Instead of also using the momentum tuned by YF, we continuously feed prescribed momentum value 0.0 and 0.9 to the underlying momentum SGD optimizer which YF is tuning.

In FIG0 , when comparing to YELLOWFIN with prescribed momentum 0.0 or 0.9, YELLOWFIN with adaptively tuned momentum achieves observably faster convergence on both TS LSTM and CIFAR100 ResNet.

It empirically demonstrates the essential role of momentum adaptivity in YELLOWFIN.

We conduct experiments on PTB LSTM with 16 asynchronous workers using Adam using the same protocol as in Section 5.2.

Fixing the learning rate to the value achieving the lowest smoothed loss in Section 5.1, we sweep the smoothing parameter 1 BID11 of the first order moment estimate in grid { 0.2, 0.0, 0.3, 0.5, 0.7, 0.9}.

1 serves the same role as momentum in SGD and we call it the momentum in Adam.

FIG0 shows tuning momentum for Adam under asynchrony gives measurably better training loss.

This result emphasizes the importance of momentum tuning in asynchronous settings and suggests that state-of-the-art adaptive methods can perform sub-optimally when using prescribed momentum.

As an adaptive tuner, YELLOWFIN does not involve manual tuning.

It can present faster development iterations on model architectures than grid search on optimizer hyperparameters.

In deep learning practice for computer vision and natural language processing, after fixing the model architecture, extensive optimizer tuning (e.g. grid search or random search) can further improve the performance of a model.

A natural question to ask is can we also slightly tune YELLOWFIN to accelerate convergence and improve the model performance.

Specifically, we can manually multiply a positive number, the learning rate factor, to the auto-tuned learning rate in YELLOWFIN to further accelerate.

In this section, we empirically demonstrate the effectiveness of learning rate factor on a 29-layer ResNext (2x64d) (Xie et al., 2016) on CIFAR10 and a Tied LSTM model (Press and Wolf, 2016) with 650 dimensions for word embedding and two hidden units layers on the PTB dataset.

When running YELLOWFIN, we search for the optimal learning rate factor in grid { 1 3 , 0.5, 1, 2(best for ResNext), 3(best for Tied LSTM), 10}. Similarly, we search the same learning rate factor grid for Adam, multiplying the factor to its default learning rate 0.001.

To further strength the performance of Adam as a baseline, we also run it on conventional logarithmic learning rate grid {5e 5 , 1e 4 , 5e 4 , 1e 3 , 5e 3 } for ResNext and {1e 4 , 5e 4 , 1e 3 , 5e 3 , 1e 2 } for Tied LSTM.

We report the best metric from searching the union of learning rate factor grid and logarithmic learning rate grid as searched Adam results.

Empirically, learning factor 1 3 and 1.0 works best for Adam respectively on ResNext and Tied LSTM.As shown in FIG0 , with the searched best learning rate factor, YELLOWFIN can improve validation perplexity on Tied LSTM from 88.7 to 80.5, an improvement of more than 9%.

Similarly, the searched learning rate factor can improve test accuracy from 92.63 to 94.75 on ResNext.

More importantly, we can observe, with learning rate factor search on the two models, YELLOWFIN can achieve better validation metric than the searched Adam results.

It demonstrates that finer-grain learning rate tuning, i.e. the learning rate factor search, can be effectively applied on YELLOWFIN to improve the performance of deep learning models.

@highlight

YellowFin is an SGD based optimizer with both momentum and learning rate adaptivity.

@highlight

Proposes a method to automatically tuning the momentum parameter in momentum SGD methods, which achieves better results and fast convergence speed that state-of-the-art Adam algorithm.