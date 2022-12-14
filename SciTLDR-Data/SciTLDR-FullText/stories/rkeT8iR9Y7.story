Although stochastic gradient descent (SGD) is a driving force behind the recent success of deep learning, our understanding of its dynamics in a high-dimensional parameter space is limited.

In recent years, some researchers have used the stochasticity of minibatch gradients, or the signal-to-noise ratio, to better characterize the learning dynamics of SGD.

Inspired from these work, we here analyze SGD from a geometrical perspective by inspecting the stochasticity of the norms and directions of minibatch gradients.

We propose a model of the directional concentration for minibatch gradients through von Mises-Fisher (VMF) distribution, and show that the directional uniformity of minibatch gradients increases over the course of SGD.

We empirically verify our result using deep convolutional networks and observe a higher correlation between the gradient stochasticity and the proposed directional uniformity than that against the gradient norm stochasticity, suggesting that the directional statistics of minibatch gradients is a major factor behind SGD.

Stochastic gradient descent (SGD) has been a driving force behind the recent success of deep learning.

Despite a series of work on improving SGD by incorporating the second-order information of the objective function BID26 BID21 BID6 BID22 BID7 , SGD is still the most widely used optimization algorithm for training a deep neural network.

The learning dynamics of SGD, however, has not been well characterized beyond that it converges to an extremal point BID1 due to the non-convexity and highdimensionality of a usual objective function used in deep learning.

Gradient stochasticity, or the signal-to-noise ratio (SNR) of the stochastic gradient, has been proposed as a tool for analyzing the learning dynamics of SGD.

BID28 identified two phases in SGD based on this.

In the first phase, "drift phase", the gradient mean is much higher than its standard deviation, during which optimization progresses rapidly.

This drift phase is followed by the "diffusion phase", where SGD behaves similarly to Gaussian noise with very small means.

Similar observations were made by BID18 and BID4 who have also divided the learning dynamics of SGD into two phases.

BID28 have proposed that such phase transition is related to information compression.

Unlike them, we notice that there are two aspects to the gradient stochasticity.

One is the L 2 norm of the minibatch gradient (the norm stochasticity), and the other is the directional balance of minibatch gradients (the directional stochasticity).

SGD converges or terminates when either the norm of the minibatch gradient vanishes to zeros, or when the angles of the minibatch gradients are uniformly distributed and their non-zero norms are close to each other.

That is, the gradient stochasticity, or the SNR of the stochastic gradient, is driven by both of these aspects, and it is necessary for us to investigate not only the holistic SNR but also the SNR of the minibatch gradient norm and that of the minibatch gradient angles.

In this paper, we use a von Mises-Fisher (vMF hereafter) distribution, which is often used in directional statistics BID20 , and its concentration parameter ?? to characterize the directional balance of minibatch gradients and understand the learning dynamics of SGD from the perspective of directional statistics of minibatch gradients.

We prove that SGD increases the direc-tional balance of minibatch gradients.

We empirically verify this with deep convolutional networks with various techniques, including batch normalization BID12 and residual connections BID9 , on MNIST and CIFAR-10 ( BID15 ).

Our empirical investigation further reveals that the proposed directional stochasticity is a major drive behind the gradient stochasticity compared to the norm stochasticity, suggesting the importance of understanding the directional statistics of the stochastic gradient.

Contribution We analyze directional stochasticity of the minibatch gradients via angles as well as the concentration parameter of the vMF distribution.

Especially, we theoretically show that the directional uniformity of the minibatch gradients modeled by the vMF distribution increases as training progresses, and verify this by experiments.

In doing so, we introduce gradient norm stochasticity as the ratio of the standard deviation of the minibatch gradients to their expectation and theoretically and empirically show that this gradient norm stochasticity decreases as the batch size increases.

Related work Most studies about SGD dynamics have been based on two-phase behavior BID28 BID18 BID4 .

BID18 investigated this behavior by considering a shallow neural network with residual connections and assuming the standard normal input distribution.

They showed that SGD-based learning under these setups has two phases; search and convergence phases.

BID28 on the other hand investigated a deep neural network with tanh activation functions, and showed that SGD-based learning has drift and diffusion phases.

They have also proposed that such SNR transition (drift + diffusion) is related to the information transition divided into empirical error minimization and representation compression phases.

However, Saxe et al. (2018) have reported that the information transition is not generally associated with the SNR transition with ReLU BID23 ) activation functions.

BID4 instead looked at the inner product between successive minibatch gradients and presented transient and stationary phases.

Unlike our work here, the experimental verification of the previous work conducted under limited settings -the shallow network BID18 , the specific activation function BID28 , and only MNIST dataset BID28 BID4 -that conform well with their theoretical assumptions.

Moreover, their work does not offer empirical result about the effect of the latest techniques including both batch normalization BID12 layers and residual connections BID9 .

Norms and Angles Unless explicitly stated, a norm refers to L 2 norm.

?? and ??, ?? thus correspond to L 2 norm and the Euclidean inner product on R d , respectively.

We use x n ??? x to indicate that "a random variable x n converges to x in distribution."

Similarly, x n P ??? x means convergence in probability.

An angle ?? between d-dimensional vectors u and v is defined by ?? = Loss functions A loss function of a neural network is written as f (w) = 1 n n i=1 f i (w), where w ??? R d is a trainable parameter.

f i is "a per-example loss function" computed on the i-th data point.

We use I and m to denote a minibatch index set and its batch size, respectively.

Further, we call f I (w) = 1 m i???I f i (w) "a minibatch loss function given I".

In Section 3.1, we use g i (w) and g(w) to denote ?????? w f i (w) and ?????? w f I (w), respectively.

In Section 3.3, the index i is used for the corresponding minibatch index set I i .

For example, the negative gradient of f Ii (w) is written a?? g i (w).

During optimization, we denote a parameter w at the i-th iteration in the t-th epoch as w Figure 1: Characteristics of the vMF distribution in a 2-dimensional space.

100 random samples are drawn from vMF(??, ??) where ?? = (1, 0) and ?? = {0, 5, 50}.on the hypersphere DISPLAYFORM0 Here, the concentration parameter ?? determines how the samples from this distribution are concentrated on the mean direction ?? and C d (??) is constant determined by d and ??.

If ?? is zero, then it is a uniform distribution on the unit hypersphere, and as ?? ??? ???, it becomes a point mass on the unit hypersphere ( Figure 1 ).

The maximum likelihood estimates for ?? and ?? ar?? DISPLAYFORM1 1???r 2 where x i 's are random samples from the vMF distribution andr = n i=1 xi n .

The formula for?? is approximate since the exact computation is intractable BID0 .

It is a usual practice for SGD to use a minibatch gradient??(w) = ?????? w f I (w) instead of a full batch gradient g(w) = ?????? w f (w).

The minibatch index set I is drawn from {1,. . .

,n} randomly.??(w) satisfies E[??(w)] = g(w) and Cov(??(w),??(w)) ??? 1 mn n i=1 g i (w)g i (w) for n m where n is the number of full data points and g i (w) = ?????? w f i (w) BID11 .

As the batch size m increases, the randomness of??(w) decreases.

Hence E ??(w) tends to g(w) , and Var( ??(w) ), which is the variance of the norm of the minibatch gradient, vanishes.

The convergence rate analysis is as the following: Theorem 1.

Let??(w) be a minibatch gradient induced from the minibatch index set I of batch size m from {1, . . .

, n} and suppose ?? = max i,j???{1,...,n} | g i (w), g j (w) |.

Then DISPLAYFORM0 and DISPLAYFORM1 Proof.

See Supplemental A.According to Theorem 1, a large batch size m reduces the variance of ??(w) centered at E ??(w) with convergence rate O(1/m).

We empirically verify this by estimating the gradient norm stochasticity at random points while varying the minibatch size, using a fully-connected neural network (FNN) with MNIST, as shown in FIG2 (a) (see Supplemental E for more details.)

This theorem however only demonstrate that the gradient norm stochasticity is (l.h.s. of FORMULA3 ) is low at random initial points.

It may blow up after SGD updates, since the upper bound (r.h.s. of FORMULA3 ) is inversely proportional to g(w) .

This implies that the learning dynamics and convergence of SGD, measured in terms of the vanishing gradient, i.e., n b i=1?? i (w) ??? 0, is not necessarily explained by the vanishing norms of minibatch gradients, but rather by the balance of the directions of?? i (w)'s, which motivates our investigation of the directional statistics of minibatch gradients.

See FIG2 (b) as an illustration.

In order to investigate the directions of minibatch gradients and how they balance, we start from an angle between two vectors.

First, we analyze an asymptotic behavior of angles between uniformly random unit vectors in a high-dimensional space.

Theorem 2.

Suppose that u and v are mutually independent d-dimensional uniformly random unit vectors.

Then, DISPLAYFORM0 Proof.

See Supplemental B.According to Theorem 2, the angle between two independent uniformly random unit vectors is normally distributed and becomes increasingly more concentrated as d grows FIG3 ).

If SGD iterations indeed drive the directions of minibatch gradients to be uniform, then, at least, the distribution of angles between minibatch gradients and a given uniformly sampled unit vector follows asymptotically DISPLAYFORM1 Figures 3(b) and 3(c) show that the distribution of the angles between minibatch gradients and a given uniformly sampled unit vector converges to an asymptotic distribution (2) after SGD iterations.

Although we could measure the uniformity of minibatch gradients how the angle distribution between minibatch gradients is close to (2), it is not as trivial to compare the distributions as to compare numerical values.

This necessitates another way to measure the uniformity of minibatch gradients.

We draw a density plot ??(u,?? j (w)) ?? j (w)) ) for 3, 000 minibatch gradients (black) at w = w 0 0 (b) and w = w 0 final , with training accuracy of > 99.9%, (c) when u is given.

After SGD iterations, the density of ??(u,?? j (w)) converges to an asymptotic density (red).

The dimension of FNN is 635,200.

To model the uniformity of minibatch gradients, we propose to use the vMF distribution in Definition 1.

The concentration parameter ?? measures how uniformly the directions of unit vectors are distributed.

By Theorem 1, with a large batch size, the norm of minibatch gradient is nearly deterministic, and?? is almost parallel to the direction of full batch gradient.

In other words, ?? measures the concentration of the minibatch gradients directions around the full batch gradient.

The following Lemma 1 introduces the relationship between the norm of averaged unit vectors and ??, the approximate estimator of ??.

Lemma 1.

The approximated estimator of ?? induced from the d-dimensional unit vectors DISPLAYFORM0 Moreover, h(??) and h (??) are strictly increasing and increasing on [0, n b ), respectively.

Proof.

See Supplemental C.1.

DISPLAYFORM1 which is measured from the directions from the current location w to the fixed points p i 's, where h(??) is a function defined in Lemma 1.

Since h(??) is an increasing function, we may focus only on DISPLAYFORM2 pi???w pi???w to see how?? behaves with respect to its argument.

Lemma 2 implies that the estimated directional concentration?? decreases if we move away from w FIG5 ).

In other words,??(w ) <??(w 0 0 ).

DISPLAYFORM3 If all p i 's are not on a single ray from the current location w, then there exists positive number ?? such that DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 Proof.

See Supplemental C.2.We make the connection between the observation above and SGD by first viewing p i 's as local minibatch solutions.

Definition 2.

For a minibatch index set I i , p i (w) = arg min w ???N (w;ri) f Ii (w ) is a local minibatch solution of I i at w, where N (w; r i ) is a neighborhood of radius r i at w. Here, r i is determined by w and I i for p i (w) to exist uniquely.

Under this definition, p i (w) is local minimum of a minibatch loss function f Ii near w.

Then we reasonably expect that the direction of?? i (w) = ?????? w f Ii (w) is similar to that of p i (w) ??? w.

Each epoch of SGD with a learning rate ?? computes a series of w DISPLAYFORM7 . .

, n b } with a large batch size or at the early stage of SGD iterations.

Combining these approximations, ?? i (w DISPLAYFORM8 For example, suppose that t = 0, n b = 3 and ?? = 1, and assume that p i (w DISPLAYFORM9 Hence, we have??(w DISPLAYFORM10 DISPLAYFORM11 for a sufficiently small ?? > 0, then there exists positive number ?? such that DISPLAYFORM12 Proof.

See Supplemental C.3.This Theorem 3 asserts that??(??) decreases even with some perturbation along the averaged direction Without the corollary above, we need to solve p i (w 0 t ) = arg min w???N (w 0 t ;r) f Ii (w) for all i ??? {1, . . .

, n s }, where n s is the number of samples to estimate ??, in order to compute??(w 0 t ).

Corollary 3.1 however implies that we can compute??(w 0 t ) by using?? DISPLAYFORM13 In Practice Although the number of all possible minibatches in each epoch is n b = n m , it is often the case to use n b ??? n/m minibatches at each epoch in practice to go from w 0 t to w 0 t+1 .

Assuming that these n b minibatches were selected uniformly at random, the average of the n b normalized minibatch gradients is the maximum likelihood estimate of ??, just like the average of all n b normalized minibatch gradients.

Thus, we expect with a large n b , DISPLAYFORM14 and that SGD in practice also satisfies??(w

In order to empirically verify our theory on directional statistics of minibatch gradients, we train various types of deep neural networks using SGD and monitor the following metrics for analyzing the learning dynamics of SGD:??? Training loss Figure 5: We show the average?? (black curve) ?? std. (shaded area), as the function of the number of training epochs (in log-log scale) across various batch sizes in MNIST classifications using FNN with fixed learning rate 0.01 and 5 random initializations.

Although?? with the large batch size decreases more smoothly rather than the small batch size, we observe that?? still decreases well with minibatches of size 64.

We did not match the ranges of the y-axes across the plots to emphasize the trend of monotonic decrease.??? Validation loss DISPLAYFORM0 The latter three quantities are statistically estimated using n s = 3, 000 minibatches.

We use?? to denote the ?? estimate.

We train the following types of deep neural networks (Supplemental E):??? FNN: a fully connected network with a single hidden layer ??? DFNN: a fully connected network with three hidden layers ??? CNN: a convolutional network with 14 layers BID16 In the case of the CNN, we also evaluate its variant with skip connections (+Res) BID9 .As it was shown recently by BID27 that batch normalization BID12 improves the smoothness of a loss function in terms of its Hessian, we also test adding batch normalization to each layer right before the ReLU BID23 ) nonlinearity (+BN).

We use MNIST for the FNN, DFNN and their variants, while CIFAR-10 ( BID15 for the CNN and its variants.

Our theory suggests a sufficiently large batch size for verification.

We empirically analyze how large a batch size is needed in Figure 5 .

From these plots,?? decreases monotonically regardless of the minibatch size, but the variance over multiple training runs is much smaller with a larger minibatch size.

We thus decide to use a practical size of 64.

With this fixed minibatch size, we use a fixed learning rate of 0.01, which allows us to achieve the training accuracy of > 99.9% for every training run in our experiments.

We repeat each setup five times starting from different random initial parameters and report both the mean and standard deviation.

FNN and DFNN We first observe that?? decreases over training regardless of the network's depth in FIG12 (a,b).

We however also notice that?? decrease monotonically with the FNN, but less so with its deeper variant (DFNN).

We conjecture this is due to the less-smooth loss landscape of a deep neural network.

This difference between FNN and DFNN however almost entirely vanishes when batch normalization (+BN) is applied ( FIG12 (e,f)).

This was expected as batch normalization is known to make the loss function behave better, and our theory assumes a smooth objective function.

CNN The CNN is substantially deeper than either FNN or DFNN and is trained on a substantially more difficult problem of CIFAR-10.

In other words, the assumptions underlying our theory may not hold as well.

Nevertheless, as shown in FIG12 Effect of +BN and +Res Based on our observations that the uniformity of minibatch gradients increases monotonically, when a deep neural network is equipped with residual connection (+Res) and trained with batch normalization (+BN), we conjecture that the loss function induced from these two techniques better satisfies the assumptions underlying our theoretical analysis, such as its well-behavedness.

This conjecture is supported by for instance BID27 , who demonstrated batch normalization guarantees the boundedness of Hessian, and BID25 , who showed residual connections eliminate some singularities of Hessian.?? near the end of training The minimum average?? of DFNN+BN, which has 1, 920, 000 parameters, is 71, 009.20, that of FNN+BN, which has 636, 800 parameters, is 23, 059.16, and that of CNN+BN+Res, which has 207, 152 parameters, is 20, 320.43.

These average?? are within a constant multiple of estimated ?? using 3,000 samples from the vMF distribution with true ?? = 0 (35, 075.99 with 1, 920, 000 dimensions, 11, 621.63 with 636, 800 dimensions, and 3, 781.04 with 207, 152 dimensions.)

This implies that we cannot say that the underlying directional distribution of minibatch gradients in all these cases at the end of training is not close to uniform BID5 .

For more detailed analysis, see Supplementary F.

The gradient stochasticity (GS) was used by BID28 as a main metric for identifying two phases of SGD learning in deep neural networks.

This quantity includes both the gradient norm stochasticity (GNS) and the directional uniformity ??, implying that either or both of GNS and ?? could drive the gradient stochasticity.

We thus investigate the relationship among these three quantities as well as training and validation losses.

We focus on CNN, CNN+BN and CNN+Res+BN trained on CIFAR-10.

and directional uniformity??.

We normalized each quantity by its maximum value over training for easier comparison on a single plot.

In all the cases, SNR (orange) and?? (red) are almost entirely correlated with each other, while normSNR is less correlated. (Second row) We further verify this by illustrating SNR-?? scatter plots (red) and SNR-normSNR scatter plots (blue) in log-log scales.

These plots suggest that the SNR is largely driven by the directional uniformity.

From FIG13 (First row), it is clear that the proposed metric of directional uniformity?? correlates better with the gradient stochasticity than the gradient norm stochasticity does.

This was especially prominent during the early stage of learning, suggesting that the directional statistics of minibatch gradients is a major explanatory factor behind the learning dynamics of SGD.

This difference in correlations is much more apparent from the scatter plots in FIG13 (Second row).

We show these plots created from other four training runs per setup in Supplemental G.

Stochasticity of gradients is a key to understanding the learning dynamics of SGD BID28 and has been pointed out as a factor behind the success of SGD (see, e.g., BID17 BID14 .

In this paper, we provide a theoretical framework using von Mises-Fisher distribution, under which the directional stochasticity of minibatch gradients can be estimated and analyzed, and show that the directional uniformity increases over the course of SGD.

Through the extensive empirical evaluation, we have observed that the directional uniformity indeed improves over the course of training a deep neural network, and that its trend is monotonic when batch normalization and skip connections were used.

Furthermore, we demonstrated that the stochasticity of minibatch gradients is largely determined by the directional stochasticity rather than the gradient norm stochasticity.

Our work in this paper suggests two major research directions for the future.

First, our analysis has focused on the aspect of optimization, and it is an open question how the directional uniformity relates to the generalization error although handling the stochasticity of gradients has improved SGD BID24 BID11 BID29 BID13 .

Second, we have focused on passive analysis of SGD using the directional statistics of minibatch gradients, but it is not unreasonable to suspect that SGD could be improved by explicitly taking into account the directional statistics of minibatch gradients during optimization.

In proving Theorem 1, we use Lemma A.1.

Define selector random variables BID11 as below: DISPLAYFORM0 Then we have?? DISPLAYFORM1 Lemma A.1.

Let??(w) be a minibatch gradient induced from the minibatch index set I with batch size m from {1, . . .

, n}. Then DISPLAYFORM2 where ?? = max i,j???{1,...,n} | g i (w), g j (w) |.

DISPLAYFORM3 Theorem 1.

Let??(w) be a minibatch gradient induced from the minibatch index set I of batch size m from {1, . . .

, n} and suppose ?? = max i,j???{1,...,n} | g i (w), g j (w) |.

Then DISPLAYFORM4 and DISPLAYFORM5 Hence, Var( ??(w) ) DISPLAYFORM6

2 ??? E ??(w) 2 .

From the second inequality and Lemma A.1, DISPLAYFORM0 .

DISPLAYFORM1

For proofs, Slutsky's theorem and delta method are key results to describe limiting behaviors of random variables in distributional sense.

Theorem B.1. (Slutsky's theorem, BID3 ) Let {x n }, {y n } be a sequence of random variables that satisfies x n ??? x and y n P ??? ?? when n goes to infinity and ?? is constant.

Then x n y n ??? cx Theorem B.2. (Delta method, BID3 ) Let y n be a sequence of random variables that satisfies ??? n(y n ??? ??) ??? N (0, ?? 2 ).

For a given smooth function f : R ??? R, suppose that f (??) exists and is not 0 where f is a derivative.

Then DISPLAYFORM0 Lemma B.1.

Suppose that u and v are mutually independent d-dimensional uniformly random unit vectors.

Then, DISPLAYFORM1 Proof.

Note that d-dimensional uniformly random unit vectors u can be generated by normalization of d-dimensional multivariate standard normal random vectors x ??? N (0, DISPLAYFORM2 Suppose that two independent uniformly random unit vector u and v are generated by two indepen- DISPLAYFORM3 By SLLN, we have DISPLAYFORM4 Since almost sure convergence implies convergence in probability, DISPLAYFORM5 Therefore, by Theorem B.1 (Slutsky's theorem), DISPLAYFORM6 Theorem 2.

Suppose that u and v are mutually independent d-dimensional uniformly random unit vectors.

Then, DISPLAYFORM7 Proof.

Suppose that ?? = 0, ?? = 1, and f (??) = DISPLAYFORM8 Moreover, h(??) and h (??) are strict increasing and increasing on [0, n b ), respectively.

(1 ???r 2 ) 2 and its numerator is always positive for d > 2.

When d = 2, DISPLAYFORM0 (1 ???r 2 ) 2 > 0.

So?? increases asr increases.

The Lipschitz continuity of h(??) directly comes from the continuity of DISPLAYFORM1 Recall that any continuous function on the compact interval [0, n b (1 ??? )] is bounded.

Hence the derivative of?? with respect to u is bounded.

This implies the Lipschitz continuity of h(??).h(??) is strictly increasing sincer = u n b. Further, DISPLAYFORM2 If all p i 's are not on a single ray from the current location w, then there exists positive number ?? such that DISPLAYFORM3 Proof.

Without loss of generality, we regard w as the origin.

DISPLAYFORM4 .

Therefore, we only need to show DISPLAYFORM5 we have DISPLAYFORM6 Note that p j (0) = p j and x j = 1.

We have DISPLAYFORM7 Since the equality holds when u, x j 2 = u 2 x j 2 for all j, we have strict inequality when all p i 's are not located on a single ray from the origin.

The proof of Theorem 3 is very similar to that of Lemma 2.

DISPLAYFORM0 for sufficiently small ?? > 0, then there exists positive number ?? such that DISPLAYFORM1 for all ??? (0, ??].Proof.

We regard w 0 t as the origin 0.

For simplicity, write DISPLAYFORM2 Now we differentiatef ( ) with respect to , that is, DISPLAYFORM3 Recall thatp j (0) = p j .

Rewrite pj pj = x j and use f (0) in the proof of Lemma 2 DISPLAYFORM4 Since f (0) < 0 by the proof of Lemma 2, DISPLAYFORM5 By using x j = 1 and applying the Cauchy inequality, DISPLAYFORM6 Define r = min j p j .

If ) where ?? is a learning rate.

To prove Corollary 3.1, we need to show??(w 0 t+1 ) <??(w 0 t ) which is equivalent to DISPLAYFORM7 DISPLAYFORM8 Since DISPLAYFORM9 is Lipschitz continuous on R BID2 .

If the batch size is sufficiently large and the learning rate ?? is sufficiently small, ?? i (w DISPLAYFORM10 If we denote ?? ?? as , we can convert (8) to (9).

DISPLAYFORM11 Since both w DISPLAYFORM12 where ?? max (A) and ?? min (A) are maximal and minimal singular values of A, respectively.

If A is positive-definite matrix, then DISPLAYFORM13 Here ?? max (A) and ?? min (A) are maximal and minimal eigenvalues of A, respectively.

Lemma D.1.

If the condition number of the positive definite Hessian matrix of f Ii at a local minibatch solution p i , denoted by H i = ??? w 2 f Ii (p i ), is close to 1 (well-conditioned), then the direction to p i from w is approximately parallel to its negative gradient at w. That is, for all w ??? R, DISPLAYFORM14 Proof.

By the second order Taylor expansion, DISPLAYFORM15 Then, we only need to show DISPLAYFORM16 Since H i is positive definite, we can diagonalize it as H i = P i ?? i P i where P i is an orthonormal transition matrix for H i .

DISPLAYFORM17 for sufficiently small ??.

This implies (6) since DISPLAYFORM18 .Then we can apply Theorem 3 and??(w DISPLAYFORM19 where h(??) is increasing and Lipschitz continuous(Lemma 1).

By Lemma D.1, we have DISPLAYFORM20 t ) where rhs is bounded by ??.

Hence, Lipschitz continuity of h(??) implies that DISPLAYFORM21 Since t is arbitrary, we can apply this for all w ??? R including w 0 t+1 .

For all cases, their weighted layers do not have biases, and dropout BID30 is not applied.

We use Xavier initializations BID8 and cross entropy loss functions for all experiments.

FNN The FNN is a fully connected network with a single hidden layer.

It has 800 hidden units with ReLU BID23 activations and a softmax output layer.

DFNN The DFNN is a fully connected network with three hidden layers.

It has 800 hidden units with ReLU activations in each hidden layers and a softmax output layer.

CNN The network architecture of CNN is similar to the network introduced in BID10 as a CIFAR-10 plain network.

The first layer is 3 ?? 3 convolution layer and the number of output filters are 16.

After that, we stack of {4, 4, 3, 1} layers with 3 ?? 3 convolutions on the feature maps of sizes {32, 16, 8, 4} and the numbers of filters {16, 32, 64, 128}, respectively.

The subsampling is performed with a stride of 2.

All convolution layers are activated by ReLU and the convolution part ends with a global average pooling BID19 , a 10-way fully-conneted layers, and softmax.

Note that there are 14 stacked weighted layers.+BN We apply batch normalization right before the ReLU activations on all hidden layers.+Res The identity skip connections are added after every two convolution layers before ReLU nonlinearity (After batch normalization, if it is applied on it.).

We concatenate zero padding slices backwards when the number of filters increases.

We use neither data augmentations nor preprocessings except scaling pixel values into [0, 1] both MNIST and CIFAR-10.

In the case of CIFAR-10, for validation, we randomly choose 5000 images out of 50000 training images.

Figure 8: We show?? estimated from {1, 000 (black), 2, 000 (blue), 3, 000 (red)} random samples of the vMF distribution with underlying true ?? in 10, 000-dimensional space, as the function of ?? (in log-log scale except 0).

For large ??, it is well-estimated by?? regardless of sample sizes.

When the true ?? approaches 0, we need a larger sample size to more accurately estimate this.

635, 200 20, 111.90 ?? 13.04 14, 196.89 ?? 14.91 11, 607.39 ?? 9.27 FNN+BN 636, 800 20, 157.57 ?? 14.06 14, 259.83 ?? 16.38 11, 621.63 ?? 6.83 DFNN 1, 915, 200 60, 619.02 ?? 13.49 42, 849.86 ?? 18.90 34, 983.31 ?? 15.62 DFNN+BN 1, 920, 000 60, 789.84 ?? 17.93 42, 958.71 ?? 25.61 35, 075.99 ?? 12.39 F SOME NOTES ABOUT THE ?? ESTIMATE We point out that, for a small ??, the absolute value of?? is not a precise indicator of the uniformity due to its dependence on the dimensionality, as was investigated earlier by BID5 .

In order to verify this claim, we run some simulations.

First, we vary the number of samples and the true underlying ?? with the fixed dimensionality (Unfortunately, we could not easily go over 10, 000 dimensions due to the difficulty in sampling from the vMF distribution with positive ??.).

We draw {1, 000, 2, 000, 3, 000} random samples from the vMF distribution with the designated ??.

We compute?? from these samples.

As can be seen from Figure 8 , the?? approaches the true ?? from above as the number of samples increases.

When the true ?? is large, the estimation error rapidly becomes zero as the number of samples approaches 3, 000.

When the true ?? is low, however, the gap does not narrow completely even with 3, 000 samples.

While fixing the true ?? to 0 and the number of samples to {1, 000, 2, 000, 3, 000}, we vary the dimensionality to empirically investigate the??.

We choose to use 3, 000 samples to be consistent with our experiments in this paper.

We run five simulations each and report both mean and standard deviation TAB1 .We clearly observe the trend of increasing??'s with respect to the dimensions.

This suggests that we should not compare the absolute values of??'s across different network architectures due to the differences in the number of parameters.

This agrees well with BID5 which empirically showed that the threshold for rejecting the null hypothesis of ?? = p by using?? where p is a fixed value grows with respect to the dimensions.

We show plots from other four training runs in FIG12 .

For all runs, the curves of GS (inverse of SNR) and?? are strongly correlated while GNS (inverse of normSNR) is less correlated to GS.

Figure 9: (a,c,e) We plot the evolution of the training loss (Train loss), validation loss (Valid loss), inverse of gradient stochasticity (SNR), inverse of gradient norm stochasticity (normSNR) and directional uniformity ??.

We normalized each quantity by its maximum value over training for easier comparison on a single plot.

In all the cases, SNR (orange) and?? (red) are almost entirely correlated with each other, while normSNR is less correlated. (b,d,f) We further verify this by illustrating SNR-?? scatter plots (red) and SNR-normSNR scatter plots (blue) in log-log scales.

These plots suggest that the SNR is largely driven by the directional uniformity.

Figure 10: (a,c,e) We plot the evolution of the training loss (Train loss), validation loss (Valid loss), inverse of gradient stochasticity (SNR), inverse of gradient norm stochasticity (normSNR) and directional uniformity ??.

We normalized each quantity by its maximum value over training for easier comparison on a single plot.

In all the cases, SNR (orange) and?? (red) are almost entirely correlated with each other, while normSNR is less correlated. (b,d,f) We further verify this by illustrating SNR-?? scatter plots (red) and SNR-normSNR scatter plots (blue) in log-log scales.

These plots suggest that the SNR is largely driven by the directional uniformity.

Under review as a conference paper at ICLR 2019 Figure 11: (a,c,e) We plot the evolution of the training loss (Train loss), validation loss (Valid loss), inverse of gradient stochasticity (SNR), inverse of gradient norm stochasticity (normSNR) and directional uniformity??.

We normalized each quantity by its maximum value over training for easier comparison on a single plot.

In all the cases, SNR (orange) and ?? (red) are almost entirely correlated with each other, while normSNR is less correlated. (b,d,f) We further verify this by illustrating SNR-?? scatter plots (red) and SNR-normSNR scatter plots (blue) in log-log scales.

These plots suggest that the SNR is largely driven by the directional uniformity.

Figure 12: (a,c,e) We plot the evolution of the training loss (Train loss), validation loss (Valid loss), inverse of gradient stochasticity (SNR), inverse of gradient norm stochasticity (normSNR) and directional uniformity ??.

We normalized each quantity by its maximum value over training for easier comparison on a single plot.

In all the cases, SNR (orange) and?? (red) are almost entirely correlated with each other, while normSNR is less correlated. (b,d,f) We further verify this by illustrating SNR-?? scatter plots (red) and SNR-normSNR scatter plots (blue) in log-log scales.

These plots suggest that the SNR is largely driven by the directional uniformity.

@highlight

One of theoretical issues in deep learning