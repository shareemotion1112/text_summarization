We propose a new algorithm to learn a one-hidden-layer convolutional neural network where both the convolutional weights and the outputs weights are parameters to be learned.

Our algorithm works for a general class of (potentially overlapping) patches, including commonly used structures for computer vision tasks.

Our algorithm draws ideas from (1) isotonic regression for learning neural networks and (2) landscape analysis of non-convex matrix factorization problems.

We believe these findings may inspire further development in designing provable algorithms for learning neural networks and other complex models.

While our focus is theoretical, we also present experiments that illustrate our theoretical findings.

Giving provably efficient algorithms for learning neural networks is a core challenge in machine learning theory.

The case of convolutional architectures has recently attracted much interest due to their many practical applications.

Recently BID2 showed that distributionfree learning of one simple non-overlapping convolutional filter is NP-hard.

A natural open question is whether we can design provably efficient algorithms to learn convolutional neural networks under mild assumptions.

We consider a convolutional neural network of the form f px, w, aq " k ÿ j"1 a j σ`w J P j x˘(1)where w P R r is a shared convolutional filter, a P R k is the second linear layer and P j " r 0 lo omo on pj´1qs I lo omo on r 0 lo omo on d´pj´1qs`r s P R rˆd selects the ppj´1qs`1q-th to ppj´1qs`rq-th coordinates of x with stride s and σ p¨q is the activation function.

Note here that both w and a are unknown vectors to be learned and there may be overlapping patches because the stride size s may be smaller than the filter size r.

Our Contributions We give the first efficient algorithm that can provably learn a convolutional neural network with two unknown layers with commonly used overlapping patches.

Our main result is the following theorem.

Theorem 1.1 (Main Theorem (Informal)).

Suppose s ě t r 2 u`1 and the marginal distribution is symmetric and isotropic.

Then the convolutional neural network defined in equation 1 with piecewise linear activation functions is learnable in polynomial time.

We refer readers to Theorem 3.1 for the precise statement.

Technical Insights Our algorithm is a novel combination of the algorithm for isotonic regression and the landscape analysis of non-convex problems.

First, inspired by recent work on isotonic regression, we extend the idea in BID13 to reduce learning a CNN with piecewise linear activation to learning a convolutional neural network with linear activation (c.f.

Section 4).

Second, we show learning a linear convolutional filter can be reduced to a non-convex matrix factorization problem which admits a provably efficient algorithm based on non-convex geometry BID8 .

Third, in analyzing our algorithm, we present a robust analysis of Convotron algorithm proposed by BID13 , in which we draw connections to the spectral properties of Toeplitz matrices.

We believe these ideas may inspire further development in designing provable learning algorithms for neural networks and other complex models.

Related Work From the point of view of learning theory, it is well known that training is computational infeasible in the worst case BID12 BID2 .

Thus distributional assumptions are needed for efficient learning.

A line of research has focused on analyzing the dynamics of gradient descent conditioned on the input distribution being standard Gaussian BID29 BID28 BID21 BID32 BID2 BID31 BID5 .

Specifically for convolutional nets, existing analyses heavily relied on the analytical formulas which can only be derived if the input is Gaussian and patches are non-overlapping.

Recent work has tried to relax the Gaussian input assumption and the non-overlapping structure for learning convolutional filters.

BID5 showed if the patches are sufficiently close to each other then stochastic gradient descent can recover the true filter.

BID13 proposed a modified iterative algorithm inspired from isotonic regression that gives the first recovery guarantees for learning a filter for commonly used overlapping patches under much weaker assumptions on the distribution.

However, these two analyses only work for learning one unknown convoutional filter.

Moving away from gradient descent, various works have shown positive results for learning general simple fully connected neural networks in polynomial time and sample complexity under certain assumptions using techniques such as kernel methods BID12 BID30 BID10 BID0 and tensor decomposition BID27 BID18 .

The main drawbacks include the shift to improper learning for kernel methods and the knowledge of the probability density function for tensor methods.

In contrast to this, our algorithm is proper and does not assume that the input distribution is known.

Learning a neural network is often formulated as a non-convex problem.

If the objective function satisfies (1) all saddle points and local maxima are strict (i.e., there exists a direction with negative curvature), and (2) all local minima are global (no spurious local minmum), then noise-injected (stochastic) gradient descent BID7 BID19 ) finds a global minimum in polynomial time.

Recent work has studied these properties for the landscape of neural networks BID20 BID3 BID15 BID14 BID22 BID6 BID25 Zhou & Feng, 2017; BID23 BID0 BID9 Zhou & Feng, 2017; BID26 BID4 .

A crucial step in our algorithm is reducing the convolutional neural network learning problem to matrix factorization and using the geometric properties of matrix factorization.

We use bold-faced letters for vectors and matrices.

We use }¨} 2 to denote the Euclidean norm of a finite-dimensional vector.

For a matrix A, we use λ max pAq to denote its eigenvalue and λ min pAq its smallest singular value.

Let Op¨q and Ω p¨q denote standard Big-O and Big-Omega notations, only hiding absolute constants.

In our setting, we have n data points tx i , y i u n i"1 where x i P R d and y P R. We assume the label is generated by a two-layer convolutional neural network with filter size r, stride s and k hidden neurons.

Compactly we can write the formula in the following form: y i " f px i , w˚, a˚q, x i " Z where the prediction function f is defined in equation 1.

To obtain a proper scaling, we let }w˚} 2 }a˚} 2 " σ 1 .

We also define the induced patch matrix as P pxq " rP 1 x . . .

P k xs P R rˆk which will play an important role in our algorithm design.

Our goal is to properly learn this convolutional neural network, i.e., design a polynomial time algorithm which outputs a pair pw, aq that satisfies E x"Z " pf pw, a, xq´f pw˚, a˚, xqq 2 ı ď .

Input: Input distribution Z. Number of iterations: T 1 , T 2 .

Number of samples: T 3 .Step sizes: DISPLAYFORM0 Output: Parameters of the one-hidden-layer CNN: w and a.

In this section we describe our main result.

We first list our main assumptions, followed by the detailed description of our algorithm.

Lastly we state the main theorem which gives the convergence guarantees of our algorithm.

Our first assumption is on the input distribution Z. We assume the input distribution is symmetric, bounded and has identity covariance.

The symmetry assumption is used in BID13 and many learning theory papers BID0 .

The identity covariance assumption is true if the data whitened.

Further, in many architectures, the input of certain layers is assumed having these properties because of the use of batch normalization BID17 or other techniques.

Lastly, the boundedness is a standard regularity assumption to exclude pathological input distributions.

We remark that this assumption considerably weaker than the standard Gaussian input distribution assumption used in Tian FORMULA12 ; BID31 ; Du et al. FORMULA12 , which has the rotational invariant property.

Assumption 3.1 (Input Distribution Assumptions).

We assume the input distribution satisfies the following conditions.• Symmetry: P pxq " P p´xq .•

Identity covariance: DISPLAYFORM0 • Boundedness: @x " Z, }x} 2 ď B almost surely for some B ą 0.Our second assumption is on the patch structure.

In this paper we assume the stride is larger than half of the filter size.

This is indeed true for a wide range of convolutional neural network used in computer vision.

For example some architecture has convolutional filter of size 3 and stride 2 and some use non-overlapping architectures BID16 .

DISPLAYFORM1 Next we assume the activation function is piecewise linear.

Commonly used activation functions like rectified linear unit (ReLU), Leaky ReLU and linear activation all belong to this class.

Assumption 3.3 (Piece-wise Linear Activation).σpxq " DISPLAYFORM2

Now we are ready to describe our algorithm (see Algorithm 1).

The algorithm has three stages, first we learn the outer layer weights upto sign, second we use these fixed outer weights to recover the filter weight and last we choose the best weight combination thus recovered.

Our first observation is even if there may be overlapping patches, as long as there exists some nonoverlapping part, we can learn this part and the second layer jointly.

To be specific, with filter size being r and stride being s, if s ě t r 2 u`1, for j " 1. . . .

, k we define the selection matrix for the non-overlapping part of each patch DISPLAYFORM0 Note that for any j 1 ‰ j 2 , there is no overlapping between the selected coordinates by P non j1and P non j2 .

Therefore, for a filter w, there is a segment rw r´s`1 , . . .

, w s s with length p2s´rq which acts on the non-overlapping part of each patches.

We denote w non " rw r´s`1 , . . .

, w s s and our goal in this stage is to learn wn on and a˚jointly.

In this stage, our algorithm proceeds as follows.

Given w non , a and a sample px, yq, we define g pw non , a, x, yq " 2 1`γ´f pw non , a, xq´y¯k DISPLAYFORM1 x˘is the prediction function only using w non .As will be apparent in Section 4, g and h are unbiased estimates of the gradient for the loss function of learning a linear CNN.

The term , which is used to balance the magnitude between w non and a and make the algorithm more stable.

With some initialization w p0q non and a p0q , we use the following iterative updates inspired by isotonic regression BID13 DISPLAYFORM2 DISPLAYFORM3 where η 1 ą 0 is the step size parameter, ξ ptq wnon and ξ ptq a are uniformly sampled a unit sphere and at iteration we use a fresh sample`x ptq , y ptq˘. Here we add isotropic noise ξ ptq wnon and ξ ptq a because the objective function for learning a linear CNN is non-convex and there may exist saddle points.

Adding noise can help escape from these saddle points.

We refer readers to BID7 for more technical details regarding this.

As will be apparent in Section 4, after sufficient iterations, we obtain a pair`w pT1q , a pT1q˘s uch that either it is close to the truth pwn on , a˚q or close to the negative of the truth p´wn on ,´a˚q.

Remark 3.1 (Non-overlapping Patches).

If there is no overlap between patches, we can skip Stage 2 because after Stage 1 we have already learned a and w non " w.

Stage 2: Convotron with fixed Linear Layer In Stage 1 we have learned a good approximation to the second layer (either a pT1q or´a pT1q ).

Therefore, the problem reduces to learning a convolutional filter.

We run Convotron (Algorithm 3) proposed in BID13 Algorithm 3 Convotron BID13 Initialize DISPLAYFORM4 the right one.

To do this, we simply use T 3 " poly`k, B, 1 ˘f resh samples and output the solution which gives lower squared error.

arg min pw,aqPtpw p`q ,a pT q q,pw p´q ,´a pT q qu DISPLAYFORM0 Since we draw many samples, the empirical estimates will be close to the true loss using standard concentration bounds and choosing the minimum will give us the correct solution.

The following theorem shows that Algorithm 1 is guaranteed to learn the target convolutinoal neural network in polynomial time.

To our knowledge, this is the first polynomial time proper learning algorithm for convolutional neural network with overlapping patches.

Theorem 3.1 (Theorem 1.1 (Formal)).

Under Assumptions 3.1-3.3, if we set DISPLAYFORM0 ˘˘t hen with high probability, Algorithm 1 returns a pair pw, aq which satisfies DISPLAYFORM1 pf pw, a, xq´f pw˚, a˚, xqq 2 ı ď .

In this section we list our key ideas used for designing the Algorithm 1 and proving its correctness.

We discuss the analysis stage-wise for ease of understnading.

Now, taking expectation with respect to x, we have DISPLAYFORM0 where the last step we used our assumptions that patches are non-overlapping and the covariance of x is the identity.

From equation 7, it is now apparent that the population L 2 loss is just the standard loss for rank-1 matrix factorization problem.

Recent advances in non-convex optimization shows the following regularized loss function DISPLAYFORM1 satisfies all local minima are global and all saddles points and local maxima has a negative curvature BID8 and thus allows simple local search algorithm to find a global minimum.

Though the objective function in equation 8 is a population risk, we can obtain its stochastic gradient by our samples if we use fresh sample at each iteration.

We define DISPLAYFORM2 where`x ptq , y ptq˘i s the sample we use in the t-th iteration.

In expectation this is the standard gradient descent algorithm for solving equation 8: DISPLAYFORM3 With this stochastic gradient oracle at hand, we can implement the noise-injected stochastic gradient descent proposed in BID7 .

DISPLAYFORM4 where ξ ptq w and ξ ptq a are sampled from a unit sphere.

Theorem 6 in BID7 implies after polynomial iterations, this iterative procedure returns an -optimal solution of the objective function equation 8 with high probability.

Learning non-overlapping part of a CNN with piece-wise linear activation function Now we consider piece-wise linear activation function.

Our main observation is that we can still obtain a stochastic gradient oracle for the linear convolutional neural network using equation 2 and equation 3.

Formally, we have the following theorem.

Lemma 4.1 (Properties of Stochastic Gradient for Linear CNN).

Define DISPLAYFORM5 Under Assumption 3.1, we have E x rg pw non , a, x, yqs " BL reg pw non , aq Bw non , E x rh pw non , a, x, yqs " BL reg pw non , aq Ba where g pw non , a, x, yq and h pw non , a, x, yq are defined in equation 2 and equation 3, respectively.

Further, if }w non } 2 " Oppoly pσ 1 qq, }a} 2 " Oppoly pσ 1 qq, then the differences are also bounded DISPLAYFORM6 Here the expectation of g and h are equal to the gradient of the objective function for linear CNN because we assume the input distribution is symmetric and the activation function is piece-wise linear.

This observation has been stated in BID13 and based on this property, BID13 DISPLAYFORM7

After Stage 1, we have approximately recovered the outer layer weights.

We use these as fixed weights and run Convotron to obtain the filter weights.

The analysis of Convotron inherently handles average pooling as the outer layer.

Here we extend the analysis of Convotron to handle any fixed outer layer weights and also handle noise in these outer layer weights.

Formally, we obtain the following theorem: Convotron (modified) returns w such that with a constant probability, }w´w˚} 2 ď Opk 3 }w˚} q in polypk, }w˚} , B, logp1{ qq iterations.

Note that we present the theorem and proof for covariance being identity and no noise in the label but it can be easily extended to handle non-identity convariance with good condition number and bounded (in expectation) probabilistic concept noise.

Our analysis closely follows that from BID13 .

However, in contrast to the known second layer setting considered in BID13 , we only know an approximation to the second layer and a robust analysis is needed.

Another difficulty arises from the fact that the convergence rate depends on the least eigenvalue of P a :" ř 1ďi,jďk a i a j P i P T j .

By simple algebra, we can show that the matrix has the following form: DISPLAYFORM0 Using property of Toeplitz matrices, we show the least eigenvalue of P a is lower bounded by 1ć DISPLAYFORM1 2) for all a with norm 1.

Here we show how we can pick the correct hypothesis from the two possible hypothesis.

Under our assumptions, the individual loss py piq´f pw, a, x piq qq 2 is bounded.

sThus, a direct application of Hoeffding inequality gives the following guarantee.

Theorem 4.3.

Suppose T 3 " Ω`poly`r, k, B, 1 ˘˘a nd let pw, aq.

If either`w p`q , a T1˘o r w p´q ,´a T1˘h as population risk smaller than 2 , then let pw, aq be the output according to equation 6, then with high probability Now we put our analyses for Stage 1-3 together and prove Theorem 3.1.

By Theorem 4.1, we know we have a pT1q such that › › a DISPLAYFORM0

› ď O´ r 1{2 k 5{2 σ1¯( without loss of generality, we assume a and a˚are normalized) with DISPLAYFORM0 Now with Theorem 4.2, we know with η " O`poly`1 k , DISPLAYFORM1 Lastly, the following lemma bounds the loss of each instance in terms of the closeness of parameters.

Lemma 4.2.

For any a and w, we havé f pw˚, a˚, xq´f´w ptq , a, x¯¯2 ď 2k´}a}

Therefore, we know either`w p`q , a pT1q˘o r`w p´q ,´a pT1q˘a chieves prediction error.

Now combining Theorem 4.3 and Lemma A.1 we obtain the desired result.

In this section we use simulations to verify the effectiveness of our proposed method.

We fix input dimension d " 160 and filter size r " 16 for all experiments and vary the stride size s " 9, 12, 16.

For all experiments, we generate w˚and a˚from a standard Gaussian distribution and use 10, 000 samples to calculate the test error.

Note in Stage 2 of Algorithm we need to test a " a pT1q and a pT1q .

Here we only report the one with better performance in the Stage 2 because in Stage 3 we can decide which one is better.

To measure the performance of Stage 1, we use the angle between a t an a˚(in radians).

We first test Gaussian input distribution x " N p0, Iq.

FIG2 shows the convergence in Stage 1 of Algorithm 1 with T 1 " 10000 and η 1 " 0.0001.

FIG2 shows the convergence in Stage 2 of Algorithm 1 with T 2 " 10000 and η 2 " 0.0001.

We then test uniform input distribution x " Unifr´?3, ?

3s d (this distribution has identity covariance).

FIG2 shows the convergence in Stage 1 of Algorithm 1 with T 1 " 40000 and η 1 " 0.0001.

FIG2 shows the convergence in Stage 2 of Algorithm 1 with T 2 " 100000 and η 2 " 0.00001.

Note for both input distributions and all choices of stride size, our algorithm achieves low test error..

In this paper, we propose the first efficient algorithm for learning a one-hidden-layer convolutional neural network with possibly overlapping patches.

Our algorithm draws ideas from isotonic regression, landscape analysis of non-convex problem and spectral analysis of Toeplitz matrices.

These findings can inspire further development in this field.

Our next step is extend our ideas to design provable algorithms that can learn complicated models consisting of multiple filters.

To solve this problem, we believe the recent progress on landscape design BID9 may be useful.

In this section we present a few lemmas/theorems that are useful for our analysis.

Proof of Lemma 4.2.

Observe that, f pw˚, a˚, xq´f´w ptq , a, x¯¯2 ď 2´pf pw˚, a˚, xq´f pw˚, a, xqq 2`p f pw˚, a, xq´f pw, a, xqq 2s ince pa`bq 2 ď 2`a 2`b2˘f or all a, b P R.The first term can be bounded as follows, pf pw˚, a˚, xq´f pw˚, a, xqq 2 "˜k ÿ DISPLAYFORM0 .

Here the first inequality follows from observing that σ paq ď |a| and the last follows from }v} 1 ď ?

DISPLAYFORM1 Similarly, the other term can be bounded as follows, pf pw˚, a, xq´f pw, a, xqq 2 "˜k ÿ DISPLAYFORM2 Here we use the Lipschitz property of σ to get the first inequality.

The lemma follows from combining the above two.

The following lemma extends this to the overall loss.

Lemma A.1.

For any a and w, Er´f pw˚, a˚, xq´f´w ptq , a, x¯¯2s ď 2kB´}a} The following lemma from BID13 is key to our analysis.

Lemma A.2 (Lemma 1 of BID13 ).

For all a, b P R n , if Z is symmetric then, DISPLAYFORM3 The following well-known theorem is useful for bounding eigenvalues of matrices.

Theorem A.1 (Gershgorin Circle Theorem Weisstein (2003) ).

For a nˆn matrix A, define R i :" ř n j"1,j‰i |A i,j |.

Each eigenvalue of A must lie in at least one of the disks tz : DISPLAYFORM4 The following lemma bounds the eigenvalue of the weighted patch matrices.

Theorem A.2.

For all a P S k´1 , DISPLAYFORM5 Proof.

Since s ě t r 2 u`1, only adjacent patches overlap, and it is easy to verify that the matrix P a has the following structure: DISPLAYFORM6 Using the Gershgorin Circle Theorem, stated below, we can bound the eigenvalues, λ min pP a q ě DISPLAYFORM7 To bound the eigenvalues, we will boundˇˇř k´1 i"1 a i a i`1ˇb y maximizing it over all a such that }a} 2 " 1.

We have max DISPLAYFORM8 ) since the maximum can be achieved by setting all a i to be non-negative.

This can alternatively be viewed as max }a} 2 "1 a T Ma " λ max pMq where M is a tridiagonal symmetric Toeplitz matrix as follows: DISPLAYFORM9 It is well known that the eigenvalues of this matrix are of the form cos´i BID1 Thus E x rg pw non , a, x, yqs " BL reg pw, aq Bw .

DISPLAYFORM10 The proof for h pw non , a, x, yq is similar.

To obtain a bound of the gradient, note that Similar argument applies to y ř k j"1 a j P non jx.

We follow the Convotron analysis and include the changes.

Define S t " tpx 1 , y 1 q , . . .

, px t , y t qu.

The modified gradient update is as follows,

@highlight

We propose an algorithm for provably recovering parameters (convolutional and output weights) of a convolutional network with overlapping patches.

@highlight

This paper studies the theoretical learning of one-hidden-layer convolutional neural nets, resulting in a learning algorithm and provable guarantees using the algorithm.

@highlight

This paper gives a new algorithm for learning a two layer neural network which involves a single convolutional filter and a weight vector for different locations.