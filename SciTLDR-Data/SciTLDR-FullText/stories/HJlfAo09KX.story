We study model recovery for data classification, where the training labels are generated from a one-hidden-layer fully -connected neural network with sigmoid activations, and the goal is to recover the weight vectors of the neural network.

We prove that under Gaussian inputs, the empirical risk function using cross entropy exhibits strong convexity and smoothness uniformly in a local neighborhood of the ground truth, as soon as the sample complexity is sufficiently large.

This implies that if initialized in this neighborhood, which can be achieved via the tensor method, gradient descent converges linearly to a critical point that is provably close to the ground truth without requiring a fresh set of samples at each iteration.

To the best of our knowledge, this is the first global convergence guarantee established for the empirical risk minimization using cross entropy via gradient descent for learning one-hidden-layer neural networks, at the near-optimal sample and computational complexity with respect to the network input dimension.

Neural networks have attracted a significant amount of research interest in recent years due to the success of deep neural networks BID18 in practical domains such as computer vision and artificial intelligence BID24 BID15 BID27 .

However, the theoretical underpinnings behind such success remains mysterious to a large extent.

Efforts have been taken to understand which classes of functions can be represented by deep neural networks BID7 BID16 BID0 Telgarsky, 2016) , when (stochastic) gradient descent is effective for optimizing a non-convex loss function BID8 , and why these networks generalize well BID1 .One important line of research that has attracted extensive attention is a model-recovery setup, i.e., given that the training samples (x i , y i ) ∼ (x, y) are generated i.i.d.

from a distribution D based on a neural network model with the ground truth parameter W , the goal is to recover the underlying model parameter W , which is important for the network to generalize well BID22 .

Previous studies along this topic can be mainly divided into two types of data generations.

First, a regression problem, for example, assumes that each sample y is generated as y = 1 K K k=1 φ(w k x), where w k ∈ R d is the weight vector of the kth neuron, 1 ≤ k ≤ K, and the input x ∈ R d is Gaussian.

This type of regression problem has been studied in various settings.

In particular, BID28 studied the single-neuron model under ReLU activation, BID38 ) studied the onehidden-layer multi-neuron network model, and BID19 ) studied a two-layer feedforward networks with ReLU activations and identity mapping.

Second, for a classification problem, suppose each label y ∈ {0, 1} is drawn under the conditional distribution P(y = 1|x) = 1 K K k=1 φ(w k x), where w k ∈ R d is the weight vector of the kth neuron, 1 ≤ k ≤ K, and the input x ∈ R d is Gaussian.

Such a problem has been studied in BID21 in the case with a single neuron.

For both the regression and the classification settings, in order to recover the neural network parameters, all previous studies considered (stochastic) gradient descent over the squared loss, i.e., qu (W ; x, y) = DISPLAYFORM0 Furthermore, previous studies provided two types of statistical guarantees for such model recovery problems using the squared loss.

More specifically, BID38 showed that in the local neighborhood of the ground truth, the Hessian of the empirical loss function is positive definite for each given point under independent high probability event.

Hence, their guarantee for gradient descent to converge to the ground truth requires a fresh set of samples at every iteration, thus the total sample complexity will depend on the number of iterations.

On the other hand, studies such as BID21 BID28 establish certain types of uniform geometry such as strong convexity so that resampling per iteration is not needed for gradient descent to have guaranteed linear convergence as long as it enters such a local neighborhood.

However, such a stronger statistical guarantee without per-iteration resampling have only been shown for the squared loss function.

In this paper, we aim at developing such a strong statistical guarantee for the loss function in eq. (2), which is much more challenging but more practical than the squared loss for the classification problem.

This study provides the first performance guarantee for the recovery of one-hidden-layer neural networks using the cross entropy loss function, to the best of our knowledge.

More specifically, our contributions are summarized as follows.• For multi-neuron classification problem with sigmoid activations, we show that, if the input is Gaussian, the empirical risk function f n (W ) = 1 n n i=1 (W ; x i ) based on the cross entropy loss in eq. (2) is uniformly strongly convex in a local neighborhood of the ground truth W of size O(1/K 3/2 ) as soon as the sample size is O(dK 5 log 2 d), where d is the input dimension and K is the number of neurons.• We further show that, if initialized in this neighborhood, gradient descent converges linearly to a critical point W n (which we show to exist), with a sample complexity of O(dK 5 log 2 d), which is near-optimal up to a polynomial factor in K and log d. Due to the nature of quantized labels here, the recover of W is only up to certain statistical accuracy, and W n converges to W at a rate of O( dK 9/2 log n/n) in the Frobenius norm.

Furthermore, such a convergence guarantee does not require a fresh set of samples at each iteration due to the uniform strong convexity in the local neighborhood.

To obtain -accuracy, it requires a computational complexity of O(ndK 2 log(1/ )).•

We adopt the tensor method proposed in BID38 , and show it provably provides an initialization in the neighborhood of the ground truth.

In particular, our proof replaces the homogeneous assumption on activation functions in BID38 ) by a mild condition on the curvature of activation functions around W , which holds for a larger class of activation functions including sigmoid and tanh.

In order to analyze the challenging cross-entropy loss function, our proof develops various new machineries in order to exploit the statistical information of the geometric curvatures, including the gradient and Hessian of the empirical risk, and to develop covering arguments to guarantee uniform concentrations.

Our technique also yields similar performance guarantees for the classification problem using the squared loss in eq.(1), which we omit due to space limitations, as it is easier to analyze than cross entropy.

Due to page limitations we focus on the most relevant literature on theoretical and algorithmic aspects of learning shallow neural networks via nonconvex optimization.

The parameter recovery viewpoint is relevant to the success of non-convex learning in signal processing problems such as matrix completion, phase retrieval, blind deconvolution, dictionary learning and tensor decomposition BID31 BID6 BID13 BID30 BID2 , to name a few.

The statistical model for data generation effectively removes worst-case instances and allows us to focus on average-case performance, which often possess much benign geometric properties that enable global convergence of simple local search algorithms.

The studies of one-hidden-layer network model can be further categorized into two classes, landscape analysis and model recovery.

In the landscape analysis, it is known that if the network size is large enough compared to the data input, then there are no spurious local minima in the optimization landscape, and all local minima are global BID3 BID25 BID23 .

For the case with multiple neurons (2 ≤ K ≤ d) in the under-parameterized setting, the work of Tian BID33 studied the landscape of the population squared loss surface with ReLU activations.

In particular, there exist spurious bad local minima in the optimization landscape BID26 even at the population level.

Zhong et.

al. BID38 provided several important characterizations for the local Hessian for the regression problem for a variety of activation functions for the squared loss.

In the model recovery problem, the number of neurons is smaller than the dimension of inputs.

In the case with a single neuron (K = 1), under Gaussian input, BID28 showed that gradient descent converges linearly when the activation function is ReLU, i.e. φ(z) = max{z, 0}, with a zero initialization, as long as the sample complexity is O(d) for the regression problem.

On the other end, BID21 showed that when φ(·) has bounded first, second and third derivatives, there is no other critical points than the unique global minimum (within a constrained region of interest), and (projected) gradient descent converges linearly with an arbitrary initialization, as long as the sample complexity is O(d log 2 d) with sub-Gaussian inputs for the classification problem using the squared loss.

Moreover, BID38 shows that the ground truth From a technical perspective, our study differs from all the aforementioned work in that the cross entropy loss function we analyze has a very different form.

Furthermore, we study the model recovery classification problem under the multi-neuron case, which has not been studied before.

Finally, we note that several papers study one-hidden-layer or two-layer neural networks with different structures under Gaussian input.

For example, BID9 b; BID37 ) studied the non-overlapping convolutional neural network, BID19 ) studied a two-layer feedforward networks with ReLU activations and identity mapping, and BID11 introduced the Porcupine Neural Network.

These results are not directly comparable to ours since both the networks and the loss functions are different.

The rest of the paper is organized as follows.

Section 2 describes the problem formulation.

Section 3 presents the main results on local geometry and local linear convergence of gradient descent.

Section 4 discusses the initialization method.

Numerical examples are demonstrated in Section 5, and finally, conclusions are drawn in Section 6.Throughout this paper, we use boldface letters to denote vectors and matrices, e.g. w and W .

The transpose of W is denoted by W , and W , W F denote the spectral norm and the Frobenius norm.

For a positive semidefinite (PSD) matrix A, we write A 0.

The identity matrix is denoted by I. The gradient and the Hessian of a function f (W ) is denoted by ∇f (W ) and ∇ 2 f (W ), respectively.

Let σ i (W ) denote the i-th singular value of W .

Denote · ψ1 as the sub-exponential norm of a random variable.

We use c, C, C 1 , . . .

to denote constants whose values may vary from line to line.

For nonnegative functions f (x) and g(x), f (x) = O (g(x)) means there exist positive constants c and a such that f (x) ≤ cg(x) for all x ≥ a; f (x) = Ω (g(x)) means there exist positive constants c and a such that f (x) ≥ cg(x) for all x ≥ a.

We first describe the generative model for training data, and then describe the gradient descent algorithm for learning the network weights.

Suppose we are given n training samples {(x i , y i )} n i=1 ∼ (x, y) that are drawn i.i.d., where x ∼ N (0, I).

Assume the activation function is sigmoid, i.e. φ (z) = 1/(1 + e −z ) for all z. Conditioned on x ∈ R d , we consider the classification setting, where y is mapped to a discrete label using the one-hidden layer neural network model as follows: DISPLAYFORM0 and P(y = 0|x) = 1 − P(y = 1|x), where K is the number of neurons.

Our goal is to estimate W = [w 1 , · · · , w K ], via minimizing the following empirical risk function: DISPLAYFORM1 where (W ; x) := (W ; x, y) is the cross entropy loss, i.e., the negative log-likelihood function, i.e., DISPLAYFORM2 With slight abuse of notation, we denote the gradient and Hessian of (W ; x) with respect to the vector w.

To estimate W , since (4) is a highly nonconvex function, vanilla gradient descent with an arbitrary initialization may get stuck at local minima.

Therefore, we implement the gradient descent algorithm with a well-designed initialization scheme that is described in detail in Section 4.

The update rule is given as DISPLAYFORM0 , where η is the step size.

The algorithm is summarized in Algorithm 1.

DISPLAYFORM1

We note that throughout the execution of the algorithm, the same set of training samples is used which is the standard implementation of gradient descent.

This is in sharp contrast to existing work such as BID38 that employs the impractical scheme of resampling, where a fresh set of training samples is used at every iteration of gradient descent.

Before stating our main results, we first introduce an important quantity regarding φ(z) that captures the geometric properties of the loss function, distilled in BID38 .

Figure 1: ρ (σ) for sigmoid activation.

DISPLAYFORM0 Note that the definition here is different from that in (Zhong et al., 2017b, Property 3 .2) but consistent with (Zhong et al., 2017b, Lemma D.4) which removes the third term in (Zhong et al., 2017b, Property 3.2) .

For the activation function considered in this paper, the first two terms suffice.

We depict ρ(σ) as a function of σ in a certain range for the sigmoid activation in Fig. 1 .

It is easy to observe that ρ(σ) > 0 for all σ > 0.

We first characterize the local strong convexity of f n (W ) in a neighborhood of the ground truth W .

Let B (W , r) denote a Euclidean ball centered at W ∈ R d×K with a radius r, i.e. DISPLAYFORM0 Let σ i := σ i (W ) denote the i-th singular value of W .

Let the condition number be κ = σ 1 /σ K , and λ = DISPLAYFORM1 The following theorem guarantees the Hessian of the empirical risk function f n (W ) in the local neighborhood of W is positive definite with high probability.

Theorem 1.

For the classification model (3) with sigmoid activation function, assume W F ≤ 1, then there exists some constant C, such that if DISPLAYFORM2 then with probability at least 1 − d −10 , for all W ∈ B(W , r), DISPLAYFORM3 hold, where r := min DISPLAYFORM4 We note that all column permutations of W are equivalent global minima of the loss function, and Theorem 1 applies to all such permutation matrices of W .

The proof of Theorem 1 is outlined in Appendix A. Theorem 1 guarantees that the Hessian of the empirical cross-entropy loss function f n (W ) is positive definite (PD) in a neighborhood of the ground truth W , as long as ρ(σ K ) > 0 (i.e. W is full-column rank), when the sample size n is sufficiently large for the sigmoid activation.

The bounds in Theorem 1 depend on the dimension parameters of the network (n and K), as well as the activation function and the ground truth (ρ(σ K ), λ).

As a special case, suppose W is composed of orthonormal columns with ρ(σ K ) = O(1), κ = 1, λ = 1.

Then, Theorem 1 guarantees DISPLAYFORM5 , as soon as the sample complexity n = Ω(dK 5 log 2 d).

The sample complexity is order-wise near-optimal in d up to polynomial factors of K and log d, since the number of unknown parameters is dK.

For the classification problem, due to the nature of quantized labels, W is no longer a critical point of f n (W ).

By the strong convexity of the empirical risk function f n (W ) in the local neighborhood of W , there can exist at most one critical point in B(W , r), which is the unique local minimizer in B (W , r) if it exists.

The following theorem shows that there indeed exists such a critical point W n , which is provably close to the ground truth W , and gradient descent converges linearly to W n .

Theorem 2.

For the classification model (3) with sigmoid activation function, and assume W F ≤ 1, there exist some constants C, C 1 > 0 such that if the sample size n ≥ C · dK DISPLAYFORM0 then with probability at least 1 − d −10 , there exists a unique critical point W n in B(W , r) with DISPLAYFORM1 Moreover, if the initial point W 0 ∈ B (W , r), then gradient descent converges linearly to W n , i.e. DISPLAYFORM2 where DISPLAYFORM3 , as long as the step size DISPLAYFORM4 Similarly to Theorem 1, Theorem 2 also holds for all column permutations of W .

The proof can be found in Appendix B. Theorem 2 guarantees that there exists a critical point W n in B(W , r) which converges to W at the rate of O(K 9/4 d log n/n), and therefore W can be recovered consistently as n goes to infinity.

Moreover, gradient descent converges linearly to W n at a linear rate, as long as it is initialized in the basin of attraction.

To achieve -accuracy, i.e. DISPLAYFORM5 requires a computational complexity of O ndK 2 log (1/ ) , which is linear in n, d and log(1/ ).

Our initialization adopts the tensor method proposed in BID38 .

In this section, we first briefly describe this method, and then present the performance guarantee of the initialization with remarks on the differences from that in BID38 .

This subsection briefly introduces the tensor method proposed in BID38 , to which a reader can refer for more details.

We first define a product ⊗ as follows.

If v ∈ R d is a vector and I is the identity matrix, then DISPLAYFORM0 Definition 3.

Let α ∈ R d denote a randomly picked vector.

We define P 2 and P 3 as follows: DISPLAYFORM1 1 where j 2 = min{j ≥ 2|M j = 0}, and P 3 = M j3 (I, I, I, α, · · · , α), where j 3 = min{j ≥ 3|M j = 0}.We further denote w = w/ w .

The initialization algorithm based on the tensor method is summarized in Algorithm 2, which includes two major steps.

Step 1 first estimates the direction of each column of W by decomposing P 2 to approximate the subspace spanned by {w 1 , w 2 , · · · , w K } (denoted by V ), then reduces the third-order tensor P 3 to a lower-dimension tensor R 3 = P 3 (V , V , V ) ∈ R K×K×K , and applys non-orthogonal tensor decomposition on R 3 to output the estimate s i V w i , where s i ∈ {1, −1} is a random sign.

Step 2 approximates the magnitude of w i and the sign s i by solving a linear system of equations.

For the classification problem, we make the following technical assumptions, similarly in (Zhong et al., 2017b, Assumption 5.

3) for the regression problem.

Assumption 1.

The activation function φ(z) satisfies the following conditions: DISPLAYFORM0 2.

At least one of M 3 and M 4 is non-zero.

Furthermore, we do not require the homogeneous assumption ((i.e., φ(az) = a p z for an integer p)) required in BID38 , which can be restrictive.

Instead, we assume the following condition on the curvature of the activation function around the ground truth, which holds for a larger class of activation functions such as sigmoid and tanh.

Assumption 2.

Let l 1 be the index of the first nonzero M i where i = 1, . . .

, 4.

For the activation function φ (·), there exists a positive constant δ such that m l1,i (·) is strictly monotone over the interval ( w i − δ, w i + δ), and the derivative of m l1,i (·) is lower bounded by some constant for all i.

We next present the performance guarantee for the initialization algorithm in the following theorem.

Theorem 3.

For the classification model (3), under Assumptions 1 and 2, if the sample size n ≥ dpoly (K, κ, t, log d, 1/ ), then the output W 0 ∈ R d×K of Algorithm 2 satisfies DISPLAYFORM1 with probability at least 1 − d−Ω(t) .The proof of Theorem 3 consists of (a) showing the estimation of the direction of W is sufficiently accurate and (b) showing the approximation of the norm of W is accurate enough.

Our proof of part (a) is the same as that in BID38 ), but our argument in part (b) is different, where we relax the homogeneous assumption on activation functions.

More details can be found in the supplementary materials in Appendix C.

In this section, we first implement gradient descent to verify that the empirical risk function is strongly convex in the local region around W .

If we initialize multiple times in such a local region, it is expected that gradient descent converges to the same critical point W n , with the same set of training samples.

Given a set of training samples, we randomly initialize multiple times, and then calculate the variance of the output of gradient descent.

Denote the output of the th run as w DISPLAYFORM0

, where L = 20 is the total number of random initializations.

Adopted in BID21 , it quantifies the standard deviation of the estimator W n under different initializations with the same set of training samples.

We say an experiment is successful, if SD n ≤ 10 −2 .Figure 2 (a) shows the successful rate of gradient descent by averaging over 50 sets of training samples for each pair of n and d, where K = 3 and d = 15, 20, 25 respectively.

The maximum iterations for gradient descent is set as iter max = 3500.

It can be seen that as long as the sample complexity is large enough, gradient descent converges to the same local minima with high probability.

We next show that the statistical accuracy of the local minimizer for gradient descent if it is initialized close enough to the ground truth.

Suppose we initialize around the ground truth such that W 0 − W F ≤ 0.1 · W F .

We calculate the average estimation error as DISPLAYFORM0 Monte Carlo simulations with random initializations.

FIG1 shows the average estimation error with respect to the sample complexity when K = 3 and d = 20, 35, 50 respectively.

It can be seen that the estimation error decreases gracefully as we increase the sample size and matches with the theoretical prediction of error rates reasonably well.

We further compare the performance of gradient descent algorithm applied to both the cross entropy loss and the squared loss, respectively.

As shown in FIG1 , when K = 3, d = 20, cross entropy loss with gradient descent achieves a much lower error than the squared loss.

Clearly, the cross entropy loss is favored in the classification problem over the squared loss.

In this paper, we have studied the model recovery of a one-hidden-layer neural network using the cross entropy loss in a multi-neuron classification problem.

In particular, we have characterized the sample complexity to guarantee local strong convexity in a neighborhood (whose size we have characterized as well) of the ground truth when the training data are generated from a classification model.

This guarantees that with high probability, gradient descent converges linearly to the ground truth if initialized properly.

In the future, it will be interesting to extend the analysis in this paper to more general class of activation functions, particularly ReLU-like activations; and more general network structures, such as convolutional neural networks BID10 BID37 .

To begin, denote the population loss function as DISPLAYFORM0 where the expectation is taken with respect to the distribution of the training sample (x; y).The proof of Theorem 1 follows the following steps:1.

We first show that the Hessian ∇ 2 f (W ) of the population loss function is smooth with respect to ∇ 2 f (W ) (Lemma 1);2.

We then show that ∇ 2 f (W ) satisfies local strong convexity and smoothness in a neighborhood of W , B(W , r) with appropriately chosen radius by leveraging similar properties of ∇ 2 f (W ) (Lemma 2); 3.

Next, we show that the Hessian of the empirical loss function ∇ 2 f n (W ) is close to its popular counterpart ∇ 2 f (W ) uniformly in B(W , r) with high probability (Lemma 3).4.

Finally, putting all the arguments together, we establish ∇ 2 f n (W ) satisfies local strong convexity and smoothness in B(W , r).We will first show that the Hessian of the population risk is smooth enough around W in the following lemma.

Lemma 1.

For sigmoid activations, assume W F ≤ 1, we have DISPLAYFORM1 holds for some large enough constant C, when W − W F ≤ 0.7.The proof is given in Appendix D.2.

Lemma 1 together with the fact that ∇ 2 f (W ) be lower and upper bounded, will allow us to bound ∇ 2 f (W ) in a neighborhood around ground truth, given below.

Lemma 2 (Local Strong Convexity and Smoothness of Population Loss).

For sigmoid activations, there exists some constant C, such that DISPLAYFORM2 holds for all W ∈ B(W , r) with r := min DISPLAYFORM3 The proof is given in Appendix D.3.

The next step is to show the Hessian of the empirical loss function is close to the Hessian of the population loss function in a uniform sense, which can be summarized as following.

Lemma 3.

For sigmoid activations, there exists constant C such that as long as n ≥ C · dK log dK, with probability at least 1 − d −10 , the following holds DISPLAYFORM4 where r := min DISPLAYFORM5 The proof can be found in Appendix D.4.The final step is to combine Lemma 3 and Lemma 1 to obtain Theorem 1 as follows,Proof of Theorem 1.

By Lemma 3 and Lemma 2, we have with probability at least DISPLAYFORM6 As long as the sample size n is set such that DISPLAYFORM7 holds for all W ∈ B (W , r).

Similarly, we have DISPLAYFORM8 holds for all W ∈ B (W , r).

We have established that f n (W ) is strongly convex in B(W , r) in Theorem 1, thus there exists at most one critical point in B(W , r).

The proof of Theorem 2 follows the steps below:1.

We first show that the gradient ∇f n (W ) concentrates around ∇f (W ) in B(W , r) (Lemma 4), and then invoke BID21 , Theorem 2) to guarantee there indeed exists a critical point W n in B(W , r);2.

We next show W n is close to W and gradient descent converges linearly to W n with a properly chosen step size.

The following lemma establishes that ∇f n (W ) uniformly concentrates around ∇f (W ).

Lemma 4.

For sigmoid activation function, assume W F ≤ 1, there exists constant C such that as long as n ≥ CdK log(dK), with probability at least 1 − d −10 , the following holds DISPLAYFORM0 where r := min DISPLAYFORM1 Notice that for the population risk function, f (W ), W is the unique critical point in B(W , r) due to local strong convexity.

With Lemma 3 and Lemma 4, we can invoke (Mei et al., 2016, Theorem 2) , which guarantees the following.

Corollary 1.

There exists one and only one critical point W n ∈ B (W * , r) that satisfies DISPLAYFORM2 We first show that W n is close to W .

By the intermediate value theorem, ∃W ∈ B (W , r) such that DISPLAYFORM3 where the last inequality follows from the optimality of W n .

By Theorem 1, we have DISPLAYFORM4 On the other hand, by the Cauchy-Schwarz inequality, we have DISPLAYFORM5 where the last line follows from Lemma 4.

Plugging FORMULA0 and FORMULA0 into FORMULA0 , we have DISPLAYFORM6 Now we have established there indeed exists a critical point in B(W , r).

We can establish local linear convergence of gradient descent as below.

Let W t be the estimate at the t-th iteration.

According to the update rule, we have DISPLAYFORM7 Moreover, by the fundamental theorem of calculus BID17 , ∇f n (W t ) can be written as DISPLAYFORM8 .

By Theorem 1, we have DISPLAYFORM9 where DISPLAYFORM10 and H max = C. Therefore, we have DISPLAYFORM11 Hence, DISPLAYFORM12 as long as we set η < DISPLAYFORM13 .

In summary, gradient descent converges linearly to the local minimizer W n .

The proof contains two parts.

Part (a) proves that the estimation of the direction of W is sufficiently accurate, which follows the arguments similar to those in BID38 and is only briefly summarized below.

Part (b) is different, where we do not require the homogeneous condition for the activation function, and instead, our proof is based on a mild condition in Assumption 2.

We detail our proof in part (b).We first define a tensor operation as follows.

For a tensor T ∈ R n1×n2×n3 and three matrices A ∈ R n1×d1 , B ∈ R n2×d2 , C ∈ R n3×d3 , the (i, j, k)-th entry of the tensor T (A, B, C) is given by DISPLAYFORM0 (a) In order to estimate the direction of each w i for i = 1, . . .

, K, BID38 shows that for the regression problem, if the sample size n ≥ dpoly (K, κ, t, log d), then DISPLAYFORM1 holds with high probability.

Such a result also holds for the classification problem with only slight difference in the proof as we describe as follows.

The main idea of the proof is to bound the estimation error of P 2 and R 3 via Bernstein inequality.

For the regression problem, Bernstein inequality was applied to terms associated with each neuron individually, and the bounds were then put together via triangle inequality in BID38 , whereas for the classification problem here, we apply Bernstein inequality to terms associated with all neurons all together.

Another difference is that the label y i of the classification model is bounded by nature, whereas the output y i in the regression model needs to be upper bounded via homogeneously bounded conditions of the activation function.

A reader can refer to BID38 for the details of the proof for this part.(b) In order to estimate w i for i = 1, . . .

, K, we provide a different proof from BID38 , which does not require the homogeneous condition on the activation function, but assumes a more relaxed condition in Assumption 2.We define a quantity Q 1 as follows: DISPLAYFORM2 where l 1 is the first non-zero index such that M l1 = 0.

For example, if l 1 = 3, then Q 1 takes the following form DISPLAYFORM3 where w = w/ w and by definition m 3,i ( DISPLAYFORM4 Clearly, Q 1 has information of w i , which can be estimated by solving the following optimization problem: DISPLAYFORM5 where each entry of the solution takes the form DISPLAYFORM6 In the initialization, we substitute Q 1 (estimated from training data) for Q 1 , V u i (estimated in part (a)) for s i w i into FORMULA55 , and obtain an estimate β of β .

We then substitute β for β and V u i for s i w i into (21) to obtain an estimate a i of w i via the following equation DISPLAYFORM7 Furthermore, since m l1,i (x) has fixed sign for x > 0 and for l 1 ≥ 1, s i can be estimated correctly from the sign of β i for i = 1, . . .

, K.For notational simplicity, let β 1,i := DISPLAYFORM8 2 , and then FORMULA0 and FORMULA57 DISPLAYFORM9 By Assumption 2 and (21), there exists a constant δ > 0 such that the inverse function g(·) of m 3,1 (·) has upper-bounded derivative in the interval (β 1,i − δ , β 1,i + δ ), i.e., |g (x)| < Γ for a constant Γ. By employing the result in BID38 , if the sample size n ≥ dpoly (K, κ, t, log d), then Q 1 and Q 1 , V u i and s i w i can be arbitrarily close so that |β DISPLAYFORM10 Thus, by (23) and mean value theorem, we obtain DISPLAYFORM11 where ξ is between β 1,i and β 1,i , and hence |g (ξ)| < Γ. Therefore, DISPLAYFORM12 , which is the desired result.

We introduce some useful definitions and results that will be used in the proofs.

The first one is the definition of norms of random variable, i.e.

Definition 4 (Sub-gaussian and Sub-exponential norm).

The sub-gaussian norm of a random variable X, denotes as X ψ2 , is defined as DISPLAYFORM0 and the sub-exponential norm of X, denoted as X ψ1 , is defined as DISPLAYFORM1 The definition is summarized from (Vershynin, 2012, Def 5.7,Def 5.13) , and if X ψ2 is upper bounded, then X is a sub-gaussian random variable and it satisfies DISPLAYFORM2 Next we provide the calculations of the gradient and Hessian of E [ (W ; DISPLAYFORM3 where if j = l, DISPLAYFORM4 and if j = l, DISPLAYFORM5 Next we will evaluate ∆ j,l .

From FORMULA28 we can write the hessian block more concisely as DISPLAYFORM6 where g j,l (W ) = ξ j,l (W ) (p(W )(1−p(W ))) 2 ∈ R, and then by the mean value theorem, we can write g j,l (W ) as DISPLAYFORM7 where W = η · W + (1 − η) W for some η ∈ (0, 1).

Thus we can calculate ∆ j,l as DISPLAYFORM8 and plug it back to (30) we can obtain DISPLAYFORM9 for the third equality we have used the fact that DISPLAYFORM10 since the variable of g j,l W is in the form of w i x. and for the last two inequalities, we have used Cauchy-Schwarz inequality.

Our next goal is to upper bound E T 2 j,l,k .

Further since DISPLAYFORM11 which aligns with x and the scalar coefficient is upper bounded by DISPLAYFORM12 are all upper bounded, thus we leave only the denominator.

And then DISPLAYFORM13 holds for some constant C, where the second inequality follows from Lemma 5.Lemma 5.

Let x ∼ N (0, I), t = max { w 1 2 , · · · w K 2 } and z ∈ Z such that z ≥ 1 , for the sigmoid activation function φ (x) = 1 1+e −x , the following DISPLAYFORM14 holds for a large enough constant C which depends on the constant z.

Plugging FORMULA1 into FORMULA1 , we can obtain DISPLAYFORM15 Further since e DISPLAYFORM16 , where we have used the assumption that W F ≤ 1 thus we can conclude that if DISPLAYFORM17 holds for some constant C.

Proof.

We will first present upper and lower bounds of the Hessian of the population risk at ground truth, i.e. ∇ 2 f (W ), and then apply Lemma 1 to obtain a uniform bound in the neighborhood of W .

As a reminder, DISPLAYFORM0 and let a = [a 1 , · · · , a K ] ∈ R dK , we can write DISPLAYFORM1 the second inequality holds due to the fact that DISPLAYFORM2 , and the last inequality follows from (Zhong et al., 2017b, Lemmas D.4 and D.6 ).Further more, we can uppder bound ∇ 2 f (W ) as DISPLAYFORM3 where for the third and fourth inequality we have used the fact that φ w i x 1 − φ w i x ≤ 1 4and DISPLAYFORM4 Thus together with the lower bound (41) we can conclude that DISPLAYFORM5 From Lemma 1, we have DISPLAYFORM6 therefore, when W − W F ≤ 0.7 and DISPLAYFORM7 i.e., when W − W F ≤ min DISPLAYFORM8 κ 2 λ , 0.7 for some constant C, we have DISPLAYFORM9 Moreover, within the same neighborhood, by the triangle inequality we have DISPLAYFORM10 D.4 PROOF OF LEMMA 3Proof.

We adapt the analysis in BID21 to our setting.

Let N be the -covering number of the Euclidean ball B (W , r).

It is known that log N ≤ dK log (3r/ ) BID35 .

Let W = {W 1 , · · · , W N } be the -cover set with N elements.

For any W ∈ B (W , r), let j (W ) = argmin j∈[N ]

W − W j(W ) F ≤ for all W ∈ B (W , r).For any W ∈ B (W , r), we have DISPLAYFORM11 Hence, we have DISPLAYFORM12 where the events A t , B t and C t are defined as DISPLAYFORM13 DISPLAYFORM14 DISPLAYFORM15 In the sequel, we will bound the terms P (A t ), P (B t ), and P (C t ), separately.1.

Upper bound P (B t ).

Before continuing, let us state a simple technical lemma that is useful for our proof, whose proof can be found in BID21 .

Let G i = v, ∇ 2 (W ; x i ) − E ∇ 2 (W ; x) v where E[G i ] = 0.

Let a = a 1 , · · · , a K ∈ R dK .

Then we can show that G i ψ1 is upper bounded, which we summariz as follows.

Lemma 7.

There exists some constant C such that DISPLAYFORM16

@highlight

We provide the first theoretical analysis of guaranteed recovery of one-hidden-layer neural networks under cross entropy loss for classification problems.