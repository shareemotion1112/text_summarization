We consider the problem of learning a one-hidden-layer neural network: we assume the input x is from Gaussian distribution and the label $y = a \sigma(Bx) + \xi$, where a is a nonnegative vector and  $B$ is a full-rank weight matrix, and $\xi$ is a noise vector.

We first give an analytic formula for the population risk of the standard squared loss and demonstrate that it implicitly attempts to decompose a sequence of low-rank tensors simultaneously.

Inspired by the formula, we design a non-convex objective function $G$ whose landscape is guaranteed to have the following properties:	  1.

All local minima of $G$ are also global minima.

2.

All global minima of $G$ correspond to the ground truth parameters.

3.

The value and gradient of $G$ can be estimated using samples.

With these properties, stochastic gradient descent on $G$ provably converges to the global minimum and learn the ground-truth parameters.

We also prove finite sample complexity results and validate the results by simulations.

Scalable optimization has played an important role in the success of deep learning, which has immense applications in artificial intelligence.

Remarkably, optimization issues are often addressed through designing new models that make the resulting training objective functions easier to be optimized.

For example, over-parameterization BID19 , batch-normalization BID14 , and residual networks BID12 b) are often considered as ways to improve the optimization landscape of the resulting objective functions.

How do we design models and objective functions that allow efficient optimization with guarantees?

Towards understanding this question in a principled way, this paper studies learning neural networks with one hidden layer.

Roughly speaking, we will show that when the input is from Gaussian distribution and under certain simplifying assumptions on the weights, we can design an objective function G(·), such that[a] all local minima of G(·) are global minima [b] all the global minima are the desired solutions, namely, the ground-truth parameters (up to permutation and some fixed transformation).We note that designing such objective functions is challenging because 1) the natural 2 loss objective does have bad local minimum, and 2) due to the permutation invariance 1 , the objective function inherently has to contain an exponential number of isolated local minima.

We aim to learn a neural network with a one-hidden-layer using a non-convex objective function.

We assume input x comes from Gaussian distribution and the label y comes from the model y = a σ(B x) + ξ (1.1)where a ∈ R m , B ∼ R m×d are the ground-truth parameters, σ(·) is a element-wise non-linear function, and ξ is a noise vector with zero mean.

Here we can without loss of generality assume x comes from spherical Gaussian distribution N (0, Id d×d ).

For technical reasons, we will further assume m ≤ d and that a has non-negative entries.

The most natural learning objective is perhaps the 2 loss function, given the additive noise.

Concretely, we can parameterize with training parameters a ∈ R m , B ∼ R m×d of the same dimension as a and B correspondingly,ŷ = a σ(Bx) ,( 1.2) and then use stochastic gradient descent to optimize the 2 loss function.

In many parts of our paper, we consider σ to be the ReLU function σ(x) = max{x, 0}. In such settings we assume rows of B have norm 1 because a and rows of B can be scaled simultaneously without changing the model or the objective.

When we have enough training examples, we are effectively minimizing the following population risk with stochastic updates, DISPLAYFORM0 However, empirically stochastic gradient descent cannot converge to the ground-truth parameters in the synthetic setting above when σ(x) = ReLU(x) = max{x, 0}, even if we have access to an infinite number of samples, and B is a orthogonal matrix.

We also show this phenomena generalizes to the case when σ(x) is the sigmoid function and the learned network also have the same architecture.

Such empirical results have been reported in BID19 previously, and we also provide our version in FIG1 and Figure 2 of Section 4.

This is consistent with observations and theory that over-parameterization is important for training neural networks successfully BID19 BID11 BID22 .These empirical findings suggest that the population risk f (a, B) has spurious local minima with inferior error compared to that of the global minimum.

This phenomenon occurs even if we assume we know a or a = 1 is merely just the all one's vector.

Empirically, such landscape issues seem to be alleviated by over-parameterization.

By contrast, our method described in the next section does not require over-parameterization and might be suitable for applications that demand the recovery of the true parameters.

Towards learning with the same number of training parameters as the ground-truth model, we first study the landscape of the population risk f (·) and give an analytic formula for it -as an explicit function of the ground-truth parameters and training parameters with the randomness of the data being marginalized out.

The formula in equation (2.3) shows that f (·) is implicitly attempting to solve simultaneously an infinite number of low-rank tensor decomposition problems with commonly shared components.

Inspired by the formula, we design a new training model whose associated loss function -named f and formally defined in equation (2.5) -corresponds to the loss function for decomposing a matrix (2-nd order tensor) and a 4-th order tensor (Theorem 2.2).

Empirically, stochastic gradient descent on f learns the network as shown Section 4.Despite the empirical success of f , we still lack a provable guarantee on the landscape of f .

The second contribution of the paper is to design a more sophisticated objective G(·) whose landscape Published as a conference paper at ICLR 2018 is provably nice -all the local minima of G(·) are proven to be global, and they correspond to the permutation of the true parameters.

See Theorem 2.3.Moreover, the value and the gradient of G can be estimated using samples, and there are no constraints in the optimization.

These allow us to use straightforward SGD (see guarantees in BID7 ; ) to optimize G(·) and converge to a local minimum, which is also a global minimum (Corollary 2.4).Finally, we also prove a finite-sample complexity result.

We will show that with a polynomial number of samples, the empirical version of G share almost the same landscape properties as G itself (Theorem 2.7).

Therefore, we can also use an empirical version of G as a surrogate in the optimization.

The work of BID0 is one of the early results on provable algorithms for learning deep neural networks, where the authors give an algorithm for learning deep generative models with sparse weights.

BID19 , Zhang et al. (2016; 2017b) , and BID5 study the learnability of special cases of neural networks using ideas from kernel methods.

Janzamin et al. (2015) give a polynomial-time algorithm for learning one-hidden-layer neural networks with twice-differentiable activation function and known input distributions.

Their approach uses the idea of score function to estimate the high order tensors related to the true components, and then apply tensor decompositions to recover the true parameters.

When applied to Gaussian input distribution, the score function becomes Hermite polynomials.

A series of recent papers study the theoretical properties of non-convex optimization algorithms for one-hidden-layer neural networks.

BID4 and Tian (2017) analyze the landscape of the population risk for one-hidden-layer neural networks with Gaussian inputs under the assumption that the weights vector associated to each hidden variable (that is, the filters) have disjoint supports.

BID18 proves that stochastic gradient descent recovers the ground-truth parameters when the parameters are known to be close to the identity matrix.

Zhang et al. (2017a) study the optimization landscape of learning one-hidden-layer neural networks with a specific activation function, and they design a specific objective function that can recover a single column of the weight matrix.

Zhong et al. (2017) study the convergence of non-convex optimization from a good initializer that is produced by tensor methods.

Our algorithm works for a large family of activation functions (including ReLU) and any full-rank weight matrix.

To our best knowledge, we give the first global convergence result for gradient-based methods for our general setting.

The optimization landscape properties have also been investigated on simplified neural networks models.

BID17 shows that the landscape of deep neural nets does not have bad local minima but has degenerate saddle points.

BID10 show that re-parametrization using identity connection as in residual networks BID12 can remove the degenerate saddle points in the optimization landscape of deep linear residual networks.

BID22 show that an over-parameterized neural network does not have bad differentiable local minimum.

BID11 analyze the power of over-parameterization in a linear recurrent network (which is equivalent to a linear dynamical system.)The optimization landscape has also been analyzed for other machine learning problems, including SVD/PCA phase retrieval/synchronization, orthogonal tensor decomposition, dictionary learning, matrix completion, matrix sensing BID1 ; Srebro & Jaakkola (2013); BID7 ; Sun et al. (2015) ; BID2 BID8 ; BID3 .

Our analysis techniques build upon that for tensor decomposition in BID7 -we add two additional regularization terms to deal with spurious local minimum caused by the weights a and to remove the constraints.

We use · to denote the Euclidean norm of a vector and spectral norm of a matrix.

We use · F to denote the Frobenius/Euclidean norm of a matrix or high-order tensor.

For a vector x, let x 0 denotes its infinity norm and for a matrix A, let |A| 0 be a shorthand for vec(A) 0 where vec(A) is the vectorization of A.We use A ⊗ B to denote the Kronecker product of A and B, and A ⊗k is a shorthand for A ⊗ · · · ⊗ A where A appears k times.

For vectors a ⊗ b and a ⊗k denote the tensor product.

We denote the identity matrix in dimension d × d by Id d×d , or Id when the dimension is clear from the context.

We will define other notations when we first use them.

We first show that a natural 2 loss for the one-hidden-layer neural network can be interpreted as simultaneously decomposing tensors of different orders.

A straightforward approach of learning the model (1.1) is to parameterize the prediction bŷ DISPLAYFORM0 where a ∈ R d , B ∼ R m×d are the training parameters.

Naturally, we can use 2 as the empirical loss, which means the population risk is DISPLAYFORM1 Throughout the paper, we use b 1 , . . .

, b m to denote the row vectors of B and similarly for B.That is, we have B = DISPLAYFORM2 Let a i and a i 's be the coordinates of a and a respectively.

We give the following analytic formula for the population risk defined above.

Theorem 2.1.

Assume vectors b i , b i 's are unit vectors.

Then, the population risk f defined in equation (2.2) satisfies that DISPLAYFORM3 whereσ k is the k-th Hermite coefficient of the function σ.

See section A.1 for a short introduction of Hermite polynomial basis.

Connection to tensor decomposition: We see from equation (2.3) that the population risk of f is essentially an average of infinite number of loss functions for tensor decomposition.

For a fixed k ∈ N, we have that the k-th summand in equation (2.3) is equal to (up to the scaling factorσ DISPLAYFORM0 where DISPLAYFORM1 We note that the objective f k naturally attempts to decompose the k-order rank-m tensor T k into m rank-1 components a 1 b ⊗k i , . . .

, a m b ⊗k m .

The proof of Theorem 2.1 follows from using techniques in Hermite Fourier analysis, which is deferred to Section A.2.Issues with optimizing f :.

It turns out that optimizing the population risk using stochastic gradient descent is empirically difficult.

FIG1 shows that in a synthetic setting where the noise is zero, the test error empirically doesn't converge to zero for sufficiently long time with various learning rate schemes, even if we are using fresh samples in iteration.

This suggests that the landscape of the population risk has some spurious local minimum that is not a global minimum.

See Section 4 for more details on the experiment setup.

4 When σ = ReLU , we have thatσ0 DISPLAYFORM2 .

For n ≥ 2 and even,σn = DISPLAYFORM3 .

For n ≥ 2 and odd,σn = 0.An empirical fix:.

Inspired by the connection to tensor decomposition objective described earlier in the subsection, we can design a new objective function that takes exactly the same form as the tensor decomposition objective function f 2 + f 4 .

Concretely, let's defineŷ = a γ(Bx) where γ =σ 2 h 2 +σ 4 h 4 and h 2 (t) = normalized probabilists' Hermite polynomials Wikipedia (2017a).

We abuse the notation slightly by using the same notation to denote the its element-wise application on a vector.

Now for each example we use ŷ − y 2 as loss function.

The corresponding population risk is DISPLAYFORM4 Now by an extension of Theorem 2.1, we have that the new population risk is equal to theσ It turns out stochastic gradient descent on the objective f (a, B) (with projection to the set of matrices B with row norm 1) converges empirically to the ground truth (a , B ) or one of its equivalent permutations.

(See Figure 3 .)

However, we don't know of any existing work for analyzing the landscape of the objective f (or f k for any k ≥ 3).

We conjecture that the landscape of f doesn't have any spurious local minimum under certain mild assumptions on (a , B ).

Despite recent attempts on other loss functions for tensor decomposition BID6 , we believe that analyzing f is technically challenging and its resolution will be potentially enlightening for the understanding landscape of loss function with permutation invariance.

See Section 4 for more experimental results.

The population risk defined in equation (2.5) -though works empirically for randomly generated ground-truth (a , B ) -doesn't have any theoretical guarantees.

It's also possible that when (a , B ) are chosen adversarially or from a different distribution, SGD no longer converges to the true parameters.

To solve this problem, we design another objective function G(·), such that the optimizer of G(·) still corresponds to the ground-truth, and G() has provably nice landscape -all local minima of G() are global minima.

In this subsection, for simplicity, we work with the case when B is an orthogonal matrix and state our main result.

The discussion of the general case is deferred to the end of this Section and Section C.We define our objective function G(B) as DISPLAYFORM0 where ϕ(·, ·) is defined as DISPLAYFORM1 and φ(·, ·, ·) is defined as DISPLAYFORM2 The rationale behind of the choices of φ and ϕ will only be clearer and relevant in later sections.

For now, the only relevant property of them is that both are smooth functions whose derivatives are easily computable.

We remark that we can sample G(·) using the samples straightforwardly -it's defined as an average of functions of examples and the parameters.

We also note that only parameter B appears in the loss function.

We will infer the value of a using straightforward linear regression after we get the (approximately) accurate value of B .Due to technical reasons, our method only works for the case when a i > 0 for every i.

We will assume this throughout the rest of the paper.

The general case is left for future work.

Let a max = max a i , a min = min a i , and κ = max a i / min a i .

Our result will depend on the value of κ .

Essentially we treat κ as an absolute constant that doesn't scale in dimension.

The following theorem characterizes the properties of the landscape of G(·).

Theorem 2.3.

Let c be a sufficiently small universal constant (e.g. c = 0.01 suffices) and suppose the activation function σ satisfiesσ 4 = 0.

Assume µ ≤ c/κ , λ ≥ c −1 a max , and B is an orthogonal matrix.

The function G(·) defined as in equation (2.7) satisfies that 1.

A matrix B is a local minimum of G if and only if B can be written as B = DP B where P is a permutation matrix and D is a diagonal matrix with D ii ∈ {±1 ± O(µa max /λ)}.

Furthermore, this means that all local minima of G are also global.2.

Any saddle point B has a strictly negative curvature in the sense that λ min (∇ 2 G(B)) ≥ −τ 0 where τ 0 = c min{µa min /(κ d), λ} 3.

Suppose B is an approximate local minimum in the sense that B satisfies DISPLAYFORM0 Then B can be written as B = P DB + EB where P is a permutation matrix, D is a diagonal matrix satisfying the same bound as in bullet 1, and |E| ∞ ≤ O(ε/(σ 4 a min )).As a direct consequence, B is O d (ε)-close to a global minimum in Euclidean distance, where O d (·) hides polynomial dependency on d and other parameters.

The theorem above implies that we can learn B (up to permutation of rows and sign-flip) if we take λ to be sufficiently large and optimize G(·) using stochastic gradient descent.

In this case, the diagonal matrix D in bullet 1 is sufficiently close to identity (up to sign flip) and therefore a local minimum B is close to B up to permutation of rows and sign flip.

The sign of each b i can be recovered easily after we recover a (see Lemma 2.5 below.)

SGD converges to a local minimum BID7 (under the additional property as established in bullet 2 above), which is also a global minimum for the function G(·).

We will prove the theorem in Section B as a direct corollary of Theorem B.1.

The technical bullet 2 and 3 of the theorem is to ensure that we can use SGD to converge to a local minimum as stated below.

Corollary 2.4.

In the setting of Theorem 2.3, we can use stochastic gradient descent to optimize function G(·) (with fresh samples at each iteration) and converge to an approximate global minimum B that is ε-close to a global minimum in time poly(d, 1/ε).After approximately recovering the matrix B , we can also recover the coefficient a easily.

Note that fixing B, we can fit a using simply linear regression.

For the ease of analysis, we analyze a slightly different algorithm.

The lemma below is proved in Section D. Lemma 2.5.

Given a matrix B whose rows have unit norm, and are δ-close to B in Euclidean distance up to permutation and sign flip with δ ≤ 1/(2κ ).

Then, we can give estimates a, B (using e.g., Algorithm 1) such that there exists a permutation P where a − P a ∞ ≤ δa max and B is row-wise δ-close to P B .The key step towards analyzing objective G(B) is the following theorem that gives an analytic formula for G(·).5 More precisely, |Dii| = DISPLAYFORM0 In the most general setting, converging to a local minimum of a non-convex function is NP-hard.

Theorem 2.6.

The function G(·) satisfies DISPLAYFORM1 Theorem 2.6 is proved in Section A.

We will motivate our design choices with a brief overview in Section 3 and formally analyze the landscape of G in Section B (see Theorem B.1).Finite sample complexity bounds.

Extending Theorem 2.3, we can characterize the landscape of the empirical risk G, which implies that stochastic gradient on G also converges approximately to the ground-truth parameters with polynomial number of samples.

Theorem 2.7.

In the setting of Theorem 2.3, suppose we use N empirical samples to approximate G and obtain empirical risk G. There exists a fixed polynomial poly(d, 1/ε) such that if N ≥ poly(d, 1/ε), then with high probability the landscape of G has the properties to that of G in bullet 2 and 3 of Theorem 2.3.All of the results above assume that B is orthogonal.

Since the local minimum are preserved by linear transformation of the input space, these results can be extended to the general case when B is not orthogonal but full rank (with some additional technicality) or the case when the dimension is larger than the number of neurons (m < d).

See Section C.

In this section, we present a general overview of ideas behind the design of objective function G(·).

Inspired by the formula (2.3), in Section 3.1, we envision a family of possible objective functions for which we have unbiased estimators via samples.

In Section 3.2, we pick a specific function that feeds our needs: a) it has no spurious local minimum; b) the global minimum corresponds to the ground-truth parameters.

Recall that in equation (2.2) of Theorem 2.1 we give an analytic formula for the straightforward population risk f .

Although the population risk f doesn't perform well empirically, the lesson that we learn from it help us design better objective functions.

One of the key fact that leads to the proof of Theorem 2.1 is that for any continuous and bounded function γ, we have that DISPLAYFORM0 Hereσ k andγ k are the k-th Hermite coefficient of the function σ and γ.

That is, letting h k the k-th normalized probabilists' Hermite polynomials Wikipedia (2017a) and ·, · be the standard inner product between functions, we haveσ k = h k , σ .Note that γ can be chosen arbitrarily to extract different terms.

For example, by choosing γ = h k , we obtain that DISPLAYFORM1 That is, we can always access functions forms that involves weighted sum of the powers of b i , b i , as in RHS of equation (3.1).

Using a bit more technical tools in Fourier analysis (see details in Section A), we claim that most of the symmetric polynomials over variables b i , b j can be estimated by samples: Claim 3.1 (informal).

For any polynomial p() over a single variable, there exits a corresponding function φ p such that DISPLAYFORM2 Moreover, for an any polynomial q(·, ·) over two variables, there exists corresponding φ q such that DISPLAYFORM3 We will not prove these two general claims.

Instead, we only focus on the formulas in Theorem A.5 and Theorem A.6, which are two special cases of the claims above.

Motivated by Claim A.3, in the next subsection, we will pick an objective function which has no spurious local minimum among those functional forms on the right-hand sides of equation (3.2) and (3.3).

As discussed briefly in the introduction, one of the technical difficulties to design and analyze objective functions for neural networks comes from the permutation invariance -if a matrix B is a good solution, then any permutation of the rows of B still gives an equally good solution (if we also permute the coefficients in a accordingly).

We only know of a very limited number of objective functions that guarantee to enjoy permutation invariance and have no spurious local minima BID7 .

We start by considering the objective function used in BID7 , DISPLAYFORM0 Note that here we overload the notation by using b i 's to denote a set of fixed vectors that we wanted to recover and using b i 's to denote the variables.

Careful readers may notice that P (B) doesn't fall into the family of functions that we described in the previous section (that is, RHS equation of (3.2) and (3.3)), because it lacks the weighting a i '

s. We will fix this issue later in the subsection.

Before that we first summarize the nice properties of the landscape of P (B).For the simplicity of the discussion, let's assume B forms an orthonormal matrix in the rest of the subsection.

Then, any permutation and sign-flip of the rows of B leads to a global minimum of P (·) -when B = SQB with a permutation matrix Q and a sign matrix S (diagonal with ±1), we have that P (B) = 0 because one of b i , b j 2 and b i , b k 2 has to be zero for all i, j, k 7 ).It turns out that these permutations/sign-flips of B are also the only local minima 8 of function P (·).

To see this, notice that P (B) is a degree-4 polynomial of B. Thus if we pick an index s and fix every row except for b s , then P (B) is a quadratic function over unit vector b s -reduces to a smallest eigenvector problem.

Eigenvector problems are known to have no spurious local minimum.

Thus the corresponding function (w.r.t b s ) has no spurious local minimum.

It turns out the same property still holds when we treat all the rows as variables and add the row-wise norm constraints.

However, there are two issues with using objective function P (B).

The obvious one is that it doesn't involve the coefficients a i 's and thus doesn't fall into the forms of equation (3. 3).

Optimistically, we would hope that for nonnegative a i 's the weighted version of P below would also enjoy the similar landscape property DISPLAYFORM1 When a i 's are positive, indeed the global minimum of P are still just all the permutations of the B .

9 However, when max a i > 2 min a i , we found that P starts to have spurious local minima .

It seems that spurious local minimum often occurs when a row of B is a linear combination of a smaller number of rows of B .

See Section F for a concrete example.

To remove such spurious local minima, we add a regularization term below that pushes each row of B to be close to one of the rows of B , DISPLAYFORM2 We see that for each fixed j, the part in R(B) that involves b j has the form DISPLAYFORM3 This is commonly used objective function for decomposing tensor i a i b i ⊗4 .

It's known that for orthogonal b i 's, the only local minima are ±b 1 , . . .

, ±b d BID7 .

Therefore, intuitively R(B) pushes each of the b i 's towards one of the b i 's.10 Choosing µ to be small enough, it turns out that P (B) + R(B) doesn't have any spurious local minimum as we will show in Section B.Another issue with the choice of P (B) + R(B) is that we are still having a constraint minimization problem.

Such row-wise norm constraints only make sense when the ground-truth B is orthogonal and thus has unit row norm.

A straightforward generalization of P (B) to non-orthogonal case requires some special constraints that also depend on the covariance matrix B B , which in turn requires a specialized procedure to estimate.

Instead, we move the constraints into the objective function by considering adding another regularization term that approximately enforces the constraints.

It turns out the following regularizer suffices for the orthogonal case: DISPLAYFORM4 2 .

Moreover, we can extend this easily to the non-orthogonal case (see Section C) without estimating any statistics of B in advance.

We note that S(B) is not the Lagrangian multiplier and it does change the global minima slightly.

We will take λ to be large enough so that b i has to be close to 1.

As a summary, we finally use the unconstrained objective DISPLAYFORM5 Since R(B) and S(B) are degree-4 polynomials of B, the analysis of G(B) is much more delicate, and we cannot use much linear algebra as we could for P (B).

See Section B for details.

Finally we note that a feature of this objective G(·) is that it only takes B as variables.

We will estimate the value of a after we recover the value of B. (see Section D).

In this section, we provide simple simulation results that verify that minimizing G(B) with SGD recovers a permutation of B ; however, minimizing Equation (2.2) with SGD results in finding spurious local minima.

Based on the formula for the population risk in Equation (2.3), we also verified empirically the conjecture that SGD would successfully recover B using the activation functions γ(z) =σ 2 h 2 (z) +σ 4 h 4 (z), 11 even if the data were generated via a model with ReLU activation.

(See Section 2.1 for the rationale behind such conjectures.)

For all of our experiments, we chose B = Id d×d with dimension d = 50 and a = 1 for simplicity, and the data is generated from a one-hidden-layer network with ReLU or Sigmoid activation without noise.

We use stochastic gradient descent with fresh samples at each iteration, and we plot the (expected) population error (that is, the error on a fresh batch of examples).To test whether SGD converges to a matrix B which is equivalent to B up to permutation of rows, we use a surrogate error metric to evaluate whether B −1 B is close to a permutation matrix.

Given a matrix Q with row norm 1, let DISPLAYFORM0 (4.1)Then we have that if e(Q) ≤ ε for some ε < 1/3, then it implies that Q is √ 2ε-close to a permutation matrix in infinity norm.

On the other direction, we know that if e(Q) > ε, then Q is not ε-close to any permutation matrix in infinity norm.

The latter statement also holds when Q doesn't have row norm 1.

FIG1 shows that without over-parameterization, using ReLU as an activation function, SGD doesn't converge to zero test error and the ground-truth parameters.

We decreased step-size by a 11 We also observed that using γ(z) = 1 2 |z| also works but due to the space limitation we don't report the experimental results here.

factor of 4 every 5000 number of iterations after the error plateaus at 10000 iterations.

For the final 5000 iterations, the step-size is less than 10 −9 , so we can be confident that the non-zero objective value is not due to the variance of SGD.

We see that none of the five runs of SGD converged to a global minimum.

Figure 2 shows the result for sigmoid activation which is quantitatively similar.

Figure 3 shows that usingσ 2 h 2 +σ 4 h 4 as the activation function, SGD with projection to the set of matrices B with row norm 1 converges to the ground-truth parameters.

We also plot the loss function which converges the value of a global minimum.

(We subtracted the constant term in equation (2.6) so that the global minimum has loss 0.)

FIG3 shows that using our objective function G(B), the iterate converges to a permutation of the ground truth matrix B .

The fact that the parameter error goes up and down is not surprising, because the algorithm first gets close to a saddle point and then breaks ties and converges to a one of the global minima.

Finally we note that using the loss function G(·) seems to require significantly larger batch (and sample complexity) to reduce the variance in the gradients estimation.

We used batch size 262144 in the experiment for G(·).

However, in contrast, for theσ 2 h 2 +σ 4 h 4 we used batch size 8192 and for relu we used batch size 256.

In this paper we first give an analytic formula for the population risk of the standard 2 loss, which empirically may converge to a spurious local minimum.

We then design a novel population loss that is guaranteed to have no spurious local minimum.

Designing objective functions with well-behaved landscape is an intriguing and potentially fruitful direction.

We hope that our techniques can be useful for characterizing and designing the optimization landscape for other settings.

We conjecture that the objective αf 2 + βf 4 12 has no spurious local minimum when α, β are reasonable constants and the ground-truth parameters are in general position.

We provided empirical evidence to support the conjecture.

Our results assume that the input distribution is Gaussian.

Extending them to other input distributions is a very interesting open problem.

2 /2 ) in the following sense 13 .

For two functions f, g that map R to R, define the inner product f, g with respect to the Gaussian measure as DISPLAYFORM0 The polynomials h 0 , . . .

, h m , . . .

are orthogonal to each other under this inner product: DISPLAYFORM1 2 /2 ) , let the k-th Hermite coefficient of σ be defined asσ DISPLAYFORM2 Since h 0 , . . .

, h m , . . . , forms a complete orthonormal basis, we have the expansion that DISPLAYFORM3 We will leverage several other nice properties of the Hermite polynomials in our proofs.

The following claim connects the Hermite polynomial to the coefficients of Taylor expansion of a certain exponential function.

It can also serve as a definition of Hermite polynomials.

13 We denote by Donnell, 2014, Equation 11 .8)).

We have that for t, z ∈ R, DISPLAYFORM4 DISPLAYFORM5 The following Claims shows that the expectation E [h n (x)h m (y)] can be computed easily when x, y are (correlated) Gaussian random variables.

Claim A.2 ((O'Donnell, 2014, Section 11.2)).

Let (x, y) be ρ-correlated standard normal variables (that is, both x,y have marginal distribution N (0, 1) and E[xy] = ρ).

Then, DISPLAYFORM6 As a direct corollary, we can compute Ex∼N (0,Id d×d ) σ(u x)γ(v x) by expanding in the Hermite basis and applying the Claim above.

Claim A.3.

Let σ, γ be two functions from R to R such that DISPLAYFORM7 2 /2 ).

Then, for any unit vectors u, v ∈ R d , we have that DISPLAYFORM8 Proof of Claim A.3.

Let s = u x and t = v x. Then s, t are two spherical standard normal random variables that are u, v -correlated, and we have that DISPLAYFORM9 We expand σ(s) and γ(t) in the Fourier basis and obtain that DISPLAYFORM10 In this section we prove Theorem 2.1 and Theorem 2.2, which both follow from the following more general Theorem.

DISPLAYFORM11 2 /2 ), andŷ = a γ(Bx) with parameter a ∈ R and B ∈ R ×d .

Define the population risk f γ as DISPLAYFORM12 DISPLAYFORM13 whereσ k ,γ k are the k-th Hermite coefficients of the function σ and γ respectively.

We can see that Theorem 2.1 follows from choosing γ = σ and Theorem 2.2 follows from choosing γ =σ 2 h 2 +σ 4 h 4 .

The key intuition here is that we can decompose σ into a weighted combination of Hermite polynomials, and each Hermite polynomial influence the population risk more or less independently (because they are orthogonal polynomials with respect to the Gaussian measure).Proof of Theorem A.4.

We have DISPLAYFORM14

In this section we show that the population risk G(·) (defined as in equation (2.7)) has the following analytical formula: DISPLAYFORM0 The formula will be crucial for the analysis of the landscape of G(·) in Section B. The formula follows straightforwardly from the following two theorems and the definition (2.7).

Theorem A.5.

Let φ(·, ·, ·) be defined as in equation (2.9), we have that DISPLAYFORM1 Theorem A.6.

Let ϕ(·, ·) be defined as in equation (2.8), then we have that DISPLAYFORM2 In the rest of the section we prove Theorem A.5 and A.6.We start with a simple but fundamental lemma.

Essentially all the result in this section follows from expanding the two sides of equation (A.1) below.

Lemma A.7.

Let u, v ∈ R d be two fixed vectors and x ∼ N (0, Id d×d ).

Then, for any s, t ∈ R, DISPLAYFORM3 Proof.

Using the fact that E exp(v x) = exp( 1 2 v 2 ), we have that, DISPLAYFORM4 Next we extend some of the results in the previous section to the setting with different scaling (such as when v in Claim A.3 is no longer a unit vector.)

Lemma A.8.

Let u be a fixed unit vector and v be an arbitrary vector in DISPLAYFORM5 As a sanity check, we can verify that when v is a unit vector, ϕ(v, x) = √ 24h 4 (v x) and th Lemma reduces to a special case of Claim A.2.Proof.

Let A, B be formal power series in variable s, t defined as A = exp( u, v st) and B = E exp(u xt − (A.3) which implies that Now we are ready prove Theorem A.6 using Lemma A.8.

DISPLAYFORM6 DISPLAYFORM7 Proof of Theorem A.6.

Using the fact that σ(v x) = ∞ k=0σ k h k (v x), we have that DISPLAYFORM8 (by Lemma A.8 and DISPLAYFORM9 Towards proving Theorem A.5, we start with the following Lemma.

Inspired by the proofs above, we design a function φ(v, w, x) such that we can estimate u, v 2 u, w 2 by taking expectation of E σ(u x)φ(v, w, x) .

Lemma A.9.

Let a be a fixed unit vector in R d and v, w two fixed vectors in R d .

Let ϕ(·, ·) be defined as in Lemma A.8.

Define φ(v, w, x) as DISPLAYFORM10 Then, we have that DISPLAYFORM11 Proof.

Using the fact that u, v + w 2 + u, v − w 4 − 2 u, v 2 − 2 u, w 4 = 12 u, v 2 u, w 2 and Lemma A.8, we have that DISPLAYFORM12 Using the fact that σ(u x) = ∞ k=0σ k h k (u x), we conclude that DISPLAYFORM13 (by Lemma A.8 again)Now we are ready to prove Theorem A.5 by using Lemma A.9 for every summand.

Proof of Theorem A.5.

We have that DISPLAYFORM14 (by Lemma A.9)

In this section we prove Theorem 2.3.

Since the landscape property is invariant with respect to rotations of parameters, without loss of generality we assume B is the identity matrix Id throughout this section.

(See Section C for a precise statement for the invariance.)

Recall that by Theorem 2.6, the population risk G(·) in the case of B = Id is equal to DISPLAYFORM0 In the rest of section we work with the formula above for G(·) instead of the original definition.

In fact, for future reference, we study a more general version of the function G. For nonnegative vectors α, β and nonnegative number µ, let G α,β,µ be defined as DISPLAYFORM1 Here e i denotes the i-th natural basis vector.

We see that G is sub-case of G α,β,µ and we prove the following extension of Theorem 2.3.

Let α max = max i α i and α min = min i α i .Theorem B.1.

Let κ α = α max /α min and c be a sufficiently small universal constant (e.g. c = 10 Then B can be written as B = DP + E where P is a permutation matrix, D is a diagonal matrix with the entries satisfying DISPLAYFORM2 and E is an error matrix satisfying |E| ∞ ≤ 3ε/β min .

Here we recall that |E| ∞ denotes the largest entries in the matrix E. Theorem 2.3 follows straightforwardly from Theorem B.1 by setting α = 2 √ 6|σ 4 |a and β = |σ 4 |a / √ 6.

In the rest of the section we prove Theorem B.1. . . .

DISPLAYFORM0 Naturally, towards analyzing the properties of a local minimum B, the first step is that we pick a row b s of B and treat only b s as variables and others rows as fixed.

We will show that local optimality of b s will imply that b s is equal to one of the basis vector e j up to some scaling factor.

This step is done in Section B.1.

Then in Section B.2 we show that the local optimality of all the variables in B implies that each of the rows of B corresponds to different basis vector, which implies that B is a permutation matrix (up to scaling of the rows).

Suppose we fix b 1 , · · · , b s−1 , b s+1 , · · · , b d , and optimize only over b s , we obtain the objective h of the following form: DISPLAYFORM0 We can see that setting α i = a i k =s (b k e i ) 2 , β i = a i , and x = b s gives us the original objective G(B).

In this subsection, we will work with h(·) and analyze the properties of the local minima of h(·).The following lemma shows that a local minimum x of the objective h(·) must be a scaling of a basis vector.

For a vector x, let |x| 2nd denotes the second largest absolute values of the entries for x. We note that |·| 2nd is not a norm.

The lemma deals generally an approximate local minimum, though we suggest casual readers simply think of ε, τ = 0 in the lemma.

Lemma B.2.

Let h(·) be defined in equation (B.3) with non-negative vectors α and β in R d .

Suppose parameters ε, τ ≥ 0 satisfy that ε ≤ τ 3 /β min .

If some point x satisfies ∇h(x) ≤ ε and λ min (∇ 2 h(x)) ≥ −τ , then we have DISPLAYFORM1 Proof.

Without loss of generality, we can take ε = τ 3 /β min which means τ = ε 2/3 β 1/3 min .

The gradient and Hessian of function h(·) are DISPLAYFORM2 where γ 4λ( x 2 − 1).Let S = {i : |x i | ≥ δ} be the indices of the coordinates that are significantly away from zero, where DISPLAYFORM3 .

Since ∇h(x) ≤ ε, we have that |∇h(x) i | ≤ ε for every i ∈ [d], which implies that DISPLAYFORM4 which further implies that DISPLAYFORM5 If |S| = 1, then we are done because |x| 2nd ≤ δ.

Next we prove that |S| ≥ 2.

For the sake of contradiction, we assume that |S| ≥ 2.

Moreover, WLOG, we assume that |x| 1 ≥ |x| 2 are the two largest entries of |x| in absolute values.

We take v ∈ R d such that v 1 = −x 2 / x 2 1 + x 2 2 , and v 2 = x 1 / x 2 1 + x 2 2 , and v j = 0 for j ≥ 2.

Then we have that v x = 0 and v = 1.

We evaluate the quadratic form and have that .

Then we conclude that DISPLAYFORM6 DISPLAYFORM7 This contradicts with the assumption that λ min (∇ 2 h(x)) ≥ −β 1/3 min ε 2/3 = τ and that v = 1.

Therefore we have |S| = 1 and DISPLAYFORM8 For future reference, we can also show that for a sufficiently strong regularization term (sufficiently large λ), the norm of a local minimum x should be bounded from below and above by 1/2 and 2.

This are rather coarse bounds that suffice for our purpose in this subsection.

In Section B.2 we will show that all the rows of a local minimum B of G have norm close to 1.

Lemma B.3.

In the setting of Lemma B.2, 1.

Suppose in addition that λ ≥ 4 max(β max , τ ) and ε ≤ 0.

DISPLAYFORM9 2.

Let i = arg max i |x i |.

In addition to the previous conditions in bullet 1, assume that λ ≥ 4α i .

Then, DISPLAYFORM10 We remark that we have to state the conditions for the upperbounds and lowerbounds separately since they will be used with these different conditions.

Proof.

Let S = {i : |x i | ≥ δ} be the indices of the coordinates that are significantly away from zero, where δ = ε βmin

.

We first show that x 2 ≤ 2.

We divide into two cases:1 S is empty.

Since ε ≤ 0.

DISPLAYFORM0 .

We conclude that x 2 ≤ 2.2 S is non-empty.

For i ∈ S, recall equation (B.6) which implies that DISPLAYFORM1 , and thus from the display above we have that x 2 ≤ 2.Next we show that x 2 ≥ 1 2 .

Again we divide into two cases:1.

S is empty.

For the sake of contradiction, assume that x 2 ≤ 1 2 , then γ ≤ −2λ.

We show that there is sufficient negative curvature.

Recall that DISPLAYFORM2 Choose index j so that α j = α min , then DISPLAYFORM3 This contradicts with the fact that λ min (∇ 2 h(x)) ≥ −τ .

Thus when S is empty, x 2 ≥ 1 2 .

2.

S is non-empty.

Recall that i = arg max i |x i |, and by definition i ∈ S. Using Equation (B.6) γ ≥ −2α i − ε δ which implies that DISPLAYFORM4 Since λ ≥ 4α i , and λ ≥ β 1/3 min ε 2/3 ≥ ε δ , we conclude that x 2 ≥ 1/2.We have shown that a local minimum x of h should be a scaling of the basis vector e i .

The following lemma strengthens the result by demonstrating that not all basis vector can be a local minimumthe corresponding coefficient α i has to be reasonably small for e i being a local minimum.

The key intuition here is that if α i is very large compared to other entries of α, then if we move locally the mass of e i from entry i to some other index j, the objective function will be likely to decrease because α j x 2 j is likely to be smaller than α i x 2 i .

(Indeed, we will show that such movement will cause a second-order decrease of the objective function in the proof.)

Lemma B.4.

In the setting of Lemma B.2, let i = arg max i |x i |.

If ∇h(x) ≤ ε, and λ min (∇ 2 h(x)) > −τ for 0 ≤ τ ≤ 0.1β min /d and ε ≤ τ 3 /β min , then DISPLAYFORM5 Proof.

For the ease of notation, assume WLOG that i = 1.

Let δ = (τ /β min ) 1/2 .

By the assumptions, we have that δ ≤ 1 √ 6d.

By Lemma B.2, we have x 2 ≥ 1 2 , which implies that DISPLAYFORM6 Define v = − x k x1 e 1 + e k .

Since x 1 is the largest entry of x, we can verify that 1 DISPLAYFORM7 By the assumption, we have that DISPLAYFORM8 On the other hand, recall the form of Hessian (equation FIG3 ), by straightforward algebraic manipulation, we have that DISPLAYFORM9 (by x ≤ 2 using Lemma B.2)Combining equation (B.8) and the equation above gives DISPLAYFORM10 Since k is arbitrary we complete the proof.

The previous lemma implies that it's very likely that the local minimum x can be written as x = x i e i and the index i is also likely to be the argmin of α.

The following technical lemma shows that when this indeed happens, then we can strengthen Lemma B.2 in terms of the error bound's dependency on ε and τ .

In Lemma B.2, we have that |x| 2nd is bounded by a function of τ .

Here we strengthen the bound to be a function that only depends on ε.

Thus as long as τ be small enough so that we can apply Lemma B.2 and Lemma B.4 to meet the condition of the lemma below, then we get an error bound that goes to zero as ε goes to zero.

This translates to the error bound in bullet 3 of Theorem B.1 where the bound on E only depends on ε.

For casual readers we suggest to skip this Lemma since its precise functionality will only be clearer in the proof of Theorem B.1.Lemma B.5.

In the setting of Lemma B.2, in addition we assume that i = argmin k |α k | and that x can be written as x = x i e i + x −i satisfying DISPLAYFORM11 Then, we can strengthen the bound to DISPLAYFORM12 DISPLAYFORM13 Combining the above two displays, g(B) ) ≥ −τ for parameters τ, ε satisfying 0 ≤ τ ≤ c min{µβ min /(κ α d), λ} and ε ≤ c min{α min , τ 3 /β min }.

Then, the matrix B can be written as DISPLAYFORM14 DISPLAYFORM15 where D is diagonal such that ∀i, |D ii | ∈ [1/4, 2], and P is a permutation matrix, and |E| ∞ ≤ δ DISPLAYFORM16 As alluded before, in the proof we will first apply the results in Section B.1 to show that when B is a local minimum, each row b s has a unique large entry.

Then we will show that the largest entries of each row sit on different columns.

The key intuition behind the proof is that if two rows, say row s, t, have their large entries on the same column, then it means that there exists a column-say column k -that doesn't contain largest entry of any row.

Then either row s or t will violate Lemma B.4.

Or in other words, either row s or t can move their mass into the column k to decrease the function value.

This contradicts the assumption that B is a local minimum.

Proof.

As pointed in the paragraph below equation (B.3), when we restrict our attention to a particular row of B and fix the rest of the rows the function G α,β,µ reduces to the function h(·) in equation (B.3) so that we can apply lemmas in Section B.1.

2 , and DISPLAYFORM0 We view the function above as h(x).

Now we apply Lemma B.2 (by replacing α, β in Lemma B.2 byᾱ,β).

The assumption that DISPLAYFORM1 Hence by Lemma B.2, we have that the second largest entry of |b s | satisfies DISPLAYFORM2 for the ease of notation.

We can check that δ ≤ 1 4 √ καd by the assumption.

Therefore, we have essentially shown that each row of B has only one single large entry, since the second largest entry is at most δ.

Next we show that each row of B has largest entries on distinct columns.

For each row j ∈ [d], let i j = arg max i |e i b j | be the index of the largest entry of b j .

We will show that i 1 , . . .

, i d are distinct.

For the sake of contradiction, suppose they are not distinct, that is, there are two distinct rows s, t that have the same largest entries on column l, that is, we assume that i s = i t =

l.

This implies that {i 1 , . . .

, DISPLAYFORM3 We note that by the DISPLAYFORM4 .

We first bound from aboveᾱ k DISPLAYFORM5 Assume in addition without loss of generality that |b s e l | ≤ |b t e l |.

Let DISPLAYFORM6 be the sum of squares of the entries on the column l without entry b j e l , and thatᾱ l = α l z l .

We first prove that z l ≥ 1/3.For the sake of contradiction, assume z l < 1/3.

Then we have thatᾱ l = α l z ≤ 1 3 α l .

This implies that λ ≥ 4 max{ᾱ l , τ }, and since l is the index of the largest column of b s we can invoke Lemma B.3 and conclude that b s 2 ≥ 1/2.

This further implies that DISPLAYFORM7 Since we have assumed that |b s e l | ≤ |b t e l |.

Then we obtain that DISPLAYFORM8 which contradicts the assumption.

Therefore, we conclude that z l ≥ 1/3.

Then we are ready to boundᾱ l from below:ᾱ DISPLAYFORM9 The display above and Equation (B.13) implies that DISPLAYFORM10 Note that l is the largest entry in absolute value in the vector b s .

We will apply Lemma B. Finally, let Q be the matrix that only contain the largest entries (in absolute value) of each columns of B. Since i 1 , . . .

, i d are distinct, we have that Q contains exactly one entry per row and per column.

Therefore Q can be written as DP where P is a permutation matrix and D is a diagonal matrix.

Moreover, we have that DISPLAYFORM11 2nd ≥ 1/4 and b s 2 ≤ 2.

Therefore, the largest entry of each row has absolute value between 1/4 and 2.

Therefore |D| ii ∈ [1/4, 2].

Let E = B − P D. Then we have that |E| ∞ ≤ max s |b s | 2nd ≤ δ,which completes the proof.

Applying Lemma B.5, we can further strengthen Proposition B.6 with better error bounds and better control of the largest entries of each column.

Proposition B.7 (Strengthen of Proposition B.6).

In the setting of Proposition B.6.

Suppose in addition that τ satisfies τ ≤ cµβ 2 min /β max .

Then, the matrix B can be written as DISPLAYFORM12 where P is a permutation matrix, D is diagonal such that DISPLAYFORM13 and DISPLAYFORM14 Published as a conference paper at ICLR 2018Proof.

By Proposition B.6, we know that DISPLAYFORM15 .

Now we use Lemma B.5 to strength the error bound.

As we have done in the proof of Proposition B.6, we again fix an arbitrary s ∈ [d] and all the rows except b s and view G α,β,µ as a function of DISPLAYFORM16 2 , and β i = µβ i and view G α,β,µ as a function of the form h(x) with α, β replaced byᾱ,β, namely, DISPLAYFORM17 We will verify the condition of Lemma B.5.

Let i be the index of the largest entry in absolute value of the vector b s .

Since we have shown that the largest entry in each row sits on different columns, and the second largest entry is always less than δ, we have that, DISPLAYFORM18 For any k = i, we know that the column k contains some entry (k, j k ) which is the largest entry of some row, and we also have that j k = s since the largest entry of row s is on column i.

Therefore, we have thatᾱ DISPLAYFORM19 Therefore,ᾱ k ≥ᾱ i for any k = i and thus i = argmin k |ᾱ k |.

By the fact that |E| ∞ ≤ δ, we have that DISPLAYFORM20 }.

Now we are ready to apply Lemma B.5 and obtain that |b s | 2nd ≤ 3ε βmin .

Applying the argument for every row s gives |E| ∞ ≤ 3ε βmin .

Finally, we give the bound for the entires in D. Let v be a short hand for ∇h(b s ) which is equal to the s-th column of ∇G(B).

Since B is an ε-approximate stationary point, then we have that v ≤ ε and by straightforward calculation of the gradient, we have DISPLAYFORM21 Since x i = 0, dividing by x i gives, DISPLAYFORM22 Rearranging the equation above gives, DISPLAYFORM23 i , we note that |v i | < ε,ᾱ i > 0, and j =i x 2 j > 0, so DISPLAYFORM24 Moreover, we have proved that each rows has largest entry at different columns.

Also note that the largest entry of row b s is on column i.

Therefore, we haveᾱ i = α i j =s (b T j e i ) 2 ≤ α max dδ 2 .

Using these two estimates and δ = 3ε βmin , we have DISPLAYFORM25 Finally we are ready to prove Theorem B.1 by applying Proposition B.6.Proof of Theorem B.1.

By setting ε = 0, τ = 0 in Proposition B.6, we have that any local minimum B satisfies that B = DP where P is a permutation matrix and D is a diagonal and the precise diagonal entries of D. It can be verified that all these points have the same function value, so that they are all global minimizers.

Towards proving the second bullet, we note that a saddle point B satisfies that ∇G(B) = 0.

We will prove that λ min (∇ 2 G(B)) ≤ −τ 0 .

For the sake of contradiction, suppose λ min (∇ 2 G(B)) ≥ −τ 0 .

Then setting ε = 0 and τ = τ 0 in Propostion B.7, we have that B = DP and D ii = ± 1 1−µβi/λ , which by bullet 1 implies that B is a local minimum.

This contradicts the assumption that B is a saddle point.

The 3rd bullet is a just a rephrasing of Proposition B.7.

In this section, we first show that when the weight vectors {b i } s are not orthonormal, the local optimum of a slight variant of G(B) still allow us to recover B .

The main observation is that the set of local minima are preserved (in a certain sense) by linear transformation of the variables.

We design an objective function F (B) that is equivalent to G(B) up to a linear transformation.

This allows us to use Theorem 2.3 as a black box to characterize all the local minima of F .We use λ max (·), λ min (·) to denote the largest and smallest eigenvalues of a square matrix.

Similarly, σ max (·) and σ min (·) are used to denote the largest and smallest singular values.

Given a function f (y), we say function g(·) is a linear transformation of f (·) if there is a matrix W such that g(x) = f (W x).

If W has full rank, the local minima of f are closely related to the local minima of g.

We recall some standard notation in calculus first.

We use ∇f (t) to denote the gradient of f evaluated at t. For example, ∇f (W x) is a shorthand for ∂f (y) ∂y | y=W x , and similarly DISPLAYFORM0 The following theorem then connects the gradients and Hessians of f (W x) and g(x).

Essentially, it shows that the set of local minima and saddle points have a 1-1 mapping between f and g, and the corresponding norms/eigenvalues only differ multiplicatively by quantities related to the spectrum of W .

Theorem C.1.

Let W ∈ R d×m (d ≥ m) be a full rank matrix.

Suppose g : R m → R and f : R d → R are twice-differentiable functions such that g(x) = f (W x) for any x ∈ R m .

Then, for all x ∈ R m , the following three properties hold: DISPLAYFORM1 3.

The point x satisfies the first and second order optimality condition for g iff y = W x also satisfy the first and second order optimality condition for f .Proof.

The proof follows from the relationship between the gradients of g and the gradients of f .

By basic calculus, we have DISPLAYFORM2 which immediately implies bullet 1.

Similarly, we can compute the second order derivative: DISPLAYFORM3 To simplify notation, let A = ∇ 2 f (W x).

Let x = arg min x =1 x W AW x, and y = (W x)/ W x .

Therefore DISPLAYFORM4 On the other hand, let y be the unit vector that minimizes y Ay, we know y is in column span of W because f is only defined on the row span, so there must exist a unit vector x such that W x = λy where λ ≥ σ min (W ).

For this x we have DISPLAYFORM5 .

This finishes the proof for 2.

Finally, notice that W is full rank, so ∇g( DISPLAYFORM6

Now we will design a new objective function that can be linearly transformed to the orthonormal case.

The main idea is to view the rows of B as the new basis that we work on (which is not necessarily orthogonal).

Note that this is already the case for the first two terms of the objective function G(B), we change the objective function as follows: More concretely, we define DISPLAYFORM0 Note that the only change in the objective is the regularizer for the norm of b j .

It is now replaced by (( DISPLAYFORM1 2 , which tries to ensure the "norm" of b j in the basis defined by row of B to be 1.

The objective function that we will optimize corresponds to choosing α i = a i .

Similar as before, this function can be computed as expectations DISPLAYFORM2 where (x , y ) is an independent sample, and φ 2 (v, DISPLAYFORM3 Intuitively, if we can find a linear transformation that makes {b i }'s orthonormal, that will reduce the problem to the orthonormal case.

This is in fact the whitening matrix: DISPLAYFORM4 DISPLAYFORM5 For notational convenience, let's extend the definition of the G(B) in equation by using the putting the relevant information in the subscript DISPLAYFORM6 (That is, the index o denotes the ground-truth solution with respect to which G is defined.)The next Theorem shows that we can rotate the objective function F properly so that it matches the objective G with a ground-truth vector o i 's. Theorem C.2.

Let W be defined as above, and let 1/a be the vector whose i-th entry is 1/a i .

Then, we have that DISPLAYFORM7 Note this can be interpreted as a linear transformation as in vector format BW is equal to B · (W ⊗ Id d×d ).Proof.

The equality can be obtained by straightforward calculation.

We note that since B =   b Therefore, we have that DISPLAYFORM8 (by the definition of o i 's)From Theorem 2.3 we can immediately get the following Corollary (note that the only difference is that the coefficients now are 1/a i instead of a i ).

Recall a max = max i a i and a min = min i a min , we havePublished as a conference paper at ICLR 2018 Corollary C.3.

Let κ a = a max /a min .

Let c be a sufficiently small universal constant (e.g. c = 0.01 suffices).

Assume µ ≤ c/κ a and λ ≥ (ca min ) −1 .

The function G 1/a ,µ,λ,oi (·) defined as in Theorem C.2 satisfies that 1.

A matrix B is a local minimum of G if and only if B can be written as B = P DO where O is a matrix whose rows are o i 's, P is a permutation matrix and D is a diagonal matrix with D ii ∈ {±1 ± O(µ/λa min )}.2.

Any saddle point B has a strictly negative curvature in the sense that λ min (∇ 2 G(B)) ≥ −τ 0 where τ 0 = c min{µ/(κ a a max d), λ} 3.

Suppose B is an approximate local minimum in the sense that B satisfies ∇g(B) ≤ ε and λ min (∇ 2 g(B)) ≥ −τ 0Then B can be written as B = P DO +E where P is a permutation matrix, D is a diagonal matrix and |E| ∞ ≤ O(εa max /σ 4 ).Finally, we can combine the theorem above and Theorem B.1 to give a guarantee for optimizing F .

Let Γ be a diagonal matrix with DISPLAYFORM9 Theorem C.4.

Let c be a sufficiently small universal constant (e.g. c = 0.01 suffices).

Let κ a = a max /a min .

Assume µ ≤ c/κ a and λ ≥ 1/(c · a min ).

The function F (·) defined as in Theorem C.2 satisfies that 1.

A matrix B is a local minimum of F if and only if B satisfy B − = P DΓB where P is a permutation matrix, Γ is a diagonal matrix with Γ ii = a i , and D is a diagonal matrix with D ii ∈ {±1 ± O(µ/λa min )}.

Furthermore, this means that all local minima of F are also global.2.

Any saddle point B has a strictly negative curvature in the sense that λ min (∇ 2 F (B)) ≥ −τ 0 where τ 0 = c min{µ/(κ a da max ), λ}σ min (M ).3.

Suppose B is an approximate local minimum in the sense that B satisfies DISPLAYFORM10 Then B can be written as B − = P DΓB + E where Γ, D, P are as in 1, the error term DISPLAYFORM11 Proof.

Note that we can immediately apply Theorem 2.3 to G 1/a ,µ,λ,oi (B) to characterize all its local minima.

See Corollary C.3.Next we will transform the properties for local minima of G (stated in Corollary C.3) to F using Theorem C.1.

First we note that the transformation matrix W and M are closely related:

DISPLAYFORM12 This is because according to the definition of W , the SVD of M is M = U DU and DISPLAYFORM13 The claims of the singular values follow immediately from the SVD of M and W .As a result, all local minimum of F are of the form BW where B is a local minimum of G. For B = BW , the gradient and Hessian of F (B) and G(B) are also related by Theorem C.1.Let us first prove 1.

By Corollary C.3, we know every local minimum of G is of the form B = P DO.

According to the definition of O in Theorem C.2, we know each row vector o i is equal to W (a i ) 1/2 b i , therefore O = ΓB W .

As a result, all local minima of G are of the form B = P DΓB W .

By Theorem C.1 and Theorem C.2, we know all local minima of F must be of the form B = BW = P DΓB W W = P DΓB M −1 .

DISPLAYFORM14 Note that P is still a permutation matrix, and D − is still a matrix whose diagonal entries are {±1 ± O(µ/λa min )}, so this is exactly the form we stated in 1.

More concretely, the rows of B − are permutations of a i b i .For bullet 2, it follows immediately from Property 2 in Theorem C.1.

Note that by property 2, DISPLAYFORM15 Finally we will prove 3.

Let B = BW − , so that G(B) = F (B).

We will prove properties of B using the properties of B from Corollary C.3.First we observe that by Theorem C.1, DISPLAYFORM16 Therefore the second order condition for Claim 3 in Corollary C.3 is satisfied.

Now when DISPLAYFORM17 .

By Corollary C.3, we know B can be expressed as P DO + E where D is the diagonal matrix, P is a permutation matrix and DISPLAYFORM18 1/2 )).

We will apply perturbation Theorem C.9 for matrix inversion.

Since σ min (P DO) ≥ 1/2, we know when E ≤ 1/4, DISPLAYFORM19 Here E is bounded by DISPLAYFORM20 , which is smaller than 1/4 when ε is small enough.

The corresponding point in F is B = BW , and in 1 we have already proved (P DOW )− is of the form we want, therefore we can define DISPLAYFORM21 , and DISPLAYFORM22 This finishes the proof.

The objective function F can handle the case when the weights b i 's are not orthogonal, but still requires the number of components m to be equal to the number of dimensions d. In this section we show how to use similar ideas for the case when the number of components is smaller than the dimension (m < d).Note that all the terms in F (B) only depends on the inner-products b j , b i .

Let S be the span of {b i }'s and P S be the projection matrix to this subspace, it is easy to see that F (B) satisfies DISPLAYFORM0 That is, the previous objective function only depends on the projection of B in the space S. Using similar argument as Theorem C.4, it is not hard to show the only local optimum in S satisfies the same conditions, and allow us to recover B .

However, without modifying the objective, the local optimum of F (B) can have arbitrary components in the orthogonal subspace S ⊥ .In order to prevent the components from S ⊥ , we add an additional 2 regularizer: define F α,µ,λ,δ as follows: DISPLAYFORM1 Intuitively, since the first term F α,µ,λ (B) only cares about the projection BP S , minimizing B 2 F will remove the components in the orthogonal subspace of S. We will choose δ carefully to make

We will show in this section that if we have are given a δ-approximation of B , then it is easy to recover a .

The key observation here is that the correlation between the b i , x and the output y is exactly proportional to a i .

We also note that there could be multiple other ways to recover a , e.g., using linear regression with the σ(Bx) as input and the y as output.

We chose this algorithm mostly because of the ease of analysis.

Algorithm 1 Recovering a Input: A matrix B with unit row norms that is row-wise δ-close to B in Euclidean distance.

Return: DISPLAYFORM0 Lemma (Restatement of Lemma 2.5).

Given a matrix B whose rows are δ-close to B in Euclidean distance up to permutation and sign flip with δ ≤ 1/(2κ ).

Then, we can give estimates a, B (using e.g., Algorithm 1) such that there exists a permutation P where a − P a ∞ ≤ δa max and B is row-wise δ-close to P B .To see why this simple algorithm works for recovering a , we need the following simple claim.

Claim D.1.

For any vector v we have DISPLAYFORM1 The proof of this claim follows immediately from the property of Hermite polynomials.

Now we are ready to prove the corollary.

Proof.

Without loss of generality we assume B is close to a sign flip of B .

The unknown permutation does not change the proof.

Since b i is δ close to B i , let u be the vector where DISPLAYFORM2 Therefore a i is always positive, a i is in the desirable range and B i − B i ≤ δ.

Similarly, if −b i is δ close to B i , we have a i ∈ −a i ± a max δ, and the conclusion still holds.

For the settings considered in Section C, the vectors b i are not necessarily orthogonal.

In this case we use the following algorithm:Algorithm 2 Recovering a for general case Input: A matrix B with unit row norms, and B is δ-close to B in spectral norm up to permutation and sign flip.

Proof.

We again use Claim D.1: in this case we know the vector u satisfies u = B(B ) a .

As a result, for the vector a , we have a = (BB ) −1 (B(B ) )a = (B † ) (B ) a = (B B † ) a .By assumption we know B = SP B + E where E ≤ δ.

By the perturbation of matrix inverse (Theorem C.9), we know if E ≤ δ ≤ σ min (B)/2, then B † = (B ) † P −1 S −1 + E where E ≤ 2 √ 2σ min (B) −2 δ.

Therefore a = (P −1 S −1 + E ) a = S − P − a + (E ) a = SP a + (E ) a .(Here the last equality is because for both permutation matrix P and sign flip matrix S, P − = P and S − = S.) Therefore, coordinates of a are permutation and sign flips of a , up to an error term (E ) a . , and B − P B ≤ δ.

In this section we will show that our algorithm only requires polynomially many samples to find the desired solution.

Note that we did not try to optimize the polynomial dependency.

Theorem E.1 (Theorem 2.7 Restated).

In the setting of Theorem 2.3, suppose we use N empirical samples to approximate G and obtain function G. There exists a fixed polynomial such that if N ≥ poly(d, a max /a min , 1/ε), with high probability for any point B with λ min (∇ 2 G(B)) ≥ −τ 0 /2 and ∇ G(B) ≤ ε/2, then B can be written as B = DP + E where P is a permutation matrix, D is a diagonal matrix and |E| ∞ ≤ O(ε/(σ 4 a min )).In order to bound the sample complexity, we will prove a uniform convergence result: we show that with polynomially many samples, the gradient and Hessian of G are point-wise close to the gradient and Hessian of G, therefore any approximate local minimum of G must also be an approximate local minimum of G.However, there are two technical issues in showing the uniform convergence result.

The first issue is that when the norm of B is very large, both the gradient and Hessian of G and G are very large and we cannot hope for good concentration.

We deal with this issue by showing when B has a large norm, the empirical gradient ∇ G(B) must also have large norm, and therefore it can never be an approximate local minimum (we do this later in Lemma E.5).

The second issue is that our objective function involves high-degree polynomials over Gaussian variables x, y, and is therefore not subGaussian or sub-exponential.

We use a standard truncation argument to show that the function does not change by too much if we restrict to the event that the Gaussian variables have bounded norm.

Lemma E.2.

Suppose P (B) + R(B) = E (x,y) [f (x, y, B)] where f is a polynomial of degree at most 5 in x, y and at most 4 in B. Also assume that the sum of absolute values of coefficients is bounded by Γ. For any ε ≤ Γ/2, let R = Cd log(a max Γ/ε) for a large enough constant C, let F be the event that x 2 ≤ R, and let G trunc = E (x,y) [f (x, y, B)1 F ].

For any B such that b i ≤ 2 for all rows, we have ∇G(B) − ∇G trunc (B) ≤ ε, and ∇ 2 G(B) − ∇ 2 G trunc (B) ≤ ε,Proof.

By standard χ 2 concentration bounds, for large enough C and any z > R, the probability that x 2 ≥ z is at most exp(−10z).By simple calculation, it is easy to check that ∇ B f (x, y, B) ≤ 4Γd 1.5 a max x 5 , and ∇ 2 B f (x, y, B) ≤ 12Γd 2 a max x 5 .

We know ∇G(B)−∇G trunc (B) = E[∇ B f (x, y, B)(1− 1 F ) .

The expectation between x 2 ∈ [2 i R, 2 i+1 R], for i = 0, 1, 2, ..., is always bounded by 4Γd 1.5 a max 2 i+1 R 5 exp(−2 i R) < ε/2 i+1 .

Therefore DISPLAYFORM0 The bound for the Hessian follows from the same argument.

Finally, we combine this truncation with a result of BID20 that proves universal convergence of gradient and Hessian.

For completeness here we state a version of their theorem with bounded gradient/Hessian: Theorem E.3 (Theorem 1 in BID20 ).

Let f (θ) be a function from R p → R andf be its empirical version.

If the norm of the gradient and Hessian of a function is always bounded by τ , for variables in a ball of radius r in p dimensions, there exists a universal constant C 0 such that for C = C 0 max{log rτ /δ, 1}, the following hold: As an immediate corollary of this theorem and Lemma E.2, we have Corollary E.4.

In the setting of Theorem 2.7, for every B whose rows have norm at most 2, we have with high probability, ∇G(B) − ∇ G(B) ≤ ε/2, and ∇ 2 G(B) − ∇ 2 G(B) ≤ τ 0 /2.Proof.

On the other hand, for all such matrices B, by Lemma E.2 we know the gradient and Hessian of G is close to the gradient and Hessian of G trunc .

The corollary then follows from triangle inequality.

Finally we handle the case when B has a row with large norm.

We will show that in this case ∇ G(B) must also be large, so B cannot be an approximate local minimum.

Lemma E.5.

If b i is the row with largest norm and b i ≥ 2, then when N ≥ poly(d, a max /a min ) for some fixed polynomial, we have with high probability ∇ G(B), b i ≥ cλ b i 4 for some universal constant c > 0.Assume we have a local perturbation B , where b 1 = 1 − ε 2 1 e 1 + ε 1 u 1 , b 2 = 1 − ε 2 2 e 1 + ε 2 u 2 .

Here u 1 , u 2 are unit vectors that are orthogonal to e 1 .

Also, since this is a local perturbation, we make sure ε 1 , ε 2 ≤ ε, and b 3 (2) ≥ 1 − ε, [b 4 (3)] 2 , [b 4 (4)] 2 ≥ 0.5 − ε.

We will show that when ε is small enough, the objective function P (B ) ≥ 1.

Similarly we have the same equation for b 2 .

Note that all the terms we analyzed are disjoint, therefore P (B ) ≥ (1 − ε 2 1 )(1 − ε 2 2 ) + ε 2 1 (2 + δ)(0.5 − ε) + ε 2 2 (2 + δ)(0.5 − ε).

By removing higher order terms of ε, it is easy to see that P (B ) ≥ 1 when ε is small enough.

Therefore B is a local minima of P .

<|TLDR|>

@highlight

The paper analyzes the optimization landscape of one-hidden-layer neural nets and designs a new objective that provably has no spurious local minimum. 

@highlight

This paper studies the problem of learning one-hidden layer neural networks, establishes a connection between least squares population loss and Hermite polynomials, and proposes a new loss function.

@highlight

A tensor factorization-type method for leaning one hidden-layer neural netowrk