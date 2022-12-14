The tremendous success of deep neural networks has motivated the need to better understand the fundamental properties of these networks, but many of the theoretical results proposed have only been for shallow networks.

In this paper, we study an important primitive for understanding the meaningful input space of a deep network: span recovery.

For $k<n$, let $\mathbf{A} \in \mathbb{R}^{k \times n}$ be the innermost weight matrix of an arbitrary feed forward neural network $M: \mathbb{R}^n \to  \mathbb{R}$, so $M(x)$ can be written as $M(x) = \sigma(\mathbf{A} x)$, for some network $\sigma: \mathbb{R}^k \to  \mathbb{R}$.

The goal is then to recover the row span of $\mathbf{A}$ given only oracle access to the value of $M(x)$. We show that if $M$ is a multi-layered network with ReLU activation functions, then partial recovery is possible: namely, we can provably recover $k/2$ linearly independent vectors in the row span of $\mathbf{A}$ using poly$(n)$ non-adaptive queries to $M(x)$.  Furthermore, if $M$ has differentiable activation functions, we demonstrate that \textit{full} span recovery is possible even when the output is first passed through a sign or $0/1$ thresholding function; in this case our algorithm is adaptive.

Empirically, we confirm that full span recovery is not always possible, but only for unrealistically thin layers.

For reasonably wide networks, we obtain full span recovery on both random networks and networks trained on MNIST data.

Furthermore, we demonstrate the utility of span recovery as an attack by inducing neural networks to misclassify data obfuscated by controlled random noise as sensical inputs.

Consider the general framework in which we are given an unknown function f : R n → R, and we want to learn properties about this function given only access to the value f (x) for different inputs x.

There are many contexts where this framework is applicable, such as blackbox optimization in which we are learning to optimize f (x) (Djolonga et al., 2013) , PAC learning in which we are learning to approximate f (x) (Denis, 1998) , adversarial attacks in which we are trying to find adversarial inputs to f (x) (Szegedy et al., 2013) , or structure recovery in which we are learning the structure of f (x).

For example in the case when f (x) is a neural network, one might want to recover the underlying weights or architecture (Arora et al., 2014; .

In this work, we consider the setting when f (x) = M (x) is a neural network that admits a latent low-dimensional structure, namely M (x) = σ(Ax) where A ∈ R k×n is a rank k matrix for some k < n, and σ : R k → R is some neural network.

In this setting, we focus primarily on the goal of recovering the row-span of the weight matrix A. We remark that we can assume that A is full-rank as our results extend to the case when A is not full-rank.

Span recovery of general functions f (x) = g(Ax), where g is arbitrary, has been studied in some contexts, and is used to gain important information about the underlying function f .

By learning Span(A), we in essence are capturing the relevant subspace of the input to f ; namely, f behaves identically on x as it does on the projection of x onto the row-span of A. In statistics, this is known as effective dimension reduction or the multi-index model Li (1991) ; Xia et al. (2002) .

Another important motivation for span recovery is for designing adversarial attacks.

Given the span of A, we compute the kernel of A, which can be used to fool the function into behaving incorrectly on inputs which are perturbed by vectors in the kernel.

Specifically, if x is a legitimate input correctly classified by f and y is a large random vector in the kernel of A, then x + y will be indistinguishable from noise but we will have f (x) = f (x + y).

Several works have considered the problem from an approximation-theoretic standpoint, where the goal is to output a hypothesis function f which approximates f well on a bounded domain.

For instance, in the case that A ∈ R n is a rank 1 matrix and g(Ax) is a smooth function with bounded derivatives, Cohen et al. (2012) gives an adaptive algorithm to approximate f .

Their results also give an approximation A to A, under the assumption that A is a stochastic vector (A i ≥ 0 for each i and i A i = 1).

Extending this result to more general rank k matrices A ∈ R k×n , Tyagi & Cevher (2014) and Fornasier et al. (2012) give algorithms with polynomial sample complexity to find approximations f to twice differentiable functions f .

However, their results do not provide any guarantee that the original matrix A

In this paper, we provably show that span recovery for deep neural networks with high precision can be efficiently accomplished with poly(n) function evaluations, even when the networks have poly(n) layers and the output of the network is a scalar in some finite set.

Specifically, for deep networks M (x) : R n → R with ReLU activation functions, we prove that we can recover a subspace V ⊂ Span(A) of dimension at least k/2 with polynomially many non-adaptive queries.

1 First, we use a volume bounding technique to show that a ReLU network has sufficiently large piece-wise linear sections and that gradient information can be derived from function evaluations.

Next, by using a novel combinatorial analysis of the sign patterns of the ReLU network along with facts in polynomial algebra, we show that the gradient matrix has sufficient rank to allow for partial span recovery.

Theorem 3.4 (informal) Suppose we have the network M (x) = w T φ(W 1 φ(W 2 φ(. . .

W d φ(Ax)) . . . ), where φ is the ReLU and W i ∈ R ki×ki+1 are weight matrices, with k i possibly much smaller than k. Then, under mild assumptions, there is a non-adaptive algorithm that makes O(kn log k) queries to M (x) and returns in poly(n, k)-time a subspace V ⊆ span(A) of dimension at least k 2 with probability 1 − δ.

We remark that span recovery of the first weight layer is provably feasible even in the surprising case when the neural network has many "bottleneck" layers with small O(log(n)) width.

Because this does not hold in the linear case, this implies that the non-linearities introduced by activations in deep learning allow for much more information to be captured by the model.

Moreover, our algorithm is non-adaptive, which means that the points x i at which M (x i ) needs to be evaluated can be chosen in advance and span recovery will succeed with high probability.

This has the benefit of being parallelizable, and possibly more difficult to detect when being used for an adversarial attack.

In contrast with previous papers, we do not assume that the gradient matrix has large rank; rather our main focus and novelty is to prove this statement under minimal assumptions.

We require only two mild assumptions on the weight matrices.

The first assumption is on the orthant probabilities of the matrix A, namely that the distribution of sign patterns of a vector Ag, where g ∼ N (0, I n ), is not too far from uniform.

Two examples of matrices which satisfy this property are random matrices and matrices with nearly orthogonal rows.

The second assumption is a non-degeneracy condition on the matrices W i , which enforces that products of rows of the matrices W i result in vectors with non-zero coordinates.

Our next result is to show that full span recovery is possible for thresholded networks M (x) with twice differentiable activation functions in the inner layers, when the network has a 0/1 threshold function in the last layer and becomes therefore non-differentiable, i.e., M (x) ∈ {0, 1}. Since the activation functions can be arbitrarily non-linear, our algorithm only provides an approximation of the true subspace Span(A), although the distance between the subspace we output and Span(A) can be made exponentially small.

We need only assume bounds on the first and second derivatives of the activation functions, as well as the fact that we can find inputs x ∈ R n such that M (x) = 0 with good probability, and that the gradients of the network near certain points where the threshold evaluates to one are not arbitrarily small.

We refer the reader to Section 4 for further details on these assumptions.

Under these assumptions, we can apply a novel gradient-estimation scheme to approximately recover the gradient of M (x) and the span of A.

Theorem 4.3 (informal) Suppose we have the network M (x) = τ (σ(Ax)), where τ : R → {0, 1} is a threshold function and σ : R k → R is a neural network with twice differentiable activation functions, and such that M satisfies the conditions sketched above (formally defined in Section 4).

Then there is an algorithm that runs in poly(N ) time, making at most poly(N ) queries to M (x), where N = poly(n, k, log( 1 ), log( 1 δ )), and returns with probability 1 − δ a subspace V ⊂ R n of dimension k such that for any x ∈ V , we have

where P Span(A) is the orthogonal projection onto the span of A.

Empirically, we verify our theoretical findings by running our span recovery algorithms on randomly generated networks and trained networks.

First, we confirm that full recovery is not possible for all architectures when the network layer sizes are small.

This implies that the standard assumption that the gradient matrix is full rank does not always hold.

However, we see that realistic network architectures lend themselves easily to full span recovery on both random and trained instances.

We emphasize that this holds even when the network has many small layers, for example a ReLU network that has 6 hidden layers with [784, 80, 40, 30, 20, 10] nodes, in that order, can still admit full span recovery of the rank 80 weight matrix.

Furthermore, we observe that we can effortlessly apply input obfuscation attacks after a successful span recovery and cause misclassifications by tricking the network into classifying noise as normal inputs with high confidence.

Specifically, we can inject large amounts of noise in the null space of A to arbitrarily obfuscate the input without changing the output of the network.

We demonstrate the utility of this attack on MNIST data, where we use span recovery to generate noisy images that are classified by the network as normal digits with high confidence.

We note that this veers away from traditional adversarial attacks, which aim to drastically change the network output with humanly-undetectable changes in the input.

In our case, we attempt the arguably more challenging problem of drastically changing the input without affecting the output of the network.

Notation For a vector x ∈ R k , the sign pattern of x, denoted sign(x) ∈ {0, 1} k , is the indicator vector for the nonzero coordinates of x. Namely, sign(x) i = 1 if x i = 0 and sign(x) i = 0 otherwise.

Given a matrix A ∈ R n×m , we denote its singular values as σ min (A) = σ min{n,m} , . . . , σ 1 (A) = σ max (A).

The condition number of A is denoted κ(A) = σ max (A)/σ min (A).

We let I n ∈ R n×n denote the n × n identity matrix.

For a subspace V ⊂ R n , we write P V ∈ R n×n to denote the orthogonal projection matrix onto V .

If µ ∈ R n and Σ ∈ R n×n is a PSD matrix, we write N (µ, Σ) to denote the multi-variate Gaussian distribution with mean µ and covariance Σ.

Gradient Information For any function f (x) = g(Ax), note that ∇f (x) = A g(Ax) must be a vector in the row span of A. Therefore, span recovery boils down to understanding the span of the gradient matrix as x varies.

Specifically, note that if we can find points x 1 , .., x k such that {∇f (x i )} are linearly independent, then the full span of A can be recovered using the span of the gradients.

To our knowledge, previous span recovery algorithms heavily rely on the assumption that the gradient matrix is full rank and in fact well-conditioned.

Specifically, for some distribution D, it is assumed that H f = x∼D ∇f (x)∇f (x) dx is a rank k matrix with a minimum non-zero singular value bounded below by α and the number of gradient or function evaluations needed depends inverse polynomially in α.

In contrast, in this paper, when f (x) is a neural network, we provably show that H f is a matrix of sufficiently high rank or large minimum non-zero singular value under mild assumptions, using tools in polynomial algebra.

In this section, we demonstrate that partial span recovery is possible for deep ReLU networks.

Specifically, we consider neural networks M (x) : R n → R of the form

where φ(x) i = max{x i , 0} is the RELU (applied coordinate-wise to each of its inputs), and W i ∈ R ki×ki+1 , and w ∈ R k d , and A has rank k. We note that k i can be much smaller than k. In order to obtain partial span recovery, we make the following assumptions parameterized by a value γ > 0 (our algorithms will by polynomial in 1/γ):

• Assumption 2:

Si is the matrix with the rows j / ∈ S i set equal to 0.

Moreover, we assume

Our first assumption is an assumption on the orthant probabilities of the distribution Ag.

Specifically, observe that Ag ∈ R k follows a multi-variate Gaussian distribution with covariance matrix AA T .

Assumption 1 then states that the probability that a random vector x ∼ N (0, AA T ) lies in a certain orthant of R k is not too far from uniform.

We remark that orthant probabilities of multivariate Gaussian distributions are well-studied (see e.g., Miwa et al. (2003) ; Bacon (1963) ; Abrahamson et al. (1964) ), and thus may allow for the application of this assumption to a larger class of matrices.

In particular, we show it is satisfied by both random matrices and orthogonal matrices.

Our second assumption is a non-degeneracy condition on the weight matrices W i -namely, that products of w T with non-empty sets of rows of the W i result in entry-wise non-zero vectors.

In addition, Assumption 2 requires that the network is non-zero with probability that is not arbitrarily small, otherwise we cannot hope to find even a single x with M (x) = 0.

In the following lemma, we demonstrate that these conditions are satisfied by randomly initialized networks, even when the entries of the W i are not identically distributed.

Lemma 3.1.

If A ∈ R k×n is an arbitrary matrix with orthogonal rows, or if n > Ω(k 3 ) and A has entries that are drawn i.i.d.

from some sub-Gaussian distribution D with expectation 0, unit variance, and constant sub-Gaussian norm

1/p then with probability at least 1 − e −k 2 , A satisfies Assumption 1 with

have entries that are drawn independently (and possibly non-identically) from continuous symmetric distributions, and if

, then Assumption 2 holds with probability 1 − δ.

The algorithm for recovery is given in Algorithm 1.

Our algorithm computes the gradient ∇M (g i ) for different Gaussian vectors g i ∼ N (0, I k ), and returns the subspace spanned by these gradients.

To implement this procedure, we must show that it is possible to compute gradients via the perturbational method (i.e. finite differences), given only oracle queries to the network M .

Namely, we firstly must show that if g ∼ N (0, I n ) then ∇M (g) exists, and moreover, that ∇M (x) exists for all x ∈ B (g), where B (g) is a ball of radius centered at g, and is some value with polynomial bit complexity which we can bound.

To demonstrate this, we show that for any fixing of the sign patterns of the network, we can write the region of R n which satisfies this sign pattern and is -close to one of the O(dk) ReLU thresholds of the network as a linear program.

We then show that the feasible polytope of this linear program is contained inside a Euclidean box in R n , which has one side of length .

Using this containment, we upper bound the volume of the polytope in R n which is close to each ReLU, and union bound over all sign patterns and ReLUs to show that the probability that a Gaussian lands in one of these polytopes is exponentially small.

Compute Gradient:

There is an algorithm which, given g ∼ N (0, I k ), with probability 1 − exp(−n c ) for any constant c > 1 (over the randomness in g), computes ∇M (g) ∈ R n with O(n) queries to the network, and in poly(n) runtime.

Now observe that the gradients of the network lie in the row-span of A. To see this, for a given input x ∈ R n , let S 0 (x) ∈ R k be the sign pattern of φ(Ax) ∈ R k , and more generally define

, which demonstrates the claim that the gradients lie in the row-span of

, and let Z be the matrix where the i-th row is equal to z i .

We will prove that Z has rank at least k/2.

To see this, first note that we can write Z = VA, where V is some matrix such that the non-zero entries in the i-th row are precisely the coordinates in the set S i 0 , where S i j = S j (g i ) for any j = 0, 1, 2, . . .

, d and i = 1, 2, . . .

, r. We first show that V has rank at least ck for a constant c > 0.

To see this, suppose we have computed r gradients so far, and the rank of V is less than ck for some 0 < c < 1/2.

Now V ∈ R r×k is a fixed rank-ck matrix, so the span of the matrix can be expressed as a linear combination of some fixed subset of ck of its rows.

We use this fact to show in the following lemma that the set of all possible sign patterns obtainable in the row span of V is much smaller than 2 k .

Thus a gradient z r+1 with a uniform (or nearly uniform) sign pattern will land outside this set with good probability, and thus will increase the rank of Z when appended.

Lemma 3.3.

Let V ∈ R r×k be a fixed at most rank ck matrix for c ≤ 1/2.

Then the number of sign patterns S ⊂ [k]

with at most k/2 non-zeros spanned by the rows of V is at most

.

In other words, the set

, where φ is the ReLU, satisfies Assumptions 1 and 2.

Then the algorithm given in Figure 1 makes O(kn log(k/δ)/γ) queries to M (x) and returns in poly(n, k, 1/γ, log(1/δ)) time a subspace V ⊆ span(A) of dimension at least k 2 with probability 1 − δ.

In this section, we consider networks that have a threshold function at the output node, as is done often for classification.

Specifically, let τ : R → {0, 1} be the threshold function: τ (x) = 1 if x ≥ 1, and τ (x) = 0 otherwise.

Again, we let A ∈ R k×n where k < n, be the innermost weight matrix.

The networks M : R n → R we consider are then of the form:

and each φ i is a continuous, differentiable activation function applied entrywise to its input.

We will demonstrate that even for such functions with a binary threshold placed at the end, giving us minimal information about the network, we can still achieve full span recovery of the weight matrix A, albeit with the cost of anapproximation to the subspace.

Note that the latter fact is inherent, since the gradient of any function that is not linear in some ball around each point cannot be obtained exactly without infinitely small perturbations of the input, which we do not allow in our model.

We can simplify the above notation, and write σ(x) = W 1 φ 1 (W 2 φ 2 (. . .

φ d Ax)) . . . ), and thus M (x) = τ (σ(x)).

Our algorithm will involve building a subspace V ⊂ R n which is a good approximation to the span of A. At each step, we attempt to recover a new vector which is very close to a vector in Span(A), but which is nearly orthogonal to the vectors in V .

Specifically, after building V , on an input x ∈ R n , we will query M for inputs M ((I n − P V )x).

Recall that P V is the projection matrix onto V , and P V ⊥ is the projection matrix onto the subspace orthogonal to V .

Thus, it will help here to think of the functions M, σ as being functions of x and not (I n − P V )x, and so we define

For the results of this section, we make the following assumptions on the activation functions.

1.

The function φ i : R → R is continuous and twice differentiable, and φ i (0) = 0.

2.

φ i and φ i are L i -Lipschitz, meaning:

The network is non-zero with bounded probability: for every subspace V ⊂ R n of dimension dim(V ) < k, we have that Pr g∼N (0,In) [σ V (g) ≥ 1] ≥ γ for some value γ > 0.

4.

Gradients are not arbitrarily small near the boundary: for every subspace V ⊂ R n of dimension dim(V ) < k

for some values η, γ > 0, where ∇ g σ V (cg) is the directional derivative of σ V in the direction g.

The first two conditions are standard and straightforward, namely φ i is differentiable, and has bounded first and second derivatives (note that for our purposes, they need only be bounded in a ball of radius poly(n)).

Since M (x) is a threshold applied to σ(x), the third condition states that it is possible to find inputs x with non-zero network evaluation M (x).

Our condition is slightly stronger, in that we would like this to be possible even when x is projected away from any k < k dimensional subspace (note that this ensures that Ax is non-zero, since A has rank k).

The last condition simply states that if we pick a random direction g where the network is non-zero, then the gradients of the network are not arbitrarily small along that direction at the threshold points where σ(c · g) = 1.

Observe that if the gradients at such points are vanishingly small, then we cannot hope to recover them.

Moreover, since M only changes value at these points, these points are the only points where information about σ can be learned.

Thus, the gradients at these points are the only gradients which could possibly be learned.

We note that the running time of our algorithms will be polynomial in log(1/η), and thus we can even allow the gradient size η to be exponentially small.

We now formally describe and analyze our span recovery algorithm for networks with differentiable activation functions and 0/1 thresholding.

Let κ i be the condition number of the i-th weight matrix W i , and let δ > 0 be a failure probability, and let > 0 be a precision parameter which will affect the how well the subspace we output will approximate Span(A).

Now fix N = poly(n, k,

The running time and query complexity of our algorithm will be polynomial in N .

Our algorithm for approximate span recovery is given formally in Algorithm 2.

Proposition 4.1.

Let V ⊂ R n be a subspace of dimension k < k, and fix any 0 > 0.

Then we can find a vector x with 0 ≤ σ V (x) − 1 ≤ 2 0 in expected O(1/γ + N log(1/ 0 )) time.

Moreover, with probability γ/2 we have that ∇ x σ V (x) > η/4 and the tighter bound of 0 η2

We will apply the above proposition as input to the following Lemma 4.2, which is the main technical result of this section.

Our approach involves first taking the point x from Proposition 4.1 such that σ V (x) is close but bounded away from the boundary, and generating n perturbations at this point M V (x + u i ) for carefully chosen u i .

While we do not know the value of σ V (x + u i ), we can tell for a given scaling c > 0 if σ V (x + cu i ) has crossed the boundary, since we will then have M V (x + cu i ) = 0.

Thus, we can estimate the directional derivative ∇ ui σ(x) by finding a value c i via a binary search such that σ V (x + c i u i ) is exponentially closer to the boundary than σ V (x).

In order for our estimate to be accurate, we must carefully upper and lower bound the gradients and Hessian of σ v near x, and demonstrate that the linear approximation of σ v at x is still accurate at the point x + c i u i where the boundary is crossed.

Since each value of 1/c i is precisely proportional to ∇ ui σ(x) = ∇σ(x), u i , we can then set up a linear system to approximately solve for the gradient ∇σ(x) (lines 8 and 9 of Algorithm 2).

Lemma 4.2.

Fix any , δ > 0, and let N be defined as above.

Then given any subspace V ⊂ R n with dimension dim(V ) < k, and given x ∈ R n , such that 0 η2

, and such that ∇ x σ V (x) > η/2, then with probability 1 − 2 −N/n 2 , we can find a vector v ∈ R n in expected poly(N ) time, such that P Span(A) v 2 ≥ (1 − ) v 2 , and such that P V v 2 ≤ v 2 .

Theorem 4.3.

Suppose the network M (x) = τ (σ(Ax)) satisfies the conditions described at the beginning of this section.

Then Algorithm 2 runs in poly(N ) time, making at most poly(N ) queries to M (x), where

Find a scaling α > 0 via binary search on values τ (σ V (αg)) such that x = αg satisfies

Generate g 1 , . . . , g n ∼ N (0, I n ), and set u i = g i 2 −N − x/ x 2 .

For each i ∈ [n], binary search over values c to find c i such that

If any c i satisfies |c i | ≥ (10 · 2 −N 0 /η), restart from line 5 (regenerate the Gaussian g).

Otherwise, define B ∈ R n×n via B * ,i = u i , where B * ,i is the i-th column of B. Define b ∈ R n by b i = 1/c i .

Let y * be the solution to: min

Set v i = y * and V ← Span(V, v i ).

10 end 11 return V poly(n, k,

, and returns with probability 1 − δ a subspace V ⊂ R n of dimension k such that for any x ∈ V , we have P Span(A) x 2 ≥ (1 − ) x 2 .

Figure 1 : Partial span recovery of small networks with layer sizes specified in the legend.

Note that 784->80->[6,3] indicates a 4 layer neural network with hidden layer sizes 784, 80, 6, and 3, in that order.

Full span recovery is not always possible and recovery deteriorates as width decreases and depth increases.

When applying span recovery for a given network, we first calculate the gradients analytically via auto-differentiation at a fixed number of sample points distributed according to a standard Gaussian.

Our networks are feedforward, fully-connected with ReLU units; therefore, as mentioned above, using analytic gradients is as precise as using finite differences due to piecewise linearity.

Then, we compute the rank of the resulting gradient matrix, where the rank is defined to be the number of singular values that are above 1e-5 of the maximum singular value.

In our experiments, we attempt to recover the full span of a 784-by-80 matrix with decreasing layer sizes for varying sample complexity, as specified in the figures.

For the MNIST dataset, we use a size 10 vector output and train according to the softmax cross entropy loss, but we only calculate the gradient with respect to the first output node.

Our recovery algorithms are GradientsRandom (Algorithm 1), GradientsRandomAda (Algorithm 2), and GradientsM-NIST.

GradientsRandom is a direct application of our first span recovery algorithm and calculates gradients via pertur- bations at random points for a random network.

GradientsRandomAda uses our adaptive span recovery algorithm for a random network.

Finally, GradientsMNIST is an application of GradientsRandom on a network with weights trained on MNIST data.

In general, we note that the experimental outcomes are very similar among all three scenarios.

Figure 3: Fooling ReLU networks into misclassifying noise as digits by introducing Gaussian noise into the null space after span recovery.

The prediction of the network is presented above the images, along with its softmax probability.

For networks with very small widths and multiple layers, we see that span recovery deteriorates as depth increases, supporting our theory (see Figure 1 ).

This holds both in the case when the networks are randomly initialized with Gaussian weights or trained on a real dataset (MNIST) and whether we use adaptive or non-adaptive recovery algorithms.

However, we note that these small networks have unrealistically small widths (less than 10) and when trained on MNIST, these networks fail to achieve high accuracy, all falling below 80 percent.

The small width case is therefore only used to support, with empirical evidence, why our theory cannot possibly guarantee full span recovery under every network architecture.

For more realistic networks with moderate or high widths, however, full span recovery seems easy and implies a real possibility for attack (see Figure 2 ).

Although we tried a variety of widths and depths, the results are robust to reasonable settings of layer sizes and depths.

Therefore, we only present experimental results with sub-networks of a network with layer sizes [784, 80, 40, 30, 20, 10] .

Note that full span recovery of the first-layer weight matrix with rank 80 is achieved almost immediately in all cases, with less than 100 samples.

On the real dataset MNIST, we demonstrate the utility of span recovery algorithms as an attack to fool neural networks to misclassify noisy inputs (see Figure 3 ).

We train a ReLU network (to around 95 percent accuracy) and recover its span by computing the span of the resulting gradient matrix.

Then, we recover the null space of the matrix and generate random Gaussian noise projected onto the null space.

We see that our attack successfully converts images into noisy versions without changing the output of the network, implying that allowing a full (or even partial) span recovery on a classification network can lead to various adversarial attacks despite not knowing the exact weights of the network.

We first restate the results which have had their proofs omitted, and include their proofs subsequently.

Assumption 2 holds with probability 1 − δ.

Proof.

By Theorem 5.58 of Vershynin (2010) , if the entries A are drawn i.i.d.

from some sub-Gaussian isotropic distribution D over R n such that A j 2 = √ n almost surely, then

, for some constants c, C > 0 depending only on D ψ2 .

Since the entries are i.i.d.

with variance 1, it follows that the rows of A are isotropic.

Moreover, we can always condition on the rows having norm exactly √ n, and pulling out a positive diagonal scaling through the first Relu of M (x), and absorbing this scaling into W d .

It follows that the conditions of the theorem hold, and we have

for a suitably large re scaling of the constant C. Setting n > Ω(k 3 ), it follows that κ(A) < (1 + 1/(100k)), which holds immediately if A has orthogonal rows.

Now observe that Ag is distributed as a multi-variate Gaussian with co-variance A T A, and is therefore given by the probability density function (pdf)

x T x be the pdf of an identity covariance Gaussian N (0, I k ).

We lower bound p (x)/p(x) for x with x 2 2 ≤ 16k.

In this case, we have

Thus for any sign pattern S, Pr[sign(Ag) = S : Ag

−k , and spherical symmetry of Gaussians, Pr[sign(g) = S : g For the second claim, by an inductive argument, the entries in the rows i ∈ S j of the product W Sj

Si is non-zero with probability 1.

It follows that w,

is the inner product of a non-zero vector with a vector w with continuous, independent entries, and is thus non-zero with probability 1.

By a union bound over all possible non-empty sets S j , the desired result follows.

We now show that the second part of Assumption 2 holds.

To do so, first let g ∼ N (0, I n ).

We demonstrate that Pr W1,W2,...,W d ,g [M (x) = 0] ≤ 1 − γδ/100.

Here the entries of the W i 's are drawn independently but not necessarily identically from a continuous symmetric distribution.

To see this, note that we can condition on the value of g, and condition at each step on the non-zero value of y i = φ(W i+1 φ(W i+2 φ(. . .

φ(Ag) . . . ).

Then, over the randomness of W i , note that the inner product of a row of W i and y i is strictly positive with probability at least 1/2, and so each coordinate of W i y i is strictly positive independently with probability ≥ 1/2.

It follows that φ(W i y i ) is non-zero with probability at least 1 − 2 −ki .

Thus

where the second inequality is by assumption.

It follows by our first part that

So by Markov's inequality,

Thus with probability 1 − δ over the choice of

Lemma 3.2 There is an algorithm which, given g ∼ N (0, I k ), with probability 1 − exp(−n c ) for any constant c > 1 (over the randomness in g), computes ∇M (g) ∈ R n with O(n) queries to the network, and in poly(n) running time.

where φ is the ReLU.

If ∇M (g) exists, there is an > 0 such that M is differentiable on B (g).

We show that with good probability, if g ∼ N (0, I n ) (or in fact, almost any continuous distribution), then M (g) is continuous in the ball B (g) = {x ∈ R n | x − g 2 < } for some which we will now compute.

First, we can condition on the event that g 2 2 ≤ (nd) 10c , which occurs with probability at least 1 − exp(−(nd) 5c ) by concentration results for χ 2 distributions Laurent & Massart (2000) .

Now, fix any sign pattern

, and let S = (S 1 , S 2 , . . .

, S d+1 ).

We note that we can enforce the constraint that for an input x ∈ R n , the sign pattern of M i (x) is precisely S i .

To see this, note that after conditioning on a sign pattern for each layer, the entire network becomes linear.

Thus each constraint that (−poly(nd) ) is a value we will later choose.

Thus, we obtain a linear program with k + d i=1 k i constraints and n variables.

The feasible polytope P represents the set of input points which satisfy the activation patterns S and are η-close to the discontinuity given by the j-th neuron in the i-th layer.

We can now introduce the following non-linear constraint on the input that x 2 ≤ (nd)

10c .

Let B = B (nd) 10c ( 0) be the feasible region of this last constraint, and let P * = P ∩ B. We now bound the Lesbegue measure (volume) V (P * ) of the region P. First note that V (P * ) ≤ V (P ), where V (P ) is the region defined by the set of points which satisfy:

where each coordinate of the vector y ∈ R n is a linear combination of products of the weight matrices W , ≥ i. One can see that the first two constraints for P are also constraints for P * , and the last constraint is precisely B, thus P * ⊂ P which completes the claim of the measure of the latter being larger.

Now we can rotate P by the rotation which sends y → y 2 · e 1 ∈ R n without changing the volume of the feasible region.

The resulting region is contained in the region P given by

Finally, note that P ⊂ R n is a Eucledian box with n − 1 side lengths equal to (nd) 10c and one side length of y 2 η, and thus V (P ) ≤ y 2 η(nd)

10nc .

Now note we can assume that the entries of the weight matrices A, W 1 , . . .

, W d are specified in polynomially many (in n) bits, as if this were not the case the output M (x) of the network would not have polynomial bit complexity, and could not even be read in poly(n) time.

Equivalently, we can assume that our running time is allowed to be polynomial in the number of bits in any value of M (x), since this is the size of the input to the problem.

Given this, since the coordinates of y were linear combinations of products of the coordinates of the weight matrices, and note that each of which is at most 2 n C for some constant C (since the matrices have polynomial bit-complexity), we have that P * ≤ η2 n C (nd) 10nc as needed.

Now the pdf of a multi-variate Gaussian is upper bounded by 1, so

10nc .

It follows that the probability that a multivariate Gaussian g ∼ N (0, I n ) satisfies the sign pattern S and is η close to the boundary for the j-th neuron in the i-th layer.

Now since there are at most

nd possible combinations of sign patterns S, it follows that the the probability that a multivariate Gaussian g ∈ N (0, I n ) is η close to the boundary for the j-th neuron in the i-th layer is at most η2

10nc 2 nd .

Union bounding over each of the k i neurons in layer i, and then each of the d layers, it follows that g ∈ N (0, I n ) is η close to the boundary for any discontinuity in M (x) is at most η2

, it follows that with probability at least 1 − exp(−(nd) c ), the network evaluated at g ∈ N (0, I n ) is at least η close from all boundaries (note that C is known to us by assumption).

Now we must show that perturbing the point g by any vector with norm at most results in a new point g which still has not hit one of the boundaries.

Note that M (g) is linear in an open ball around g, so the change that can occur in any intermediate neuron after perturbing g by some v ∈ R n is at most

where · 2 is the spectral norm.

Now since each entry in the weight matrix can be specified in polynomially many bits, the Frobenius norm of each matrix (and therefore the spectral norm), is bounded by n 2 2 n C for some constant C. Thus

and setting = η/β, it follows that M (x) is differentiable in the ball B (x) as needed.

We now generate u 1 , u 2 , . . . , u n ∼ N (0, I n ), which are linearly independent almost surely.

We set v i = ui 2 ui 2 .

Since M (g) is a ReLU network which is differentiable on B (g), it follows that M (g) is a linear function on B (g), and moreover v i ∈ B (g) for each i ∈ [n].

Thus for any c < 1 we have

.

Finally, since the directional derivative is given by ∇ vi M (x) = ∇M (x), v i / v i 2 , and since v 1 , . . .

, v n are linearly independent, we can set up a linear system to solve for ∇M (x) exactly in polynomial time, which completes the proof.

with at most k/2 non-zeros spanned by the rows of V is at most

.

In other words, the set

Proof.

Any vector w in the span of the rows of V can be expressed as a linear combination of at most ck rows of V. So create a variable x i for each coefficient i ∈ [ck] in this linear combination, and let f j (x) be the linear function of the x i s which gives the value of the j-th coordinate of w. Then f (x) = (f 1 (x), . . .

, f k (x)) is a k-tuple of polynomials, each in ck-variables, where each polynomial has degree 1.

By Theorem 4.1 of Hall et al. (2010) , it follows that the number of sign patterns which contain at most k/2 non-zero entries is at most ck+k/2 ck .

Setting c ≤ 1/2, this is at most

, where φ is the ReLU, satisfies Assumption 1 and 2.

Then the algorithm given in Figure 1 makes O(kn log(k/δ)/γ) queries to M (x) and returns in poly(n, k, 1/γ, log(1/δ))-time a subspace V ⊆ span(A) of dimension at least k 2 with probability 1 − δ.

1/γ repetitions is O(log(1/γ) √ n), and thus the expected running time reduces to the stated bound, which completes the first claim of the Proposition.

For the second claim, note that ∇ gi σ V (c * g i ) > 0 by construction of the binary search, and since σ V (c * g i ) > 0 = 1, by Property 4 with probability γ we have that ∇ gi σ V (g i ) > η.

Now with probability 1 − γ/2, we have that g i 2 2 ≤ O(n log(1/γ)) (see Lemma 1 Laurent & Massart (2000) ), so by a union bound both of these occur with probability γ/2.

Now since (c * − c)x 2 ≤ 0 2 −N (after rescaling N by a factor of log( g i 2 ) = O(log(n))), and since 2 N is also an upper bound on the spectral norm of the Hessian of σ by construction, it follows that ∇ gi σ V (cg i ) > η/2.

Now we set x ← cg i +c 0 2 −N g i /( cg i 2 ).

First note that this increases σ V (cx)−1 by at most 0 , so σ(cx)−1 ≤ 2 0 , so this does not affect the first claim of the Proposition.

But in addition, note that conditioned on the event in the prior paragraph, we now have that σ V (x) > 1 + η 0 2 −N .

The above facts can be seen by the fact that 2 N is polynomially larger than the spectral norm of the Hessian of σ, thus perturbing x by 0 2 −N additive in the direction of x will result in a positive change of at least 1 2 (η/4)( 0 2 −N ) in σ.

Moreover, by applying a similar argument as in the last paragraph, we will have ∇ x σ V (cx) > η/4 still after this update to x. Lemma 4.2 Fix any , δ > 0, and let N = poly(n, k,

where 0 = Θ(2 −N C / ) for a sufficiently large constant C = O(1), and ∇ x σ V (x) > η/2, then with probability

We first claim that the c i which achieves this value satisfies c i u i 2 ≤ (10 · 2 −N 0 /η).

To see this, first note that by Proposition 4.1, we have ∇ x σ V (x) > η/4 with probability γ.

We will condition on this occurring, and if it fails to occur we argue that we can detect this and regenerate x. Now conditioned on the above, we first claim that ∇ ui σ V (x) ≥ η/8, which follows from the fact that we can bound the angle between the unit vectors in the directions of u i and x by cos (angle(u i , x)) = u i u 2 , x x 2 ≥ (1 − n/2 −N ) > (1 − η/2 −N/2 ) along with the fact that we have ∇ x σ V (x) > η/4.

Since |σ V (x) − 1| < 2 0 < 2 −N C , and since 2 N is an upper bound on the spectral norm of the Hessian of σ, we have that ∇ ui σ V (x + cu i ) > η/8 + 2 −N > η/10 for all c < 2 −2N .

In other words, if H is the hessian of σ, then perturbing x by a point with norm O(c) ≤ 2 −2N can change the value of the gradient by a vector of norm at most 2 2N H 2 ≤ 2 −N , where H 2 is the spectral norm of the Hessian.

It follows that setting c = (10 · 2 −N 0 /η) is sufficient for σ V (x + cu i ) < 1, which completes the above claim.

Now observe that if after binary searching, the property that c ≤ (10 · 2 −N 0 /η) does not hold, then this implies that we did not have ∇σ(x) > η/4 to begin with, so we can throw away this x and repeat until this condition does hold.

By Assumption 4, we must only repeat O(1/γ) times in expectation in order for the assumption to hold.

Next, also note that we can bound c i ≥ 0 η2 −N /N , since 2 N again is an upper bound on the norm of the gradient of σ and we know that σ V (x) − 1 > 0 η2 −N .

Altogether, we now have that |Ξ(c i u i )| ≤ c poly(n, k,

log(κ i ), log( 1 η ), log( 1 ), log( 1 δ )), and returns with probability 1 − δ a subspace V ⊂ R n of dimension k such that for any x ∈ V , we have

Proof.

We iteratively apply Lemma 4.2, each time appending the output v ∈ R n of the proposition to the subspace V ⊂ R n constructed so far.

WLOG we can assume v is a unit vector by scaling it.

Note that we have the property at any given point in time k < k that V = Span(v 1 , . . .

, v k ) where each v i satisfies that P Span{v1,...,vi−1} v i 2 ≤ .

Note that the latter fact implies that v 1 , . . .

v k are linearly independent.

Thus at the end, we recover a rank k subspace V = Span(v 1 , . . .

, v k ), with the property that P Span(A) v i 2 2 ≥ (1 − ) v i 2 2 for each i ∈ [k].

Now let V ∈ R n×n be the matrix with i-th column equal to v i .

Fix any unit vector x = Va ∈ V , where a ∈ R n is uniquely determined by x. Let V = V + + V − where V + = P Span(A) V and V − = V − V + Then x = V + a + V − a, and (I n − P Span(A) )x 2 ≤ (I n − P Span(A) )V + a 2 + (I n − P Span(A) )V − a 2

First note that by the construction of the v i 's, each column of V − has norm O( ), thus V

Thus σ min (V) ≥ (1 − O( )), so we have (I n − P Span(A) )x 2 ≤ V − 2 1 σmin(V) ≤ 2 √ n .

By the Pythagorean theorem: P Span(A) )x 2 2 = 1 − (I n − P Span(A) )x 2 2 ≥ 1 − O(n 2 ).

Thus we can scale by a factor of Θ(1/ √ n) in the call to Lemma 4.2, which gives the desired result of P Span(A) x 2 ≥ 1 − .

<|TLDR|>

@highlight

We provably recover the span of a deep multi-layered neural network with latent structure and empirically apply efficient span recovery algorithms to attack networks by obfuscating inputs.