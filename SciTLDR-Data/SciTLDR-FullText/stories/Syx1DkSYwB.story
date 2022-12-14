Variance reduction methods which use a mixture of large and small batch gradients, such as SVRG (Johnson & Zhang, 2013) and SpiderBoost (Wang et al., 2018), require significantly more computational resources per update than SGD (Robbins & Monro, 1951).

We reduce the computational cost per update of variance reduction methods by introducing a sparse gradient operator blending the top-K operator (Stich et al., 2018; Aji & Heafield, 2017) and the randomized coordinate descent operator.

While the computational cost of computing the derivative of a model parameter is constant, we make the observation that the gains in variance reduction are proportional to the magnitude of the derivative.

In this paper, we show that a sparse gradient based on the magnitude of past gradients reduces the computational cost of model updates without a significant loss in variance reduction.

Theoretically, our algorithm is at least as good as the best available algorithm (e.g. SpiderBoost) under appropriate settings of parameters and can be much more efficient if our algorithm succeeds in capturing the sparsity of the gradients.

Empirically, our algorithm consistently outperforms SpiderBoost using various models to solve various image classification tasks.

We also provide empirical evidence to support the intuition behind our algorithm via a simple gradient entropy computation, which serves to quantify gradient sparsity at every iteration.

Optimization tools for machine learning applications seek to minimize the finite sum objective

where x is a vector of parameters, and f i : R d → R is the loss associated with sample i. Batch SGD serves as the prototype for modern stochastic gradient methods.

It updates the iterate x with x − η∇f I (x), where η is the learning rate and f I (x) is the batch stochastic gradient, i.e. ∇f I (x) = 1 |I| i∈I ∇f i (x).

The batch size |I| in batch SGD directly impacts the stochastic variance and gradient query complexity of each iteration of the update rule.

Lower variance improves convergence rate without any changes to learning rate, but the step-size in the convergence analysis of SGD decreases with variance (Robbins & Monro, 1951) , which suggests that learning rates can be increased when stochastic variance is decreased to further improve the convergence rate of gradient-based machine learning optimization algorithms.

This is generally observed behavior in practice (Smith et al., 2018; Hoffer et al., 2017) .

In recent years, new variance reduction techniques have been proposed by carefully blending large and small batch gradients (e.g. Roux et al., 2012; Johnson & Zhang, 2013; Defazio et al., 2014; Xiao & Zhang, 2014; Allen-Zhu & Yuan, 2016; Allen-Zhu & Hazan, 2016; Reddi et al., 2016a; b; Allen-Zhu, 2017; Lei & Jordan, 2017; Lei et al., 2017; Allen-Zhu, 2018b; Fang et al., 2018; Zhou et al., 2018; Wang et al., 2018; Pham et al., 2019; Nguyen et al., 2019; Lei & Jordan, 2019) .

They are alternatives to batch SGD and are provably better than SGD in various settings.

While these methods allow for greater learning rates than batch SGD and have appealing theoretical guarantees, they require a per-iteration query complexity which is more than double than that of batch SGD.

This 1.

We introduce a novel way to reduce the computational complexity of SVRG-style variance reduction methods using gradient sparsity estimates.

Concretely, we define an algorithm which applies these ideas to SpiderBoost (Wang et al., 2018) .

2.

We provide a complete theoretical complexity analysis of our algorithm, which shows algorithmic improvements in the presence of gradient sparsity structure.

3.

We experimentally show the presence of sparsity structure for some deep neural networks, which is an important assumption of our algorithm.

Our experiments show that, for those deep neural networks, sparse gradients improve the empirical convergence rate by reducing both variance and computational complexity.

4.

We include additional experiments on natural language processing and sparse matrix factorization, and compare our algorithms to two different SGD baselines.

These experiments demonstrate different ways in which variance reduction methods can be adapted to obtain competitive performance on challenging optimization tasks.

The rest of the paper is organized as follows.

We begin by providing a sparse variance reduction algorithm based on a combination of SCSG (Lei et al., 2017) and SpiderBoost (Wang et al., 2018) .

We then explain how to perform sparse back-propagation in order to realize the benefits of sparsity.

We prove both that our algorithm is as good as SpiderBoost, and under reasonable assumptions, has better complexity than SpiderBoost.

Finally, we present our experimental results which include an empirical analysis of the sparsity of various image classification problems, and a comparison between our algorithm and SpiderBoost.

Generally, variance reduction methods reduce the variance of stochastic gradients by taking a snapshot ∇f (y) of the gradient ∇f (x) every m steps of optimization, and use the gradient information in this snapshot to reduce the variance of subsequent smaller batch gradients ∇f I (x) (Johnson & Zhang, 2013; Wang et al., 2018) .

Methods such as SCSG (Lei & Jordan, 2017) utilize a large batch gradient, which is typically some multiple in size of the small batch gradient b, which is much more practical and is what we do in this paper.

To reduce the cost of computing additional gradients, we use sparsity by only computing a subset k of the total gradients d, where y ∈ R d .

In what follows, we define an operator which takes vectors x, y and outputs y , where y retains only k of the entries in y, k 1 of which are selected according to the coordinates in x which have the k 1 largest absolute values, and the remaining k 2 entries are randomly selected from y. The k 1 coordinate indices and k 2 coordinate indices are disjoint.

Formally, the operator rtop k1,k2 :

where |x| denotes a vector of absolute values, |x| (1) ≥ |x| (2) ≥ . . .

≥ |x| (d) denotes the order statistics of coordinates of x in absolute values, and S denotes a random subset with size k 2 that is uniformly drawn from the set { : |x| < |x| (k1) }.

For instance, if x = (11, 12, 13, −14, −15), y = (−25, −24, 13, 12, 11) and k 1 = k 2 = 1, then S is a singleton uniformly drawn from {1, 2, 3, 4}.

On the other hand, if k 1 = 0, rtop 0,k2 (x, y) does not depend on x and returns a rescaled random subset of y. This is the operator used in coordinate descent methods.

Finally, rtop k1,k2 (x, y) is linear in y. The following Lemma shows that rtop k1,k2 (x, y) is an unbiased estimator of y, which is a crucial property in our later analysis.

where E is taken over the random subset S involved in the rtop k1,k2 operator and

Our algorithm is detailed as below.

Algorithm 1: SpiderBoost with Sparse Gradients.

Input: Learning rate η, inner loop size m, outer loop size T , large batch size B, small batch size b, initial iterate x 0 , memory decay factor α, sparsity parameters k 1 , k 2 .

The algorithm includes an outer-loop and an inner-loop.

In the theoretical analysis, we generate N j as Geometric random variables.

This trick is called "geometrization", proposed by Lei & Jordan (2017) and dubbed by Lei & Jordan (2019) .

It greatly simplifies analysis (e.g. Lei et al., 2017; Allen-Zhu, 2018a) .

In practice, as observed by Lei et al. (2017) , it does not make a difference if N j is simply set to be m. For this reason, we apply "geometrization" in theory to make arguments clean and readable.

On the other hand, in theory the output is taken as uniformly random elements from the set of last iterates in each outer loop.

This is a generic strategy for nonconvex optimization, as an analogue of the average iterates for convex optimization, proposed by Nemirovski et al. (2009) .

In practice, we simply use the last iterate as convention.

Similar to Aji & Heafield (2017) , we maintain a memory vector at each iteration of our algorithm.

We assume the optimization procedure is taking place locally and thus do not transmit and zero out any components.

Instead, we maintain an exponential moving average M (j) t of the magnitudes of each coordinate of our gradient estimate ν (j) t .

We then use M (j) t as an approximation to the variance of each gradient coordinate in our rtop k1,k2 operator.

With M (j) t as input, the rtop k1,k2 operator targets k 1 high variance gradient coordinates in addition to the k 2 randomly selected coordinates.

The cost of invoking rtop k1,k2 is dominated by the algorithm for selecting the top k coordinates, which has linear worst case complexity when using the introselect algorithm.

Algorithmic implementation details for a sparse back-propagation algorithm can be found in appendix B.

We assume that sampling an index i and accessing the pair ∇f i (x) incur a unit of cost and accessing the truncated version rtop k1,k2 (y, ∇f i (x)) incur (k 1 + k 2 )/d units of cost.

Note that calculating rtop k1,k2 (y, ∇f I (x)) incurs |I|(k 1 + k 2 )/d units of computational cost.

Given our framework, the theoretical complexity of the algorithm is

3 THEORETICAL COMPLEXITY ANALYSIS

Denote by · the Euclidean norm and by a ∧ b the minimum of a and b. For a random vector

We say a random variable N has a geometric distribution, N ∼ Geom(m), if N is supported on the non-negative integers with

for some γ such that EN = m. Here we allow N to be zero to facilitate the analysis.

Assumption A1 on the smoothness of individual functions will be made throughout the paper.

As a direct consequence of assumption A1, it holds for any x,

To formulate our complexity bounds, we define

Further we define σ 2 as an upper bound on the variance of the stochastic gradients:

3.2 WORST-CASE GUARANTEE Theorem 1.

Under the following setting of parameters

, the complexity to achieve the above condition is

Recall that the complexity of SpiderBoost (Wang et al., 2018) is

, our algorithm has the same complexity as SpiderBoost under appropriate settings.

The penalty term O( b(k 1 + k 2 )/k 2 ) is due to the information loss by sparsification.

Let g

and

By Cauchy-Schwarz inequality and the linearity of top −k1 , it is easy to see that g

t .

If our algorithm succeeds in capturing the sparsity, both g (j)

t and G (j) t will be small.

In this subsection we will analyze the complexity under this case.

Further define R j as

where E j is taken over all randomness in j-th outer loop (line 4-13 of Algorithm 1).

Theorem 2.

Under the following setting of parameters

If we further set

, the complexity to achieve the above condition is

In practice, m is usually much larger than b. As a result, the complexity of our algorithm is

We ran a variety of experiments to demonstrate the performance of Sparse SpiderBoost, as well as to illustrate the potential of sparsity as a way to improve the gradient query complexity of variance reduction methods.

We include performance on image classification to further illustrate the performance of SpiderBoost with and without sparsity.

We also provide additional experiments, including a natural language processing task and sparse matrix factorization to evaluate our algorithm on a variety of tasks.

For all experiments, unless otherwise specified, we run SpiderBoost and Sparse SpiderBoost with a learning rate η = 0.1, large-batch size B = 1000, small-batch size b = 100, inner loop length of m = 10, memory decay factor of α = 0.5, and k 1 and k 2 both set to 5% of the total number of model parameters.

We call the sum k 1 + k 2 = 10% the sparsity of the optimization algorithm.

Our experiments in this section test a number of image classification tasks for gradient sparsity, and plot the learning curves of some of these tasks.

We test a 2-layer fully connected neural network with hidden layers of width 100, a simple convolutional neural net which we describe in detail in appendix C, and Resnet-18 (He et al., 2015) .

All models use ReLu activations.

For datasets, we use CIFAR-10 (Krizhevsky et al.), SVHN (Netzer et al., 2011), and MNIST (LeCun & Cortes, 2010) .

None of our experiments include Resnet-18 on MNIST as MNIST is an easier dataset; it is included primarily to provide variety for the other models we include in this work.

Our method relies partially on the assumption that the magnitude of the derivative of some model parameters are greater than others.

To measure this, we compute the entropy of the empirical distribution over the magnitude of the derivative of the model parameters.

In Algorithm 1, the following term updates our estimate of the variance of each coordinate's derivative:

Consider the entropy of the following probability vector p = Mt Mt 1

.

The entropy of p provides us with a measure of how much structure there is in our gradients.

To see this, consider the hypothetical scenario where p i = 1 d .

In this scenario we have no structure; the top k 1 component of our sparsity operator is providing no value and entropy is maximized.

On the other hand, if a single entry p i = 1 and all other entries p j = 0, then the top k 1 component of our sparsity operator is effectively identifying the only relevant model parameter.

To measure the potential of our sparsity operator, we compute the entropy of p while running SpiderBoost on a variety of datasets and model architectures.

The results of running this experiment are summarized in the following table.

The entries of tables 1a and 1b correspond to the entropy of the memory vector before and after training.

For each model, the entropy at the beginning of training is almost maximal.

For example, maximum entropy of the convolutional model, which consists of 62, 006 parameters, is 15.92.

This is mainly due to random initialization of model parameters.

After 150 epochs, the entropy of M t for the convolutional model drops to approximately 3, which suggests a substantial amount of gradient structure.

Note that for the datasets that we tested, the gradient structure depends primarily on the model and not the dataset.

In particular, for Resnet-18, the entropy appears to vary minimally after 150 epochs.

, etc.

Our results of fitting the convolutional neural network to MNIST show that sparsity provides a significant advantage compared to using SpiderBoost alone.

We only show 2 epochs of this experiment since the MNIST dataset is fairly simple and convergence is rapidly achieved.

The results of training Resnet-18 on CIFAR-10 suggests that our sparsity algorithm works well on large neural networks, and non-trivial datasets.

Results for the rest of these experiments can be found in appendix C.

To further evaluate SpiderBoost with sparse gradients, as well as variance reduction methods in general, we test our algorithm on an LSTM (Hochreiter & Schmidhuber, 1997) The matrix factorization model is trained on the MovieLens database (Harper & Konstan, 2015) with a latent dimension of 20.

Further details can be found in C. We run SpiderBoost and Sparse SpiderBoost with a large-batch size B = 1030, small-batch size b = 103, inner loop length of m = 10.

For this experiment, we run SpiderBoost with a a learning rate schedule that interpolates from η = 1.0 to η = 0.1 as the algorithm progresses through the inner loop.

For instance, within the inner loop, at iteration 0 the learning rate is 1.0, and at iteration m it is 0.1.

We believe this is a natural way to utilize the low variance at the beginning of the inner loop, and is a fair comparison to an exponential decay learning rate schedule for SGD.

Details of the SGD baselines are provided in figure 2.

We see that for both tasks, SpiderBoost is slightly worse than SGD, and sparsity provides a slight improvement over SGD.

In this paper, we show how sparse gradients with memory can be used to improve the gradient query complexity of SVRG-type variance reduction algorithms.

While we provide a concrete sparse variance reduction algorithm for SpiderBoost, the techniques developed in this paper can be adapted to other variance reduction algorithms.

We show that our algorithm provides a way to explicitly control the gradient query complexity of variance reduction methods, a problem which has thus far not been explicitly addressed.

Assuming our algorithm captures the sparsity structure of the optimization problem, we also prove that the complexity of our algorithm is an improvement over SpiderBoost.

The results of our comparison to SpiderBoost validates this assumption, and our entropy experiment empirically supports the hypothesis that gradient sparsity does exist.

The results of our entropy experiment also support the results in Aji & Heafield (2017) , which show that the top k operator generally outperforms the random k operator.

Not every problem we tested exhibited sparsity structure.

While this is true, our analysis proves that our algorithm performs no worse than SpiderBoost in these settings.

Even when there is no structure, our algorithm reduces to a random sampling of k 1 + k 2 coordinates.

The results of our experiments on natural language processing and matrix factorization demonstrate that, with extra engineering effort, variance reduction methods can be competitive with SGD baselines.

While we view this as progress toward improving the practical viability of variance reduction algorithms, we believe further improvements can be made, such as better utilization of reduced variance during training, and better control over increased variance in very high dimensional models such as dense net (Defazio, 2019) .

We recognize these issues and hope to make progress on them in future work.

A TECHNICAL PROOFS

Lemma 2 (Lemma 3.1 of Lei & Jordan (2019)).

Let N ∼ Geom(m).

Then for any sequence

Proof of Lemma 1.

WLOG, assume that |x 1 | ≥ |x 2 | ≥ . . .

≥ |x d |.

Let S be a random subset of {k 1 + 1, . . .

, d} with size k 2 .

Then

As a result,

and

Therefore,

where (i) uses assumption A1 and (ii) uses the definition that x

Lemma 5.

For any j, t,

where E j,t and Var j,t are taken over the randomness of I (j) t and the random subset S involved in the rtop k1,k2 operator.

Proof.

By Lemma 4, we have

As a result,

The proof is then completed by Lemma 4.

Lemma 6.

For any j,

where E j is taken over all randomness in j-th outer loop (line 4-13 of Algorithm 1).

4.

Proof.

By definition,

As a result, ν

This implies that we can apply Lemma 2 on the sequence

Letting j = N j in Lemma 5 and taking expectation over all randomness in E j , we have

By Lemma 2,

where the last line uses the definition that

Nj .

By Lemma 3,

The proof is completed by putting equation 9, equation 10 and equation 11 together.

Lemma 7.

For any j, t,

Proof.

By equation 3,

The proof is then completed.

Lemma 8.

For any j,

where E j is taken over all randomness in j-th outer loop (line 4-13 of Algorithm 1).

Proof.

Since ∇f (x) ≤ σ for any x,

This implies that

As shown in equation 8, ν (j) t = Poly(t) and thus |f (x (j) t )| = Poly(t).

This implies that we can apply Lemma 2 on the sequence

Letting j = N j in Lemma 7 and taking expectation over all randomness in E j , we have

By Lemma 2,

The proof is then completed.

Combining Lemma 6 and Lemma 8, we arrive at the following key result on one inner loop.

Theorem 3.

For any j,

A weakness of our method is the technical difficulty of implementing a sparse backpropagation algorithm in modern machine learning libraries, such as Tensorflow (Abadi et al., 2015) and Pytorch (Paszke et al., 2017) .

Models implemented in these libraries generally assume dense structured parameters.

The optimal implementation of our algorithm makes use of a sparse forward pass and assumes a sparse computation graph upon which backpropagation is executed.

Libraries that support dynamic computation graphs, such as Pytorch, will construct the sparse computation graph in the forward pass.

This makes the required sparse backpropagation trivial and suggests that our algorithm will perform best on libraries that support dynamic computation graphs.

Consider the forward pass of a deep neural network, where φ is a deep composition of parametric functions,

The unconstrained problem of minimizing over the θ can be rewritten as a constrained optimization problem as follows:

In this form, z L+1 i is the model estimate for data point i. Consider φ (x; θ ) = σ(x T θ ) for 1 ≤ < L, φ L be the output layer, and σ be some subdifferentiable activation function.

If we apply the rtop k1,k2 operator per-layer in the forward-pass, with appropriate scaling of k 1 and k 2 to account for depth, we see that the number of multiplications in the forward pass is reduced to k 1 + k 2 : σ(rtop k1,k2 (v, x) T rtop k1,k2 (v, θ )).

A sparse forward-pass yields a computation graph for a (k 1 + k 2 )-parameter model, and back-propagation will compute the gradient of the objective with respect to model parameters in linear time (Chauvin & Rumelhart, 1995) .

The simple convolutional neural network used in the experiments consists of a convolutional layer with a kernel size of 5, followed by a max pool layer with kernel size 2, followed by another convolutional layer with kernel size 5, followed by a fully connected layer of input size 16 * side 2 × 120 (side is the size of the second dimension of the input), followed by a fully connected layer of size 120 × 84, followed by a final fully connected layer of size 84× the output dimension.

The natural language processing model consists of a word embedding of dimension 128 of 1, 000 tokens, which is jointly learned with the task.

The LSTM has a hidden and cell state dimension of 1024.

The variance reduction training algorithm for this type of model is given below.

The model M can be thought of as a classifier with cross entropy loss L and additional dependence on s i .

The batch gradient objective can therefore be formulated by considering the full sequence of predictions from i = 0 to i = |D| − 1, generating for each step i the outputD i+1 , s i+1 .

Each token is one-hot encoded, so the empirical risk is given by

In this setting, a dataset of length |D| is split into b contiguous sequences of length |D|/b and stored in a matrix Z b ∈ R b×(|D|/b) .

Taking a pass over Z b requires maintaining a state s i for each entry in a batch, which is reset before every pass over Z b .

To deal with maintaining state for batches at different time scales, we define a different matrix Z B ∈ R b×(|D|/B) which maintains a different set of states S i for each entry of batch size B.

@highlight

We use sparsity to improve the computational complexity of variance reduction methods.