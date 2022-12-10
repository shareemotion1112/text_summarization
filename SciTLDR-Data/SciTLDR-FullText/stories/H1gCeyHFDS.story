First-order methods such as stochastic gradient descent (SGD) are currently the standard algorithm for training deep neural networks.

Second-order methods, despite their better convergence rate, are rarely used in practice due to the pro- hibitive computational cost in calculating the second-order information.

In this paper, we propose a novel Gram-Gauss-Newton (GGN) algorithm to train deep neural networks for regression problems with square loss.

Our method draws inspiration from the connection between neural network optimization and kernel regression of neural tangent kernel (NTK).

Different from typical second-order methods that have heavy computational cost in each iteration, GGN only has minor overhead compared to first-order methods such as SGD.

We also give theoretical results to show that for sufficiently wide neural networks, the convergence rate of GGN is quadratic.

Furthermore, we provide convergence guarantee for mini-batch GGN algorithm, which is, to our knowledge, the first convergence result for the mini-batch version of a second-order method on overparameterized neural net- works.

Preliminary experiments on regression tasks demonstrate that for training standard networks, our GGN algorithm converges much faster and achieves better performance than SGD.

First-order methods such as Stochastic Gradient Descent (SGD) are currently the standard choice for training deep neural networks.

The merit of first-order methods is obvious: they only calculate the gradient and therefore are computationally efficient.

In addition to better computational efficiency, SGD has even more advantages among the first-order methods.

At each iteration, SGD computes the gradient only on a mini-batch instead of all training data.

Such randomness introduced by sampling the mini-batch can lead to better generalization (Hardt et al., 2015; Keskar et al., 2016; Masters & Luschi, 2018; Mou et al., 2017; Zhu et al., 2018) and better convergence (Ge et al., 2015; Jin et al., 2017a; b) , which is crucial when the function class is highly overparameterized deep neural networks.

Recently there is a huge body of works trying to develop more efficient first-order methods beyond SGD (Duchi et al., 2011; Kingma & Ba, 2014; Luo et al., 2019; Liu et al., 2019) .

Second-order methods, despite their better convergence rate, are rarely used to train deep neural networks.

At each iteration, the algorithm has to compute second order information, for example, the Hessian or its approximation, which is typically an m by m matrix where m is the number of parameters of the neural network.

Moreover, the algorithm needs to compute the inverse of this matrix.

The computational cost is prohibitive and usually it is not even possible to store such a matrix.

Formula and require subtle implementation tricks to use backpropagation.

In contrast, GGN has simpler update rule and better guarantee for neural networks.

In a concurrent and independent work, Zhang et al. (2019a) showed that natural gradient method and K-FAC have a linear convergence rate for sufficiently wide networks in full-batch setting.

In contrast, our method enjoys a higher-order (quadratic) convergence rate guarantee for overparameterized networks, and we focus on developing a practical and theoretically sound optimization method.

We also reveal the relation between our method and NTK kernel regression, so using results based on NTK (Arora et al., 2019b) , one can easily give generalization guarantee of our method.

Another independent work (Achiam et al., 2019) proposed a preconditioned Q-learning algorithm which has similar form of our update rule.

Unlike the methods considered in Zhang et al. (2019a) ; Achiam et al. (2019) which contain the learning rate that needed to be tuned, our derivation of GGN does not introduce a learning rate term (or understood as suggesting that the learning rate can be fixed to be 1 to get good performance which is verified in Figure 2 (c)).

Nonlinear least squares regression problem is a general machine learning problem.

Given data pairs {x i , y i } n i=1 and a class of nonlinear functions f , e.g. neural networks, parameterized by w, the nonlinear least squares regression aims to solve the optimization problem min w∈R m L(w) = 1 2

In the seminal work (Jacot et al., 2018) , the authors consider the case when f is a neural network with infinite width.

They showed that optimization on this problem using gradient flow involves a special kernel which is called neural tangent kernel (NTK).

The follow-up works further extended the relation between optimization and NTK which can be concluded in the following lemma: Lemma 1 (Lemma 3.1 in Arora et al. (2019a) , see also Dou & Liang (2019) ; Mei et al. (2019) ).

Consider optimizing problem (1) by gradient descent with infinitesimally small learning rate: dwt dt = −∇L(w t ).

where w t is the parameters at time t. Let f t = (f (w t , x i )) n i=1 ∈ R n be the network outputs on all x i 's at time t, and y = (y i )

n i=1 be the desired outputs.

Then f t follows the following evolution:

where G t is an n × n positive semidefinite matrix, i.e. the Gram matrix w.r.t.

the NTK at time t, whose (i, j)-th entry is ∇ w f (w t , x i ), ∇ w f (w t , x j ) .

The key idea of Jacot et al. (2018) and its extensions (Du et al., 2018b; a; Zou et al., 2018; Allen-Zhu et al., 2018a; Oymak & Soltanolkotabi, 2018; Lee et al., 2019; Yang, 2019; Arora et al., 2019b; a; Cao & Gu, 2019; Zou & Gu, 2019) is that when the network is sufficiently wide, the Gram matrix at initialization G 0 is close to a fixed positive definite matrix defined by the infinite-width kernel and G t is close to G 0 during training for all t. Under this situation, G t remains invertible, and the above dynamics is then identical to the dynamics of solving kernel regression with gradient flow w.r.t.

the current kernel at time t. In fact, Arora et al. (2019a) rigorously proves that a fully-trained sufficiently wide ReLU neural network is equivalent to the kernel regression predictor.

As pointed out in Chizat & Bach (2018) , the idea of NTK can be summarized as a linear approximation using first order Taylor expansion.

We give an example of this idea on the NTK at initialization:

where ∇ w f (w 0 , x) can then be viewed as an explicit expression of feature map at x, w − w 0 is the parameter in reproducing kernel Hilbert space (RKHS) induced by NTK and f (w,

The idea of linear approximation is also used in the classic Gauss-Newton method (Golub, 1965) to obtain an acceleration algorithm for solving nonlinear least squares problem (1).

Concretely, at iteration t, Gauss-Newton method takes the following first-order approximation:

where w t stands for the parameter at iteration t. We note that this is also the linear expansion for deriving NTK at time t. According to Eq. (1) and (4), to update the parameter, one can instead solve the following problem.

where f t , y have the same meaning as in Lemma 1, and J t = (∇ w f (w t , x 1 ), · · · , ∇ w f (w t , x n )) ∈ R n×m is the Jacobian matrix.

A necessary and sufficient condition for w to be the solution of Eq. (5) is

Below we will denote H t := J t J t ∈ R m×m .

For under-parameterized model (i.e., the number of parameters m is less than the number of data n), H t is invertible, and the update rule is

This can also be viewed as an approximate Newton's method using H t = J t J t to approximate the Hessian matrix.

In fact, the exact Hessian matrix is

In the case when f is only mildly nonlinear w.r.t.

w at data point x i 's, ∇ 2 w f (w t , x i ) ≈ 0, and H t is close to the real Hessian.

In this situation, the behavior of the Gauss-Newton method is similar to that of Newton's method, and thus can achieve a superlinear convergence rate (Golub, 1965) .

The classic second-order methods using approximate Hessian such as Gauss-Newton method described in the previous section face obvious difficulties dealing with the intractable approximate Hessian matrix when the regression model is an overparameterized neural network.

In Section 3.1, we develop a Gram-Gauss-Newton (GGN) method which is inspired by NTK kernel regression and does not require the computation of the approximate Hessian.

In Section 3.2, we show that for sufficiently wide neural networks, GGN has quadratic convergence rate.

In Section 3.3, we show that the additional computational cost (per iteration) of GGN compared to SGD is small.

We now describe our GGN method to learn overparameterized neural networks for regression problems.

As mentioned in the previous sections, for sufficiently wide networks, using gradient descent for solving the regression problem (1) has similar dynamics as using gradient descent for solving NTK kernel regression (Lemma 1) w.r.t.

NTK at each step.

However, one can also solve the kernel regression problem w.r.t.

the NTK at each step immediately using the explicit formula of kernel regression.

By explicitly solving instead of using gradient descent to solve NTK kernel regression, one can expect the optimization to get accelerated.

We propose our Gram-Gauss-Newton (GGN) method to directly solve the NTK kernel regression with Gram matrix G t at each time step t. Note that the feature map of NTK at time t, based on the approximation in Eq. (4), can be expressed as x → ∇ w f (w t , x), and the linear parameters in RKHS are w − w t , also the target is f (w, x i ) − f (w t , x i ).

Therefore, the kernel (ridgeless) regression solution (Mohri et al., 2018; Liang & Rakhlin, 2018)

where J t,S is the matrix of features at iteration t computed on the training data set S which is equal to the Jacobian , f t,S and y S are the vectorized outputs of neural network and the corresponding targets on S respectively, and

is the Gram matrix of the NTK on S.

One may wonder what is the relation between our derivation from NTK kernel regression and the Gauss-Newton method.

We point out that for overparameterized models, there are infinitely many solutions of Eq. (5) but our update rule (9) essentially uses the minimum norm solution.

In other words, the GGN update rule re-derives the Gauss-Newton method with the minimum norm solution.

This somewhat surprising connection is due to the fact that in kernel learning, people usually choose a kernel with powerful expressivity, i.e. the dimension of feature space is large or even infinite.

However, by the representer theorem (Mohri et al., 2018) , the solution of kernel (ridgeless) regression lies in the n-dimensional subspace of RKHS and minimizes the RKHS norm.

We refer the readers to Chapter 11 of Mohri et al. (2018) for details.

As mentioned in Section 1, the design of learning algorithms should consider not only optimization but also generalization.

It has been shown that using mini-batch instead of full batch to compute derivatives is crucial for the learned model to have good generalization ability (Hardt et al., 2015; Keskar et al., 2016; Masters & Luschi, 2018; Mou et al., 2017; Zhu et al., 2018) .

Therefore, we propose a mini-batch version of GGN.

The update rule is the following:

where B t is the mini-batch used at iteration t, J t,Bt and G t,Bt are the Jacobian and the Gram matrix computed using the data of B t respectively, and f t,Bt , y Bt are the vectorized outputs and the corresponding targets on B t respectively.

G t,Bt = J t,Bt J t,Bt is a very small matrix when using a typical batch size.

One difference between Eq. (10) and Eq. (7) is that our update rule only requires to compute the Gram matrix G t,Bt and its inverse.

Note that the size of G t,Bt is equal to the size of the mini-batch and is typically very small.

So this also greatly reduces the computational cost.

Using the idea of kernel ridge regression (which can also be viewed as Levenberg-Marquardt extension (Levenberg, 1944) of Gauss-Newton method), we introduce the following variant of GGN:

where λ > 0 is another hyper-parameter controlling the learning process.

Our algorithm is formally described in Algorithm 1.

Fetch a mini-batch B t from the dataset.

Calculate the Jacobian matrix J t,Bt .

Calculate the Gram matrix G t,Bt = J t,Bt J t,Bt .

Update the parameter by w t+1 = w t − J t,Bt (λG t,Bt + αI) −1 (f t,Bt − y Bt ).

8:

In this subsection, we show that for two-layer neural networks, if the width is sufficiently large, then: (1) Full-batch GGN converges with quadratic convergence rate.

(2) Mini-batch GGN converges linearly. (For clarity, here we only present a proof for two-layer neural networks, but we believe that it is not hard for the conclusion to be extended to deep neural networks using the techniques developed in Du et al. (2018a) ; Zou & Gu (2019) ).

As we explained through the lens of NTK, the result is a consequence of the fact that for wide enough neural networks, if the weights are initialized according to a suitable probability distribution, then with high probability the output of the network is close to a linear function w.r.t.

the parameters (but nonlinear w.r.t.

the input of the network) in a neighborhood containing the initialization point and a global optimum.

Although the neural networks used in practice are far from that wide, this still motivates us to design the GGN algorithm.

Neural network structure.

We use the following two-layer network

where x ∈ R d is the input, M is the network width, W = (w 1 , · · · , w M ) and σ(·) is the activation function.

Each entry of W is i.i.d.

initialized with the standard Gaussian distribution w r ∼ N (0, I d ) and each entry of a is initialized from the uniform distribution on {±1}. Similar to Du et al. (2018b) , we only train the network on parameter W just for the clarity of the proof.

We also assume the activation function σ(·) is -Lipschitz and β-smooth, and and β are regarded as O(1) absolute constants.

The key finding, as pointed out in Jacot et al. (2018); Du et al. (2018b; a) , is that under such initialization, the Gram matrix G has an asymptotic limit, which is, under mild conditions (e.g. input data not being degenerate etc.

, see Lemma F.2 of Du et al. (2018a) ), a positive definite matrix

(13) Assumption 1 (Least Eigenvalue of the Limit of Gram Matrix).

We assume the matrix K defined in (13) above is positive definite, and denote its least eigenvalue as

Now we are ready to state our theorem of full-batch GGN: Theorem 1 (Quadratic Convergence of Full-batch GGN on Overparameterized Neural Networks).

Assume Assumption 1 holds.

Assume the scale of the data is

, then with probability 1 − δ over the random initialization, the full-batch version of GGN whose update rule is given in Eq. (9) satisfies the following:

1) The Gram matrix G t,S at each iteration is invertible;

2) The loss converges to zero in a way that

for some C that is independent of M , which is a second-order convergence.

For the mini-batch version of GGN, by the analysis of its NTK limit, the algorithm is essentially doing serial subspace correction (Xu, 2001) on subspaces induced by mini-batch.

So mini-batch GGN is similar to the Gauss-Siedel method (Golub & Van Loan, 1996) applied to solving systems of linear equations, as shown in the proof of the following theorem.

Similar to the full batch situation, GGN takes the exact solution of the "kernel regression problem on the subspace" which is faster than just doing a gradient step to optimize on the subspace.

Moreover, we note that existing results of the convergence of SGD on overparameterized networks usually use the idea that when the step size is bounded by a quantity related to smoothness, the SGD can be reduced to GD.

However, our analysis takes a different way from the analysis of GD, thus does not rely on small step size.

In the following, we denote G 0 ∈ R n×n as the initial Gram matrix.

Let n = bk, where b is the batch size and k is the number of batches, and let

where

represents the block-diagonal and block-lower-triangular parts of G 0 .

We will show that the convergence of mini-batch GGN is highly related to the spectral radius of A. To simplify the proof, we make the following mild assumption on A: Assumption 2 (Assumption on the Iteration Matrix).

Assume the matrix A defined in (15) above is diagonalizable.

So we choose an arbitary diagonalization of A as A = P −1 QP and denote

We note that Assumption 2 is only for the sake of simplicity.

Even if it does not hold, an infinitesimally small perturbation can make any matrix diagonalizable, and it will not affect the proof.

Now we are ready to state the theorem for mini-batch GGN.

Theorem 2 (Convergence of Mini-batch GGN on Overparameterized Neural Networks).

Assume Assumption 1 and 2 hold.

Assume the scale of the data is

We use the mini-batch version of GGN whose update rule is given in Eq. (10), and the batch B t is chosen sequentially and cyclically with a fixed batch size b and k = n/b updates per epoch.

If the network width

then with probability 1 − δ over the random initialization, we have the following:

1) The Gram matrix G t,Bt at each iteration is invertible;

2) The loss converges to zero in a way that after T epochs, we have

Proof sketch for Theorem 1 and 2.

Denote J t = J(W t ) and

For the full-batch version, we have

Then we control the first term in Eq. (17) in a way similar to the following:

can be upper bounded, we get our result Eq. (14).

For the mini-batch version, similarly we have

where the subscript B t denotes the sub-matrix/vector corresponding to the batch, andG

is a zero-padded version of G −1 t,Bt to make Eq. (18) hold.

Therefore, after one epoch (from the (tk + 1)-th to the ((t + 1)k)-th update), we have

We will see that the matrix A t is close to the matrix A defined in (15), so it boils down to analyzing the spectral properties of A.

For both theorems, we can compute that as M increases, the norm of the update W t − W t+1 F does not increase with M , so the update is small compared to the Gaussian initialization where

.

From this we can derive that the matrices J, G etc. remain close to their initialization, which makes bounding their norms possible.

The full proof is in the appendix.

In conclusion, the accelerated convergence is related to the local linearity and the stability of the Jacobian and Gram matrix.

We emphasize that our theorems serve more as a motivation than a justification of our GGN algorithm, because we expect that GGN works in practice, even under milder situations when M is not as large as the theorem demands or for deep networks with different architectures, and that GGN would still perform much better than first-order methods.

We have proved that for sufficiently overparametrized deep neural networks, full-batch GGN has quadratic convergence rate.

In this subsection, we analyze the per-iteration computational cost of GGN, and compare it to that of SGD.

For every mini-batch (i.e., iteration), there are two major steps of computation in GGN:

• (A).

Forward, and then backpropagate for computing the Jacobian matrix J.

• (B).

Use J to compute the update J (λG + αI)

We show that the computational complexity of (A) is the same as that of SGD with the same batch size; and the computational complexity of (B) is small compared to (A) for typical networks and batch sizes.

Thus, the per-iteration computation overhead of GGN is very small compared to SGD.

Overall, in terms of training time, GGN can be much faster than SGD.

For the computation in step (A), the forward part is just the same as that of SGD.

For the backward part, for every input data, GGN keeps track of the output's derivative for the nodes in the middle of the computational graph.

This part is just the same as backpropagation in SGD.

What is different is that GGN also, for every input data, keeps track of the output's derivative for the parameters; while in SGD the derivatives for the parameters are averaged over a batch of data.

However, it is not difficult to see the computational costs of GGN and SGD are the same.

For the computation in step (B), observe that the size of the Jacobian is b × m where b is the batch size and m is the number of parameters.

The Gram matrix G t,Bt = J t,Bt J t,Bt in our GramGauss-Newton method is of size b × b and it only requires O(b 2 m + b 3 ) for computing G t,Bt and a matrix inverse.

Multiplying the two matrices to f − y requires even less computation.

Overall, the computational cost in step (B) is small compared to that of step (A).

Given the theoretical findings above, in this section, we compare our proposed GGN algorithm with several baseline algorithms in real applications.

In particular, we mainly study two regression tasks, AFAD-LITE (Niu et al., 2016) and RSNA Bone Age (rsn, 2017).

AFAD-LITE task is to predict the age of human from the facial information.

The training data of the AFAD-LITE task contains 60k facial images and the corresponding age for each image.

We choose ResNet-32 (He et al., 2016) as the base model architecture.

During training, all input images are resized to 64 * 64.

We study two variants of the ResNet-32 architecture: ResNet-32 with batch normalization layer (referred to as ResNetBN), and ResNet-32 with Fixup initialization (Zhang et al., 2019b ) (referred to as ResNetFixup).

In both settings, we use SGD as our baseline algorithm.

In particular, we follow Qian (1999) to use its momentum variant and set the hyper-parameters lr=0.003 and momentum=0.9 determined by selecting the best optimization performance using grid search.

Since batch normalization is computed over all samples within a mini-batch, it is not consistent with our assumption in Section 2 that the regression function has the form of f (w, x), which only depends on w and a single input datum x. For this reason, the GGN algorithm does not directly apply to ResNetBN, and we test our proposed algorithm on ResNetFixup only.

We set λ = 1 and α = 0.3 for GGN.

We follow the common practice to set the batch size to 128 for our proposed method and all baseline algorithms.

Mean square loss is used for training.

RSNA Bone Age task is a part of the 2017 Pediatric Bone Age Challenge organized by the Radiological Society of North America (RSNA).

It contains 12,611 labeled images.

Each image in this dataset is a radiograph of a left hand labeled with the corresponding bone age.

During training, all input images are resized to 64 * 64.

We also choose ResNetBN and ResNetFixup for this experiment, and use ResNetBN and ResNetFixup trained in the first task as warm-start initialization.

We use lr= 0.01 and momentum= 0.9 for SGD, and use λ = 1 and α = 0.1 for GGN.

Batch size is set to 128 in these experiments, and mean square loss is used for training.

Convergence.

The training loss curves of different optimization algorithms for AFAD-LITE and RSNA Bone Age tasks are shown in Figure 1 .

On both tasks, our proposed method converges much faster than the baselines.

We can see from Figure 1a and Figure 1b that, on the AFAD-LITE task, the loss using our GGN method quickly decreases to nearly zero in 30 epochs.

On the contrary, for both baselines using SGD, the loss decays much slower than our method in terms of wall clock time and epochs.

Similar advantage of GGN can also be observed on the RSNA bone age task.

Generalization performance and different hyper-parameters.

We can see that our proposed method trains much faster than other baselines.

However, as a machine learning model, generalization performance also needs to be evaluated.

Due to space limitation, we only provide the test curve for the RSNA Bone Age task in Figure 2a .

From the figure, we can see that the test loss of our proposed method also decreases faster than the baseline methods.

Furthermore, the loss of our GGN algorithm is lower than those of the baselines.

These results show that the GGN algorithm can not only accelerate the whole training process, but also learn better models.

We then study the effect of hyper-parameters used in the GGN algorithm.

We try different λ and α on the RSNA Bone Age task and report the training loss of all experiments at the 10 th epoch.

All results are plotted in Figure 2c .

In the figure, the x-axis is the value of λ and the y-axis is the value of α.

The gray value of each point corresponds to the loss, the lighter the color, the higher the loss.

We can see that the model converges faster when λ is close to 1.

In GGN, α can be considered as the inverse value of the learning rate in SGD.

Empirically, we find that the convergence speed of training loss is not that sensitive to α given a proper λ, such as λ = 1.

Some training loss curves of different hyper-parameter configurations are shown in Figure 2b .

We propose a novel Gram-Gauss-Newton (GGN) method for solving regression problems with square loss using overparameterized neural networks.

Despite being a second-order method, the computation overhead of the GGN algorithm at each iteration is small compared to SGD.

We also prove that if the neural network is sufficiently wide, GGN algorithm enjoys a quadratic convergence rate.

Experimental results on two regression tasks demonstrate that GGN compares favorably to SGD on these data sets with standard network architectures.

Our work illustrates that second-order methods have the potential to compete with first-order methods for learning deep neural networks with huge number of parameters.

In this paper, we mainly focus on the regression task, but our method can be easily generalized to other tasks such as classification as well.

Consider the k-category classification problem, the neural network outputs a vector with k entries.

Although this will increase the computational complexity of getting the Jacobian whose size increases k times, i.e., J ∈ R (bk)×m , each row of J can be still computed in parallel, which means the extra cost only comes from parallel computation overhead when we calculate in a fully parallel setting.

While most first-order methods for training neural networks can hardly make use of the computational resource in parallel or distributed settings to accelerate training, our GGN method can exploit this ability.

For first-order methods, basically extra computational resource can only be used to calculate more gradients at a time by increasing batch size, which harms generalization a lot.

But for GGN, more resource can be used to refine the gradients and achieve accelerated convergence speed with the help of second-order information.

It is an important future work to study the application of GGN to classification problems.

Notations.

We use the following notations for the rest of the sections.

• Let [n] = {1, · · · , n}.

• J W,x ∈ R M ×d denotes the gradient

∂W , which is of the same size of W.

• The bold J W or J(W) denotes the Jacobian with regard to all n data, with each gradient for W vectorized, i.e.

• w r denotes the r-th row of W, which is the incoming weights of the r-th neuron.

• W 0 stands for the parameters at initialization.

• d W,x := σ (Wx) ∈ R M ×1 denotes the (entry-wise) derivative of the activation function.

• We use ·, · to denote the inner product, · 2 to denote the Euclidean norm for vectors or the spectral norm for matrices, and · F to denote the Frobenius norm for matrices.

where • is the point-wise product.

So we can also easily solve G as

Our analysis is based on the fact that G stays not too far from its infinite-width limit

which is a positive definite matrix with least eigenvalue denoted λ 0 , and we assume λ 0 > 0.

λ 0 is a small data-dependent constant, and without loss of generality we assume λ 0 ≤ 1, or else we can just

The first lemma is about the estimation of relevant norms at initialization.

Lemma 2 (Bounds on Norms at Initialization).

If M = Ω (d log(16n/δ)), then with probability at least 1 − δ/2 the following holds

(c). (2010)).

Notice that W 0 ∈ R M ×d is a Gaussian random matrix, the Corollary states that with probability 1 − 2e

By choosing M = max(d, 2 log(8/δ)), we obtain W 0 2 ≤ 3 √ M with probability 1 − δ/4.

(b).

First, a r , r ∈ [M ] are Rademacher variables, thereby 1-sub-Gaussian, so with probability 1 − 2e

This means if we take M = Ω (log(16/δ)),

Next, the vector

is a standard Gaussian vector.

Suppose the activation σ(·) is l-Lipschitz and l is O(1) by our assumption, with the vector a fixed, the function

According to the classic result on the concentration of a Lipschitz function over Gaussian variables (see Theorem 2.26 of Wainwright (2019)), we have

holds jointly for all i ∈ [n] with probability 1 − δ/8.

Note that

Plugging in (22) and (24) into (23), we see that as long as M = Ω(log(16n/δ)), then with probability

According to (20) we can easily know that

The next lemma is about the least eigenvalue of the Gram matrix G at initialization.

It shows that when M is large, G W0 is close to K and has a lower bounded least eigenvalue.

It is the same as Lemma 3.1 in Du et al. (2018b) , but here we restate it and its proof for the reader's convenience.

, then with probability at least 1 − δ/2 over random initialization, we have

Proof.

Because σ is Lipschitz, σ (wx i )σ (wx j ) is bounded by O(1).

For every fixed (i, j) pair, at initialization G ij is an average of independent random variables, and by Hoeffding's inequality, applying union bound for all n 2 of (i, j) pairs, with probability 1 − δ/2 at initialization we have

and then

Next, in Lemma 4 and 5 we will bound the relevant norms and the least eigenvalue of G inside some scope of W that covers the whole optimization trajectory starting from W 0 .

Specifically, we consider the range

where R is determined later to make sure that the optimization trajectory remains inside B(R).

The idea of the whole convergence theorem is that when the width M is large, R is very small compared to its initialization scale:

.

This way, neither the Jacobian nor the Gram matrix changes much during optimization.

Lemma 4 (Bounds on Norms in the Optimization Scope).

Suppose the events in Lemma 2 hold.

There exists a constant C > 0 such that if M ≥ CR 2 , we have the following:

(a) For any W ∈ B(R), we have

Also, for any W ∈ B(R), we have

Proof.

(a).

This is straightforward from Lemma 2(a), the definition of B(R), and M = Ω(R 2 ).

(b).

According to the O(1)-smoothness of the activation, we have

so we can bound

And according to (19), we have

Also, taking W 1 = W and W 2 = W 0 , combining with Lemma 2(c), we see there exists C such that for M ≥ CR 2 we have J W F = O(1), and naturally , we have

and thus combined with Lemma 3, we know that G W remains invertible when W ∈ B(R) and satisfies

Proof.

Based on the results in Lemma 4(b), we have

To make the above less than

Proof idea.

In this section, we use W t , t ∈ {0, 1, · · · } to represent the parameter W after t iterations.

For convenience, J t , G t , f t is short for J Wt , G Wt , f Wt respectively.

We introduce

For each iteration, if G t is invertible, we have

Then we control the first term of the right hand side based on Lemma 4 in the following form

and plugging into (32) along with norm bounds on J and G we obtain a second-order convergence.

Formal proof.

Let R t = W t − W t+1 F for t ∈ {0, 1, · · · }.

We take R = Θ n λ0 in Lemma 4 and 5 (the constant is chosen to make the right hand side of (34) hold).

We prove that there exists an M = Ω max

(with enough constant) that suffices.

First we can easily verify that all the requirements for M in Lemma 2-5 can be satisfied .

Hence, with probability at least 1 − δ all the events in Lemma 2-5 hold.

Under this situation, we do induction on t to show the following:

• (a).

W t ∈ B(R).

•

As long as (b) is true for all t, then choosing M large enough to make sure the series { f t − y 2 } ∞ t=0

converges to zero, we obtain the second-order convergence property.

For t = 0, (a) and (b) hold by definition.

Suppose the proposition holds for t = 0, · · · , T .

Then for t = 0, · · · , T , G t is invertible.

Recall that the update rule is vec(

(Lemma 4 and 5)

According to Lemma 2(b) and the assumption that the target label y i = O(1), we have f 0 − y 2 2 = O(n).

When T > 1, the decay ratio at the first step is bounded as ) with enough constant can make sure r is a constant less than 1, in which case the second-order convergence property (b) will ensure a faster ratio of decay at each subsequent step in f t − y T t=0 .

Combining (33), we have

for variables r ∈ R n×1 (where vec(W) = vec(W 0 ) + J r), using the Gauss-Siedel method, which means solving

for the i-th batch in every epoch.

Therefore, it is natural that the matrix (15) is introduced.

We will show later that the real update follows

In order to prove the theorem, we need some additional lemmas as follows.

Lemma 6 (Formula for the Update).

If the Gram matrix at each step is invertible, then:

(b) The formula for f − y is

(c) The update of f − y satisfies

where

Or, if we denote

then we have

Specifically, we have

where

Proof.

(a) For (38), this is exactly the update formula (10) for the i-th batch where the Jacobian and Gram matrix are J ti,i and G ti,ii respectively.

Note that (39)), we obtain (40).

(c).

Based on (40) we know that

where the index goes in decreasing order from left to right.

So in order to prove (41) we only need to prove that

which we will prove by induction on i. For i = 0 it is trivial that D t − L t = U t0 by definition.

Suppose (44) holds for i, then

. . .

which proves (41).

Note that by the definition, we have U ti = D ti − L ti , we can then obtain (42) by

By (43), we can see that the iteration matrix A t is close to the matrix A defined in (15).

The convergence of the algorithm is much related to the eigenvalues of A. In the next two lemmas, we bound the spectral radius of A and provide an auxiliary result on the convergence on perturbed matrices based on the spectral radius.

Lemma 7.

Suppose the least eigenvalue of the initial Gram matrix G 0 satisfies λ min (G 0 ) ≥ 3 4 λ 0 , (which is true with probability 1 − δ if M = Ω n 2 log(2n/δ) λ 2 0 , according to Lemma 3).

Also assume J W0,x l F = O(1) for any l ∈ [n] (which is true with probability 1 − δ, according to Lemma 2).

Then the spectral radius A, or equivalently, maximum norm of eigenvalues of A, denoted ρ(A), satisfies

Proof.

For any eigenvalue λ ∈ C of A, it is an eigenvalue of A , so there exists a corresponding

where

.

It is not hard to see that

Also, since by our assumption each entry of L (or of G 0 ) is at most O(1), we have

Now we use take an inner product of v with (46) and get

and therefore

This concludes the proof for this lemma.

Lemma 8.

Denote ρ(A) = ρ 0 ≤ 1.

Let A be diagonalized as A = P −1 QP and µ = P 2 P −1 2 (see Assumption 2).

Suppose we have A t − A 0 2 ≤ δ for t ≤ T , then

In addition to the bounds used in Theorem 1, we provide the next lemma with useful bounds of the norms and eigenvalues of relevant matrices in the optimization scope

Lemma 9 (Relevant Bounds for the Matrices in the Optimization Scope).

We assume the events in Lemma 2 and 3 hold, and let M =

for some large enough constant C, then:

For any (t, i)-pair, we assume W t i ∈ B(R) for all i ∈ [k] when t ≤ t and i ∈ [i] when t = t in the following propositions.

.

Suppose up to t, W t is in B(R) (which means for i ∈ [k + 1], and t ∈ [t − 1], W ti ∈ B(R)).

(By the positive-definiteness of G 0 and Lemma 3) (b).

By Lemma 4 we know that within B(R), the size of the Jacobian J x l F w.r.t.

data x l is O(1),

, this can be applied to each entry of D, L, U, etc., including those J t (i ,i +1) terms by the convexity of B(R), and we can see that each entry of these matrices has size at most O(1) and varies inside an O R √ M range.

Therefore we get

and

suffices.

With all the preparations, now we are ready to prove Theorem 2.

The logic of the proof is just the same as what we did in the formal proof of Theorem 1, where we then used induction on t and now we use induction on the pair (t, i).

The key, is still selecting some R so that in each step W ti remains in B(R).

Combined with the previous Lemmas, in B(R) we have A t being close to A, and then convergence is guaranteed.

Formal proof of Theorem 2.

Let R ti = W t(i+1) − W ti F .

We take

in the range B(R) (where the constant is chosen to make the right hand side of (52) hold).

We prove that there exists an

with a large enough constant that suffices.

First we can easily verify that all the requirements for M in Lemma 2-9 (most importantly, Lemma 9) can be satisfied.

Hence, with probability at least 1 − δ all the events in Lemma 2-9 hold.

Under this situation, we do induction on (t, i) (t ∈ {1, 2, · · · }, i ∈ [k], by dictionary order) to show that:

For (t, i) = (1, 1), it holds by definition.

Suppose the proposition holds up to (t, i).

Then since W ti ∈ B(R), by Lemma 4 we know that

2 .

This naturally gives us λ min (G ti,ii ) ≥ λ0 2 , which means G ti,ii is invertible.

Similar to the argument in the proof of Lemma 9, we know that each entry of J ti , J t(i,i+1) , D ti , L ti , U ti , etc., whose index of (t , i ) only contains i with i ≤ i, is of scale Based on the update rule (38), we have

Since this also hold for previous (t, i) pairs, we have

which is the reason why we need to take R = Θ .

This means that W ti ∈ B(R) holds.

And by induction, we have proved that W remains in B(R) throughout the optimization process.

The last thing to do is to bound f t − y. By the same logic from above, we have , which proves our theorem.

In this section, we give test performance curve of AFAD-LITE dataset in Figure 3 under the same setting with Section 4.

In addition, we provide more baseline results, e.g. Adam (Kingma & Ba, 2014) and K-FAC (Martens & Grosse, 2015) on RSNA Bone Age dataset.

Since we find that, as another alternation of BN, Group Normalization (GN) (Wu & He, 2018) can largely improve the performance of Adam, we also implement the GN layer for our GGN method.

We use grid search to obtain approximately best hyper-parameters for every experiment.

All experiments are performed with batch size 128, input size 64*64 and weight decay 10 −4 .

We set the number of groups to 8 for ResNetGN.

Other hyper-parameters are listed below.

• SGD+ResNetBN: learning rate 0.01, momentum 0.9.

• Adam+ResNetBN: learning rate 0.001.

• SGD+ResNetGN: learning rate 0.002, momentum 0.9.

• Adam+ResNetGN: learning rate 0.0005.

• K-FAC+ResNet: learning rate 0.02, momentum 0.9, = 0.1, update frequency 100.

• GGN+ResNetGN: λ = 1, α = 0.075.

The convergence results are summarized in Figure 4 .

Note to make comparison clearer, we use logarithmic scale for training curves.

@highlight

A novel Gram-Gauss-Newton method to train neural networks, inspired by neural tangent kernel and Gauss-Newton method, with fast convergence speed both theoretically and experimentally.