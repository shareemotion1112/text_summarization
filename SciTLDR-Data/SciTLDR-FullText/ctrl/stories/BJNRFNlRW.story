We relate the minimax game of generative adversarial networks (GANs) to finding the saddle points of the Lagrangian function for a convex optimization problem, where the discriminator outputs and the distribution of generator outputs play the roles of primal variables and dual variables, respectively.

This formulation shows the connection between the standard GAN training process and the primal-dual subgradient methods for convex optimization.

The inherent connection does not only provide a theoretical convergence proof for training GANs in the function space, but also inspires a novel objective function for training.

The modified objective function forces the distribution of generator outputs to be updated along the direction according to the primal-dual subgradient methods.

A toy example shows that the proposed method is able to resolve mode collapse, which in this case cannot be avoided by the standard GAN or Wasserstein GAN.

Experiments on both Gaussian mixture synthetic data and real-world image datasets demonstrate the performance of the proposed method on generating diverse samples.

Generative adversarial networks (GANs) are a class of game theoretical methods for learning data distributions.

It trains the generative model by maintaining two deep neural networks, namely the discriminator network D and the generator network G. The generator aims to produce samples resembling real data samples, while the discriminator aims to distinguish the generated samples and real data samples.

The standard GAN training procedure is formulated as the following minimax game: DISPLAYFORM0 where p d (x) is the data distribution and p z (z) is the noise distribution.

The generated samples G(z) induces a generated distribution p g (x).

Theoretically, the optimal solution to (1) is p * g = p d and D * (x) = 1/2 for all x in the support of data distribution.

In practice, the discriminator network and the generator network are parameterized by θ θ θ d and θ θ θ g , respectively.

The neural network parameters are updated iteratively according to gradient descent.

In particular, the discriminator is first updated either with multiple gradient descent steps until convergence or with a single gradient descent step, then the generator is updated with a single descent step.

However, the analysis of the convergence properties on the training approaches is challenging, as noted by Ian Goodfellow in BID10 , "For GANs, there is no theoretical prediction as to whether simultaneous gradient descent should converge or not.

Settling this theoretical question, and developing algorithms guaranteed to converge, remain important open research problems.".

There have been some recent studies on the convergence behaviours of GAN training (Nowozin et al., 2016; BID18 BID14 BID24 BID22 .The simultaneous gradient descent method is proved to converge assuming the objective function is convex-concave in the network parameters (Nowozin et al., 2016) .

The local stability property is established in BID14 BID24 .One notable inconvergence issue with GAN training is referred to as mode collapse, where the generator characterizes only a few modes of the true data distribution BID11 BID18 .

Various methods have been proposed to alleviate the mode collapse problem.

Feature matching for intermediate layers of the discriminator has been proposed in (Salimans et al., 2016) .

In BID23 , the generator is updated based on a sequence of previous unrolled discriminators.

A mixture of neural networks are used to generate diverse samples (Tolstikhin et al., 2017; BID15 BID2 .

In , it was proposed that adding noise perturbation on the inputs to the discriminator can alleviate the mode collapse problem.

It is shown that this training-with-noise technique is equivalent to adding a regularizer on the gradient norm of the discriminator (Roth et al., 2017) .

The Wasserstein divergence is proposed to resolve the problem of incontinuous divergence when the generated distribution and the data distribution have disjoint supports BID12 .

Mode regularization is used in the loss function to penalize the missing modes BID6 Srivastava et al., 2017) .

The regularization is usually based on heuristics, which tries to minimize the distance between the data samples and the generated samples, but lacks theoretical convergence guarantee.

In this paper, we formulate the minimax optimization for GAN training (1) as finding the saddle points of the Lagrangian function for a convex optimization problem.

In the convex optimization problem, the discriminator function D(·) and the probabilities of generator outputs p g (·) play the roles of the primal variables and dual variables, respectively.

This connection not only provides important insights in understanding the convergence of GAN training, but also enables us to leverage the primal-dual subgradient methods to design a novel objective function that helps to alleviate mode collapse.

A toy example reveals that for some cases when standard GAN or WGAN inevitably leads to mode collapse, our proposed method can effectively avoid mode collapse and converge to the optimal point.

In this paper, we do not aim at achieving superior performance over other GANs, but rather provide a new perspective of understanding GANs, and propose an improved training technique that can be applied on top of existing GANs.

The contributions of the paper are as follows:• The standard training of GANs in the function space is formulated as primal-dual subgradient methods for solving convex optimizations.• This formulation enables us to show that with a proper gradient descent step size, updating the discriminator and generator probabilities according to the primal-dual algorithms will provably converge to the optimal point.• This formulation results in a novel training objective for the generator.

With the proposed objective function, the generator is updated such that the probabilities of generator outputs are pushed to the optimal update direction derived by the primal-dual algorithms.

Experiments have shown that this simple objective function can effectively alleviate mode collapse in GAN training.• The convex optimization framework incorporates different variants of GANs including the family of f -GAN (Nowozin et al., 2016) and an approximate variant of WGAN.

For all these variants, the training objective can be improved by including the optimal update direction of the generated probabilities.

In this section, we first describe the primal-dual subgradient methods for convex optimization.

Later, we explicitly construct a convex optimization and relate the subgradient methods to standard GAN training.

Consider the following convex optimization problem: DISPLAYFORM0 where x ∈ R k is a length-k vector, X is a convex set, and f i (x), i = 0 · · · , , are concave functions mapping from R k to R. The Lagrangian function is calculated as DISPLAYFORM1 In the optimization problem, the variables x ∈ R k and λ λ λ ∈ R + are referred to as primal variables and dual variables, respectively.

The primal-dual pair (x * , λ λ λ * ) is a saddle-point of the Lagrangian fuction, if it satisfies: DISPLAYFORM2 Primal-dual subgradient methods have been widely used to solve the convex optimization problems, where the primal and dual variables are updated iteratively, and converge to a saddle point (Nedić & Ozdaglar, 2009; BID16 .There are two forms of algorithms, namely dual-driven algorithm and primal-dual-driven algorithm.

For both approaches, the dual variables are updated according to the subgradient of L(x(t), λ λ λ(t)) with respect to λ λ λ(t) at each iteration t. For the dual-driven algorithm, the primal variables are updated to achieve maximum of L(x, λ λ λ(t)) over x. For the primal-dual-driven algorithm, the primal variables are updated according to the subgradient of L(x(t), λ λ λ(t)) with respect to x(t).

The iterative update process is summarized as follows: DISPLAYFORM3 where P X (·) denotes the projection on set X and (x) + = max(x, 0).The following theorem proves that the primal-dual subgradient methods will make the primal and dual variables converge to the optimal solution of the convex optimization problem.

Theorem 1 Consider the convex optimization (2).

Assume the set of saddle points is compact.

Suppose f 0 (x) is a strictly concave function over x ∈ X and the subgradient at each step is bounded.

There exists some step size α (t) such that both the dual-driven algorithm and the primal-dual-driven algorithm yield x (t) → x * and λ λ λ (t) → λ λ λ * , where x * is the solution to (2), and λ λ λ * satisfies DISPLAYFORM4 Proof: See Appendix 7.1.

We explicitly construct a convex optimization problem and relate it to the minimax game of GANs.

We assume that the source data and generated samples belong to a finite set {x 1 , · · · , x n } of arbitrary size n. The extension to uncountable sets can be derived in a similar manner BID20 .

The finite case is of particular interest, because any real-world data has a finite size, albeit the size could be arbitrarily large.

We construct the following convex optimization problem: DISPLAYFORM0 where D is some convex set.

The primal variables are DISPLAYFORM1 ) is the Lagrangian dual associated with the i-th constraint.

The Lagrangian function is thus DISPLAYFORM2 When D = {D : 0 ≤ D i ≤ 1, ∀i}, finding the saddle points for the Lagrangian function is exactly equivalent to solving the GAN minimax problem(1).

This inherent connection enables us to utilize the primal-dual subgradient methods to design update rules for D(x) and p g (x) such that they converge to the saddle points.

The following theorem provides a theoretical guideline for the training of GANs.

Theorem 2 Consider the Lagrangian function given by (9) with D = {D : ≤ D i ≤ 1 − , ∀i}, where 0 < < 1/2.

If the discriminator and generator have enough capacity, and the discriminator output and the generated distribution are updated according to the primal-dual update rules (5) and FORMULA4 with DISPLAYFORM3 Proof: The optimization problem (8) is a particularized form of (2) , where DISPLAYFORM4 The objective function is strictly concave over D. Moreover, since D is projected onto the compact set [ , 1 − ] at each iteration t, the subgradients ∂f i (D (t) ) are bounded.

The assumptions of Theorem 1 are satisfied.

Since the constraint (8b) gives an upper bound of D i ≤ 1/2, the solution to the above convex optimization is obviously DISPLAYFORM5 Since the problem is convex, the optimal primal solution is the primal saddle point of the Lagrangian function (Bertsekas, 1999, Chapter 5) .

DISPLAYFORM6 , and the saddle point is unique.

By Theorem 1, the primal-dual update rules will guarantee convergence of DISPLAYFORM7 It can be seen that the standard training of GAN corresponds to either dual-driven algorithm (Nowozin et al., 2016) or primal-dual-driven algorithm BID11 .

A natural question arises: Why does the standard training fail to converge and lead to mode collapse?

As will be shown later, the underlying reason is that standard training of GANs in some cases do not update the generated distribution according to (6).

Theorem 2 inspires us to propose a training algorithm to tackle this issue.

First, we present our training algorithm.

Later, we will use a toy example to give intuitions of why our algorithm is effective to avoid mode collapse.

The algorithm is described in Algorithm 1.

The maximum step of discriminator update is k 0 .

In the context of primal-dual-driven algorithms, k 0 = 1.

In the context of dual-driven algorithms, k 0 is some large constant, such that the discriminator is updated till convergence at each training epoch.

The update of the discriminator is the same as standard GAN training.

The main difference is the modified loss function for the generator update (13).

The intuition is that when the generated samples have disjoint support from the data, the generated distribution at the data support may not be updated using standard training.

This is exactly one source of mode collapse.

Ideally, the modified loss function will always update the generated probabilities at the data support along the optimal direction.

The generated probability mass at DISPLAYFORM0 where 1{·} is the indicator function.

The indicator function is not differentiable, so we use a continuous kernel to approximate it.

Define DISPLAYFORM1 where σ is some positive constant.

The constant σ is also called bandwidth for kernel density estimation.

The empirical generated distribution is thus approximately calculated as (17).

There

Initialization: Choose the objective function f 0 (·) and constraint function f 1 (·) according to the GAN realization.

For the original GAN based on Jensen-Shannon divergence, f 0 (D) = log (D) and f 1 (D) = log(2(1 − D)).

while the stopping criterion is not met do Sample minibatch m 1 data samples DISPLAYFORM0 Update the discriminator parameters with gradient ascent: DISPLAYFORM1 end for Update the target generated distribution as: DISPLAYFORM2 where α is some step size and DISPLAYFORM3 Withp g (x i ) fixed, update the generator parameters with gradient descent: DISPLAYFORM4 end while are different bandwidth selection methods BID5 BID13 .

It can be seen that as σ → 0, k σ (x − y) tends to the indicator function, but it will not give large enough gradients to far areas that experience mode collapse.

A larger σ implies a coarser quantization of the space in approximating the distribution.

In practical training, the kernel bandwidth can be set larger at first and gradually decreases as the iteration continues.

By the dual update rule (6) , the generated probability of every x i should be updated as DISPLAYFORM5 This motivates us to add the second term of (13) in the loss function, such that the generated distribution is pushed towards the target distribution (15).Although having good convergence guarantee in theory, the non-parametric kernel density estimation of the generated distribution may suffer from the curse of dimension.

Previous works combining kernel learning and the GAN framework have proposed methods to scale the algorithms to deal with high-dimensional data, and the performances are promising BID19 BID17 Sinn & Rawat, 2017) .

One common method is to project the data onto a low dimensional space using an autoencoder or a bottleneck layer of a pretrained neurual network, and then apply the kernel-based estimates on the feature space.

Using this approach, the estimated probability of x i becomes DISPLAYFORM6 where f φ (.) is the projection of the data to a low dimensional space.

We will leave the work of generating high-resolution images using this approach as future work.

Mode collapse occurs when the generated samples have a very small probability to overlap with some families of the data samples, and the discriminator D(·) is locally constant around the region of the generated samples.

We use a toy example to show that the standard training of GAN and Wasserstein may fail to avoid mode collapse, while our proposed method can succeed.

Claim 1 Suppose the data distribution is p d (x) = 1{x = 1}, and the initial generated distribution is p g (x) = 1{x = 0}. The discriminator output D(x) is some function that is equal to zero for |x − 0| ≤ δ and is equal to one for |x − 1| ≤ δ, where 0 < δ < 1/2.

Standard training of GAN and WGAN leads to mode collapse.

Proof: We first show that the discriminator is not updated, and then show that the generator is not updated during the standard training process.

In standard training of GAN and WGAN, the discriminator is updated according to the gradient of (10).

For GAN, since 0 ≤ D(x) ≤ 1, the objective funtion for the discriminator is at most zero, i.e., DISPLAYFORM0 which is achieved by the current D(x) by assumption.

For WGAN, the optimal discrminator output D(x) is some 1-Lipschitz function such that DISPLAYFORM1 where (19) is due to the Lipschitz condition |D(1) − D(0)| ≤ 1.

The current D(x) is obviously optimal.

Thus, for both GAN and WGAN, the gradient of the loss function with respect to θ θ θ d is zero and the discriminator parameters are not updated.

On the other hand, in standard training, the generator parameters θ θ θ g are updated with only the first term of (13).

By the chain rule, DISPLAYFORM2 where (21) is due to the assumption that D(x) is locally constant for x = 0.

Therefore, the generator and the discriminator reach a local optimum point.

The generated samples are all zeros.

In our proposed training method, when x = 1, the optimal update direction is given by (11), wherẽ p g is a large value because D(1) = 1.

Therefore, by (13), the second term in the loss function is very large, which forces the generator to generate samples at G(z) = 1.

As the iteration continues, the generated distribution gradually converges to data distribution, and D(x) gradually converges to 1/2, which makes ∂ pg(x) L(D(x), p g (x)) = log(2(1 − D(x))) become zero.

The experiment in Section 5 demonstrates this training dynamic.

In this paper, the standard training of GANs in function space has been formulated as primal-dual updates for convex optimization.

However, the training is optimized over the network parameters in practice, which typically yields a non-convex non-concave problem.

Theorem 2 tells us that as long as the discriminator output and the generated distribution are updated according to the primal-dual update rule, mode collapse should not occur.

This insight leads to the addition of the second term in the modified loss function for the generator (13).

In Section 5, experiments on the above-mentioned toy example and real-world datasets show that the proposed training technique can greatly improve the baseline performance.

Consider the following optimization problem: DISPLAYFORM0 DISPLAYFORM1 where f 0 (·) and f 1 (·) are concave functions.

Compared with the generic convex optimization problem (2), the number of constraint functions is set to be the variable alphabet size, and the constraint functions are DISPLAYFORM2 The objective and constraint functions in (22) can be tailored to produce different GAN variants.

For example, TAB0 shows the large family of f -GAN (Nowozin et al., 2016) .

The last row of TAB0 gives a new realization of GAN with a unique saddle point of D * (x) = 2 and DISPLAYFORM3 We also derive a GAN variant similar to WGAN, which is named "Approximate WGAN".

As shown in TAB0 , the objective and constraint functions yield the following minimax problem: DISPLAYFORM4 where is an arbitrary positive constant.

The augmented term D 2 (x) is to make the objective function strictly concave, without changing the original solution.

It can be seen that this problem has a unique saddle point p * g (x) = p d (x).

As tends to 0, the training objective function becomes identical to WGAN.

The optimal D(x) for WGAN is some Lipschitz function that maximizes E x∼p d (x) {D(x)} − E x∼pg(x) {D(x)}, while for our problem is D * (x) = 0.

Weight clipping can still be applied, but serves as a regularizer to make the training more robust BID21 .The training algorithms for these variants of GANs follow by simply changing the objective function f 0 (·) and constraint function f 1 (·) accordingly in Algorithm 1.

5.1 SYNTHETIC DATA FIG0 shows the training performance for a toy example.

The data distribution is p g (x) = 1{x = 1}. The inital generated samples are concentrated around x = −3.0.

The details of the neural network parameters can be seen in Appendix 7.3.

FIG0 shows the generated samples in the 90 quantile as the training iterates.

After 8000 iterations, the generated samples from standard training of GAN and WGAN are still concentrated around x = −3.0.

As shown in FIG0 , the discrminators hardly have any updates throughout the training process.

Using the proposed training approach, the generated samples gradually converge to the data distribution and the discriminator output converges to the optimal solution with D(1) = 1/2.

Fig. 2 shows the performance of the proposed method for a mixture of 8 Gaussain data on a circle.

While the original GANs experience mode collapse BID15 BID23 , our proposed method is able to generate samples over all 8 modes.

In the training process, the bandwidth of the Gaussian kernel (14) is inialized to be σ 2 = 0.1 and decreases at a rate of 0.8 DISPLAYFORM0 , where t is the iteration number.

The generated samples are dispersed initially, and then gradually converge to the Gaussian data samples.

Note that our proposed method involves a low complexity with a simple regularization term added in the loss function for the generator update.

Figure 2: Performance of the proposed algorithm on 2D mixture of Gaussian data.

The data samples are marked in blue and the generated samples are marked in orange.

We also evaluate the performance of the proposed method on two real-world datasets: MNIST and CIFAR-10.

Please refer to the appendix for detailed architectures.

Inception score (Salimans et al., 2016 ) is employed to evaluate the proposed method.

It applies a pretrained inception model to every generated image to get the conditional label distribution p(y|x).

The Inception score is calculated as exp (E x {KL(p(y|x) p(y)}).

It measures the quality and diversity of the generated images.

The MNIST dataset contains 60000 labeled images of 28 × 28 grayscale digits.

We train a simple LeNet-5 convolutional neural network classifier on MNIST dataset that achieves 98.9% test accuracy, and use it to compute the inception score.

The proposed method achieves an inception score of 9.8, while the baseline method achieves an inception score of 8.8.

The examples of generated images are shown in Fig. 3 .

The generated images are almost indistinguishable from real images.

We further evaluated our algorithm on an augmented 1000-class MNIST dataset to further demonstrate the robustness of the proposed algorithm against mode collapse problem.

More details of the experimental results can be found in the Appendix.

CIFAR is a natural scene dataset of 32 × 32.

We use this dataset to evaluate the visual quality of the generated samples.

Table 2 shows the inception scores of different GAN models on CIFAR-10 dataset.

The inception score of the proposed model is much better than the baseline method WGAN MNIST CIFAR Figure 3 : Examples of generated images using MNIST and CIFAR dataset.

Method ScoreReal data 11.24 ± 0.16 WGAN 3.82 ± 0.06 MIX + WGAN BID2 4.04 ± 0.07 Improved-GAN (Salimans et al., 2016) 4.36 ± 0.04 ALI BID7 5.34 ± 0.05 DCGAN (Radford et al., 2015) 6.40 ± 0.05Proposed method 4.53 ± 0.04 Table 2 : Inception scores on CIFAR-10 dataset.that uses similar network architecture and training method.

Note that although DCGGAN achieves a better score, it uses a more complex network architecture.

Examples of the generated images are shown in Fig. 3 .

In this paper, we propose a primal-dual formulation for generative adversarial learning.

This formulation interprets GANs from the perspective of convex optimization, and gives the optimal update of the discriminator and the generated distribution with convergence guarantee.

By framing different variants of GANs under the convex optimization framework, the corresponding training algorithms can all be improved by pushing the generated distribution along the optimal direction.

Experiments on two synthetic datasets demonstrate that the proposed formulation can effectively avoid mode collapse.

It also achieves competitive quantitative evaluation scores on two benchmark real-world image datasets.

The proof of convergence for dual-driven algorithms can be found in BID4 , Chapter 3).The primal-dual-driven algorithm for continuous time update has been studied in BID8 .

Here, we show the convergence for the discrete-time case.

We choose a step size α(t) that satisfies DISPLAYFORM0 Let z(t) = [x(t), λ λ λ(t)]

T be a vector consisting of the primal and dual variables at the t-th iteration.

The primal-dual-driven update can be expressed as: DISPLAYFORM1 where DISPLAYFORM2 and DISPLAYFORM3 Since the subgradient is bounded by assumption, there exists M > 0 such that ||T (·)|| 2 2 < M , where ||.|| 2 stands for the L 2 norm.

Modes generated Inception Score BID23 , and the networks are trained with Root Mean Square Propagation (RMSProp) with a learning rate of 1e-4.

For GAN, the networks are trained with Adam with a learning rate of 1e-4.

The minibatch size is 32.

The bandwidth parameter for the Gaussian kernel is initialized to be σ = 0.5 and then is changed to 0.1 after 2000 iterations.

We use the network structure in BID23 to evaluate the performance of our proposed method.

The data is sampled from a mixture of 8 Gaussians of standard deviation of 0.02 uniformly located on a circle of radius 2.

The noise samples are a vector of 256 independent and identically distributed (i.i.d.)

Gaussian variables with mean zero and standard deviation of 1.The generator has two hidden layers of size 128 with ReLU activation.

The last layer is a linear projection to two dimensions.

The discriminator has one hidden layer of size 128 with ReLU activation followed by a fully connected network to a sigmoid activation.

All the biases are initialized to be zeros and the weights are initalilzed via the "Xavier" initialization BID9 .

The training follows the primal-dual-driven algorithm, where both the generator and the discriminator are updated once at each iteration.

The Adam optimizer is used to train the discriminator with 8e-4 learning rate and the generator with 4e-4 learning rate.

The minibatch sample number is 64.

For MNIST dataset, the generator network is a deconvolutional neural network.

It has two fully connected layer with hidden size 1024 and 7 × ×7 × 128, two deconvolutional layers with number of units 64, 32, stride 2 and deconvolutional kernel size 4 × 4 for each layer, respectively, and a final convolutional layer with number of hidden unit 1 and convolutional kernel 4 × 4..

The discriminator network is a two layer convolutional neural network with number of units 64, 32 followed by two fully connected layer of hidden size 1024 and 1.

The input noise dimension is 64.We employ ADAM optimization algorithm with initial learning rate 0.01 and β = 0.5.

For CIFAR dataset, the generator is a 4 layer deconvolutional neural network, and the discriminator is a 4 layer convolutional neural network.

The number of units for discriminator is [64, 128, 256, 1024] , and the number of units for generator is [1024, 256, 128, 64] .

The stride for each deconvolutional and convolutional layer is two.

We employ RMSProp optimization algorithm with initial learning rate of 0.0001, decay rate 0.95, and momentum 0.1.

<|TLDR|>

@highlight

We propose a primal-dual subgradient method for training GANs and this method effectively alleviates mode collapse.