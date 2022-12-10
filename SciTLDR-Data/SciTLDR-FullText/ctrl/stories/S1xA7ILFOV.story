Optimal Transport (OT) naturally arises in many machine learning applications, where we need to handle cross-modality data from multiple sources.

Yet the heavy computational burden limits its wide-spread uses.

To address the scalability issue, we propose an implicit generative learning-based framework called SPOT (Scalable Push-forward of Optimal Transport).

Specifically, we approximate the optimal transport plan by a pushforward of a reference distribution, and cast the optimal transport problem into a minimax problem.

We then can solve OT problems efficiently using primal dual stochastic gradient-type algorithms.

We also show that we can recover the density of the optimal transport plan using neural ordinary differential equations.

Numerical experiments on both synthetic and real datasets illustrate that SPOT is robust and has favorable convergence behavior.

SPOT also allows us to efficiently sample from the optimal transport plan, which benefits downstream applications such as domain adaptation.

The Optimal Transport (OT) problem naturally arises in a variety of machine learning applications, where we need to handle cross-modality data from multiple sources.

One example is domain adaptation: We collect multiple datasets from different domains, and we need to learn a model from a source dataset, which can be further adapted to target datasets BID18 BID8 .

Another example is resource allocation: We want to assign a set of assets (one data source) to a set of receivers (another data source) so that an optimal economic benefit is achieved BID46 BID17 .

Recent literature has shown that both aforementioned applications can be formulated as optimal transport problems.

The optimal transport problem has a long history, and its earliest literature dates back to Monge (1781).

Since then, it has attracted increasing attention and been widely studied in multiple communities such as applied mathematics, probability, economy and geography BID51 Carlier, 2012; BID23 .

Specifically, we consider two sets of data, which are generated from two different distributions denoted by X ∼ µ and Y ∼ ν.1 We aim to find an optimal joint distribution γ of X and Y , which minimizes the expectation on some ground cost function c, i.e., γ * = arg min γ∈Π(µ,ν) DISPLAYFORM0 The constraint γ ∈ Π(µ, ν) requires the marginal distribution of X and Y in γ to be identical to µ and ν, respectively.

The cost function c measures the discrepancy between input X and Y .

For crossmodality structured data, the form of c incorporates prior knowledge into optimal transport problem.

Existing literature often refers to the optimal expected cost W * (µ, ν) = E (X,Y )∼γ * [c(X, Y )] as Wasserstein distance when c is a distance, and γ * as the optimal transport plan.

For domain adaptation, the function c measures the discrepancy between X and Y , and the optimal transport plan γ * essentially reveals the transfer of the knowledge from source X to target Y .

For resource allocation, the function c is the cost of assigning resource X to receiver Y , and the optimal transport plan γ To address the scalability and efficiency issues, we propose a new implicit generative learning-based framework for solving optimal transport problems.

Specifically, we approximate γ * by a generative model, which maps from some latent variable Z to (X, Y ).

For simplicity, we denote DISPLAYFORM1 where ρ is some simple latent distribution and G is some operator, e.g., deep neural network or neural ordinary differential equation (ODE) .

Accordingly, instead of directly estimating the probability density of γ * , we estimate the mapping G between Z and (X, Y ) by solving DISPLAYFORM2 We then cast equation 3 into a minimax optimization problem using the Lagrangian multiplier method.

As the constraints in equation 3 are over the space of continuous distributions, the Lagrangian multiplier is actually infinite dimensional.

Thus, we propose to approximate the Lagrangian multiplier by deep neural networks, which eventually delivers a finite dimensional generative learning problem.

Our proposed framework has three major benefits: (1) Our formulated minimax optimization problem can be efficiently solved by primal dual stochastic gradient-type algorithms.

Many empirical studies have corroborated that these algorithms can easily scale to very large minimax problems in machine learning BID2 ; (2) Our framework can take advantage of recent advances in deep learning.

Many empirical evidences have suggested that deep neural networks can effectively adapt to data with intrinsic low dimensional structures BID33 .

Although they are often overparameterized, due to the inductive biases of the training algorithms, the intrinsic dimensions of deep neural networks are usually controlled very well, which avoids the curse of dimensionality; (3) Our adopted generative models allow us to efficiently sample from the optimal transport plan.

This is very convenient for certain downstream applications such as domain adaptation, where we can generate infinitely many data points paired across domains BID35 .Moreover, the proposed framework can also recover the density of entropy regularized optimal transport plan.

Specifically, we adopt the neural Ordinary Differential Equation (ODE) approach in to model the dynamics that how Z gradually evolves to G(Z).

We then derive the ODE that describes how the density evolves, and solve the density of the transport plan from the ODE.

The recovery of density requires no extra parameters, and can be evaluated efficiently.

Notations:

Given a matrix A ∈ R d×d , det(A) denotes its determinant, tr(A) = i A ii denotes its trace, A F = i,j A 2 ij denotes its Frobenius norm, and |A| denotes a matrix with [|A|] ij = |A ij |.

We use dim(v) to denote the dimension of a vector v.

We review some background knowledge on optimal transport and implicit generative learning.

Optimal Transport: The idea of optimal transport (OT) originally comes from Monge (1781), which proposes to solve the following problem, DISPLAYFORM0 where T (·) is a mapping from the space of µ to the space of ν.

The optimal mapping T * is referred to as Monge map, and equation 4 is referred to as Monge formulation of optimal transport.

Monge formulation, however, is not necessarily feasible.

For example, when X is a constant random variable and Y is not, there does not exist such a map T satisfying T (X) ∼ ν.

The Kantorovich formulation of our interest in equation 1 is essentially a relaxation of equation 4 by replacing the deterministic mapping with the coupling between µ and ν.

Consequently, Kantorovich formulation is guaranteed to be feasible and becomes the classical formulation of optimal transport in existing literature BID1 BID6 BID16 BID50 .

Implicit Generative Learning:

For generative learning problems, direct estimation of a probability density function is not always convenient.

For example, we may not have enough prior knowledge to specify an appropriate parametric form of the probability density function (pdf).

Even when an appropriate parametric pdf is available, computing the maximum likelihood estimator (MLE) can be sometimes neither efficient nor scalable.

To address these issues, we resort to implicit generative learning, which do not directly specify the density.

Specifically, we consider that the observed variable X is generated by transforming a latent random variable Z (with some known distribution ρ) through some unknown mapping G(·), i.e., X = G(Z).

We then can train a generative model by estimating G(·) with a properly chosen loss function, which can be easier to compute than MLE.

Existing literature usually refer to the distribution of G(Z) as the push-forward of reference distribution ρ.

Such an implicit generative learning approach also enjoys an additional benefit: We only need to choose ρ that is convenient to sample, e.g., uniform or Gaussian distribution, and we then can generate new samples from our learned distribution directly through the estimated mapping G very efficiently.

For many applications, the target distribution can be quite complicated, in contrast to the distribution ρ being simple.

This actually requires the mapping G to be flexible.

Therefore, we choose to represent G using deep neural networks (DNNs), which are well known for its universal approximation property, i.e., DNNs with sufficiently many neurons and properly chosen activation functions can approximate any continuous functions over compact support up to an arbitrary error.

Early empirical evidence, including variational auto-encoder (VAE, Kingma & Welling (2013) ) and generative adversarial networks (GAN, BID21 ) have shown great success of parameterizing G with DNNs.

They further motivate a series of variants, which adopt various DNN architectures to learn more complicated generative models BID45 BID5 BID54 BID10 BID28 .Although the above methods cannot directly estimate the density of the target distribution, for certain applications, we can actually recover the density of G(Z).

For example, generative flow methods such as NICE BID13 , Real NVP BID14 Glow (Kingma & BID31 ) impose sparsity constraints on weight matrices, and exploit the hierarchical nature of DNNs to compute the densities layer by layer.

Specifically, NICE proposed in BID13 denotes the transitions of densities within a neural network as DISPLAYFORM1 , where h i represents the hidden units of the i-th layer and f i is the transition function.

NICE suggest to restrict the Jacobian matrices of f i 's to be triangular.

Therefore, f i 's are reversible and the transition of density in each layer can be easily computed.

More recently, propose a neural ordinary differential equation (neural ODE) approach to compute the transition from Z to G(Z).

Specifically, they introduce a dynamical formulation and parameterizing the mapping G using DNNs with recursive structures: They use an ODE to describe how the input Z gradually evolves towards the output G(Z) in continuous time, dz/dt = ξ(z(t), t), where z(t) denotes the continuous time interpolation of Z, and ξ(·, ·) denotes a feedforward-type DNN.

Without loss of generality, we choose z(0) = Z and z(1) = G(Z).

Then under certain regularity conditions, the mapping G(·) is guaranteed to be reversible, and the density of G(Z) can be computed in O(d) time, where d is the dimension of Z BID22 .

For better efficiency and scalability, we propose a new framework -named SPOT (Scalable Pushforward of Optimal Transport) -for solving the optimal transport problem.

Before we proceed with the derivation, we first introduce some notations and assumptions.

Recall that we aim to find an optimal joint distribution γ given by equation 1.

For simplicity, we assume that the two marginal distributions X ∼ µ and Y ∼ ν have densities p X (x) and p Y (y) for X ∈ X and Y ∈ Y with compact X and Y, respectively.

Moreover, we assume that the joint distribution γ has density p γ .

Then we rewrite equation 1 as the following integral form, DISPLAYFORM0 We then convert equation 5 into a minmax optimization problem using the Lagrangian multiplier method.

Note that equation 5 contains infinitely many constraints, i.e., the equality constraints need to hold for every x ∈ X and y ∈ Y.

Therefore, we need infinitely many Lagrangian multipliers.

For notational simplicity, we denote the Lagrangian multipliers associated with x and y by two functions λ X (x) : X → R and λ Y (y) : Y → R, respectively.

Eventually we obtain DISPLAYFORM1 As mentioned earlier, solving p γ in the space of all continuous distributions is generally intractable.

Thus, we adopt the push-forward method, which introduces a mapping G from some latent variable DISPLAYFORM2 The latent variable Z follows some distribution ρ that is easy to sample.

We then rewrite equation 6 as min DISPLAYFORM3 (7) Note that we have replaced the integrals with expectations, since x∈X p γ (x, y)dx, y∈Y p γ (x, y)dy, p X (x), and p Y (y) are probability density functions.

Then we further parameterize G, λ X , and λ Y with neural networks 2 .

We denote G as the class of neural networks for parameterizing G and similarly F X and F Y as the classes of functions for λ X and λ Y , respectively.

Although G, F X , and F Y are finite classes, our parameterization of G cannot exactly represent any continuous distributions of (X, Y ) (only up to a small approximation error with sufficiently many neurons).

Then the marginal distribution constraints, G X (Z) ∼ µ and G Y (Z) ∼ ν, are not necessarily satisfied.

Therefore, the equilibrium of equation 7 does not necessarily exist, since the Lagrangian multipliers can be unbounded.

Motivated by BID0 , we require the neural networks for parameterizing λ X and λ Y to be η-Lipschitz, denoting as F η X and F η Y , respectively.

Here η can be treated as a tuning parameter, and provides a refined control of the constraint violation.

Since each η-Lipschitz function can be represented by ηf with f being 1-Lipschitz, we rewrite equation 7 as min DISPLAYFORM4 We apply alternating stochastic gradient algorithm to solve equation 8: in each iteration, we perform a few steps of gradient ascent on λ X and λ Y , respectively for a fixed G, followed by one-step gradient descent on G for fixed λ X and λ Y .

We use Spectral Normalization (SN, BID38 ) to control the Lipschitz constant of λ X and λ Y being smaller than 1.

Specifically, SN constrains the spectral norm of each weight matrix W by SN(W ) = W/σ(W ) in every iteration, where σ(W ) denotes the spectral norm of W .

Note that σ(W ) can be efficiently approximated by a simple one-step power method BID20 .

Therefore, the computationally intensive SVD can be avoided.

We summarize the algorithm in Algorithm 1 with SN omitted.

Algorithm 1 Mini-batch Primal Dual Stochastic Gradient Algorithm for SPOT Require: DISPLAYFORM5 Initialized networks G, λ X , and λ Y with parameters w, θ, and β, respectively; α, the learning rate; n critic , the number of gradient ascent for λ X and λ Y ; n, the batch size while w not converged do DISPLAYFORM6 Connection to Wasserstein Generative Adversarial Networks (WGANs): Our proposed framework equation 8 can be viewed as a multi-task learning version of Wasserstein GANs BID35 .

Specifically, the mapping G can be viewed as a generator that generates samples in the domains X and Y. The Lagrangian multipliers λ X and λ Y can be viewed as discriminators that evaluate the discrepancies of the generated sample distributions and the target marginal distributions.

By restricting DISPLAYFORM7 essentially approximates the Wasserstein distance between the distributions of G X (Z) and X under the Euclidean ground cost BID51 , the same holds for Y ).

Denote DISPLAYFORM8 which essentially learns two Wasserstein GANs with a joint generator G through the regularizer R. Extension to Multiple Marginal Distributions: Our proposed framework can be straightforwardly extended to more than two marginal distributions.

Consider the ground cost function c taking m inputs X 1 , . . .

, X m with X i ∼ µ i for i = 1, . . .

, m. Then the optimal transport problem equation 1 becomes the multi-marginal problem BID42 : DISPLAYFORM9 where Π(µ 1 , µ 2 , · · · , µ m ) denotes all the joint distributions with marginal distributions satisfying X i ∼ µ i for all i = 1, . . . , m. Following the same procedure for two distributions, we cast equation 10 into the following form min DISPLAYFORM10 , where G and λ Xi 's are all parameterized by neural networks.

Existing methods for solving the multi-marginal problem equation 10 suggest to discretize the support of the joint distribution using a refined grid.

For complex distributions, the grid size needs to be very large and can be exponential in m BID51 .

Our parameterization method actually only requires at most 2m neural networks, which further corroborates the scalability and efficiency of our framework.

Existing literature has shown that entropy-regularized optimal transportation outperforms the unregularized counterpart in some applications BID15 BID9 .

This is because the entropy regularizer can tradeoff the estimation bias and variance by controlling the smoothness of the density function.

We demonstrate how to efficiently recover the density p γ of the transport plan with entropy regularization.

Instead of parameterizing G by a feedforward neural network, we choose the neural ODE approach, which uses neural networks to approximate the transition from input Z towards output G(Z) in the continuous time.

Specifically, we take z(0) = Z and z(1) = G(Z).

Let z(t) be the continuous interpolation of Z with density p(t) varying according to time t. We split z(t) into z 1 (t) and z 2 (t) such that dim(z 1 ) = dim(X) and dim(z 2 ) = dim(Y ).

We then write the neural ODE as DISPLAYFORM0 where ξ 1 and ξ 2 capture the dynamics of z(t).

We parameterize ξ = (ξ 1 , ξ 2 ) by a neural network with parameter w.

We describe the dynamics of the joint density p(t) in the following proposition.

Proposition 1.

Let z, z 1 , z 2 , ξ 1 and ξ 2 be defined as above.

Suppose ξ 1 and ξ 2 are uniformly Lipschitz continuous in z (the Lipschitz constant is independent of t) and continuous in t. The log joint density satisfies the following ODE: DISPLAYFORM1 where ∂ξ1 ∂z1 and ∂ξ2 ∂z2 are Jacobian matrices of ξ 1 and ξ 2 with respect to z 1 and z 2 , respectively.

Proposition 1 is a direct result of Theorem 1 in .

We can now recover the joint density by taking p γ = p(1), which further enables us to efficiently compute the entropy regularizer defined as DISPLAYFORM2 Then we consider the entropy regularized Wasserstein distance DISPLAYFORM3 is the objective function in equation 8.

Note that here G is a functional operator of ξ, and hence parameterized with w. The training algorithm follows Algorithm 1, except that updating G becomes more complex due to involving the neural ODE and the entropy regularizer.

To update G, we are essentially updating w using the gradient g w = ∂(L c + H)/∂w, where is the regularization coefficient.

First we compute ∂L c /∂w.

We adopt the integral form from in the following DISPLAYFORM4 where a(t) = ∂L c /∂z(t) is the so-called "adjoint variable".

The detailed derivation is slightly involved due to the complicated terms in the chain rule.

We refer the readers to for a complete argument.

The advantage of introducing a(t) is that we can compute a(t) using the following ODE, DISPLAYFORM5 Then we can use a well developed numerical method to compute equation 13 efficiently BID12 .

Next, we compute ∂H/∂w in a similar procedure with a(t) replaced by b(t) = ∂H/∂ log p(t).

We then write DISPLAYFORM6 Using the same numerical method, we can compute ∂H/∂w, which eventually allows us to compute g w and update w.

We evaluate the SPOT framework on various tasks: Wasserstein distance approximation, density recovery, paired sample generation and domain adaptation.

All experiments are implemented with PyTorch using one GTX1080Ti GPU and a Linux desktop computer with 32GB memory, and we adopt the Adam optimizer with configuration parameters 0.5 and 0.999 (Kingma & Ba, 2014).

We first demonstrate that SPOT can accurately and efficiently approximate the Wasserstein distance.

We take the Euclidean ground cost, i.e. c(x, y) = x − y .

Then DISPLAYFORM0 essentially approximates the Wasserstein distance.

We take the marginal distributions µ and ν as two Gaussian distributions in R 2 with the same identity covariance matrix.

The means are (−2.5, 0) and (2.5, 0) , respectively.

We find the Wasserstein distance between µ and ν equal to 5 by evaluating its closed-form solution.

We generate n = 10 5 samples from both distributions µ and ν, respectively.

Note that naively applying discretization-based algorithms by dividing the support according to samples requires at least 40 GB memory, which is beyond the memory capability.

We parameterize G X , G Y , λ X , and λ Y with fully connected neural networks without sharing parameters.

All the networks use the Leaky-ReLU activation BID37 .

G X and G Y have 2 hidden layers.

λ X and λ Y have 1 hidden layer.

The latent variable Z follows the standard Gaussian distribution in R 2 .

We take the batch size equal to 100.WD vs. Number of Epochs.

We compare the algorithmic behavior of SPOT and Regularized Optimal Transport (ROT, BID47 ) with different regularization coefficients.

For SPOT, we set the number of units in each hidden layer equal to 8 and η = 10 4 .

For ROT, we adopt the code from the authors 3 with only different input samples, learning rates, and regularization coefficients.

FIG0 shows the convergence behavior of SPOT and ROT for approximating the Wasserstein distance between µ and ν with different learning rates.

We observe that SPOT converges to the true Wasserstein distance with only 0.6%, 0.3%, and 0.3% relative errors corresponding to Learning Rates (LR) 10 −3 , 10 −4 , and 10 −5 , respectively.

In contrast, ROT is very sensitive to its regularization coefficient.

Thus, it requires extensive tuning to achieve a good performance.

WD vs. Number of Hidden Units.

We then explore the adaptivity of SPOT by increasing the network size, while the input data are generated from some low dimensional distribution.

Specifically, the number of hidden units per layer varies from 2 to 2 10 .

Recall that we parameterize G with two 2-hidden-layer neural networks, and λ X , λ Y with two 1-hidden-layer neural networks.

Accordingly, the number of parameters in G varies from 36 to about 2 × 10 6 , and that in λ X or λ Y varies from 12 to about 2, 000.

The tuning parameter η also varies corresponding to the number of hidden units in λ X , λ Y .

We use η = 10 5 for 2 1 , 2 2 and 2 3 hidden units per layer, η = 2×10 4 for 2 4 , 2 5 and 2 6 hidden units per layer, η = 10 4 for 2 7 and 2 8 hidden units per layer, η = 2 × 10 3 for 2 9 , and 2 10 hidden units per layer.

FIG1 shows the estimated WD with respect to the number of hidden units per layer.

For large neural networks that have 2 9 or 2 10 hidden units per layer, i.e., 5.2 × 10 5 or 2.0 × 10 6 parameters, the number of parameters is far larger than the number of samples.

Therefore, the model is heavily overparameterized.

As we can observe in FIG1 , the relative error however, does not increase as the number of parameters grows.

This suggests that SPOT is robust with respect to the network size.

We demonstrate that SPOT can effectively recover the joint density with entropy regularization.

We adopt the neural ODE approach as described in Section 4.

Denote φ(a, b) as the density N (a, b) .

We take the marginal distributions µ and ν as (1) Gaussian distributions φ(0, 1) and φ(2, 0.5); (2) mixtures of Gaussian 1 2 φ(−1, 0.5) + 1 2 φ(1, 0.5) and 1 2 φ(−2, 0.5)+ 1 2 φ(2, 0.5).

The ground cost is the Euclidean square function, i.e., c(x, y) = x−y 2 .

We run the training algorithm for 6 × 10 5 iterations and in each iteration, we generate 500 samples from µ and ν, respectively.

We parameterize ξ with a 3-hidden-layer fully-connected neural network with 64 hidden units per layer, and the latent dimension is 2.

We take η = 10 6 .Figure 4: Visualization of the marginal distributions and the joint density of the optimal transport plan.

Figure 4 shows the input marginal densities and heat maps of output joint densities.

We can see that a larger regularization coefficient yields a smoother joint density for the optimal transport plan.

Note that with continuous marginal distributions and the Euclidean square ground cost, the joint density of the unregularized optimal transport degenerates to a generalized impulse function (i.e., a generalized Dirac δ function that has nonzero value on a manifold instead of one atom, as shown in Rachev FORMULA0 ; Onural FORMULA1 ).

Entropy regularization prevents such degeneracy by enforcing smoothness of the density.

We show that SPOT can generate paired samples (G X (Z), G Y (Z)) from unpaired data X and Y that are sampled from marginal distributions µ and ν, respectively.

Synthetic Data.

We take the squared Euclidean cost, i.e. c(x, y) = x−y 2 , and adopt the same implementation and sample size as in Section 5.1 with learning rate 10 −3 and 32 hidden units per layer.

FIG3 illustrates the input samples and the generated samples with two sets of different marginal distributions: The upper row corresponds to the same Gaussian distributions as in Section 5.1.

The lower row takes X as Gaussian distribution with mean (−2.5, 0) and covariance 0.5I, Y as (sin(Y 1 ) + Y 2 , 2Y 1 − 3) , where Y 1 follows a uniform distribution on [0, 3] , and Y 2 follows a Gaussian distribution N (2, 0.1).

We observe that the generated samples and the input samples are approximately identically distributed.

Additionally, the paired relationship is as expected -the upper mass is transported to the upper region, and the lower mass is transported to the lower region.

Real Data.

We next show SPOT is able to generate high quality paired samples from two unpaired real datasets: MNIST (LeCun et al., 1998) and MNISTM BID18 .

The handwritten digits in MNIST and MNISTM datasets have different backgrounds and foregrounds (see FIG2 .

The digits in paired images however, are expected to have similar contours.

We leverage this prior knowledge 4 by adopting a semantic-aware cost function BID34 to extract the edge of handwritten letters, i.e., we use the following cost function DISPLAYFORM0 where C 1 and C 2 denote the Sobel filter BID49 , and x j 's and y j 's are the three channels of RGB images.

The operator * denotes the matrix convolution.

We set We now use separate neural networks to parameterize G X and G Y instead of taking G X and G Y as outputs of a common network.

Note that G X and G Y does not share parameters.

Specifically, we use two 4-layer convolutional layers in each neural network for G X or G Y , and two 5-layer convolutional neural networks for λ X and λ Y .

More detailed network settings are provided in Appendix A.2.

The batch size is 32, and we train the framework with 2 × 10 5 iterations until the generated samples become stable.

FIG2 shows the generated samples of SPOT.

We also reproduce the results of CoGAN with the code from the authors 5 .

As can be seen, with approximately the same network size, SPOT yields paired images with better quality than CoGAN: The contours of the paired results of SPOT are nearly identical, while the results of CoGAN have no clear paired relation.

Besides, the images corresponding to G Y (Z) in SPOT have colorful foreground and background, while in CoGAN there are only few colors.

Recall that in SPOT, the paired relation is encouraged by ground cost c, and in CoGAN it is encouraged by sharing parameters.

By leveraging prior knowledge in ground cost c, the paired relation is more accurately controlled without compromising the quality of the generated images.

DISPLAYFORM1 We further test our framework on more complex real datasets: Photo-Monet dataset and Edges-Shoes dataset .

We adopt the Euclidean cost function for Photo-Monet dataset, and the semantic-aware cost function as in MNIST-MNISTM for Edges-Shoes dataset.

Other implementations remain the same as the MNIST-MINSTM experiment.

FIG5 demonstrates the generated samples of both datasets.

We observe that the generated images have a desired paired relation: For each Z, G X (Z) and G Y (Z) gives a pair of corresponding scenery and shoe.

The generated images are also of high quality, especially considering that Photo-Monet dataset is a pretty small but complex dataset with 6,288 photos and 1,073 paintings.

Optimal transport has been used in domain adaptation, but existing methods are either computationally inefficient BID7 , or cannot achieve a state-of-the-art performance BID48 .

Here, we demonstrate that SPOT can tackle large scale domain adaptation problems with state-of-the-art performance.

In particular, we receive labeled source data {x i } ∼ µ, where each data point is associated with a label v i , and target data {y j } ∼ ν with unknown labels.

For simplicity, we use X and Y to denote the random vectors following distributions µ and ν, respectively.

The two distributions µ and ν can be coupled in a way that each paired samples of (X, Y ) from the coupled joint distribution are likely to have the same label.

In order to identify such coupling information between source and target data, we propose a new OT-based domain adaptation method -DASPOT (Domain Adaptation with SPOT) as follows.

Specifically, we jointly train an optimal transport plan and two classifiers for X and Y (denoted by D X and D Y , respectively).

Each classifier is a composition of two neural networks -an embedding network and a decision network.

For simplicity, we denote D X = D e,X • D c,X , where D e,X denotes the embedding network, and D c,X denotes the decision network (respectively for D Y = D e,Y • D c,Y ).

We expect the embedding networks to extract high level features of the source and target data, and then find an optimal transport plan to align X and Y based on these high level features using SPOT.

Here we choose a ground cost c(x, y) = D e,X (x) − D e,Y (y) 2 .

Let G denote the generator of SPOT.

The Wasserstein distance of such an OT problem can be written as DISPLAYFORM0 Meanwhile, we train D X by minimizing the empirical risk DISPLAYFORM1 , where E denotes the cross entropy loss function, and train D Y by minimizing DISPLAYFORM2 where [ Eventually, the joint training optimize DISPLAYFORM3 where DISPLAYFORM4 is the objective function of OT problem in equation 8 with c defined above, and η s , η da are the tuning parameters.

We choose η s = 10 3 for all experiments.

We set η da = 0 for the first 10 5 iteration to wait the generators to be well trained.

Then we set η da = 10 for the next 3 × 10 5 iteration.

We take totally 4 × 10 5 iterations, and set the learning rate equal to 10 −4 and batch size equal to 128 for all experiments.

We evaluate DASPOT with the MNIST, MNISTM, USPS BID26 , and SVHN BID40 datasets.

We denote a domain adaptation task as Source Domain → Target Domain.

We compare the performance of DASPOT with other optimal transport based domain adaptation methods: ROT BID48 , StochJDOT (Damodaran et al., 2018) and DeepJDOT (Damodaran et al., 2018) .

As can be seen in TAB0 , DASPOT achieves equal or better performances on all the tasks.

Moreover, we show that DeepJDOT is not as efficient as DASPOT.

For example, in the MNIST → USPS task, DASPOT requires 169s running time to achieve a 95% accuracy, while DeepJDOT requires 518s running time to achieve the same accuracy.

The reason behind is that DeepJDOT needs to solve a series of optimal transport problems using Sinkhorn algorithm.

The implementation of DeepJDOT is adapted from the authors' code 6 .

Existing literature shows that several stochastic algorithms can efficiently compute the Wasserstein distance between two continuous distributions.

These algorithms, however, only apply to the dual of the OT problem equation 1, and cannot provide the optimal transport plan.

For example, BID19 suggest to expand the dual variables in two reproducing kernel Hilbert spaces.

They then apply the Stochastic Averaged Gradient (SAG) algorithm to compute the optimal objective value of OT with continuous marginal distributions or semi-discrete marginal distributions (i.e., one marginal distribution is continuous and the other is discrete).

The follow-up work, BID47 , parameterize the dual variables with neural networks and apply the Stochastic Gradient Descent (SGD) algorithm to eventually achieve a better convergence.

These two methods can only provide the optimal transport plan and recover the joint density when the densities of the marginal distributions are known.

This is prohibitive in most applications, since we only have access to the empirical data.

Our framework actually allows us to efficiently compute the joint density from the transformation of the latent variable Z as in Section 4.

TAB1 shows the architecture of two discriminators λ X , λ Y .

The two networks have identical architechture and do not share parameters.

The CNN architecture for USPS, MNIST and MNISTM.

PReLU activation is applied BID24 .

TAB2 shows the architecture of two generators G X and G Y .

The last column in TAB2 means whether G X and G Y share the same parameter.

TAB3 shows the architecture of two discriminators λ X , λ Y , and two classifiers D X , D Y .

The last column in TAB2 uses (·, ·) to denote which group of discriminators share the same parameter.

TAB4 shows the architecture of two generators G X and G Y .

The last column in TAB4 means whether G X and G Y share the same parameter.

The Residual block is the same as the one in BID38 . [3 × 3, ch, stride = 1, padding =0]

Sigmoid False TAB5 shows the architecture of two discriminators λ X , λ Y , and two classifiers D X , D Y .

The last column in TAB5 uses (·, ·) to denote which group of discriminators share the same parameter.

<|TLDR|>

@highlight

Use GAN-based method to scalably solve optimal transport