We provide a novel perspective on the forward pass through a block of layers in a deep network.

In particular, we show that a forward pass through a standard dropout layer followed by a linear layer and a non-linear activation is equivalent to optimizing a convex objective with a single iteration of a $\tau$-nice Proximal Stochastic Gradient method.

We further show that replacing standard Bernoulli dropout with additive dropout is equivalent to optimizing the same convex objective with a variance-reduced proximal method.

By expressing both fully-connected and convolutional layers as special cases of a high-order tensor product, we unify the underlying convex optimization problem in the tensor setting and derive a formula for the Lipschitz constant $L$ used to determine the optimal step size of the above proximal methods.

We conduct experiments with standard convolutional networks applied to the CIFAR-10 and CIFAR-100 datasets and show that replacing a block of layers with multiple iterations of the corresponding solver, with step size set via $L$, consistently improves classification accuracy.

Deep learning has revolutionized computer vision and natural language processing and is increasingly applied throughout science and engineering BID20 .

This has motivated the mathematical analysis of various aspects of deep networks, such as the capacity and uniqueness of their representations BID28 BID24 and their global training convergence properties BID10 .

However, a complete characterization of deep networks remains elusive.

For example, Bernoulli dropout layers are known to improve generalization BID29 , but a thorough theoretical understanding of their behavior remains an open problem.

While basic dropout layers have proven to be effective, there are many other types of dropout with various desirable properties BID22 .

This raises many questions.

Can the fundamental block of layers that consists of a dropout layer followed by a linear transformation and a non-linear activation be further improved for better generalization?

Can the choice of dropout layer be made independently from the linear transformation and non-linear activation?

Are there systematic ways to propose new types of dropout?We attempt to address some of these questions by establishing a strong connection between the forward pass through a block of layers in a deep network and the solution of convex optimization problems of the following form: DISPLAYFORM0 Note that when f i (a i x) = 1 2 (a i x − y i ) 2 and g(x) = x 2 2 , Eq. (1) is standard ridge regression.

When g(x) = x 1 , Eq. (1) has the form of LASSO regression.

We show that a block of layers that consists of dropout followed by a linear transformation (fullyconnected or convolutional) and a non-linear activation has close connections to applying stochastic solvers to (1).

Interestingly, the choice of the stochastic optimization algorithm gives rise to commonly used dropout layers, such as Bernoulli and additive dropout, and to a family of other types of dropout layers that have not been explored before.

As a special case, when the block in question does not include dropout, the stochastic algorithm reduces to a deterministic one.

Our contributions can be summarized as follows.

(i) We show that a forward pass through a block that consists of Bernoulli dropout followed by a linear transformation and a non-linear activation is equivalent to a single iteration of τ -nice Proximal Stochastic Gradient, Prox-SG BID34 when it is applied to an instance of (1).

We provide various conditions on g that recover (either exactly or approximately) common non-linearities used in practice. (ii) We show that the same block with an additive dropout instead of Bernoulli dropout is equivalent to a single iteration of mS2GD BID16 ) -a mini-batching form of variance-reduced SGD BID12 ) -applied to an instance of (1). (iii) By expressing both fully-connected and convolutional layers (referred to as linear throughout) as special cases of a high-order tensor product BID2 , we derive a formula for the Lipschitz constant L of ∇F (x).

As a consequence, we can compute the optimal step size for the stochastic solvers that correspond to blocks of layers.

We note that concurrent work BID26 used a different analysis strategy to derive an equivalent result for computing the singular values of convolutional layers. (iv) We validate our theoretical analysis experimentally by replacing blocks of layers in standard image classification networks with corresponding solvers and show that this improves the accuracy of the models.

Optimization algorithms can provide insight and guidance in the design of deep network architectures BID31 BID14 BID35 BID36 .

For example, BID35 have proposed a deep network architecture for compressed sensing.

Their network, dubbed ADMM-Net, is inspired by ADMM updates BID4 on the compressed sensing objective.

Similarly, BID36 demonstrated that unrolling a proximal gradient descent solver BID1 ) on the same problem can further improve performance.

The work of BID14 demonstrated a relation between incremental proximal methods and ResNet blocks; based on this observation, they proposed a new architecture (variational networks) for the task of image reconstruction.

BID0 proposed to embed optimization problems, in particular linearly-constrained quadratic programs, as structured layers in deep networks.

BID21 replaced proximal operators in optimization algorithms by neural networks.

BID11 proposed a new matrix layer, dubbed ReEig, that applies a thresholding operation to the eigenvalues of intermediate feature representations that are stacked in matrix form.

ReEig can be tightly connected to a proximal operator of the set of positive semi-definite matrices.

proposed a new architecture based on a sparse representation construct, Multi-Layer Convolutional Sparse Coding (ML-CSC), initially introduced by BID23 .

Sparsity on the intermediate representations was enforced by a multi-layer form of basis pursuit.

This body of work has demonstrated the merits of connecting the design of deep networks with optimization algorithms in the form of structured layers.

Yet, with few exceptions BID0 , previous works propose specialized architectures for specific tasks.

Our work aims to contribute to a unified framework that relates optimization algorithms to deep layers.

A line of work aims to provide rigorous interpretation for dropout layers.

For example, BID32 showed that dropout is linked to an adaptively balanced 2 -regularized loss.

BID33 showed that approximating the loss with a normal distribution leads to a faster form of dropout.

BID7 BID38 developed a framework that connects dropout with approximate variational inference in Bayesian models.

We provide a complementary perspective, in which dropout layers arise naturally in an optimization-driven framework for network design.

This section is organized as follows.

We introduce our notation and preliminaries in Section 3.1.

In Section 3.2, we present a motivational example relating a single iteration of proximal gradient descent (Prox-GD) on (1) to the forward pass through a fully-connected layer followed by a nonlinear activation.

We will show that several commonly used non-linear activations can be exactly or approximately represented as proximal operators of g(x).

In Section 3.3, we unify fully-connected and convolutional layers as special cases of a high-order tensor product.

We propose a generic instance of (1) in a tensor setting, where we provide a formula for the Lipschitz constant L of the finite sum structure of (1).

In Section 3.4, we derive an intimate relation between stochastic solvers, namely τ -nice Prox-SG and mS2GD, and two types of dropout layers.

FIG6 shows an overview of the connections that will be developed.

FIG6 : An overview of the tight relation between a single iteration of a stochastic solver and the forward pass through the l th layer in a network that consists of dropout followed by a linear transformation and a non-linear activation.

We study an instance of problem (1) with quadratic F (x), where x l−1 are the input activations and x l , the variables being optimized, correspond to the output activations.

Varying the type of stochastic solver changes the nature of the dropout layer, while the prior g(x) on the output activations determines the non-linearity Prox 1 L g (.).

As we will be working with tensors, we will follow the tensor notation of BID15 .

The order of a tensor is the number of its dimensions.

In particular, scalars are tensors of order zero, vectors are tensors of order one, and matrices are tensors of order two.

We denote scalars by lowercase letters a, vectors by bold lowercase letters a, and matrices by bold capital letters A. We use subscripts a i to refer to individual elements in a vector.

Tensors of order three or more will be denoted by cursive capital letters A ∈ R J1×J2×···×Jn .

Throughout the paper, we will handle tensors that are of at most order four.

High-order tensors with a second dimension of size equal to one are traditionally called vector tensors and denoted A ∈ R J1×1×J3×J4 .

We use A(i, j, k, z) to refer to an element in a tensor and A(i, j, k, :) to refer to a slice of a tensor.

The inner product between tensors of the same size is denoted A, B = i1,...,i N A (i 1 , . . .

, i N ) B (i 1 , . . .

, i N ).

The squared Frobenius norm of a tensor A is defined as A 2 F = A, A .

Lastly, the superscripts and H are used to denote the transpose and the Hermitian transpose, respectively.

As a motivating example, we consider the l th linear layer in a deep network that is followed by a non-linear activation ρ, i.e. x l = ρ(Ax l−1 + b), where A ∈ R n2×n1 and b ∈ R n2 are the weights and biases of the layer and x l−1 and x l are the input and output activations, respectively.

Now consider an instance of (1) with a convex function g(x) and DISPLAYFORM0 where A (i, :) is the i th row of A .

Such an objective can be optimized iteratively in x l using Prox-GD with the following update equation: DISPLAYFORM1 where the Lipschitz constant L = λ max AA and λ max (.) denotes the maximum eigenvalue.

By initializing the iterative optimization at x l = 0, it becomes clear that a single iteration of (3) is equivalent to a fully-connected layer followed by a non-linearity that is implemented by the proximal operator BID6 .

The choice of g(x) determines the specific form of the non-linearity ρ.

Several popular activation functions can be traced back to their corresponding g(x).

The ReLU, which enforces non-negative output activations, corresponds to the indicator function g(x) = 1 x≥0 ; the corresponding instance of problem (1) is a non-negative quadratic program.

Similar observations for the ReLU have been made in other contexts BID0 BID23 .

We observe that many other activation functions fit this framework.

For example, when g(x) is a squared hinge loss, i.e. DISPLAYFORM2 DISPLAYFORM3 , a single update of FORMULA2 is equivalent to a linear layer followed by a Leaky ReLU.

TAB0 lists some other choices of g(x) and their induced activations.

is not required to exhibit a simple, coordinate-wise separable structure.

More complex functions can be used, as long as the proximal operator is easy to evaluate.

Interesting examples arise when the output activations have matrix structure.

For instance, one can impose nuclear norm regularization g(X) = X * to encourage X to be low rank.

Alternatively, one can enforce positive semi-definite structure on the matrix X by defining g(X) = 1 X 0 .

A similar activation has been used for higher-order pooling BID11 .In what follows, we will show that this connection can be further extended to explain dropout layers.

Interestingly, specific forms of dropout do not arise from particular forms of objective FORMULA0 , but from different stochastic optimization algorithms that are applied to it.

Before presenting our main results on the equivalence between a forward pass through a block of layers and solving (1) with stochastic algorithms, we provide some key lemmas.

These lemmas will be necessary for a unified treatment of fully-connected and convolutional layers as generic linear layers.

This generic treatment will enable efficient computation of the Lipschitz constant for both fully-connected and convolutional layers.

Lemma 1.

Consider the l th convolutional layer in a deep network with some non-linear activation, e.g. Prox g (.), where the weights A ∈ R n2×n1×W ×H , biases B ∈ R n2×1×W ×H , and input activations X l−1 ∈ R n1×1×W ×H are stacked into 4 th -order tensors.

We can describe the layer as DISPLAYFORM0 where HO is the high-order tensor product.

Here n 1 is the number of input features, n 2 is the number of output features (number of filters), and W and H are the spatial dimensions of the features.

As a special case, a fully-connected layer follows naturally, since HO reduces to a matrix-vector multiplication when W = H = 1.The proof can be found in supplementary material.

Note that the order of the dimensions is essential in this notation, as the first dimension in A corresponds to the number of independent filters while the second corresponds to the input features that will be aggregated after the 2D convolutions.

Also note that according to the definition of HO in BID2 , the spatial size of the filters in A, namely W and H, has to match the spatial dimensions of the input activations X l−1 , since the operator HO performs 2D circular convolutions while convolutions in deep networks are 2D linear convolutions.

This is not a restriction, since one can perform linear convolution through a zero-padded circular convolution.

Lastly, we assume that the values in B are replicated along the spatial dimensions W and H in order to recover the behaviour of biases in deep networks.

Given this notation, we will refer to either a fully-connected or a convolutional layer as a linear layer throughout the rest of the paper.

Since we are interested in a generic linear layer followed by a non-linearity, we will consider the tensor quadratic version of F (x), denoted F ( X ): DISPLAYFORM1 Note that if A ∈ R n2×n1×W ×H , then A H ∈ R n1×n2×W ×H , where each of the frontal slices of A(:, :, i, j) is transposed and each filter, A(i, j, :, :), is rotated by 180• .

This means that A H HO X aggregates the n 2 filters after performing 2D correlations.

This is performed n 1 times independently.

This operation is commonly referred to as a transposed convolution.

Details can be found in supplementary material.

Next, the following lemma provides a practical formula for the computation of the Lipschitz constant L of the finite sum part of FORMULA6 : DISPLAYFORM2 whereÂ is the 2D discrete Fourier transform along the spatial dimensions W and H.The proof can be found in supplementary material.

Lemma 2 states that the Lipschitz constant L is the maximum among the set of maximum eigenvalues of all the possible W × H combinations of the outer product of frontal slicesÂ (:, :, i, j)Â H (:, :, i, j).

Note that if W = H = 1, thenÂ = A ∈ R n2×n1 since the 2D discrete Fourier transform of scalars (i.e. matrices of size 1 × 1) is an identity mapping.

As a consequence, we can simplify (6) to L = max i=j=1 {λ max A(:, :, i, j)A H (:, :, i, j) } = λ max AA , which recovers the Lipschitz constant for fully-connected layers.

In this subsection, we present two propositions.

The first shows the relation between standard Bernoulli dropout (p is the dropout rate), BerDropout p BID29 , and τ -nice Prox-SG.

The second proposition relates additive dropout, AddDropout, to mS2GD BID16 .

We will first introduce a generic notion of sampling from a set.

This is essential as the stochastic algorithms sample unbiased function estimates from the set of n 1 functions in (5).

Definition 3.1.

BID9 .

A sampling is a random set-valued mapping with values being the subsets of [n 1 ] = {1, . . .

, n 1 }.

A sampling S is τ -nice if it is uniform, i.e. Prob (i ∈ S) = Prob (j ∈ S) ∀ i, j, and assigns equal probabilities to all subsets of [n 1 ] of cardinality τ and zero probability to all others.

Various other types of sampling can be found in BID9 .

We are now ready to present our first proposition.

Proposition 1.

A single iteration of Prox-SG with τ -nice sampling S on (5) with τ = (1 − p)n 1 , zero initialization, and unit step size can be shown to exhibit the update DISPLAYFORM0 which is equivalent to a forward pass through a BerDropout p layer that drops exactly n 1 p input activations followed by a linear layer and a non-linear activation.

We provide a simplified sketch for fully-connected layers here.

The detailed proof is in the supplement.

To see how (7) reduces to the functional form of BerDropout p followed by a fully-connected layer and a non-linear activation, consider W = H = 1.

The argument of Prox 1 L g in (7) (without the bias term) reduces to n 1 τ i∈S A(:, i, :, : DISPLAYFORM1 The first equality follows from the definition of HO , while the second equality follows from trivially reparameterizing the sum, with BerDropout p (.) being equivalent to a mask that zeroes out exactly pn 1 input activations.

Note that if τ -nice Prox-SG was replaced with Prox-GD, i.e. τ = n 1 , then this corresponds to having a BerDropout p layer with dropout rate p = 0; thus, (8) reduces to A BerDropout p ( X l−1 ) = A X l−1 , which recovers our motivating example (3) that relates Prox-GD with the forward pass through a fully-connected layer followed by a non-linearity.

Note that Proposition 1 directly suggests how to apply dropout to convolutional layers.

Specifically, complete input features from n 1 should be dropped and the 2D convolutions should be performed only on the τ -sampled subset, where τ = (1 − p)n 1 .Similarly, the following proposition shows that a form of additive dropout, AddDropout, can be recovered from a different choice of stochastic solver.

Proposition 2.

A single outer-loop iteration of mS2GD BID16 ) with unit step size and zero initialization is equivalent to a forward pass through an AddDropout layer followed by a linear layer and a non-linear activation.

The proof is given in the supplement.

It is similar to Proposition 1, with mS2GD replacing τ -nice Prox-SG.

Note that any variance-reduced algorithm where one full gradient is computed at least once can be used here as a replacement for mS2GD.

For instance, one can show that the serial sampling version of mS2GD, S2GD BID16 , and SVRG BID12 can also be used.

Other algorithms such as Stochastic Coordinate Descent with arbitrary sampling are discussed in the supplement.

A natural question arises as a consequence of our framework: If common layers in deep networks can be understood as a single iteration of an optimization algorithm, what happens if the algorithm is applied for multiple iterations?

We empirically answer this question in our experiments.

In particular, we embed solvers as a replacement to their corresponding blocks of layers and show that this improves the accuracy of the models without an increase in the number of network parameters.

Experimental setup.

We perform experiments on CIFAR-10 and CIFAR-100 BID17 ).

In all experiments, training was conducted on 90% of the training set while 10% was left for validation.

The networks used in the experiments are variants of LeNet (LeCun et al., 1999) , AlexNet BID18 , and VGG16 BID27 .

We used stochastic gradient descent with a momentum of 0.9 and a weight decay of 5 × 10 −4 .

The learning rate was set to (10 −2 , 10 −3 , 10 −4 ) for the first, second, and third 100 epochs, respectively.

For finetuning, the learning rate was initially set to 10 −3 and reduced to 10 −4 after 100 epochs.

Moreover, when a block of layers is replaced with a deterministic solver, i.e. Prox-GD, the step size is set to the optimal constant 1/L, where L is computed according to Lemma 2 and updated every epoch without any zero padding as a circular convolution operator approximates a linear convolution in large dimensions BID38 .

In Prox-SG, a decaying step size is necessary for convergence; therefore, the step size is exponentially decayed as suggested by BID3 , where the initial step size is again set according to Lemma 2.

Finally, to guarantee convergence of the stochastic solvers, we add the strongly convex function λ 2 X 2 F to the finite sum in (5), where we set λ = 10 −3 in all experiments.

Note that for networks that include a stochastic solver, the network will be stochastic at test time.

We thus report the average accuracy and standard deviation over 20 trials.

Replacing fully-connected layers with solvers.

In this experiment, we demonstrate that (i) training networks with solvers replacing one or more blocks of layers can improve accuracy when trained from scratch, and (ii) the improvement is consistently present when one or more blocks are replaced with solvers at different layers in the network.

To do so, we train a variant of LeNet on the CIFAR-10 dataset with two BerDropout p layers.

The last two layers are fully-connected layers with ReLU activation.

We consider three variants of this network: Both fully-connected layers are augmented with BerDropout p (LeNet-D-D), only the last layer is augmented with BerDropout p (LeNet-ND-D), and finally only the penultimate layer is augmented with BerDropout p (LeNet-D-ND) .

In all cases, we set the dropout rate to p = 0.5.

We replace the BerDropout p layers with their corresponding stochastic solvers and run them for 10 iterations with τ = n 1 /2 (the setting corresponding to a dropout rate of p = 0.5).

We train these networks from scratch using the same procedure as the baseline networks.

The results are summarized in Table 2 .

It can be seen that replacing BerDropout p with the corresponding stochastic solver (τ -nice Prox-SG) improves performance significantly, for any choice of layer.

The results indicate that networks that incorporate stochastic solvers can be trained stably and achieve desirable generalization performance.

LeNet-D-ND LeNet-ND-D Baseline 64.39% 71.72% 68.54% Prox-SG 72.86% ± 0.177 75.20% ± 0.205 76.23% ± 0.206 Table 2 : Comparison in accuracy between variants of the LeNet architecture on the CIFAR-10 dataset.

The variants differ in the location (D or ND) and number of BerDropout p layers for both the baseline networks and their stochastic solver counterpart Prox-SG.

Accuracy consistently improves when Prox-SG is used.

Accuracy is reported on the test set.

Convolutional layers and larger networks.

We now demonstrate that solvers can be used to improve larger networks.

We conduct experiments with variants of AlexNet 1 and VGG16 on both CIFAR-10 and CIFAR-100.

We start by training strong baselines for both AlexNet and VGG16, achieving 77.3% and 92.56% test accuracy on CIFAR-10, respectively.

Note that performance on this dataset is nearly saturated.

We then replace the first convolutional layer in AlexNet with the deterministic Prox-GD solver, since this layer is not preceded by a dropout layer.

The results are summarized in Table 3 .

We observe that finetuning the baseline network with the solver leads to an improvement of ≈ 1.2%, without any change in the network's capacity.

A similar improvement is observed on the harder CIFAR-100 dataset.

AlexNet AlexNet-Prox-GD CIFAR-10 77.30% 78.51% CIFAR-100 44.20%45.53% Table 3 : Replacing the first convolutional layer of AlexNet by the deterministic Prox-GD solver yields consistent improvement in test accuracy on CIFAR-10 and CIFAR-100.Results on VGG16 are summarized in Table 4 .

Note that VGG16 has two fully-connected layers, which are preceded by a BerDropout p layer with dropout rate p = 0.5.

We start by replacing only the last layer with Prox-SG with 30 iterations and τ = n 1 /2 (VGG16-Prox-SG-ND-D).

We further replace both fully-connected layers that include BerDropout p with solvers (VGG16-Prox-SG-D-D).

We observe comparable performance for both settings on CIFAR-10.

We conjecture that this might be due to the dataset being close to saturation.

On CIFAR-100, a more pronounced increase in accuracy is observed, where VGG-16-Prox-SG-ND-D outperforms the baseline by about 0.7%.We further replace the stochastic solver with a deterministic solver and leave the dropout layers unchanged.

We denote this setting as VGG16-Prox-GD in Table 4 .

Interestingly, this setting performs the best on CIFAR-10 and comparably to VGG16-Prox-SG-ND-D on CIFAR-100.

CIFAR-10 92.56% 92.44% ± 0.028 92.57% ± 0.029 92.80% CIFAR-100 70.27%70.95% ± 0.042 70.44% ± 0.077 71.10% Table 4 : Experiments with the VGG16 architecture on CIFAR-10 and CIFAR-100.

Accuracy is reported on the test set.

Dropout rate vs. τ -nice sampling.

In this experiment, we demonstrate that the improvement in performance is still consistently present across varying dropout rates.

Since Proposition 1 has established a tight connection between the dropout rate p and the sampling rate τ in (5), we observe that for different choices of dropout rate the baseline performance improves upon replacing a block of layers with a stochastic solver with the corresponding sampling rate τ .

We conduct experiments with 1 AlexNet BID18 was adapted to account for the difference in spatial size of the images in CIFAR-10 and ImageNet BID5 ).

The first convolutional layer has a padding of 5, and all max-pooling layers have a kernel size of 2.

A single fully-connected layer follows at the end.

VGG16 on CIFAR-100.

We train four different baseline models with varying choices of dropout rate p ∈ {0, 0.1, 0.9.0.95} for the last layer.

We then replace this block with a stochastic solver with a sampling rate τ and finetune the network.

Table 5 reports the accuracy of the baselines for varying dropout rates p and compares to the accuracy of the stochastic solver with corresponding τ (Prox-SG).

With a high dropout rate, the performance of the baseline network drops drastically.

When using the stochastic solver, we observe a much more graceful degradation.

For example, with a sampling rate τ that corresponds to an extreme dropout rate of p = 0.95 (i.e. 95% of all input activations are masked out), the baseline network with BerDropout p suffers a 56% reduction in accuracy while the stochastic solver declines by only 5%.

Prox-SG Table 5 : Comparison of the VGG16 architecture trained on CIFAR-100 with varying dropout rates p in the last BerDropout p layer.

We compare the baseline to its stochastic solver counterpart with corresponding sampling rate τ = (1 − p)n 1 .

Accuracy is reported on the test set.

In summary, our experiments show that replacing common layers in deep networks with stochastic solvers can lead to better performance without increasing the number of parameters in the network.

The resulting networks are stable to train and exhibit high accuracy in cases where standard dropout is problematic, such as high dropout rates.

We have presented equivalences between layers in deep networks and stochastic solvers, and have shown that this can be leveraged to improve accuracy.

The presented relationships open many doors for future work.

For instance, our framework shows an intimate relation between a dropout layer and the sampling S from the set [n 1 ] in a stochastic algorithm.

As a consequence, one can borrow theory from the stochastic optimization literature to propose new types of dropout layers.

For example, consider a serial importance sampling strategy with Prox-SG to solve (5) BID37 BID34 , where serial sampling is the sampling that satisfies Prob (i ∈ S, j ∈ S) = 0.

A serial importance sampling S from the set of functions f i ( X ) is the sampling such that Prob DISPLAYFORM0 i.e. each function from the set [n 1 ] is sampled with a probability proportional to the norm of the gradient of the function.

This sampling strategy is the optimal serial sampling S that maximizes the rate of convergence solving (5) BID37 .

From a deep layer perspective, performing Prox-SG with importance sampling for a single iteration is equivalent to a forward pass through the same block of layers with a new dropout layer.

Such a dropout layer will keep each input activation with a non-uniform probability proportional to the norm of the gradient.

This is in contrast to BerDropout p where all input activations are kept with an equal probability 1 − p. Other types of dropout arise when considering non-serial importance sampling where |S| = τ > 1.In summary, we have presented equivalences between stochastic solvers on a particular class of convex optimization problems and a forward pass through a dropout layer followed by a linear layer and a non-linear activation.

Inspired by these equivalences, we have demonstrated empirically on multiple datasets and network architectures that replacing such network blocks with their corresponding stochastic solvers improves the accuracy of the model.

We hope that the presented framework will contribute to a principled understanding of the theory and practice of deep network architectures.

A LEAKY RELU AS A PROXIMAL OPERATOR Proof.

The proximal operator is defined as Prox g (a) = arg min DISPLAYFORM1 Note that the problem is both convex and smooth.

The optimality conditions are given by: DISPLAYFORM2 Since the problem is separable in coordinates, we have: DISPLAYFORM3 The Leaky ReLU is defined as DISPLAYFORM4 which shows that Prox g is a generalized form of the Leaky ReLU with a shift of λ and a slope α = Proof.

The proximal operator is defined as Prox g (a) = arg min DISPLAYFORM5 Note that the function g(x) is elementwise separable, convex, and smooth.

By equating the gradient to zero and taking the positive solution of the resulting quadratic polynomial, we arrive at the closedform solution: DISPLAYFORM6 where denotes elementwise multiplication.

It is easy to see that this operator is close to zero for x i << 0, and close to x i for x i >> 0, with a smooth transition for small |x i |.Note that the function Prox g (a) approximates the activation SoftPlus = log(1 + exp (a)) very well.

An illustrative example is shown in FIG2 .

Proposition 5.

The proximal operator to g(x) = −γ i log(1 − x i x i ) approximates the Tanh non-linearity.

Proof.

To simplify the exposition, we derive the proximal operator for the case γ = 1.

The general case for γ > 0 follows analogously.

The proximal operator is defined as Prox g (a) = arg min DISPLAYFORM0 Note that the logarithm is taken element wise, and the objective is convex and smooth.

By equating the gradient to zero, it can be seen that the optimal solution is a root of a cubic equation: DISPLAYFORM1 which is defined in each coordinate i separately.

DISPLAYFORM2 Since q 2 − p 3 < 0, ∀a i ∈ R, it is guaranteed that all roots are real and distinct.

Consequently, the roots can be described as DISPLAYFORM3 Since g(x) is only defined on x ∈ [−1, 1] d , the root that minimizes (11) has to satisfy DISPLAYFORM4 DISPLAYFORM5 It is straightforward to check that DISPLAYFORM6 By substituting f k into (14) and checking inequality (15) it becomes clear that the root corresponding to k = 2 minimizes (11).

By using trigonometric identities the root corresponding to k = 2 can be further simplified to DISPLAYFORM7 which has the approximate shape of the Tanh activation.

An example of this operator is shown in FIG2 .

The proximal operator corresponding to the Sigmoid activation can be derived in a similar fashion by setting g(x) = −γ log(x) − γ log(1 − x).

The exposition presented in the paper requires some definitions related to tensors and operators on tensors.

We summarize the material here.

In all subsequent definitions, we assume D ∈ R n1×n2×n3×n4 and X ∈ R n2×1×n3×n4 .Definition D.1.

BID2 The t-product between high-order tensors is defined as DISPLAYFORM0 where circ HO (D) ∈ R n1n3n4×n2n3n4 and MatVec HO X ∈ R n2n3n4×1 .The operator circ HO (.) unfolds an input tensor into a structured matrix.

On the other hand, MatVec HO (.) unfolds an input tensor into a vector.

The fold and unfold procedures are detailed in BID2 .

Definition D.2.

BID2 The operator DISPLAYFORM1 DISPLAYFORM2 where bdiag (.) : C n1×n2×n3×n4 → C n1n3n4×n2n3n4 , maps a tensor to a block diagonal matrix of all the frontal faces D(:, :, i, j).

Note that if n 3 = n 4 = 1, bdiag () is an identity mapping.

Moreover if n 1 = n 2 = 1, bdiag (.) is a diagonal matrix.

Due to the structure of the tensor unfold of circ HO (.), the resultant matrix circ HO (D) exhibits the following blockwise diagonalization: DISPLAYFORM3 where F n is the n × n Normalized Discrete Fourier Matrix.

Note thatD has the dimensions n 3 and n 4 replaced with the corresponding 2D Discrete Fourier Transforms.

That isD(i, j, :, :) is the 2D Discrete Fourier Transform of D(i, j, :, :).For more details, the reader is advised to start with third order tensors in the work of BID13 and move to the work of BID2 is equivalent to (i) performing 2D convolutions spatially along the third and fourth dimensions.(ii) It aggregates the result along the feature dimension n 1 . (iii) It repeats the procedure for each of the n 2 filters independently.

We will show the following using direct manipulation of the properties of HO BID2 .

DISPLAYFORM4 Note that Equation FORMULA1 shows that features are aggregated along the n 1 dimension.

Now, by showing that A (:, i, :, :) HO X l−1 (i, :, :, :) performs n 2 independent 2D convolutions along on the i th channel, the Lemma 1 is proven.

For ease of notation, consider two tensors U ∈ R n2×1×W ×H and Y ∈ R 1×1×W ×H then we have the following: DISPLAYFORM5 2D-Inverse Fourier Transform with stride of of n2 DISPLAYFORM6 Note that Ĝ is the elementwise product of the 2D Discrete Fourier Transform between a feature of an input activation Y and the 2D Discrete Fourier Transform of every filter of the n 2 in Y. Since DISPLAYFORM7 is the inverse 2D Fourier transform along each of the n 2 filters resulting in Ĝ .

Thus U HO Y performs 2D convolutions independently along each of the n 2 filters, combined with (22); thus, Lemma 1 is proven.

DISPLAYFORM8 Lemma 3.

For τ -nice Prox-SG, DISPLAYFORM9 is an unbiased estimator to F ( X ).Proof.

DISPLAYFORM10 The first equality follows by introducing an indicator function where 1 i∈S = 1 if i ∈ S and zero otherwise.

The last equality follows from the uniformity across elements of the τ -nice S.From Lemma 3, and with zero initialization it follows that DISPLAYFORM11 is an unbiased estimator of ∇F ( X ) X =0 .

The last iteration follows by noting that A = A H H .

Therefore, the first iteration of τ -nice Prox-SGD with zero initialization and unit step size is: DISPLAYFORM12 Note that the previous stochastic sum in (26) with τ = (1 − p)n 1 can be reparameterized as follows: DISPLAYFORM13 where M ∈ R n1×1×W ×H is a mask tensor.

Note that since τ = (1 − p)n 1 , M has exactly pn 1 slices M(i, :, :, :) that are completely zero.

This equivalent to a dropout layer where the layer drops exactly pn 1 input activations.

It follows that (27) is equivalent to a forward pass through a BerDropout p layer followed by a linear layer and non-linear activation.

H PROOF OF PROPOSITION 2 mS2GD BID16 with zero initialization at the first epoch defines the following update: DISPLAYFORM14 With zero initialization at the first epoch we have Y = 0, therefore DISPLAYFORM15 I RANDOMIZED COORDINATE DESCENT PERMUTES DROPOUT AND LINEAR LAYERWe present an additional insight to the role of stochastic solvers on (5) in network design.

In particular, we show that performing a randomized coordinate descent, RCD, on (5) ignoring the finite sum structure, is equivalent to a linear transformation followed by BerDropout p and a non-linear activation.

That is, performing RCD permutes the order of linear transformation and dropout.

For ease of notation, we show this under the special case of fully connected layers.

Proposition 6.

A single iteration of Randomized Coordinate Descent, e.g. NSync , with τ -nice sampling of coordinates of (5) with τ = (1 − p)n 2 , unit step sizes along each partial derivative, and with zero initialization is equivalent to:Prox 1 L g i∈S A(i, :, :, :) HO X l−1 + B(i, :, :, :) , which is equivalent to a forward pass through a linear layer followed by a BerDropout p layer (that drops exactly n 2 p output activations) followed by a non-linear activation.

Proof.

We provide a sketch of the proof on the simple quadratic F (x) = 1 2 A x − x l−1 2 − b x where the linear layer is a fully-connected layer.

Considering a randomized coordinate descent, e.g. NSync, with τ -nice sampling of the coordinates we have the following: DISPLAYFORM16 Note that e i is a vector of all zeros except the i th coordinate which is equal to 1.

Moreover, since the step sizes along each partial derivative is 1, v = 1.

Equation (32) is equivalent to a forward pass through a linear layer followed by a BerDropout p layer and a non-linear activation.

<|TLDR|>

@highlight

A framework that links deep network layers to stochastic optimization algorithms; can be used to improve model accuracy and inform network design.