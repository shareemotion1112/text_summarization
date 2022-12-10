Abstract In this work, we describe a set of rules for the design and initialization of well-conditioned neural networks, guided by the goal of naturally balancing the diagonal blocks of the Hessian at the start of training.

We show how our measure of conditioning of a block relates to another natural measure of conditioning, the ratio of weight gradients to the weights.

We prove that for a ReLU-based deep multilayer perceptron, a simple initialization scheme using the geometric mean of the fan-in and fan-out satisfies our scaling rule.

For more sophisticated architectures, we show how our scaling principle can be used to guide design choices to produce well-conditioned neural networks, reducing guess-work.

The design of neural networks is often considered a black-art, driven by trial and error rather than foundational principles.

This is exemplified by the success of recent architecture random-search techniques (Zoph and Le, 2016; Li and Talwalkar, 2019) , which take the extreme of applying no human guidance at all.

Although as a field we are far from fully understanding the nature of learning and generalization in neural networks, this does not mean that we should proceed blindly.

In this work we define a scaling quantity γ l for each layer l that approximates the average squared singular value of the corresponding diagonal block of the Hessian for layer l.

This quantity is easy to compute from the (non-central) second moments of the forward-propagated values and the (non-central) second moments of the backward-propagated gradients.

We argue that networks that have constant γ l are better conditioned than those that do not, and we analyze how common layer types affect this quantity.

We call networks that obey this rule preconditioned neural networks, in analogy to preconditioning of linear systems.

As an example of some of the possible applications of our theory, we:

• Propose a principled weight initialization scheme that can often provide an improvement over existing schemes; • Show which common layer types automatically result in well-conditioned networks;

• Show how to improve the conditioning of common structures such as bottlenecked residual blocks by the addition of fixed scaling constants to the network (Detailed in Appendix E).

We will use the multilayer perceptron (i.e. a classical feed-forward deep neural network) as a running example as it is the simplest non-trivial deep neural network structure.

We use ReLU activation functions, and use the following notation for layer l of L (following He et al., 2015) :

x l+1 = ReLU (y l ), where W l is a n out l × n in l matrix of weights, b l is the bias vector, y l the preactivation vector and x l is the input activation vector for the layer.

The quantities n fan-in of the layer respectively.

We also denote the gradient of a quantity with respect to the loss (i.e. the back-propagated gradient) with the prefix ∆. We initially focus on the least-squares loss.

Additionally, we assume that each bias vector is initialized with zeros unless otherwise stated.

Our proposed approach focuses on the singular values of the diagonal blocks of the Hessian.

In the case of a multilayer perceptron network (MLP) network, each diagonal block corresponds to the weights from a single weight matrix W l or bias vector b l .

This block structure is used by existing approaches such as K-FAC and variants (Martens and Grosse, 2015; Grosse and Martens, 2016; Ba et al., 2017; George et al., 2018) , which correct the gradient step using estimates of secondorder information.

In contrast, our approach modifies the network to improve the Hessian without modifying the step.

Estimates of the magnitude of the singular values σ i (G l ) of the diagonal blocks G 1 ,. . .

,G L of the Hessian G provide information about the singular values of the full matrix.

Proposition 1.

Let G l be the lth diagonal block of a real symmetric matrix G : n × n. Then for all i = 1 . . .

n:

We can use this simple bound to provide some insight into the conditioning of the full matrix:

Corollary 2.

Let S = {s 1 , . . . , } be the union of the sets of singular values of the diagonal blocks G 1 , . . .

, G l of a real symmetric matrix G : n × n. Then the condition number κ(G) = σ max (G)/σ min (G) is bounded as:

In particular, a Hessian matrix with a very large difference between the singular values of each block must be ill-conditioned.

This provides strong motivation for balancing the magnitude of the singular values of each diagonal block, the goal of this work.

Although ideally, we would like to state the converse, that a matrix with balanced blocks is well conditioned, we can not make such a statement without strong assumptions on the off-diagonal behavior of the matrix.

We use the average squared singular value of each block as a proxy for the full spectrum, as it is particularly easy to estimate in expectation.

Although the minimum and maximum for each block would seem like a more natural quantity to work with, we found that any such bounds tend to be too pessimistic to reflect the behavior of the actual singular values.

When using the ReLU activation function, as we consider in this work, a neural network is no longer a smooth function of its inputs, and the Hessian becomes ill-defined at some points in the parameter space.

Fortunately, the spectrum is still well-defined at any twice-differentiable point, and this gives a local measure of the curvature.

ReLU networks are typically twice-differentiable almost everywhere.

We assume this throughout the remainder of this work.

Our analysis will proceed with batch-size 1 and a network with k outputs.

We consider the network at initialization, where weights are centered, symmetric and i.i.d random variables and biases are set to zero.

ReLU networks have a particularly simple structure for the Hessian with respect to any set of activations, as the network's output is a piecewise-linear function g fed into a final layer consisting of a loss.

This structure results in greatly simplified expressions for diagonal blocks of the Hessian with respect to the weights.

We will consider the output of the network as a composition two functions, the current layer g, and the remainder of the network h. We write this as a function of the weights, i.e. f (W l ) = h(g(W l )).

The dependence on the input to the network is implicit in this notation, and the network below layer l does not need to be considered.

h(y l ) be the Hessian of h, the remainder of the network after application of layer l (recall y l = W l x l ).

Let J l be the Jacobian of y l with respect to W l .

The Jacobian has shape

Given these quantities, the diagonal block of the Hessian corresponding to W l is equal to:

The lth diagonal block of the (Generalized) Gauss-Newton matrix G (Martens, 2014) .

We discuss this decomposition further in Appendix A.1.

We use the notation E[X 2 ] for any matrix or vector X to denote the expectation of the element-wise non-central second moment.

Proposition 3. (The GR scaling) Under the assumptions outlined in Section 3.2, the average squared singular value of G l is equal to the following quantity, which we call the GR scaling for MLP layers:

.

We define a "balanced" or "preconditioned" network as one in which γ l is equal for all l (full derivation in Appendix A).

Balancing this theoretically derived GR scaling quantity in a network will produce an initial optimization problem for which the blocks of the Hessian are expected to be approximately balanced with respect to their average squared singular value.

Due to the large number of approximations needed for this derivation, which we discuss further in the next section, we don't claim that this theoretical approximation is accurate, or that the blocks will be closely matched in practice.

Rather, we make the lesser claim that a network with very disproportionate values of γ l between layers is likely to have convergence difficulties during the early stages of optimization due to Cor.

2.

To check the quality of our approximation, we computed the ratio of the convolutional version of the GR scaling equation (Equation 1) to the actual E[(G l r) 2 ] product for a strided (rather than max-pooled, see Table 1 ) LeNet model, where we use random input data and a random loss (i.e. for outputs y we use y T Ry for an i.i.d normal matrix R), with batch-size 1024, and 32 × 32 input images.

The results are shown in Figure 2 for 100 sampled setups; there is generally good agreement with the theoretical expectation.

The following strong assumptions are used in the derivation of the GR scaling:

(A1) The input and target values are drawn element-wise i.i.d from a centered symmetric distribution with known variance.

(A2) The Hessian of the remainder of the network above each block, with respect to the output, has Frobenius norm much larger than 1.

More concretely, we assume that all but the highest order terms that are polynomial in this norm may be dropped.

(A3) All activations, pre-activations and gradients are independently distributed element-wise.

In practice due to the mixing effect of multiplication by random weight matrices, only the magnitudes of these quantities are correlated, and the effect is small for wide networks due to the law of large numbers.

Independence assumptions of this kind are common when approximating second-order methods; the block-diagonal variant of K-FAC (Martens and Grosse, 2015) makes similar assumptions for instance.

Assumption A2 is the most problematic of these assumptions, and we make no claim that it holds in practice.

However, we are primarily interested in the properties of blocks and their scaling with respect to each other, not their absolute scaling.

Assumption A2 results in very simple expressions for the scaling of the blocks without requiring a more complicated analysis of the top of the network.

Similar theory can be derived for other assumptions on the output structure, such as the assumption that the target values are much smaller than the outputs of the network.

We provide further motivation for the utility of preconditioning by comparing it to another simple quantity of interest.

Consider at network initialization, the ratio of the (element-wise non-central) second moments of each weight-matrix gradient to the weight matrix itself:

This ratio approximately captures the relative change that a single SGD step with unit step-size on W l will produce.

We call this quantity the weight-to-gradient ratio.

A network with constant ν l is also well-behaved under weight-decay, as the ratio of weight-decay second moments to gradient second moments will stay constant throughout the network, keeping the push-pull of gradients and decay constant across the network.

Remarkably, the weight-to-gradient ratio ν l turns out to be equivalent to the GR scaling for MLP networks: Proposition 4. (Appendix 8) ν l is equal to the GR scaling γ l for i.i.d mean-zero randomly-initialized multilayer perceptron layers under the independence assumptions of Appendix 3.2.

The concept of GR scaling may be extended to scaled convolutional layers y l = α l Conv W l (x l ) + b l with scaling factor α l , kernel width k l , batch-size b, and output resolution ρ l × ρ l .

A straight-forward derivation gives expressions for the convolution weight and biases of:

This requires an assumption of independence of the values of activations within a channel that is not true in practice, so γ l tends to be further away from empirical estimates for convolutional layers than for non-convolutional layers, although it is still a useful guide.

The effect of padding is also ignored here.

Sequences of convolutions are well-scaled against each other along as the kernel size remains the same.

The scaling of layers involving differing kernel sizes can be corrected using the alpha parameter (Appendix E), and more generally any imbalance between the conditioning of layers can be fixed by modifying α l while at the same time changing the initialization of W l so that the forward variance remains the same as the unmodified version.

This adjusts γ l while leaving all other γ the same.

For ReLU networks with a classical multilayer-perceptron (i.e. non-convolutional, non-residual) structure, we show in this section that initialization using i.i.d mean-zero random variables with (non-central) second moment inversely proportional to the geometric mean of the fans:

for some fixed constant c, results in a constant GR scaling throughout the network.

Proposition 5.

Let W 0 : m × n and W 1 : p × m be weight matrices satisfying the geometric initialization criteria of Equation 2, and let b 0 , b 1 be zero-initialized bias parameters.

Then consider the following sequence of two layers where x 0 and ∆y 1 are i.i.d, mean 0, uncorrelated and symmetrically distributed:

Proof.

Note that the ReLU operation halves both the forward and backward (non-central) second moments, due to our assumptions on the distributions of x 0 and ∆y 1 .

So:

Consider the first weight-gradient ratio, using E[∆W

Under our assumptions, back-propagation to ∆x 1 results in E[∆x

Now consider the second weight-gradient ratio:

Under our assumptions, applying forward propagation gives E[y

0 ], and so from Equation 3 we have:

which matches Equation 4, so ν 0 = ν 1 .

Remark 6.

This relation also holds for sequences of (potentially) strided convolutions, but only if the same kernel size is used everywhere and circular padding is used.

The initialization should be modified to include the kernel size, changing the expression to c/ k

Under review as a conference paper at ICLR 2020

The most common approaches are the Kaiming (He et al., 2015) and Xavier (Glorot and Bengio, 2010) initializations.

The Kaiming technique for ReLU networks is actually one of two approaches:

For the feed-forward network above, assuming random activations, the forward-activation variance will remain constant in expectation throughout the network if fan-in initialization of weights (LeCun et al., 2012 ) is used, whereas the fan-out variant maintains a constant variance of the back-propagated signal.

The constant factor 2 in the above expressions corrects for the variance-reducing effect of the ReLU activation.

Although popularized by He et al. (2015) , similar scaling was in use in early neural network models that used tanh activation functions (Bottou, 1988) .

These two principles are clearly in conflict; unless n in l = n out l , either the forward variance or backward variance will become non-constant, or as more commonly expressed, either explode or vanish.

No prima facie reason for preferring one initialization over the other is provided.

Unfortunately, there is some confusion in the literature as many works reference using Kaiming initialization without specifying if the fan-in or fan-out variant is used.

The Xavier initialization (Glorot and Bengio, 2010) is the closest to our proposed approach.

They balance these conflicting objectives using the arithmetic mean:

to "... approximately satisfy our objectives of maintaining activation variances and back-propagated gradients variance as one moves up or down the network".

This approach to balancing is essentially heuristic, in contrast to the geometric mean approach that our theory directly guides us to.

We can use the same proof technique to compute the GR scaling for the bias parameters in a network.

Our update equations change to include the bias term: y l = W l x l + b l , with b l assumed to be initialized at zero.

We show in Appendix D that:

.

It is easy to show using the techniques of Section 6 that the biases of consecutive layers have equal GR scaling as long as geometric initialization is used.

However, unlike in the case of weights, we have less flexibility in the choice of the numerator.

Instead of allowing all weights to be scaled by c for any positive c, we require that c = 2, so that:

Proposition 7

It is traditional to normalize a dataset before applying a neural network so that the input vector has mean 0 and variance 1 in expectation.

This principle is rarely quested in modern neural networks, even though there is no longer a good justification for its use in modern ReLU based networks.

In contrast, our theory provides direct guidance for the choice of input scaling.

We show that the (non-central) second moment of the input affects the GR scaling of bias and weight parameters differently and that they can be balanced by careful choice of the initialization.

Consider the GR scaling values for the bias and weight parameters in the first layer of a ReLU-based multilayer perceptron network, as considered in previous sections.

We assume the data is already

.

We can cancel terms to find the value of E x 2 0 that makes these two quantities equal:

In common computer vision architectures, the input planes are the 3 color channels, and the kernel size is k = 3, giving E x 2 0 ≈ 0.2.

Using the traditional variance-one normalization will result in the effective learning rate for the bias terms being lower than that of the weight terms.

This will result in potentially slower learning of the bias terms than for the input scaling we propose.

A neural network's behavior is also very sensitive to the (non-central) second moment of the outputs.

For a convolutional network without pooling layers (but potentially with strided dimensionality reduction), if geometric-mean initialization is used the activation (non-central) second moments are given by:

The application of a sequence of these layers gives a telescoping product:

We potentially have independent control over this (non-central) second moment at initialization, as we can insert a fixed scalar multiplication factor at the end of the network that modifies it.

This may be necessary when adapting a network architecture that was designed and tested under a different initialization scheme, as the success of the architecture may be partially due to the output scaling that happens to be produced by that original initialization.

We are not aware of any existing theory guiding the choice of output variance at initialization for the case of log-softmax losses, where it has a non-trivial effect on the back-propagated signals, although output variances of 0.01 to 0.1 appear to work well.

The output variance should always be checked and potentially corrected when switching initialization schemes.

Consider a network where γ l is constant throughout.

We may add a layer between any two existing layers without affecting this conditioning, as long as the new layer maintains the activation-gradient (non-central) second-moment product: and dimensionality; this follows from Equation 1.

For instance, adding a simple scaling layer of the form x l+1 = 2x l doubles the (non-central) second moment during the forward pass and doubles the backward (non-central) second moment during back-propagation, which maintains this product:

When spatial dimensionality changes between layers we can see that the GR scaling is no longer maintained just by balancing this product, as γ depends directly on the square of the spatial dimension.

Instead, a pooling operation that changes the forward and backward signals in a way that counteracts the change in spatial dimension is needed.

The use of stride-2 convolutions, as well as average pooling, results in the correct scaling, but other common types of spatial reduction generally do not.

It is particularly interesting to note that the evolution in state-of-the-art architectures corresponds closely to a move from poorly scaled building blocks to well-scaled ones.

Early shallow architectures like LeNet-5 used tanh nonlinearities, which were replaced by the (well-scaled) ReLU, used for instance in the seminal AlexNet architecture (Krizhevsky et al., 2012) .

AlexNet and the latter VGG architectures made heavy use of max-pooling and reshaping before the final layers, both operations which have been replaced in modern fully-convolutional architectures with (well-scaled) striding and average-pooling respectively.

The use of large kernel sizes is also in decline.

The AlexNet architecture used kernel sizes of 11, 5 and 3, whereas modern ResNet (He et al., 2016) architectures only use 7, 3 and 1.

Furthermore, recent research has shown that replacing the single 7x7 convolution used with a sequence of three 3x3 convolutions improves performance (He et al., 2018) .

We considered a selection of dense and moderate-sparsity multi-class classification datasets from the LibSVM repository, 26 in total.

The same model was used for all datasets, a non-convolutional ReLU network with 3 weight layers total.

The inner two layer widths were fixed at 384 and 64 nodes respectively.

These numbers were chosen to result in a larger gap between the optimization methods, less difference could be expected if a more typical 2× gap was used.

Our results are otherwise generally robust to the choice of layer widths.

For every dataset, learning rate and initialization combination we ran 10 seeds and picked the median loss after 5 epochs as the focus of our study (The largest differences can be expected early in training).

Learning rates in the range 2 1 to 2 −12 (in powers of 2) were checked for each dataset and initialization combination, with the best learning rate chosen in each case based off of the median of the 10 seeds.

Training loss was used as the basis of our comparison as we care primarily about convergence rate, and are comparing identical network architectures.

Some additional details concerning the experimental setup and which datasets were used is available in the Appendix.

Table 1 shows that geometric initialization is the most consistent of the initialization approaches considered.

It has the lowest loss, after normalizing each dataset, and it is never the worst of the 4 methods on any dataset.

Interestingly, the fan-out method is most often the best method, but consideration of the per-dataset plots (Appendix F) shows that it often completely fails to learn for some problems, which pulls up its average loss and results in it being the worst for 9 datasets.

Testing an initialization method on modern computer vision problems is problematic due to the heavy architecture search, both automated and manual, that is behind the current best methods.

This search will fit the architecture to the initialization method, in a sense, so any other initialization is at a disadvantage compared to the one used during architecture search.

This is further complicated by the prevalence of BatchNorm which is not handled in our theory.

Instead, to provide a clear comparison we use an older network with a large variability in kernel sizes, the AlexNet architecture.

This architecture has a large variety of filter sizes (11, 5, 3, linear) , which according to our theory will affect the conditioning adversely, and which should highlight the differences between the methods.

We found that a network with consistent kernel sizes through-out showed only negligible differences between the initialization methods.

The network was modified to replace max-pooling with striding as max-pooling is not well-scaled by our theory (further details in Appendix F).

Following Section 6.4, we normalize the output of the network at initialization by running a single batch through the network and adding a fixed scaling factor to the network to produce output standard deviation 0.05.

For our preconditioned variant, we added alpha correction factors following Section 5 in conjunction with geometric initialization, and compared this against other common initialization methods.

We tested on CIFAR-10 following the standard practice as closely as possible, as detailed in Appendix F. We performed a geometric learning rate sweep over a power-of-two grid.

Results are shown in Figure 3 for an average of 10 seeds for each initialization.

Preconditioning improves training loss over all other initialization schemes tested, although only by a small margin.

Although not a panacea, by using the scaling principle we have introduced, neural networks can be designed with a reasonable expectation that they will be optimizable by stochastic gradient methods, minimizing the amount of guess-and-check neural network design.

As a consequence of our scaling principle, we have derived an initialization scheme that automatically preconditions common network architectures.

Most developments in neural network theory attempt to explain the success of existing techniques post-hoc.

Instead, we show the power of the scaling law approach by deriving a new initialization technique from theory directly.

we then combine the simplifications from Equations 9, 10, and 11 to give:

.

Standard ReLU classification and regression networks have a particularly simple structure for the Hessian with respect to the input, as the network's output is a piecewise-linear function g feed into a final layer consisting of a convex log-softmax operation, or a least-squares loss.

This structure results in the Hessian with respect to the input being equivalent to its Gauss-Newton approximation.

The Gauss-Newton matrix can be written in a factored form, which is used in the analysis we perform in this work.

We emphasize that this is just used as a convenience when working with diagonal blocks, the GN representation is not an approximation in this case.

The (Generalized) Gauss-Newton matrix G is a positive semi-definite approximation of the Hessian of a non-convex function f , given by factoring f into the composition of two functions

where h is convex, and g is approximated by its Jacobian matrix J at x, for the purpose of computing G:

The GN matrix also has close ties to the Fisher information matrix (Martens, 2014) , providing another justification for its use.

Surprisingly, the Gauss-Newton decomposition can be used to compute diagonal blocks of the Hessian with respect to the weights W l as well as the inputs (Martens, 2014) .

To see this, note that for any activation y l , the layers above may be treated in a combined fashion as the h in a f (W l ) = h(g(W l )) decomposition of the network structure, as they are the composition of a (locally) linear function and a convex function and thus convex.

In this decomposition g(W l ) =W l x l + b l is a function of W l with x l fixed, and as this is linear in W l , the Gauss-Newton approximation to the block is thus not an approximation.

We make heavy use of the equations for forward propagation and backward propagation of second moments, under the assumption that the weights are uncorrelated to the activations or gradients.

For a convolution

with input channels n in , output channels n out , and square k × k kernels, these formulas are (recall our notation for the second moments is element-wise for vectors and matrices):

C THE WEIGHT GRADIENT RATIO IS EQUAL TO GR SCALING FOR MLP MODELS Proposition 8.

The weight-gradient ratio ν l is equal to the GR scaling γ l for i.i.d mean-zero randomly-initialized multilayer perceptron layers under the independence assumptions of Appendix 3.2.

Proof.

To see the equivalence, note that under the zero-bias initialization, we have from

and so:

The gradient of the weights is given by ∆W ij = ∆y li x lj and so its second moment is:

Combining these quantities gives:

.

We consider the case of a convolutional neural network with spatial resolution ρ × ρ for greater generality.

Consider the Jacobian of y l with respect to the bias.

It has shape J

Each row corresponds to a y l output, and each column a bias weight.

As before, we will approximate the product of G with a random i.i.d unit variance vector r:

The structure of J b l is that each block of ρ 2 rows has the same set of 1s in the same column.

Only a single 1 per row.

It follows that:

The calculation of the product of R l with J b l r is approximated in the same way as in the weight scaling calculation.

For the J bT product, note that there is an additional ρ 2 as each column has ρ 2 non-zero entries, each equal to 1.

Combining these three quantities gives:

.

Proposition 9.

Consider the setup of Proposition 5, with the addition of biases:

As long as the weights are initialized following Equation 7 and the biases are initialized to 0, we have that

We will include c = 2 as a variable as it clarifies it's relation to other quantities.

We reuse some calculations from Proposition 5.

Namely that:

Plugging these into the definition of γ b 0 :

.

b 1 , we require the additional quantity:

Again plugging this in:

.

There has been significant recent interest in training residual networks without the use of batchnormalization or other normalization layers (Zhang et al., 2019) .

In this section, we explore the modifications that are necessary to a network for this to be possible and show how to apply our preconditioning principle to these networks.

The building block of a ResNet model is the residual block:

where F is a composition of layers.

Unlike classical feedforward architectures, the pass-through connection results in an exponential increase in the variance of the activations in the network as the depth increases.

A side effect of this is the output of the network becomes exponentially more sensitive to the input of the network as depth increases, a property characterized by the Lipschitz constant of the network (Hanin, 2018) .

This exponential dependence can be reduced by the introduction of scaling constants s l to each block:

The introduction of these constants requires a modification of the block structure to ensure constant conditioning between blocks.

A standard bottleneck block, as used in the ResNet-50 architecture, has the following form:

In this notation, C 0 is a 1 × 1 convolution that reduces the number of channels 4 fold, C 1 is a 3 × 3 convolution with equal input and output channels, and C 2 is a 1 × 1 convolution at increases the number of channels back up 4 fold to the original input count.

If we introduce a scaling factor s l to each block l, then we must also add conditioning multipliers β l to each convolution to change their GR scaling, as we described in Section 5.

The correct scaling constant depends on the scaling constant of the previous block.

A simple calculation gives the equation:

The initial β 0 and s 0 may be chosen arbitrarily.

If a flat s l = s is used for all l, then we may use β l = 1.

The block structure including the β l factors is:

x 1 = ReLU(y 0 ),

x 2 = ReLU(y 1 ),

The weights of each convolution must then be initialized with the standard deviation modified such that the combined convolution-scaling operation gives the same output variance as would be given if the geometric-mean initialization scheme is used without extra scaling constants.

For instance, the initialization of the C 0 convolution must have standard deviation scaled down by dividing by 1 β so that the multiplication by 1 β during the forward pass results in the correct forward variance.

The 1/ √ 3 factor is an α correction that corrects for change in kernel shape for the middle convolution.

The variance at initialization must be scaled to correct for the α factor also.

Since the initial convolution in a ResNet-50 model is also not within a residual block, it's GR scaling is different from the convolutions within residual blocks.

Consider the composition of a non-residual followed by a residual block, without max-pooling or ReLUs for simplicity of exposition:

Without loss of generality, we assume that E x 2 0 = 1, and assume a single channel input and output.

Our goal is to find a constant α, so that γ 0 = γ 1 .

Recall that when using α scaling factors we must initialize C 0 so that the variance of y 0 is independent of the choice of α.

Our scaling factor will also depend on the kernel sizes used in the two convolutions, so we must include those in the calculations.

From Equation 1, the GR scaling for C 0 is

Note that E[∆y

For the residual convolution, we need to use a modification of the standard GR equation due to the residual branch.

The derivation of γ for non-residual convolutions assumes that the remainder of the network above the convolution responds linearly (locally) with the scaling of the convolution, but here due to the residual connection, this is no longer the case.

For instance, if the weight were scaled to zero, the output of the network would not also become zero (recall our assumption of zero-initialization for bias terms).

This can be avoided by noting that the ratio E[∆y

] in the GR scaling may be computed further up the network, as long as any scaling in between is corrected for.

In particular, we may compute this ratio at the point after the residual addition, as long as we include the factor s 4 1 to account for this.

So we in fact have:

We now equate γ 0 = γ 1 :

Therefore to ensure that γ 0 = γ 1 we need:

.

A similar calculation applies when the residual block is before the non-residual convolution, as in the last layer linear in the ResNet network, giving a scaling factor for the linear layer (effective kernel size 1) of:

To prevent the results from being skewed by the number of classes and the number of inputs affecting the output variance, the logit output of the network was scaled to have standard deviation 0.05 after the first minibatch evaluation for every method, with the scaling constant fixed thereafter.

LayerNorm was used on the input to whiten the data.

Weight decay of 0.00001 was used for every dataset.

To aggregate the losses across datasets we divided by the worst loss across the initializations before averaging.

LIBSVM PLOTS Figure 4 shows the interquartile range (25%, 50% and 75% quantiles) of the best learning rate for each case.

Following standard practice, training used random augmentations consisting of horizontal flips and random crops to 32x32, as well as normalization to the interval [-1,+1].

We used SGD with momentum 0.9, a learning rate schedule of decreases at 150 and 225 epochs, and no weight decay.

The network architecture is the following sequence, with circular "equal" padding used and ReLU nonlinearities after each convolution:

1. 11x11 stride-1 convolution with 3 input and 64 output channels, 2. 5x5 stride-2 convolution with 64 input and 192 output channels, 3.

3x3 stride-2 convolution with 192 input and 384 output channels,

<|TLDR|>

@highlight

A theory for initialization and scaling of ReLU neural network layers