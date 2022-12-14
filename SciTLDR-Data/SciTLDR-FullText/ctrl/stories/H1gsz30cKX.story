Normalization layers are a staple in state-of-the-art deep neural network architectures.

They are widely believed to stabilize training, enable higher learning rate, accelerate convergence and improve generalization, though the reason for their effectiveness is still an active research topic.

In this work, we challenge the commonly-held beliefs by showing that none of the perceived benefits is unique to normalization.

Specifically, we propose fixed-update initialization (Fixup), an initialization motivated by solving the exploding and vanishing gradient problem at the beginning of training via properly rescaling a standard initialization.

We find training residual networks with Fixup to be as stable as training with normalization -- even for networks with 10,000 layers.

Furthermore, with proper regularization, Fixup enables residual networks without normalization to achieve state-of-the-art performance in image classification and machine translation.

Artificial intelligence applications have witnessed major advances in recent years.

At the core of this revolution is the development of novel neural network models and their training techniques.

For example, since the landmark work of BID13 , most of the state-of-the-art image recognition systems are built upon a deep stack of network blocks consisting of convolutional layers and additive skip connections, with some normalization mechanism (e.g., batch normalization BID16 ) to facilitate training and generalization.

Besides image classification, various normalization techniques (Ulyanov et al., 2016; BID0 BID26 Wu & He, 2018) have been found essential to achieving good performance on other tasks, such as machine translation (Vaswani et al., 2017) and generative modeling (Zhu et al., 2017) .

They are widely believed to have multiple benefits for training very deep neural networks, including stabilizing learning, enabling higher learning rate, accelerating convergence, and improving generalization.• Training without normalization.

We propose Fixup, a method that rescales the standard initialization of residual branches by adjusting for the network architecture.

Fixup enables training very deep residual networks stably at maximal learning rate without normalization.

In the remaining of this paper, we first analyze the exploding gradient problem of residual networks at initialization in Section 2.

To solve this problem, we develop Fixup in Section 3.

In Section 4 we quantify the properties of Fixup and compare it against state-of-the-art normalization methods on real world benchmarks.

A comparison with related work is presented in Section 5.

Standard initialization methods BID6 BID12 Xiao et al., 2018) attempt to set the initial parameters of the network such that the activations neither vanish nor explode.

Unfortunately, it has been observed that without normalization techniques such as BatchNorm they do not account properly for the effect of residual connections and this causes exploding gradients.

BID1 characterizes this problem for ReLU networks, and we will generalize this to residual networks with positively homogenous activation functions.

A plain (i.e. without normalization layers) ResNet with residual blocks {F 1 , . . .

, F L } and input x 0 computes the activations as DISPLAYFORM0 ResNet output variance grows exponentially with depth.

Here we only consider the initialization, view the input x 0 as fixed, and consider the randomness of the weight initialization.

We analyze the variance of each layer x l , denoted by Var[x l ] (which is technically defined as the sum of the variance of all the coordinates of x l .)

For simplicity we assume the blocks are initialized to be zero mean, i.e., DISPLAYFORM1 , and the law of total variance, we have DISPLAYFORM2 Resnet structure prevents x l from vanishing by forcing the variance to grow with depth, i.e. DISPLAYFORM3 with initialization methods such as BID12 , the output variance of each residual branch This causes the output variance to explode exponentially with depth without normalization BID10 for positively homogeneous blocks (see Definition 1).

This is detrimental to learning because it can in turn cause gradient explosion.

As we will show, at initialization, the gradient norm of certain activations and weight tensors is lower bounded by the cross-entropy loss up to some constant.

Intuitively, this implies that blowup in the logits will cause gradient explosion.

Our result applies to convolutional and linear weights in a neural network with ReLU nonlinearity (e.g., feed-forward network, CNN), possibly with skip connections (e.g., ResNet, DenseNet), but without any normalization.

Our analysis utilizes properties of positively homogeneous functions, which we now introduce.

Definition 1 (positively homogeneous function of first degree).

A function f : R m → R n is called positively homogeneous (of first degree) (p.h.) if for any input x ∈ R m and α > 0, f (αx) = αf (x).Definition 2 (positively homogeneous set of first degree).

Let θ = {θ i } i∈S be the set of parameters of f (x) and θ ph = {θ i } i∈S ph ⊂S .

We call θ ph a positively homogeneous set (of first degree) (p.h. set) if for any α > 0, f (x; θ \ θ ph , αθ ph ) = αf (x; θ \ θ ph , θ ph ), where αθ ph denotes {αθ i } i∈S ph .Intuitively, a p.h.

set is a set of parameters θ ph in function f such that for any fixed input x and fixed DISPLAYFORM4 Examples of p.h.

functions are ubiquitous in neural networks, including various kinds of linear operations without bias (fully-connected (FC) and convolution layers, pooling, addition, concatenation and dropout etc.) as well as ReLU nonlinearity.

Moreover, we have the following claim: Proposition 1.

A function that is the composition of p.h.

functions is itself p.h.

We study classification problems with c classes and the cross-entropy loss.

We use f to denote a neural network function except for the softmax layer.

Cross-entropy loss is defined as (z, y) −y T (z − logsumexp(z)) where y is the one-hot label vector, z f (x) ∈ R c is the logits where z i denotes its i-th element, and logsumexp(z) DISPLAYFORM5 and the average cross-entropy loss DISPLAYFORM6 , where we use (m) to index quantities referring to the m-th example.

· denotes any valid norm.

We only make the following assumptions about the network f : These assumptions hold at initialization if we remove all the normalization layers in a residual network with ReLU nonlinearity, assuming all the biases are initialized at 0.

DISPLAYFORM7 Our results are summarized in the following two theorems, whose proofs are listed in the appendix: Theorem 1.

Denote the input to the i-th block by x i−1 .

With Assumption 1, we have DISPLAYFORM8 where p is the softmax probabilities and H denotes the Shannon entropy.

Since H(p) is upper bounded by log(c) and x i−1 is small in the lower blocks, blowup in the loss will cause large gradient norm with respect to the lower block input.

Our second theorem proves a lower bound on the gradient norm of a p.h.

set in a network.

Theorem 2.

With Assumption 1, we have DISPLAYFORM9 Furthermore, with Assumptions 1 and 2, we have DISPLAYFORM10 It remains to identify such p.h.

sets in a neural network.

In Figure 2 Figure 2 : Examples of p.h.

sets in a ResNet without normalization: (1) the first convolution layer before max pooling; (2) the fully connected layer before softmax; (3) the union of a spatial downsampling layer in the backbone and a convolution layer in its corresponding residual branch.3 FIXUP: UPDATE A RESIDUAL NETWORK Θ(η) PER SGD STEP Our analysis in the previous section points out the failure mode of standard initializations for training deep residual network: the gradient norm of certain layers is in expectation lower bounded by a quantity that increases indefinitely with the network depth.

However, escaping this failure mode does not necessarily lead us to successful training -after all, it is the whole network as a function that we care about, rather than a layer or a network block.

In this section, we propose a top-down design of a new initialization that ensures proper update scale to the network function, by simply rescaling a standard initialization.

To start, we denote the learning rate by η and set our goal: DISPLAYFORM11 Put another way, our goal is to design an initialization such that SGD updates to the network function are in the right scale and independent of the depth.

We define the Shortcut as the shortest path from input to output in a residual network.

The Shortcut is typically a shallow network with a few trainable layers.

1 We assume the Shortcut is initialized using a standard method, and focus on the initialization of the residual branches.

Residual branches update the network in sync.

To start, we first make an important observation that the SGD update to each residual branch changes the network output in highly correlated directions.

This implies that if a residual network has L residual branches, then an SGD step to each residual branch should change the network output by Θ(η/L) on average to achieve an overall Θ(η) update.

We defer the formal statement and its proof until Appendix B.1.Study of a scalar branch.

Next we study how to initialize a residual branch with m layers so that its SGD update changes the network output by Θ(η/L).

We assume m is a small positive integer (e.g., 2 or 3).

As we are only concerned about the scale of the update, it is sufficiently instructive to study the scalar case, i.e., F (x) = ( m i=1 a i ) x where a 1 , . . .

, a m , x ∈ R + .

For example, the standard initialization methods typically initialize each layer so that the output (after nonlinear activation) preserves the input variance, which can be modeled as setting ∀i ∈ [m], a i = 1.

In turn, setting a i to a positive number other than 1 corresponds to rescaling the i-th layer by a i .Through deriving the constraints for F (x) to make Θ(η/L) updates, we will also discover how to rescale the weight layers of a standard initialization as desired.

In particular, we show the SGD update to F (x) is Θ(η/L) if and only if the initialization satisfies the following constraint: DISPLAYFORM12 We defer the derivation until Appendix B.2.Equation (5) suggests new methods to initialize a residual branch through rescaling the standard initialization of i-th layer in a residual branch by its corresponding scalar a i .

For example, we DISPLAYFORM13 Alternatively, we could start the residual branch as a zero function by setting a m = 0 and ∀i DISPLAYFORM14 In the second option, the residual branch does not need to "unlearn" its potentially bad random initial state, which can be beneficial for learning.

Therefore, we use the latter option in our experiments, unless otherwise specified.

The effects of biases and multipliers.

With proper rescaling of the weights in all the residual branches, a residual network is supposed to be updated by Θ(η) per SGD step -our goal is achieved.

However, in order to match the training performance of a corresponding network with normalization, there are two more things to consider: biases and multipliers.

Using biases in the linear and convolution layers is a common practice.

In normalization methods, bias and scale parameters are typically used to restore the representation power after normalization.

Intuitively, because the preferred input/output mean of a weight layer may be different from the preferred output/input mean of an activation layer, it also helps to insert bias terms in a residual network without normalization.

Empirically, we find that inserting just one scalar bias before each weight layer and nonlinear activation layer significantly improves the training performance.

Multipliers scale the output of a residual branch, similar to the scale parameters in batch normalization.

They have an interesting effect on the learning dynamics of weight layers in the same branch.

Specifically, as the stochastic gradient of a layer is typically almost orthogonal to its weight, learning rate decay tends to cause the weight norm equilibrium to shrink when combined with L2 weight decay (van Laarhoven, 2017) .

In a branch with multipliers, this in turn causes the growth of the multipliers, increasing the effective learning rate of other layers.

In particular, we observe that inserting just one scalar multiplier per residual branch mimics the weight norm dynamics of a network with normalization, and spares us the search of a new learning rate schedule.

Put together, we propose the following method to train residual networks without normalization:Fixup initialization (or: How to train a deep residual network without normalization) 1.

Initialize the classification layer and the last layer of each residual branch to 0.

2.

Initialize every other layer using a standard method (e.g., BID12 ), and scale only the weight layers inside residual branches by L − 1 2m−2 .

3.

Add a scalar multiplier (initialized at 1) in every branch and a scalar bias (initialized at 0) before each convolution, linear, and element-wise activation layer.

It is important to note that Rule 2 of Fixup is the essential part as predicted by Equation (5).

Indeed, we observe that using Rule 2 alone is sufficient and necessary for training extremely deep residual networks.

On the other hand, Rule 1 and Rule 3 make further improvements for training so as to match the performance of a residual network with normalization layers, as we explain in the above text.

3 We find ablation experiments confirm our claims (see Appendix C.1).Our initialization and network design is consistent with recent theoretical work BID11 ; BID20 , which, in much more simplified settings such as linearized residual nets and quadratic neural nets, propose that small initialization tend to stabilize optimization and help generalizaiton.

However, our approach suggests that more delicate control of the scale of the initialization is beneficial.

One of the key advatanges of BatchNorm is that it leads to fast training even for very deep models BID16 .

Here we will determine if we can match this desirable property by relying only on proper initialization.

We propose to evaluate how each method affects training very deep nets by measuring the test accuracy after the first epoch as we increase depth.

In particular, we use the wide residual network (WRN) architecture with width 1 and the default weight decay 5e−4 (Zagoruyko & Komodakis, 2016).

We specifically use the default learning rate of 0.1 because the ability to use high learning rates is considered to be important to the success of BatchNorm.

We compare Fixup against three baseline methods -(1) rescale the output of each residual block by BID1 , FORMULA8 post-process an orthogonal initialization such that the output variance of each residual block is close to 1 (Layer-sequential unit-variance orthogonal initialization, or LSUV) BID22 , (3) batch normalization BID16 .

We use the default batch size of 128 up to 1000 layers, with a batch size of 64 for 10,000 layers.

We limit our budget of epochs to 1 due to the computational strain of evaluating models with up to 10,000 layers.

To demonstrate the generality of Fixup, we also apply it to replace layer normalization BID0 in Transformer (Vaswani et al., 2017) , a state-of-the-art neural network for machine translation.

Specifically, we use the fairseq library BID5 and follow the Fixup template in Section 3 to modify the baseline model.

We evaluate on two standard machine translation datasets, IWSLT German-English (de-en) and WMT English-German (en-de) following the setup of BID23 .

For the IWSLT de-en dataset, we cross-validate the dropout probability from {0.3, 0.4, 0.5, 0.6} and find 0.5 to be optimal for both Fixup and the LayerNorm baseline.

For the WMT'16 en-de dataset, we use dropout probability 0.4.

All models are trained for 200k updates.

DISPLAYFORM15 It was reported BID2 that "Layer normalization is most critical to stabilize the training process... removing layer normalization results in unstable training runs".

However we find training with Fixup to be very stable and as fast as the baseline model.

Results are shown in TAB6 .

Surprisingly, we find the models do not suffer from overfitting when LayerNorm is replaced by Fixup, thanks to the strong regularization effect of dropout.

Instead, Fixup matches or supersedes the state-of-the-art results using Transformer model on both datasets.

Normalization methods.

Normalization methods have enabled training very deep residual networks, and are currently an essential building block of the most successful deep learning architectures.

All normalization methods for training neural networks explicitly normalize (i.e. standardize) some component (activations or weights) through dividing activations or weights by some real number computed from its statistics and/or subtracting some real number activation statistics (typically the mean) from the activations.

6 In contrast, Fixup does not compute statistics (mean, variance or norm) at initialization or during any phase of training, hence is not a normalization method.

Theoretical analysis of deep networks.

Training very deep neural networks is an important theoretical problem.

Early works study the propagation of variance in the forward and backward pass for different activation functions BID6 BID12 .Recently, the study of dynamical isometry (Saxe et al., 2013) provides a more detailed characterization of the forward and backward signal propogation at initialization BID24 BID9 , enabling training 10,000-layer CNNs from scratch (Xiao et al., 2018) .

For residual networks, activation scale BID10 , gradient variance BID1 and dynamical isometry property (Yang & Schoenholz, 2017) have been studied.

Our analysis in Section 2 leads to the similar conclusion as previous work that the standard initialization for residual networks is problematic.

However, our use of positive homogeneity for lower bounding the gradient norm of a neural network is novel, and applies to a broad class of neural network architectures (e.g., ResNet, DenseNet) and initialization methods (e.g., Xavier, LSUV) with simple assumptions and proof.

BID11 analyze the optimization landscape (loss surface) of linearized residual nets in the neighborhood around the zero initialization where all the critical points are proved to be global minima.

Yang & Schoenholz (2017) study the effect of the initialization of residual nets to the test performance and pointed out Xavier or He initialization scheme is not optimal.

In this paper, we give a concrete recipe for the initialization scheme with which we can train deep residual networks without batch normalization successfully.

Understanding batch normalization.

Despite its popularity in practice, batch normalization has not been well understood.

BID16 attributed its success to "reducing internal covariate shift", whereas BID27 argued that its effect may be "smoothing loss surface".

Our analysis in Section 2 corroborates the latter idea of BID27 by showing that standard initialization leads to very steep loss surface at initialization.

Moreover, we empirically showed in Section 3 that steep loss surface may be alleviated for residual networks by using smaller initialization than the standard ones such as Xavier or He's initialization in residual branches.

van Laarhoven (2017); BID15 studied the effect of (batch) normalization and weight decay on the effective learning rate.

Their results inspire us to include a multiplier in each residual branch.

ResNet initialization in practice.

BID5 ; BID1 proposed to address the initialization problem of residual nets by using the recurrence x l = 1 / 2 (x l−1 + F l (x l−1 )).

BID22 proposed a data-dependent initialization to mimic the effect of batch normalization in the first forward pass.

While both methods limit the scale of activation and gradient, they would fail to train stably at the maximal learning rate for very deep residual networks, since they fail to consider the accumulation of highly correlated updates contributed by different residual branches to the network function (Appendix B.1).

Srivastava et al. FORMULA0 ; BID11 ; BID7 ; BID17 found that initializing the residual branches at (or close to) zero helped optimization.

Our results support their observation in general, but Equation (5) suggests additional subtleties when choosing a good initialization scheme.

In this work, we study how to train a deep residual network reliably without normalization.

Our theory in Section 2 suggests that the exploding gradient problem at initialization in a positively homogeneous network such as ResNet is directly linked to the blowup of logits.

In Section 3 we develop Fixup initialization to ensure the whole network as well as each residual branch gets updates of proper scale, based on a top-down analysis.

Extensive experiments on real world datasets demonstrate that Fixup matches normalization techniques in training deep residual networks, and achieves state-of-the-art test performance with proper regularization.

Our work opens up new possibilities for both theory and applications.

Can we analyze the training dynamics of Fixup, which may potentially be simpler than analyzing models with batch normalization is?

Could we apply or extend the initialization scheme to other applications of deep learning?

It would also be very interesting to understand the regularization benefits of various normalization methods, and to develop better regularizers to further improve the test performance of Fixup.

Proof of Theorem 1.

We use f i→j to denote the composition DISPLAYFORM0 .

Note that z is p.h.

with respect to the input of each network block, i.e. f i→L ((1 + )x i−1 ) = (1 + )f i→L (x i−1 ) for > −1.

This allows us to compute the gradient of the cross-entropy loss with respect to the scaling factor at = 0 as DISPLAYFORM1 Since the gradient L 2 norm ∂ /∂xi−1 must be greater than the directional derivative DISPLAYFORM2 xi−1 ), y), defining = t / xi−1 we have DISPLAYFORM3

Proof of Theorem 2.

The proof idea is similar.

Recall that if θ ph is a p.h.

set, thenf DISPLAYFORM0 hence we again invoke the directional derivative argument to show DISPLAYFORM1 In order to estimate the scale of this lower bound, recall the FC layer weights are i.i.d.

sampled from a symmetric, mean-zero distribution, therefore z has a symmetric probability density function with mean 0.

We hence have DISPLAYFORM2 where the inequality uses the fact that logsumexp(z) ≥ max i∈[c]

z i ; the last equality is due to y and z being independent at initialization and Ez = 0.

Using the trivial bound EH(p) ≤ log(c), we get DISPLAYFORM3 which shows that the gradient norm of a p.h.

set is of the order Ω(E[max i∈ [c] z i ]) at initialization.

A common theme in previous analysis of residual networks is the scale of activation and gradient BID1 Yang & Schoenholz, 2017; BID10 .

However, it is more important to consider the scale of actual change to the network function made by a (stochastic) gradient descent step.

If the updates to different layers cancel out each other, the network would be stable as a whole despite drastic changes in different layers; if, on the other hand, the updates to different layers align with each other, the whole network may incur a drastic change in one step, even if each layer only changes a tiny amount.

We now provide analysis showing that the latter scenario more accurately describes what happens in reality at initialization.

For our result in this section, we make the following assumptions:• f is a sequential composition of network blocks DISPLAYFORM0 , consisting of fully-connected weight layers, ReLU activation functions and residual branches.• f L is a fully-connected layer with weights i.i.d.

sampled from a zero-mean distribution.• There is no bias parameter in f .For l < L, let x l−1 be the input to f l and F l (x l−1 ) be a branch in f l with m l layers.

Without loss of generality, we study the following specific form of network architecture: DISPLAYFORM1 Furthermore, we always choose 0 as the gradient of ReLU when its input is 0.

As such, with input x, the output and gradient of ReLU (x) l .

We define the following terms to simplify our presentation: DISPLAYFORM2 We have the following result on the gradient update to f : Theorem 3.

With the above assumptions, suppose we update the network parameters by ∆θ = −η ∂ ∂θ (f (x 0 ; θ), y), then the update to network output ∆f ( DISPLAYFORM3 where z f (x 0 ) ∈ R c is the logits.

Let us discuss the implecation of this result before delving into the proof.

As each J Simply put, to allow the whole network be updated by Θ(η) per step independent of depth, we need to ensure each residual branch contributes only a Θ(η/L) update on average.

Proof.

The first insight to prove our result is to note that conditioning on a specific input x 0 , we can replace each ReLU activation layer by a diagonal matrix and does not change the forward and backward pass.

(In fact, this is valid even after we apply a gradient descent update, as long as the learning rate η > 0 is sufficiently small so that all positive preactivation remains positive.

This observation will be essential for our later analysis.)

We thus have the gradient w.r.t.

the i-th weight layer in the l-th block is ∂ ∂Vec(W DISPLAYFORM4 where ⊗ denotes the Kronecker product.

The second insight is to note that with our assumptions, a network block and its gradient w.r.t.

its input have the following relation: DISPLAYFORM5 We then plug in Equation FORMULA0 to the gradient update ∆θ = −η ∂ ∂θ (f (x 0 ; θ), y), and recalculate the forward pass f (x 0 ; θ +∆θ).

The theorem follows by applying Equation FORMULA0 and a first-order Taylor series expansion in a small neighborhood of η = 0 where f (x 0 ; θ + ∆θ) is smooth w.r.t.

η.

For this section, we focus on the proper initialization of a scalar branch F (x) = ( m i=1 a i )x. We have the following result: Theorem 4.

Assuming ∀i, a i ≥ 0, x = Θ(1) and DISPLAYFORM0 Proof.

We start by calculating the gradient of each parameter: DISPLAYFORM1 and a first-order approximation of ∆F (x): DISPLAYFORM2 where we conveniently abuse some notations by defining DISPLAYFORM3 Denote DISPLAYFORM4 as M and min k a k as A, we have DISPLAYFORM5 and therefore by rearranging Equation FORMULA0 and letting ∆F (x) = Θ(η/L)

we get DISPLAYFORM6 Hence the "only if" part is proved.

For the "if" part, we apply Equation (19) to Equation (17) and observe that by Equation (15) DISPLAYFORM7 The result of this theorem provides useful guidance on how to rescale the standard initialization to achieve the desired update scale for the network function.

C ADDITIONAL EXPERIMENTS

In this section we present the training curves of different architecture designs and initialization schemes.

Specifically, we compare the training accuracy of batch normalization, Fixup, as well as a few ablated options: (1) removing the bias parameters in the network; (2) use 0.1x the suggested initialization scale and no bias parameters; (3) use 10x the suggested initialization scale and no bias parameters; and (4) remove all the residual branches.

The results are shown in FIG7 .

We see that initializing the residual branch layers at a smaller scale (or all zero) slows down learning, whereas training fails when initializing them at a larger scale; we also see the clear benefit of adding bias parameters in the network.

We perform additional experiments to validate our hypothesis that the gap in test error between Fixup and batch normalization is primarily due to overfitting.

To combat overfitting, we use Mixup (Zhang et al., 2017) and Cutout BID4 with default hyperparameters as additional regularization.

On the CIFAR-10 dataset, we perform experiments with WideResNet-40-10 and on SVHN we use WideResNet-16-12 (Zagoruyko & Komodakis, 2016) , all with the default hyperparameters.

We observe in TAB8 that models trained with Fixup and strong regularization are competitive with state-of-the-art methods on CIFAR-10 and SVHN, as well as our baseline with batch normalization.

Figure 5 shows that without additional regularization Fixup fits the training set very well, but overfits significantly.

We see in Figure 6 that Fixup is competitive with networks trained with normalization when the Mixup regularizer is used.

The first use of normalization in neural networks appears in the modeling of biological visual system and dates back at least to Heeger (1992) in neuroscience and to BID25 BID21 in computer vision, where each neuron output is divided by the sum (or norm) of all of the outputs, a module called divisive normalization.

Recent popular normalization methods, such as local response normalization BID18 , batch normalization BID16 and layer normalization BID0 mostly follow this tradition of dividing the neuron activations by their certain summary statistics, often also with the activation mean subtracted.

An exception is weight normalization BID26 , which instead divides the weight parameters by their statistics, specifically the weight norm; weight normalization also adopts the idea of activation normalization for weight initialization.

The recently proposed actnorm BID17 removes the normalization of weight parameters, but still use activation normalization to initialize the affine transformation layers.

<|TLDR|>

@highlight

All you need to train deep residual networks is a good initialization; normalization layers are not necessary.

@highlight

A method is presented for initialization and normalization of deep residual networks. This is based on observations of forward and backward explosion in such networks. The method performance is on par with the best results obtained by other networks with more explicit normalization.

@highlight

The authors propose a novel way to initialize residual networks, which is motivated by the need to avoid exploding/vanishing gradients.

@highlight

Proposes a new initialization method used to train very deep RedNets without using batch-norm.