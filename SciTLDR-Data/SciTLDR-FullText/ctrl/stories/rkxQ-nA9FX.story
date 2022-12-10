Batch Normalization (BN) has become a cornerstone of deep learning across diverse architectures, appearing to help optimization as well as generalization.

While the idea makes intuitive sense, theoretical analysis of its effectiveness has been lacking.

Here theoretical support is provided for one of its conjectured properties, namely, the ability to allow gradient descent to succeed with less tuning of learning rates.

It is shown that even if we fix the learning rate of scale-invariant parameters (e.g., weights of each layer with BN) to a constant (say, 0.3), gradient descent still approaches a stationary point (i.e., a solution where gradient is zero) in the rate of T^{−1/2} in T iterations, asymptotically matching the best bound for gradient descent with well-tuned learning rates.

A similar result with convergence rate T^{−1/4} is also shown for stochastic gradient descent.

Batch Normalization (abbreviated as BatchNorm or BN) (Ioffe & Szegedy, 2015) is one of the most important innovation in deep learning, widely used in modern neural network architectures such as ResNet BID8 , Inception (Szegedy et al., 2017) , and DenseNet (Huang et al., 2017) .

It also inspired a series of other normalization methods (Ulyanov et al., 2016; BID0 Ioffe, 2017; Wu & He, 2018) .BatchNorm consists of standardizing the output of each layer to have zero mean and unit variance.

For a single neuron, if x 1 , . . .

, x B is the original outputs in a mini-batch, then it adds a BatchNorm layer which modifies the outputs to DISPLAYFORM0 where µ = i=1 (x i − µ) 2 are the mean and variance within the minibatch, and γ, β are two learnable parameters.

BN appears to stabilize and speed up training, and improve generalization.

The inventors suggested (Ioffe & Szegedy, 2015) that these benefits derive from the following:1.

By stabilizing layer outputs it reduces a phenomenon called Internal Covariate Shift, whereby the training of a higher layer is continuously undermined or undone by changes in the distribution of its inputs due to parameter changes in previous layers.

, 2.

Making the weights invariant to scaling, appears to reduce the dependence of training on the scale of parameters and enables us to use a higher learning rate;3.

By implictly regularizing the model it improves generalization.

But these three benefits are not fully understood in theory.

Understanding generalization for deep models remains an open problem (with or without BN).

Furthermore, in demonstration that intuition can sometimes mislead, recent experimental results suggest that BN does not reduce internal covariate shift either (Santurkar et al., 2018) , and the authors of that study suggest that the true explanation for BN's effectiveness may lie in a smoothening effect (i.e., lowering of the Hessian norm) on the objective.

Another recent paper (Kohler et al., 2018) tries to quantify the benefits of BN for simple machine learning problems such as regression but does not analyze deep models.

Provable quantification of Effect 2 (learning rates).

Our study consists of quantifying the effect of BN on learning rates.

Ioffe & Szegedy (2015) observed that without BatchNorm, a large learning rate leads to a rapid growth of the parameter scale.

Introducing BatchNorm usually stabilizes the growth of weights and appears to implicitly tune the learning rate so that the effective learning rate adapts during the course of the algorithm.

They explained this intuitively as follows.

After BN the output of a neuron z = BN(w x) is unaffected when the weight w is scaled, i.e., for any scalar c > 0, BN(w x) = BN((cw) x).Taking derivatives one finds that the gradient at cw equals to the gradient at w multiplied by a factor 1/c.

Thus, even though the scale of weight parameters of a linear layer proceeding a BatchNorm no longer means anything to the function represented by the neural network, their growth has an effect of reducing the learning rate.

Our paper considers the following question: Can we rigorously capture the above intuitive behavior?

Theoretical analyses of speed of gradient descent algorithms in nonconvex settings study the number of iterations required for convergence to a stationary point (i.e., where gradient vanishes).

But they need to assume that the learning rate has been set (magically) to a small enough number determined by the smoothness constant of the loss function -which in practice are of course unknown.

With this tuned learning rate, the norm of the gradient reduces asymptotically as T −1/2 in T iterations.

In case of stochastic gradient descent, the reduction is like T −1/4 .

Thus a potential way to quantify the rate-tuning behavior of BN would be to show that even when the learning rate is fixed to a suitable constant, say 0.1, from the start, after introducing BN the convergence to stationary point is asymptotically just as fast (essentially) as it would be with a hand-tuned learning rate required by earlier analyses.

The current paper rigorously establishes such auto-tuning behavior of BN (See below for an important clarification about scale-invariance).We note that a recent paper (Wu et al., 2018) introduced a new algorithm WNgrad that is motivated by BN and provably has the above auto-tuning behavior as well.

That paper did not establish such behavior for BN itself, but it was a clear inspiration for our analysis of BN.Scale-invariant and scale-variant parameters.

The intuition of Ioffe & Szegedy (2015) applies for all scale-invariant parameters, but the actual algorithm also involves other parameters such as γ and β whose scale does matter.

Our analysis partitions the parameters in the neural networks into two groups W (scale-invariant) and g (scale-variant).

The first group, W = {w (1) , . . . , w (m) }, consists of all the parameters whose scales does not affect the loss, i.e., scaling w (i) to cw (i) for any c > 0 does not change the loss (see Definition 2.1 for a formal definition); the second group, g, consists of all other parameters that are not scale-invariant.

In a feedforward neural network with BN added at each layer, the layer weights are all scale-invariant.

This is also true for BN with p normalization strategies (Santurkar et al., 2018; Hoffer et al., 2018) and other normalization layers, such as Weight Normalization (Salimans & Kingma, 2016) , Layer Normalization BID0 ), Group Normalization (Wu & He, 2018 ) (see Table 1 in BID0 for a summary).

In this paper, we show that the scale-invariant parameters do not require rate tuning for lowering the training loss.

To illustrate this, we consider the case in which we set learning rates separately for scale-invariant parameters W and scale-variant parameters g. Under some assumptions on the smoothness of the loss and the boundedness of the noise, we show that 1.

In full-batch gradient descent, if the learning rate for g is set optimally, then no matter how the learning rates for W is set, (W ; g) converges to a first-order stationary point in the rate DISPLAYFORM0 , which asymptotically matches with the convergence rate of gradient descent with optimal choice of learning rates for all parameters (Theorem 3.1); 2.

In stochastic gradient descent, if the learning rate for g is set optimally, then no matter how the learning rate for W is set, (W ; g) converges to a first-order stationary point in the rate O(T −1/4 polylog(T )), which asymptotically matches with the convergence rate of gradient descent with optimal choice of learning rates for all parameters (up to a polylog(T ) factor) (Theorem 4.2).In the usual case where we set a unified learning rate for all parameters, our results imply that we only need to set a learning rate that is suitable for g. This means introducing scale-invariance into neural networks potentially reduces the efforts to tune learning rates, since there are less number of parameters we need to concern in order to guarantee an asymptotically fastest convergence.

In our study, the loss function is assumed to be smooth.

However, BN introduces non-smoothness in extreme cases due to division by zero when the input variance is zero (see equation 1).

Note that the suggested implementation of BN by Ioffe & Szegedy (2015) uses a smoothening constant in the whitening step, but it does not preserve scale-invariance.

In order to avoid this issue, we describe a simple modification of the smoothening that maintains scale-invariance.

Also, our result cannot be applied to neural networks with ReLU, but it is applicable for its smooth approximation softplus BID5 ).We include some experiments in Appendix D, showing that it is indeed the auto-tuning behavior we analysed in this paper empowers BN to have such convergence with arbitrary learning rate for scale-invariant parameters.

In the generalization aspect, a tuned learning rate is still needed for the best test accuracy, and we showed in the experiments that the auto-tuning behavior of BN also leads to a wider range of suitable learning rate for good generalization.

Previous work for understanding Batch Normalization.

Only a few recent works tried to theoretically understand BatchNorm.

Santurkar et al. (2018) was described earlier.

Kohler et al. (2018) aims to find theoretical setting such that training neural networks with BatchNorm is faster than without BatchNorm.

In particular, the authors analyzed three types of shallow neural networks, but rather than consider gradient descent, the authors designed task-specific training methods when discussing neural networks with BatchNorm.

BID1 observes that the higher learning rates enabled by BatchNorm improves generalization.

Convergence of adaptive algorithms.

Our analysis is inspired by the proof for WNGrad (Wu et al., 2018) , where the author analyzed an adaptive algorithm, WNGrad, motivated by Weight Normalization (Salimans & Kingma, 2016) .

Other works analyzing the convergence of adaptive methods are (Ward et al., 2018; Li & Orabona, 2018; Zou & Shen, 2018; Zhou et al., 2018) .Invariance by Batch Normalization.

BID3 proposed to run riemmanian gradient descent on Grassmann manifold G(1, n) since the weight matrix is scaling invariant to the loss function.

Hoffer et al. (2018) observed that the effective stepsize is proportional to ηw wt 2 .

In this section, we introduce our general framework in order to study the benefits of scale-invariance.

Scale-invariance is common in neural networks with BatchNorm.

We formally state the definition of scale-invariance below:Definition 2.1. (Scale-invariance) Let F(w, θ ) be a loss function.

We say that w is a scale-invariant parameter of F if for all c > 0, F(w, θ ) = F(cw, θ ); if w is not scale-invariant, then we say w is a scale-variant parameter of F.We consider the following L-layer "fully-batch-normalized" feedforward network Φ for illustration: DISPLAYFORM0 } is a mini-batch of B pairs of input data and ground-truth label from a data set D. f y is an objective function depending on the label, e.g., f y could be a cross-entropy loss in classification tasks.

W (1) , . . .

, W (L) are weight matrices of each layer.

σ : R → R is a nonlinear activation function which processes its input elementwise (such as ReLU, sigmoid).

Given a batch of inputs z 1 , . . .

, z B ∈ R m , BN(z b ) outputs a vectorz b defined as DISPLAYFORM1 where DISPLAYFORM2 are the mean and variance of z b , γ k and β k are two learnable parameters which rescale and offset the normalized outputs to retain the representation power.

The neural network Φ is thus parameterized by weight matrices W (i) in each layer and learnable parameters γ k , β k in each BN.BN has the property that the output is unchanged when the batch inputs z 1,k , . . .

, z B,k are scaled or shifted simultaneously.

For z b,k = w kx b being the output of a linear layer, it is easy to see that w k is scale-invariant, and thus each row vector of weight matrices DISPLAYFORM3 In convolutional neural networks with BatchNorm, a similar argument can be done.

In particular, each filter of convolutional layer normalized by BN is scale-invariant.

With a general nonlinear activation, other parameters in Φ, the scale and shift parameters γ k and β k in each BN, are scale-variant.

When ReLU or Leaky ReLU (Maas et al., 2013) are used as the activation σ, the vector (γ 1 , . . .

, γ m , β 1 , . . . , β m ) of each BN at layer 1 ≤ i < L (except the last one) is indeed scale-invariant.

This can be deduced by using the the (positive) homogeneity of these two types of activations and noticing that the output of internal activations is processed by a BN in the next layer.

Nevertheless, we are not able to analyse either ReLU or Leaky ReLU activations because we need the loss to be smooth in our analysis.

We can instead analyse smooth activations, such as sigmoid, tanh, softplus BID5 , etc.

Now we introduce our general framework.

Let Φ be a neural network parameterized by θ.

Let D be a dataset, where each data point z ∼ D is associated with a loss function F z (θ) (D can be the set of all possible mini-batches).

We partition the parameters θ into (W ; g), where W = {w (1) , . . .

, w (m) } consisting of parameters that are scale-invariant to all F z , and g contains the remaining parameters.

The goal of training the neural network is to minimize the expected loss over the dataset: DISPLAYFORM0 In order to illustrate the optimization benefits of scaleinvariance, we consider the process of training this neural network by stochastic gradient descent with separate learning rates for W and g: DISPLAYFORM1

Thanks to the scale-invariant properties, the scale of each weight w (i) does not affect loss values.

However, the scale does affect the gradients.

Let V = {v(1) , . . .

, v (m) } be the set of normalized weights, where DISPLAYFORM0 2 .

The following simple lemma can be easily shown: Lemma 2.2 (Implied by Ioffe & Szegedy (2015) ).

For any W and g, DISPLAYFORM1 To make ∇ w (i) F z (W ; g) 2 to be small, one can just scale the weights by a large factor.

Thus there are ways to reduce the norm of the gradient that do not reduce the loss.

For this reason, we define the intrinsic optimization problem for training the neural network.

Instead of optimizing W and g over all possible solutions, we focus on parameters θ in which w DISPLAYFORM2 This does not change our objective, since the scale of W does not affect the loss.

2 = 1 for all i} be the intrinsic domain.

The intrinsic optimization problem is defined as optimizing the original problem in U: DISPLAYFORM0 For {θ t } being a sequence of points for optimizing the original optimization problem, we can define {θ t }, whereθ t = (V t ; g t ), as a sequence of points optimizing the intrinsic optimization problem.

In this paper, we aim to show that training neural network for the original optimization problem by gradient descent can be seen as training by adaptive methods for the intrinsic optimization problem, and it converges to a first-order stationary point in the intrinsic optimization problem with no need for tuning learning rates for W .

We assume F z (W ; g) is defined and twice continuously differentiable at any θ satisfying none of DISPLAYFORM0 2 , we assume that the following bounds on the smoothness: DISPLAYFORM1 In addition, we assume that the noise on the gradient of g in SGD is upper bounded by G g : DISPLAYFORM2 Smoothed version of motivating neural networks.

Note that the neural network Φ illustrated in Section 2.1 does not meet the conditions of the smooothness at all since the loss function could be non-smooth.

We can make some mild modifications to the motivating example to smoothen it 1 :(1).

The activation could be non-smooth.

A possible solution is to use smooth nonlinearities, e.g., sigmoid, tanh, softplus BID5 , etc.

Note that softplus can be seen as a smooth approximation of the most commonly used activation ReLU.(2).

The formula of BN shown in equation 3 may suffer from the problem of division by zero.

To avoid this, the inventors of BN, Ioffe & Szegedy (2015) , add a small smoothening parameter > 0 to the denominator, i.e.,z DISPLAYFORM3 However, when z b,k = w kx b , adding a constant directly breaks the scale-invariance of w k .

We can preserve the scale-invariance by making the smoothening term propositional to w k 2 , i.e., replacing with w k 2 .

By simple linear algebra and letting DISPLAYFORM4 , this smoothed version of BN can also be written as DISPLAYFORM5 Since the variance of inputs is usually large in practice, for small , the effect of the smoothening term is negligible except in extreme cases.

Using the above two modifications, the loss function is already smooth.

However, the scale of scale-variant parameters may be unbounded during training, which could cause the smoothness unbounded.

To avoid this issue, we can either project scale-variant parameters to a bounded set, or use weight decay for those parameters (see Appendix C for a proof for the latter solution).

The following lemma is our key observation.

It establishes a connection between the scale-invariant property and the growth of weight scale, which further implies an automatic decay of learning rates:Lemma 2.4.

For any scale-invariant weight w (i) in the network Ψ, we have: DISPLAYFORM0 (2).

w DISPLAYFORM1 Proof.

Let θ t be all the parameters in θ t other than w (i)t .

Taking derivatives with respect to c for the both sides of F zt (w DISPLAYFORM2 t , so the first proposition follows by taking c = 1.

Applying Pythagorean theorem and Lemma 2.2, the second proposition directly follows.

Using Lemma 2.4, we can show that performing gradient descent for the original problem is equivalent to performing an adaptive gradient method for the intrinsic optimization problem: DISPLAYFORM3 where Π is a projection operator which maps any vector w to w/ w 2 .Remark 2.6.

Wu et al. FORMULA0 noticed that Theorem 2.5 is true for Weight Normalization by direct calculation of gradients.

Inspiring by this, they proposed a new adaptive method called WNGrad.

Our theorem is more general since it holds for any normalization methods as long as it induces scale-invariant properties to the network.

The adaptive update rule derived in our theorem can be seen as WNGrad with projection to unit sphere after each step.

Proof for Theorem 2.5.

Using Lemma 2.2, we have DISPLAYFORM4 which implies the first equation.

The second equation is by Lemma 2.4.While popular adaptive gradient methods such as AdaGrad BID4 , RMSprop (Tieleman

Assumptions on learning rates.

We consider the case that we use fixed learning rates for both W and g, i.e., η w,0 = · · · = η w,T −1 = η w and η g,0 = · · · = η g,T −1 = η g .

We assume that η g is tuned carefully to η g = (1 − c g )/L gg for some constant c g ∈ (0, 1).

For η w , we do not make any assumption, i.e., η w can be set to any positive value.

Theorem 3.1.

Consider the process of training Φ by gradient descent with η g = 2(1 − c g )/L gg and arbitrary η w > 0.

Then Φ converges to a stationary point in the rate of DISPLAYFORM0 where DISPLAYFORM1 This matches the asymptotic convergence rate of GD by BID2 .

The high level idea is to use the decrement of loss function to upper bound the sum of the squared norm of the gradients.

Note that ∇L( DISPLAYFORM0 Thus the core of the proof is to show that the monotone increasing w DISPLAYFORM1 T 2 has an upper bound for all T .

It is shown that for every w (i) , the whole training process can be divided into at most two phases.

In the first phase, the effective learning rate η w / w DISPLAYFORM2 2 is larger than some threshold 1 Ci (defined in Lemma 3.2) and in the second phase it is smaller.

2 is large enough and that the process enters the second phase, then by Lemma 3.2 in each step the loss function L will decrease by DISPLAYFORM0 by Lemma 2.4).

Since L is lower-bounded, we can conclude w DISPLAYFORM1 2 is also bounded .

For the second part, we can also show that by Lemma 3.2 DISPLAYFORM2 Thus we can concludeÕ( DISPLAYFORM3 ) convergence rate of ∇L(θ t ) 2 as follows.

DISPLAYFORM4 The full proof is postponed to Appendix A.

In this section, we analyze the effect related to the scale-invariant properties when training a neural network by stochastic gradient descent.

We use the framework introduced in Section 2.2 and assumptions from Section 2.4.

Assumptions on learning rates.

As usual, we assume that the learning rate for g is chosen carefully and the learning rate for W is chosen rather arbitrarily.

More specifically, we consider the case that the learning rates are chosen as DISPLAYFORM0 We assume that the initial learning rate η g of g is tuned carefully to η g = (1 − c g )/L gg for some constant c g ∈ (0, 1).

Note that this learning rate schedule matches the best known convergence rate O(T −1/4 ) of SGD in the case of smooth non-convex loss functions BID6 .For the learning rates of W , we only assume that 0 ≤ α ≤ 1/2, i.e., the learning rate decays equally as or slower than the optimal SGD learning rate schedule.

η w can be set to any positive value.

Note that this includes the case that we set a fixed learning rate η w,0 = · · · = η w,T −1 = η w for W by taking α = 0.

Remark 4.1.

Note that the auto-tuning behavior induced by scale-invariances always decreases the learning rates.

Thus, if we set α > 1/2, there is no hope to adjust the learning rate to the optimal strategy Θ(t −1/2 ).

Indeed, in this case, the learning rate 1/G t in the intrinsic optimization process decays exactly in the rate ofΘ(t −α ), which is the best possible learning rate can be achieved without increasing the original learning rate.

Theorem 4.2.

Consider the process of training Φ by gradient descent with η w,t = η w · (t + 1) DISPLAYFORM1 gg and η w > 0 is arbitrary.

Then Φ converges to a stationary point in the rate of DISPLAYFORM2 where DISPLAYFORM3 and we see L gg = Ω(1).Note that this matches the asymptotic convergence rate of SGD, within a polylog(T ) factor.

We delay the full proof into Appendix B and give a proof sketch in a simplified setting where there is no g and α ∈ [0, 1 2 ).

We also assume there's only one w i , that is, m = 1 and omit the index i. By Taylor expansion, we have DISPLAYFORM0 We can lower bound the effective learning rate η w,T w T 2 and upper bound the second order term respectively in the following way:(1).

For all 0 ≤ α < 1 2 , the effective learning rate DISPLAYFORM1 .

DISPLAYFORM2 Taking expectation over equation 14 and summing it up, we have DISPLAYFORM3 Plug the above bounds into the above inequality, we complete the proof.

DISPLAYFORM4

In this paper, we studied how scale-invariance in neural networks with BN helps optimization, and showed that (stochastic) gradient descent can achieve the asymptotic best convergence rate without tuning learning rates for scale-invariant parameters.

Our analysis suggests that scale-invariance in nerual networks introduced by BN reduces the efforts for tuning learning rate to fit the training data.

However, our analysis only applies to smooth loss functions.

In modern neural networks, ReLU or Leaky ReLU are often used, which makes the loss non-smooth.

It would have more implications by showing similar results in non-smooth settings.

Also, we only considered gradient descent in this paper.

It can be shown that if we perform (stochastic) gradient descent with momentum, the norm of scale-invariant parameters will also be monotone increasing.

It would be interesting to use it to show similar convergence results for more gradient methods.

By the scale-invariant property of w (i) , we know that L(W ; g) = L(V ; g).

Also, the following identities about derivatives can be easily obtained: DISPLAYFORM0 Thus, the assumptions on the smoothness imply DISPLAYFORM1 Proof for Lemma 3.2.

Using Taylor expansion, we have ∃γ ∈ (0, 1), such that for w DISPLAYFORM2 Note that w DISPLAYFORM3 t , we have DISPLAYFORM4 Thus, DISPLAYFORM5 By the inequality of arithmetic and geometric means, we have DISPLAYFORM6 Taking ∆w DISPLAYFORM7 We can complete the proof by replacing DISPLAYFORM8 Using the assumption on the smoothness, we can show that the gradient with respect to w (i) is essentially bounded: Lemma A.1.

For any W and g, we have DISPLAYFORM9 Proof.

A.1 Fix all the parameters except w (i) .

Then L(W ; g) can be written as a function f (w DISPLAYFORM10 Since f is continuous and S is compact, there must exist v DISPLAYFORM11 min is also a minimum in the entire domain and ∇f (w DISPLAYFORM12 For an arbitrary DISPLAYFORM13 , and h goes along the geodesic from v (i) min to v (i) on the unit sphere S with constant speed.

Let H(τ ) = ∇f (h(τ )).

By Taylor expansion, we have DISPLAYFORM14 where we use the fact that w DISPLAYFORM15 at the third line.

We can bound S DISPLAYFORM16 Combining them together, we have DISPLAYFORM17 Taking sum over all i = 1, . . .

, m and also subtracting G T on the both sides, we have DISPLAYFORM18 where DISPLAYFORM19 T + G T is used at the second line.

Combining the lemmas above together, we can obtain our results.

Proof for Theorem 3.1.

By Lemma A.2, we have DISPLAYFORM20 Thus min 0≤t<T ∇L(V t , g t ) 2 converges in the rate of DISPLAYFORM21

Let F t = σ{z 0 , . . .

, z t−1 } be the filtration, where σ{·} denotes the sigma field.

We use L t := L zt (θ t ), F t := F zt (θ t ) for simplicity.

As usual, we define v DISPLAYFORM0 i .

Let k be the maximum i such that t i exists.

Let t k+1 = T + 1.

Then we know that DISPLAYFORM1 Thus, DISPLAYFORM2 Proof.

Conditioned on F t , by Taylor expansion, we have DISPLAYFORM3 where Q t is DISPLAYFORM4 By the inequality DISPLAYFORM5 Taking this into equation 21 and summing up for all t, we have DISPLAYFORM6 and the right hand side can be expressed as DISPLAYFORM7 Proof for Theorem 4.2.

Combining Lemma B.2 and Lemma B.4, for 0 ≤ α < 1/2, we have DISPLAYFORM8 2 +Õ(log T ).Thus, DISPLAYFORM9 Similarly, for α = 1/2, we have DISPLAYFORM10 2 +Õ(log T ).Thus, DISPLAYFORM11

In this section we prove that the modified version of the motivating neural network does meet the assumptions in Section 2.4.

More specifically, we assume:• We use the network structure Φ in Section 2.1 with the smoothed variant of BN as described in Section 2.4;• The objective f y (·) is twice continuously differentiable, lower bounded by f min and Lipschitz (|f y (ŷ)| ≤ α f );• The activation σ(·) is twice continuously differentiable and Lipschitz (|f y (ŷ)| ≤ α σ );• We add an extra weight decay (L2 regularization) term First, we show that g t (containing all scale and shift parameters in BN) is bounded during the training process.

Then the smoothness follows compactness using Extreme Value Theorem.

We use the following lemma to calculate back propagation: DISPLAYFORM0 Proof.

Letx ∈ R B be the vector wherex b := (w (x b −u))/ w S+ I .

It is easy to see x 2 2 ≤ B. Then DISPLAYFORM1 For x b , we have DISPLAYFORM2 Thus, DISPLAYFORM3 Lemma C.2.

If g 0 2 is bounded by a constant, there exists some constant K such that g t 2 ≤ K.Proof.

Fix a time t in the training process.

Consider the process of back propagation.

Define DISPLAYFORM4 b,k is the output of the k-th neuron in the i-th layer in the b-th data sample in the batch.

By the Lipschitzness of the objective, R L can be bounded by a constant.

If R i can be bounded by a constant, then by the Lipschitzness of σ and Lemma C.1, the gradient of γ and β in layer i can also be bounded by a constant.

Note that DISPLAYFORM5 Thus γ and β in layer i can be bounded by a constant since DISPLAYFORM6 Also Lemma C.1 and the Lipschitzness of σ imply that R i−1 can be bounded if R i and γ in the layer i can be bounded by a constant.

Using a simple induction, we can prove the existence of K for bounding the norm of g t 2 for all time t.

Theorem C.3.

If g 0 2 is bounded by a constant, then Φ satisfies the assumptions in Section 2.4.Proof.

Let C be the set of parameters θ satisfying g ≤ K and w (i) 2 = 1 for all 1 ≤ i ≤ m. By Lemma C.2, C contains the set ofθ associated with the points lying between each pair of θ t and θ t+1 (including the endpoints).It is easy to show that F z (θ) is twice continously differentiable.

Since C is compact, by the Extreme Value Theorem, there must exist such constants L

In this section, we provide experimental evidence showing that the auto rate-tuning behavior does empower BN in the optimization aspect.

We trained a modified version of VGGNet (Simonyan & Zisserman, 2014) on Tensorflow.

This network has 2 × conv64, pooling, 3 × conv128, pooling, 3 × conv256, pooling, 3 × conv512, pooling, 3 × conv512, pooling, fc512, fc10 layers in order.

Each convolutional layer has kernel size 3 × 3 and stride 1.

ReLU is used as the activation function after each convolutional or fullyconnected layer.

We add a BN layer right before each ReLU.

We set = 0 in each BN, since we observed that the network works equally well for being 0 or an small number (such as 10 −3 , the default value in Tensorflow).

We initialize the parameters according to the default configuration in Tensorflow: all the weights are initialized by Glorot uniform initializer BID7 ; β and γ in BN are initialized by 0 and 1, respectively.

In this network, every kernel is scale-invariant, and for every BN layer except the last one, the concatenation of all β and γ parameters in this BN is also scale-invariant.

Only β and γ parameters in the last BN are scale-variant (See Section 2.1).

We consider the training in following two settings:1.

Train the network using the standard SGD (No momentum, learning rate decay, weight decay and dropout);2.

Train the network using Projected SGD (PSGD): at each iteration, one first takes a step proportional to the negative of the gradient calculated in a random batch, and then projects each scale-invariant parameter to the sphere with radius equal to its 2-norm before this iteration, i.e., rescales each scale-invariant parameter so that each maintains its length during training.

Note that the projection in Setting 2 removes the adaptivity of the learning rates in the corresponding intrinsic optimization problem, i.e., Gt in equation 9 remains constant during the training.

Thus, by comparing Setting 1 and Setting 2, we can know whether or not the auto-tuning behavior of BN shown in theory is effective in practice.

The relationship between the training loss and the learning rate.

For learning rate larger than 10, the training loss of PSGD or SGD with BN removed is always either very large or NaN, and thus not invisible in the figure.

Left: The average training loss of the last 5 epochs (averaged across 10 experiments).

In rare cases, the training loss becomes NaN in the experiments for the green curve (SGD, BN removed) with learning rate larger than 10 −0.7 .

We removed such data when taking the average.

Right: The average training loss of each epoch (each curve stands for a single experiment).

The relationship between the test accuracy and the learning rate.

Left: The average test accuracy of the last 5 epochs (averaged across 10 experiments).

Right: The test accuracy after each epoch (each curve stands for a single experiment).

Due to the implementation of Tensorflow, outputing NaN leads to a test accuracy of 10%.

Note that the magenta dotted curve (PSGD, lr=100), red dashed curve (SGD, BN removed, lr=1) and cyan dashed curve (SGD, BN removed, lr=10) are covered by the magenta dashed curve (SGD, BN removed, lr=100).

They all have 10% test accuracy.

As in our theoretical analysis, we consider what will happen if we set two learning rates separately for scale-invariant and scale-variant parameters.

We train the network in either setting with different learning rates ranging from 10 −2 to 10 2 for 100 epochs.

First, we fix the learning rate for scale-variant ones to 0.1, and try different learning rates for scaleinvariant ones.

As shown in FIG3 , for small learning rates (such as 0.1), the training processes of networks in Setting 1 and 2 are very similar.

But for larger learning rates, networks in Setting 1 can still converge to 0 for all the learning rates we tried, while networks in Setting 2 got stuck with relatively large training loss.

This suggests that the auto-tuning behavior of BN does takes effect when the learning rate is large, and it matches with the claimed effect of BN in (Ioffe & Szegedy, 2015) that BN enables us to use a higher learning rate.

Though our theoretical analysis cannot be directly applied to the network we trained due to the non-smoothness of the loss function, the experiment results match with what we expect in our analysis.

Next, we consider the case in which we train the network with a unified learning rate for both scaleinvariant and scale-variant parameters.

We also compare Setting 1 and 2 with the setting in which we train the network with all the BN layers removed using SGD (we call it Setting 3).As shown in FIG4 , the training loss of networks in Setting 1 converges to 0.

On the contrast, the training loss of networks in Setting 2 and 3 fails to converge to 0 when a large learning rate is used, and in some cases the loss diverges to infinity or NaN. This suggests that the auto-tuning behavior of BN has an effective role in the case that a unified learning rate is set for all parameters.

For a fair comparison, we also trained neural networks in Setting 3 with initialization essentially equivalent to the ones with BN.

This is done in the same way as (Krähenbühl et al., 2015; Mishkin & Matas, 2015) and Section 3 of (Salimans & Kingma, 2016): we first randomly initialize the parameters, then feed the first batch into the network and adjust the scaling and bias of each neuron to make its outputs have zero mean and unit variance.

In this way, the loss of the networks converges to 0 when the learning rate is smaller than 10 −2.0 , but for a slightly larger learning rate such as 10 −1.8 , the loss fails to converge to 0, and sometimes even diverges to infinity or NaN. Compared with experimental results in Setting 1, this suggests that the robustness of training brought by BN is independent of the fact that BN changes the effective initialization of parameters.

Despite in Setting 1 the convergence of training loss for different learning rates, the convergence points can be different, which lead to different performances on test data.

In FIG5 , we plot the test accuracy of networks trained in Setting 1 and 2 using different unified learning rates, or separate learning rates with the learning rate for scale-variant parameters fixed to 0.1.

As shown in the FIG5 , the test accuracy of networks in Setting 2 decreases as the learning rate increases over 0.1, while the test accuracy of networks in Setting 1 remains higher than 75%.

The main reason that the network in Setting 2 doesn't perform well is underfitting, i.e. the network in Setting 2 fails to fit the training data well when learning rate is large.

This suggests that the autotuning behavior of BN also benefits generalization since such behavior allows the algorithm to pick learning rates from a wider range while still converging to small test error.

<|TLDR|>

@highlight

We give a theoretical analysis of the ability of batch normalization to automatically tune learning rates, in the context of finding stationary points for a deep learning objective.