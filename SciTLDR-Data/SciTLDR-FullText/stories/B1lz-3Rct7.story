Weight decay is one of the standard tricks in the neural network toolbox, but the reasons for its regularization effect are poorly understood, and recent results have cast doubt on the traditional interpretation in terms of $L_2$ regularization.

Literal weight decay has been shown to outperform $L_2$ regularization for optimizers for which they differ.

We empirically investigate weight decay for three optimization algorithms (SGD, Adam, and K-FAC) and a variety of network architectures.

We identify three distinct mechanisms by which weight decay exerts a regularization effect, depending on the particular optimization algorithm and architecture: (1) increasing the effective learning rate, (2) approximately regularizing the input-output Jacobian norm, and (3) reducing the effective damping coefficient for second-order optimization.

Our results provide insight into how to improve the regularization of neural networks.

Weight decay has long been a standard trick to improve the generalization performance of neural networks (Krogh & Hertz, 1992; Bos & Chug, 1996) by encouraging the weights to be small in magnitude.

It is widely interpreted as a form of L 2 regularization because it can be derived from the gradient of the L 2 norm of the weights in the gradient descent setting.

However, several findings cast doubt on this interpretation:• Weight decay has sometimes been observed to improve training accuracy, not just generalization performance (e.g. Krizhevsky et al. (2012) ).•

Loshchilov & Hutter (2017) found that when using Adam (Kingma & Ba, 2014) as the optimizer, literally applying weight decay (i.e. scaling the weights by a factor less than 1 in each iteration) enabled far better generalization than adding an L 2 regularizer to the training objective.• Weight decay is widely used in networks with Batch Normalization (BN) (Ioffe & Szegedy, 2015) .

In principle, weight decay regularization should have no effect in this case, since one can scale the weights by a small factor without changing the network's predictions.

Hence, it does not meaningfully constrain the network's capacity.

The effect of weight decay remains poorly understood, and we lack clear guidelines for which tasks and architectures it is likely to help or hurt.

A better understanding of the role of weight decay would help us design more efficient and robust neural network architectures.

In order to better understand the effect of weight decay, we experimented with both weight decay and L 2 regularization applied to image classifiers using three different optimization algorithms: SGD, Adam, and Kronecker-Factored Approximate Curvature (K-FAC) BID1 .

Consistent with the observations of Loshchilov & Hutter (2017), we found that weight decay consistently outperformed L 2 regularization in cases where they differ.

Weight decay gave an especially strong performance boost to the K-FAC optimizer, and closed most of the generalization gaps between first-and second-order optimizers, as well as between small and large batches.

We then investigated the reasons for weight decay's performance boost.

Surprisingly, we identified three distinct mechanisms by which weight decay has a regularizing effect, depending on the particular algorithm and architecture: Comparison of test accuracy of the networks trained with different optimizers on both CIFAR10 and CIFAR100.

We compare Weight Decay regularization to L2 regularization and the Baseline (which used neither).

Here, BN+Aug denotes the use of BN and data augmentation.

K-FAC-G and K-FAC-F denote K-FAC using Gauss-Newton and Fisher matrices as the preconditioner, respectively.

The results suggest that weight decay leads to improved performance across different optimizers and settings.1.

In our experiments with first-order optimization methods (SGD and Adam) on networks with BN, we found that it acts by way of the effective learning rate.

Specifically, weight decay reduces the scale of the weights, increasing the effective learning rate, thereby increasing the regularization effect of gradient noise BID2 Keskar et al., 2016) .

As evidence, we found that almost all of the regularization effect of weight decay was due to applying it to layers with BN (for which weight decay is meaningless).

Furthermore, when we computed the effective learning rate for the network with weight decay, and applied the same effective learning rate to a network without weight decay, this captured the full regularization effect.

2.

We show that when K-FAC is applied to a linear network using the Gauss-Newton metric (K-FAC-G), weight decay is equivalent to regularizing the squared Frobenius norm of the input-output Jacobian (which was shown by BID3 to improve generalization).

Empirically, we found that even for (nonlinear) classification networks, the Gauss-Newton norm (which K-FAC with weight decay is implicitly regularizing) is highly correlated with the Jacobian norm, and that K-FAC with weight decay significantly reduces the Jacobian norm.

3.

Because the idealized, undamped version of K-FAC is invariant to affine reparameterizations, the implicit learning rate effect described above should not apply.

However, in practice the approximate curvature matrix is damped by adding a multiple of the identity matrix, and this damping is not scale-invariant.

We show that without weight decay, the weights grow large, causing the effective damping term to increase.

If the effective damping term grows large enough to dominate the curvature term, it effectively turns K-FAC into a first-order optimizer.

Weight decay keeps the effective damping term small, enabling K-FAC to retain its second-order properties, and hence improving generalization.

Hence, we have identified three distinct mechanisms by which weight decay improves generalization, depending on the optimization algorithm and network architecture.

Our results underscore the subtlety and complexity of neural network training: the final performance numbers obscure a variety of complex interactions between phenomena.

While more analysis and experimentation is needed to understand how broadly each of our three mechanisms applies (and to find additional mechanisms!), our work provides a starting point for understanding practical regularization effects in neural network training.

Supervised learning.

Given a training set S consisting of training pairs {x, y}, and a neural network f θ (x) with parameters θ (including weights and biases), our goal is to minimize the emprical risk expressed as an average of a loss over the training set: DISPLAYFORM0 To minimize the empirical risk L(θ), stochastic gradient descent (SGD) is used extensively in deep learning community.

Typically, gradient descent methods can be derived from the framework of steepest descent with respect to standard Euclidean metric in parameter space.

Specifically, gradient descent minimizes the following surrogate objective in each iteration: DISPLAYFORM1 where the distance (or dissimilarity) function D(θ, θ + ∆θ) is chosen as 1 2 ∆θ 2 2 .

In this case, solving equation 1 yields ∆θ = −η∇ θ L(θ), where η is the learning rate.

Natural gradient.

Though popular, gradient descent methods often struggle to navigate "valleys" in the loss surface with ill-conditioned curvature (Martens, 2010) .

Natural gradient descent, as a variant of second-order methods (Martens, 2014) , is able to make more progress per iteration by taking into account the curvature information.

One way to motivate natural gradient descent is to show that it can be derived by adapting steepest descent formulation, much like gradient descnet, except using an alternative local distance.

The distance function which leads to natural gradient is the KL divergence on the model's predictive distribution D KL (p θ p θ+∆θ ) ≈ 1 2 ∆θ F∆θ, where F(θ) is the Fisher information matrix 1 (Amari, 1998): DISPLAYFORM2 Applying this distance function to equation 1, we have DISPLAYFORM3 Gauss-Newton algorithm.

Another sensible distance function in equation 1 is the L 2 distance on the output (logits) of the neural network, i.e. .

This leads to the classical Gauss-Newton algorithm which updates the parameters by DISPLAYFORM4 , where the Gauss-Newton (GN) matrix is defined as DISPLAYFORM5 and J θ is the Jacobian of f θ (x) w.r.t θ.

The Gauss-Newton algorithm, much like natural gradient descent, is also invariant to the specific parameterization of neural network function f θ .Two curvature matrices.

It has been shown that the GN matrix is equivalent to the Fisher matrix in the case of regression task with squared error loss (Heskes, 2000) .

However, they are not identical for the case of classification, where cross-entropy loss is commonly used.

Nevertheless, Martens (2014) showed that the Fisher matrix is equivalent to generalized GN matrix when model prediction p(y|x, θ) corresponds to exponential family model with natural parameters given by f θ (x), where the generalized GN matrix is given by DISPLAYFORM6 and H is the Hessian of (y, z) w.r.t z, evaluated at z = f θ (x).

In regression with squared error loss, the Hessian H happens to be identity matrix.

Preconditioned gradient descent.

Given the fact that both natural gradient descent and GaussNewton algorithm precondition the gradient with an extra curvature matrix C(θ) (including the Fisher matrix and GN matrix), we also term them preconditioned gradient descent for convenience.

As modern neural networks may contain millions of parameters, computing and storing the exact curvature matrix and its inverse is impractical.

Kronecker-factored approximate curvature (K-FAC) BID1 uses a Kronecker-factored approximation to the curvature matrix to perform efficient approximate natural gradient updates.

As shown by Luk & Grosse (2018) , K-FAC can be applied to general pullback metric, including Fisher metric and the Gauss-Newton metric.

For more details, we refer reader to Appendix F or BID1 .Batch Normalization.

Broadly speaking, Batch Normalization (BN) is a mechanism that aims to stabilize the distribution (over a mini-batch) of inputs to a given network layer during training.

This is achieved by augmenting the network with additional layers that subtract the mean µ and divide by the standard deviation σ.

Typically, the normalized inputs are also scaled and shifted based on trainable parameters γ and β: DISPLAYFORM0 For clarity, we ignore the parameters γ and β, which do not impact the performance in practice.

This is not surprising, since with ReLU activations, only the γ of the last layer affects network's outputs which can be merged with the softmax layer weights (as also pointed out by van Laarhoven (2017)).

Our goal is to understand weight decay regularization in the context of training deep neural networks.

Towards this, we first discuss the relationship between L 2 regularization and weight decay in different optimizers.

Gradient descent with weight decay is defined by the following update rule: DISPLAYFORM0 , where β defines the rate of the weight decay per step and η is the learning rate.

In this case, weight decay is equivalent to L 2 regularization.

However, the two differ when the gradient update is preconditioned by a matrix C −1 , as in Adam or K-FAC.

The preconditioned gradient descent update with L 2 regularization is given by DISPLAYFORM1 whereas the weight decay update is given by DISPLAYFORM2 The difference between these updates is whether the preconditioner is applied to θ t .

The latter update can be interpreted as the preconditioned gradient descent update on a regularized objective where the regularizer is the squared C-norm θ 2 C = θ Cθ.

If C is adapted based on statistics collected during training, as in Adam or K-FAC, this interpretation holds only approximately because gradient descent on θ 2 C would require differentiating through C. However, this approximate regularization term can still yield insight into the behavior of weight decay.

(As we discuss later, this observation informs some, but not all, of the empirical phenomena we have observed.)

Though the difference between the two updates may appear subtle, we find that it makes a substantial difference in terms of generalization performance.

Initial Experiments.

We now present some empirical findings about the effectiveness of weight decay which the rest of the paper is devoted to explaining.

Our experiments were carried out on two different datasets: CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton, 2009) with varied batch sizes.

We test VGG16 BID5 and ResNet32 (He et al., 2016) on both CIFAR-10 and CIFAR-100 (for more details, see Appendix A).

In particular, we investigate three different optimization algorithms: SGD, Adam and K-FAC.

We consider two versions of K-FAC, which use the Gauss-Newton matrix (K-FAC-G) and Fisher information matrix (K-FAC-F).

FIG0 shows the comparison between weight decay, L 2 regularization and the baseline.

We also compare weight decay to the baseline on more settings and report the final test accuracies in TAB0 .

Finally, the results for large-batch training are summarized in TAB3 .

Based on these results, we make the following observations regarding weight decay:1.

In all experiments, weight decay regularization consistently improved the performance and was more effective than L 2 regularization in cases where they differ (See FIG0 ).

2.

Weight decay closed most of the generalization gaps between first-and second-order optimizers, as well as between small and large batches (See TAB0 ).

3.

Weight decay significantly improved performance even for BN networks (See TAB0 ), where it does not meaningfully constrain the networks' capacity.

4. Finally, we notice that weight decay gave an especially strong performance boost to the K-FAC optimizer when BN was disabled (see the first and fourth rows in TAB0 ).In the following section, we seek to explain these phenomena.

With further testing, we find that weight decay can work in unexpected ways, especially in the presence of BN.

Test accuracy as a function of training epoch for SGD and Adam on CIFAR-100 with different weight decay regularization schemes.

baseline is the model without weight decay; wd-conv is the model with weight decay applied to all convolutional layers; wd-all is the model with weight decay applied to all layers; wd-fc is the model with weight decay applied to the last layer (fc).

Most of the generalization effect of weight decay is due to applying it to layers with BN.

As discussed in Section 3, when SGD is used as the optimizer, weight decay can be interpreted as penalizing the L 2 norm of the weights.

Classically, this was believed to constrain the model by penalizing explanations with large weight norm.

However, for a network with Batch Normalization (BN), an L 2 penalty does not meaningfully constrain the reprsentation, because the network's predictions are invariant to rescaling of the weights and biases.

More precisely, if BN(x; θ l ) denotes the output of a layer with parameters θ l in which BN is applied before the activation function, then DISPLAYFORM0 for any α > 0.

By choosing small α, one can make the L 2 norm arbitrarily small without changing the function computed by the network.

Hence, in principle, adding weight decay to layers with BN should have no effect on the optimal solution.

But empirically, weight decay appears to significantly improve generalization for BN networks (e.g. see FIG0 )

.van Laarhoven (2017) observed that L 2 regularization has an influence on the effective learning rate in (stochastic) gradient descent.

In this work, we extend this result to first-order optimizers (including SGD and Adam) that weight decay increases the effective learning rate by reducing the scale of the weights.

Since higher learning rates lead to larger gradient noise, which has been shown to act as a stochastic regularizer BID2 Keskar et al., 2016; Jastrzębski et al., 2017; Hoffer et al., 2017) , this means weight decay can indirectly exert a regularizing effect through the effective learning rate.

In this section, we provide additional evidence supporting the hypothesis of van Laarhoven (2017).

For simplicity, this section focuses on SGD, but we've observed similar behavior when Adam is used as the optimizer.

Due to its invariance to the scaling of the weights, the key property of the weight vector is its direction.

As shown by Hoffer et al. (2018) , the weight directionθ l = θ l / θ l 2 is updated according tô DISPLAYFORM1 Therefore, the effective learning rate is approximately proportional to η/ θ l 2 2 .

Which means that by decreasing the scale of the weights, weight decay regularization increases the effective learning rate.

FIG3 shows the effective learning rate over time for two BN networks trained with SGD (the results for Adam are similar), one with weight decay and one without it.

Each network is trained with a typical learning rate decay schedule, including 3 factor-of-10 reductions in the learning rate parameter, spaced 60 epochs apart.

Without weight decay, the normalization effects cause an additional effective learning rate decay (due to the increase of weight norm), which reduces the effective learning rate by a factor of 10 over the first 50 epochs.

By contrast, when weight decay is applied, the effective learning rate remains more or less constant in each stage.

We now show that the effective learning rate schedule explains nearly the entire generalization effect of weight decay.

First, we independently varied whether weight decay was applied to the top layer of the network, and to the remaining layers.

Since all layers except the top one used BN, it's only in the top layer that weight decay would constrain the model.

Training curves for SGD and Adam under all four conditions are shown in FIG2 .

In all cases, we observe that whether weight decay was applied to the top (fully connected) layer did not have a significant impact; whether it was applied to the reamining (convolution) layers explained most of the generalization effect.

This supports the effective learning rate hypothesis.

We further tested this hypothesis using a simple experimental manipulation.

Specifically, we trained a BN network without weight decay, but after each epoch, rescaled the weights in each layer to match that layer's norm from the corresponding epoch for the network with weight decay.

This rescaling does not affect the network's predictions, and is equivalent to setting the effective learning rate to match the second network.

As shown in FIG4 , this effective learning rate transfer scheme (wn-conv) eliminates almost the entire generalization gap; it is fully closed by also adding weight decay to the top layer (wd-fc+wn-conv).

Hence, we conclude that for BN networks trained with SGD or Adam, weight decay achieves its regularization effect primarily through the effective learning rate.

In Section 3, we observed that when BN is disabled, weight decay has the strongest regularization effect when K-FAC is used as the optimizer.

Hence, in this section we analyze the effect of weight decay for K-FAC with networks without BN.

First, we show that in a certain idealized setting, K-FAC with weight decay regularizes the input-output Jacobian of the network.

We then empirically investigate whether it behaves similarly for practical networks.

As discussed in Section 3, when the gradient updates are preconditioned by a matrix C, weight decay can be viewed as approximate preconditioned gradient descent on the norm θ 2 C = θ Cθ.

This interpretation is only approximate because the exact gradient update requires differentiating through C.2 When C is taken to be the (exact) Gauss-Newton (GN) matrix G, we obtain the Gauss-Newton norm θ 2 G = θ G(θ)θ.

Similarly, when C is taken to be the K-FAC approximation to G, we obtain what we term the K-FAC Gauss-Newton norm.

These norms are interesting from a regularization perspective.

First, under certain conditions, they are proportional to the average L 2 norm of the network's outputs.

Hence, the regularizer ought to make the network's predictions less extreme.

This is summarized by the following results:Lemma 1 (Gradient structure).

For a feed-forward neural network of depth L with ReLU activation function and no biases, the network's outputs are related to the input-output Jacobian and parameteroutput Jacobian as follows: DISPLAYFORM0 Lemma 2 (Gauss-Newton Norm).

Under the same assumptions of Lemma 1, we observe: DISPLAYFORM1 If we further restrict the network to be a deep linear neural network, we have K-FAC Gauss-Newton norm as follows: Using these results, we show that for linear networks 3 with whitened inputs, the (K-FAC) GaussNewton norm is proportional to the squared Frobenius norm of the input-output Jacobian.

This is interesting from a regularization perspective, since BID3 found the norm of the input-output Jacobian to be consistently coupled to generalization performance.

Theorem 1 (Approximate Jacobian norm).

For a deep linear network of depth L without biases, if we further assume that E[x] = 0 and Cov(x) = I, then: DISPLAYFORM2 DISPLAYFORM3 and θ DISPLAYFORM4 Proof.

It follows from Lemma 2 that θ DISPLAYFORM5 When the network is linear, the input-output Jacobian J x is independent of the input x.

Then we use the assumption of whitened inputs: DISPLAYFORM6 Frob .

The proof for K-FAC Gauss-Newton norm follows immediately with equation 12.While the equivalence between the (K-FAC) GN norm and the Jacobian norm holds only for linear networks, we note that linear networks have been useful for understanding the dynamics of neural net training more broadly (e.g. BID4 ).

Hence, Jacobian regularization may help inform our understanding of weight decay in practical (nonlinear) networks.

To test whether the K-FAC GN norm correlates with the Jacobian norm for practical networks, we trained feed-forward networks with a variety optimizers on both MNIST (LeCun et al., 1998) and CIFAR-10.

For MNIST, we used simple fully-connected networks with different depth and width.

For CIFAR-10, we adopted the VGG family (From VGG11 to VGG19).

We defined the generalization gap to be the difference between training and test loss.

FIG5 shows the relationship of the Jacobian norm to the K-FAC GN norm and to generalization gap for these networks.

We observe that the Jacobian norm correlates strongly with the generalization gap (consistent with BID3 ) and also with the K-FAC GN norm.

Hence, Theorem 1 can inform the regularization of nonlinear networks.

To test if K-FAC with weight decay reduces the Jacobian norm, we compared the Jacobian norms at the end of training for networks with and without weight decay.

As shown in TAB1 , weight decay reduced the Jacboian norm by a much larger factor when K-FAC was used as the optimizer than when SGD was used as the optimizer. : Test accuracy as a function of training epoch for K-FAC on CIFAR-100 with different weight decay regularization schemes.

baseline is the model without weight decay regularization; wd-conv is the model with weight decay applied to all convolutional layers; wd-all is the model with weight decay applied to all layers; wd-fc is the model with weight decay applied to the last layer (fc).

Consistent with the Jacobian regularization hypothesis, applying weight decay to the non-BN layers have the largest regularization effect.

However, applying weight decay to the BN layers also lead to noticeable gains.

Our discussion so far as focused on the GN version of K-FAC.

Recall that, in many cases, the Fisher information matrix differs from the GN matrix only in that it accounts for the output layer Hessian.

Hence, this analysis may help inform the behavior of K-FAC-F as well.

We also note that θ 2 F , the Fisher-Rao norm, has been proposed as a complexity measure for neural networks (Liang et al., 2017) .

Hence, unlike in the case of SGD and Adam for BN networks, we interpret K-FAC with weight decay as constraining the capacity of the network.

We now return our attention to the setting of architectures with BN.

The Jacobian regularization mechanism from Section 4.2 does not apply in this case, since rescaling the weights results in an equivalent network, and therefore does not affect the input-output Jacobian.

Similarly, if the network is trained with K-FAC, then the effective learning rate mechanism from Section 4.1 also does not apply because the K-FAC update is invariant to affine reparameterization (Luk & Grosse, 2018) and therefore not affected by the scaling of the weights.

More precisely, for a layer with BN, the curvature matrix C (either the Fisher matrix or the GN matrix) has the following property: DISPLAYFORM0 whereθ l = θ l / θ l 2 as in Section 4.1.

Hence, the θ l 2 2 factor in the preconditioner counteracts the θ l −2 2 factor in the effective learning rate, resulting in an equivlaent effective learning rate regardless of the norm of the weights.

These observations raise the question of whether it is still useful to apply weight decay to BN layers when using K-FAC.

To answer this question, we repeated the experiments in FIG2 (applying weight decay to subsets of the layers), but with K-FAC as the optimizer.

The results are summarized in FIG6 .

Applying it to the non-BN layers had the largest effect, consistent with the Jacobian regularization hypothesis.

However, applying weight decay to the BN layers also led to significant gains, especially for K-FAC-F.The reason this does not contradict the K-FAC invariance property is that practical K-FAC implementations dampen the updates (like many second-order optimziers) by adding a multiple of the identity matrix to the curvature before inversion.

According to equation 15, as the norm of the weights gets larger, C gets smaller, and hence the damping term comes to dominate the preconditioner.

Mathematically, we can understand this effect by deriving the following update rule for the normalized weightŝ θ (see Appendix D for proof): DISPLAYFORM1 where λ is the damping parameter.

Hence, for large C(θ l ) or small θ l , the update is close to the idealized second-order update, while for small enough C(θ l ) or large enough θ l , K-FAC effectively becomes a first-order optimizer.

Hence, by keeping the weights small, weight decay helps K-FAC to retain its second-order properties.

Most implementations of K-FAC keep the damping parameter λ fixed throughout training.

Therefore, it would be convenient if C(θ l ) and θ l do not change too much during training, so that a single value of λ can work well throughout training.

Interestingly, the norm of the GN matrix appears to be much more stable than the norm of the Fisher matrix.

FIG7 shows the norms of the Fisher matrix F(θ l ) and GN matrix G(θ l ) of the normalized weights for the first layer of a CIFAR-10 network throughout training.

While the norm of F(θ l ) decays by 4 orders of magnitude over the first 50 epochs, the norm of G(θ l ) increases by only a factor of 2.The explanation for this is as follows: in a classification task with cross-entropy loss, the Fisher matrix is equivalent to the generalized GN matrix E[J θ H J θ ] (see Section 2).

This differs from the GN matrix E[J θ J θ ] only in that it incudes the output layer Hessian H = diag(p) − pp , where p is the vector of estimated class probabilities.

It is easy to see that H goes to zero as p collapses to one class, as is the case for tasks such as CIFAR-10 and CIFAR-100 where networks typically achieve perfect training accuracy.

Hence, we would expect F to get much smaller over the course of training, consistent with FIG7 .To summarize, when K-FAC is applied to BN networks, it can be advantageous to apply weight decay even to layers with BN, even though this appears unnecessary based on invariance considerations.

The reason is that weight decay reduces the effective damping, helping K-FAC to retain its second-order properties.

This effect is stronger for K-FAC-F than for K-FAC-G because the Fisher matrix shrinks dramatically over the course of training.

Despite its long history, weight decay regularization remains poorly understood.

We've identified three distinct mechanisms by which weight decay improves generalization, depending on the architecture and optimization algorithm: increasing the effective learning rate, reducing the Jacobian norm, and reducing the effective damping parameter.

We would not be surprised if there remain additional mechanisms we have not found.

The dynamics of neural net training is incredibly complex, and it can be tempting to simply do what works and not look into why.

But we think it is important to at least sometimes dig deeper to determine exactly why an algorithm has the effect that it does.

Some of our analysis may seem mundane, or even tedious, as the interactions between different hyperparameters are not commonly seen as a topic worthy of detailed scientific study.

But our experiments highlight that the dynamics of the norms of weights and curvature matrices, and their interaction with optimization hyperparameters, can have a substantial impact on generalization.

We believe these effects deserve more attention, and would not be surprised if they can help explain the apparent success or failure of other neural net design choices.

We also believe our results highlight the need for automatic adaptation of optimization hyperparameters, to eliminate potential experimental confounds and to allow researchers and practitioners to focus on higher level design issues.

Jimmy Ba, Roger Grosse, and James Martens.

Distributed second-order optimization using kroneckerfactored approximations.

2016.

Throughout the paper, we perform experiments on image classification with three different datasets, MNIST (LeCun et al., 1998), CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton, 2009) .

For MNIST, we use simple fully-connected networks with different depth and width.

For CIFAR-10 and CIFAR-100, we use VGG16 BID5 BID5 ResNet32 (He et al., 2016) .

To make the network more flexible, we widen all convolutional layers in ResNet32 by a factor of 4, according to BID8 .We investigate three different optimization methods, including Stochastic Gradient Descent (SGD), Adam (Kingma & Ba, 2014) and K-FAC BID1 .

In K-FAC, two different curvature matrices are studied, including Fisher information matrix and Gauss-Newton matrix.

In default, batch size 128 is used unless stated otherwise.

In SGD and Adam, we train the networks with a budge of 200 epochs and decay the learning rate by a factor of 10 every 60 epochs for batch sizes of 128 and 640, and every 80 epochs for the batch size of 2K.

Whereas we train the networks only with 100 epochs and decay the learning rate every 40 epochs in K-FAC.

Additionally, the curvature matrix is updated by running average with re-estimation every 10 iterations and the inverse operator is amortized to 100 iterations.

For K-FAC, we use fixed damping term 1e −3 unless state otherwise.

For each algorithm, best hyperparameters (learning rate and regularization factor) are selected using grid search on held-out 5k validation set.

For the large batch setting, we adopt the same strategies in Hoffer et al. (2017) for adjusting the search range of hyperparameters.

Finally, we retrain the model with both training data and validation data.

Claim.

For a feed-forward neural network of depth L with ReLU activation function and no biases, one has the following property: DISPLAYFORM0 The key observation of Lemma 1 is that rectified neural networks are piecewise linear up to the output f θ (x).

And ReLU activation function satisfies the property σ(z) = σ (z)z.

Summing over all the layers, we conclude the following equation eventually: DISPLAYFORM1 Claim.

For a feed-forward neural network of depth L with ReLU activation function and no biases, we observe: DISPLAYFORM2 Furthermore, if we restrict the network to be linear with only fully-connected layers, we have K-FAC Gauss-Newton norm as follows DISPLAYFORM3 Proof.

We first prove the equaility θ DISPLAYFORM4 Using the definition of the Gauss-Newton norm in equation 3, we have DISPLAYFORM5 Combining above equalities, we arrive at the conclusion.

DISPLAYFORM6 2 , we note that kronecker-product is exact under the condition that the network is linear (Bernacchia et al., 2018) , which means G K−FAC is the diagonal block version of Gauss-Newton matrix G. Therefore, we have DISPLAYFORM7 2 , therefore we conclude that DISPLAYFORM8 Claim.

During training, the weight directionθ DISPLAYFORM9 Proof.

Natural gradient update is given by DISPLAYFORM10 Denote ρ t = θ t 2 .

Then we have DISPLAYFORM11 and therefore DISPLAYFORM12 Additionally, we can rewrite the natural gradient update as follows DISPLAYFORM13 And therefore, DISPLAYFORM14 , its gradient has the following form: DISPLAYFORM15 According to Lemma 1, we have f θ (x) = 1 L+1 J θ θ, therefore we can rewrite equation 17 DISPLAYFORM16 Surprisingly, the resulting gradient has the same form as the case where we take Gauss-Newton matrix as a constant of θ up to a constant (L + 1).F KRONECKER-FACTORED APPROXIMATE CURVATURE (K-FAC) BID1 proposed K-FAC for performing efficient natural gradient optimization in deep neural networks.

Following on that work, K-FAC has been adopted in many tasks BID7 BID9 to gain optimization benefits, and was shown to be amendable to distributed computation (Ba et al., 2016) .

DISPLAYFORM17 As shown by Luk & Grosse (2018), K-FAC can be applied to general pullback metric, including Fisher metric and the Gauss-Newton metric.

For convenience, we introduce K-FAC here using the Fisher metric.

Considering l-th layer in the neural network whose input activations are a l ∈ R n1 , weight W l ∈ R n1×n2 , and output s l ∈ R n2 , we have DISPLAYFORM18 .

With this gradient formula, K-FAC decouples this layer's fisher matrix F l using mild approximations, DISPLAYFORM19 Where A l = E aa and S l = E {∇ s L}{∇ s L} .

The approximation above assumes independence between a and s, which proves to be accurate in practice.

Further, assuming between-layer independence, the whole fisher matrix F can be approximated as block diagonal consisting of layerwise fisher matrices F l .

Decoupling F l into A l and S l not only avoids the memory issue saving F l , but also provides efficient natural gradient computation.

DISPLAYFORM20 As shown by equation 20, computing natural gradient using K-FAC only consists of matrix transformations comparable to size of W l , making it very efficient.

Algorithm 1 K-FAC with L 2 regularization and K-FAC with weight decay.

Subscript l denotes layers, w l = vec(W l ).

We assume zero momentum for simplicity.

Require: η: stepsize Require: β: weight decay Require: stats and inverse update intervals T stats and T inv k ← 0 and initialize DISPLAYFORM0 while stopping criterion not met do DISPLAYFORM1

It has been shown that K-FAC scales very favorably to larger mini-batches compared to SGD, enjoying a nearly linear relationship between mini-batch size and per-iteration progress for medium-to-large sized mini-batches BID1 Ba et al., 2016) .

However, Keskar et al. (2016) showed that large-batch methods converge to sharp minima and generalize worse.

In this subsection, we measure the generalization performance of K-FAC with large batch training and analyze the effect of weight decay.

In TAB3 , we compare K-FAC with SGD using different batch sizes.

In particular, we interpolate between small-batch (BS128) and large-batch (BS2000).

We can see that in accordance with previous works (Keskar et al., 2016; Hoffer et al., 2017 ) the move from a small-batch to a large-batch indeed incurs a substantial generalization gap.

However, adding weight decay regularization to K-FAC almost close the gap on CIFAR-10 and cause much of the gap diminish on CIFAR-100.

Surprisingly, the generalization gap of SGD also disappears with well-tuned weight decay regularization.

Moreover, we observe that the training loss cannot decrease to zero if weight decay is not used, indicating weight decay may also speed up the training.

Figure 8: Test accuracy as a function of training epoch.

We plot baseline vs L2 regularization vs weight decay regularization on CIFAR-10 and CIFAR-100 datasets.

The '+' denotes with BN and data augmentation.

Note that training accuracies of all the models are 100% in the end of the training.

We smooth all the curves for visual clarity.

While this paper mostly focus on generalization, we also report the convergence speed of different optimizers in deep neural networks; we report both per-epoch performance and wall-clock time performance.

We consider the task of image classification on CIFAR-10 (Krizhevsky & Hinton, 2009) dataset.

The models we use consist of VGG16 BID5 and ResNet32 (He et al., 2016) .

We compare our K-FAC-G, K-FAC-F with SGD, Adam (Kingma & Ba, 2014) .

We experiment with constant learning for K-FAC-G and K-FAC-F. For SGD and Adam, we set batch size as 128.

For K-FAC, we use batch size of 640, as suggested by BID1 .In FIG9 , we report the training curves of different algorithms.

FIG9 show that K-FAC-G yields better optimization than other baselines in training loss per epoch.

We highlight that the training loss decreases to 1e-4 within 10 epochs with K-FAC-G. Although K-FAC based algorithms take more time for each epoch, FIG9 still shows wall-clock time improvements over the baselines.

In FIG9 and 9d, we report similar results on the ResNet32.

Note that we make the network wider with a widening factor of 4 according to BID8 .

K-FAC-G outperforms both K-FAC-F and other baselines in term of optimization per epoch, and compute time.

@highlight

We investigate weight decay regularization for different optimizers and identify three distinct mechanisms by which weight decay improves generalization.

@highlight

Discusses the effect of weight decay on the training of deep network models with and without batch normalization and when using first/second order optimization methods and hypothesizes that a larger learning rate has a regularization effect.