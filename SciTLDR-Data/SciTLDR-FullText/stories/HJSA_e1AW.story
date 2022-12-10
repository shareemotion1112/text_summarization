Optimization algorithms for training deep models not only affects the convergence rate and stability of the training process, but are also highly related to the generalization performance of trained models.

While adaptive algorithms, such as Adam and RMSprop, have shown better optimization performance than stochastic gradient descent (SGD) in many scenarios, they often lead to worse generalization performance than SGD, when used for training deep neural networks (DNNs).

In this work, we identify two problems regarding the direction and step size for updating the weight vectors of hidden units, which may degrade the generalization performance of Adam.

As a solution, we propose the normalized direction-preserving Adam (ND-Adam) algorithm, which controls the update direction and step size more precisely, and thus bridges the generalization gap between Adam and SGD.

Following a similar rationale, we further improve the generalization performance in classification tasks by regularizing the softmax logits.

By bridging the gap between SGD and Adam, we also shed some light on why certain optimization algorithms generalize better than others.

In contrast with the growing complexity of neural network architectures BID10 BID12 , the training methods remain relatively simple.

Most practical optimization methods for deep neural networks (DNNs) are based on the stochastic gradient descent (SGD) algorithm.

However, the learning rate of SGD, as a hyperparameter, is often difficult to tune, since the magnitudes of different parameters can vary widely, and adjustment is required throughout the training process.

To tackle this problem, several adaptive variants of SGD have been developed, including Adagrad BID6 ), Adadelta (Zeiler, 2012 , RMSprop BID24 , Adam BID15 , etc.

These algorithms aim to adapt the learning rate to different parameters automatically, based on the statistics of gradient.

Although they usually simplify learning rate settings, and lead to faster convergence, it is observed that their generalization performance tend to be significantly worse than that of SGD in some scenarios BID25 .

This intriguing phenomenon may explain why SGD (possibly with momentum) is still prevalent in training state-of-the-art deep models, especially feedforward DNNs BID10 BID12 .

Furthermore, recent work has shown that DNNs are capable of fitting noise data BID31 , suggesting that their generalization capabilities are not the mere result of DNNs themselves, but are entwined with optimization BID2 .This work aims to bridge the gap between SGD and Adam in terms of the generalization performance.

To this end, we identify two problems that may degrade the generalization performance of Adam, and show how these problems are (partially) avoided by using SGD with L2 weight decay.

First, the updates of SGD lie in the span of historical gradients, whereas it is not the case for Adam.

This difference has been discussed in rather recent literature BID25 , where the authors show that adaptive methods can find drastically different but worse solutions than SGD.

Second, while the magnitudes of Adam parameter updates are invariant to rescaling of the gradient, the effect of the updates on the same overall network function still varies with the magnitudes of parameters.

As a result, the effective learning rates of weight vectors tend to decrease during training, which leads to sharp local minima that do not generalize well BID11 .To fix the two problems for Adam, we propose the normalized direction-preserving Adam (NDAdam) algorithm, which controls the update direction and step size more precisely.

We show that ND-Adam is able to achieve significantly better generalization performance than vanilla Adam, and matches that of SGD in image classification tasks.

We summarize our contributions as follows:• We observe that the directions of Adam parameter updates are different from that of SGD, i.e., Adam does not preserve the directions of gradients as SGD does.

We fix the problem by adapting the learning rate to each weight vector, instead of each individual weight, such that the direction of the gradient is preserved.• For both Adam and SGD without L2 weight decay, we observe that the magnitude of each vector's direction change depends on its L2-norm.

We show that, using SGD with L2 weight decay implicitly normalizes the weight vectors, and thus remove the dependence in an approximate manner.

We fix the problem for Adam by explicitly normalizing each weight vector, and by optimizing only its direction, such that the effective learning rate can be precisely controlled.• We further show that, without proper regularization, the learning signal backpropagated from the softmax layer may vary with the overall magnitude of the logits in an undesirable way.

Based on the observation, we apply batch normalization or L2-regularization to the logits, which further improves the generalization performance in classification tasks.

In essence, our proposed methods, ND-Adam and regularized softmax, improve the generalization performance of Adam by enabling more precise control over the directions of parameter updates, the learning rates, and the learning signals.

Adaptive moment estimation (Adam) BID15 ) is a stochastic optimization method that applies individual adaptive learning rates to different parameters, based on the estimates of the first and second moments of the gradients.

Specifically, for n trainable parameters, θ ∈ R n , Adam maintains a running average of the first and second moments of the gradient w.r.t.

each parameter as DISPLAYFORM0 DISPLAYFORM1 Here, t denotes the time step, m t ∈ R n and v t ∈ R n denote respectively the first and second moments, and β 1 ∈ R and β 2 ∈ R are the corresponding decay factors.

BID15 further notice that, since m 0 and v 0 are initialized to 0's, they are biased towards zero during the initial time steps, especially when the decay factors are large (i.e., close to 1).

Thus, for computing the next update, they need to be corrected aŝ DISPLAYFORM2 where β t 1 , β t 2 are the t-th powers of β 1 , β 2 respectively.

Then, we can update each parameter as DISPLAYFORM3 where α t is the global learning rate, and is a small constant to avoid division by zero.

Note the above computations between vectors are element-wise.

A distinguishing merit of Adam is that the magnitudes of parameter updates are invariant to rescaling of the gradient, as shown by the adaptive learning rate term, αt √v t+ .

However, there are two potential problems when applying Adam to DNNs.

First, in some scenarios, DNNs trained with Adam generalize worse than that trained with stochastic gradient descent (SGD) BID25 .

BID31 demonstrate that overparameterized DNNs are capable of memorizing the entire dataset, no matter if it is natural data or meaningless noise data, and thus suggest much of the generalization power of DNNs comes from the training algorithm, e.g., SGD and its variants.

It coincides with another recent work BID25 , which shows that simple SGD often yields better generalization performance than adaptive gradient methods, such as Adam.

As pointed out by the latter, the difference in the generalization performance may result from the different directions of updates.

Specifically, for each hidden unit, the SGD update of its input weight vector can only lie in the span of all possible input vectors, which, however, is not the case for Adam due to the individually adapted learning rates.

We refer to this problem as the direction missing problem.

Second, while batch normalization BID13 can significantly accelerate the convergence of DNNs, the input weights and the scaling factor of each hidden unit can be scaled in infinitely many (but consistent) ways, without changing the function implemented by the hidden unit.

Thus, for different magnitudes of an input weight vector, the updates given by Adam can have different effects on the overall network function, which is undesirable.

Furthermore, even when batch normalization is not used, a network using linear rectifiers (e.g., ReLU, leaky ReLU) as activation functions, is still subject to ill-conditioning of the parameterization BID8 , and hence the same problem.

We refer to this problem as the ill-conditioning problem.

L2 weight decay is a regularization technique frequently used with SGD.

It often has a significant effect on the generalization performance of DNNs.

Despite the simplicity and crucial role of L2 weight decay in the training process, it remains to be explained how it works in DNNs.

A common justification for L2 weight decay is that it can be introduced by placing a Gaussian prior upon the weights, when the objective is to find the maximum a posteriori (MAP) weights BID3 .

However, as discussed in Sec. 2.1, the magnitudes of input weight vectors are irrelevant in terms of the overall network function, in some common scenarios, rendering the variance of the Gaussian prior meaningless.

We propose to view L2 weight decay in neural networks as a form of weight normalization, which may better explain its effect on the generalization performance.

Consider a neural network trained with the following loss function: DISPLAYFORM0 where L (θ; D) is the original loss function specified by the task, D is a batch of training data, N is the set of all hidden units, and w i denotes the input weights of hidden unit i, which is included in the trainable parameters, θ.

For simplicity, we consider SGD updates without momentum.

Therefore, the update of w i at each time step is DISPLAYFORM1 where α is the learning rate.

As we can see from Eq. (5), the gradient magnitude of the L2 penalty is proportional to w i 2 , thus forms a negative feedback loop that stabilizes w i 2 to an equilibrium value.

Empirically, we find that w i 2 tends to increase or decrease dramatically at the beginning of the training, and then varies mildly within a small range, which indicates w i 2 ≈

w i + ∆w i 2 .

In practice, we usually have ∆w i 2 / w i 2 1, thus ∆w i is approximately orthogonal to w i , i.e. DISPLAYFORM2 Let l wi and l ⊥wi be the vector projection and rejection of ∂L ∂wi on w i , which are defined as DISPLAYFORM3 From Eq. FORMULA5 and FORMULA7 , it is easy to show DISPLAYFORM4 As discussed in Sec. 2.1, when batch normalization is used, or when linear rectifiers are used as activation functions, the magnitude of w i 2 is irrelevant.

Thus, it is the direction of w i that actually makes a difference in the overall network function.

If L2 weight decay is not applied, the magnitude of w i 's direction change will decrease as w i 2 increases during the training process, which can potentially lead to overfitting (discussed in detail in Sec. 3.2).

On the other hand, Eq. FORMULA8 shows that L2 weight decay implicitly normalizes the weights, such that the magnitude of w i 's direction change does not depend on w i 2 , and can be tuned by the product of α and λ.

In the following, we refer to ∆w i 2 / w i 2 as the effective learning rate of w i .While L2 weight decay produces the normalization effect in an implicit and approximate way, we will show that explicitly doing so enables more precise control of the effective learning rate.

We first present the normalized direction-preserving Adam (ND-Adam) algorithm, which essentially improves the optimization of the input weights of hidden units, while employing the vanilla Adam algorithm to update other parameters.

Specifically, we divide the trainable parameters, θ, into two sets, θ v and θ s , such that θ v = {w i |i ∈ N }, and θ s = {θ \ θ v }.

Then we update θ v and θ s by different rules, as described by Alg.

1.

The learning rates for the two sets of parameters are denoted respectively by α

In Alg.

1, the iteration over N can be performed in parallel, and thus introduces no extra computational complexity.

Compared to Adam, computing g t (w i ) and w i,t may take slightly more time, which, however, is negligible in practice.

On the other hand, to estimate the second order moment of each w i ∈ R n , Adam maintains n scalars, whereas ND-Adam requires only one scalar, v t (w i ).

Thus, ND-Adam has smaller memory overhead than Adam.

In the following, we address the direction missing problem and the ill-conditioning problem discussed in Sec. 2.1, and explain Alg.

1 in detail.

We show how the proposed algorithm jointly solves the two problems, as well as its relation to other normalization schemes.

Assuming the stationarity of a hidden unit's input distribution, the SGD update (possibly with momentum) of the input weight vector is a linear combination of historical gradients, and thus can only lie in the span of the input vectors.

As a result, the input weight vector itself will eventually converge to the same subspace.

On the contrary, the Adam algorithm adapts the global learning rate to each scalar parameter independently, such that the gradient of each parameter is normalized by a running average of its magnitudes, which changes the direction of the gradient.

To preserve the direction of the gradient w.r.t.

each input weight vector, we generalize the learning rate adaptation scheme from scalars to vectors.

DISPLAYFORM0 is a linear combination of historical gradients, it can be extended to vectors without any change; or equivalently, we can rewrite it for each vector as DISPLAYFORM1 We then extend Eq. (1b) as DISPLAYFORM2 i.e., instead of estimating the average gradient magnitude for each individual parameter, we estimate the average of g t (w i ) 2 2 for each vector w i .

In addition, we modify Eq. FORMULA2 and FORMULA3 accordingly aŝ DISPLAYFORM3 and DISPLAYFORM4 Here,m t (w i ) is a vector with the same dimension as w i , whereasv t (w i ) is a scalar.

Therefore, when applying Eq. (11), the direction of the update is the negative direction ofm t (w i ), and thus is in the span of the historical gradients of w i .It is worth noting that only the input to the first layer (i.e., the training data) is stationary throughout training.

Thus, for the weights of an upper layer to converge to the span of its input vectors, it is necessary for the lower layers to converge first.

Interestingly, this predicted phenomenon may have been observed in practice BID4 .Despite the empirical success of SGD, a question remains as to why it is desirable to constrain the input weights in the span of the input vectors.

A possible explanation is related to the manifold hypothesis, which suggests that real-world data presented in high dimensional spaces (images, audios, text, etc) concentrates on manifolds of much lower dimensionality BID5 BID18 .

In fact, commonly used activation functions, such as (leaky) ReLU, sigmoid, tanh, can only be activated (not saturating or having small gradients) by a portion of the input vectors, in whose span the input weights lie upon convergence.

Assuming the local linearity of the manifolds of data or hidden-layer representations, constraining the input weights in the subspace that contains some of the input vectors, encourages the hidden units to form local coordinate systems on the corresponding manifold, which can lead to good representations BID21 .

The ill-conditioning problem occurs when the magnitude change of an input weight vector can be compensated by other parameters, such as the scaling factor of batch normalization, or the output weight vector, without affecting the overall network function.

Consequently, suppose we have two DNNs that parameterize the same function, but with some of the input weight vectors having different magnitudes, applying the same SGD or Adam update rule will, in general, change the network functions in different ways.

Thus, the ill-conditioning problem makes the training process inconsistent and difficult to control.

More importantly, when the weights are not properly regularized (e.g., without using L2 weight decay), the magnitude of w i 's direction change will decrease as w i 2 increases during the training process.

As a result, the effective learning rate for w i tends to decrease faster than expected, making the network converge to sharp minima.

It is well known that sharp minima generalize worse than flat minima BID11 BID14 .As shown in Sec. 2.2, L2 weight decay can alleviate the ill-conditioning problem by implicitly and approximately normalizing the weights.

However, we still do not have a precise control over the effective learning rate, since l ⊥wi 2 / l wi 2 is unknown and not necessarily stable.

Moreover, the approximation fails when w i 2 is far from the equilibrium due to improper initialization, or drastic changes in the magnitudes of the weight vectors.

This problem is also addressed by BID19 , by employing a geometry invariant to rescaling of weights.

However, their proposed methods do not preserve the direction of gradient.

To address the ill-conditioning problem in a more principled way, we restrict the L2-norm of each w i to 1, and only optimize its direction.

In other words, instead of optimizing w i in a n-dimensional space, we optimize w i on a (n − 1)-dimensional unit sphere.

Specifically, we first obtain the raw gradient w.r.t.

w i ,ḡ t (w i ) = ∂L/∂w i , and project the gradient onto the unit sphere as DISPLAYFORM0 Here, w i,t−1 2 = 1.

Then we follow Eq. FORMULA11 - FORMULA13 , and replace (11) with DISPLAYFORM1 and DISPLAYFORM2 In Eq. FORMULA2 , we keep only the component that is orthogonal to w i,t−1 .

However,m t (w i ) is not necessarily orthogonal as well.

In addition, even whenm t (w i ) is orthogonal to w i,t−1 , Eq. (13a) can still increase w i 2 , according to the Pythagorean theorem.

Therefore, we explicitly normalize w i,t in Eq. (13b), to ensure w i,t 2 = 1 after each update.

Also note that, since w i,t−1 is a linear combination of its historical gradients, g t (w i ) still lies in the span of the historical gradients after the projection in Eq. (12).Compared to SGD with L2 weight decay, spherical weight optimization explicitly normalizes the weight vectors, such that each update to the weight vectors only changes their directions, and strictly keeps the magnitudes constant.

As a result, the effective learning rate of a weight vector is DISPLAYFORM3 which enables precise control over the learning rate of w i through a single hyperparameter, α v t , rather than two as required by Eq. (7).

Note that it is possible to control the effective learning rate more precisely, by normalizingm t (w i ) with m t (w i ) 2 , instead of by v t (w i ).

However, by doing so, we lose the information provided by m t (w i ) 2 at different time steps.

In addition, sincê m t (w i ) is less noisy than g t (w i ), m t (w i ) 2 / v t (w i ) becomes small near convergence, which is considered a desirable property of Adam BID15 .

Thus, we keep the gradient normalization scheme intact.

We note the difference between various gradient normalization schemes and the normalization scheme employed by spherical weight optimization.

As shown in Eq. 11, ND-Adam generalizes the gradient normalization scheme of Adam, and thus both Adam and ND-Adam normalize the gradient by a running average of its magnitude.

This, and other similar schemes BID9 BID27 ) make the optimization less susceptible to vanishing and exploding gradients.

On the other hand, the proposed spherical weight optimization serves a different purpose.

It normalizes each weight vector and projects the gradient onto a unit sphere, such that the effective learning rate can be controlled more precisely.

Moreover, it provides robustness to improper weight initialization, since the magnitude of each weight vector is kept constant.

For nonlinear activation functions, such as sigmoid and tanh, an extra scaling factor is needed for each hidden unit to express functions that require unnormalized weight vectors.

For instance, given an input vector x ∈ R n , and a nonlinearity φ (·), the activation of hidden unit i is then given by DISPLAYFORM4 where γ i is the scaling factor, and b i is the bias.

A related normalization and reparameterization scheme, weight normalization BID22 , has been developed as an alternative to batch normalization, aiming to accelerate the convergence of SGD optimization.

We note the difference between spherical weight optimization and weight normalization.

First, the weight vector of each hidden unit is not directly normalized in weight normalization, i.e, w i 2 = 1 in general.

At training time, the activation of hidden unit i is DISPLAYFORM0 which is equivalent to Eq. (15) for the forward pass.

For the backward pass, the effective learning rate still depends on w i 2 in weight normalization, hence it does not solve the ill-conditioning problem.

At inference time, both of these two schemes can combine w i and γ i into a single equivalent weight vector, DISPLAYFORM1 While spherical weight optimization naturally encompasses weight normalization, it can further benefit from batch normalization.

When combined with batch normalization, Eq. (15) evolves into DISPLAYFORM2 where BN (·) represents the transformation done by batch normalization without scaling and shifting.

Here, γ i serves as the scaling factor for both the normalized weight vector and batch normalization.

At training time, the distribution of the input vector, x, changes over time, slowing down the training of the sub-network composed by the upper layers.

BID22 observe that, such problem cannot be eliminated by normalizing the weight vectors alone, but can be substantially mitigated by combining weight normalization and mean-only batch normalization.

Additionally, in linear rectifier networks, the scaling factors, γ i , can be removed (or set to 1), without changing the overall network function.

Since w i · x is standardized by batch normalization, we have DISPLAYFORM3 and hence DISPLAYFORM4 Therefore, y i 's that belong to the same layer, or different dimensions of x that fed to the upper layer, will also have comparable variances, which potentially makes the weight updates of the upper layer more stable.

For these reasons, we combine the use of spherical weight optimization and batch normalization, as shown in Eq. (17).

For multi-class classification tasks, the softmax function is the de facto activation function for the output layer.

Despite its simplicity and intuitive probabilistic interpretation, we observe a related problem to the ill-conditioning problem we have addressed.

Similar to how different magnitudes of weight vectors result in different updates to the same network function, the learning signal backpropagated from the softmax layer varies with the overall magnitude of the logits.

Specifically, when using cross entropy as the surrogate loss with one-hot target vectors, the prediction is considered correct as long as arg max c∈C (z c ) is the target class, where z c is the logit before the softmax activation, corresponding to category c ∈ C. Thus, the logits can be positively scaled together without changing the predictions, whereas the cross entropy and its derivatives will vary with the scaling factor.

Concretely, denoting the scaling factor by η, the gradient w.r.t.

each logit is DISPLAYFORM0 and DISPLAYFORM1 whereĉ is the target class, andc ∈ C\ {ĉ}.For Adam and ND-Adam, since the gradient w.r.t.

each scalar or vector are normalized, the absolute magnitudes of Eq. (20a) and (20b) are irrelevant.

Instead, the relative magnitudes make a difference here.

When η is small, we have DISPLAYFORM2 which indicates that, when the magnitude of the logits is small, softmax encourages the logit of the target class to increase, while equally penalizing that of the other classes.

On the other end of the spectrum, assuming no two digits are the same, we have DISPLAYFORM3 wherec = arg max c∈C\{ĉ} (z c ), andc ∈ C\ {ĉ,c }.

Eq. FORMULA2 indicates that, when the magnitude of the logits is large, softmax penalizes only the largest logit of the non-target classes.

The latter case is related to the saturation problem of softmax discussed in BID20 .

However, they focus on the problem of small absolute gradient magnitude, which does not affect Adam and ND-Adam.

It is worth noting that both of these two cases can happen without the scaling factor.

For instance, varying the norm of the weights of the softmax layer is equivalent to varying the value of η, in terms of the relative magnitude of the gradient.

In the case of small η, the logits of all non-target classes are penalized equally, regardless of the difference inẑ −z for differentz ∈ C\ {ẑ}. However, it is more reasonable to penalize more the logits that are closer toẑ, which are more likely to cause misclassification.

In the case of large η, although the logit that is most likely to cause misclassification is strongly penalized, the logits of other non-target classes are ignored.

As a result, the logits of the non-target classes tend to be similar at convergence, ignoring the fact that some classes are closer to each other than the others.

We propose two methods to exploit the prior knowledge that the magnitude of the logits should not be too small or too large.

First, we can apply batch normalization to the logits.

But instead of setting γ c 's as trainable variables, we consider them as a single hyperparameter, γ C , such that γ c = γ C , ∀c ∈ C. Tuning the value of γ C can lead to a better trade-off between the two extremes described by Eq. (21) and (22).

The optimal value of γ C tends to remain the same for different optimizers or different network widths, but varies with dataset and network depth.

We refer to this method as batch-normalized softmax (BN-Softmax).Alternatively, since the magnitude of the logits tends to grow larger than expected (in order to minimize the cross entropy), we can apply L2-regularization to the logits by adding the following penalty to the loss function: DISPLAYFORM4 where λ C is a hyperparameter to be tuned.

Different from BN-Softmax, λ C can be shared by different datasets and networks of different depths.

In this section, we provide empirical evidence for the analysis in Sec. 2.2, and evaluate the performance of ND-Adam and regularized softmax on CIFAR-10 and CIFAR-100.

To empirically examine the effect of L2 weight decay, we train a wide residual network (WRN) BID29 of 22 layers, with a width of 7.5 times that of a vanilla ResNet.

Using the notation in BID29 , we refer to this network as WRN-22-7.5.

We train the network on the CIFAR-10 dataset BID16 , with a small modification to the original WRN architecture, and with a different learning rate annealing schedule.

Specifically, for simplicity and slightly better performance, we replace the last fully connected layer with a convolutional layer with 10 output feature maps.

I.e., we change the layers after the last residual block from BN-ReLU-GlobalAvgPool-FC-Softmax toIn FIG2 , we show how the effective learning rate varies in different hyperparameter settings.

By Eq. FORMULA8 , ∆w i 2 / w i 2 is expected to remain the same as long as αλ stays constant, which is confirmed by the fact that the curve for α 0 = 0.1, λ = 0.001 overlaps with that for α 0 = 0.05, λ = 0.002.

However, comparing the curve for α 0 = 0.1, λ = 0.001, with that for α 0 = 0.1, λ = 0.0005, we can see that the value of ∆w i 2 / w i 2 does not change proportionally to αλ.

On the other hand, by using ND-Adam, we can control the value of ∆w i 2 / w i 2 more precisely by adjusting the learning rate for weight vectors, α v .

For the same training step, changes in α v lead to approximately proportional changes in ∆w i 2 / w i 2 , as shown by the two curves corresponding to ND-Adam in FIG2 .

To compare the generalization performance of SGD, Adam, and ND-Adam, we train the same WRN-22-7.5 network on the CIFAR-10 and CIFAR-100 datasets.

For SGD and ND-Adam, we first tune the hyperparameters for SGD (α 0 = 0.1, λ = 0.001, momentum 0.9), then tune the initial learning rate of ND-Adam for weight vectors to match the effective learning rate to that of SGD (α v 0 = 0.05), as shown in FIG2 .

While L2 weight decay can greatly affect the performance of SGD, it does not noticeably benefit Adam in our experiments.

For Adam and ND-Adam, β 1 and β 2 are set to the default values of Adam, i.e., β 1 = 0.9, β 2 = 0.999.

Although the learning rate of Adam is usually set to a constant value, we observe better performance with the cosine learning rate schedule.

The initial learning rate of Adam (α 0 ), and that of ND-Adam for scalar parameters (α s 0 ) are both tuned to 0.001.

We use the same data augmentation scheme as used in BID29 , including horizontal flips and random crops, but no dropout is used.

We first experiment with the use of trainable scaling parameters (γ i ) of batch normalization.

As shown in FIG4 , at convergence, the test accuracies of ND-Adam are significantly improved upon that of vanilla Adam, and matches that of SGD.

Note that at the early stage of training, the training losses of Adam drop dramatically as shown in FIG4 , and the test accuracies also increase more rapidly than that of ND-Adam and SGD.

However, the test accuracies remain at a high level afterwards, which indicates that Adam tends to quickly find and get stuck in bad local minima that do not generalize well.

The average results of 3 runs are summarized in the first part of Table 1 .

Interestingly, compared to SGD, ND-Adam shows slightly better performance on CIFAR-10, but worse performance on CIFAR-100.

This inconsistency may be related to the problem of softmax discussed in Sec. 4, that there is a lack of proper control over the magnitude of the logits.

But overall, given comparable effective learning rates, ND-Adam and SGD show similar generalization performance.

In this sense, the effective learning rate is a more natural learning rate measure than the learning rate hyperparameters.

Next, we repeat the experiments with the use of BN-Softmax.

As discussed in Sec. 3.2, γ i 's can be removed from a linear rectifier network, without changing the overall network function.

Although this property does not strictly hold for residual networks due to the skip connections, we find that simply removing the scaling factors results in slightly improved generalization performance when using ND-Adam.

However, the improvement is not consistent as it degrades performance of SGD.

Interestingly, when BN-Softmax is further used, we observe consistent improvement over all three algorithms.

Thus, we only report results for this setting.

The scaling factor of the logits, γ C , is set to 2.5 for CIFAR-10, and 1 for CIFAR-100.

As shown in the second part of Table 1 , BN-Softmax significantly improves the performance of Adam and ND-Adam.

Moreover, in this setting, we obtain the best generalization performance with ND-Adam, outperforming SGD and Adam on both CIFAR-10 and CIFAR-100.While the TensorFlow implementation we use already provides an adequate test bed, we notice that it is different from the original implementation of WRN in several aspects.

For instance, they use different nonlinearities (leaky ReLU vs. ReLU), and use different skip connections for downsampling (average pooling vs. strided convolution).

A seemingly subtle but important difference is that, L2-regularization is applied not only to weight vectors, but also to the scales and biases of batch normalization in the original implementation, which leads to better generalization performance.

For further comparison between SGD and ND-Adam, we reimplement ND-Adam and test its performance on a PyTorch version of the original implementation BID28 .Due to the aforementioned differences, we use a slightly different hyperparameter setting in this experiment.

Specifically, for SGD λ is set to 5e−4, while for ND-Adam λ is set to 5e−6 (L2-regularization for biases), and both α does not yield improved performance for SGD, since the L2-regularization applied to γ i 's and the last layer weights can serve a similar purpose.

Thus, we only apply L2-regularized softmax for ND-Adam with λ C = 0.001.

The average results of 3 runs are summarized in TAB2 .

Note that the performance of SGD for WRN-28-10 is slightly better than that reported with the original implementation (i.e., 4.00 and 19.25), due to the modifications described in Sec. 5.

@highlight

A tailored version of Adam for training DNNs, which bridges the generalization gap between Adam and SGD.

@highlight

Proposes a variant of ADAM optimization algorithm that normalizes weights of each hidden unit using batch normalization

@highlight

Extension of the Adam optimization algorithm to preserve the update direction by adapting the learning rate for the incoming weights to a hidden unit jointly using the L2 norm of the gradient vector