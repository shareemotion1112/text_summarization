We conduct a mathematical analysis on the Batch normalization (BN) effect on gradient backpropagation in residual network training in this work, which is believed to play a critical role in addressing the gradient vanishing/explosion problem.

Specifically, by analyzing the mean and variance behavior of the input and the gradient in the forward and backward passes through the BN and residual branches, respectively, we show that they work together to confine the gradient variance to a certain range across residual blocks in backpropagation.

As a result, the gradient vanishing/explosion problem is avoided.

Furthermore, we use the same analysis to discuss the tradeoff between depth and width of a residual network and demonstrate that shallower yet wider resnets have stronger learning performance than deeper yet thinner resnets.

Convolutional neural networks (CNNs) BID10 BID1 BID8 aim at learning a feature hierarchy where higher level features are formed by the composition of lower level features.

The deep neural networks act as stacked networks with each layer depending on its previous layer's output.

The stochastic gradient descent (SGD) method BID12 has proved to be an effective way in training deep networks.

The training proceeds in steps with SGD, where a mini-batch from a given dataset is fed at each training step.

However, one factor that slows down the stochastic-gradient-based learning of neural networks is the internal covariate shift.

It is defined as the change in the distribution of network activations due to the change in network parameters during the training.

To improve training efficiency, BID7 introduced a batch normalization (BN) procedure to reduce the internal covariate shift.

The BN changes the distribution of each input element at each layer.

Let x = (x 1 , x 2 , · · · , x K ), be a K-dimensional input to a layer.

The BN first normalizes each dimension of x as DISPLAYFORM0 and then provide the following new input to the layer DISPLAYFORM1 where k = 1, · · · , K and γ k and β k are parameters to be determined.

BID7 offered a complete analysis on the BN effect along the forward pass.

However, there was little discussion on the BN effect on the backpropagated gradient along the backward pass.

This was stated as an open research problem in BID7 .

Here, to address this problem, we conduct a mathematical analysis on gradient propagation in batch normalized networks.

The number of layers is an important parameter in the neural network design.

The training of deep networks has been largely addressed by normalized initialization BID12 BID3 BID11 and intermediate normalization layers BID7 .

These techniques enable networks consisting of tens of layers to converge using the SGD in backpropagation.

On the other hand, it is observed that the accuracy of conventional CNNs gets saturated and then degrades rapidly as the network layer increases.

Such degradation is not caused by over-fitting since adding more layers to a suitably deep model often results in higher training errors BID13 .

To address this issue, BID6 introduced the concept of residual branches.

A residual network is a stack of residual blocks, where each residual block fits a residual mapping rather than the direct input-output mapping.

A similar network, called the highway network, was introduced by BID13 .

Being inspired by the LSTM model BID2 , the highway network has additional gates in the shortcut branches of each block.

There are two major contributions in this work.

First, we propose a mathematical model to analyze the BN effect on gradient propogation in the training of residual networks.

It is shown that residual networks perform better than conventional neural networks because residual branches and BN help maintain the gradient variation within a range throughout the training process, thus stabilizing gradient-based-learning of the network.

They act as a check on the gradients passing through the network during backpropagation so as to avoid gradient vanishing or explosion.

Second, we provide insights into wide residual networks based on the same mathematical analysis.

The wide residual network was recently introduced by BID16 .

As the gradient goes through the residual network, the network may not learn anything useful since there is no mechanism to force the gradient flow to go through residual block weights during the training.

In other words, it might be possible that there are only a few blocks that learn useful representations while a large number of blocks share very little information with small contributions to the ultimate goal.

We will show that residual blocks that stay dormant are the chains of blocks at the end of each scale of the residual network.

The rest of this paper is organized as follows.

Related previous work is reviewed in Sec. 2.

Next, we derive a mathematical model for gradient propagation through a layer defined as a combination of batch normalization, convolution layer and ReLU in Sec. 3.

Then, we apply this mathematical model to a resnet block in Sec. 4.

Afterwards, we use this model to show that the dormant residual blocks are those at the far-end of a scale in deep residual networks in Sec. 5.

Concluding remarks and future research directions are given in Sec. 6.

One major obstacle to the deep neural network training is the vanishing/exploding gradient problem BID0 .

It hampers convergence from the beginning.

Furthermore, a proper initialization of a neural network is needed for faster convergence to a good local minimum.

BID12 proposed to initialize weights randomly, in such a way that the sigmoid is activated in its linear region.

They implemented this choice by stating that the standard deviation of the output of each node should be close to one.

BID3 proposed to adopt a properly scaled uniform distribution for initialization.

Its derivation was based on the assumption of linear activations used in each layer .

Most recently, took the ReLU/PReLU activation into consideration in deriving their proposal.

The basic principle used by both is that a proper initialization method should avoid reducing or magnifying the magnitude of the input and its gradient exponentially.

To achieve this objective, they first initialized weight vectors with zero mean and a certain variance value.

Then, they derived the variance of activations at each layer, and equated them to yield an initial value for the variance of weight vectors at each layer.

Furthermore, they derived the variance of gradients that are backpropagated at each layer, and equated them to obtain an initial value for the variance of weight vectors at each layer.

They either took an average of the two initialized weight variances or simply took one of them as the initial variance of weight vectors.

Being built up on this idea, we attempt to analyze the BN effect by comparing the variance of gradients that are backpropagated at each layer below.3 GRADIENT PROPAGATION THROUGH A LAYER

We first consider the simplest case where a layer consists of the BN operation only.

We use x andx to denote a batch of input and output values to and from a batch normalized (BN) layer, respectively.

The standard normal variate of x is z.

In gradient backpropagation, the batch of input gradient values to the BN layer is ∆x while the batch of output gradient values from the BN layer is ∆x.

By simple manipulation of the formulas given in BID7 , we can get DISPLAYFORM0 where x i is the ith element of batch x and Std is the standard deviation.

Then, it is straightforward to derive E(∆x i ) = 0, and V ar( DISPLAYFORM1 3.2 CASCADED BN/RELU/CONV LAYER Next, we examine a more complex but common case, where a layer consists of three operations in cascade.

They are: 1) batch normalization, 2) ReLU activation, and 3) convolution.

To simplify the gradient flow calculation, we make some assumptions which will be mentioned whenever needed.

The input to the Lth Layer of a deep neural network is y L−1 while its output is y L .

We use BN , ReLU and CON V to denote the three operations in each sub-layer.

Then, we have the following three equations: DISPLAYFORM2 The relationship between DISPLAYFORM3

We will derive the mean and variance of output y L,i from the input y L−1 .

First, we examine the effect of the BN sub-layer.

The output of a batch normalization layer is γ i z i + β i , where z i is the standard normal variate of y L−1,i , calculated across a batch of activations.

Clearly, we have DISPLAYFORM0 Next, we consider the effect of the ReLU sub-layer.

Let a = βi γi .

We assume that a is small enough so that the standard normal variate z i follows a nearly uniform distribution in interval (0, a).

In Appendix A, we show a step-by-step procedure to derive the mean and variance of the output of the ReLU sub-layer when it is applied to the output of a BN layer.

Here, we summarize the main results below: DISPLAYFORM1 Finally, we consider the influence of the CONV sub-layer.

To simplify the analysis, we assume that all elements in W f L are mutually independent and with the same distribution of mean 0 and all elements in y L−1 are also mutually independent and with the same distribution across a batch of activations.

Furthermore, DISPLAYFORM2 Note that assuming the weight elements come from a distribution with mean 0 is a fair assumption because we initialize the weight elements from a distribution with mean 0 and in the next section, we see that the mean of gradient that reaches the convolution layer during backpropagation has mean 0 across a batch.

We consider backward propagation from the Lth layer to the (L − 1)th layer and focus on gradient propagation.

Since, the gradient has just passed through the BN sub-layer of Lth layer, using FORMULA4 we get E(∆y L ) = 0.

First, gradients go through the CONV sub-layer.

Under the following three assumptions: 1) elements in W b L are mutually independent and with the same distribution of mean 0, 2) elements in ∆y L are mutually independent and with the same distribution across a batch, and 3) ∆y L and W b L are independent of each other.

Then, we get DISPLAYFORM0 Next, gradients go through the ReLU sub-layer.

It is assumed that the function applied to the gradient vector on passing through ReLU and the elements of gradient are independent of each other.

Since the input in the forward pass was a shifted normal variate (a = βi γi ), we get DISPLAYFORM1 (12) In the final step, gradients go through the BN sub-layer.

If the standard normal variate, z, to the BN sub-layer and the incoming gradients ∆y are independent, we have E(z i ∆y L−1,i ) = E(z i )E(∆y L−1,i ) = 0.

The last equality holds since the mean of the standard normal variate is zero.

The final result is DISPLAYFORM2 Note that the last product term in the derived formula is the term under consideration for checking gradient explosion or vanishing.

The other two fractions are properties of the network, that compare two adjacent Layers.

The skipped steps are given in Appendix B.3.5 DISCUSSION Initially, we set β i = 0 and γ i = 1 so that a = 0.

Then, the last product term in the RHS of Eq. (13) is equal to one.

Hence, if the weight initialization stays equal across all the layers, propagated gradients are maintained throughout the network.

In other words, the BN simplifies the weight initialization job.

For intermediate steps, we can estimate the gradient variance under simplifying assumptions that offer a simple minded view of gradient propagation.

Note that, when a = β γ is small, the last product term is nearly equal to one.

The major implication is that, the BN helps maintain gradients across the network, throughout the training, thus stabilizing optimization.

The resnet blocks in the forward pass and in the gradient backpropagation pass are shown in Figs. 2 and 3, respectively.

A residual network has multiple scales, each scale has a fixed number of residual blocks, and the convolutional layer in residual blocks at the same scale have the same number of filters.

In the analysis, we adopt the model where the filter number increases k times from one scale to the next one.

Although no bottleneck blocks are explicitly considered here, our analysis holds for bottleneck blocks as well.

As shown in FIG2 , the input passes through a sequence of BN, ReLU and CONV sub-layers along the shortcut branch in the first residual block of a scale, which shapes the input to the required number of channels in the current scale.

For all other residual blocks in the same scale, the input just passes through the shortcut branch.

For all residual blocks, the input goes through the convolution branch which consists of two sequences of BN, ReLU and CONV sub-layers.

We use a layer to denote a sequence of BN, ReLU and CONV sub-layers as used in the last section and F to denote the compound function of one layer.

To simplify the computation of the mean and variance of y L,i and ∆y L,i , we assume that a = βi γi is small (<1) across all the layers so that we can assume a as constant for all the layers.

We define the following two associated constants.

DISPLAYFORM0 c 2 = 0.5 + 2 π a + 0.5a DISPLAYFORM1 which will be needed later.

As shown in FIG2 , block L is the Lth residual block in a scale with its input y L−1 and output y L .

The outputs of the first and the second BN-ReLU-CONV layers in the convolution branch are DISPLAYFORM0 DISPLAYFORM1 For L>1, block L receives an input of size n s in the forward pass and an input gradient of size n s in the backpropagation pass.

Since block 1 receives its input y 0 from the previous scale, it receives an input of size ns k in the forward pass.

By assuming y L andŷ L are independent, we have DISPLAYFORM2 We will show how to compute the variance of y L,i step by step in Appendix C for DISPLAYFORM3 where c 2 is defined in Eq. (15).We use ∆ as prefix in front of vector representations at the corresponding positions in forward pass to denote the gradient in FIG3 in the backward gradient propagation.

Also, as shown in FIG3 , we represent the gradient vector at the tip of the convolution branch and shortcut branch by ∆ L and∆ L respectively.

As shown in the figure, we have DISPLAYFORM4 A step-by-step procedure in computing the variance of ∆y L−1,i is given in Appendix D. Here, we show the final result below: DISPLAYFORM5 (20)

We can draw two major conclusions from the analysis conducted above.

First, it is proper to relate the above variance analysis to the gradient vanishing and explosion problem.

The gradients go through a BN sub-layer in one residual block before moving to the next residual block.

As proved in Sec. 3, the gradient mean is zero when it goes through a BN sub-layer and it still stays at zero after passing through a residual block.

Thus, if it is normally distributed, the probability of the gradient values between ± 3 standard deviations is 99.7%.

A smaller variance would mean lower gradient values.

In contrast, a higher variance implies a higher likelihood of discriminatory gradients.

Thus, we take the gradient variance across a batch as a measure for stability of gradient backpropagation.

Second, recall that the number of filters in each convolution layer of a scale increases by k times with respect to its previous scale.

Typically, k = 1 or 2.

Without loss of generality, we can assume the following: the variance of weights is about equal across layers, c 1 /c 2 ≈ 1, and k = 2.

Then, Eq. (20) can be simplified to DISPLAYFORM0 We see from above that the change in the gradient variance from one residual block to its next is little.

This is especially true when the L value is high.

This point will be further discussed in the next section.

We trained a Resnet-15 model that consists of 15 residual blocks and 3 scales on the CIFAR-10 dataset, and checked the gradient variance across the network throughout the training.

We plot the mean of the gradient variance and the l 2 -norm of the gradient at various residual block locations in FIG4 and 5, respectively, where the gradient variance is calculated for each feature across one batch.

Since gradients backpropagate from the output layer to the input layer, we should read each plot from right to left to see the backpropagation effect.

The behavior is consistent with our analysis.

There is a gradual increase of the slope across a scale.

The increase in the gradient variance between two residual blocks across a scale is inversely proportional to the distance between the residual blocks and the first residual block in the scale.

Also, there is a dip in the gradient variance value when we move from one scale to its previous scale.

Since the BN sub-layer is used in the shortcut branch of the first block of a scale, it ensures the decrease of the gradient variance as it goes from one scale to another.

Some other experiments that we conducted to support our theory can be found in Appendix E. BID14 showed that the paths which gradients take through a ResNet are typically far shorter than the total depth of that network.

For this reason, they introduced the "effective depth" idea as a measure for the true length of these paths.

They showed that almost all of gradient updates in the training come from paths of 5-17 modules in their length.

BID15 also presented a similar concept.

That is, residual networks are actually an ensemble of various sub-networks and it echoes the concept of effective depth.

Overall, the main point is that some residual blocks stay dormant during gradient backpropagation.

Based on our analysis in Sec. 4, the gradient variance should increase by L/(L − 1) after passing through a residual block, where (L − 1) is the distance of the current residual block from the first residual block in a scale.

Thus, the gradient variance should not change much as the gradient backpropagates through the chain of residual networks at the far end of a scale if we use a residual network of high depth.

Since a lower gradient variance value implies non-varying gradient values, it supports the effective path concept as well.

As a result, the weights in the residual blocks see similar gradient variation without learning much discriminatory features.

In contrast, for networks of lower depth , the gradient variance changes more sharply as we go from one residual block to another.

As a result, all weights present in the network are used in a more discriminatory way, thus leading to better learning.

We compare the performance of the following three Resnet models:1.

Resnet-99 with 99 resnet blocks, 2.

Resnet-33 with 33 resnet blocks and tripled filter numbers in each resnet block, 3.

Resnet-9 with 9 resnet blocks and 11 times filter numbers in each resnet block.

Note that the total filter numbers in the three models are the same for fair comparison.

We trained them on the CIFAR-10 dataset.

First, we compare the training accuracy between Resent-9 and Resnet-99 in FIG6 and that between Resent-9 and Resnet-33 in FIG7 , where the horizontal axis shows the epoch number.

We see that Resnet-9 reaches the higher accuracy faster than both Resnet-99 and Resnet-33, yet the gap between Resnet-9 and Resnet-33 is smaller than that between Resnet-9 and Resnet-99.

This supports our claim that a shallow-wide Resnet learns faster than a deep-narrow Resnet.

Next, we compare their test set accuracy in TAB1 .

We see that Resnet-9 has the best performance while Resnet-99 the worst.

This is in alignment with our prediction and the experimental results given above.

Resnet with 99 resnet blocks 93.4% Resnet with 33 resnet blocks 93.8% Resnet with 9 resnet blocks 94.4% Furthermore, we plot the mean of the gradient variance, calculated for each feature across one batch, as a function of the residual block index at epochs 1, 25,000 and 50,000 in Figs. 8, 9 and 10, respectively, where the performance of Resnet-99, Resnet-33 and Resnet-9 is compared.

We observe that the gradient variance does not change much across a batch as it passes through the residual blocks present at the far end of a scale in Resnet-99.

For Resnet-33, there are fewer resnet blocks at the far end of a scale that stay dormant.

We can also see clearly the gradient variance changes more sharply during gradient backpropagation in resnet-9.

Hence, the residual blocks present at the end of a scale have a slowly varying gradient passing through them in Resnet-99, compared to Resnet-33 and Resnet-9.

These figures show stronger learning performance of shallower but wider resnets.

Mathematical analysis was conducted to analyze the BN effect on gradient propagation in residual network training in this work.

We explained how BN and residual branches work together to maintain gradient stability across residual blocks in back propagation.

As a result, the gradient does not explode or vanish in backpropagation throughout the whole training process.

Furthermore, we applied this mathematical analysis to the decision on the residual network architecture -whether it should be deeper or wider.

We showed that a slowly varying gradient across residual blocks results in lower learning capability and deep resnets tend to learn less than their corresponding wider form.

The wider resnets tend to use their parameter space better than the deeper resnets.

The Saak transform has been recently introduced by BID9 , which provides a brand new angle to examine deep learning.

The most unique characteristics of the Saak transform approach is that neither data labels nor backpropagation is needed in training the filter weights.

It is interesting to study the relationship between multi-stage Saak transforms and residual networks and compare their performance in the near future.

We apply the ReLU to the output of a BN layer, and show the step-by-step procedure in calculating the variance and the mean of the output of the ReLU operation.

In the following derivation, we drop the layer and the element subscripts (i.e., L and i) since there is no confusion.

It is assumed that scaling factors, β and γ, in the BN are related such that a = β/γ is a small number and the standard normal variable z has a nearly uniform distribution in (−a,0).

Then, we can write the shifted Gaussian variate due to the BN operation as DISPLAYFORM0 Let y = ReLU (z + a).

Let a > 0.

We can write DISPLAYFORM1 The first right-hand-side (RHS) term of Eq. FORMULA1 is zero since y = 0 if z < −a due to the ReLU operation.

Thus, E(y|z < −a) = 0.

For the second RHS term, z is uniformly distributed with probability density function equal to a −1 in range (-a, 0) if 0 < a << 1.

Then, we have DISPLAYFORM2 , and E(y| − a < z < 0) = a 2 .For the third RHS term, P (z > 0) = 0.5.

Besides, z > 0 is half-normal distributed.

Thus, we have DISPLAYFORM3 Based on the above results, we get DISPLAYFORM4 Similarly, we can derive a formula for E(y 2 ) as DISPLAYFORM5 For the first RHS term of Eq. (27), we have E(y 2 |z < −a) = 0 due to the ReLU operation.

For the second RHS term of Eq. (27), z is uniformly distributed with probability density function a −1 for -a<z<0 so that P (−a < z < 0) = a √ 2πand E(y 2 | − a < z < 0) = a 2 3 .

For the third RHS term P (z > 0) = 0.5 for z > 0.

The random variable z > 0 is half normal distributed so that DISPLAYFORM6 Then, we obtain DISPLAYFORM7 We can follow the same procedure for a < 0.

The final results are summarized below.

E(ReLU (γz + β)) = γE(y) , and E((ReLU (γz + β)) DISPLAYFORM8 where E(y) and E(y 2 ) are given in Eqs. FORMULA1 and FORMULA1 , respectively.

• We assumed that the function(F) applied by ReLU to the gradient vector and the gradient elements are independent of each other.

Function F is defined as DISPLAYFORM0 where ∆y denotes input gradient in gradient backpropagation and y denotes the input activation during forward pass to the ReLU layer.

Coming back to our analysis, sinceỹ L−1,i is a normal variate shifted by a, the probability that the input in forward pass to the ReLU layer, i.e.ỹ L−1,i is greater than 0 is DISPLAYFORM1 Similarly, we can solve for Var(∆ỹ L−1,i ) and thus, get Eq. (12).• First, using eq 5 and the assumption that the input standard normal variate in forward pass and the input gradient in gradient pass are independent, we have DISPLAYFORM2 Then, using Eq. (10) for Y L−1 (yet with L replaced with L − 1), we can get Eq. (13).

For L = 1, y 1 = F (y 0 ).

Since the receptive field for the last scale is k times smaller, we get the following from Eq. (10), DISPLAYFORM0 Also, sinceŷ 1 = F (F (y 0 )), we have DISPLAYFORM1 based on Eq. (10).

Therefore, we get DISPLAYFORM2 For L = N > 1, the input just passes through the shortcut branch.

Then, DISPLAYFORM3 due to using Eq. (10).

Thus, DISPLAYFORM4 Doing this recursively from L = 1 to N , we get DISPLAYFORM5 APPENDIX D For block L = N > 1, the gradient has to pass through two BN-ReLU-Conv Layers in convolution branch.

Since, the receptive field doesn't change in between the two BN-ReLU-Conv Layers in the convolution branch of the block, we use Eq. (13) and find that for same receptive field between the two layers i. DISPLAYFORM6 When gradient passes through the first BN-ReLU-Conv Layer, the variance of the forward activation that BN component sees is actually the variance of the output of previous block.

Hence, using Var(y L−1,i ), which is the output of previous residual block, in place of the denominator in Eq. (31), we get DISPLAYFORM7 We assume that∆ L and ∆ L are independent of each other.

Since we are calculating for Block L>1 where there is no BN-ReLU-Conv Layer in shortcut branch, we have V ar(∆ L,i ) = V ar(∆y L,i ).

DISPLAYFORM8 Finally, we obtain All the models had 15 residual blocks, 5 in each scale.

The parameters of each model were initialized similarly and were trained for same number of epochs.

The weights were initialized with xavier initialization and the biases were initialized to 0.

First, we compare the training accuracy among the three models in FIG1 , where the horizontal axis shows the epoch number.

We see that M odel 1 reaches higher accuracy faster than the other two models.

However, M odel 2 isn't far behind.

But DISPLAYFORM9

Final accuracy M odel 1 92.5% M odel 2 90.6% M odel 3 9.09% Table 2 : Comparison of test accuracy of three Resnet models.

M odel 3 , which has BN removed, doesn't learn anything.

Next, we compare their test set accuracy in Table 2 .

We see that M odel 1 has the best performance while M odel 2 isn't far behind.

Furthermore, we plot the mean of the gradient variance, calculated for each feature across one batch, as a function of the residual block index at epochs 25,000, 50,000 and 75,000 in Figs. 12, 13 and 14, respectively, where the performance of M odel 1 and M odel 2 is compared.

We observe that the gradient variance also stays within a certain range, without exploding or vanishing, in case of M odel 2 .

However, the change in gradient variance across a scale doesn't follow a fixed pattern compared to M odel 1 .

We also plot a similar kind of plot for M odel 3 at epoch-1 in FIG1 We observed gradient explosion, right from the start, in case of M odel 3 and the loss function had quickly become undefined.

This was the reason, why M odel 3 didn't learn much during the course of training.

This experiment shows that BN plays a major role in stabilizing training of residual networks.

Even though we remove the residual branches, the network still tries to learn from the training set, with its gradient fixed in a range across layers.

However, removing BN hampers the training process right from the start.

Thus, we can see that batch normalization helps to stop gradient vanishing and explosion throughout training, thus stabilizing optimization.

<|TLDR|>

@highlight

Batch normalisation maintains gradient variance throughout training, thus stabilizing optimization.

@highlight

This paper analyzed the effect of batch normalization on gradient backpropagation in residual networks