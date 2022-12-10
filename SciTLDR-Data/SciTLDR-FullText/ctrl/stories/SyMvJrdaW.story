We propose a Warped Residual Network (WarpNet) using a parallelizable warp operator for forward and backward propagation to distant layers that trains faster than the original residual neural network.

We apply a perturbation theory on residual networks and decouple the interactions between residual units.

The resulting warp operator is a first order approximation of the output over multiple layers.

The first order perturbation theory exhibits properties such as binomial path lengths and exponential gradient scaling found experimentally by Veit et al (2016).

We demonstrate through an extensive performance study that the proposed network achieves comparable predictive performance to the original residual network with the same number of parameters, while achieving a significant speed-up on the total training time.

As WarpNet performs model parallelism in residual network training in which weights are distributed over different GPUs, it offers speed-up and capability to train larger networks compared to original residual networks.

Deep Convolution Neural Networks (CNN) have been used in image recognition tasks with great success.

Since AlexNet BID6 , many other neural architectures have been proposed to achieve start-of-the-art results at the time.

Some of the notable architectures include, VGG BID7 , Inception and Residual networks (ResNet) BID3 .Training a deep neural network is not an easy task.

As the gradient at each layer is dependent upon those in higher layers multiplicatively, the gradients in earlier layers can vanish or explode, ceasing the training process.

The gradient vanishing problem is significant for neuron activation functions such as the sigmoid, where the gradient approaches zero exponentially away from the origin on both sides.

The standard approach to combat vanishing gradient is to apply Batch Normalization (BN) BID5 followed by the Rectified Linear Unit (ReLU) BID1 activation.

More recently, skip connections BID9 have been proposed to allow previous layers propagate relatively unchanged.

Using this methodology the authors in BID9 were able to train extremely deep networks (hundreds of layers) and about one thousand layers were trained in residual networks BID3 .As the number of layers grows large, so does the training time.

To evaluate the neural network's output, one needs to propagate the input of the network layer by layer in a procedure known as forward propagation.

Likewise, during training, one needs to propagate the gradient of the loss function from the end of the network to update the model parameters, or weights, in each layer of the network using gradient descent.

The complexity of forward and propagation is O(K), where K is the number of layers in the network.

To speed up the process, one may ask if there exist a shallower network that accurately approximates a deep network so that training time is reduced.

In this work we show that there indeed exists a neural network architecture that permits such an approximation, the ResNet.

Residual networks typically consist of a long chain of residual units.

Recent investigations suggest that ResNets behave as an ensemble of shallow networks BID11 .

Empirical evidence supporting this claim includes one that shows randomly deactivating residual units during training (similar to drop-out BID8 ) appears to improve performance BID4 .

The results imply that the output of a residual unit is just a small perturbation of the input.

In this work, we make an approximation of the ResNet by using a series expansion in the small perturbation.

We find that merely the first term in the series expansion is sufficient to explain the binomial distribution of path lengths and exponential gradient scaling experimentally observed by BID11 .

The approximation allows us to effectively estimate the output of subsequent layers using just the input of the first layer and obtain a modified forward propagation rule.

We call the corresponding operator the warp operator.

The backpropagation rule is obtained by differentiating the warp operator.

We implemented a network using the warp operator and found that our network trains faster on image classification tasks with predictive accuracies comparable to those of the original ResNet.

• We analytically investigate the properties of ResNets.

In particular, we show that the first order term in the Taylor series expansion of the layer output across K residual units has a binomial number of terms, which are interpreted as the number of paths in BID11 , and that for ReLU activations the second and higher order terms in the Taylor series vanish almost exactly.• Based on the above-mentioned analysis, we propose a novel architecture, WarpNet, which employs a warp operator as a parallelizable propagation rule across multiple layers at a time.

The WarpNet is an approximation to a ResNet with the same number of weights.• We conduct experiments with WarpNet skipping over one and two residual units and show that WarpNet achieves comparable predictive performance to the original ResNet while achieving significant speed-up.

WarpNet also compares favorably with data parallelism using mini-batches with ResNet.

As opposed to data parallelized ResNet where nearly all the weights are copied to all GPUs, the weights in WarpNet are distributed over various GPUs which enables training of a larger network.

The organization of this paper is as follow.

In Section 2 we analyze the properties of ResNet and show that the binomial path length arises from a Taylor expansion to first order.

In Section 3 we describe Warped Residual Networks.

In Section 4 we show that WarpNet can attain similar performence as the original ResNet while offering a speed-up.

In this section we show that recent numerical results BID11 is explained when the perturbation theory is applied to ResNets.

Consider the input x i of the i-th residual unit and its output x i+1 , where DISPLAYFORM0 Typically, h(x i ) is taken to be an identity mapping, h i (x i ) = x i .

When the feature maps are down sampled, h is usually taken to be an 1 × 1 convolution layer with a stride of 2.

The functions F i is a combination of convolution, normalization and non-linearity layers, so that W i collectively represents the weights of all layers in F i .

In this work we only consider the case where the skip connection is the identity, DISPLAYFORM1 Perturbative feature map flow First, we show that the interpretation of ResNets as an ensemble of subnetworks is accurate up to the first order in F with identity mapping.

One can approximate the output of a chain of residual units by a series expansion.

For instance, the output of two residual units x 3 is related to the input of the first unit by the following (we call the process where x k is expressed in terms of x k−1 an iteration.

The following equations show two iterations).

DISPLAYFORM2 where F 2 (x 1 , W * 2 ) denotes the partial derivative of F 2 with respect to x 1 and W * i denotes the weights at the loss minimum.

A Taylor series expansion in powers of F 1 was performed on F 2 in the second line above.

1 The O( 2 ) term arises from the Taylor series expansion, representing higher order terms.

Equation (2) can be interpreted as an ensemble sum of subnetworks.

Below we show that the second and higher order terms are negligible, that is, the first order Taylor series expansion is almost exact, when ReLU activations are used.

The second order perturbation terms all contain the Hessian F (x).

But after the network is trained, the only non-linear function in F, ReLU, is only non-linear at the origin 2 .

Therefore all second order terms vanish almost exactly.

The same argument applies to higher orders.

DISPLAYFORM3 where the sum is over all subsets σ c and P(S K ) denotes the power set of S K .

We have omitted the O( 2 ) term because the first order approximation is almost exact when ReLU is used as discussed above.

The right hand side of Equation FORMULA3 is interpreted as the sum over subnetworks or paths in the sense of BID11 .

The identity path corresponding to σ c = {∅} gives x 1 in the first term.

If there is only one element in σ c , such that its cardinality |σ c | = 1, the product on the right hand side in parentheses is absent and only terms proportional to F c(1) appears in the sum, where c(1) ∈ {1, . . .

, K}. We provide the proof of Equation FORMULA3 in Appendix A.We can make the equation simpler, solely for simplicity, by setting all weights to be the same such that F c(i) = F and W * c(i) = W * for all i, DISPLAYFORM4 The binomial coefficients appear because the number of subsets of S K with cardinality k is K k .

Note that the implementations of our proposed method (described in Section 3) do not use this simplification.

Exponential gradient scaling Similarly, one observes that the gradient is the sum from all subnetwork contributions, including the identity network.

The magnitudes of subnetwork gradients for an 110 layer ResNet have been measured by BID11 .

If one takes F to have ReLU nonlinearity, then F (x, W * ) = 0 except at the origin.

The non-trivial gradient can be expressed almost exactly as DISPLAYFORM5 This validates the numerical results that the gradient norm decreases exponentially with subnetwork depth as reported in BID11 .

Their experimental results indicate that the average gradient norm for each subnetwork of depth k is given by ||F (x, W * )|| k .All aforementioned properties apply only after the ResNets are trained.

However, if an approximation in the network is made, it would still give similar results after training.

We show in the following sections that our network can attain similar performances as the original ResNet, validating our approximation.

The Warped Residual Network (WarpNet) is an approximation to the residual network, where K consecutive residual units are compressed into one warp layer.

The computation in a warp layer is different from that in a conventional neural network.

It uses a warp operator to compute the output (i.e., x K+1 ) of the layer directly from the input (i.e., x 1 ), as shown in Equation (4).

The number of weights in a warped layer is the same of the one in the original residual network for K consecutive residual units.

For instance, the weights W 1 , W 2 up to W K are present in a warped layer.

But these weights can be used and updated in parallel due to the use of the warp operator.

Below we first describe the forward and backward propagation rules used in warped residual network.

This section shows the propagation rules of the Warped Residual Network using the warp operator T warp .The expression for T warp is derived from Equation 3, that is, by using the Taylor series expansion to the first order: DISPLAYFORM0 Note that T warp can be calculated in a parallelizable manner for all K. This is shown in FIG0 with K = 2, where DISPLAYFORM1 and W i corresponds to the weights in the i-th residual unit in the original ResNet.

The formula for the K = 3 case is shown in Appendix A.

Now we derive the backpropagation rules.

Suppose that the upstream gradient ∂L/∂x 5 is known and we wish to compute ∂L/∂W 1 for gradient descent.

We first back propagate the gradient down from x 5 to x 3 .

With x 5 = T warp (x 3 ), we can derive the backpropagated gradient DISPLAYFORM0 where I is the identity matrix and we have set the derivative of F 4 to zero for ReLU non-linearities.

Note that we have removed all BN layers from F 4 in our implementation.

One sees that the same kind of parallelism in the warp operator is also present for back propagation.

Now we can evaluate the weight gradient for updates DISPLAYFORM1 Similarly for the update rule for W 2 .

Rules for the all other weights in WarpNet can be obtained in the same way, DISPLAYFORM2 The weights W 1 and W 2 can be updated in parallel independently.

The derivative ∂F 2 (x 1 , W 2 )/∂x 1 (in ∂L/∂W 1 ) is already computed in the forward pass which could be saved and reused.

Furthermore, derivatives other than F 3 needed in ∂L/∂x 3 can also be computed in the forward pass.

For higher warp factors K, only the derivative F K+1 is not available after the forward pass.

In this section we discuss our implementation of the WarpNet architecture and the experimental results.

In order to ensure the validity of the series expansion we replace the 1 × 1 convolution layers on skip connections by an average pooling layer and a concatenate layer before the residual unit to reduce the spatial dimensions of feature maps and multiply their channels.

In this way all skip connections are identity mappings.

We adopt a wide residual architecture (WRN) BID12 .

The convolution blocks F comprised of the following layers, from input to output, BN-Conv-BN-ReLU-Conv-BN BID2 DISPLAYFORM0 . .

, W i+K−1 ) and the indices i correspond to the indices in the original residual network.

Using Tensorflow, we implemented a WarpNet with various parameters, k w , K and N warp .

The widening factor BID12 ) is k w , K is the warp factor and with the scheme shown in FIG0 .

We employ Tensorflow's automatic differentiation for backpropagation, where the gradients are calculated by sweeping through the network through the chain rule.

Although the gradients computed in the forward pass can be re-used in the backward pass, we do not do so in our experiment and leave it to future work to potentially further speed up our method.

Even so, the experimental results indicate that WarpNet can be trained faster than WRN with comparable predictive accuracy.

Consider the case K = 2, we found that the computation bottleneck arises from the BN layers in F 2 .

The reason being the gradient of BN layers contains an averaging operation that is expensive to compute.

In our final implementation we removed all BN layers in F 2 from our network.

This results in a departure from our series approximation but it turns out the network still trains well.

This is because the normalizing layers are still being trained in F 1,2 .

To further improve the speed-up we replace the F 1 block in the derivative term F 2 F 1 with the input x 1 so that the term becomes F 2 x 1 .

Similar approximations are made in cases where K > 2.

We have conducted extensive experiments of this modification and found that it has similar predictive accuracies while improving speed-up.

In the following, we refer to this modification of WarpNet as WarpNet1 and the one with F 2 F 1 as WarpNet2.

For K = 3 we replace all F j F i by F j x 1 in WarpNet1.

We also drop the term F 3 F 2 F 1 in computing x 4 in both versions of WarpNet due to the limited GPUs we have in the expriements.

To investigate the speed-up provided by WarpNet and its predictive performance with various approximations on the warp operators, we define the relative speed-up, RS, compared to the corresponding wide residual network (WRN) as DISPLAYFORM1 where t warp is the total time to process a batch for WarpNet during training, and t res is that for the baseline WRN.

For the CIFAR-10 and CIFAR-100 data sets, we trained for 80000 iterations, or 204 epochs.

We took a training batch size of 128.

Initial learning rate is 0.1.

The learning rate drops by a factor of 0.1 at epochs 60, 120, and 160, with a weight decay of 0.0005.

We use common data augmentation techniques, namely, whitening, flipping and cropping.

We study the performance of WarpNet with K = 2 and K = 3.

The averaged results over two runs each are shown in TAB2

warp ] ×N warp with K × N warp residual units in TAB1 .

The total number of convolution layers (represented by n in WRN-n-k w ) is 6KN warp + 1, where the factor of 6 arise from two convolution layers in each residual unit and 3 stages in the network, plus 1 convolution layer at the beginning.

The number of layers in WRN is always odd as we do not use the 1 × 1-convolution layer across stages.

We see that in most cases, WarpNet can achieve similar, if not better, validation errors than the corresponding wide ResNet while offering speed-up.

The experiments also show that the modification of replacing F F by F x 1 , where x 1 is the input of the warp operator, achieves better accuracy most of the time while improving the speed-up.

We observe that increasing from K = 2 to K = 3, using only one more GPU, significantly improves speed-up with only a slight drop in validation accuracy compared to the K = 2 case.

We have also performed experiments on the speed-up as the widening factor k w increases.

We found that the speed-up increases as the WarpNet gets wider.

For k w = 4, 8 and 16, the speed-up in total time for K = 2 is 35%, 40% and 42% respectively.

The speed-up also increases with the warp factor K, for K = 3 using the F x modification, the speed-ups are 44%, 48% and 50% respectively.

We also tested WarpNet on a down-sampled (32x32) ImageNet data set BID0 .

The data set contains 1000 classes with 1281167 training images and 50000 validation images with 50 images each class.

The training batch size is 512, initial learning rate is 0.4 and drops by a factor of 0.1 at every 30 epochs.

The weight decay is set to be 0.0001.

We use the overall best performing warp operator in the CIFAR experiments, namely, the one containing F x.

The results are shown in Table 4 and Figure 2 .

First, we show directly that for a given ResNet there exists a WarpNet that obtains a higher validation accuracy with shorter training time.

We increase K from 2 to 3 and keep everything else fixed.

This corresponds to WarpNet-109-2.

The network has more residual units than WRN-73-2.

We observed that WarpNet-109-2 trains 12% faster than WRN-73-2 while resulting in a better validation accuracy.

Second, WarpNet can achieve close to the benchmark validation error of 18.9% with WRN-28-10 in BID0 .

Note that we were not able to train the corresponding WRN-73-4 on the dataset as the model requires too much memory on a single GPU.

This shows that the weight distribution of WarpNet across GPUs GPU assignment DISPLAYFORM0 DISPLAYFORM1 allows a bigger network to be trained.

Remarkably, the validation error curve for WRN-73-2 and its approximation WarpNet 73-2 (K = 2, N warp = 6) lie almost exactly on top of each other.

This suggests that our implementation of WarpNet is a good approximation of the corresponding WRN throughout training.

WarpNet offers model parallelism to ResNet learning, in which different sets of weights are learned in parallel.

In comparison, a popular way to parallelize deep learning is to split the batch in each training iteration into subsets and allow a different GPU to compute gradients for all weights based on a different subset and synchronization can be done, e.g., by averaging the gradients from all GPUs and updating the weights based on the average.

We refer to such methods as data parallelism methods.

Below we compare WarpNet with a data parallelism method on 2 or 4 GPUs on CIFAR-10 for which we divide each batch into 2 or 4 mini-batches, respectively, and synchronization is done right after all GPUs finish their job on their mini-batch to avoid harming the accuracy.

TAB5 shows the average result over 2 runs for each method.

All methods see the same volume of data during training, which means that the number of epochs is the same for all methods.

We chose the warp operators containing F x in this experiment, that is, WarpNet1 whose operations are specified in the first rows of each block in TAB2 .

We use the GPU assignment DISPLAYFORM0 for the case with 3 GPUs.

The results show that WarpNet is more accurate than data parallelism in both 2-GPU and 4-GPU cases.

When 3 or 4 GPUs are used, WarpNet is much faster than data-parallelized ResNet with 4 GPUs.

We believe this is because the data parallelism method needs to store all the weights of the model in all GPUs and its speed is slowed by the need to update all the weights across all GPUs at the time of synchronization.

In comparison, WarpNet splits the weights among GPUs and each GPU only maintains and updates a subset of weights.

Such weight distributions in WarpNet require less GPU memory, which allows it to train larger networks.

Furthermore, data parallelism can be applied to WarpNet as well to potentially further speed up WarpNet, which is a topic beyond the scope of this paper.

In this paper, we proposed the Warped Residual Network (WarpNet) that arises from the first order Taylor series expansion with ReLU non-linearity.

We showed analytically that the first order expansion is sufficient to explain the ensemble behaviors of residual networks BID11 .

The Taylor series approximation has the structure that allows WarpNet to train consecutive residual units in parallel while ensuring that the performance is similar to the corresponding ResNet.

The weights of different residual units are distributed over the vairous GPUs which enables the training of bigger networks compared to ResNets given limited GPU memory.

Experimental results show that WarpNet can provide a significant speed-up over wide ResNets with similar predictive accuracy, if not better.

We also show that WarpNet outperforms a data parallelism method on ResNet, achieving better predictive accuracies and a much better speed up when more than 2 GPUs are used.

In this section we explicitly work out the expressions for x 3 and x 4 using the Taylor expansion and show that in the general case the path lengths k corresponds to the binomial number of terms with power k in F and F together in the first order Taylor expansion.

The terms of order O( 2 ) will be omitted in this section.

The expression for x 3 is DISPLAYFORM0 Taylor expanding the last term in powers of F 1 gives DISPLAYFORM1 where in the last equality we simplified the notation for the partial derivative, where ∂/∂x = (∂/∂x 1 , . . .

, ∂/∂x D ) and D is the dimensionality of x. Counting the powers of F and F reveals that there are (1,2,1) terms for each power 0, 1 and 2, respectively.

The same (1,2,1) coefficients can also be obtained by setting the weights to be the same DISPLAYFORM2 This is similar to x 3 but with indices on the right hand side increased by 1.

One more iteration of Taylor expansion gives x 4 in terms of x 1 DISPLAYFORM3 where we have organized all terms having the same power of F and F together to be in the same row.

We also assume ReLU is used so that F 3 = 0 almost exactly.

We say that a term in the first order expansion has power k if the term is proportional to (F ) k−1 F.

Then there are (1,3,3,1) terms for each power k ∈ {1, 2, 3, 4}. A pattern begins to emerge that the number of terms for each power of F satisfy K k , where K is the number skipped, i.e. K = 3 for the x 4 to x 1 case above.

Now we show that the number of terms in the first order expansion is the binomial coefficient for all k. We aim to derive a recursion relationship between each iteration of index reduction.

We define the index reduction as operations that reduce the index of the outputs x i by one.

For instance, residual unit formula x i = x i−1 + F i−1 is an index reduction, where the index is reduced from i to i − 1.

Note that this operation generates a term of power 1, F i−1 , from a power 0 term x i .

The first order Taylor expansion generates a term of an additional power with a derivative, DISPLAYFORM4 where an index reduction is used in the first equality and the Taylor expansion is used in the second.

The dependence on F upon the weights and higher order corrections are omitted to avoid clutter.

We see the the combination of an index reduction and Taylor expansion generate terms of powers k and k + 1 with index i − 1 from a term of power k of index i. Let C(K, k) be the number of terms of K index reduction operations and power k. For instance, K = 3 corresponds to expressing x 4 in terms of x 1 as in Equation 9 with C(3, 1) = C(3, 2) = 3 and C(3, 0) = C(3, 3) = 1.

We now derive a relationship between the number of terms of power k + 1 after K + 1 index reductions with those after K index reductions.

Consider the terms corresponding to K + 1 with power k + 1.

There are two sources of such terms.

First, those generated by an additional index reduction after K operations and the zeroth order Taylor expansion in terms of power k + 1, there are C(K, k + 1) such terms.

Second, those generated by the first order Taylor expansion in terms of power k, there are C(K, k) such terms.

Therefore the total number of terms with power k + 1 after K + 1 index reductions is C(K + 1, k + 1) = C(K, k + 1) + C(K, k).

This is precisely the recursion formula satisfied by the binomial coefficients.

We have explicitly shown earlier that for K = 3 and K = 4 the coefficients are binomial coefficients.

Therefore the number of terms at any K and power k are the binomial coefficients, C(K, k) = Of course, the number of unordered subsets with cardinality k from a set of cardinality K is K k .

To write down a term of power k explicitly in the first order Taylor expansion, we first choose a unordered subset of k indices from S K then we order the indices to form σ c = {c(k), . . .

, c(1)}. Then the output after K residual units with input x i is the sum over all these subsets DISPLAYFORM0 where P(S K ) denotes the power set of S K .

Note that when σ c is empty, the right hand side gives the identity mapping.

This is the same as Equation FORMULA3 .

Setting all weights to be the same gives the form in Equation 4.

The series of index reduction operations can be identified with a Bernoulli process with parameters K and p = 0.5.

Each term in Equation (3) arises from a realization of the Bernoulli process.

Summing over terms from all possible realizations results in Equation (3).

Recall that to express x K+1 in terms of x 1 similar to Equation (3), we need K index reduction operations.

Let X K:1 := {X K , X K−1 , . . .

, X 1 } be a Bernoulli process, where X i ∼ B(K, p = 0.5).

Then the realizations X i = 0 represents the power of a term remains the same after an index reduction, and X i = 1 denotes an increase in the power of a term by one.

For example, consider K = 2, the terms corresponding to the realizations of the Bernoulli process X 2:1 = {X 2 , X 1 } are DISPLAYFORM0 One sees that x 3 can be obtained by summing over all terms corresponding to all realizations of X 3:1 .

This generalizes to X K:1 for x K+1 .

The probability of a term having power k is 2 DISPLAYFORM1 Since the total number of terms is 2 K , the number of terms having power k is the binomial coefficient K k .

If we let σ c to be the term corresponding to a realization of X K:1 , then consecutive Taylor expansions corresponds to summing over all σ c and Equation FORMULA3 follows.

<|TLDR|>

@highlight

We propose the Warped Residual Network using a parallelizable warp operator for forward and backward propagation to distant layers that trains faster than the original residual neural network. 