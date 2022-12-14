The recent success of neural networks for solving difficult decision tasks has incentivized incorporating smart decision making "at the edge."

However, this work has traditionally focused on neural network inference, rather than training, due to memory and compute limitations, especially in emerging non-volatile memory systems, where writes are energetically costly and reduce lifespan.

Yet, the ability to train at the edge is becoming increasingly important as it enables applications such as real-time adaptability to device drift and environmental variation, user customization, and federated learning across devices.

In this work, we address four key challenges for training on edge devices with non-volatile memory: low weight update density, weight quantization, low auxiliary memory, and online learning.

We present a low-rank training scheme that addresses these four challenges while maintaining computational efficiency.

We then demonstrate the technique on a representative convolutional neural network across several adaptation problems, where it out-performs standard SGD both in accuracy and in number of weight updates.

Deep neural networks have shown remarkable performance on a variety of challenging inference tasks.

As the energy efficiency of deep-learning inference accelerators improves, some models are now being deployed directly to edge devices to take advantage of increased privacy, reduced network bandwidth, and lower inference latency.

Despite edge deployment, training happens predominately in the cloud.

This limits the privacy advantages of running models on-device and results in static models that do not adapt to evolving data distributions in the field.

Efforts aimed at on-device training address some of these challenges.

Federated learning aims to keep data on-device by training models in a distributed fashion (Konecný et al., 2016) .

On-device model customization has been achieved by techniques such as weight-imprinting (Qi et al., 2018) , or by retraining limited sets of layers.

On-chip training has also been demonstrated for handling hardware imperfections (Zhang et al., 2017; Gonugondla et al., 2018) .

Despite this progress with small models, on-chip training of larger models is bottlenecked by the limited memory size and compute horsepower of edge processors.

Emerging non-volatile (NVM) memories such as resistive random access memory (RRAM) have shown great promise for energy and area-efficient inference (Yu, 2018) .

However, on-chip training requires a large number of writes to the memory, and RRAM writes cost significantly more energy than reads (e.g., 10.9 pJ/bit versus 1.76 pJ/bit (Wu et al., 2019) ).

Additionally, RRAM endurance is on the order of 10 6 writes (Grossi et al., 2019) , shortening the lifetime of a device due to memory writes for on-chip training.

In this paper, we present an online training scheme amenable to NVM memories to enable next generation edge devices.

Our contributions are (1) an algorithm called Streaming Kronecker Sum Approximation (SKS), and its analysis, which addresses the two key challenges of low write density and low auxiliary memory; (2) two techniques "gradient max-norm" and "streaming batch norm" to help training specifically in the online setting; (3) a suite of adaptation experiments to demonstrate the advantages of our approach.

Efficient training for resistive arrays.

Several works have aimed at improving the efficiency of training algorithms on resistive arrays.

Of the three weight-computations required in training (forward, backprop, and weight update), weight updates are the hardest to parallelize using the array structure.

Stochastic weight updates (Gokmen & Vlasov, 2016) allow programming of all cells in a crossbar at once, as opposed to row/column-wise updating.

Online Manhattan rule updating (Zamanidoost et al., 2015) can also be used to update all the weights at once.

Several works have proposed new memory structures to improve the efficiency of training (Soudry et al., 2015; Ambrogio et al., 2018) .

The number of writes has also been quantified in the context of chip-in-the-loop training (Yu et al., 2016) .

Distributed gradient descent.

Distributed training in the data center is another problem that suffers from expensive weight updates.

Here, the model is replicated onto many compute nodes and in each training iteration, the mini-batch is split across the nodes to compute gradients.

The distributed gradients are then accumulated on a central node that computes the updated weights and broadcasts them.

These systems can be limited by communication bandwidth, and compressed gradient techniques (Aji & Heafield, 2017) have therefore been developed.

In Lin et al. (2017) , the gradients are accumulated over multiple training iterations on each compute node and only gradients that exceed a threshold are communicated back to the central node.

In the context of on-chip training with NVM, this method helps reduce the number of weight updates.

However, the gradient accumulator requires as much memory as the weights themselves, which negates the density benefits of NVM.

Low-Rank Training.

Our work draws heavily from previous low-rank training schemes that have largely been developed for use in recurrent neural networks to uncouple the training memory requirements from the number of time steps inherent to the standard truncated backpropagation through time (TBPTT) training algorithm.

Algorithms developed since then to address the memory problem include Real-Time Recurrent Learning (RTRL) (Williams & Zipser, 1989) , Unbiased Online Recurrent Optimization (UORO) (Tallec & Ollivier, 2017) , Kronecker Factored RTRL (KF-RTRL) (Mujika et al., 2018) , and Optimal Kronecker Sums (OK) (Benzing et al., 2019) .

These latter few techniques rely on the weight gradients in a weight-vector product looking like a sum of outer products (i.e., Kronecker sums) of input vectors with backpropagated errors.

Instead of storing a growing number of these sums, they can be approximated with a low-rank representation involving fewer sums.

The meat of most deep learning systems are many weight matrix -activation vector products W · a. Fully-connected (dense) layers use them explicitly:

layer , where σ is a non-linear activation function (more details are discussed in detail in Appendix C.1).

Recurrent neural networks use one or many matrix-vector products per recurrent cell.

Convolutional layers can also be interpreted in terms of matrix-vector products by unrolling the input feature map into strided convolution-kernel-size slices.

Then, each matrix-vector product takes one such input slice and maps it to all channels of the corresponding output pixel (more details are discussed in Appendix C.2).

The ubiquity of matrix-vector products allows us to adapt the techniques discussed in "Low-Rank Training" of Section 2 to other network architectures.

Instead of reducing the memory across time steps, we can reduce the memory across training samples in the case of a traditional feedforward neural network.

However, in traditional training (e.g., on a GPU), this technique does not confer advantages.

Traditional training platforms often have ample memory to store a batch of activations and backpropagated gradients, and the weight updates ∆W can be applied directly to the weights W once they are computed, allowing temporary activation memory to be deleted.

The benefits of low-rank training only become apparent when looking at the challenges of proposed NVM devices:

Low write density (LWD).

In NVM, writing to weights at every sample is costly in energy, time, and endurance.

These concerns are exacerbated in multilevel cells, which require several steps of an iterative write-verify cycle to program the desired level.

We therefore want to minimize the number of writes to NVM.

.

NVM is the densest form of memory.

In 40nm technology, RRAM 1T-1R bitcells @ 0.085 um 2 (Chou et al., 2018) are 2.8x smaller than 6T SRAM cells @ 0.242 um 2 (TSMC, 2019).

Therefore, NVM should be used to store the memory-intensive weights.

By the same token, no other on-chip memory should come close to the size of the on-chip NVM.

In particular, if our b−bit NVM stores a weight matrix of size n o × n i , we should use at most r(n i + n o )b auxiliary non-NVM memory, where r is a small constant.

Despite these space limitations, the reason we might opt to use auxiliary (large, high endurance, low energy) memory is because there are places where writes are frequent, violating LWD if we were to use NVM.

In the traditional minibatch SGD setting with batch size B, an upper limit on the write density per cell per sample is easily seen: 1/B. However, to store such a batch of updates without intermediate writes to NVM would require auxiliary memory proportional to B. Therefore, a trade-off becomes apparent.

If B is reduced, LAM is satisfied at the cost of LWD.

If B is raised, LWD is satisfied at the cost of LAM.

Using low-rank training techniques, the auxiliary memory requirements are decoupled from the batch size, allowing us to increase B while satisfying both LWD and LAM 1 .

Additionally, because the low-rank representation uses so little memory, a larger bitwidth can be used, potentially allowing for gradient accumulation in a way that is not possible with low bitwidth NVM weights.

In the next section, we elaborate on the low-rank training method.

Let z (i) = W a (i) + b be the standard affine transformation building block of some larger network, e.g., y

where

.

A minibatch SGD weight update accumulates this gradient over B samples:

For a rank-r training scheme, approximate the sum

by iteratively updating two rankr matricesL ∈ R no×r ,R ∈ R ni×r with each new outer product:

.

Therefore, at each sample, we convert the rank-q = r + 1 systemLR + dz

into the rank-rLR .

In the next sections, we discuss how to compute rankReduce.

One option for rankReduce(X) to convert from rank q = r + 1 X to rank r is a minimum error estimator, which is implemented by selecting the top r components of a singular value decomposition (SVD) of X. However, a naïve implementation is computationally infeasible and biased: Benzing et al. (2019) solves these problems by proposing a minimum variance unbiased estimator for rankReduce, which they call the OK algorithm 2 .

The OK algorithm can be understood in two key steps: first, an efficient method of computing the SVD of a Kronecker sum; second, a method of splitting the singular value matrix Σ into two rank-r matrices whose outer product is a minimum-variance, unbiased estimate of Σ. Details can be found in their paper, however we include a high-level explanation in Sections 4.1.1 and 4.1.2 to aid our discussions.

Note that our variable notation differs from Benzing et al. (2019) .

Recall that rankReduce should turn rank-q LR into an updated rank-rLR .

q×q .

Then we can find the SVD of & Dhillon, 2006) , making it computationally feasible on small devices.

Now we have:

which gives the SVD of LR since Q L U C and Q R V C are orthogonal and Σ is diagonal.

This SVD computation has a time complexity of O((n i +n o +q)q 2 ) and a space complexity of O((n i +n o +q)q).

In Benzing et al. (2019) , it is shown that the problem of finding a rank-r minimum variance unbiased estimator of LR can be reduced to the problem of finding a rank-r minimum variance unbiased estimator of Σ and plugging it in to (1).

Further, it is shown that such an optimal approximator for Σ = diag(σ 1 , σ 2 , . . .

, σ q ), where σ 1 ≥ σ 2 ≥ · · · ≥ σ q will involve keeping the m − 1 largest singular values and mixing the smaller singular values σ m , . . .

, σ q within their (k + 1) × (k + 1) submatrix with m, k defined below.

Let:

Note that ||x 0 || 2 = 1.

Let X ∈ R (k+1)×(k) be orthogonal such that its left nullspace is the span of x 0 .

Then XX = I − x 0 x 0 .

Now, let s ∈ {−1, 1} (k+1)×1 be uniform random signs and define:

where is an element-wise product.

ThenΣ LΣ R =Σ is a minimum variance, unbiased 3 rank-r approximation of Σ. PluggingΣ into (1),

gives us a minimum variance, unbiased, rank-r approximationLR .

Although the standalone OK algorithm presented by Benzing et al. (2019) has good asymptotic computational complexity, our vector-vector outer product sum use case permits further optimizations.

In this section we present these optimizations, and we refer readers to the explicit implementation called Streaming Kronecker Sum Approximation (SKS) in Algorithm 1 of Appendix A.

The main optimization is a method of avoiding recomputing the QR factorization of L and R at every step.

Instead, we keep track of orthogonal matrices Q L , Q R , and weightings c x such that

Upon receiving a new sample, a single inner loop of the numerically-stable modified Gram-Schmidt (MGS) algorithm (Björck, 1967) can be used to update Q L and

computed during MGS can be used to find the new value of C = c L c R + diag(c x ).

3 The fact that it is unbiased: E[Σ] = Σ can be easily verified.

After computingΣ L =Σ R in (2), we can orthogonalize these matrices intoΣ

.

With this formulation, we can maintain orthogonality in Q L , Q R by setting:

These matrix multiplies require O((n i +n o )q 2 ) multiplications, so this optimization does not improve asymptotic complexity bounds.

This optimization may nonetheless be practically significant since matrix multiplies are easy to parallelize and would typically not be the bottleneck of the computation compared to Gram-Schmidt.

The next section discusses how to orthogonalizeΣ L efficiently and why

Orthogonalization ofΣ L is relatively straightforward.

From (2), the columns ofΣ L are orthogonal since Z is orthogonal.

However, they do not have unit norm.

We can therefore pull out the norm into a separate diagonal matrix R x with diagonal elements √ c x :

We generated X by finding an orthonormal basis that was orthogonal to a vector x 0 so that we could have XX = I − x 0 x 0 .

An efficient method of producing this basis is through Householder matrices (x 0 , X) = I − 2 vv /||v|| 2 where v = x 0 − e (1) and (x 0 , X) is a k + 1 × k + 1 matrix with first column x 0 and remaining columns X (Householder, 1958; user1551, 2013 .

The OK/SKS methods require O((n i + n o + q)q

2 ) operations per sample and O(n i n o q) operations after collecting B samples, giving an amortized cost of O((n i + n o + q)q 2 + n i n o q/B) operations per sample.

Meanwhile, a standard approach expands the Kronecker sum at each sample, costing O(n i n o ) operations per sample.

If q B, n i , n o then the low rank method is superior to minibatch SGD in both memory and computational cost.

SKS introduces variance into the gradient estimates, so here we analyze the implications for online convex convergence.

We analyze the case of strongly convex loss landscapes f t (w t ) for flattened weight vector w t and online sample t. In Appendix B, we show that with inverse squareroot learning rate, when the loss landscape Hessians satisfy 0 ≺ cI ∇ 2 f t (w t ) and under constraint (4) for the size of gradient errors ε t , where w * is the optimal offline weight vector, the online regret (5) is sublinear in the number of online steps T .

We can approximate ||ε|| and show that convex convergence is likely when (6) is satisfied in the biased, zero-variance case (equivalent to raw SVD, i.e., not applying Section 4.1.2), or when (7) is satisfied in the unbiased, minimum-variance case.

Equations (6, 7) suggest conditions under which fast convergence may be more or less likely and also point to methods for improving convergence.

We discuss these in more detail in Appendix B.3.

We validate (4) with several linear regression experiments on a static input batch X ∈ R

and target Y t ∈ R 256×100 .

In Figure 1 (a), Gaussian noise at different strengths (represented by different colors) is added to the true batch gradients at each update step.

Notice that convergence slows significantly to the right of the dashed lines, which is the region where (4) no longer holds 4 .

In Figure 1 (b), we validate Equations (4, 6, 7) by testing the SVD and SKS cases with rank r = 10.

In these particular experiments, SKS adds too much variance, causing it to operate to the right of the dashed lines.

However, both SVD and SKS can be seen to reduce their variance as training progresses.

In the case of SVD, it is able to continue training as it tracks the right dashed line.

Quantization.

The NN is quantized in both the forward and backward directions with uniform power-of-2 quantization, where the clipping ranges are fixed at the start of training 5 .

Weights are quantized to 8 bits between -1 and 1, biases to 16 bits between -8 and 8, activations to 8 bits between 0 and 2, and gradients to 8 bits between -1 and 1.

Both the weights W and weight updates ∆W are quantized to the same LSB so that weights cannot be used for accumulation beyond the fixed quantization dynamic range.

This is in contrast to using high bitwidth (Zhou et al., 2016; Banner et al., 2018) or floating point accumulators.

See Appendix D for more details on quantization.

Gradient Max-Norming.

State-of-the-art methods in training, such as Adam (Kingma & Ba, 2014) , use auxiliary memory per parameter to normalize the gradients.

Unfortunately, we lack the memory budget to support these additional variables, especially if they must be updated every sample 6 .

Instead, we propose dividing each gradient tensor by the maximum absolute value of its elements.

This stabilizes the range of gradients across samples.

See Appendix E for more details on gradient max-norming.

In the experiments, we refer to this method as "max-norm" (opposite "no-norm").

Streaming Batch Normalization.

Batch normalization (Ioffe & Szegedy, 2015 ) is a powerful technique for improving training performance which has been suggested to work by smoothing the loss landscape (Santurkar et al., 2018) .

We hypothesize that this may be especially helpful when parameters are quantized as in our case.

However, in the online setting, we receive samples one-at-a-time rather than in batches.

We therefore propose a streaming batch norm that uses moving average statistics rather than batch statistics as described in detail in Appendix F.

To test the effectiveness of SKS, experiments are performed on a representative CNN with four 3 × 3 convolution layers and two fully-connected layers.

We generate "offline" and "online" datasets based on MNIST (see Appendix G), including one in which the statistical distribution shifts every 10k images.

We then optimize an online SGD and rank-4 SKS model for fair comparison (see Appendix H).

To see the importance of different training techniques, we run several ablations in Appendix I. Finally, we compare these different training schemes in different environments, meant to model real life.

In these hypothetical scenarios, a model is first trained on the offline training set, and is then deployed to a number of devices at the edge that make supervised predictions (they make a prediction, then are told what the correct prediction would have been).

We present results on four hypothetical scenarios.

First, a control case where both external/environment and internal/NVM drift statistics are exactly the same as during offline training.

Second, a case where the input image statistical distribution shifts every 10k samples, selecting from augmentations such as spatial transforms and background gradients (see Section G).

Third and fourth are cases where the NVM drifts from the programmed values, roughly modeling NVM memory degradation.

In the third case, Gaussian noise is applied to the weights as if each weight was a single multi-level memory cell whose analog value drifted in a Brownian way.

In the fourth case, random bit flips are applied as if each weight was represented by b memory cells (see Appendix G for details).

For each hypothetical scenario, we plot five different training schemes: pure quantized inference (no training), bias-only training, standard SGD training, SKS training, and SKS training with max-normed gradients.

In SGD training and for training biases, parameters are updated at every step in an online fashion.

These are seen as different colored curves in Figure 2 .

Inference does best in the control case, but does poorly in adaptation experiments.

SGD doesn't improve significantly on bias-only training, likely because SGD cannot accumulate gradients less than a weight LSB.

SKS, on the other hand, shows significant improvement, especially after several thousand samples in the weight drift cases.

Additionally, SKS shows about three orders of magnitude improvement compared to SGD in the worst case number of weight updates.

Much of this reduction is due to the convolutions, where updates are applied at each pixel.

However, reduction in fullyconnected writes is still important because of potential energy savings.

SKS/max-norm performs best in terms of accuracy across all environments and has similar weight update cost to SKS/no-norm.

To test the broader applicability of low rank training techniques, we run several experiments on ImageNet with ResNet-34 (Deng et al., 2009; He et al., 2016) , a potentially realistic target for dense NVM inference on-chip.

For ImageNet-size images, updating the low-rank approximation at each pixel quickly becomes infeasible, both because of the single-threaded nature of the algorithm, and because of the increased variance of the estimate at larger batch sizes.

Instead, we focus on training the final layer weights (1000 × 512).

ResNet-34 weights are initialized to those from Paszke et al. (2017) and the convolution layers are used to generate feature vectors for 10k ImageNet training images 7 , which are quantized and fed to a one-layer quantized 8 neural network.

To speed up experiments, the layer weights are initialized to the pretrain weights, modulated by random noise that causes inference top-1 accuracy to fall to 52.7% ± 0.9%.

In Table 1 , we see that the unbiased SKS has the strongest recovery accuracies, although biased SVD also does quite well.

The high-variance UORO and true SGD have weak or non-existent recoveries.

-+0.3 ± 0.2 +0.3 ± 0.2 +0.3 ± 0.2 +0.9 ± 0.2 −3.9 ± 0.8 UORO 1 +0.4 ± 0.2 +0.3 ± 0.4 −1.8 ± 0.9 −7.6 ± 1.6 −31.7 ± 1.6 SVD 1 +1.9 ± 0.2 +5.8 ± 1.0 −3.4 ± 1.0 −19.4 ± 0.9 −40.7 ± 1.1 2 +1.4 ± 0.4 +6.5 ± 0.7 +6.3 ± 0.6 −5.2 ± 0.9 −36.3 ± 0.9 4 +1.3 ± 0.4 +6.5 ± 0.7 +5.2 ± 0.8 −3.3 ± 1.0 −33.8 ± 0.8 8 +1.4 ± 0.3 +5.6 ± 0.8 +4.3 ± 0.9 −2.4 ± 1.0 −32.8 ± 0.9 SKS 1 +0.3 ± 0.2 +0.3 ± 0.2 −0.7 ± 0.4 −2.7 ± 1.7 −26.5 ± 2.6 2 +0.3 ± 0.2 +0.4 ± 0.3 −0.1 ± 0.4 +1.3 ± 0.9 −12.9 ± 1.1 4 +0.4 ± 0.2 +0.6 ± 0.2 +1.9 ± 0.3 +8.0 ± 1.1 −5.1 ± 1.1 8 +0.4 ± 0.2 +1.1 ± 0.2 +3.3 ± 0.7 +4.8 ± 1.5 −15.8 ± 1.7

We demonstrated the potential for SKS to solve the major challenges facing online training on NVM-based edge devices: low write density and low auxiliary memory.

SKS is a computationallyefficient, memory-light algorithm capable of decoupling batch size from auxiliary memory, allowing larger effective batch sizes, and consequently lower write densities.

Additionally, we noted that SKS may allow for training under severe weight quantization constraints as rudimentary gradient accumulations are handled by the L, R matrices, which can have high bitwidths (as opposed to SGD, which may squash small gradients to 0).

We found expressions for when SKS might have better convergence properties.

Across a variety of online adaptation problems and a large-scale transfer learning demonstration, SKS was shown to match or exceed the performance of SGD while using a small fraction of the number of updates.

Finally, we suspect that these techniques could be applied to a broader range of problems.

Auxiliary memory minimization may be analogous to communication minimization in training strategies such as federated learning, where gradient compression is important.

State:

In this section we will attempt to bound the regret (defined below) of an SGD algorithm using noisy SKS estimatesg = g + ε in the convex setting, where g are the true gradients and ε are the errors introduced by the low rank SKS approximation.

Here, g is a vector of size N and can be thought of as a flattened/concatenated version of the gradient tensors (e.g., N = n i · n o ).

Our proof follows the proof in Zinkevich (2003) .

We define F as the convex feasible set (valid settings for our weight tensors) and assume that F is bounded with D = max w,v∈F ||w − v|| being the maximum distance between two elements of F. Further, assume a batch t of B samples out of T total batches corresponds to a loss landscape f t (w t ) that is strongly convex in weight parameters w t , so there are positive constants C ≥ c > 0 such that cI ∇ 2 f t (w t ) CI for all t (Boyd & Vandenberghe, 2004, Section 9.

3).

We define regret as

where w * = argmin w T t=1 f t (w) (i.e., it is an optimal offline minimizer of

The gradients seen during SGD are g t = ∇f t (w t ) and we assume they are bounded by G = max w∈F,t∈ [1,T ]

For the unbiased, minimum-variance case, Theorem A.4 from Benzing et al. (2019) states that the minimum variance is s .

Assuming errors between samples are uncorrelated, this leads to a total variance:

For either case, ||ε|| 2 ≈ N σ 2 ε .

For the t-th batch and i-th sample, we denote σ (t,i) q as the q-th singular value.

For simplicity, we focus on the biased, zero-variance case (the unbiased case is similar).

From (15), an approximately sufficient condition for sublinear-regret convergence is:

B.3 DISCUSSION ON CONVERGENCE Equation (19) suggests that as w t → w * , the constraints for achieving sublinear-regret convergence become more difficult to maintain.

However, in practice this may be highly problem-dependent as the σ q will also tend to decrease near optimal solutions.

To get a better sense of the behavior of the left-hand side of (19), suppose that: (no×ni) are the matrix weight W t gradients at batch t and || · || F is a Frobenius norm.

We therefore expect both the left (proportional to ||G t || 2 F ) and the right (proportional to ||w t − w * || 2 ) of (19) to decrease during training as w t → w * .

This behavior is in fact what is seen in Figure 1(b) .

If achieving convergence is found to be difficult, (19) provides some insight for convergence improvement methods.

One solution is to reduce batch size B to satisfy the inequality as necessary.

This minimizes the weight updates during more repetitive parts of training while allowing dense weight updates (possibly approaching standard SGD with small batch sizes) during more challenging parts of training.

Another solution is to reduce σ q .

One way to do this is to increase the rank r so that the spectral energy of the updates are spread across more singular components.

There may be alternate approaches based on conditioning the inputs to shape the distribution of singular values in a beneficial way.

A third method is to focus on c, the lower bound on curvature of the convex loss functions.

Perhaps a technique such as weight regularization can increase c by adding constant curvature in all Eigendirections of the loss function Hessian (although this may also increase the LHS of (19)).

Alternatively, perhaps low-curvature Eigen-directions are less important for loss minimization, allowing us to raise the c that we effectively care about.

This latter approach requires no particular action on our part, except the recognition that fast convergence may only be guaranteed for high-curvature directions.

This is exemplified in Figure 1(b) , where we can see SVD track the curve for C more so than c.

Finally, we note that this analysis focuses solely on the errors introduced by a floating-point version of SKS.

Quantization noise can add additional error into the ε t term.

We expect this to add a constant offset to the LHS of (19).

For a weight LSB ∆, quantization noise has variance ∆ 2 /12, so we desire:

C KRONECKER SUMS IN NEURAL NETWORK LAYERS

A dense or fully-connected layer transforms an input a ∈ R ni×1 to an intermediate z = W · a + b to an output y = σ(z) ∈ R no×1 where σ is a non-linear activation function.

Gradients of the loss function with respect to the weight parameters can be found as:

which is exactly the per-sample Kronecker sum update we saw in linear regression.

Thus, at every training sample, we can add (dz (i) ⊗ a (i) ) to our low rank estimate with SKS.

A convolutional layer transforms an input feature map A ∈ R hin×win×cin to an intermediate feature map Z = W kern * A + b ∈ R hout×wout×cout through a 2D convolution * with weight kernel W kern ∈ R cout×k h ×kw×cin .

Then it computes an output feature map y = σ(z) where σ is a non-linear activation function.

Convolutions can be interpreted as matrix multiplications through the im2col operation which converts the input feature map A into a matrix A col ∈ R (houtwout)×(k h kwcin) where the i th row is a flattened version of the sub-tensor of a which is dotted with W kern to produce the i th pixel of the output feature map (Ren & Xu, 2015) .

We can multiply A col by a flattened version of the kernel, W ∈ R cout×(k h hwcin) to perform the W kern * A convolution operation with a matrix multiplication.

Under the matrix multiplication interpretation, weight gradients can be represented as:

which is the same as h out w out Kronecker sum updates.

Thus, at every output pixel j of every training sample i, we can add (dZ

col,j ) to our low rank estimate with SKS.

Note that while we already save an impressive factor of B/q in memory when computing gradients for the dense layer, we save a much larger factor of Bh out w out /q in memory when computing gradients for the convolution layers, making the low rank training technique even more crucial here.

However, some care must be taken when considering activation memory for convolutions.

For compute-constrained edge devices, image dimensions may be small and result in minimal intermediate feature map memory requirements.

However, if image dimensions grow substantially, activation memory could dominate compared to weight storage.

Clever dataflow strategies may provide a way to reduce intermediate activation storage even when performing backpropagation 9 .

In a real device, operations are expected to be performed in fixed point arithmetic.

Therefore, all of our training experiments are conducted with quantization in the loop.

Our model for quantization is shown in Figure 3 .

The green arrows describe the forward computation.

Ignoring quantization for a moment, we would have a = ReLU α W * a −1 + b , where * can represent either a convolution or a matrix multiply depending on the layer type and α is the closest power-of-2 to He initialization (He et al., 2015) .

For quantization, we rely on four basic quantizers: Qw, Qb, Qa, Qg, which describe weight quantization, bias and intermediate accumulator quantization, activation quantization, and gradient quantization, respectively.

All quantizers use fixed clipping ranges as depicted and quantize uniformly within those ranges to the specified bitwidths.

In the backward pass, follow the orange arrows from δ .

Backpropagation follows standard backpropagation rules including using the straight-through estimator (Bengio et al., 2013) for quantizer gradients.

However, because we want to perform training on edge devices, these gradients must themselves be quantized.

The first place this happens is after passing backward through the ReLU derivitive.

The other two places are before feeding back into the network parameters W , b , so that W , b cannot be used to accumulate values smaller than their LSB.

Finally, instead of deriving ∆W from a backward pass through the * operator, the SKS method is used.

SKS collects a −1 , dz for many samples before computing the approximate ∆W .

It accumulates information in two low rank matrices L, R which are themselves quantized to 16 bits with clipping ranges determined dynamically by the max absolute value of elements in each matrix.

While SKS accumulates for B samples, leading to a factor of B reduction in the rate of updates to W , b is updated at every sample.

This is feasible in hardware because b is small enough to be stored in more expensive forms of memory that have superior endurance and write power performance.

Because of the coarse weight LSB size, weight gradients may be consistently quantized to 0, preventing them from accumulating.

To combat this, we only apply an update if a minimum update density ρ min = 0.01 would be achieved, otherwise we continue accumulating samples in L and R, which have much higher bitwidths.

When an update does finally happen, the "effective batch size" will be a multiple of B and we increase the learning rate correspondingly.

In the literature, a linear scaling rule is suggested (see Goyal et al. (2017) ), however we empirically find square-root scaling works better (see Appendix H).

E GRADIENT MAX-NORMING Figure 4 : Maximum magnitude of weight gradients versus training step for standard SGD on a CNN trained on MNIST.

Figure 4 plots the magnitude of gradients seen in a weight tensor over training steps.

One apparent property of these gradients is that they have a large dynamic range, making them difficult to quantize.

Even when looking at just the spikes, they assume a wide range of magnitudes.

One potential method of dealing with this dynamic range is to scale tensors so that their max absolute element is 1 (similar to a per-tensor AdaMax (Kingma & Ba, 2014) or Range Batch-Norm (Banner et al., 2018) applied to gradients).

Optimizers such as Adam, which normalize by gradient variance, provide a justification for why this sort of scaling might work well, although they work at a per-element rather than per-tensor level.

We choose max-norming rather than variance-based norming because the former is easier computational and potentially more ammenable to quantization.

However, a problem with the approach of normalizing tensors independently at each sample is that noise might be magnified during regions of quiet as seen in the Figure.

What we therefore propose is normalization by the maximum of both the current max element and a moving average of the max element.

Explicitly, max-norm takes two parameters -a decay factor β = 0.999 and a gradient floor ε = 10 −4

and keeps two state variables -the number of evaluations k := 0 and the current maximum moving average x mv := ε.

Then for a given input x, max-norm modifies its internal state and returns x norm :

Standard batch normalization (Ioffe & Szegedy, 2015) normalizes a tensor X along some axes, then applies a trainable affine transformation.

For each slice X of X that is normalized independently:

where µ b , σ b are mean and standard deviation statistics of a minibatch and γ, β are trainable affine transformation parameters.

In our case, we do not have the memory to hold a batch of samples at a time and must compute µ b , σ b in an online fashion.

To see how this works, suppose we knew the statistics of each sample µ i , σ i for i = 1 . . .

B in a batch of B samples.

For simplicity, assume the i th sample is a vector X i,: ∈ R n containing elements X i,j .

Then:

In other words, the batch variance is not equal to the average of the sample variances.

However, if we keep track of the sum-of-square values of samples σ for each sample i. After B samples, we divide both state variables by B and apply (23, 24) to get the desired batch statistics.

Unfortunately, in an online setting, all samples prior to the last one in a given batch will only see statistics generated from a portion of the batch, resulting in noisier estimates of µ b , σ b .

In streaming batch norm, we alter the above formula slightly.

Notice that in online training, only the most recently viewed sample is used for training, so there is no reason to weight different samples of a given batch equally.

Therefore we can use an exponential moving average instead of a true average to track µ s , sq s .

Specifically, let:

If we set η = 1 − 1/B, a weighting of 1/B is seen on the current sample, just as in standard averages with a batch of size B, but now all samples receive similarly clean batch statistic estimates, not just the last few samples in a batch.

For our experiments, we construct a dataset comprising an offline training, validation, and test set, as well as an online training set.

Specifically, we start with the standard MNIST dataset of LeCun et al. (1998) and split the 60k training images into partitions of size 9k, 1k, and 50k.

Elastic transforms (Simard et al., 2003; Ernestus, 2016) are used to augment each of these partitions to 50k offline training samples, 10k offline validation samples, and 100k online training samples, respectively.

Elastic transforms are also applied to the 10k MNIST test images to generate the offline test samples.

The source images for the 100k online training samples are randomly drawn with replacement, so there is a certain amount of data leakage in that an online algorithm may be graded on an image that has been generated from the same image a previous sample it has trained on has been generated from.

This is intentional and is meant to mimic a real-life scenario where a deployed device is likely to see a restrictive and repetitive set of training samples.

Our experiments include comparisons to standard SGD to show that SKS's improvement is not merely due to overfitting the source images.

From the online training set, we also generate a "distribution shift" dataset by applying unique additional augmentations to every contiguous 10k samples of the 100k online training samples.

Four types of augmentations are explored.

Class distribution clustering biases training samples belonging to similar classes to have similar indices.

For example, the first thousand images may be primarily "0"s and "3"s, whereas the next thousand might have many "5"s.

Spatial transforms rotate, scale, and shift images by random amounts.

Background gradients both scale the contrast of the images and apply black-white gradients across the image.

Finally, white noise is random Gaussian noise added to each pixel.

In addition to distribution shift for testing adaptation, we also look at internal statistical shift of weights in two ways -analog and digital.

For analog weight drift, we apply independent additive Gaussian noise to each weight every d = 10 steps with σ = σ 0 / 1M/d where σ 0 = 10 and re-clip the weights between -1 and 1.

This can be interpreted as each cell having a Gaussian cumulative error with σ = σ 0 after 1M steps.

For digital weight drift, we apply independent binary random flips to the weight matrix bits every d steps with probability p = p 0 /(1M/d) where p 0 = 10.

This can be interpreted as each cell flipping an average of p 0 times over 1M steps.

Note that in real life, σ 0 , p 0 depend on a host of issues such as the environmental conditions of the device (temperature, humidity, etc), as well as the rate of seeing training samples.

In order to compare standard SGD with the SKS approach, we sweep the learning rates of both to optimize accuracy.

In Figure 6 , we compare accuracies across a range of learning rates for four different cases: SGD or SKS with or without max-norming gradients.

Optimal accuracies are found when learning rate is around 0.01 for all cases.

For most experiments, 8b weights, activations, and gradients, and 16b biases are used.

Experiments similar to those in Section I are used to select some of the hyperparameters related to the SKS method in particular.

In most experiments, rank-4 SKS with batch sizes of 10 (for convolution layers) or 100 (for fully-connected layers) are used.

Additional details can be found in the supplemental code.

Accuracy (Last 500 of 10k) Figure 6 : The left two heat maps are used to select the base / standard SGD learning rate.

The right two heat maps are used to select the SKS learning rate using the optimal SGD learning rate for bias training from the previous sweeps.

For the SKS sweeps, the learning rate is scaled proportional to the square-root of the batch size B. This results in an approximately constant optimal learning rate across batch size, especially for the max-norm case.

Accuracy is reported averaged over the last 500 samples from a 10k portion of the online training set, trained from scratch.

In Figure 7 , rank and weight bitwidth is swept for SKS with gradient max-norming.

As expected, training accuracy improves with both higher SKS rank and bitwidth.

In dense NVM applications, higher bitwidths may be achievable, allowing for corresponding reductions in the SKS rank and therefore, reductions in the auxiliary memory requirements.

In Table 2 , biased (zero-variance) and unbiased (low-variance) versions of SKS are compared.

Accuracy improvements are generally seen moving from biased to unbiased SKS although the pattern differs between the no-norm and max-norm cases.

In the no-norm case, a significant improvement is seen favoring unbiased SKS for fully-connected layers.

In the max-norm case, the choice of biased or unbiased SKS has only a minor impact on accuracy.

It might be expected that as the number of accumulated samples for a given pseduobatch increases, lower variance would be increasingly important at the expense of bias.

For our network, this implies convolutions, which receive updates at every pixel of an output feature map, would preferentially have biased SKS, while the fully-connected layer would preferentially be unbiased.

This hypothesis is supported by the no-norm experiments, but not by the max-norm experiments.

In Table 3 , several ablations are performed on SKS with max-norm.

Most notably, weight training is found to be extremely important for accuracy as bias-only training shows a ≈ 15 − 30% accuracy hit depending on whether max-norming is used.

Streaming batch norm is also found to be quite helpful, especially in the no-norm case.

Now, we explain the κ th ablation.

In Section 4.1.1, we found the SVD of a small matrix C and its singular values σ 1 , . . .

, σ q .

This allows us to easily find the condition number of C as κ(C) = σ 1 /σ q .

We suspect high condition numbers provide relatively useless update information akin to noise, especially in the presence of L, R quantization.

Therefore, we prefer not to update L, R on samples whose condition number exceeds threshold κ th .

We can avoid performing an actual SVD (saving computation) by noting that C is often nearly diagonal, leading to the approximation κ(C) ≈ C 1,1 /C q,q .

Empirically, this rough heuristic works well to reduce computation load while having minor impact on accuracy.

In Table 3 , κ th = 10 8 does not appear to ubiquitously improve on the default κ th = 100, despite being ≈ 2× slower to compute.

Table 3 : Miscellaneous selected ablations.

Accuracy is calculated from the last 500 samples of 10k samples trained from scratch.

Mean and unbiased standard deviation are calculated from five runs of different random seeds.

Accuracy (no-norm) Accuracy (max-norm) baseline (no modifications) 80.2% ± 1.0%

83.0% ± 1.1% bias-only training 51.8% ± 3.2% 68.6% ± 1.4% no streaming batch norm 68.2% ± 1.9% 81.8% ± 1.3% no bias training 81.3% ± 1.0% 83.0% ± 1.4% κ th = 10 8 instead of 100 79.8% ± 1.4% 84.2% ± 1.4%

@highlight

We use Kronecker sum approximations for low-rank training to address challenges in training neural networks on edge devices that utilize emerging memory technologies.