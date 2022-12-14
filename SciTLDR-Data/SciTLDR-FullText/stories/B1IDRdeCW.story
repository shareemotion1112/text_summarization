Recent research has shown that one can train a neural network with binary weights and activations at train time by augmenting the weights with a high-precision continuous latent variable that accumulates small changes from stochastic gradient descent.

However, there is a dearth of work to explain why one can effectively capture the features in data with binary weights and activations.

Our main result is that the neural networks with binary weights and activations trained using the method of Courbariaux, Hubara et al. (2016) work because of the high-dimensional geometry of binary vectors.

In particular, the ideal continuous vectors that extract out features in the intermediate representations of these BNNs are well-approximated by binary vectors in the sense that dot products are approximately preserved.

Compared to previous research that demonstrated good classification performance with BNNs, our work explains why these BNNs work in terms of HD geometry.

Furthermore, the results and analysis used on BNNs are shown to generalize to neural networks with ternary weights and activations.

Our theory serves as a foundation for understanding not only BNNs but a variety of methods that seek to compress traditional neural networks.

Furthermore, a better understanding of multilayer binary neural networks serves as a starting point for generalizing BNNs to other neural network architectures such as recurrent neural networks.

The rapidly decreasing cost of computation has driven many successes in the field of deep learning in recent years.

Consequently, researchers are now considering applications of deep learning in resource limited hardware such as neuromorphic chips, embedded devices and smart phones , BID26 , BID2 ).

A recent realization for both theoretical researchers and industry practitioners is that traditional neural networks can be compressed because they are highly over-parameterized.

While there has been a large amount of experimental work dedicated to compressing neural networks (Sec. 2), we focus on the particular approach that replaces costly 32-bit floating point multiplications with cheap binary operations.

Our analysis reveals a simple geometric picture based on the geometry of high dimensional binary vectors that allows us to understand the successes of the recent efforts to compress neural networks.

and showed that one can efficiently train neural networks with binary weights and activations that have similar performance to their continuous counterparts.

Such BNNs execute 7 times faster using a dedicated GPU kernel at test time.

Furthermore, they argue that such BNNs require at least a factor of 32 fewer memory accesses at test time that should result in an even larger energy savings.

There are two key ideas in their papers FIG0 .

First, a continuous weight, w c , is associated with each binary weight, w b , that accumulates small changes from stochastic gradient descent.

Second, the non-differentiable binarize function (??(x) = 1 if x > 0 and ???1 otherwise) is replaced with a continuous one during backpropagation.

These modifications allow one to train neural networks that have binary weights and activations with stochastic gradient descent.

While the work showed how to train such networks, the existence of neural networks with binary weights and activations needs to be reconciled with previous work that has sought to understand weight matrices as extracting out continuous features in data (e.g. BID30 ).

Summary of contributions: Each oval corresponds to a tensor and the derivative of the cost with respect to that tensor.

Rectangles correspond to transformers that specify forward and backward propagation functions.

Associated with each binary weight, w b , is a continuous weight, w c , that is used to accumulate gradients.

k denotes the kth layer of the network.

(b) Each binarize transformer has a forward function and a backward function.

The forward function simply binarizes the inputs.

In the backward propagation step, one normally computes the derivative of the cost with respect to the input of a transformer via the Jacobian of the forward function and the derivative of the cost with respect to the output of that transformer (??u ??? dC/du where C is the cost function used to train the network).

Since the binarize function is non-differentiable, the straight-through estimator BID3 ), which is a smoothed version of the forward function, is used for the backward function .

the direction of high-dimensional vectors.

In particular, we show that the angle between a random vector (from a standard normal distribution) and its binarized version converges to arccos 2/?? ??? 37??? as the dimension of the vector goes to infinity.

This angle is an exceedingly small angle in high dimensions.

Furthermore, we show that this property is present in the weight vectors of a network trained using the method of .2.

Dot Product Proportionality Property:

First, we empirically show that the weight-activation dot products, an important intermediate quantity a neural network, are approximately proportional under the binarization of the weight vectors.

Next, we argue that if these weight activation dot products are proportional, then the continuous weights in the method aren't just a learning artifact.

The continuous weights obtained from the BNN training algorithm (which decouples the forward and backward propagation steps) are an approximation of the weights one would learn if the network were trained with continuous weights and regular backpropagation.

We show that the computations done by the first layer of the network are fundamentally different than the computations being done in the rest of the network because correlations in the data result in high variance principal components that are not randomly oriented relative to the binarization.

Thus we recommend an architecture that uses a continuous convolution for the first layer to embed the image in a high-dimensional binary space, after which it can be manipulated with cheap binary operations.

Furthermore, we illustrate how a GBT (rotate, binarize, rotate back) is useful for embedding low dimensional data in a high-dimensional binary space.4.

Generalization to Ternary Neural Networks: We show that the same analysis applies to ternary neural networks.

In particular, the angle between a random vector from a standard normal distribution and the ternarized version of that vector predicts the empirical distribution of such angles in a network trained on CIFAR10.

Furthermore, the dot product proportionality property is shown to hold for ternary neural networks.

Neural networks that achieve good performance on tasks such as IMAGENET object recognition are highly computationally intensive.

For instance, AlexNet has 61 million parameters and executes 1.5 billion operations to classify one 224 by 224 image (30 thousand operations/pixel) BID28 ).

Researchers have sought to reduce this computational cost for embedded applications using a number of different approaches.

The first approach is to try and compress a pre-trained network.

BID19 uses a Tucker decomposition of the kernel tensor and fine tunes the network afterwards.

BID13 train a network, prune low magnitude connections, and retrain.

BID12 extend their previous work to additionally include a weight sharing quantization step and Huffman coding of the weights.

More recently, BID14 train a dense network, sparsify it, and then retrain a dense network with the pruned weights initialized to zero.

Second, researchers have sought to train networks using either low precision floating point numbers or fixed point numbers, which allow for cheaper multiplications BID4 , BID10 , BID16 , BID11 , BID23 , BID21 ).Third, one can train networks that have quantized weights and or activations.

BID3 looked at estimators for the gradient through a stochastic binary unit.

train networks with binary weights, and then later with binary weights and activations ).

BID28 replace a continuous weight matrix with a scalar times a binary matrix (and have a similar approximation for weight activation dot products).

BID18 train a network with weights restricted in the range ???1 to 1 and then use a noisy backpropagation scheme train a network with binary weights and activations.

BID1 , BID22 and BID33 focus on networks with ternary weights.

Further work seeks to quantize the weights and activations in neural networks to an arbitrary number of bits BID32 , ).

BID31 use weights and activations that are zero or powers of two.

BID24 and BID32 quantize backpropagation in addition to the forward propagation.

Beyond merely seeking to compress neural networks, the analysis of the internal representations of neural networks is useful for understanding how to to compress them.

BID0 found that feature magnitudes in higher layers do not matter (e.g. binarizing features barely changes classification performance).

analyze the robustness of neural network representations to a collection of different distortions.

BID7 observe that binarizing features in intermediate layers of a CNN and then using backpropagation to find an image with those features leads to relatively little distortion of the image compared to dropping out features.

These papers naturally lead into our work where we are seeking to better understand the representations in neural networks based on the geometry of high-dimensional binary vectors.

We investigate the internal representations of neural networks with binary weights and activations.

A binary neural network is trained on CIFAR-10 (same learning algorithm and architecture as in ).

Experiments on MNIST were carried out using both fully connected and convolutional networks and produced similar results.

The CIFAR-10 convolutional neural network has six layers of convolutions, all of which have a 3 by 3 spatial kernel.

The number of feature maps in each layer are 128, 128, 256, 256, 512, 512 .

After the second, fourth, and sixth convolutions, there is a 2 by 2 max pooling operation.

Then there are two fully connected layers with 1024 units each.

Each layer has a batch norm layer in between.

The experiments using ternary neural networks use the same network architecture.

The dimensionality of the weight vectors in these networks (i.e. convolution converted to a matrix multiply) is the patch size (= 3 * 3 = 9) times the number of channels.

In this section, we analyze the angle distributions (i.e. geometry) of high-dimensional binary vectors.

This is crucial for understanding binary neural networks because we can imagine that at each layer of a neural network, there are some ideal continuous weight vectors that extract out features.

A binary neural network approximates these ideal continuous vectors with a binary vectors.

In low dimensions, binarization strongly impacts the direction of a vector.

However, we argue that binarization does not substantially change the direction of a high-dimensional continuous vector.

It is often the case that the geometric properties of high-dimensional vectors are counter-intuitive.

For instance, one key idea in the hyperdimensional computing theory of BID17 is that two random, high-dimensional vectors of dimension d whose entries are chosen uniformly from the set {???1, 1} are approximately orthogonal.

The result follows from the central limit theorem because the cosine angle between two such random vectors is normally distributed with ?? = 0 and ?? ??? 1/ DISPLAYFORM0 Building upon this work, we study the way in which binary vectors are distributed relative to continuous vectors.

As binarizing a continuous vector gives the binary vector closest in angle to that continuous vector, we can get a sense of how binary vectors are distributed relative to continuous vectors in high dimensions by binarizing continuous vectors.

The standard normal distribution, which serves as an informative null distribution because it is rotationally invariant, is used to generate random continuous vectors which are then binarized.

This analysis gives a fundamental insight into understanding the recent success of binary neural networks.

Binarizing a random continuous vector changes its direction by a small amount relative to the angle between two random vectors in moderately high dimensions FIG1 .

Binarization changes the direction of a vector by approximately 37??? in high dimensions.

This seems like a large change based on our low-dimensional intuition.

Indeed, the angle between two randomly chosen vectors from a rotationally invariant distribution is uniform in two dimensions.

However, two randomly chosen vectors are approximately orthogonal in high dimensions.

Thus while it is common for two random vectors to have an angle less than 37??? in low dimensions, it is exceedingly rare in high dimensions.

Therefore 37??? is a small angle in high dimensions.

In order to test our theory of the binarization of random vectors chosen from a rotationally invariant distribution, we train a multilayer binary CNN on CIFAR10 (using the Courbariaux et al. FORMULA29 method) and study the weight vectors 1 of that network.

Remarkably, there is a close correspondence between the experimental results and the theory for the angles between the binary and continuous weights FIG1 .

For each layer, the distribution of the angles between the binary and continuous weights is sharply peaked near the d ??? ??? expectation of arccos 2/??.

We note that there is a small but systematic deviation from the theory towards larger angles for the higher layers of the network (Fig. 6 ).

Ternary neural networks are considered in (SI Sec. 5.5) and yield a similar result.

Given the previous discussion, an important question to ask is: are the so-called continuous weights a learning artifact without a clear correspondence to the binary weights?

While we know that w b = ??(w c ), there are many continuous weights that map onto a particular binary weight vector.

Which one is found when using the straight-through estimator to backpropagate through the binarize function?

Remarkably, there is a clear answer to this question.

In numerical experiments, we see that one gets the continuous weight vector such that the dot products of the activations with the prebinarization and post-binarization weights are highly correlated FIG2 .

In equations, a ?? w b ??? a ?? w c .

We call this relation the Dot Product Proportionality (DPP) property.

The proportionality constant, which is subsequently normalized away by a batch norm layer, depends on the magnitudes of the continuous and binary weight vectors and the cosine angle between the binary and continuous weight vectors.

The theoretical consequences of the DPP property are explored in the rest of this section.

We show that the modified gradient of the BNN training algorithm can be viewed as an estimator of the gradient that would be used to train the continuous weights in traditional backpropagation.

This establishes the fundamental point that while the weights and activations are technically binary, they are operating as if the weights are continuous.

For instance, one could imagine using an exhaustive search over all binary weights in the network.

However, the additional structure in the problem associated with taking dot products makes the optimization simpler than that.

Furthermore, we show that if the dot products of the activations with the pre-binarized and post-binarized weights are proportional then straight-through estimator gradient is proportional to the continuous weight network gradient.

The key to the analysis is to focus on the transformers in the network whose forward and backward propagation functions are not related in the way that they would normally be related in typical gradient descent.

Suppose that there is a neural network where two tensors, u, and v and the associated derivatives of the cost with respect to those tensors, ??u, and ??v, are allocated.

Suppose that the loss as a function of v is L(x)| x=v .

Further, suppose that there are two potential forward propagation functions, f , and g. If the network is trained under normal conditions using g as the forward propagation function, then the following computations are done: DISPLAYFORM0 where L (x) denotes the derivative of L with respect to the vector x. In a modified backpropagation scheme, the following computations are done DISPLAYFORM1 A sufficient condition for ??u to be the same in both cases is DISPLAYFORM2 ) where a ??? b means that the vector a is a scalar times the vector b.

Now this general observation is applied to the binarize transformer of FIG0 .

Here, u is the continuous weight, w c , f (u) is the pointwise binarize function, g(u) is the identity function 2 , and L is the loss of the network as a function of the weights in a particular layer.

Given the network architecture, L(x) = M (a ?? x) where a are the activations corresponding to that layer and M is the loss as a function of the weight-activation dot products.

Then L (x) = M (a ?? x) a where denotes a pointwise multiply.

Thus the sufficient condition is DISPLAYFORM3 Since the dot products are followed by a batch normalization, DISPLAYFORM4 , which is the DPP property.

When the DPP only approximately holds, the second derivative can be used to bound the error between the two gradients of the two learning procedures.

In summary, the learning dynamics where g is used for the forward and backward passes (i.e. training the network with continuous weights) is approximately equivalent to the modified learning dynamics (f on the forward pass, and g on the backward pass) when we have the DPP property.

While we demonstrated that the BNN learning dynamics approximate the dynamics that one would have by training a network with continuous weights using a mixture of empirical and theoretical arguments, the ideal result would be that the learning algorithm implies the DPP property.

It should be noted that in the case of stochastic binarization where E(w b ) = w c is chosen by definition, the DPP property is true by design.

However, it is remarkable that the property still holds in the case of deterministic binarization, which is revealing of the fundamental nature of the representations used in neural networks.

While the main focus of this section is the binarization of the weights, the arguments presented can also be applied to the binarize block that corresponds to the non-linearity of the network.

The analogue of the DPP property for this binarize block is: DISPLAYFORM5 where a c denotes the pre-binarized (post-batch norm) activations and a b = a denotes the binarized activations.

This property is empirically verified to hold.

For the sake of completeness, the dot product histogram corresponding to w c ?? a c ??? w b ?? a b is also computed, although it doesn't directly correspond to removing one instance of a binarize transformer.

This property is also empirically verified to hold (SI, FIG4 ).Impact on Classification: It is natural to ask to what extent the classification performance depends on the binarization of the weights.

In experiments on CIFAR10, if the binarization of the weights on all of the convolutional layers is removed, the classification performance drops by only 3 percent relative to the original network.

Looking at each layer individually, removing the weight binarization for the first layer accounts for this entire percentage, and removing the binarization of the weights for each other layer causes no degradation in performance.

This result is evident by looking at the 2D dot product histograms in FIG2 The off-diagonal quadrants show where switching the weights from binary to continuous changes the sign of the binarized weight-activation dot product.

In all of the layers except the first layer, there are very few dot products in the off-diagonal quadrants.

Thus we recommend the use of the dot product histograms for studying the performance of binary neural networks.

Removing the binarization of the activations has a substantial impact on the classification performance because that removes the main non-linearity of the network.

Not surprisingly, some distributions are impacted more strongly by binarization than others.

A binary neural network must adapt its internal representations in such a way to not be degraded too much by binarization at each layer.

In this section we explore the idea that the principal components of the input to the binarization function should be randomly oriented relative to the binarization.

While the network can adapt the higher level representations to satisfy this property, the part of the network that interfaces with the input doesn't have that flexibility.

We make the novel observation that the difficulties in training the first layer of the network are tied to the intrinsic correlations in the input data.

In order to be more precise, we define the Generalized Binarization Transformation (GBT) DISPLAYFORM0 where x is a column vector, R is a fixed rotation matrix, and ?? is the pointwise binarization function from before.

The rows of R are called the axes of binarization.

If R is the identity matrix, then ?? R = ?? and the axes of binarization are the canonical basis vectors (..., 0, 1, 0, ...).

R can either be chosen strategically or randomly.

The GBT changes the distribution being binarized through a rotation.

For appropriate choices of the rotation, R, the directions of the input vectors, x, are changed insignificantly by binarization.

The angle between a vector and its binarized version is dependent on the dot product: x ?? ?? R (x), which is equal to x T ?? R (x) = (Rx) T ??(Rx) = y ?? ??(y) where y = Rx.

As a concrete example of the benefits .

Surprisingly, the dot products are highly correlated (r is the Pearson correlation coefficient).

Thus replacing w b with w c changes the overall constant in front of the dot products, while still preserving whether the dot product is zero or not zero.

This overall constant is divided out by the subsequent batch norm layer.

The shaded quadrants correspond to dot products where the sign changes when replacing the binary weights with the continuous weights.

Notice that for all but the first layer, a very small fraction of the dot products lie in these off diagonal quadrants.

The top left figure (labeled as Layer 1) corresponds to the input and the first convolution.

Note that the correlation is weaker in the first layer.of the GBT, consider the case where x ??? N (0, ??) and ?? i,j = ?? i,j exp(2ki) for k = 0.1 (therefore y ??? N (0, R??R T )).

As the dimension goes to infinity, the angle between a vector drawn from this distribution and its binarized version approaches ??/2.

Thus binarization is destructive to vectors from this distribution.

However, if the GBT is applied with a fixed random matrix 3 , the angle between the vector and its binarized version converges to 37??? FIG3 .

Thus a random rotation can compensate for the errors incurred from directly binarizing a non-isotropic Gaussian.

Moving into how this analysis applies to a binary neural network, the network weights must approximate the important directions in the activations using binary vectors.

For instance, Gabor filters are intrinsic features in natural images and are often found in the first layer weights of neural networks trained on natural images (e.g. BID27 ; BID20 ).

While the network has flexibility in the higher layers, the first layer must interface directly with the input where the features are not necessarily randomly oriented.

For instance, consider the 27 dimensional input to the first set of convolutions in our network: 3 color channels of a 3 by 3 patch of an image from CIFAR10 with the mean removed.

3 PCs capture 90 percent of the variance of this data and 4 PCs capture 94.5 percent of the variance.

The first two PCs are spatially uniform colors.

More generally, large images such as those in IMAGENET have the same issue.

Translation invariance of the image covariance matrix implies that the principal components are the filters of the 2D Fourier transform.

Scale invariance implies a 1/f 2 power spectrum, which results in the largest PCs corresponding to low frequencies BID9 ).Another manifestation of this issue can be seen in our trained networks.

The first layer has a much smaller dot product correlation than the other layers FIG2 .

To study this, we randomly permute the activations in order to generate a distribution with the same marginal statistics as the original data but independent joint statistics (a different permutation for each input image).

Such a transformation gives a distribution with a correlation equal to the normalized dot product of the weight vectors Random vectors are drawn from a Gaussian of dimension d with a diagonal covariance matrix whose entries vary exponentially.

As in FIG1 , the red curve shows the angle between a random vector and its binarized version.

Since the Gaussian is no longer isotropic, the red curve no longer peaks at ?? = arccos 2/??.

However, if the binarization is replaced with a GBT with a fixed random matrix, the direction of the vector is again approximately preserved.

Right: Permuting the activations shows that the correlations observed in FIG2 are not merely due to correlations between the binary and continuous weight vectors.

The correlations are due to these weight vectors corresponding to high variance directions in the data.(SI Sec. 3).

The correlations for the higher layers decrease substantially but the correlation in the first layer increases FIG3 .

For the first layer, the shuffling operation randomly permutes the pixels in the image.

Thus we demonstrate that the binary weight vectors in the first layer are not well-aligned with the continuous weight vectors relative to the input data.

Our theoretically grounded analysis is consistent with previous work.

BID13 find that compressing the first set of convolutional weights of a particular layer by the same fraction has the highest impact on performance if done on the first layer.

BID32 find that accuracy degrades by about 0.5 to 1 percent on SHVN when quantizing the first layer weights.

Thus it is recommended to rotate the input data before normalization or to use continuous weights for the first layer.

Neural networks with binary weights and activations have similar performance to their continuous counterparts with substantially reduced execution time and power usage.

We provide an experimentally verified theory for understanding how one can get away with such a massive reduction in precision based on the geometry of HD vectors.

First, we show that binarization of high-dimensional vectors preserves their direction in the sense that the angle between a random vector and its binarized version is much smaller than the angle between two random vectors (Angle Preservation Property).

Second, we take the perspective of the network and show that binarization approximately preserves weight-activation dot products (Dot Product Proportionality Property).

More generally, when using a network compression technique, we recommend looking at the weight activation dot product histograms as a heuristic to help localize the layers that are most responsible for performance degradation.

Third, we discuss the impacts of the low effective dimensionality of the data on the first layer of the network.

We recommend either using continuous weights for the first layer or a Generalized Binarization Transformation.

Such a transformation may be useful for architectures like LSTMs where the update for the hidden state declares a particular set of axes to be important (e.g. by taking the pointwise multiply of the forget gates with the cell state).

Finally, we show that neural networks with ternary weights and activations can also be understood with our approach.

More broadly speaking, our theory is useful for analyzing a variety of neural network compression techniques that transform the weights, activations or both to reduce the execution cost without degrading performance.

Random n dimensional vectors are drawn from a rotationally invariant distribution.

The angle between two random vectors and the angle between a vector and its binarized version are compared.

A rotationally invariant distribution can be factorized into a pdf for the magnitude of the vector times a distribution on angles.

In the expectations that we are calculating, the magnitude cancels out and there is only one rotationally invariant distribution on angles.

Thus it suffices to compute these expectations using a Gaussian.

Lemmas:1.

Consider a vector, v, chosen from a standard normal distribution of dimension n. DISPLAYFORM0 where ?? is the Gamma function.

Proof: Begin by considering the integral DISPLAYFORM1 where I is an indicator function.

The desired distribution comes from taking the derivative of this cumulative distribution g( DISPLAYFORM2 Thus we can write out the integral DISPLAYFORM3 The integral factorizes and all of the terms are independent of ?? 0 except the integral over DISPLAYFORM4 Using the substitution ?? = cos ?? (which is also consistent with the definition of ?? above), d?? = ??? sin(??)d??, DISPLAYFORM5 Taking the derivative with respect to ?? 0 and using the fundamental theorem of calculus gives g(??) ??? (1 ??? ?? 2 ) (n???3)/2 .

The normalization constant is equal to a beta function that evaluates to the desired result (substitute t = ?? 2 ).

??(z+??) DISPLAYFORM0 ??? Distribution of angles between two random vectors.

Since a Gaussian is a rotationally invariant distribution, we can say without loss of generality that one of the vectors is (1, 0, 0, . . .

0).

Then the cosine angle between those two vectors is ?? as defined above.

While the exact distribution of ?? is given by Lemma 1, we note that -E(??) = 0 due to the symmetry of the distribution.

DISPLAYFORM1 ??? Angles between a vector and the binarized version of that vector, ?? = DISPLAYFORM2 ??((n???1)/2) (substitute u = ?? 2 and use ??(x + 1) = x??(x) ).

Lemma two gives the n ??? ??? limit.

DISPLAYFORM3 Thus we have the normal scaling as in the central limit theorem of the large n variance.

We can calculate this explicitly following the approach of 5 .

As E(??) has been calculated, it suffices to calculate E(?? 2 ).

Expanding out ?? 2 , E(?? DISPLAYFORM4 ).

Below we show that E( DISPLAYFORM5 ??n .

Thus the variance is: DISPLAYFORM6 Using Lemma 2 to expand out the last term, we get [ DISPLAYFORM7 Plugging this in gives the desired result.

Going back to the calculation of that expectation, change variables to v 1 = r cos ??, DISPLAYFORM8 The integration over the volume element dv 3 . . .

dv n is rewritten as dzdA n???3 where dA n denotes the surface element of a n sphere.

Since the integrand only depends on the magnitude, z, dA n???3 = z n???3 * S n???3 where DISPLAYFORM9 denotes the surface area of a unit n-sphere.

Then DISPLAYFORM10 Then substitute r = p cos ??, z = p sin ?? where DISPLAYFORM11 The first integral is 2 n(n???2) using u = sin 2 ??.

The second integral is 2 (n???2)/2 ??(n/2) using u = p 2 /2 and the definition of the gamma function.

Simplifying, the result is 2 ?? * n .Thus the angle between a vector and a binarized version of that vector converges to arccos DISPLAYFORM12 which is a very small angle in high dimensions.

In this subsection, we look at the learning dynamics for the BNN training algorithm in a simple case and gain some insight about the learning algorithm.

Consider the case of regression where the target output, y, is predicted with a binary linear predictor with x as the input.

Using a squared error loss, DISPLAYFORM0 (In this notation, x is a column vector.) Taking the derivative of this loss with respect to the continuous weights and using the rule for back propagating through the binarize function gives DISPLAYFORM1 Finally, averaging over the training data gives DISPLAYFORM2 It is worthwhile to compare this equation the corresponding equation from typical linear regression: ???w c ??? C yx ??? w c ?? C xx .

For simplicity, consider the case where C xx is the identity matrix.

In this case, all of the components of w become independent: ??w = * (?? ??? ??(w)) where is the learning rate and ?? is the entry of C yx corresponding to a particular element, w. Compared to regular linear regression, it is clear that the stable point of these equations is when w = ??.

Since the weight is binarized, that equation cannot be satisfied.

However, it can be shown ( ) that in this special case of binary weight linear regression, E(??(w c )) = ??.

Intuitively, if we consider a high dimensional vector and the fluctuations of each component are likely to be out of phase, then w b ?? x ??? w c ?? x is going to be correct in expectation with a variance that scales as 1 n .

During the actual learning process, we anneal the learning rate to a very small number, so the particular state of a fluctuating component of the vector is frozen in.

Relatedly, the equation C yx ??? wC xx is easier to satisfy in high dimensions, whereas in low dimensions, it is only satisfied in expectation.

Proof for ( ): Suppose that |??| ??? 1.

The basic idea of these dynamics is that steps of size proportional to are taken whose direction depends on whether w > 0 or w < 0.

In particular, if w > 0, then the step is ??? ?? |1 ??? ??| and if w < 0, the step is ?? (?? + 1).

It is evident that after a sufficient burn-in period, |w| ??? * max(|1 ??? ??|, 1 + ??) ??? 2 .

Suppose w > 0 occurs with fraction p and w < 0 occurs with fraction 1 ??? p.

In order for w to be in equilibrium, oscillating about zero, these steps balance out on average: p(1 ??? ??) = (1 ??? p)(1 + ??) ??? p = (1 + ??)/2.

Then the expected value of ??(w) is 1 * p + (???1) * (1 ??? p) = ??.

When |??| > 1, the dynamics diverge because ?? ??? ??(w) will always have the same sign.

This divergence demonstrates the importance of some normalization technique such as batch normalization or attempting to represent w with a constant times a binary matrix.

Suppose that A = w ?? a and B = v ?? a where w, v are weight vectors and a is the vector of activations.

What is the correlation, r, between A and B?

Assuming that E(a) = 0, E(A) = E(B) = 0.

Then DISPLAYFORM0 In the case where the activations are randomly permuted, C is proportional to the identity matrix, and thus the correlation between A and B is the cosine angle between u and v. Figure 6 : Angle distribution between continuous and binary weight vectors by layer for a binary CNN trained on CIFAR-10 (same plot as in FIG1 except zoomed in).

Notice that there is a small but systematic deviation towards larger angles relative to the theoretical expectation (vertical dotted line).

As the dimension of the vectors in the layer goes up, the distribution gets sharper.

The theory predicts that the standard deviation of these distributions scales as 1/ ??? d. This relationship is shown to approximately hold in FIG1 .

Moving beyond binarization, recent work has shown how to train neural networks where the activations are quantized to three (or more) values (e.g. ).

Indeed, ternarization may be a more natural quantization method than binarization for neural network weights because one can express a positive association (+1), a negative association (???1), or no association (0) between two features in a neural network.

We show that the analysis used on BNNs holds for ternary neural networks.

The quantization function: ter a (x) = 1 if x > a, 0 if |x| < a, and ???1 if x < ???a is used in place of the binarize function with the same straight-through estimator for the gradient.

Call a the ternarization threshold.

The exact value of a is only important at initialization because the scaling constant of the batch normalization layer allows the network to adapt the standard deviation of the pre-nonlinearity activations to the value of a. The network architecture from the previous experiments was used to classify images in CIFAR-10 and a was chosen to be equal to 0.02.

In practice, roughly 10 percent of the ternarized weights were zero FIG5 ) and 2 percent of the activations were zero.

Thus the network learning process did not ignore the possibility of using zero weights.

However, more work is needed to effectively use the zero value for the activations.

The empirical distribution of angles between the continuous vectors and their ternarized counterparts is highly peaked at the value predicted by the theory FIG5 .

Random vectors are chosen from a standard normal distribution of dimension d. As in the case of binarization, the ternarized version of a vector is close in angle to the original vector in high dimensions FIG5 .

These vectors are quantized using ter a for different values of a. The peak angle varies substantially as a function of a FIG5 .

Note that for a = 0, the ternarization function collapses into the binarization function.

The empirical value of a is the ratio of the empirical threshold to the empirical standard deviation of the continuous weights.

Thus a ??? 0.02/0.18 ??? 0.11 for the higher layers.

Remarkably, the theoretical prediction for the peak angle as a function of a matches closely with the empirical result FIG5 .Finally, the dot product proportionality property is also shown to hold for ternary neural networks FIG5 .

Thus the continuous weights found using the TNN training algorithm approximate the continuous weights that one would get if the network were trained with continuous weights and regular backpropagation.

??? ) indicates the theoretical prediction given the empirical ternarization threshold (??? 0.11).

d is the dimension of the filters at each layer.

(c) Distribution of angles between two random vectors (blue), and between a vector and its quantized version (red), for a rotationally invariant distribution of dimension d. The ternarization threshold is chosen to match the trained network.

As the red and blue curves have little overlap, ternarization causes a small change in angle in high dimensions.

(d) Angle between a random vector and its ternarized version as a function of ternarization threshold for d = 1000.

There is a large variation in the angle over different thresholds.

(e) Ternarization preserves dot products.

The 2D histogram shows the dot products between the ternarized weights and the activations (horizontal axis) and the dot products between the continuous weights and the activations (vertical axis) for layer 3 of the network.

The dot products are highly correlated.

@highlight

Recent successes of Binary Neural Networks can be understood based on the geometry of high-dimensional binary vectors

@highlight

Investigates numerically and theoretically the reasons behind the empirical success of binarized neural networks.

@highlight

This paper analyzes the effectiveness of binary neural networks and why binarization is able to preserve model performance.