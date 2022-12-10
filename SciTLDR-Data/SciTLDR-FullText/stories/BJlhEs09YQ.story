In this paper we study the problem of learning the weights of a deep convolutional neural network.

We consider a network where convolutions are carried out over non-overlapping patches with a single kernel in each layer.

We develop an algorithm for simultaneously learning all the kernels from the training data.

Our approach dubbed Deep Tensor Decomposition (DeepTD) is based on a rank-1 tensor decomposition.

We theoretically investigate DeepTD under a realizable model for the training data where the inputs are chosen i.i.d.

from a Gaussian distribution and the labels are generated according to planted convolutional kernels.

We show that DeepTD is data-efficient and provably works as soon as the sample size exceeds the total number of convolutional weights in the network.

Our numerical experiments demonstrate the effectiveness of DeepTD and verify our theoretical findings.

Deep neural network (DNN) architectures have led to state of the art performance in many domains including image recognition, natural language processing, recommendation systems, and video analysis (He et al. (2016) ; Krizhevsky et al. (2012) ; Van den Oord et al. (2013) ; Collobert & Weston (2008) ).

Convolutional neural networks (CNNs) are a class of deep, feed-forward neural networks with a specialized DNN architecture.

CNNs are responsible for some of the most significant performance gains of DNN architectures.

In particular, CNN architectures have led to striking performance improvements for image/object recognition tasks.

Convolutional neural networks, loosely inspired by the visual cortex of animals, construct increasingly higher level features (such as mouth and nose) from lower level features such as pixels.

An added advantage of CNNs which makes them extremely attractive for large-scale applications is their remarkable efficiency which can be attributed to: (1) intelligent utilization of parameters via weight-sharing, (2) their convolutional nature which exploits the local spatial structure of images/videos effectively, and (3) highly efficient matrix/vector multiplication involved in CNNs compared to fully-connected neural network architectures.

Despite the wide empirical success of CNNs the reasons for the effectiveness of neural networks and CNNs in particular is still a mystery.

Recently there has been a surge of interest in developing more rigorous foundations for neural networks (Soltanolkotabi et al. (2017) FORMULA108 ).

Most of this existing literature however focus on learning shallow neural networks typically consisting of zero or one hidden layer.

In practical applications, depth seems to play a crucial role in constructing progressively higher-level features from pixels.

Indeed, state of the art Resnet models typically have hundreds of layers.

Furthermore, recent results suggest that increasing depth may substantially boost the expressive power of neural networks (Raghu et al. (2016) ; Cohen et al. (2016) )

.;In this paper, we propose an algorithm for approximately learning an arbitrarily deep CNN model with rigorous guarantees.

Our goal is to provide theoretical insights towards better understanding when training deep CNN architectures is computationally tractable and how much data is required for successful training.

We focus on a realizable model where the inputs are chosen i.i.d.

from a Gaussian distribution and the labels are generated according to planted convolutional kernels.

We use both labels and features in the training data to construct a tensor.

Our first insight is that, in the limit of infinite data this tensor converges to a population tensor which is approximately rank one and whose factors reveal the direction of the kernels.

Our second insight is that even with finite data this empirical tensor is still approximately rank one.

We show that the gap between the population and empirical tensors provably decreases with the increase in the size of the training data set and becomes negligible as soon as the size of the training data becomes proportional to the total numbers of the parameters in the planted CNN model.

Combining these insights we provide a tensor decomposition algorithm to learn the kernels from training data.

We show that our algorithm approximately learns the kernels (up to sign/scale ambiguities) as soon as the size of the training data is proportional to the total number of parameters of the planted CNN model.

Our results can be viewed as a first step towards provable end-to-end learning of practical deep CNN models.

Extending the connections between neural networks and tensors (Janzamin et al. (2015) ; Cohen et al. (2016) ; Zhong et al. (2017a) ), we show how tensor decomposition can be utilized to approximately learn deep networks despite the presence of nonlinearities and growing depth.

While our focus in this work is limited to tensors, we believe that our proposed algorithm may provide valuable insights for initializing local search methods (such as stochastic gradient descent) to enhance the quality and/or speed of CNN training.

In this section we discuss the CNN model which is the focus of this paper.

A fully connected artificial neural network is composed of computational units called neurons.

The neurons are decomposed into layers consisting of one input layer, one output layer and a few hidden layers with the output of each layer is fed in (as input) to the next layer.

In a CNN model the output of each layer is related to the input of the next layer by a convolution operation.

In this paper we focus on a CNN model where the stride length is equal to the length of the kernel.

This is sometimes referred to as a non-overlapping convolution operation formally defined below.

Definition 2.1 (Non-overlapping convolution) For two vectors k ∈ R d and h ∈ R p=dp their nonoverlapping convolution, denoted by k * ◻ h yields a vector u ∈ Rp In this paper it is often convenient to view convolutions as matrix/vector multiplications.

This leads us to the definition of the kernel matrix below.

Definition 2.2 (Kernel matrix) Consider a kernel k ∈ R d and any vector h ∈ R p=pd .

Corresponding to the non-overlapping convolution k * ◻ h, we associate a kernel matrix K * ◻ ∈ Rp ×p defined as K * ◻ ∶= Ip ⊗ k T .

Here, A ⊗ B denotes the Kronecker product between the two matrices A and B and Ip denotes thep ×p identity matrix.

We note that based on this definition k * ◻ h = K * ◻ h. Throughout the paper we shall use K interchangeably with K * ◻ to denote this kernel matrix with the dependence on the underlying kernel and its non-overlapping form implied.

With the definition of the non-overlapping convolution and the corresponding kernel matrix in hand we are now ready to define the CNN model which is the focus of this paper.

For ease of exposition, the CNN input-output relationship along with the corresponding notation is depicted in FIG1 .•

Depth and numbering of the layers.

We consider a network of depth D where we number the input as layer 0 and the output as layer D and the hidden layers 1 to D − 1.• Layer dimensions and representations.

We assume the input of the CNN, denoted by x ∈ R p , consists of p features and the output is a one dimensional label.

We also assume the hidden layers (numbered by = 1, 2, . . .

, D − 1) consists of p units withh ( ) ∈ R p and h ( ) ∈ R p denoting the input and output values of the units in the th hidden layer.

For consistency of our notation we shall also define h (0) ∶= x ∈ R p and note that the output of the CNN is h (D) ∈ R. Also, p 0 = p and p D = 1.• Kernel dimensions and representation.

For = 1, . . .

D we assume the kernel relating the output of layer ( − 1) to the input of layer is of dimension d and is denoted by k DISPLAYFORM0 • Inter-layer relationship.

We assume the inputs of layer (denoted byh ( ) ∈ R p ) are related to the outputs of layer ( − 1) (denoted by h ( −1) ∈ R p −1 ) via a non-overlapping convolution DISPLAYFORM1 In the latter equality we have used the representation of non-overlapping convolution as a matrix/vector product involving the kernel matrix K ( ) ∈ R p ×p −1 associated with the kernel k ( ) ∈ R d per Definition 2.2.

We note that the non-overlapping nature of the convolution implies that p = p −1 d .•

Activation functions and intra-layer relationship.

We assume the input of each hidden unit is related to its output by applying an activation function φ ∶ R → R. More precisely, h ( ) ∶= φ (h ( ) ) where for a vector u ∈ R p , φ (u) ∈ R p is a vector obtained by applying the activation function φ to each of the entries of u. We allow for using distinct activation functions {φ } D =1 at every layer.

Throughout, we also assume all activations are 1-Lipschitz functions (i.e. φ (a) − φ (b) ≤ a − b ).•

Final output.

The input-output relation of our CNN model with an input x ∈ R p is given by DISPLAYFORM2

This paper introduces an approach to approximating the convolutional kernels from training data based on tensor decompositions dubbed DeepTD, which consists of a carefully designed tensor decomposition.

To connect these two problems, we begin by stating how we intend to construct the tensor from the training data.

To this aim given any input data x ∈ R p , we form a D-way tensor DISPLAYFORM0 DISPLAYFORM1 Given a set of training data consisting of n input/output pairs (x i , y i ) ∈ R p × R we construct a tensor T n by tensorizing the input vectors as discussed above and calculating a weighted combination of these tensorized inputs.

More precisely, DISPLAYFORM2 We then perform a rank-1 tensor decomposition on this tensor to approximate the convolutional kernels.

Specifically we solvê DISPLAYFORM3 In the above ⊗ D =1 v denotes the tensor resulting from the outer product of v 1 , v 2 , . . .

, v D .

This tensor rank decomposition is also known as CANDECOMP/PARAFAC (CP) decomposition Bro (1997) and can be solved efficiently using Alternating Least Squares and a variety of other algorithms Anandkumar et al. (2014a); Ge et al. (2015) ; Anandkumar et al. (2014b).

1 At this point it is completely unclear why the tensor T n or its rank-1 decomposition can yield anything useful.

The main intuition is that as the data set grows (n → ∞) the empirical tensor T n converges close to a population tensor T whose rank-1 decomposition reveals useful information about the kernels.

Specifically, we will show that DISPLAYFORM4 with α a scalar whose value shall be discussed later on.

Here, x is a Gaussian random vector with i.i.d.

N (0, 1) entries and represents a typical input with f CNN (x) the corresponding output and T (x) the tensorized input.

We will also utilize a concentration argument to show that when the training 1 We would like to note that while finding the best rank-1 TD (3.2) is NP-hard, our theoretical guarantees continue to hold when using an approximately optimal solution to (3.2).

In fact, we can show that unfolding the tensor along the ith kernel into a di × ∏ j≠i dj matrix and using the top left singular vector yields a good approximation to k (i) .

However, in our numerical simulations we instead utilize popular software packages to solve (3.2).

data set originates from an i.i.d.

distribution, for a sufficiently large training data n, T n yields a good approximation of the population tensor T .Another perhaps perplexing aspect of the construction of T n in (3.1) is the subtraction by y avg in the weights.

The reason this may be a source of confusion is that based on the intuition above DISPLAYFORM5 so that the subtraction by the average seems completely redundant.

The main purpose of this subtraction is to ensure the weights y i − y avg are centered (have mean zero).

This centering allows for a much better concentration of the empirical tensor around its population counter part and is crucial to the success of our approach.

We would like to point out that such a centering procedure is reminiscent of batch-normalization heuristics deployed when training deep neural networks.

Finally, we note that based on (3.3), the rank-1 tensor decomposition step can recover the convolutional kernels {k DISPLAYFORM6 up to sign and scaling ambiguities.

Unfortunately, depending on the activation function, it may be impossible to overcome these ambiguities.

For instance, if the activations are homogeneous (i.e. φ (ax) = aφ (x)), then scaling up one layer and scaling down the other layer by the same amount does not change the overall function f CNN (⋅).

Similarly, if the activations are odd functions, negating two of the layers at the same time preserves the overall function.

In Appendix C, we discuss some heuristics and theoretical guarantees for overcoming these sign/scale ambiguities.

In this section we introduce our theoretical results for DeepTD.

We will discuss these results in three sections.

In Section 4.1 we show that the empirical tensor concentrates around its population counterpart.

Then in Section 4.2 we show that the population tensor is well-approximated by a rank-1 tensor whose factors reveal the convolutional kernels.

Finally, in Section 4.3 we combine these results to show DeepTD can approximately learn the convolutional kernels up to sign/scale ambiguities.

Our first result shows that the empirical tensor concentrations around the population tensor.

We measure the quality of this concentration via the tensor spectral norm defined below.

Definition 4.1 Let RO be the set of rank-one tensors ⊗ DISPLAYFORM0 Let x ∈ R p be a Gaussian random vector distributed as N (0, I p ) with the corresponding labels y = f CN N (x) generated by the CNN model and X ∶= T (x) the corresponding tensorized input.

Suppose the data set consists of n training samples where the feature vectors x i ∈ R p are distributed i.i.d.

N (0, I p ) with the corresponding labels y i = f CN N (x i ) generated by the same CNN model and DISPLAYFORM1 Then the empirical tensor T n and population tensor T defined based on this dataset obey DISPLAYFORM2 1) with probability at least 1 − 5e DISPLAYFORM3 n,n , where c > 0 is an absolute constant.

The theorem above shows that the empirical tensor approximates the population tensor with high probability.

This theorem also shows that the quality of this approximation is proportional to ∏ DISPLAYFORM4 .

This is natural as ∏ DISPLAYFORM5 is an upper-bound on the Lipschitz constant of the network and shows how much the CNN output fluctuates with changes in the input.

The more fluctuations, the less concentrated the empirical tensor is, translating into a worse approximation guarantee.

Furthermore, the quality of this approximation grows with the square root of the parameters in the model (∑ D =1 d ) and is inversely proportional to the square root of the number of samples (n) which are typical scalings in statistical learning.

We would also like to note that as we will see in the forthcoming sections, in many cases T is roughly on the order of ∏ DISPLAYFORM6 .

Therefore, the relative error in the DISPLAYFORM7

Our second result shows that the population tensor can be approximated by a rank one tensor.

To explain the structure of this rank one tensor and quantify the quality of this approximation we require a few definitions.

The first quantity roughly captures the average amount by which the nonlinear activations amplify or attenuate the size of an input feature at the output.

This quantity is the product of the average slopes of the activations evaluated along a path connecting the first input feature to the first hidden units across the layers all the way to the output.

We note that this quantity is the same when calculated along any path connecting an input feature to the output passing through the hidden units.

Therefore, this quantity can be thought of as the average gain (amplification or attenuation) of a given input feature due to the nonlinear activations in the network.

To gain some intuition consider a ReLU network which is mostly inactive.

Then the network is dead and α CNN ≈ 0.

On the other extreme if all ReLU units are active the network operates in the linear regime and α CNN = 1.

We would like to point out that α CNN can in many cases be bounded from below by a constant.

For instance, as proven in Appendix B, for ReLU activations as long as the kernels obey DISPLAYFORM0 Another example is the softplus activation φ (x) = log (1 + e x ) for which we prove α CNN ≥ 0.3 under similar assumptions (Also in Appendix B).

We note that an assumption similar to (4.2) is needed for the network to be active.

This is because if the kernel sums are negative one can show that with high probability, all the ReLUs after the first layer will be inactive and the network will be dead.

With this definition in hand, we are now ready to describe the form of the rank one tensor that approximates the population tensor.

Definition 4.4 (Rank one CNN tensor) We define the rank one CNN tensor DISPLAYFORM1 That is, the product of the kernels {k DISPLAYFORM2 scaled by the CNN gain α CNN .

To quantify how well the rank one CNN tensor approximates the population tensor we need two definitions.

The first definition concerns the activation functions.

Definition 4.5 (Activation smoothness) We assume the activations are differentiable everywhere and S-smooth (i.e. φ ′ (x) − φ ′ (y) ≤ S x − y for all x, y ∈ R) for some S ≥ 0.The reason smoothness of the activations play a role in the quality of the rank one approximation is that smoother activations translate into smoother variations in the entries of the population tensor.

Therefore, the population tensor can be better approximated by a low-rank tensor.

The second definition captures how diffused the kernels are.

Definition 4.6 (Kernel diffuseness parameter) Given kernels {k DISPLAYFORM3 The less diffused (or more spiky) the kernels are, the more the population tensor fluctuates and thus the quality of the approximation to a rank one tensor decreases.

With these definitions in place, we are now ready to state our theorem on approximating a population tensor with a rank one tensor.

Theorem 4.7 Consider the setup of Theorem 4.2.

Also, assume the activations are S-smooth per Definition 4.5 and the convolutional kernels are µ-diffused per Definition 4.6.

Then, the population tensor T ∶= E[yX] can be approximated by the rank-1 tensor DISPLAYFORM4 The theorem above states that the quality of the rank one approximation deteriorates with increase in the smoothness of the activations and the diffuseness of the convolutional kernels.

As mentioned earlier increase in these parameters leads to more fluctuations in the population tensor making it less likely that it can be well approximated by a rank one tensor.

We also note that DISPLAYFORM5 and therefore the relative error in this approximation is bounded by DISPLAYFORM6 We would like to note that for many activations the smoothness is bounded by a constant.

For instance, for the softplus activation (φ(x) = log(1 + e x )) and one can show that S ≤ 1.

As stated earlier, under appropriate assumptions on the kernels and activations, the CNN gain α CNN is also bounded from below by a constant.

Assume the convolutional kernels have unit norm and are sufficiently diffused so that the diffuseness parameter is bounded by a constant.

We can then conclude that DISPLAYFORM7 .

This implies that as soon as the length of the convolutional patches scale with the square of depth of the network by a constant factor the rank one approximation is sufficiently good.

Our back-of-the-envelope calculations suggest that the correct scaling is linear in D versus the quadratic result we have established here.

Improving our result to achieve the correct scaling is an interesting future research direction.

Finally, we would like to note that while we have assumed differentiable and smooth activations we expect our results to apply to popular non-differentiable activations such as ReLU activations.

We demonstrated in the previous two sections that the empirical tensor concentrates around its population counter part and that the population tensor is well-approximated by a rank one tensor.

We combine these two results along with a perturbation argument to provide guarantees for DeepTD.

DeepTD estimates of the convolutional kernels given by (3.2) using the empirical tensor T n obeys DISPLAYFORM0 , with probability at least 1 − 5e DISPLAYFORM1 n,n , where c > 0 is an absolute constant.

The above theorem is our main result on learning a non-overlapping CNN with a single kernel at each layer.

It demonstrates that estimatesk ( ) obtained by DeepTD have significant inner product with the ground truth kernels k ( ) with high probability, using only few samples.

Indeed, similar to the discussion after Theorem 4.7 assuming the activations are sufficiently smooth and the convolutional kernels are unit norm and sufficiently diffused, the theorem above can be simplified as follows DISPLAYFORM2 Thus the kernel estimates obtained via DeepTD are well aligned with the true kernels as soon as the number of samples scales with the total number of parameters in the model and the length of the convolutional kernels (i.e. the size of the batches) scales quadratically with the depth of the network.

Our goal in this section is to numerically corroborate the theoretical predictions of Section 4.

To this aim we use a CNN model of the form (2.1) with D layers and ReLU activations and set the kernel lengths to be all equal to each other i.e. d 4 = . . .

= d 1 = d. We use the identity activation for the last layer (i.e. φ D (z) = z) with the exception of the last experiment where we use a ReLU activation (i.e. φ D (z) = max(0, z)).

We conducted our experiments in Python using the Tensorly library for the tensor decomposition in DeepTD (Kossaifi et al. (2016) ).

Each curve in every figure is obtained by averaging 100 independent realizations of the same CNN learning procedure.

Similar to our theory, we use Gaussian data points x and ground truth labels y = f CNN (x).We conduct two sets of experiments: The first set focuses on larger values of depth D and the second set focuses on larger values of width d. In all experiments kernels are generated with random Gaussian entries and are normalized to have unit Euclidean norm.

For the ReLU activation if one of the kernels have all negative entries, the output is trivially zero and learning is not feasible.

To DISPLAYFORM0 ⟩ ) between the DeepTD estimate and the ground truth kernels for different layers and over-sampling ratios N .

address this, we consider operational networks where at least 50% of the training labels are nonzero.

Here, the number 50% is arbitrarily chosen and we verified that similar results hold for other values.

Finally, to study the effect of finite samples, we let the sample size grow proportional to the total degrees of freedom ∑ DISPLAYFORM1 and carry out the experiments for N ∈ {10, 20, 50, 100}. While our theory requires N ≳ log D, in our experiments, we typically observe that improvement is marginal after N = 50.In FIG6 , we consider two networks with d = 2, D = 12 and d = 3, D = 8 configurations.

We plot the absolute correlation between the ground truth and the estimates as a function of layer depth.

For each hidden layer 1 ≤ ≤ D, our correlation measure (y-axis) is corr(k DISPLAYFORM2 This number is between 0 and 1 as the kernels and their estimates both have unit norm.

We observe that for both d = 2 and d = 3, DeepTD consistently achieves correlation values above 75% for N = 20.

While our theory requires d to scale quadratically with depth i.e. d ≳ D 2 , we find that even small d values work well in our experiments.

The effect of sample size becomes evident by comparing N = 20 and N = 50 for the input and output layers ( = 1, = D).

In this case N = 50 achieves perfect correlation.

Interestingly, correlation values are smallest in the middle layers.

In fact this even holds when N is large suggesting that the rank one approximation of the population tensor provides worst estimates for the middle layers.

In FIG7 , we use a ReLU activation in the final layer and assess the impact of the centering procedure of the DeepTD algorithm which is a major theme throughout the paper.

We define the NaiveTD algorithm which solves (3.2) without centering in the empirical tensor i.e. DISPLAYFORM3 Since the activation of the final layer is ReLU, the output has a clear positive bias in expectation which will help demonstrating the importance of centering.

We find that for smaller oversampling factors of N = 10 or N = 20, DeepTD has a visibly better performance compared with NaiveTD.

The correlation difference is persistent among different layers (we plotted only Layers 1 and 2) and appears to grow with increase in the kernel size d.

Finally, in FIG8 , we assess the impact of activation nonlinearity by comparing the ReLU and identity activations in the final layer.

We plot the first and final layer correlations for this setup.

While the correlation performances of the first layer are essentially identical, the ReLU activation (dashed lines) achieves significantly lower correlation at the final layer.

This is not surprising as the final layer passes through an additional nonlinearity.

Our work is closely related to the recent line of papers on neural networks as well as tensor decompositions.

We briefly discuss this related literature.

Neural networks: Learning neural networks is a nontrivial task involving non-linearities and nonconvexities.

Consequently, existing theory works consider different algorithms, network structures and assumptions.

A series of recent work focus on learning zero or one-hidden layer fully connected neural networks with random inputs and planted models (Goel et al. FORMULA108

In this paper we studied a multilayer CNN model with depth D. We assumed a non-overlapping structure where each layer has a single convolutional kernel and has stride length equal to the dimension of its kernel.

We establish a connection between approximating the CNN kernels and higher order tensor decompositions.

Based on this, we proposed an algorithm for simultaneously learning all kernels called the Deep Tensor Decomposition (DeepTD).

This algorithm builds a D-way tensor based on the training data and applies a rank one tensor factorization algorithm to this tensor to simultaneously estimate all of the convolutional kernels.

Assuming the input data is distributed i.i.d.

according to a Gaussian model with corresponding output generated by a planted set of convolutional kernels, we prove DeepTD can approximately learn all kernels with a near minimal amount of training data.

A variety of numerical experiments complement our theoretical findings.

In this section we will prove our main results.

Throughout, for a random variable X, we use zm(X) to denote X − E[X].

Simply stated, zm(X) is the centered version of X. For a random vector/matrix/tensor X, zm(X) denotes the vector/matrix/tensor obtained by applying the zm() operation to each entry.

For a tensor T we use T F to denote the square root of the sum of squares of the entries of the tensor.

Stated differently, this is the Euclidean norm of a vector obtained by rearranging the entries of the tensor.

Throughout we use c, c1, c2, and C to denote fixed numerical constants whose values may change from line to line.

We begin with some useful definitions and lemmas.

In this section we gather some useful definitions and well-known lemmas that will be used frequently throughout our concentration arguments.

Definition A.1 (Orlicz norms) For a scalar random variable Orlicz-a norm is defined as DISPLAYFORM0 Orlicz-a norm of a vector x ∈ R p is defined as x ψa = sup v∈B p v T x ψa where B p is the unit 2 ball.

The sub-exponential norm is the function ⋅ ψ 1 and the sub-gaussian norm the function ⋅ ψ 2 .We now state a few well-known results that we will use throughout the proofs.

This results are standard and are stated for the sake of completeness.

The first lemma states that the product of sub-gaussian random variables are sub-exponential.

DISPLAYFORM1 The next lemma connects Orlicz norms of sum of random variables to the sum of the Orlicz norm of each random variable.

Lemma A.3 Suppose X, Y are random variables with bounded ⋅ ψa norm.

Then X + Y ψa ≤ 2 max{ X ψa , Y ψa }.

In particular X − E X] ψa ≤ 2 X ψa .The lemma below can be easily obtained by combining the previous two lemmas.

DISPLAYFORM2 Finally, we need a few standard chaining definitions.

For the following discussion ∆ d (An(t)), will be the diameter of the set S ∈ An that contains t, with respect to the d metric.

DISPLAYFORM3 where the infimum is taken over all admissible sequences.

The following lemma upper bounds γα functional with covering numbers of T .

The reader is referred to Section 1.2 of Talagrand FORMULA108 , Equation (2.3) of Dirksen FORMULA69 , and Lemma D.17 of Oymak (2018).Lemma A.7 (Dudley's entropy integral) Let N (ε) be the ε covering number of the set T with respect to the d metric.

Then DISPLAYFORM4 where Cα > 0 depends only on α > 0.

To prove this theorem, first note that given labels {yi} n i=1 ∼ y and their empirical average yavg = n DISPLAYFORM0 .

Hence y − yavg = zm(y − yavg) and we can rewrite the empirical tensor as follows DISPLAYFORM1 Recall that the population tensor is equal to DISPLAYFORM2 Thus the population tensor can alternatively be written as DISPLAYFORM3 .

Combining the latter with (A.1) we conclude that DISPLAYFORM4 Xi .

Now using the triangular inequality for tensor spectral norm we conclude that DISPLAYFORM5 Xi .We now state two lemmas to bound each of these terms.

The proofs of these lemmas are defered to Sections A.2.1 and A.2.2.

DISPLAYFORM6

holds with c1 > 0 a fixed numerical constant.

Lemma A.9 Consider the setup of Lemma A.8.

Then DISPLAYFORM0 holds with c2 > 0 a fixed numerical constant.

, and c1 = c 2 together with Lemma A.9 with t1 = √ n, t2 = t, and c2 = c 2 concludes the proof of Theorem 4.2.

All that remains is to prove Lemmas A.8 and A.9 which are the subject of the next two sections.

It is more convenient to carryout the steps of the proof on ∑ n i=1 zm(zm(f (xi))Xi) in liue of 1 n ∑ n i=1 zm(zm(f (xi))Xi).

The lemma trivally follows by a scaling by a factor 1 n. We first write the tensor spectral norm as a supremum DISPLAYFORM0 Let Yi = zm(f (xi))Xi.

Define the random process g(T ) = ∑ n i=1 ⟨zm(Yi), T ⟩. We claim that g(T ) has a mixture of subgaussian and subexponential increments (see Definition A.1 for subgaussian and subexponential random variables).

Pick two tensors T , H ∈ R ⊗ D =1 d .

Increments of g satisfy the linear relation DISPLAYFORM1 By construction E[g(T ) − g(H)] = 0.

We next claim that Yi is a sub-exponential vector.

Consider a tensor T with unit length T F = 1 i.e. the sum of squares of entries are equal to one.

We have ⟨Yi, T ⟩ = zm(f (xi)) ⟨Xi, T ⟩. f (Xi) is a Lipschitz function of a Gaussian random vector.

Thus, by the concentration of Lipschitz functions of Gaussians we have DISPLAYFORM2 This immediately implies that zm(f (Xi)) ψ 2 ≤ cL for a fixed numerical constant c. Also note that ⟨Xi, T ⟩ ∼ N (0, 1) hence ⟨Xi, T ⟩ ψ 2 ≤ c. These two identities combined with Lemma A.4 implies a bound on the sub-exponential norm zm(⟨Yi, T ⟩) ψ 1 ≤ cL. Next, we observe that g(T ) − g(H) is sum of n i.i.d.

sub-exponentials each obeying ⟨zm(Yi), T − H⟩ ψ 1 ≤ cL T − H F .

Applying a standard sub-exponential Bernstein inequality, we conclude that DISPLAYFORM3 holds with γ a fixed numerical constant.

This tail bound implies that g is a mixed tail process that is studied by Talagrand and others (Talagrand (2014); Dirksen FORMULA69 ).

In particular, supremum of such processes are characterized in terms of a linear combination of Talagrand's γ1 and γ2 functionals (see Definition A.6 as well as Talagrand (2014; 2006) for an exposition).

We pick the following distance metrics on tensors induced by the Frobenius norm: d1(T , H) = L H − T F c and d2(T , H) = H − T F L n c. We can thus rewrite (A.4) in the form DISPLAYFORM4

RO with respect to ⋅ F norm is 1 hence radius with respect to d1, d2 metrics are L c, L n c respectively.

Applying Theorem 3.5 of Dirksen FORMULA69 , we obtain DISPLAYFORM0 Observe that we can use the change of variable t = L ⋅ max √ un, u to obtain DISPLAYFORM1 with some updated constant C > 0.

To conclude, we need to bound the γ2 and γ1 terms.

To achieve this we will upper bound the γα functional in terms of Dudley's entropy integral which is stated in Lemma A.7.

First, let us find the ε covering number of RO.

Pick 0 < δ ≤ 1 coverings C of the unit 2 balls B d .

These covers have size at most (1 + 2 δ) d .

Consider the set of rank 1 tensors C = C1 ⊗ . . .

⊗ C D with size (1 + 2 δ) DISPLAYFORM2 Denoting Frobenius norm covering number of RO by N (ε), this implies that, for 0 < ε ≤ 1, DISPLAYFORM3 (A.8) Thus the metrics d1, d2 metrics are ⋅ F norm scaled by a constant.

Hence, their γα functions are scaled versions of (A.8) given by DISPLAYFORM4 Substituting these in (A.5) and using n ≥ ∑ DISPLAYFORM5 Substituting t → L √ nt and recalling (A.2), concludes the proof.

We first rewrite, DISPLAYFORM0 As discussed in (A.3), zm(f (xi)) ψ 2 ≤ cL for c a fixed numerical constant.

Since xi's are i.i.d the empirical sum favg = 1 n ∑ n i=1 zm(f (xi)) obeys the bound favg ψ 2 ≤ cL √ n as well.

Hence, DISPLAYFORM1 Xi is a tensor with standard normal entries, applying (Tomioka & Suzuki, 2014, Theorem 1) we conclude that DISPLAYFORM2 holds with probability 1 − 2e −t 2 2 .

Combining (A.10) and (A.11) via the union bound together with (A.9) concludes the proof.

We begin the proof of this theorem by a few definitions regarding non-overlapping CNN models that simplify our exposition.

For these definitions it is convenient to view non-overlapping CNNs as a tree with the root of the tree corresponding to the output of the CNN and the leaves corresponding to input features.

In this visualization D − th layer of the tree corresponds to the th layer.

FIG10 depicts such a tree visualization along with the definitions discussed below.

Definition A.10 (Path vector) A vector i ∈ R D+1 is a path vector if its zeroth coordinate satisfies 1 ≤ i0 ≤ p and for all D − 1 ≥ j ≥ 0, ij+1 obeys ij+1 = ⌈ij dj⌉. This implies 1 ≤ ij ≤ pj and i D = 1.

We note that in the tree visualization a path vector corresponds to a path connecting a leaf (input feature) to the root of the tree (output).

We use I to denote the set of all path vectors and note that I = p. We also define i(i) ∈ I be the vector whose zeroth entry is i0 = i. Stated differently i(i) is the path connecting the input feature i to the output.

Given a path i and a p dimensional vector v, we define v i ∶= v i 0 .

A sample path vector is depicted in bold in FIG10 which corresponds to i = (d1, 1, 1, 1).

DISPLAYFORM0 where mod(a, b) denotes the remainder of dividing integer a by b. In words the kernel path gain is multiplication of the kernel weights along the path and the activation path gain is the multiplication of the derivatives of the activations evaluated at the hidden units along the path.

For the path depicted in FIG10 in bold the kernel path gain is equal k i = k DISPLAYFORM1 and the activation path gain is equal to φ DISPLAYFORM2 Definition A.12 (CNN offsprings) Consider a CNN model of the form (2.1) with input x and inputs of hidden units given by {h DISPLAYFORM3 .

We will associate a set set (i) ⊂ {1, . . .

, p} to the ith hidden unit of layer defined as set (i) = {(i − 1)r + 1, (i − 1)r + 2, . . .

, ir } where r = p p .

By construction, this corresponds to the set of entries of the input data x thath ( ) i (x) is dependent on.

In the tree analogy set (i) are the leaves of the tree connected to hidden unit i in the th layer i.e. the set of offsprings of this hidden node.

We depict set2(p2) which are the offsprings of the last hidden node in layer two in FIG10 .We now will rewrite the population tensor in a form such that it is easier to see why it can be well approximated by a rank one tensor.

Note that since the tensorization operation is linear the population tensor is equal to DISPLAYFORM4 (A.12) Define the vector g CNN to be the population gradient vector i.e. g CNN = E[∇fCNN(x)] and note that Stein's lemma combined with (A.12) implies that DISPLAYFORM5

1 and the activation path gain is equal to φ DISPLAYFORM0 1 ).

The set set 2 (p 2 ) (offsprings of the last hidden node in layer two) is outlined.

Define a vector k ∈ R p such that ki = k i(i) .

Since k i(i) consists of the product of the kernel values across the path i(i) it is easy to see that the tensor K ∶= T (k) is equal to DISPLAYFORM1 and define the corresponding tensor V = T (v).

Therefore, (A.14) can be rewritten in the vector form g CNN = k ⊙ v where a ⊙ b denotes entry-wise (Hadamard) product between two vectors/matrices/tensors a and b of the same size.

Thus using (A.13) the population tensor can alternatively be written as DISPLAYFORM2 Therefore, the population tensor T is the outer product of the convolutional kernels whose entries are masked with another tensor V .

If the entries of the tensor V were all the same the population tensor would be exactly rank one with the factors revealing the convolutional kernel.

One natural choice for approximating the population tensor with a rank one matrix is to replace the masking tensor V with a scalar.

That is, use the approximation DISPLAYFORM3 is exactly such an approximation with c set to αCNN given by DISPLAYFORM4 To characterize the quality of this approximation note that DISPLAYFORM5 Here, (a) follows from the fact for a tensor, its spectral norm is smaller than its Frobenius norm, (b) from the definitions of T and LCNN, (c) from the fact that K = T (k) and DISPLAYFORM6 , and (e) from the fact that the Euclidean norm of the kronecker product of of vectors is equal to the product of the Euclidean norm of the indivisual vectors.

As a result of the latter inequality to prove Theorem 4.7 it suffices to show that DISPLAYFORM7 In the next lemma we prove a stronger statement.

Lemma A.13 Assume the activations are S-smooth.

Also consider a vector v ∈ R p with entries vi = DISPLAYFORM8 2 .

Here, i is the vector path that starts at input feature i.

Before proving this lemma let us explain how (A.16) follows from this lemma.

To show this we use the kernel diffuseness assumption introduced in Definition 4.6.

This definition implies that k DISPLAYFORM9 This completes the proof of Theorem 4.7.

All that remains is to prove Lemma A.13 which is the subject of the next section.

A.3.1 PROOF OF LEMMA A.13To bound the difference between vi and αCNN consider the path i = i(i) and define the variables {ai} DISPLAYFORM10 Note that a D = vi and a0 = αCNN.

To bound the difference vi − αCNN = a D − a0 we use a telescopic sum DISPLAYFORM11 (A.17)We thus focus on bounding each of the summands a − a −1 .

Setting DISPLAYFORM12 , this can be written as DISPLAYFORM13 Using γ ≤ 1 (which follows from the assumption that the activations are 1-Lipschitz), it suffices to bound DISPLAYFORM14 (A.18)To this aim we state two useful lemmas whose proofs are deferred to Sections A.3.1.1 and A.3.1.2.Lemma A.14 Let X, Y, Z be random variables where X is independent of Z. Let f be an L-Lipschitz function.

DISPLAYFORM15 is a deterministic function of the entries of x indexed by set (i).

In other words, there exists a function f such thath DISPLAYFORM16 With these lemmas in-hand we return to bounding (A.18).

To this aim we decomposeh DISPLAYFORM17 where the r term is the contribution of the entries of h ( −1) other than i −1 .

By the non-overlapping assumption, r is independent of θ −1 as well as h DISPLAYFORM18 and θ −1 is a function of x set −1 (i −1 ) whereas r is a function of the entries over the complement set (i ) − set −1 (i −1 ).

With these observations, applying Lemma A.14 with DISPLAYFORM19 , Z = θ −1 and using the fact that θ −1 ≤ 1 which holds due to 1-Lipschitzness of σ 's, we conclude that DISPLAYFORM20 Here, S is the smoothness of σ and Lipschitz constant of φ ′ .

To conclude, we need to assess the E zm(Y )term.

Now note that starting from x, each entry of h ( −1) is obtained by applying a sequence of inner products with {k DISPLAYFORM21 i=1 and activations σ (⋅), which implies h DISPLAYFORM22 2 .

Hence, zm(Y ) obeys the tail bound DISPLAYFORM23 Using a standard integration by parts argument the latter implies that DISPLAYFORM24 concluding the upper-bound on each summand of (A.17).

Combining such upper bounds (A.17) implies DISPLAYFORM25 2 ∶= κ i .

This concludes the proof of Lemma A.13.A.3.1.1 Proof of Lemma A.14 Using the independence of X, Z, we can write DISPLAYFORM26 Now, using Lipschitzness of f , we deterministically have that DISPLAYFORM27 Taking absolute values we arrive at (1) the result is trivial becauseh DISPLAYFORM28 DISPLAYFORM29 is a weighted sum of entries corresponding to set1(i).

Suppose the claim holds for all layers less than or equal to − 1 andh DISPLAYFORM30 For layer , we can use the fact that set DISPLAYFORM31 The latter is clearly a deterministic function of x set (i) .

Also it is independent of entry i because it simply chunks the vector x set (i) into d sub-vectors and returns a sum of weighted functions of these sub-vectors.

Here, the weights are the entries of k ( ) and the functions are given by σ −1 (f −1 (⋅)) (also note that the activation output is simply h DISPLAYFORM32

The first part of the theorem follows trivially by combining Theorems 4.2 and 4.7.

To translate a bound on the tensor spectral norm of Tn − LCNN to a bound on learning the kernels, requires a perturbation argument for tensor decompositions.

This is the subject of the next lemma.

DISPLAYFORM0 v be a rank one tensor with {vi} DISPLAYFORM1 vectors of unit norm.

Also assume E is a perturbation tensor obeying E ≤ δ.

Set u1, u2, . . . , u D = arg max DISPLAYFORM2 The proof of Theorem 4.8 is complete by applying Lemma A.16 above with v = k ( ) , u =k ( ) , γ = αCNN and E = Tn − LCNN.

All that remains is to prove Lemma A.16 which is the subject of the next section.

A.4.1 PROOF OF LEMMA A.16 To prove this lemma first note that for any two rank one tensors we have DISPLAYFORM3 Using this equality together with the fact that the vectors {u } D =1 are a maximizer for (A.21) we conclude that DISPLAYFORM4 Furthermore, note that • ReLU model: φ(x) = max(0, x) with the added assumption that the kernels have mean larger than zero and are modestly diffused.

Specifically, assume DISPLAYFORM5 DISPLAYFORM6 • softplus model: φ(x) = log (1 + e x ) with the added assumption that the kernels have mean larger than zero, are modestly diffused and have a sufficiently large Euclidean norm.

Specifically, assume(1 DISPLAYFORM7 Proof ReLU: In this case note that DISPLAYFORM8 we arrive at DISPLAYFORM9 Since the entries of h DISPLAYFORM10 ].

where in the last inequality we used the fact that 1 T k ( ) ≥ 0 and applied Jenson's inequality for a convex φ.

Applying this inequality recursively we arrive at DISPLAYFORM11 Using the latter in (B.3) we arrive at DISPLAYFORM12 Thus using the diffusion assumption (B.2) in the latter inequality we arrive at DISPLAYFORM13 Thus using the fact that for 0 ≤ x1, x2, ..., xn ≤ 1 we have DISPLAYFORM14 we conclude that DISPLAYFORM15 In the above E1(x) = ∫ DISPLAYFORM16 Thus using the fact that for 0 ≤ x1, x2, ..., xn ≤ 1 we have DISPLAYFORM17 we conclude that DISPLAYFORM18 We bound the expected value of the hidden unites similar to the argument for ReLU activations.

The only difference is that in the identity (B.4) we need to use the softplus activation in lieu of the ReLU activation for φ.

Therefore, (B.4) changes to DISPLAYFORM19 DISPLAYFORM20 Thus using t = E h ( ) i 2 we arrive at DISPLAYFORM21 Thus using the diffusion assumption (B.2) with ν ≥ 4 we have DISPLAYFORM22 log ν 2 Also using (B.6) and assuming k DISPLAYFORM23 log ν Plugging the latter two inequalities in (B.5), allows to conclude that for ν ≥ 10 DISPLAYFORM24

We note that DeepTD operates by accurately approximating the rank one tensor ⊗ D =1 k ( ) from data.

Therefore, DeepTD can only recover the convolutional kernels up to Sign/Scale Ambiguities (SSA).

In general, it may not be possible to recover the ground truth kernels from the training data.

For instance, when activations are ReLU, the norms of the kernels cannot be estimated from data as multiplying a kernel and dividing another by the same positive scalar leads to the same training data.

However, we can try to learn a good approximationfCNN() of the network fCNN() to minimize the risk E[(fCNN(x) −fCNN(x)) 2 ].To this aim, we introduce Centered Empirical Risk Minimization (CERM) which is a slight modification of Empirical Risk Minimization (ERM).

Let us first describe how finding a goodfCNN() can be formulated with CERM.

Given n i.i.d.

data points {(xi, yi)} n i=1 ∼ (x, y), and a function class F, CERM applies ERM after centering the residuals.

Given f ∈ F, define the average residual function ravg(f ) = 1 n ∑ n i=1 yi − f (xi).

We define the Centered Empirical Risk Minimizer aŝ DISPLAYFORM0 The remarkable benefit of CERM over ERM is the fact that, the learning rate doesn't suffer from the label or function bias.

This is in similar nature to the DeepTD algorithm that applies label centering.

In the proofs (in particular Section C.2, Theorem C.2) we provide a generalization bound on the CERM solution (C.1) in terms of the Lipschitz covering number of the function space.

While (C.1) can be used to learn all kernels, it does not provide an efficient algorithm.

Instead, we will use CERM to resolve SSA after estimating the kernels via DeepTD.

Interestingly, this approach only requires a few (O(D)) extra training samples.

Inspired from CERM, in Section C.1, we propose a greedy algorithm to address SSA.

We will apply CERM to the following function class with bounded kernels, The above theorem states that CERM finds the sign/scale ambiguity that accurate estimates the labels on new data as long as the number of samples which are used in CERM exceeds the depth of the network by constant/log factors.

In the next section we present a greedy heuristic for finding the CERM estimate.

DISPLAYFORM1

In order to resolve SSA, inspired from CERM, we propose Algorithm 1 which operates over the function class, Fk ∶= γf ∶ R p ↦ R f is a CNN of the form (2.1) with kernels {β k ( ) } D

with β ∈ {1, −1}, γ ≥ 0 . (C.4) It first determines the signs β by locally optimizing the kernels and then finds a global scaling γ > 0.

In the first phase, the algorithm attempts to maximize the correlation between the centered labels yc,i = yi −n 2 .

While our approach is applicable to arbitrary activations, it is tailored towards homogeneous activations (φ(cx) = cφ(x)).

The reason is that for homogeneous activations, function classes (C.2) and (C.4) coincide and a single global scaling γ is sufficient.

Note that ReLU and the identity activation (i.e. no activation) are both homegeneous, in fact they are elements of a larger homogeneous activation family named Leaky ReLU.

Leaky ReLU is parametrized by some scalar 0 ≤ β ≤ 1 and defined as follows LReLU (x) =x if x ≥ 0, βx if x < 0.

In this section we prove a generalization bound for Centered Empirical Risk Minimization (CERM) (C.1).

The following theorem shows that using a finite sample size n, CERM is guaranteed to choose a function close to population's minimizer.

For the sake of this section f L∞ will be the Lipschitz constant of a function.

Combining these three inequalities ((C.10) and (C.11)) and substituting them in (C.6), we conclude that for all neighbors f δ , f , E(f ) − E(f δ ) ≤ O(Kδp).

Next we set δ = cε (pK) for a sufficiently small constant c > 0, to find that with probability at least 1−exp(−n), sup f ∈F E(f ) ≤ ε holds as long as the number of samples n obeys n ≥ O(max{ε −1 , ε −2 }K 2 s log

).

We define Lerm(f ) = In this section we will show how Theorem C.1 follows from Theorem C.2.

To this aim we need to show that Fk ,B has a small Lipscshitz covering number.

We construct the following cover F ′ for the set Fk ,B .

Let with them.

We now argue that F ′ provides a cover of F. Given f ∈ F with scalings β , there exists f Observe that fi−1 and fi have equal layers except the ith layer.

Let g1 be the function of the first i−1 layers and g2 be the function of layers i + 1 to D. We have that fi+1(x) − fi(x) = g2(φ(Ki(g1(x)))) − g2(φ(K ).

Now, since all kernels have Euclidean norm bounded by B ′ , we have fCNN() L∞ ≤ B and f L∞ ≤ B for all f ∈ F. This also implies zm(fCNN(x)) ψ 2 = O(B).

Hence, we can apply Theorem C.2 to conclude the proof of Theorem C.1.

@highlight

We consider a simplified deep convolutional neural network model. We show that all layers of this network can be approximately learned with a proper application of tensor decomposition.

@highlight

Provides theoretical guarantees for learning deep convolutional neural networks using rank-one tensor decomposition.

@highlight

This paper proposes a learning method for a restricted case of deep convolutional networks, where the layers are limited to the non-overlapping case and have only one output channel per layer

@highlight

Analyzes the problem of learning a very special class of CNNs: each layers consists of a single filter, applied to non-overlapping patches of the input.