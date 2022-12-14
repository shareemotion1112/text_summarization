Recent theoretical work has demonstrated that deep neural networks have superior performance over shallow networks, but their training is more difficult, e.g., they suffer from the vanishing gradient problem.

This problem can be typically resolved by the rectified linear unit (ReLU) activation.

However, here we show that even for such activation, deep and narrow neural networks (NNs) will converge to erroneous mean or median states of the target function depending on the loss with high probability.

Deep and narrow NNs are encountered in solving partial differential equations with high-order derivatives.

We demonstrate this collapse of such NNs both numerically and theoretically, and provide estimates of the probability of collapse.

We also construct a diagram of a safe region for designing NNs that avoid the collapse to erroneous states.

Finally, we examine different ways of initialization and normalization that may avoid the collapse problem.

Asymmetric initializations may reduce the probability of collapse but do not totally eliminate it.

The best-known universal approximation theorems of neural networks (NNs) were obtained almost three decades ago by BID5 and BID18 , stating that every measurable function can be approximated accurately by a single-hidden-layer neural network, i.e., a shallow neural network.

Although powerful, these results do not provide any information on the required size of a neural network to achieve a pre-specified accuracy.

In BID2 , the author analyzed the size of a neural network to approximate functions using Fourier transforms.

Subsequently, in BID27 , the authors considered optimal approximations of smooth and analytic functions in shallow networks, and demonstrated that −d/n neurons can uniformly approximate any C n -function on a compact set in R d with error .

This is an interesting result and it shows that to approximate a three-dimensional function with accuracy 10 −6 we need to design a NN with 10 18 neurons for a C 1 function, but for a very smooth function, e.g., C 6 , we only need 1000 neurons.

In the last 15 years, deep neural networks (i.e., networks with a large number of layers) have been used very effectively in diverse applications.

After some initial debate, at the present time, it seems that deep NNs perform better than shallow NNs of comparable size, e.g., a 3-layer NN with 10 neurons per layer may be a better approximator than a 1-layer NN with 30 neurons.

From the approximation point of view, there are several theoretical results to explain this superior performance.

In BID9 , the authors showed that a simple approximately radial function can be approximated by a small 3-layer feed-forward NN, but it cannot be approximated by any 2-layer network with the same accuracy irrespective of the activation function, unless its width is exponential in the dimension (see ; BID29 ; BID6 for further discussions).

In BID24 (see also Yarotsky (2017) ), the authors claimed that for -approximation of a large class of piecewise smooth functions using the rectified linear unit (ReLU) max(x, 0) activation function, a multilayer NN using Θ(log(1/ )) layers only needs O(poly log(1/ )) neurons, while Ω(poly(1/ )) neurons are required by NNs with o(log(1/ )) layers.

That is, the number of neurons required by a shallow network to approximate a function is exponentially larger than the corresponding number of neurons needed by a deep network for a given accuracy level of function approximation.

In BID33 , the authors studied approximation theory of a class of (possibly discontinuous) piecewise C β functions for ReLU NN, and they found that no more than O( −2(d−1)/β ) nonzero weights are required to approximate the function in the L 2 sense, which proves to be optimal.

Under this optimality condition, they also show that a minimum depth (up to a multiplicative constant) is given by β/d to achieve optimal approximation rates.

As for the expressive power of NNs in terms of the width, BID25 showed that any Lebesgue integrable function from R d to R can be approximated by a ReLU forward NN of width d + 4 with respect to L 1 distance, and cannot be approximated by any ReLU NN whose width is no more than d. BID14 showed that any continuous function can be approximated by a ReLU forward NN of width d in + d out , and they also give a quantitative estimate of the depth of the NN; here d in and d out are the dimensions of the input and output, respectively.

For classification problems, networks with a pyramidal structure and a certain class of activation functions need to have width larger than the input dimension in order to produce disconnected decision regions BID31 .With regards to optimum activation function employed in the NN approximation, before 2010 the two commonly used non-linear activation functions were the logistic sigmoid 1/(1 + e −x ) and the hyperbolic tangent (tanh); they are essentially the same function by simple re-scaling, i.e., tanh(x) = 2 sigmoid(2x) − 1.

The deep neural networks with these two activations are difficult to train BID11 .

The non-zero mean of the sigmoid induces important singular values in the Hessian BID23 , and they both suffer from the vanishing gradient problem, especially through neurons near saturation BID11 .

In 2011, ReLU was proposed, which avoids the vanishing gradient problem because of its linearity, and also results in highly sparse NNs BID12 .

Since then, ReLU and its variants including leaky ReLU (LReLU) BID26 , parametric ReLU (PReLU) BID15 and ELU BID4 are favored in almost all deep learning models.

Thus, in this study, we focus on the ReLU activation.

While the aforementioned theoretical results are very powerful, they do not necessarily coincide with the results of training of NNs in practice which is NP-hard (Šíma, 2002) .

For example, while the theory may suggest that the approximation of a multi-dimensional smooth function is accurate for NN with 10 layers and 5 neurons per layer, it may not be possible to realize this NN approximation in practice.

BID10 first proved that existence of local minima poses a serious problem in learning of NNs.

After that, more work has been done to understand bad local minima under different assumptions (Zhou & Liang, 2017; BID7 BID36 Wu et al., 2018; Yun et al., 2018) .

Besides local minima, singularity BID0 and bad saddle points BID20 ) also affect training of NNs.

Our paper focuses on a particular kind of bad local minima, i.e., those encountered in deep and narrow neural networks collapse with high probability.

This is the topic of our work presented in this paper.

Our results are summarized in FIG9 , which shows a diagram of the safe region of training to achieve the theoretically expected accuracy.

As we show in the next section through numerical simulations as well as in subsequent sections through theoretical results, there is very high probability that for deep and narrow ReLU NNs will converge to an erroneous state, which may be the mean value of the function or its partial mean value.

However, if the NN is trained with proper normalization techniques, such as batch normalization BID19 , the collapse can be avoided.

Not every normalization technique is effective, for example, weight normalization BID37 leads to the collapse of the NN.

In this section, we will present several numerical tests for one-and two-dimensional functions of different regularity to demonstrate that deep and narrow NNs collapse to the mean value or partial mean value of the function.

It is well known that it is hard to train deep neural networks.

Here we show through numerical simulations that the situation gets even worse if the neural networks is narrow.

First, we use a 10-layer ReLU network with width 2 to approximate y(x) = |x|, and choose the mean squared error (MSE) as the loss.

In fact, y(x) can be represented exactly by a 2-layer ReLU NN with width 2, DISPLAYFORM0 However, our numerical tests show that there is a high probability (∼ 90%) for the NN to collapse to the mean value of y(x) FIG20 , no matter what kernel initializers (He normal (He et al., 2015) , LeCun normal (LeCun et al., 1998; BID22 , Glorot uniform (Glorot & Bengio, 2010) ) or optimizers (first order or second order including SGD, SGDNesterov (Sutskever et al., 2013) , AdaGrad BID8 ), AdaDelta (Zeiler, 2012 , RMSProp BID17 , Adam BID21 , BFGS BID32 , L-BFGS BID3 ) are employed.

The training data were sampled from a uniform distribution on [− √ 3, √ 3], and the minibatch size was chosen as 128 during training.

We find that when this happens, in most cases the bias in the last layer is the mean value of the function y(x), and the composition of all the previous layers is equivalent to a zero function.

It can be proved that under these conditions, the gradient vanishes, i.e., the optimization stops (Corollary 5).

For functions of different regularity, we observed the same collapse problem, see FIG1 for the C ∞ function y(x) = x sin(5x) and FIG2 for the L 2 function y(x) = 1 {x>0} + 0.2 sin(5x).For multi-dimensional inputs and outputs, this collapse phenomenon is also observed in our simulations.

Here, we test the target function y(x) with d in = 2 and d out = 2, which can be represented by a 2-layer neural network with width 4, y( DISPLAYFORM1 When training a 10-layer ReLU network with width 4, there is a very high probability for the NN to collapse to the mean value or with low probability to the partial mean value of y(x) FIG3 .

FIG20 : Demonstration of the neural network collapse to the mean value (A, with very high probability) or the partial mean value (B, with low probability) for the C 0 target function y(x) = |x|.

The gradient vanishes in both cases (see Corollaries 5 and 6).

A 10-layer ReLU neural network with width 2 is employed in both (A) and (B).

The biases are initialized to 0, and the weights are randomly initialized from a symmetric distribution.

The loss function is MSE.

We also observed the same collapse problem for other losses, such as the mean absolute error (MAE); the results are summarized in FIG4 for three different functions with varying regularity.

Furthermore, we find that for MSE loss, the constant is the mean value of the target function, while for MAE it is the median value.

As we demonstrated above, when the weights of the ReLU NN are randomly initialized from a symmetric distribution, the deep and narrow NN will collapse with high probability.

This type of initialization is widely used in real applications.

Here, we demonstrate that this initialization avoids the problem of exploding/vanishing mean activation length, therefore this is beneficial for training neural networks.

We study a feed-forward neural network N : DISPLAYFORM0 .

The weights and biases in the layer l are an DISPLAYFORM1 , respectively.

The input is x 0 ∈ R din , and the neural activity in the layer DISPLAYFORM2 The feed-forward dynamics is given by DISPLAYFORM3 where φ is a component-wise activation function.

Following the work in Poole et al. FORMULA6 , we investigate how the length of the input propagates through neural networks.

The normalized squared length of the vector before activation at each layer is defined as DISPLAYFORM4 where h respectively, then the length at layer l can be obtained from its previous layer (see the proof in the appendix of BID35 , which we include here in Appendix A) DISPLAYFORM5 where DISPLAYFORM6 is the standard Gaussian measure, and the initial condition is DISPLAYFORM7 When φ is ReLU, the recursion is simplified to DISPLAYFORM8 For ReLU, He normal BID15 , i.e., σ 2 w = 2 and σ b = 0, is widely used.

This choice DISPLAYFORM9 , which neither shrinks nor expands the inputs.

In fact, this result explains the success of He normal in applications.

A parallel work by BID13 shows that initializing weights from a symmetric distribution with variance 2/fan-in (fan-in is the dimension of the input of each layer) avoids the problem of exploding/vanishing mean activation length.

Here we arrived at the same conclusion but with much less work.

In this section, we present the theoretical analysis of the collapse behavior observed in Section 2, and we also derive an estimate of the probability of this collapse.

We start by stating the following assumptions for a ReLU feed-forward neural network N (x 0 ) : DISPLAYFORM0 A1 The domain Ω ⊂ R din for N is a connected space with at least two points; DISPLAYFORM1 of any layer l ∈ {1, 2, . . .

, L} is a random matrix, where the joint distribution of (W Remark: We point out here that the connectedness in assumption A1 is a very weak requirement for the input space.

The weights in a neural network are usually sampled independently from continuous distributions in real applications, and thus the assumption A2 is satisfied at the NN initialization stage; during training, the assumption A2 is usually maintained due to stochastic gradients of minibatch.

Lemma 1.

With assumptions A1 and A2, if N (x 0 ) is a constant function, then there exists a layer l ∈ {1, . . .

, L − 1} such that h l ≤ 0 1 and x l = 0 ∀x 0 ∈ Ω, with probability 1 (wp1).Corollary 2.

With assumptions A1 and A2, if N (x 0 ) is bias-free and a constant function, then there exists a layer l ∈ {1, . . .

, L − 1} such that for any n ≥ l, it holds h n ≤ 0 and x n = 0 wp1.Lemma 3.

With assumptions A1 and A2, if N (x 0 ) is a constant function, then any order gradients of the loss function with respect to the weights and biases in layers 1, . . .

, l vanish, where l is the layer obtained in Lemma 1.

Theorem 4.

For a ReLU feed-forward neural network N (x 0 ) with assumption A1, if the assumption A2 is satisfied during the initialization, and there exists a layer l such that x l (x 0 ) ≡ 0 for any input x 0 , then for any function y(x 0 ) and x 0 ∈ Ω, N is eventually optimized to a constant function when training by a gradient based optimizer.

If using L 2 loss and DISPLAYFORM2 DISPLAYFORM3 is a constant function with the value E[y], then the gradients of the loss function with respect to any weight or bias vanish when using the L 2 loss.

Corollary 5 can be generalized to the following corollary including more general converged mean states.

Corollary 6.

With assumptions A1 and A2, for a ReLU feed-forward neural network N (x 0 ) and any bounded function y(x 0 ), x 0 ∈ Ω, if ∃K 1 , . . .

, K n ⊂ Ω and each K i is a connected domain with at least two points, such that DISPLAYFORM4 then the gradients of the loss function with respect to any weight or bias vanish when using the L 2 loss.

Here x 0 Ki is the random variable of x 0 restricted to K i .See Appendices F and G for the proofs of Corollaries 5 and 6.

We can see that Corollary 5 is a special case of Corollary 6 with ∪ n i=1 K i = Ω. Lemma 7.

Let us assume that a one-layer ReLU feed-forward neural network N 1 is initialized independently by symmetric nonzero distributions, i.e., any weight or bias of N 1 is initialized by a symmetric nonzero distribution, which can be different for different parameters.

Then, for any fixed input the corresponding output is zero with probability (1/2) dout , except the special case where all biases and the input are zero yielding that the output is always zero.

See Appendices H and I for the proofs of Lemma 7 and Theorem 8.

Although biases are initialized to 0 in most applications, for the sake of completeness, we also consider the case where biases are not initialized to 0.

Proposition 9.

If a ReLU feed-forward neural network N with L layers assembled width N 1 , . . .

, N L is initialized randomly by symmetric nonzero distributions (weights and biases), then for any fixed nonzero input, the corresponding output is zero with probability (1/2) See Appendix J for the proof of Proposition 9.

We note that Theorem 8 provides the probability for any given input, but in Theorem 4 it requires that the entire neural network is a zero function.

Hence, the probability in Theorem 8 is an upper bound.

In the following theorem, we give a theoretical formula of the probability for the NN with width 2.

Proposition 10.

Suppose the origin is an interior point of Ω. Consider a bias-free ReLU neural network with d in = 1, width 2 and L layers, and weights are initialized randomly by symmetric nonzero distributions.

Then for this neural network, the probability of being initialized to a constant function is the last component of π L , where DISPLAYFORM5 with π 1 and P being the probability distribution after the first layer and the probability transition matrix when one more layer is added, respectively.

Here every layer employs the ReLU activation.

See Appendix K for the derivation of π 1 and P .

For general cases, we found that it is hard to obtain an explicit expression for the probability, so we used numerical simulations instead, where 1 million samples of random initialization are used to calculate each probability estimation.

We show both theoretically (Theorem 8, Propositions 9 and 10) and numerically that NN has the same probability to collapse no matter what symmetric distributions are used, even if different distributions are used for different weights.

On the other hand, to keep the collapse probability less than p, because the probability obtained in Theorem 8 is an upper bound, which corresponds to a safer maximum number of layers, we have that 1 − Π L l=1 (1 − (1/2) N ) ≤ p, which implies the upper bound of the depth of NN DISPLAYFORM6 Theorem 8 shows that when the NN gets deeper and narrower, the probability of the NN initialized to a zero function is higher FIG9 .

Hence, we have higher probability of vanishing gradient in almost all the layers, rather than just some neurons.

In our experiments, we also found that there is very high probability that the gradient is 0 for all parameters except in the last layer, because ReLU is not used in the last layer.

During the optimization, the neural network thus can only optimize the parameters in the last layer (Theorem 4).

When we design a neural network, we should keep the probability less than 1% or 10%.

As a practical guide, we constructed a diagram shown in FIG9 that includes both theoretical predictions and our numerical tests.

We see that as the number of layers increases, the numerical tests match closer the theoretical results.

It is clear from the diagram that a 10-layer NN of width 10 has a probability of only 1% to collapse whereas a 10-layer NN of width 5 has a probability greater than 10% to collapse; for width of three the probability is greater than 60%.

In this section, we present some training techniques and examine which ones do not suffer from the collapse problem.

Our analysis applies for any symmetric initialization, so it is straightforward to consider asymmetric initializations.

The asymmetric initializations proposed in the literature include orthogonal initialization BID38 and layer-sequential unit-variance (LSUV) initialization BID30 .

LSUV is the orthogonal initialization combined with rescaling of weights such that the output of each layer has unit variance.

Because weight rescaling cannot make the output escape from the negative part of ReLU, it is sufficient to consider the orthogonal initialization.

The probability of collapse when using orthogonal initialization is very close to and a little lower than that when using symmetric distributions FIG10 .

Therefore, orthogonal initialization cannot treat the collapse problem.

As we have shown in the previous section, deep and narrow neural networks cannot be trained well directly with gradient-based optimizers.

Here, we employ several widely used normalization The maximum number of layers of a neural network can be used at different width to keep the probability of collapse less than 1% or 10%.

The region below the blue line is the safe region when we design a neural network.

As the width increases the theoretical predictions match closer with our numerical simulations.

techniques to train this kind of networks.

We do not consider some methods, such as Highway (Srivastava et al., 2015) and ResNet BID16 , because in these architectures the neural nets are no longer the standard feed-forward neural networks.

Current normalization methods mainly include batch normalization (BN) BID19 , layer normalization (LN) BID1 , weight normalization (WN) BID37 , instance normalization (IN) (Ulyanov et al., 2016) , group normalization (GN) (Wu & He, 2018) , and scaled exponential linear units (SELU) BID22 .

BN, LN, IN and GN are similar techniques and follow the same formulation, see Wu & He (2018) for the comparison.

Because we focus on the performance of these normalization methods on narrow nets and the width of the neural network must be larger than the dimension of the input to achieve a good approximation, we only test the normalization methods on low dimensional inputs.

However, LN, IN and GN perform normalization on each training data individually, and hence they cannot be used in our low-dimensional situations.

Hence, we only examine BN, WN and SELU.

BN is applied before activations while for SELU LeCun normal initialization is used BID22 .

Our simulations show that the neural network can successfully escape from the collapsed areas and approximate the target function with a small error, when BN or SELU are employed.

BN changes the weights and biases not only depending on the gradients, and different from ReLU the negative values do not vanish in SELU.

However, WN failed because it is only a simple re-parameterization of the weight vectors.

Moreover, our simulations show that the issue of collapse cannot be solved by dropout, which induces sparsity and more zero activations (Srivastava et al., 2014) .

We consider here ReLU neural networks for approximating multi-dimensional functions of different regularity, and in particular we focus on deep and narrow NNs due to their reportedly good approximation properties.

However, we found that training such NNs is problematic because they converge to erroneous means or partial means or medians of the target function.

We demonstrated this collapse problem numerically using one-and two-dimensional functions with C 0 , C ∞ and L 2 regularity.

These numerical results are independent of the optimizers we used; the converged state depends on the loss but changing the loss function does not lead to correct answers.

In particular, we have observed that the NN with MSE loss converges to the mean or partial mean values while the NN with MAE loss converges to the median values.

This collapse phenomenon is induced by the symmetric random initialization, which is popular in practice because it maintains the length of the outputs of each layer as we show theoretically in Section 3.We analyze theoretically the collapse phenomenon by first proving that if a NN is a constant function then there must exist a layer with output 0 and the gradients of weights and biases in all the previous layers vanish (Lemma 1, Corollary 2, and Lemma 3).

Subsequently, we prove that if such conditions are met, then the NN will converge to a constant value depending on the loss function (Theorem 4).

Furthermore, if the output of NN is equal to the mean value of the target function, the gradients of weights and biases vanish (Corollaries 5 and 6).

In Lemma 7 and Theorem 8 and Proposition 9, we derive estimates of the probability of collapse for general cases, and in Proposition 10, we derive a more precise estimate for deep NNs with width 2.

These theoretical estimates are verified numerically by tests using NNs with different layers and widths.

Based on these results, we construct a diagram which can be used as a practical guideline in designing deep and narrow NNs that do not suffer from the collapse phenomenon.

Finally, we examine different methods of preventing deep and narrow NNs from converging to erroneous states.

In particular, we find that asymmetric initializations including orthogonal initialization and LSUV cannot be used to avoid this collapse.

However, some normalization techniques such as batch normalization and SELU can be used successfully to prevent the collapse of deep and narrow NNs; on the other hand, weight normalization fails.

Similarly, we examine the effect of dropout which, however, also fails.

DISPLAYFORM0 DISPLAYFORM1 is a summation of independent Gaussian random variables and thus is a Gaussian distribution.

If l ≥ 3, by central limit theorem, DISPLAYFORM2 2 is the standard Gaussian measure.

Therefore, DISPLAYFORM3 B PROOF OF LEMMA 1Lemma 11.

Let A ∈ R n×m be a random matrix, where {A ij } i∈{1,2,...,n},j∈{1,2,...,m} are random variables, and the joint distribution of (A i1 , A i2 , . . .

, A im ) is absolutely continuous for i = 1, 2, . . .

, n. If x ∈ R m is a nonzero column vector, then P(Ax = 0) = 0.Proof.

Let us consider the first value of Ax, i.e., Proof.

By assumption A2 and Lemma 11, DISPLAYFORM4 is a constant function with respect to x 0 .

So we can assume that there is ReLU in the last layer, and prove that there exists a layer l ∈ {1, . . .

, L}, s.t., h l ≤ 0 and x l = 0 wp1 for every x 0 ∈ Ω.

We proceed in two steps.

DISPLAYFORM5 Because Ω ⊂ R din is a connected space with at least two points, then Ω has no isolated points, which impliesx 0 is not an isolated point.

Since the neural network is a continuous map, DISPLAYFORM6 , which contradicts the fact that x 1 is a constant function.

Therefore, h ≤ 0 and x 1 = 0.ii) Assume the theorem is true for L.

Then for L + 1, if x 1 = 0, choose l = 1 and we are done; otherwise, consider the NN without the first layer with x 1 ∈ Ω 1 as the input, denoted N 1 .

By i, Ω 1 is a connected space with at least two points.

Because N 1 is a constant function of x 1 and has L layers, by induction, there exists a layer whose output is zero.

Therefore, for the original neural network N , the output of such layer is also zero.

By i and ii, the statement is true for any L.

Proof.

By Lemma 1, there exists a layer l ∈ {1, . . .

, L − 1}, s.t.

h l ≤ 0 and x l = 0 wp1.

Because N is bias-free, h l+1 = W l+1 x l = 0 and x l+1 = ReLU(h l+1 ) = 0 wp1.

By induction, for any n ≥ l, h n ≤ 0 and x n = 0 wp1.

Proof.

Because x l ≡ 0, it is then obvious by backpropagation.

E PROOF OF THEOREM 4 DISPLAYFORM0 is a constant function, and then by Lemma 3, gradients of the loss function w.r.t.

the weights and biases in layers 1, . . . , l vanish.

Hence, the weights and biases in layers 1, . . .

, l will not change when using a gradient based optimizer, which implies N (x 0 ) is always a constant function depending on the weights and biases in layers l + 1, . . .

, L. Therefore, N will be optimized to a constant function, which has the smallest loss.

For L 2 loss, this constant with the smallest loss is E[y].

For L 1 loss, this constant with the smallest loss is its median.

Proof.

Because N (x 0 ) is a constant function, by Lemma 1 and Theorem 4, N is optimized to E[y].

Also, since N is equal to E[y], gradients vanish.

Proof.

It suffices to show that gradients vanish for x 0 ∈ K i , i = 1, . . .

, n and DISPLAYFORM0 Ki )].

Similar to Corollary 5, gradients vanish when using the L 2 loss.ii) For x 0 ∈ Ω \ ∪ n i=1 K i , the loss at x 0 is 0, so gradients vanish.

By i and ii, gradients vanish when using the L 2 (MSE) loss.

H PROOF OF LEMMA 7Proof.

Let x = (x 1 , x 2 , . . . , x din ) be any input, and y = (y 1 , y 2 , . . .

, y dout ) be the corresponding output.

For i = 1, . . .

, d out , DISPLAYFORM1 So P(y i = 0) = 1 2 , and then P( DISPLAYFORM2 Here P denotes the probability.

Proof.

If the last layer also employs ReLU activation, by Lemma 7, P( DISPLAYFORM0 The last equality holds because P(x 0 = 0) = 1.If in the last layer we do not apply ReLU activation, then P( DISPLAYFORM1

Proof.

If the last layer also has ReLU activation, by Lemma 7, DISPLAYFORM0 If the last layer does not have ReLU activation, and L ≥ 2, then DISPLAYFORM1 For L = 1, N is a single layer perceptron, which is a trivial case.

Proof.

We consider a ReLU neural network with d in = 1 and each hidden layer with width 2.

Because all biases are zero, then it is easy to see the following fact: when the input is 0, the output of any neuron in any layer is 0; when the input is negative, the output of any neuron in any layer is a linear function with respect to the input; when the input is positive, the output of any neuron in any layer is also a linear function with respect to the input.

Because the origin is an interior point of Ω, then it suffices to consider a subset [−a, a] ⊂ Ω with a ∈ R + .

The output of each hidden layer has 16 possible cases: Note that in this case we can assume that ω = (cos θ, sin θ), θ ∈ (0, π 2 ) and ω * = (−1, 0) is a constant vector.

It is easy to see that ∠(ω, ω * ) = π − θ, and hence the probabilities of cases (1), (6), FORMULA6 and FORMULA6 Similarly, the probabilities of cases FORMULA7 , (3) , FORMULA18 , (8) , (9) , FORMULA6 , FORMULA6 and FORMULA6 FORMULA17 , (7) , FORMULA6 and FORMULA6 iii) Case (4) (the same method can be applied for cases (8) and FORMULA6 .It is easy to see that the probabilities of cases FORMULA17 , (8) , FORMULA6 and FORMULA6 .

Note that in this case, ω 1 > 0 and ω * 1 < 0, and thus it is not hard to see that the probabilities of cases (1), (6), FORMULA6 and FORMULA6 .

Therefore, the probabilities of all the 16 cases are 1 16 .

vi) Case (13) (the same method can be applied for cases (14) and FORMULA6 Similar to the argument of the case (4), it is easily to see that the probabilities for cases (13), FORMULA6 , FORMULA6 and FORMULA6 FORMULA6 The output of the next layer is the case (16) with probability 1.

<|TLDR|>

@highlight

Deep and narrow neural networks will converge to erroneous mean or median states of the target function depending on the loss with high probability.

@highlight

This paper studies failure modes of deep and narrow networks, focusing on as small as possible models for which the undesired behavior occurs.

@highlight

This paper shows that the training of deep ReLU neural networks will converge to a constant classifier with high probability over random initialization if hidden layer widths are too small.