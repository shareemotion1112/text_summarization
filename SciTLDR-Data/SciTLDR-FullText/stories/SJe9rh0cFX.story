Compression is a key step to deploy large neural networks on resource-constrained platforms.

As a popular compression technique, quantization constrains the number of distinct weight values and thus reducing the number of bits required to represent and store each weight.

In this paper, we study the representation power of quantized neural networks.

First, we prove the universal approximability of quantized ReLU networks on a wide class of functions.

Then we provide upper bounds on the number of weights and the memory size for a given approximation error bound and the bit-width of weights for function-independent and function-dependent structures.

Our results reveal that, to attain an approximation error bound of $\epsilon$, the number of weights needed by a quantized network is no more than $\mathcal{O}\left(\log^5(1/\epsilon)\right)$ times that of an unquantized network.

This overhead is of much lower order than the lower bound of the number of weights needed for the error bound, supporting the empirical success of various quantization techniques.

To the best of our knowledge, this is the first in-depth study on the complexity bounds of quantized neural networks.

Various deep neural networks deliver state-of-the-art performance on many tasks such as object recognition and natural language processing using new learning strategies and architectures BID11 Kumar et al., 2016; Ioffe & Szegedy, 2015; Vaswani et al., 2017) .

Their prevalence has extended to embedded or mobile devices for edge intelligence, where security, reliability or latency constraints refrain the networks from running on servers or in clouds.

However, large network sizes with the associated expensive computation and memory consumption make edge intelligence even more challenging BID2 Sandler et al., 2018) .In response, as will be more detailed in Section 2, substantial effort has been made to reduce the memory consumption of neural networks while minimizing the accuracy loss.

The memory consumption of neural networks can be reduced by either directly reducing the number of weights or decreasing the number of bits (bit-width) needed to represent and store each weight, which can be employed on top of each other BID3 .

The number of weights can be reduced by pruning BID9 , weight sparsifying (Liu et al., 2015) , structured sparsity learning BID14 and low rank approximation BID5 .

The bit-width is reduced by quantization that maps data to a smaller set of distinct levels (Sze et al., 2017) .

Note that while quantization may stand for linear quantization only (Li et al., 2017; BID7 or nonlinear quantization only BID8 BID3 in different works, our discussion will cover both cases.

However, as of today quantization is still only empirically shown to be robust and effective to compress various neural network architectures (Hubara et al., 2016; BID20 BID22 .

Its theoretical foundation still remains mostly missing.

Specifically, many important questions remain unanswered.

For example:• Why even binarized networks, those most extremely quantized with bit-width down to one, still work well in some cases?• To what extent will quantization decrease the expressive power of a network?

Alternatively, what is the overhead induced by weight quantization in order to maintain the same accuracy?In this paper, we provide some insights into these questions from a theoretical perspective.

We focus on ReLU networks, which is among the most widely used in deep neural networks BID15 .

We follow the idea from BID16 to prove the complexity bound by constructing a network, but with new and additional construction components essential for quantized networks.

Specifically, given the number of distinct weight values λ and a target function f , we construct a network that can approximate f with an arbitrarily small error bound to prove the universal approximability.

The memory size of this network then naturally serves as an upper bound for the minimal network size.

The high-level idea of our approach is to replace basic units in an unquantized network with quantized sub-networks 1 that approximate these basic units.

For example, we can approximate a connection with any weight in an unquantized network by a quantized sub-network that only uses a finite number of given weight values.

Even though the approximation of a single unit can be made arbitrarily accurate in principle with unlimited resources (such as increased network depth), in practice, there exists some inevitable residual error at every approximation, all of which could propagate throughout the entire network.

The challenge becomes, however, how to mathematically prove that we can still achieve the end-to-end arbitrary small error bound even if these unavoidable residual errors caused by quantization can be propagated throughout the entire network.

This paper finds a solution to solve the above challenge.

In doing so, we have to propose a number of new ideas to solve related challenges, including judiciously choosing the proper finite weight values, constructing the approximation sub-networks as efficient as possible (to have a tight upper bound), and striking a good balance among the complexities of different approximation steps.

Based on the bounds derived, we compare them with the available results on unquantized neural networks and discuss its implications.

In particular, the main contributions of this paper include:• We prove that even the most extremely quantized ReLU networks using two distinct weight values are capable of representing a wide class of functions with arbitrary accuracy.• Given the number of distinct weights and the desired approximation error bound, we provide upper bounds on the number of weights and the memory size.

We further show that our upper bounds have good tightness by comparing them with the lower bound of unquantized ReLU networks established in the literature.• We show that, to attain the same approximation error bound , the number of weights needed by a quantized network is no more than O log 5 (1/ ) times that of an unquantized network.

This overhead is of much lower order compared with even the lower bound of the number of weights needed for the error bound.

This partially explains why many state-ofthe-art quantization schemes work well in practice.• We demonstrate how a theoretical complexity bound can be used to estimate an optimal bit-width, which in turn enables the best cost-effectiveness for a given task.

The remainder of the paper is organized as follows.

Section 2 reviews related works.

Section 3 lays down the models and assumptions of our analysis.

We prove the universal approximability and the upper bounds with function-independent structure in Section 4 and extend it to function-dependent structure in Section 5.

We analyze the bound-based optimal bit-width in Section 6.

Finally, Section 7 discusses the results and gets back to the questions raised above.

Quantized Neural Networks: There are rich literatures on how to obtain quantized networks, either by linear quantization or nonlinear quantization BID19 Leng et al., 2017; Shayar et al., 2017) .

Linear quantization does mapping with a same distance between contiguous quantization levels and is usually implemented by storing weights as fixed-point numbers with reduced bit-width (Li et al., 2017; BID7 .

Nonlinear quantization maps the data to quantization levels that are not uniformly distributed and can be either preselected or learned from training.

Then the weights are stored using lossless binary coding (the index to a lookup table) instead of the actual values BID8 BID3 .

It is reported that a pruned AlexNet can be quantized to eight bits and five bits in convolutional layers and fully connected layers, respectively, without any loss of accuracy.

Similar results are also observed in LENET-300-100, LENET-5, and VGG-16 (Han et al., 2015a) .

One may argue that some of these benchmark networks are known to have redundancy.

However, recent works show that quantization works well even on networks that are designed to be extremely small and compact.

SqueezeNet, which is a state-of-the-art compact network, can be quantized to 8-bit while preserving the original accuracy BID7 Iandola et al., 2016) .

There are some representative works that can achieve little accuracy loss on ImageNet classification even using binary or ternary weights BID4 Rastegari et al., 2016; BID14 BID21 .

More aggressively, some works also reduce the precision of activations, e.g. (Hubara et al., 2016; Rastegari et al., 2016; BID6 .

Although the classification accuracy loss can be minimized, the universal approximation property is apparently lost, as with limited output precision the network cannot achieve arbitrary accuracy.

Accordingly, we do not include them in the discussion of this paper.

The limit of quantization is still unknown while the state-of-the-art keeps getting updated.

For example, VGG-16 is quantized to 3-bit while maintaining the original accuracy (Leng et al., 2017) .

Motivated by the great empirical success, the training of quantized neural networks has been analyzed theoretically, but not the network capacity (Li et al., 2017; BID3 .Universal Approximability and Complexity Bounds: The universal approximability of ReLU networks is proved in Mhaskar & Micchelli (1992) and revisited in (Sonoda & Murata, 2017

Throughout this paper, we define ReLU networks as feedforward neural networks with the ReLU activation function σ(x) = max(0, x).

The ReLU network considered includes multiple input units, a number of hidden units, and one output unit.

Without loss of generality, each unit can only connect to units in the next layer.

Our conclusions on ReLU networks can be extended to any other networks that use piecewise linear activation functions with finite breakpoints such as leaky ReLU and ReLU-6 immediately, as one can replace a ReLU network by an equivalent one using these activation functions while only increasing the number of units and weights by constant factors BID16 .We denote the finite number of distinct weight values as λ (λ ∈ Z + and λ ≥ 2), for both linear quantization and nonlinear quantization.

For linear quantization, without loss of generality, we assume the finite number of distinct weight values are given as {−1, λ } are uniformly spaced (hence called "linear") in (0, 1) and −1 is used to obtain the negative weight values.

For nonlinear quantization, we assume the finite number of distinct weight values are not constrained to any specific values, i.e., they can take any values as needed.

To store each weight, we only need log(λ) 2 bits to encode the index, i.e. the bit-width is log(λ).

The overhead to store sparse structures can be ignored because it varies depending on the implementation and can be easily reduced to the same order as the weight storage using techniques such as compressed sparse row (CSR) for nonlinear quantization.

The number of bits needed to store the codebook can also be ignored because it has lower order of complexity.

We consider any function f in the Sobolev space: DISPLAYFORM0 The space W n,∞ consists of all locally integrable function f : DISPLAYFORM1 , where |n| ≤ n and Ω is an open set in R d .

We denote this function space as F d,n in this paper.

Note that we only assume weak derivatives up to order n exist where n can be as small as 1 where the function is non-differentiable.

We also only assume the Lipschitz constant to be no greater than 1 for the simplicity of the derivation.

When the Lipschitz constant is bigger than 1, as long as it is bounded, the whole flow of the proof remains the same though the bound expression will scale accordingly.

When constructing the network to approximate any target function f , we consider two scenarios for deriving the bounds.

The first scenario is called function-dependent structure, where the constructed network topology and their associated weights are all affected by the choice of the target function.

In contrast, the second scenario is called function-independent structure, where the constructed network topology is independent of the choice of the target function in f ∈ F d,n with a given .

The principle behind these design choices (the network topology constructions and the choice of weights) is to achieve a tight upper bound as much as possible.

One might consider that we can transform an unquantized network within the error bound to a quantized one in a straightforward way by approximating every continuous-value weight with a combination of discrete weights with arbitrary accuracy.

However, the complexity of such approximation (number of discrete weights needed) depends on the distribution of those continuous-value weights (e.g., their min and max), which may vary depending on the training data and network structure and a closed-form expression for the upper bounds is not possible.

As such, a more elegant approach is needed.

Below we will establish a constructive approach which allows us to bound the approximation analytically.

We start our analysis with function-independent structure, where the network topology is fixed for any f ∈ F d,n and a given .

We first present the approximation of some basic functions by subnetworks in Section 4.1.

We then present the sub-network that approximates any weight in Section 4.2, and finally the approximation of general functions and our main results are in Section 4.3.

Proposition 1.

Denote the design parameter that determines the approximation error bound as r. The proof and the details of the sub-network constructed are included in Appendix A.1.

Once the approximation to squaring function is obtained, we get Proposition 2 by the fact that 2xy = (x + y) 2 − x 2 − y 2 .

Proposition 2.

Denote the design parameter that determines the approximation error bound as r. Given x ∈ [−1, 1], y ∈ [−1, 1], and only two weight values 1 2 and − 1 2 , there is a ReLU sub-network with two input units that implements a function × : R 2 → R, such that (i) if x = 0 or y = 0, then × (x, y) = 0; (ii) for any x, y, the error × = | × (x, y) − xy| ≤ 6 · 2 −2(r+1) ; (iii) the depth is O (r); (iv) the width is a constant; (v) the number of weights is O (r).Proof.

Build three sub-networks f r s as described in Proposition 1 and let DISPLAYFORM0 (2) Then the statement (i) is followed by property (i) of Proposition 1.

Using the error bound in Proposition 1 and Equation (2) , we get the error bound of × : DISPLAYFORM1 Since a sub-network B abs that computes σ(x) + σ(−x) can be constructed to get the absolute value of x trivially, we can construct × (x, y) as a linear combination of three parallel f r s and feed them with

Proposition 3.

Denote the design parameter that determines the approximation error bound as t. A connection with any weight w ∈ [−1, 1] can be approximated by a ReLU sub-network that has only λ ≥ 2 distinct weights, such that (i) the sub-network is equivalent to a connection with weight w while the approximation error is bounded by 2 −t i.e., |w − w| < 2 −t ; (ii) the depth is O λt DISPLAYFORM0 Proof.

Consider that we need a weight w to feed the input x to a unit in the next layer as wx.

With a limited number of distinct weight values, we can construct the weight we need by cascade and combination.

For clarity, we first consider w ≥ 0 and x ≥ 0, and relax these assumptions later.

The connections with w = 0 can be seen as an empty sub-network while w = 1 can be easily implemented by 4 units with weight 1 2 .

Now we show how to represent all integral multiples of 2 −t from 2 −t to 1 − 2 −t , which will lead to the statement (i) by choosing the nearest one from w as w .

Without loss of generality, we assume t 1 λ−1 is an integer.

We use λ weights that include − 1 2 and W : DISPLAYFORM1 We first construct all w from W c which is defined as DISPLAYFORM2 Similar to a numeral system with radix equal to t 1 λ−1 , any w i ∈ W c can be obtained by concatenating weights from W while every weights in W is used no greater than t 1 λ−1 − 1 times.

After that, all integral multiples of 2 −t from 2 −t to 1−2 −t can be represented by a binary expansion on W c .

Note that connections in the last layer for binary expansion use weight 1 2 , thus additional 2 −1 is multiplied to scale the resolution from 2 −(t−1) to 2 −t .

Since for any weight in W c we need to concatenate no more than λ t 1 λ−1 − 1 weights in a straight line, the sub-network has no greater than λ t 1 λ−1 − 1 + 1 layers, and no greater than 4tλ t 1 λ−1 − 1 + 8t + 4 weights.

We now relax the assumption w ≥ 0.

When w < 0, the sub-network can be constructed as w = |w|, while we use − 1 2 instead of 1 2 in the last layer.

To relax the assumption x ≥ 0, we can make a duplication of the sub-network.

Let all the weights in the first layer of the sub-network be 1 2 for one and − 1 2 for the other.

Here we are utilizing the gate property of ReLU.

In this way, one sub-network is activated only when x > 0 and the other is activated only when x < 0.

The sign of the output can be adjusted by flipping the sign of weights in the last layer.

Note that the configuration of the sub-network is solely determined by w and works for any input x.

The efficiency of the weight approximation is critical to the overall complexity.

Compared with the weight selection as {2 −1 , 2 DISPLAYFORM3 λ−1 }, our approximation reduces the number of weights by a factor of t λ−2 λ−1 .

With the help of Proposition 2 and Proposition 3, we are able to prove the upper bound for general functions.

Theorem 1.

For any f ∈ F d,n , given λ distinct weights, there is a ReLU network with fixed structure that can approximate f with any error ∈ (0, 1), such that (i) the depth is DISPLAYFORM0 the number of bits needed to store the network is O λ log (λ) log DISPLAYFORM1 The complete proof and the network constructed can be found in Appendix A.2.

We first approximate f by f 2 using the Taylor polynomial of order n − 1 and prove the approximation error bound.

Note that even when f is non-differentiable (only first order weak derivative exists), the Taylor polynomial of order 0 at x = m N can still be used, which takes the form of P m = f ( m N ).

Then we approximate f 2 by a ReLU network that is denoted as f with bounded error.

After that, we present the quantized ReLU network that implements f and the complexity.

The discussion above focuses on nonlinear quantization which is a more general case compared to linear quantization.

For linear quantization, which strictly determines the available weight values once λ is given, we can use the same proof for nonlinear quantization except for a different subnetwork for weight approximation with width t and depth t log λ +1.

Here we give the theorem and the proof is included in Appendix A.3.

Theorem 2.

For any f ∈ F d,n , given weight maximum precision 1 λ , there is a ReLU network with fixed structure that can approximate f with any error ∈ (0, 1), such that (i) the depth is O (log (1/ )); (ii) the number of weights is O log (1/ ) + DISPLAYFORM2 ; (iii) the number of bits needed to store the network is O log(λ) log (1/ ) + log DISPLAYFORM3

The network complexity can be reduced if the network topology can be set according to a specific target function, i.e. function-dependent structure.

In this section, we provide an upper bound for function-dependent structure when d = 1 and n = 1, which is asymptotically better than that of a fixed structure.

Specifically, we first define an approximation to f (x) as f (x) that has special properties to match the peculiarity of quantized networks.

Then we use piecewise linear interpolation and "cached" functions BID16 to approximate f (x) by a ReLU network.

While simply using piecewise linear interpolation at the scale of can satisfy the error bound with O (1/ ) weights, the complexity can be reduced by first doing interpolation at a coarser scale and then fill the details in the intervals to make the error go down to .

By assigning a "cached" function to every interval depending on specific function and proper scaling, the number of weights is reduced to O log −1 (1/ ) 1/ when there is no constraint on weight values BID16 .The key difficulty in applying this approach to quantized ReLU networks is that the required linear interpolation at i T exactly where i = 1, 2, · · · , T is not feasible because of the constraint on weight selection.

To this end, we transform f (x) to f (x) such that the approximation error is bounded; the Lipschitz constant is preserved; f i T are reachable for the network under the constraints of weight selection without increasing the requirement on weight precision.

Then we can apply the interpolation and cached function method on f (x) and finally approximate f (x) with a quantized ReLU network.

Formally, we get the following proposition and the proof can be found in Appendix A.4.Proposition 4.

For any f ∈ F 1,1 , t ∈ Z + , and T ∈ Z + , there exists a function f (x) such that DISPLAYFORM0

With the help of Proposition 4 and the weight construction method described in Section 4.2, we are able to apply the interpolation and cached function approach.

Denoting the output of the network as f (x), we have |f (x)−f (x)| = |f (x)− f (x)|+| f (x)−f (x)| ≤ by choosing appropriate hyperparameters which are detailed in Appendix A.5 and the network complexity is obtained accordingly.

Theorem 3.

For any f ∈ F 1,1 , given λ distinct weights, there is a ReLU network with function-dependent structure that can approximate f with any error ∈ (0, 1), such that (i) the depth is O λ (log log (1/ )) 1 λ−1 + log (1/ ) ; (ii) the number of weights is O λ (log log (1/ )) 1 λ−1 +1 + (1/ ) (iii) the number of bits needed to store the network is O log λ λ (log log (1/ )) DISPLAYFORM0 Using the different weight construction approach as in the case of function-independent structure, we have the result for linear quantization: Theorem 4.

For any f ∈ F 1,1 , given weight maximum precision 1 λ , there is a ReLU network with function-dependent structure that can approximate f with any error ∈ (0, 1), such that (i) the depth is O (log (1/ )); (ii) the number of weights is O (1/ ); (iii) the number of bits needed to store the network is O (log(λ)/ ).

In this section, we first introduce the optimal bit-width problem and then show how a theoretical bound could potentially be used to estimate the optimal bit-width of a neural network.

Because of the natural need and desire of comparison with competitive approaches, most quantization techniques are evaluated on some popular reference networks, without modification of the network topology.

On the one hand, the advancement of lossless quantization almost stalls at a bit-width between two and six BID8 BID3 Sze et al., 2017; BID0 Su et al., 2018; BID6 .

A specific bit-width depends on the compactness of the reference network and the difficulty of the task.

On the other hand, the design space, especially the different combinations of topology and bit-width, is largely underexplored because of the complexity, resulting in sub-optimal results.

A recent work by Su et al. (2018) empirically validates the benefit of exploring flexible network topology during quantization.

That work adds a simple variable of network expanding ratio, and shows that a bit-width of four achieves the best cost-accuracy trade-off among limited options in {1, 2, 4, 8, 16, 32}. Some recent effort on using reinforcement learning to optimize the network hyper-parameters BID12 could potentially be used to address this issue.

But the current design space is still limited to a single variable per layer (such as the pruning ratio based on a reference network).

How to estimate an optimal bit-width for a target task without training could be an interesting research direction in the future.

The memory bound expression as derived in this paper helps us to determine whether there is an optimal λ that would lead to the lowest bound and most compact network (which can be translated to computation cost in a fully connected structure) for a given target function.

For example, by dropping the lower-order term and ignoring the rounding operator, our memory bound can be simplified as DISPLAYFORM0 where θ 1 is a constant determined by , n, and d. We can find an optimal λ that minimizes M (λ): DISPLAYFORM1 As is detailed in Appendix B, we prove that there exists one and only one local minimum (hence global minimum) in the range of [2, ∞)

whenever < 1 2 .

We also show that λ opt is determined by log 3n2 d / , which can be easily dominated by d. Based on such results, we quantitatively evaluate the derivative of M (λ), and based on which the optimal bit-width log(λ opt ) under various settings in FIG3 and FIG3 , respectively.

In FIG3 , we also mark the input dimension of a few image data sets.

It is apparent to see that the optimal bit width derived from M (λ) is dominated by d and lies between one and four for a wide range of input size.

This observation is consistent with most existing empirical research results, hence showing the potential power of our theoretical bound derivation.

Since the bounds are derived for fully connected networks and depend on the construction approach, the interesting proximity between log(λ opt ) and the empirical results cannot be viewed as a strict theoretical explanation.

Regardless, we show that the complexity bound may be a viable approach DISPLAYFORM2 is a positive monotonically increasing function and thus does not affect the trends too much.

Note that λ is the number of distinct weight values and thus log(λ) is the corresponding bit-width.

It can be seen that and n only affect log(λ opt ) when d is small (< 10 2 ).

We also mark the input dimension d of various image data set and their corresponding log(λ opt ).

It shows that the optimal bit-width increases very slowly with d.to understand the optimal bit-width problem, thus potentially accelerating the hyper-parameter optimization of deep neural networks.

We defer such a thorough investigation of the optimal bit-width or optimal hybrid bit-width configuration across the network to our future work.

In this section, we further discuss the bound of nonlinear quantization with a function-independent structure as the generality of nonlinear quantization.

The availability of unquantized functionindependent structures in literature also makes it an excellent reference for comparison.

Comparison with the Upper Bound: The quality of an upper bound lies on its tightness.

Compared with the most recent work on unquantized ReLU networks BID16 , where the upper bound on the number of weights to attain an approximation error is given by O log(1/ ) (1/ ) d n , our result for a quantized ReLU network is given by O λ log DISPLAYFORM0 , which translates to an increase by a factor of λ log 1 λ−1 (1/ ) .

Loosely speaking, this term reflects the loss of expressive power because of weight quantization, which decreases quickly as λ increases.

We also compare our bound with the lower bound of the number of weights needed to attain an error bound of to have a better understanding on the tightness of the bound.

We use the lower bound for unquantized ReLU networks from BID16 , as it is also a natural lower bound for quantized ReLU networks.

Under the same growth rate of depth, the lower bound is given by Ω(log −3 (1/ ) (1/ ) d/n ), while our upper bound is, within a polylog factor when λ is a constant, O(λ log DISPLAYFORM0 The comparison validates the good tightness of our upper bound.

The Upper Bound of Overhead: More importantly, the above comparison yields an upper bound on the possible overhead induced by quantization.

By comparing the expressions of two bounds while treating λ as a constant, we can show that, to attain the same approximation error bound , the number of weights needed by a quantized ReLU network is no more than O(log 5 (1/ )) times that needed by an unquantized ReLU network.

Note that this factor is of much lower order than the lower bound Ω(log −3 (1/ ) (1/ ) d/n ).

This little overhead introduced by weight quantization explains in part the empirical success on network compression and acceleration by quantization and also answers in part the questions as raised in Section 1.

Given the significant benefits of quantization in term of memory and computation efficiency, we anticipate that the use of quantization networks will continue to grow, especially on resource-limited platforms.

Future Work: There remain many other avenues for future investigation.

For example, although we derived the first upper bound of quantized neural networks, the lower bound is still missing.

If a tight lower bound of the network size is established, it could be combined with the upper bound to give a much better estimation of required resources and the optimal bit-width.

We believe the trends associated with the bounds can also be useful and deserve some further investigation.

For example, the trend may help hardware designers in their early stage of design exploration without the need of lengthy training.

While we assume a uniform bit-width across all layers, another area of research is to allow different bit-widths in different layers, which could achieve better efficiency and potentially provide theoretical justifications on the emerging trend of hybrid quantization BID17 DISPLAYFORM1 DISPLAYFORM2 where DISPLAYFORM3 is the i-th iterate of g(x).

Since g(x) can be implemented by a ReLU sub-network DISPLAYFORM4 •r (x) can be obtained by concatenating such implementation of g(x) for r times.

Now, to implement f r s (x) based on g•r (x), all we need are weights {2 −2 , 2 −4 , · · · , 2 −2(r−1) , 2 −2r }, which can be easily constructed with additional 2r layers and the weight 1 2 .

Note that a straightforward implementation will have to scale g •i (x) separately (multiply by different numbers of 1 2 ) before subtracting them from x because each g•i (x) have a different coefficient.

Then the width of the network will be Θ(r).

Here we use a "pre-scale" method to reduce the network width from Θ(r) to a constant.

The network constructed is shown in FIG6 .

The one-layer sub-network that implements g(x) and the one-layer sub-network that scales the input by 4 are denoted as B g and B m respectively.

Some units are copied to compensate the scaling caused by •i (x) are scaled by 2 2(r−i) respectively.

As a result, we obtain 2 2r x − r i=1 2 2(r−i) g •i (x) after the last B m .

Then it is scaled by 2 −2r in the later 2r layers to get f r s (x).

In this way, we make all g •i (x) sharing the same scaling link and a constant width can be achieved.

A.2 THE PROOF OF THEOREM 1 Theorem 1.

For any f ∈ F d,n , given λ distinct weights, there is a ReLU network with fixed structure that can approximate f with any error ∈ (0, 1), such that (i) the depth is DISPLAYFORM5 the number of bits needed to store the network is O λ log (λ) log DISPLAYFORM6 Proof.

The proof is composed of four steps.

We first approximate f by f 2 using the Taylor polynomial of order n − 1 and prove the approximation error bound.

Note that even when f is nondifferentiable (only first order weak derivative exist), the Taylor polynomial of order 0 at x = m N can still be used, which takes the form of P m = f ( m N ).

Then we approximate f 2 by a ReLU network that is denoted as f with bounded error.

After that, we present the quantized ReLU network that implements the network f and the complexity of the network.

We use a partition of unity on DISPLAYFORM7 , and h(x) is defined as follows: DISPLAYFORM8 where N is a constant and DISPLAYFORM9 Note that supp ψ m ⊂ {x : DISPLAYFORM10 For all m, we have the order n − 1 Taylor polynomial for the function f at x = m N as DISPLAYFORM11 To get a more realizable approximation for quantized networks, we define P m (x) = n−|n| .

Then we get an approximation to f using P m and ψ m as f 2 m∈{0,··· ,N } d ψ m P m .

Then the approximation error of f 2 is bounded by Equation (13).

DISPLAYFORM12 The second step follows ψ m (x) = 0 when x / ∈ suppψ m .

In the third step we turn the sum to multiplication, because for any x there are up to 2 d terms ψ m (x) that are not equal to zero.

The fourth step uses a Lagrange's form of the Taylor remainder.

The fifth step follows different round precision of β m,n in different order and the fact that the number of terms with order i is not greater than d i .We rewrite f 2 as DISPLAYFORM13 where DISPLAYFORM14 Note that β m,n is a constant and thus f 2 is a linear combination of at most d n (N + 1) d terms of f m,n (x).

Note that when d = 1, the number of terms should be n(N + 1) d instead; but for simplicity of presentation we loosely use the same expression as they are on the same order.

We define an approximation to f m,n (x) as f m,n (x).

The only difference between f m,n (x) and f m,n (x) is that all multiplication operations are approximated by × as discussed in Proposition 2.

Consider that if we construct our function × with | × (x, y) − xy| < × = 2 −2(r+1) , then DISPLAYFORM15 Applying Equation FORMULA0 to |f m,n (x) − f m,n (x)| repeatedly, we bound it to Equation FORMULA0 .

DISPLAYFORM16 Finally, we define our approximation to f (x) as f (x): DISPLAYFORM17 Using Equation FORMULA0 , we get the error bound of the approximation to f 2 (x) as in Equation FORMULA0 .

DISPLAYFORM18 The second line follows again the support property and statement (i) of Proposition 2.

The third line uses the bound |β m,n | ≤ 1.

The fourth line is obtained by inserting Equation (17).Then the final approximation error bound is as follows: DISPLAYFORM19 Using statement (ii) of Proposition 2 and choosing r as r = log(6N n (d+n−1)) 2 − 1, the approximation error turns to DISPLAYFORM20 Figure 3: A qunatized ReLU network that implements f (x).

The connections of all B fmn are the same.

Every connection from B N to other blocks has no greater than two weights.

DISPLAYFORM21 Therefore, for any f ∈ F d,n and ∈ (0, 1), there is a ReLU network f that approximate f with error bound if we choose DISPLAYFORM22 We now present the construction of the network for f (x).

If every f m,n (x) can be computed by a sub-network, then f (x) is simply a weighted sum of all outputs of f m,n (x).

By Proposition 3, we can implement the needed weights β m,n by choosing t = log .

As discussed in Proposition 3, we build a weight construction network B w in the way that all integral multiples of the minimal precision can be obtained.

Therefore, all mi N can be obtained in the same way as β m,n , except that we need to concatenate two weight construction sub-networks.

Now we analyze the complexity of the network.

The implementation of f (x) is shown in Figure 3 .

The function and size of blocks are listed in TAB2 .

Then we are able to obtain the complexity of the network.

While we can write the complexity of the network in an explicit expression, here we use the O notation for clarity.

Let N d , N w , N b be the depth, the number of weights, and the number of bits required respectively.

The weight construction blocks B w have the highest order of number of weights and we have N w = O λt DISPLAYFORM23 Inserting t = log A.3 THE PROOF OF THEOREM 2Theorem 2.

For any f ∈ F d,n , given weight maximum precision 1 λ , there is a ReLU network with fixed structure that can approximate f with any error ∈ (0, 1), such that (i) the depth is O (log (1/ )); (ii) the number of weights is O log (1/ ) + DISPLAYFORM24 ; (iii) the number of bits needed to store the network is O log(λ) log (1/ ) + log 2 (1/ ) (1/ ) DISPLAYFORM25 With λ distinct values, a linearly quantized network has a minimal resolution of 1 λ .

The proof for the approximability of linear quantization can be done in the same way as Theorem 1 except for a different sub-network for weight approximation.

We still construct W c in Proposition 3 first and any weight value from W c can be obtained by multiply at most t log λ weights.

Thus the width and depth of the weight approximation network will be t and t log λ + 1 respectively.

Updating the B w in TAB2 , we obtain the complexity accordingly.

DISPLAYFORM26 This contradicts the fact that f ( T by Equation (22) .

Before the intersection, if f (0) < f + (0), DISPLAYFORM27 , apply the same logic and we obtain 0 ≤ f DISPLAYFORM28 T .

This implies statement (iii) and concludes the proof.

Theorem 3.

For any f ∈ F 1,1 , given λ distinct weights, there is a ReLU network with function-dependent structure that can approximate f with any error ∈ (0, 1), such that (i) the depth is O λ (log log (1/ )) 1 λ−1 + log (1/ ) ; (ii) the number of weights is O λ (log log (1/ )) 1 λ−1 +1 + (1/ ) (iii) the number of bits needed to store the network is O log λ λ (log log (1/ )) 1 λ−1 +1 + (1/ ) .Proof.

We first transform f to f with Proposition 4.

Then we apply the interpolation and cached function method from [35] while using the weight construction method described in Proposition 3.

Denoting the output of the network as f (x), we have |f ( The approximation network is shown in FIG15 .

The sizes of blocks are given in TAB3 where f T is the uniform linear interpolation function of f with T − 1 breakpoints, f * is the sum of the selected cached functions, Φ(x) is a filtering function.

The inputs connections to B f * 1 and the connections inside B m have higher order to the number of weights than others.

Then the complexity can be obtained accordingly.

Theorem 4.

For any f ∈ F 1,1 , given weight maximum precision 1 λ , there is a ReLU network with function-dependent structure that can approximate f with any error ∈ (0, 1), such that (i) the

@highlight

This paper proves the universal  approximability of quantized ReLU neural networks and puts forward the complexity bound given arbitrary error.