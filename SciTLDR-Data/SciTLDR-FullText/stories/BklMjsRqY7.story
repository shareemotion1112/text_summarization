Efforts to reduce the numerical precision of computations in deep learning training have yielded systems that aggressively quantize weights and activations, yet employ wide high-precision accumulators for partial sums in inner-product operations to preserve the quality of convergence.

The absence of any framework to analyze the precision requirements of partial sum accumulations results in conservative design choices.

This imposes an upper-bound on the reduction of complexity of multiply-accumulate units.

We present a statistical approach to analyze the impact of reduced accumulation precision on deep learning training.

Observing that a bad choice for accumulation precision results in loss of information that manifests itself as a reduction in variance in an ensemble of partial sums, we derive a set of equations that relate this variance to the length of accumulation and the minimum number of bits needed for accumulation.

We apply our analysis to three benchmark networks: CIFAR-10 ResNet 32, ImageNet ResNet 18 and ImageNet AlexNet.

In each case, with accumulation precision set in accordance with our proposed equations, the networks successfully converge to the single precision floating-point baseline.

We also show that reducing accumulation precision further degrades the quality of the trained network, proving that our equations produce tight bounds.

Overall this analysis enables precise tailoring of computation hardware to the application, yielding area- and power-optimal systems.

Over the past decade, deep learning techniques have been remarkably successful in a wide spectrum of applications through the use of very large and deep models trained using massive datasets.

This training process necessitates up to 100's of ExaOps of computation and Gigabytes of storage.

It is, however, well appreciated that a range of approximate computing techniques can be brought to bear to significantly reduce this computational complexity -and amongst them, exploiting reduced numerical precision during the training process is extremely effective and has already been widely deployed BID5 .There are several reasons why reduced precision deep learning has attracted the attention of both hardware and algorithms researchers.

First, it offers well defined and scalable hardware efficiency, as opposed to other complexity reduction techniques such as pruning BID7 a) , where handling sparse data is needed.

Indeed, parameter complexity scales linearly while multiplication hardware complexity scales quadratically with precision bit-width BID20 .

Thus, any advance towards truly binarized networks BID10 corresponds to potentially 30x -1000x complexity reduction in comparison to single precision floating-point hardware.

Second, the mathematics of reduced precision has direct ties with the statistical theory of quantization BID18 .

In the context of deep learning, this presents an opportunity for theoreticians to derive analytical trade-offs between model accuracy and numerical precision BID12 BID16 .

The terminology FPa/b denotes an FPU whose multiplier and adder use a and b bits, respectively.

Our work enables convergence in reduced precision accumulation and gains an extra 1.5× ∼ 2.2× area reduction.

Most ongoing efforts on reduced precision deep learning solely focus on quantizing representations and always assume wide accumulators, i.e., ideal summations.

The reason being reduced precision accumulation can result in severe training instability and accuracy degradation, as illustrated in FIG0 (a) for ResNet 18 (ImageNet) model training.

This is especially unfortunate, since the hardware complexity in reduced precision floating-point numbers (needed to represent small gradients during training) BID17 BID14 ) is dominated by the accumulator bit-width.

To illustrate this dominance we developed a model underpinned by the hardware synthesis of low-precision floating-point units (FPU) , that translates precision into area complexity of the FPU.

Comparisons obtained from this model are shown in FIG0 (b).

We observe that accumulating in high precision severely limits the hardware benefits of reduced precision computations.

This presents a new angle to the problem of reduced precision deep learning training which concerns determining suitable accumulation precision and forms the basis of our paper.

Our findings are that the accumulation precision requirements in deep learning training are nowhere near 32-b, and in fact could enable further complexity reduction of FPUs by a factor of 1.5 ∼ 2.2×.

Our work is concerned with establishing theoretical foundations for estimating the accumulation bit precision requirements in deep learning.

While this topic has never been addressed in the past, there are prior works in both deep learning and high performance computing communities that align well with ours.

Most early works on reduced precision deep learning consider fixed-point arithmetic or a variation of it BID5 .

However, when considering quantization of signals involving the backpropagation algorithm, finding a suitable fixed-point configuration becomes challenging due to a weak handle on the scalar dynamic range of the back-propagated signals.

Thus, hardware solutions have been sought, and, accordingly, other number formats were considered.

Flexpoint BID11 ) is a hybrid version between fixed-point and floating-point where scalars in a tensor are quantized to 16-b fixed-point but share 5-b of exponenent to adjust the dynamic range.

Similarly, WAGE BID19 augments Flexpoint with stochastic quantization and enables integer quantization.

All of these schemes focused on representation precision, but mostly used 32-bit accumulation.

Another option is to use reduced precision floating-point as was done in MPT BID14 , which reduces the precision of most signals to 16-b floating-point, but observes accuracy degradation when reducing the accumulation precision from 32-b.

Recently, BID17 quantize all representations to 8-b floating-point and experimentally find that the accumulation could be in 16-b with algorithmic contrivance, such as chunk-based accumulation, to enable convergence.

The issue of numerical errors in floating-point accumulation has been classically studied in the area of high performance computing.

BID15 were among the first to statistically estimate the effects of floating-point accumulation.

Assuming a stream of uniformly and exponentially distributed positive numbers, estimates for the mean square error of the floating-point accumulation were derived via quantization noise analysis.

Because such analyses are often intractable (due to the multiplicative nature of the noise), later works on numerical stability focus on worst case estimates of the accumulation error.

BID9 provide upper bounds on the error magnitude by counting and analyzing round-off errors.

Following this style of worst case analysis, BID0 provide bounds on the accumulation error for different summing algorithms, notably using chunk-based summations.

Different approaches to chunking are considered and their benefits are estimated.

It is to be noted that these analyses are often loose as they are agnostic to the application space.

To the best of our knowledge, a statistical analysis on the accumulation precision specifically tailored to deep learning training remains elusive.

Our contribution is both theoretical and practical.

We introduce the variance retention ratio (VRR) of a reduced precision accumulation in the context of the three deep learning dot products.

The VRR is used to assess the suitability, or lack thereof, of a precision configuration.

Our main result is the derivation of an actual formula for the VRR that allows us to determine accumulation bit-width for precise tailoring of computation hardware.

Experimentally, we verify the validity and tightness of our analysis across three benchmarking networks (CIFAR-10 ResNet 32, ImageNet ResNet 18 and ImageNet AlexNet).

The following basic floating-point definitions and notations are used in our work: Floating-point representation: A b-bit floating-point number a has a signed bit, e exponent bits, and m mantissa bits so that b = 1 + e + m. Its binary representation is (B s , B 1 , . . .

, B e , B 1 , . . .

, B m ) ∈ {0, 1} b and its value is equal to: DISPLAYFORM0 Floating-point operations: One of the most pervasive arithmetic functions used in deep learning is the dot product between two vectors which is the building block of the generalized matrix multiplication (GEMM).

A dot product is computed in a multiply-accumulate (MAC) fashion and thus requires two floating-point operations: multiplication and addition.

The realization of an ideal floating-point operation requires a certain bit growth at the output to avoid loss of information.

For instance, in a typical MAC operation, if c ← c + a × b where a is (1, e a , m a ) and b is (1, e b , m b ), then c should be (1, max(e a , e b ) + 2, m a + m b + 1 + ∆ E ), which depends on the bit-precision and the relative exponent difference of the operands ∆ E .

However, it is often more practical to pre-define the precision of c as (1, e c , m c ), which requires rounding immediately after computation.

Such rounding might cause an operand to be completely or partially truncated out of the addition, a phenomenon called "swamping" BID9 , which is the primary source of accumulation errors.

The second order statistics (variance) of signals are known to be of great importance in deep learning.

For instance, in prior works on weight initialization BID3 BID8 , it is customary to initialize random weights subject to a variance constraint designed so as to prevent vanishing or explosion of activations and gradients.

Thus, such variance engineering induces fine convergence of DNNs.

Importantly, in such analyses, the second order output statistics of a dot product are studied and expressed as a function of that of the accumulated terms, which are assumed to be independent and having similar variance.

A fundamental assumption is: V ar(s) = nV ar(p), where V ar(s) and V ar(p) are the variances of the sum and individual product terms, respectively, and n is the length of the dot product.

One intuition concerning accumulation with reduced precision is that, due to swamping, some product terms may vanish from the summation, resulting in a lower variance than expected: V ar(s) =ñV ar(p), whereñ < n.

This constitutes a violation of a key assumption and effectively leads to the re-emergence of the difficulties in training neural networks with improper weight initialization which often harms the convergence behavior BID3 BID8 .

To explain the poor convergence of our ResNet 18 run FIG0 (a)), we evaluate the behavior of accumulation variance across layers.

Specifically, we check the three dot products of a backpropagation iteration: the forward propagation (FWD), the backward propagation (BWD), and the gradient computation (GRAD), as illustrated in FIG1 .

Indeed, there is an abnormality in reduced precision GRAD as shown in Figure 3 .

It is also observed that the abnormality of variance is directly related to accumulation length.

From Figure 3 , the break point corresponds to the switch from the first to the second residual block.

The GRAD accumulation length in the former is much longer (4×) than the latter.

Thus, evidence points to the direction that for a given precision, there is an accumulation length for which the expected variance cannot be properly retained due to swamping.

Motivated by these observations, we propose to study the trade-offs among accumulation variance, length, and mantissa precision.

Before proceeding, it is important to note that our upcoming analysis differs, in style, from many works on reduced precision deep learning where it is common to model quantization effects as additive noise causing increased variance BID13 .

Our work does not contradict such findings, since prior arts have considered representation quantization whose effects are, by nature, different from intermediate roundings in partial sums.

We assume sufficient exponent precision throughout and treat reduced precision floating-point arithmetic as an unbiased form of approximate computing, as is customary.

Thus, our work focuses on associating second order statistics to mantissa precision.

We consider the accumulation of n terms {p i } n i=1 which correspond to the element-wise product terms in a dot product.

The goal is to compute the correct n th partial sum s n where DISPLAYFORM0 are statistically independent, zeromean, and have the same variance σ 2 p .

Thus, under ideal computation, the variance of s n (which is equal to its second moment) should be V ar(s n ) ideal = nσ 2 p .

However, due to swamping effects, the variance of s n under reduced precision is V ar(s n ) swamping = V ar(s n ) ideal .Let the product terms {p i } n i=1 and partial sum terms {s i } n i=1 have m p and m acc mantissa bits, respectively.

Our key contribution is a formula for the variance retention ratio V RR = V ar(sn)swamping V ar(sn)ideal .

The VRR, which is always less than or equal to unity, is a function of n, m p , and m acc only, which needs no simulations to be computed.

Furthermore, to preserve quality of computation under reduced precision, it is required that V RR → 1.

As it turns out, the VRR for a fixed precision is a curve with "knee" with respect to n, where a break point in accumulation length beyond which a certain mantissa precision is no longer suitable can be easily identified.

Accordingly, for a given accumulation length, the mantissa precision requirements can be readily estimated.

Before proceeding, we formally define swamping.

As illustrated in Figure 4 , the bit-shift of p i due to exponent difference may cause partial (e.g., stage 1-4) or full truncation of p i 's bits, called swamping.

We define (1) "full swamping" which occurs when |s i | > 2 macc |p i+1 |, and (2) "partial swamping" which occurs when 2 macc−mp |p i+1 | < |s i | ≤ 2 macc |p i+1 |.

These two swamping types will be fully considered in our analysis.

In the lemma below, we first present a formula for the VRR when only full swamping is considered.

Lemma 1.

The variance retention ratio of a length n accumulation using m acc mantissa bits, and when only considering full swamping, is given by: DISPLAYFORM0 where DISPLAYFORM1 i=2 q i +q n is a normalization constant, and Q denotes the elementary Q-function.

The proof is provided in Appendix A. A preliminary check is that a very large value of m acc in (1) causes all {q i } n−1 i=1 terms to vanish andq n to approach unity.

This makes V RR → 1 for high precision as expected.

On the other hand, if we assume m acc to be small, but let n → ∞, we get nq n → 0 because the Q-function term will approach 1 exponentially fast as opposed to the n term which is linear.

Furthermore, the terms inside the summation having a large i will vanish by the same argument, while the n term in the denominator will make the ratio decrease and we would expect V RR → 0.

This means that with limited precision, there is little hope to achieve a correct result when the accumulation length is very large.

Also, the rapid change in VRR from 0 to 1 indicates that VRR can be used to provide sharp decision boundary for accumulation precision.

The above result only considers full swamping and is thus incomplete.

Next we augment our analysis to take into account the effects of partial swamping.

The corresponding formula for the VRR is provided in the following theorem.

Theorem 1.

The variance retention ratio of a length n accumulation using m p and m acc mantissa bits for the input products and partial sum terms, respectively, is given by: DISPLAYFORM2 where DISPLAYFORM3 Published as a conference paper at ICLR 2019The proof is provided in Appendix B. Observe the dependence on m acc , m p , and n. Therefore, in what follows, we shall refer to the VRR in (2) as V RR(m acc , m p , n).

Once again, we verify the extremal behavior of our formula.

A very large value of m acc in (2) causes k 1 ≈ k 2 ≈ 0 and k 3 ≈ 1.

This makes V RR → 1 for high precision as expected.

In addition, assuming small m acc and letting n → ∞, we get nk 3 → 0 because k 3 decays exponentially fast due to the Q-function term.

By the same argument, q jr → 0 for all j r and q i → 0 for all but small values of i. Thus, the numerator will be small, while the denominator will increase linearly in n causing V RR → 0.

Thus, once more, we establish that with limited accumulation precision, there is little hope for a correct result.

Next we consider an accumulation that uses chunking.

In particular, assume n = n 1 × n 2 so that the accumulation is broken into n 2 chunks, each of length n 1 .

Thus, n 2 accumulations of length n 1 are performed and the n 2 intermediate results are added to obtain s n .

This simple technique is known to greatly improve the stability of sums BID0 .

The VRR can be used to theoretically explain such improvements.

For simplicity, we assume two-level chunking (as described above) and same mantissa precision m acc for both inter-chunk and intra-chunk accumulations.

Applying the above analysis, we may obtain a formula for the VRR as provided in the corollary below, which is proved in Appendix C. Corollary 1.

The variance retention ratio of an length n = n 1 × n 2 accumulation with chunking, where n 1 is the chunk size and n 2 is the number of chunks, using m p and m acc mantissa bits for the input products and partial sum terms, respectively, is given by: DISPLAYFORM0

It is common to encounter sparse operands in deep learning dot products.

Since addition of zero is an identity operation, the effective accumulation length is often less than as described by the network topology.

Indeed, for a given accumulation, supposedly of length n, if we can estimate the non-zero ratio (NZR) of its incoming product terms, then the effective accumulation length is N ZR × n.

Thus, when an accumulation is known to have sparse inputs with known NZR, a better estimate of the VRR is DISPLAYFORM0 Similarly, when considering the VRR with chunking, we may use knowledge of sparsity to obtain the effective intra-accumulation length as N ZR × n 1 .

This change is reflected both in the VRR of the intra-chunk accumulation and the input precision of the inter-chunk accumulation: DISPLAYFORM1 In practice, the NZR can be estimated by making several observations from baseline data.

Using an estimate of the NZR makes our analysis less conservative.

For a given accumulation setup, one may compute the VRR and observe how close it is from the ideal value of 1 in order to judge the suitability of the mantissa precision assignment.

It turns out that when measured as a function of accumulation length n for a fixed precision, the VRR has a breakdown region.

This breakdown region can very well be observed when considering what we define as the normalized exponential variance lost: DISPLAYFORM0 In FIG3 (a,b) we plot v(n) for different values of m acc when considering both normal accumulation and chunk-based accumulation with a chunk-size of 64.

The value of m p is set to 5-b, corresponding to the product of two numbers in (1,5,2) floating-point format BID17 .

We consider m acc to be suitable for a given n only if v(n) < 50.

The reason being, in all plots, the variance lost rapidly increases when v(n) > 50 and n increases.

On the other hand, when v(n) < 50 and n decreases, the variance lost quickly drops to zero.

This choice of a cut-off value is thus chosen purely based on the accumulation length and precision.

In addition, when performing chunk-based accumulation, the chunk size is a hyperparameter that, a priori, cannot be determined trivially.

BID0 identified an optimal chunk size minimizing the loose upper bound on the accumulation error they derived.

In practice, they did not find the accumulation error to be sensitive to the chunk-size.

Neither did BID17 who performed numerical simulations.

By sweeping the chunk size and observing the accumulation behavior on synthetic data, it was found that chunking significantly reduces accumulation error as long as the chunk size is not too small nor too large.

Using our analysis, we provide a theoretical justification.

FIG3 (c) shows the VRR for various accumulation setups, including chunking when the chunk size is sweeped.

For each case we see that chunking raises the VRR to a value close to unity.

Furthermore, the VRR curve in that regime is "flat", meaning that a specific value of chunk size does not matter as long as it is not too small nor too large.

One intuition is that a moderate chunk size prevents both inter-and intra-chunk accumulations to be as large as the original accumulation.

In our upcoming chunking experiments we use a chunk size of 64 as was done by BID17 .

Using the above analysis, we predict the mantissa precisions required by the three GEMM functions for training the following networks: ResNet 32 on the CIFAR-10 dataset, ResNet 18 and AlexNet on the ImageNet dataset.

Those benchmarks were chosen due to both their popularity and topologies which present large accumulation lengths, making them good candidates against which we can verify our work.

We use the same configurations as BID17 , in particular, we use 6-b of exponents in the accumulations, and quantize the intermediate tensors to (1,5,2) floating-point format and keep the final layer's precision in 16 bit.

The technique of loss scaling BID14 ) is used in order to limit underflows of activation gradients.

A single scaling factor of 1000 was used for all models tested.

In order to realize rounding of partial sums, we modify the CUDA code of the GEMM function (which, in principle, can be done using any framework).

In particular, we add a custom rounding function where the partial sum accumulation occurs.

Quantization of dot product inputs is handled similarly.

The predicted precisions for each network and layer/block are listed in TAB1 for the case of normal and chunk-based accumulation with a chunk size of 64.

Several insights are to be noted.• The required accumulation precision for CIFAR-10 ResNet 32 is in general lower than that of the ImageNet networks.

This is simply because, the network topology imposes shorter dot products.• Though, topologically, the convolutional layers in the two ImageNet networks are similar, the precision requirements do vary.

Specifically, the GRAD accumulation depends on the feature map is an ordered tuple of two values which correspond to the predicted mantissa precision of both normal and chunk-based accumulations, respectively.

The precision requirements of FWD and BWD are typically smaller than those of GRAD.

The latter needs the most precision for layers/blocks close to the input as the size of the feature maps is highest in the early stages.

The benefits of chunking are non linear but range from 1 to 6 bits.

AlexNet, and (d) final accuracy degradation with respect to the baseline as a function of precision perturbation (PP).

The solid and dashed lines correspond to the no chunking and chunking case, respectively.

Using our predicted precision assignment, the converged test error is close to the baseline (no more than 0.5% degradation) but increases significantly when the precision is further reduced.

dimension which is mostly dataset dependent, yet AlexNet requires less precision than ResNet 18.

This is because the measured sparsity of the operands was found to be much higher for AlexNet.• FIG3 suggests that chunking decreases the precision requirements significantly.

This is indeed observed in TAB1 , where we see that the benefits of chunking reach up to 6-b in certain accumulations, e.g., the GRAD acccumulation in the first ResBlock of ImageNet ResNet 18.Because our predicted precision assignment ensures the VRR of all GEMM accumulations to be close to unity, we expect reduced precision training to converge with close fidelity to the baseline.

Since our work focuses on accumulation precision, in our experiments, the baseline denotes accumulation in full precision.

For a fair comparison, all upcoming results use (1,5,2) representation precision.

Thus, the effects of reduced precision representation are not taken into account.

The goal of our experiments is to investigate both the validity and conservatism of our analysis.

In FIG4 , we plot the convergence curves when training with our predicted accumulation precision for a normal accumulation.

The runs corresponding to chunk-based accumulations were also performed but are omitted since the trend is similar.

Furthermore, we repeat all experiments with precision perturbation (PP), meaning a specific reduction in precision with respect to our prediction.

For instance, P P = 0 indicates our prediction while P P = −1 corresponds to a one bit reduction.

Finally, in order to better visualize how the accumulation precision affect convergence, we plot in FIG4 (d) the accuracy degradation as a function of precision perturbation for each of our three networks with both normal and chunk-based accumulations.

The following is to be noted:• When P P = 0, the converged accuracy always lies within 0.5% of the baseline, a strong indication of the validity of our analysis.

We use a 0.5% accuracy cut-off with respect to the baseline as it corresponds to an approximate error bound for neural networks obtained by changing the random numbers seed BID4 BID2 ).•

When P P < 0, a noticeable accuracy degradation is observed, most notably for ImageNet ResNet 18.

The converged accuracy is no longer within 0.5% of the baseline.

Furthermore, a clear trend observed is that the higher the perturbation, the worse the degradation.• ImageNet AlexNet is more robust to perturbation than the two ResNets.

While P P = −1 causes a degradation strictly > 0.5%, it is not much worse than the P P = 0 case.

This observation aligns with that from neural net quantization that Alexnet is robust due to its over-parameterized network structure BID21 .

But the trend of increasing degradation remains the same.• FIG4 (d) suggests that the effects of PP are more pronounced for a chunk-based accumulation.

Since the precision assignment itself is lower TAB1 , a specific precision perturbation corresponds to a relatively higher change.

For example, decreasing one bit from a 6-b assignment is more important than decreasing one bit from a 10-b assignment.

Further justification can be obtained by comparing FIG3 and 5 (b) where consecutive lines are less closely aligned for the chunk-based accumulation, indicating more sensitivity to precision perturbation.

Thus, overall, our predictions are adequate and close to the limits beyond which training becomes unstable.

These are very encouraging signs that our analysis is both valid and tight.

We have presented an analytical method to predict the precision required for partial sum accumulation in the three GEMM functions in deep learning training.

Our results prove that our method is able to accurately pinpoint the minimum precision needed for the convergence of benchmark networks to the full-precision baseline.

Our theoretical concepts are application agnostic, and an interesting extension would be to consider recurrent architectures such as LSTMs.

In particular, training via backpropagation in time could make the GRAD accumulation very large depending on the number of past time-steps used.

In such a case, our analysis is of great relevance to training precision optimization.

On the practical side, this analysis is a useful tool for hardware designers implementing reduced-precision FPUs, who in the past have resorted to computationally prohibitive brute-force emulations.

We believe this work addresses a critical missing link on the path to truly low-precision floating-point hardware for DNN training.

The analysis of stability of sums under reduced-precision floating-point accumulation is a classically difficult problem.

Indeed, statistically characterizing recursive rounding effects is often mathematically intractable.

Therefore, most prior works have considered worst-case analyses and provided loose bounds on accuracy of computation as a function of precision BID0 ).

In contrast, the results presented in our paper were found to be tight, however they necessitate a handful of assumptions for mathematical tractability.

These assumptions are listed and discussed hereafter, and the proofs of the theoretical results presented in the main text follow in this supplementary section.

Assumption 1: The product terms {p i } n i=1 are statistically independent, zero-mean, and have the same variance σ 2 p .

This assumption, which was mentioned in the main text, is a standard one in works where the issue of variance in deep learning is studied BID8 .

Note that as a result we have V ar( DISPLAYFORM0 Assumption 2: Computation in reduced precision floating-point arithmetic is unbiased with respect to the baseline.

This assumption is also standard in works studying quantization noise and effects BID16 .

An important implication is that V ar(s n ) swamping = E swamping s DISPLAYFORM1 Assumption 3: The accumulation is monotonic in the iterations leading to a full swamping event.

This assumption means that we shall focus on a typical scenario where the partial sums {s i } n i=1 grow in magnitude while product terms {p i } n i=1 are of the same order.

In other words, we do not consider catastrophic events where full swamping occurs unexpectedly (the probability of such event is small in any case).

We consider a partial sum s i that experiences full swamping in reduced precision accumulation to be statistically independent from prior partial sums s i for i <

i.

are statistically dependent as they are computed in a recursive manner.

Our assumption is that should swamping noise be so significant to cause full swamping, then the recursive dependence on prior partial sums is broken.

Assumption 5: Once a full swamping event occurs, the computation of partial sum accumulation is halted.

It is possible, but unlikely, that the computation might recover from swamping.

A partial recovery of the computation is also possible but causes negligible effects on the final result.

Thus, such scenarios are neglected.

Assumptions 3, 4, and 5 will be particularly useful for mathematical tractability in the proof of Lemma 1.Assumption 6: The bits of the mantissa representation of partial sums {s i } n i=1 and product terms {p i } n i=1 are equally likely to be zero or one.

This is yet again a standard assumption in quantization noise analysis which will be particularly useful in the proof of Theorem 1.A PROOF OF LEMMA 1In order to compute the VRR, we first need to compute V ar(s n ) during swamping, i.e., V ar(s n ) swamping = E swamping s 2 n , where the equality holds by application of Assumptions 1 and 2.

To do so, we rely on the Law of Total Expectation.

Indeed, assume that A is the set of events that describe all manners in which the accumulation s n experiencing swamping can occur, and let P (A) be the probability of event A ∈ A. Hence, by the Law of Total Expectation, we have that DISPLAYFORM0 It is in fact a difficult task to enumerate and describe all events in A. Thus we consider a reduced set of eventsÂ ⊂ A which is representative enough of the manners in which the accumulation occurs, yet tractable so that it can be used as a surrogate to A in (7).

We shall form the setÂ by application of Assumption 3, 4, and 5 as we proceed.

We consider a scenario where the first occurrence of full swamping is at iteration i of the summation for i = 2 . . .

n − 1.

This happens if: DISPLAYFORM1 Instead of looking at the actual absolute value of an incoming product term, we replace it by its typical value of σ p .

Furthermore, we simplify the condition of no swamping prior to iteration i by only considering the accumulated sum at the previous iteration.

This is due to Assumption 3 which allows us toconsider a simplified scenario where the accumulation is monotonic in the iterations leading to full swamping.

Hence, our simplified condition for the first occurrence of full swamping at iteration i is given by: DISPLAYFORM2 As iteration i corresponds to a full swamping event, we may invoke Assumption 4 and treat each of the two inequalities above independently.

Finally, we invoke Assumption 5: since full swamping happens at itertion i, then the result of the accumulation is s n = s i since the computation in the following iterations is halted.

Thus, the event setÂ we construct for our analysis consists of the mutually exclusive events DISPLAYFORM3 i=2 where A i is the event that full swamping occurs for the first time at iteration i under the above assumptions.

The condition for event i to happen is given by (8).

By Central Limit Theorem, which we use by virtue of the s i being a summation of independent, indentically distributed product terms, we have that s i ∼ N (0, iσ 2 p ), so that: DISPLAYFORM4 Furthermore, by Assumption 5 we have: DISPLAYFORM5 We also add to our space of events, the event A n where no full swamping occurs over the course of the accumulation.

This event happens if |s n | < 2 macc σ p and thus has probability P (A n ) = 1 − 2Q 2 macc √ n =q n .

Since this event corresponds to the ideal scenario, we have E s DISPLAYFORM6 Thus, under the above conditions we have: DISPLAYFORM7 where k = n−1 i=2 q i +q n is a normalization constant needed asÂ does not describe the full probability space.

Consequently we obtain (1) as formula for the VRR completing the proof for Lemma 1.

First, we do not change the description of the events {A i } n−1 i=2 above.

Notably, by Assumption 5, once full swamping occurs, i.e., |s i | > 2 macc σ p , the computation is stuck and stops.

The probability of this event is P (A i ) as above.

However, to account for partial swamping, we alter the result of E s 2 n A i .

Indeed, partial swamping causes additional loss of variance.

When the input product terms have m p bits of mantissa, then before event A i can occur, the computation should go through each of the m p stages described in Figure 4 .

We again use Assumption 3 and consider a typical scenario for each of the m p stages of partial swamping preceding a full swamping event whereby the accumulation is monotonic and the magnitude of the incoming product term is close to its typical value σ p .

Under this assumption, stage j is expected to happen for the following number of iterations: DISPLAYFORM0 for j = 1 . . .

m p .

At stage j, j least significant bits in the representation of the incoming product term are truncated (swamped).

The variance lost because of this truncation, which we call fractional variance loss E[f 2 j ], can be computed by assuming the truncated bits are equally likely to be 0 or 1 (Assumption 6), so that: DISPLAYFORM1 Hence, the total fractional variance lost before the occurrence of even DISPLAYFORM2 .

Thus, we update the value of variance conditioned on A i as follows: DISPLAYFORM3 where we used the operator (x) + = x if x > 0 0 otherwise in order to guarantee that the variance is positive.

Effectively, we neglect the events A i where i is so small that the variance retained is less than the variance lost due to partial swamping.

In other words, an event whereby full swamping occurs very early in the accumulation is considered to have zero probability and we replace P (A i ) in (9) by: DISPLAYFORM4 where q i = 2Q In addition, some boundary conditions need to be accounted for.

These include the cases when no full swamping happens before the accumulation is complete but partial swamping does happen.

We again consider a typical scenario as above and append our event set A with m p − 1 boundary events A jr mp jr=2, where the event A jr corresponds to the case where the computation has gone through stage j r − 1 of partial swamping but has not reached stage j r yet.

The condition for this event is σ p 2 macc−mp+jr−1 < |s n | < σ p 2 macc−mp+jr and occurs typically for up to N jr−1 iterations.

The total fractional variance lost is: Finally, the event A n is updated and corresponds to the case where neither partial nor full swamping occurs.

The condition for this event is |s n | < 2 macc−mp+1 and has a probability P (A n ) = 1 − DISPLAYFORM5 Putting things together, we use the law of total expectation as in FORMULA13 where k = k 1 + k 2 + k 3 , k 1 = n−1 i=2 P (A i ), k 2 = mp jr=2 P (A jr ), and k 3 = P (A n ).

Hence, the formula for the VRR in (2) in the theorem follows and this concludes the proof.

Applying the above analysis, we may compute the variance of the intermediate results as σ 2 p n 1 V RR(m acc , m p , n 1 ).

To compute the variance of the final result, first note that the mantissa precision of the incoming terms to the inter-chunk accumulation (the results from the intra-chunk accumulation) is min (m acc , m p + log 2 (n 1 )).

The reason being that since the intra-chunk accumulation uses m acc mantissa bits, the mantissa cannot grow beyond m acc due to the rounding nature of the floating-point accumulation.

However, if m acc is large enough and n 1 is small enough, it is most likely that the mantissa has not grown to the maximum.

Assuming accumulation of terms having statistically similar absolute value as was done for the VRR analysis, then the bit growth of the mantissa is logarithmic in n 1 and starts at m p .Hence, the variance of the computed result s n when chunking is used is: V ar(s n ) chunking = σ 2 p n 1 V RR(m acc , m p , n 1 ) × n 2 V RR (m acc , min (m acc , m p + log 2 (n 1 )) , n 2 )and hence the VRR with chunking can be computed using (3) in Corollary 1.

This completes the proof.

@highlight

We present an analytical framework to determine accumulation bit-width requirements in all three deep learning training GEMMs and verify the validity and tightness of our method via benchmarking experiments.

@highlight

The authors propose an analytical method to predict the number of mantissa bits needed for partial summations for convolutional and fully connected layers

@highlight

The authors conduct a thorough analysis of the numeric precision required for the accumulation operations in neural network training and show the theoretical impact of reducing number of bits in the floating point accumulator.