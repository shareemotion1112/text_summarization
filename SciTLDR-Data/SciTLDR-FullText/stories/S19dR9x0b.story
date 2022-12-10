Recurrent neural networks have achieved excellent performance in many applications.

However, on portable devices with limited resources, the models are often too large to deploy.

For applications on the server with large scale concurrent requests, the latency during inference can also be very critical for costly computing resources.

In this work, we address these problems by quantizing the network, both weights and activations, into multiple binary codes {-1,+1}. We formulate the quantization as an optimization problem.

Under the key observation that once the quantization coefficients are fixed the binary codes can be derived efficiently by binary search tree, alternating minimization is then applied.

We test the quantization for two well-known RNNs, i.e., long short term memory (LSTM) and gated recurrent unit (GRU), on the language models.

Compared with the full-precision counter part, by 2-bit quantization we can achieve ~16x memory saving and  ~6x real inference acceleration on CPUs, with only a reasonable loss in the accuracy.

By 3-bit quantization, we can achieve almost no loss in the accuracy or even surpass the original model, with ~10.5x memory saving and ~3x real inference acceleration.

Both results beat the exiting quantization works with large margins.

We extend our alternating quantization to image classification tasks.

In both RNNs and feedforward neural networks, the method also achieves  excellent performance.

Recurrent neural networks (RNNs) are specific type of neural networks which are designed to model the sequence data.

In last decades, various RNN architectures have been proposed, such as LongShort-Term Memory (LSTM) BID9 and Gated Recurrent Units BID0 .

They have enabled the RNNs to achieve state-of-art performance in many applications, e.g., language models (Mikolov et al., 2010) , neural machine translation BID21 , automatic speech recognition BID5 , image captions BID23 , etc.

However, the models often build on high dimensional input/output,e.g., large vocabulary in language models, or very deep inner recurrent networks, making the models have too many parameters to deploy on portable devices with limited resources.

In addition, RNNs can only be executed sequentially with dependence on current hidden states.

This causes large latency during inference.

For applications in the server with large scale concurrent requests, e.g., on-line machine translation and speech recognition, large latency leads to limited requests processed per machine to meet the stringent response time requirements.

Thus much more costly computing resources are in demand for RNN based models.

To alleviate the above problems, several techniques can be employed, i.e., low rank approximation (Sainath et al., 2013; BID14 BID16 BID22 , sparsity BID19 BID7 , and quantization.

All of them are build on the redundancy of current networks and can be combined.

In this work, we mainly focus on quantization based methods.

More precisely, we are to quantize all parameters into multiple binary codes {−1, +1}.The idea of quantizing both weights and activations is firstly proposed by BID11 .

It has shown that even 1-bit binarization can achieve reasonably good performance in some visual classification tasks.

Compared with the full precision counterpart, binary weights reduce the memory by a factor of 32.

And the costly arithmetic operations between weights and activations can then be replaced by cheap XNOR and bitcount operations BID11 , which potentially leads to much acceleration.

Rastegari et al. (2016) further incorporate a real coefficient to compensate for the binarization error.

They apply the method to the challenging ImageNet dataset and achieve better performance than pure binarization in BID11 .

However, it is still of large gap compared with the full precision networks.

To bridge this gap, some recent works BID12 BID29 further employ quantization with more bits and achieve plausible performance.

Meanwhile, quite an amount of works, e.g., BID3 BID30 BID6 , quantize the weights only.

Although much memory saving can be achieved, the acceleration is very limited in modern computing devices (Rastegari et al., 2016) .Among all existing quantization works, most of them focus on convolutional neural networks (CNNs) while pay less attention to RNNs.

As mentioned earlier, the latter is also very demanding.

Recently, BID10 showed that binarized LSTM with preconditioned coefficients can achieve promising performance in some easy tasks such as predicting the next character.

However, for RNNs with large input/output, e.g., large vocabulary in language models, it is still very challenging for quantization.

Both works of BID12 and BID28 test the effectiveness of their multi-bit quantized RNNs to predict the next word.

Although using up to 4-bits, the results with quantization still have noticeable gap with those with full precision.

This motivates us to find a better method to quantize RNNs.

The main contribution of this work is as follows:(a) We formulate the multi-bit quantization as an optimization problem.

The binary codes {−1, +1} are learned instead of rule-based.

For the first time, we observe that the codes can be optimally derived by the binary search tree once the coefficients are knowns in advance, see, e.g., Algorithm 1.

Thus the whole optimization is eased by removing the discrete unknowns, which are very difficult to handle.

(b) We propose to use alternating minimization to tackle the quantization problem.

By separating the binary codes and real coefficients into two parts, we can solve the subproblem efficiently when one part is fixed.

With proper initialization, we only need two alternating cycles to get high precision approximation, which is effective enough to even quantize the activations on-line.

(c) We systematically evaluate the effectiveness of our alternating quantization on language models.

Two well-known RNN structures, i.e., LSTM and GRU, are tested with different quantization bits.

Compared with the full-precision counterpart, by 2-bit quantization we can achieve ∼16× memory saving and ∼6× real inference acceleration on CPUs, with a reasonable loss on the accuracy.

By 3-bit quantization, we can achieve almost no loss in accuracy or even surpass the original model with ∼10.5× memory saving and ∼3× real inference acceleration.

Both results beat the exiting quantization works with large margins.

To illustrate that our alternating quantization is very general to extend, we apply it to image classification tasks.

In both RNNs and feedforward neural networks, the technique still achieves very plausible performance.

Before introducing our proposed multi-bit quantization, we first summarize existing works as follows:(a) Uniform quantization method (Rastegari et al., 2016; BID12 firstly scales its value in the range x ∈ [−1, 1].

Then it adopts the following k-bit quantization: DISPLAYFORM0 after which the method scales back to the original range.

Such quantization is rule based thus is very easy to implement.

The intrinsic benefit is that when computing inner product Figure 1 : Illustration of the optimal 2-bit quantization when α 1 and α 2 (α 1 ≥ α 2 ) are known in advance.

The values are quantized into −α 1 − α 2 , −α 1 + α 2 , α 1 − α 2 , and α 1 + α 2 , respectively.

And the partition intervals are optimally separated by the middle points of adjacent quantization codes, i.e., −α 1 , 0, and α 1 , correspondingly.of two quantized vectors, it can employ cheap bit shift and count operations to replace costly multiplications and additions operations.

However, the method can be far from optimum when quantizing non-uniform data, which is believed to be the trained weights and activations of deep neural network BID28 .

(b) Balanced quantization BID28 alleviates the drawbacks of the uniform quantization by firstly equalizing the data.

The method constructs 2 k intervals which contain roughly the same percentage of the data.

Then it linearly maps the center of each interval to the corresponding quantization code in (1).

Although sounding more reasonable than the uniform one, the affine transform on the centers can still be suboptimal.

In addition, there is no guarantee that the evenly spaced partition is more suitable if compared with the non-evenly spaced partition for a specific data distribution.

(c) Greedy approximation BID6 instead tries to learn the quantization by tackling the following problem: DISPLAYFORM1 For k = 1, the above problem has a closed-form solution (Rastegari et al., 2016) .

Greedy approximation extends to k-bit (k > 1) quantization by sequentially minimizing the residue.

That is min αi,bi DISPLAYFORM2 Then the optimal solution is given as DISPLAYFORM3 Greedy approximation is very efficient to implement in modern computing devices.

Although not able to reach a high precision solution, the formulation of minimizing quantization error is very promising.

(d) Refined greedy approximation BID6 extends to further decrease the quantization error.

In the j-th iteration after minimizing problem (3), the method adds one extra step to refine all computed DISPLAYFORM4 with the least squares solution: DISPLAYFORM5 In experiments of quantizing the weights of CNN, the refined approximation is verified to be better than the original greedy one.

However, as we will show later, the refined method is still far from satisfactory for quantization accuracy.

Besides the general multi-bit quantization as summarized above, propose ternary quantization by extending 1-bit binarization with one more feasible state, 0.

It does quantization by tackling min α,t w − αt 2 2 with t ∈ {−1, 0, +1} n .

However, no efficient algorithm is proposed in .

They instead empirically set the entries w with absolute scales less than 0.7/n w 1 to 0 and binarize the left entries as (4).

In fact, ternary quantization is a special case of the 2-bit quantization in (2), with an additional constraint that α 1 = α 2 .

When the binary codes are fixed, the optimal coefficient α 1 (or α 2 ) can be derived by least squares solution similar to (5).

Algorithm 1: Binary Search Tree (BST) to determine to optimal code BST(w, v) {w is the real value to be quantized} {v is the vector of quantization codes in ascending order} DISPLAYFORM6 In parallel to the binarized quantization discussed here, vector quantization is applied to compress the weights for feedforward neural networks BID4 BID8 .

Different from ours where all weights are directly constraint to {−1, +1}, vector quantization learns a small codebook by applying k-means clustering to the weights or conducting product quantization.

The weights are then reconstructed by indexing the codebook.

It has been shown that by such a technique, the number of parameters can be reduced by an order of magnitude with limited accuracy loss BID4 .

It is possible that the multi-bit quantized binary weight can be further compressed by using the product quantization.

Now we introduce our quantization method.

We tackle the same minimization problem as (2).

For simplicity, we firstly consider the problem with k = 2.

Suppose that α 1 and α 2 are known in advance with α 1 ≥ α 2 ≥ 0, then the quantization codes are restricted to DISPLAYFORM0 For any entry w of w in problem (2), its quantization code is determined by the least distance to all codes.

Consequently, we can partition the number axis into 4 intervals.

And each interval corresponds to one particular quantization code.

The common point of two adjacent intervals then becomes the middle point of the two quantization codes, i.e., −α 1 , 0, and α 1 .

Fig. 1 gives an illustration.

For the general k-bit quantization, suppose that {α i } k i=1 are known and we have all possible codes in ascending order, i.e., DISPLAYFORM1 Similarly, we can partition the number axis into 2 k intervals, in which the boundaries are determined by the centers of two adjacent codes in DISPLAYFORM2 .

However, directly comparing per entry with all the boundaries needs 2 k comparisons, which is very inefficient.

Instead, we can make use of the ascending property in v. Hierarchically, we partition the codes of v evenly into two ordered sub-sets, i.e., By recursively evenly partition the ordered feasible codes, we can then efficiently determine the Algorithm 2: Alternating Multi-bit Quantization Require :Full precision weight w ∈ R n , number of bits k, total iterations T Ensure : DISPLAYFORM3 as FORMULA3 Construct v of all feasible codes in accending order DISPLAYFORM4 as Algorithm 1.

6 end optimal code for per entry by only k comparisons.

The whole procedure is in fact a binary search tree.

We summarize it in Algorithm 1.

Note that once getting the quantization code, it is straightforward to map to the binary code b. Also, by maintaining a mask vector with the same size as w to indicate the partitions, we could operate BST for all entries simultaneously.

To give a better illustration, we give a binary tree example for k = 2 in FIG0 .

Note that for k = 2, we can even derive the optimal codes by a closed form solution, i.e., b 1 = sign(w) and b 2 = sign(w − α 1 b 1 ) with α 1 ≥ α 2 ≥ 0.Under the above observation, let us reconsider the refined greedy approximation BID6 introduced in Section 2.

After modification on the computed DISPLAYFORM5 are no longer optimal while the method keeps all of them fixed.

To improve the refined greedy approximation, alternating minimizing DISPLAYFORM6 In real experiments, we find that by greedy initialization as (4), only two alternating cycles is good enough to find high precision quantization.

For better illustration, we summarize our alternating minimization in Algorithm 2.

For updating DISPLAYFORM7 , we need 2k 2 n binary operations and kn non-binary operations.

Combining kn non-binary operations to determine the binary code, for total T alternating cycles, we thus need 2T k 2 n binary operations and 2(T + 1)kn non-binary operations to quantize w ∈ R n into k-bit, with the extra 2kn corresponding to greedy initialization.

Implementation.

We firstly introduce the implementation details for quantizing RNN.

For simplicity, we consider the one layer LSTM for language model.

The goal is to predict the next word indexed by t in a sequence of one-hot word tokens (y * 1 , . . .

, y * N ) as follows: DISPLAYFORM0 where σ represents the activation function.

In the above formulation, the multiplication between the weight matrices and the vectors x t and h t occupy most of the computation.

This is also where we apply quantization to.

For the weight matrices, We do not apply quantization on the full but rather row by row.

During the matrix vector product, we can firstly execute the binary multiplication.

Then element-wisely multiply the obtained binary vector with the high precision scaling coefficients.

Thus little extra computation results while much more freedom is brought to better approximate the weights.

We give an illustration on the left part of FIG2 .

Due to one-hot word tokens, x t corresponds to one specific row in the quantized W e .

It needs no more quantization.

Different from the weight matrices, h t depends on the input, which needs to be quantized on-line during inference.

For consistent notation with existing work, e.g., BID12 BID28 , we also call quantizing on h t as quantizing on activation.

For W ∈ R m×n and h t ∈ R n , the standard matrix-vector product needs 2mn operations.

For the quantized product between k w -bit W and k h -bit h t , we have 2k w k h mn + 4k 2 h n binary operations and 6k h n + 2k w k h m non-binary operations, where 6k h n corresponds to the cost of alternating approximation (T = 2) and 2k w k h m corresponds to the final product with coefficients.

As the binary multiplication operates in 1 bit, whereas the full precision multiplication operates in 32 bits, despite the feasible implementations, the acceleration can be 32× in theory.

For alternating quantization here, the overall theoretical acceleration is thus computed as γ = DISPLAYFORM1 Suppose that LSTM has hidden states n = 1024, then we have W h ∈ R 4096×1024 .

The acceleration ratio becomes roughly 7.5× for (k h , k w ) = (2, 2) and 3.5× for (k h , k w ) = (3, 3).

In addition to binary operations, the acceleration in real implementations can be largely affected by the size of the matrix, where much memory reduce can result in better utilizing in the limited faster cache.

We implement the binary multiplication kernel in CPUs.

Compared with the much optimized Intel Math Kernel Library (MKL) on full precision matrix vector multiplication, we can roughly achieve 6× for (k h , k w ) = (2, 2) and 3× for (k h , k w ) = (3, 3).

For more details, please refer to Appendix A.As indicated in the left part of FIG2 , the binary multiplication can be conducted sequentially by associativity.

Although the operation is suitable for parallel computing by synchronously conducting the multiplication, this needs extra effort for parallelization.

We instead concatenate the binary codes as shown in the right part of FIG2 .

Under such modification, we are able to make full use of the much optimized inner parallel matrix multiplication, which gives the possibility for further acceleration.

The final result is then obtained by adding all partitioned vectors together, which has little extra computation.

Training.

As firstly proposed by BID3 , during the training of quantized neural network, directly adding the moderately small gradients to quantized weights will result in no change on it.

So they maintain a full precision weight to accumulate the gradients then apply quantization in every mini-batch.

In fact, the whole procedure can be mathematically formulated as a bi-level optimization BID1 problem: DISPLAYFORM2 Denote the quantized weight asŵ = k i=1 α i b i .

In the forward propagation, we deriveŵ from the full precision w in the lower-level problem and apply it to the upper-level function f (·), i.e., RNN in this paper.

During the backward propagation, the derivative ∂f ∂ŵ is propagated back to w through the lower-level function.

Due to the discreteness of b i , it is very hard to model the implicit dependence ofŵ on w.

So we also adopt the "straight-through estimate" as BID3 , i.e., ∂f ∂w = ∂f ∂ŵ .

To compute the derivative on the quantized hidden state h t , the same trick is applied.

During the training, we find the same phenomenon as BID12 that some Table 1 : Measurement on the approximation of different quantization methods, e.g., Uniform BID12 , Balanced BID28 , Greedy BID6 , Refined BID6 , and our Alternating method, see Section 2.

We apply these methods to quantize the full precision pre-trained weight of LSTM on the PTB dataset.

The best values are in bold.

W-bits represents the number of weight bits and FP denotes full precision.

In this section, we conduct quantization experiments on language models.

The two most well-known recurrent neural networks, i.e., LSTM BID9 and GRU BID0 , are evaluated.

As they are to predict the next word, the performance is measured by perplexity per word (PPW) metric.

For all experiments, we initialize with the pre-trained model and using vanilla SGD.

The initial learning rate is set to 20.

Every epoch we evaluate on the validation dataset and record the best value.

When the validation error exceeds the best record, we decrease learning rate by a factor of 1.2.

Training is terminated once the learning rate less than 0.001 or reaching the maximum epochs, i.e., 80.

The gradient norm is clipped in the range [−0.25, 0.25] .

We unroll the network for 30 time steps and regularize it with the standard dropout (probability of dropping out units equals to 0.5) BID27 .

For simplicity of notation, we denote the methods using uniform, balanced, greedy, refined greedy, and our alternating quantization as Uniform, Balanced, Greedy, Refined, and Alternating, respectively.

Peen Tree Bank.

We first conduct experiments on the Peen Tree Bank (PTB) corpus BID20 , using the standard preprocessed splits with a 10K size vocabulary (Mikolov, 2012) .

The PTB dataset contains 929K training tokens, 73K validation tokens, and 82K test tokens.

For fair comparison with existing works, we also use LSTM and GRU with 1 hidden layer of size 300.

To have a glance at the approximation ability of different quantization methods as detailed in Section 2, we firstly conduct experiments by directly quantizing the trained full precision weight (neither quantization on activation nor retraining).

Results on LSTM and GRU are shown in TAB2 , respectively.

The left parts record the relative mean squared error of quantized weight matrices with full precision one.

We can see that our proposed Alternating can get much lower error across all varying bit.

We also measure the testing PPW for the quantized weight as shown in the right parts of Table 1 and 2.

The results are in consistent with the left part, where less errors result in lower testing PPW.

Note that Uniform and Balanced quantization are rule-based and not aim at minimizing the error.

Thus they can have much worse result by direct approximation.

We also repeat the experiment on other datasets.

For both LSTM and GRU, the results are very similar to here.

We then conduct experiments by quantizing both weights and activations.

We train with the batch size 20.

The final result is shown in TAB3 .

Besides comparing with the existing works, we also conduct experiment for Refined as a competitive baseline.

We do not include Greedy as it is already shown to be much inferior to the refined one, see, e.g., Table 1 and 2.

As TAB3 shows, our full precision model can attain lower PPW than the existing works.

However, when considering the gap between quantized model with the full precision one, our alternating quantized neural network is still far better than existing works, i.e., Uniform BID12 and Balanced BID28 .

Compared with Refined, our Alternating quantization can achieve compatible performance using 1-bit less quantization on weights or activations.

In other words, under the same tolerance of accuracy drop, Alternating executes faster and uses less memory than Refined.

We can see that our 3/3 weights/activations quantized LSTM can achieve even better performance than full precision one.

A possible explanation is due to the regularization introduced by quantization BID12 .WikiText-2 (Merity et al., 2017) is a dataset released recently as an alternative to PTB.

It contains 2088K training, 217K validation, and 245K test tokens, and has a vocabulary of 33K words, which is roughly 2 times larger in dataset size, and 3 times larger in vocabulary than PTB.

We train with one layer's hidden state of size 512 and set the batch size to 100.

The result is shown in TAB4 .

Similar to PTB, our Alternating can use 1-bit less quantization to attain compatible or even lower PPW than Refined.

Text8.

In order to determine whether Alternating remains effective with a larger dataset, we perform experiments on the Text8 corpus (Mikolov et al., 2014) .

Here we follow the same setting as BID26 .

The first 90M characters are used for training, the next 5M for validation, and the final 5M for testing, resulting in 15.3M training tokens, 848K validation tokens, and 855K test tokens.

We also preprocess the data by mapping all words which appear 10 or fewer times to the unknown token, resulting in a 42K size vocabulary.

We train LSTM and GRU with one hidden layer of size 1024 and set the batch size to 100.

The result is shown in TAB5 .

For LSTM on the left part, Alternating achieves excellent performance.

By only 2-bit quantization on weights and activations, it exceeds Refined with 3-bit.

The 2-bit result is even better than that reported in BID26 , where LSTM adding noising schemes for regularization can only attain 110.6 testing PPW.

For GRU on the right part, although Alternating is much better than Refined, the 3-bit quantization still has gap with full precision one.

We attribute that to the unified setting of hyper-parameters across all experiments.

With specifically tuned hyper-parameters on this dataset, one may make up for that gap.

Note that our alternating quantization is a general technique.

It is not only suitable for language models here.

For a comprehensive verification, we apply it to image classification tasks.

In both RNNs and feedforward neural networks, our alternating quantization also achieves the lowest testing error among all compared methods.

Due to space limitation, we deter the results to Appendix B.

In this work, we address the limitations of RNNs, i.e., large memory and high latency, by quantization.

We formulate the quantization by minimizing the approximation error.

Under the key observation that some parameters can be singled out when others fixed, a simple yet effective alternating method is proposed.

We apply it to quantize LSTM and GRU on language models.

By 2-bit weights and activations, we achieve only a reasonably accuracy loss compared with full precision one, with ∼16× reduction in memory and ∼6× real acceleration on CPUs.

By 3-bit quantization, we can attain compatible or even better result than the full precision one, with ∼10.5× reduction in memory and ∼3× real acceleration.

Both beat existing works with a large margin.

We also apply our alternating quantization to image classification tasks.

In both RNNs and feedforward neural networks, the method can still achieve very plausible performance.

In this section, we discuss the implementation of the binary multiplication kernel in CPUs.

The binary multiplication is divided into two steps: Entry-wise XNOR operation (corresponding to entry-wise product in the full precision multiplication) and bit count operation for accumulation (corresponding to compute the sum of all multiplied entries in the full precision multiplication).

We test it on Intel Xeon E5-2682 v4 @ 2.50 GHz CPU.

For the XNOR operation, we use the Single instruction, multiple data (SIMD) _mm256 _xor _ps, which can execute 256 bit simultaneously.

For the bit count operation, we use the function _popcnt64 (Note that this step can further be accelerated by the up-coming instruction _mm512 _popcnt_epi64 , which can execute 512 bits simultaneously.

Similarly, the XNOR operation can also be further accelerated by the up-coming _mm512 _xor _ps instruction to execute 512 bits simultaneously).

We compare with the much optimized Intel Math Kernel Library (MKL) on full precision matrix vector multiplication and execute all codes in the single-thread mode.

We conduct two scales of experiments: a matrix of size 4096 × 1024 multiplying a vector of size 1024 and a matrix of size 42000 × 1024 multiplying a vector of size 1024, which respectively correspond to the hidden state product W h h t−1 and the softmax layer W s h t for Text8 dataset during inference with batch size of 1 (See Eq. (6)).

The results are shown in TAB6 .

We can see that our alternating quantization step only accounts for a small portion of the total executing time, especially for the larger scale matrix vector multiplication.

Compared with the full precision one, the binary multiplication can roughly achieve 6× acceleration with 2-bit quantization and 3× acceleration with 3-bit quantization.

Note that this is only a simple test on CPU.

Our alternating quantization method can also be extended to GPU, ASIC, and FPGA.

Sequential MNIST.

As a simple illustration to show that our alternating quantization is not limited for texts, we conduct experiments on the sequential MNIST classification task BID2 .

The dataset consists of a training set of 60K and a test set of 10K 28 × 28 gray-scale images.

Here we divide the last 5000 training images for validation.

In every time, we sequentially use one row of the image as the input (28×1), which results in a total of 28 time steps.

We use 1 hidden layer's LSTM of size 128 and the same optimization hyper-parameters as the language models.

Besides the weights and activations, the inputs are quantized.

The testing error rates for 1-bit input, 2-bit weight, and 2-bit activation are shown in 7, where our alternating quantized method still achieves plausible performance in this task.

MLP on MNIST.

The alternating quantization proposed in this work is a general technique.

It is not only suitable for RNNs, but also for feed-forward neural networks.

As an example, we firstly conduct a classification task on MNIST and compare with existing work .

The method proposed in is intrinsically a greedy multi-bit quantization method.

For fair comparison, we follow the same setting.

We use the MLP consisting of 3 hidden layers of 4096 units and an L2-SVM output layer.

No convolution, preprocessing, data augmentation or pre-training is ) 1.25 % Refined BID6 1.22 % Alternating (ours) 1.13 % Table 9 : Testing error rate of CNN on CIFAR-10 with 2-bit weight and 1-bit activation.

Testing Error Rate Full Precision (reported in BID10 11.90 % XNOR-Net (1-bit weight & activation, reported in BID10 ) 12.62 % Refined BID6 12.08 % Alternating (ours) 11.70 % used.

We also use ADAM BID15 with an exponentially decaying learning rate and Batch Normalization BID13 with a batch size 100.

The testing error rates for 2-bit input, 2-bit weight, and 1-bit activation are shown in TAB8 .

Among all the compared multi-bit quantization methods, our alternating one achieves the lowest testing error.

CNN on CIFAR-10.

We then conduct experiments on CIFAR-10 and follow the same setting as BID10 .

That is, we use 45000 images for training, another 5000 for validation, and the remaining 10000 for testing.

The images are preprocessed with global contrast normalization and ZCA whitening.

We also use the VGG-like architecture (Simonyan & Zisserman, 2015) :(2 × 128 C3)−MP2−(2 × 256 C3)−MP2−(2 × 512 C3)−MP2−(2 × 1024 FC)−10 SVM where C3 is a 3 × 3 convolution layer, and MP2 is a 2 × 2 max-pooling layer.

Batch Normalization, with a mini-batch size of 50, and ADAM are used.

The maximum number of epochs is 200.

The learning rate starts at 0.02 and decays by a factor of 0.5 after every 30 epochs.

The testing error rates for 2-bit weight and 1-bit activation are shown in Table 9 , where our alternating method again achieves the lowest test error rate among all compared quantization methods.

@highlight

We propose a  new  quantization method and apply it to quantize RNNs for both compression and acceleration

@highlight

This paper proposes a multi-bit quantization method for recurrent neural networks.

@highlight

A technique for quantizing neural network weight matrices, and an alternating optimization procedure to estimate the set of k binary vectors and coefficients that best represent the original vector.