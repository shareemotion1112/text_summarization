Weight pruning has been introduced as an efficient model compression technique.

Even though pruning removes significant amount of weights in a network, memory requirement reduction was limited since conventional sparse matrix formats require significant amount of memory to store index-related information.

Moreover, computations associated with such sparse matrix formats are slow because sequential sparse matrix decoding process does not utilize highly parallel computing systems efficiently.

As an attempt to compress index information while keeping the decoding process parallelizable, Viterbi-based pruning was suggested.

Decoding non-zero weights, however, is still sequential in Viterbi-based pruning.

In this paper, we propose a new sparse matrix format in order to enable a highly parallel decoding process of the entire sparse matrix.

The proposed sparse matrix is constructed by combining pruning and weight quantization.

For the latest RNN models on PTB and WikiText-2 corpus, LSTM parameter storage requirement is compressed 19x using the proposed sparse matrix format compared to the baseline model.

Compressed weight and indices can be reconstructed into a dense matrix fast using Viterbi encoders.

Simulation results show that the proposed scheme can feed parameters to processing elements 20 % to 106 % faster than the case where the dense matrix values directly come from DRAM.

Deep neural networks (DNNs) require significant amounts of memory and computation as the number of training data and the complexity of task increases BID0 .

To reduce the memory burden, pruning and quantization have been actively studied.

Pruning removes redundant connections of DNNs without accuracy degradation BID6 .

The pruned results are usually stored in a sparse matrix format such as compressed sparse row (CSR) format or compressed sparse column (CSC) format, which consists of non-zero values and indices that represent the location of non-zeros.

In the sparse matrix formats, the memory requirement for the indices is not negligible.

Viterbi-based pruning BID14 significantly reduces the memory footprint of sparse matrix format by compressing the indices of sparse matrices using the Viterbi algorithm BID3 .

Although Viterbi-based pruning compresses the index component considerably, weight compression can be further improved in two directions.

First, the non-zero values in the sparse matrix can be compressed with quantization.

Second, sparse-to-dense matrix conversion in Viterbi-based pruning is relatively slow because assigning non-zero values to the corresponding indices requires sequential processes while indices can be reconstructed in parallel using a Viterbi Decompressor (VD).Various quantization techniques can be applied to compress the non-zero values, but they still cannot reconstruct the dense weight matrix quickly because it takes time to locate non-zero values to the corresponding locations in the dense matrix.

These open questions motivate us to find a non-zero value compression method, which also allows parallel sparse-to-dense matrix construction.

The contribution of this paper is as follows.(a) To reduce the memory footprint of neural networks further, we propose to combine the Viterbibased pruning BID14 ) with a novel weight-encoding scheme, which also uses the Viterbi-based approach to encode the quantized non-zero values.

(b) We suggest two main properties of the weight matrix that increase the probability of finding "good" Viterbi encoded weights.

First, the weight matrix with equal composition ratio of '0' and '1' for each bit is desired.

Second, using the pruned parameters as "Don't Care" terms increases the probability of finding desired Viterbi weight encoding.

(c) We demonstrate that the proposed method can be applied to Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) with various sizes and depths.

(d) We show that using the same Viterbi-based approach to compress both indices and non-zero values allows us to build a highly parallel sparse-to-dense reconstruction architecture.

Using a custom cycle-simulator, we demonstrate that the reconstruction can be done fast.

DNNs have been growing bigger and deeper to solve complex nonlinear tasks.

However, BID2 showed that most of the parameters in neural networks are redundant.

To reduce the redundancy and minimize memory and computation overhead, several weight reduction methods have been suggested.

Recently, magnitude-based pruning methods became popular due to its computational efficiency BID6 .

Magnitude-based pruning methods remove weights according to weight magnitude only and retrain the pruned network to recover from accuracy loss.

The method is scalable to large and deep neural networks because of its low computation overhead.

BID6 showed 9×-13× pruning rate on AlexNet and VGG-16 networks without accuracy loss on ImageNet dataset.

Although the compression rate was high, reduction of actual memory requirement was not as high as the compression rate because conventional sparse matrix formats, such as CSR and CSC, must use large portion of memory to store the indices of surviving weights.

BID14 succeeded in reducing the amount of index-related information using a Viterbi-algorithm based pruning method and corresponding custom sparse matrix format.

BID14 demonstrated 38.1% memory reduction compared to BID6 with no accuracy loss.

The memory reduction was limited, however, due to uncompressed non-zero values.

Several weight quantization methods were also suggested to compress the parameters of neural networks.

BID1 BID15 ; BID20 demonstrated that reducing the weights to binary or ternary was possible, but the accuracy loss of the binary neural networks was significant.

BID25 reduced the bit resolution of weights to binary, activations to 2 bits and gradients to 6 bits with 9.8 % top-1 accuracy loss on AlexNet for ImageNet task.

BID5 demonstrated a binary-weight AlexNet with 2.0% top-1 accuracy loss, achieving ∼10× compression rate.

BID22 showed that RNNs can also be quantized to reduce the memory footprint.

By quantizing the weight values to 3 bits with proposed method, the memory footprint of RNN models were reduced ∼10.5× with negligible performance degradation.

BID8 suggested to combine pruning with weight quantization to achieve higher compression rate.

The results showed 35× increase in compression rate on AlexNet.

However, the reduction was limited since the memory requirement of index-related information was only slightly improved with Huffman coding.

Although several magnitude-based pruning methods showed high compression rate, computation time did not improve much, because it takes time to decode the sparse matrix formats that describe irregular weight indices of pruned networks.

BID7 suggested to use dedicated hardware, custom sparse matrix formats, and dedicated pruning methods to accelerate the computation even after pruning.

BID10 ; tried to accelerate the computation by limiting the irregularity of weight indices.

By pruning neurons or feature maps, pruned weight matrices could maintain the dense format.

These approaches successfully reduced the number of computation of neural networks, but the compression rate was limited due to additional pruning conditions.

Although BID14 could use the Viterbi encoder to construct the index matrix fast, the process of pairing the non-zero weight values with the corresponding indices is still sequential, and thus relatively slow.3 WEIGHT PRUNING AND QUANTIZATION USING DOUBLE-VITERBI APPROACH Figure 1 illustrates the flowchart of the proposed compression method.

Viterbi-based pruning BID14 ) is applied first, and the pruned matrix is quantized using alternating multi-bit quantization method BID22 .

Quantized binary code matrices are then encoded using the Viterbi-based approach, which is similar to the one used in pruning.

Figure 1: Flowchart of Double Viterbi compression.

W is the weight of an original network and W P , W P Q , andŴ P Q represent the compressed weights after each process.

M ∈ {0, 1} is an index matrix which indicates whether each weight is pruned or not, and means element-wise multiplication.

DISPLAYFORM0 ∈ {−1, +1} are constants and binary weights generated by quantization.

DISPLAYFORM1 ∈ {−1, +1} is binary weights encoded by the Viterbi algorithm.

As the first step of the proposed weight encoding scheme, we compress the indices of the non-zero values in sparse weight matrix using the Viterbi-based pruning (Figure 1 ) BID14 .

In this scheme, Viterbi algorithm is used to select a pruned index matrix which minimizes the accuracy degradation among many candidates which a Viterbi decompressor can generate.

While the memory footprint of the index portion is significantly reduced by the Viterbi-based pruning, the remaining non-zero values after pruning still require non-negligible memory when high-precision bits are used.

Hence, quantization of the non-zero values is required for further reduction of the memory requirement.

Appendix A.1 explains the Viterbi-based pruning in detail.

After Viterbi-based pruning is finished, the alternating multi-bit quantization BID22 ) is applied to the sparse matrix ( Figure 1 ).

As suggested in BID22 , real-valued non-zero weights are quantized into multiple binary codes DISPLAYFORM0 In addition to the high compression capabilities, another important reason we chose the alternating quantization is that the output distribution of the method is well suited to the Viterbi algorithm, which is used to encode the quantized non-zero values.

Detailed explanation is given in Section 3.3.

A sparse matrix that is generated by Viterbi-based pruning and quantization can be represented using the Viterbi Compression Matrix (VCM) format BID14 .

A sparse matrix stored in VCM format requires much smaller amount of memory than the original dense weight matrix does.

However, it is difficult to parallelize the process of reconstructing sparse matrix from the representation in VCM format, because assigning each non-zero value to its corresponding index requires a sequential process of counting ones in indices generated by the Viterbi encoder.

To address this issue, we encode binary weight codes DISPLAYFORM0 in addition to the indices, based on the same Viterbi algorithm BID3 .

By using similar VD structures ( While using VD structures to generate binary weight codes allows parallel sparse-to-dense matrix conversion, it requires the quantization method to satisfy a specific condition to minimize accuracy DISPLAYFORM1 Proposed process of sparse-to-dense matrix conversion for the Viterbi-based compressed matrix.

This figure shows an example such that weight values and weight index values are generated by independent Viterbi decompressors simultaneously.loss after Viterbi-based encoding.

It is known that the VD structure acts as a random number generator BID13 , which produces '0' and '1' with 50 % probability each.

Thus, generated binary weight codes will be closer to the target binary weight codes if the target binary weight code matrix also consists of equal number of '0' and '1'.

Interestingly, the composition ratio of '-1' and '+1' in each b i , which was generated by the alternating quantization method, is 50 % each.

It is because the weights in DNNs are generally initialized symmetrically with respect to '0' BID4 BID11 and the distribution is maintained even after training BID16 .

The preferable output distribution of the alternating quantization implies that the probability of finding an output matrixb i close to b i with the Viterbi algorithm is high.

For comparison, we measured the accuracy differences before and after Viterbi encoding for several quantization methods such as linear quantization BID16 , logarithmic quantization BID19 , and alternating quantization BID22 .

When the Viterbi encoding is applied to the weight quantized by alternating quantization BID22 , the validation accuracy degrades by only 2 %.

However, accuracy degrades by 71 % when the Viterbi encoding is applied to the weight quantized using other methods BID16 BID19 .

The accuracy difference mainly comes from the uneven weight distribution.

Because weights of neural networks usually have normal distribution, the composition ratio of '0' and '1' is not equal when the linear or logarithmic quantization is applied to the weights unlike alternating quantization.

Another important idea to increase the probability of finding "good" Viterbi encoded weight is to consider the pruned parameters in b i as "Don't Care" terms ( FIG2 .

The "Don't Care" elements can have any values when findingb i , because they will be masked by the zero values in the index matrix generated by the Viterbi pruning.

Next, let us describe how we use the Viterbi algorithm for weight encoding.

We select theb i that best matches with b i among all possibleb i cases that the VD can generate, as follows.

We first construct a trellis diagram as shown in FIG3 .

The trellis diagram is a state diagram represented A cost function for each transition using path and branch metrics is set and computed in the next step.

The branch metric λ i,j t is the cost of traveling along a transition from a state i to the successor state j at the time index t.

The path metric is expressed as DISPLAYFORM2 DISPLAYFORM3 where i1 and i2 are two predecessor states of j. Equation 1 denotes that one of the two possible transitions is selected to maximize the accumulated value of branch metrics.

2 The branch metric is defined as DISPLAYFORM4

To maintain the accuracy, the number of incorrect bits in the encoded binary codeb i compared to the original binary code b i needs to be minimized.

Thus, we retrain the network withŴ P Q = k i=1 α ibi M (M is the index matrix of non-zeros in W), apply the alternating quantization, and then perform the Viterbi encoding repeatedly (Figure 1) .

By repeating the retraining, quantization, and Viterbi encoding, the number of incorrect bits betweenb i and b i can be reduced because the parameters in the network are fine-tuned close to parameters inŴ P Q .

During the retraining period, we apply the straight-through estimate BID20 , i.e. ∂C ∂Ŵ P Q = ∂C ∂W as adopted in BID22 .

After the last Viterbi encoding is finished, small amount of components inb i can be still different from the corresponding values in b i .

To maintain the accuracy, location data for the incorrect components are stored separately and are used to flip the corresponding VD encoded bits during on-chip weight reconstruction period.

In our experiments, the memory requirement for the correction data was negligible.

After the retraining is finished, we can obtain a compressed parameter in Viterbi Weight Matrix (VWM) format, which includes DISPLAYFORM0 , compressed index in VCM format, and indices where DISPLAYFORM1 .

Note that entire training process used the training dataset and the validation dataset only to decide the best compressed weight data.

The accuracy measurement for the test dataset was done only after training is finished so that any hyperparameter was not tuned on the test dataset.

All the experiments in this paper followed the above training principle.

We first conduct experiments on Penn Tree Bank (PTB) corpus BID17 .

We use the standard split version of PTB corpus with 10K vocabulary BID18 , and evaluate the performance using perplexity per word (PPW).

We pretrain the RNN model 3 which contains 1 layer of LSTM with 600 memory units, then prune the parameters of LSTMs with 80 % pruning rate using the Viterbi-based pruning technique 4 and retrain the model.

Then, we quantize the parameters of LSTMs using alternating quantization technique, encode the binary weight codes by using the Viterbi algorithm, and retrain the model.

We repeat the quantization, binary code encoding, and retraining process 5 times.

We quantize the LSTM model with different numbers of quantization bits k with the fixed N o = 5.

As k increases, PPW is improved, but the memory requirement for parameters is also increased ( TAB0 ).

Note that k = 3 is the minimum number of bits that minimizes the model size without PPW degradation.

Compared to BID14 , further quantization and Viterbi-based compression reduce the parameter size by 78 % to 90 % TAB0 .

We compress the binary weight codes with different number of VD outputs N o in case of k = 3.

As N o increases, PPW degrades while the memory requirement for parameters is increased TAB0 .

Large N o implies that the binary weight codes are compressed with high compression ratio 1/N o , but the similarity betweenb i and b i decreases.

The optimal N o is 100/(100-pruning rate (%)) , where the average number of survived parameters per N o serial parameters is 1 statistically, which results in no model performance degradation.

Effectiveness of "Don't Care": To verify the effectiveness of using the "Don't Care" elements, we apply our proposed method on the original network and pruned one.

While the pruned network maintains the original PPW after applying our proposed compression method, applying our method to the dense network degrades PPW to 102.6.

This is because the ratio of incorrect bits betweenb i and b i decreases from 28.3 % to 1.7 % when we use the sparse b i .

Therefore, combination of the Viterbi pruning and alternating quantization increases the probability of findingb i close to b i using the VD for weight encoding, which results in no PPW degradation.

The latest RNN for language modeling: We further test our proposed method on the latest RNN model BID23 , which shows the best perplexity on both PTB and WikiText-2 (WT2) corpus.

We prune 75 % of the parameters in three LSTMs with the same condition as we prune the above 1-layer LSTM model, and quantize them to 3 bits (k = 3).

Note that we do not apply fine-tuning and dynamic evaluation BID12 in this experiment.

The compression result in TAB1 shows that the memory requirements for the models are reduced by 94.7 % with our VWM format on both PTB and WT2 corpus without PPW degradation.

This result implies that our proposed compression method can be applied regardless of the depth and size of the network.

Detailed experiment settings and compression results are described in Appendix A.3.

In addition, we extend our proposed method to the RNN models for machine translation , and its experimental results are presented in Appendix A.4.

We also apply our proposed method to a CNN, VGG-9 (2×128C3 -2×256C3 -2×512C3 -2×1024FC -10SM 5 ) on CIFAR-10 dataset to verify the proposed technique is valid for other types of DNNs.

We randomly select 5 K validation images among 50 K training images in order to observe validation error during retraining process and measure the test error after retraining.

We use k = 3 for all layers.

Optimal N o for each layer is chosen based on the pruning rate of the parameters; N o = 4 for convolutional layers, N o = 25 for the first two fully-connected layers, and N o = 5 for the last fully-connected layer.

We also compute the memory requirement for other compression methods.

Experimental results on VGG-9 is found in Table 3 .

Compared to BID8 , the VWM format generated by the proposed scheme has 39 % smaller memory footprint due to the compressed indices, smaller number of bits for quantization, and encoded binary weight codes.

This experiment on CIFAR-10 shows that our proposed method can be applied to DNNs with various types and sizes.

Meanwhile, it can be seen that the combination of the Viterbi-pruning BID14 ) with the alternating quantization BID22 ) requires 10% smaller memory requirement than the VWM format because the VWM format requires additional memory for indices where DISPLAYFORM0 However, additional "Viterbi-based binary code encoding" process for the VWM format allows parallel sparse-to-dense matrix conversion, which increases the parameter feeding rate up to 40.5 % compared to BID14 .

In Section 4.3, we analyze the speed of sparse-to-dense matrix conversion in detail.

a) Non-zero values are represented as 32-bit floating point numbers.

b) Convolution filters are quantized to 8-bit, and weights of fully-connected layers and indices of sparse matrices are quantized to 5-bit, which is the same quantization condition as the condition used in BID8 .

c) For the Conv1 layer, pruning is not applied and only the alternating quantization is applied.

We built a cycle-level simulator for the weight matrix reconstruction process of the proposed format to show that the sparse matrix-matrix multiplications with the proposed method can be done fast with parallel reconstruction of dense matrix.

In the simulator, baseline structure feeds two dense input matrices to processing elements (PEs) using raw data fed by DRAM FIG5 ), while the proposed structure reconstructs both index masks and binary codes using the highly compressed data fed by DRAM and sends the reconstructed values to PEs FIG5 ).

Both index masks and binary codes are reconstructed by several Viterbi encoders in parallel, and bit errors in binary codes are corrected in a serial manner using the small number of flip-bit related data, which are received from DRAM.

Simulation results show that the feeding rate of the proposed scheme is 20.0-106.4 % higher than the baseline case and 10.3-40.5 % higher than BID14 , depending on the pruning rate ( FIG5 ).

The gain mainly comes from the high compression rate and parallel reconstruction process of the proposed method.

As shown in FIG5 , higher sparsity leads to higher feeding rate.

Higher sparsity allows using many VD outputs for the index N ind , and increasing N ind leads to faster reconstruction.

Also, the reconstruction rate of binary codes becomes higher with reduced number of non-zero values and corresponding bit corrections.

(c) Rate of parameter feeding into PEs for the proposed scheme compared to those for the baseline structure, which receives the dense matrix data directly from DRAM, and BID14 .

We assumed the number of VD outputs for the index N ind = 3, 4, 5, 6, 10, 10 respectively as the reciprocal of each sparsity value.

We used N ind = 10 for 95 % sparsity since we compressed matrices with over 90 % sparsity with N ind = 10.

We also assumed k = 3, and 1 % bit-wise difference betweenb i and b i during simulation.

We also assumed that 16 non-zero parameters can be fed into the PE array in parallel and DRAM requires 10 cycles to handle a 256 bit READ operation.

We proposed a DNN model compression technique with high compression rate and fast dense matrix reconstruction process.

We adopted the Viterbi-based pruning and alternating multi-bit quantization technique to reduce the memory requirement for both non-zeros and indices of sparse matrices.

Then, we encoded the quantized binary weight codes using Viterbi algorithm once more.

As the non-zero values and the corresponding indices are generated in parallel by multiple Viterbi encoders, the sparse-to-dense matrix conversion can be done very fast.

We also demonstrated that the proposed scheme significantly reduces the memory requirements of the parameters for both RNN and CNN.

A APPENDIX A.1 PRUNING USING THE VITERBI ALGORITHM In Viterbi-based pruning scheme, the binary outputs generated by a Viterbi Decompressor (VD) are used as the index matrix that indicates whether a weight element is pruned ('0') or not ('1').

Suppose the number of elements in a target weight matrix is q, and the number of outputs generated by a VD at each time step is N ind , then only 2 q/N ind binary matrices can be generated by the VD among all 2 q binary matrices.

The index matrix which minimizes the accuracy loss should be selected among binary matrix candidates which VD can generate in this pruning scheme, and the Viterbi algorithm is used for this purpose.

The overall pruning process is similar to the binary weight encoding process using the Viterbi algorithm in Section 3.3.

First, Trellis diagram ( FIG3 ) of the VD which is used for pruning is constructed, and then the cost function is computed by using the path metric and the branch metric.

The same path metric shown in Equation 1 in Section 3.3 is used to select the branch which maximizes the path metric between two connected branches from the previous states.

On the other hand, a different branch metric λ i,j t is used for pruning, which is expressed as: DISPLAYFORM0 where W i,j,m t is the magnitude of a parameter at the m th VD output and time index t, normalized by the maximum absolute value of all elements in target weight matrix, and TH p is the pruning threshold value determined heuristically.

As β i,j,m t gives additional points (penalties) to the parameters with large magnitude to survive (be pruned), the possibility to prune small-magnitude parameters is maximized.

S 1 and S 2 are the scaling factors which is empirically determined.

BID14 uses 5.0 and 10 4 each).

After computing the cost function through the whole time steps, the state with the maximum path metric is chosen, and we trace the previous state by selecting the surviving branch and corresponding indices backward until the first state is reached.

The ideal pruning rate of the Viterbi-based pruning is 50 %, because the VD structures act like random number generator and the probability to generate '0' or '1' is 50 % each.

For various pruning rates, comparators and comparator threshold value, TH c , are used.

A N C -bit comparator receives N c VD outputs and generates 1-bit result whether the value made by the combination of the received VD outputs (e.g. {out 1 , out 2 , · · · , out N ind } where out i indicates the i th VD output) is greater than TH c or not.

For example, suppose a 4-bit comparator is used to the VD in Figure 1 and TH c = 3, then the probability for the comparator to generate '1' is 25%(= (3 + 1)/2 4 ) and this percentage is the target pruning rate.

Comparators and TH c control the value of pruning rates and the index compression ratio decreases by 1/N c times.

It is reported that a low N ind is desired to prune weights of convolutional layers while high N ind can be used to prune the weights of fully-connected layers because of the trade-off between the index compression ratio and the accuracy BID14 .

Thus, in our paper, we use N ind = 50 and N c = 5 to prune weights of LSTMs and fully-connected layers in VGG-6 on CIFAR-10.

On the other hand, we use N ind = 10 and N c = 5 to prune weights of convolutional layers in VGG-6 on CIFAR-10.

The RNN model in BID23 is composed of three LSTM layers, and use various learning techniques such as mixture-of-softmaxes (MoS) to achieve better perplexity.

As shown in TAB0 , the parameters in the first layer have high sparsity, so we use N o = 6.

In the remaining layers, however, we use N o = 3 because the parameters are pruned with only about 70 % pruning rate.

We repeat the process of quantization, binary code encoding, and retraining only once.

We also extend our experiments on the RNN models for machine translation 6 .

We use the model which consists of an encoder, a decoder and an attention layer.

4-layer LSTMs with 1024 units compose each encoder and decoder.

A bidirectional LSTM (BiLSTM) is used for the first layer of the encoder.

The weights of LSTM models are pruned with 75 % pruning rate by the Viterbi-based pruning techinque, then k = 4 is used for quantization.

Optimal N o values are used according to the sparsity of each LSTM layer (i.e. 3 ≤ N o ≤ 6 is enough to encode binary weight codes with 70 -83% of sparsity).

The process of quantization, binary code encoding, and retraining is repeated only once in this case, too.

As shown in TAB5 , we reduce the memory requirement of each baseline model by 93.5 % using our proposed technique.

This experiment results show that our proposed scheme can be extended to RNNs for other complex tasks.

<|TLDR|>

@highlight

We present a new weight encoding scheme which enables high compression ratio and fast sparse-to-dense matrix conversion.