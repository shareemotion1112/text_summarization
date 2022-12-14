In this paper, we investigate lossy compression of deep neural networks (DNNs) by weight quantization and lossless source coding for memory-efficient deployment.

Whereas the previous work addressed non-universal scalar quantization and entropy coding of DNN weights, we for the first time introduce universal DNN compression by universal vector quantization and universal source coding.

In particular, we examine universal randomized lattice quantization of DNNs, which randomizes DNN weights by uniform random dithering before lattice quantization and can perform near-optimally on any source without relying on knowledge of its probability distribution.

Moreover, we present a method of fine-tuning vector quantized DNNs to recover the performance loss after quantization.

Our experimental results show that the proposed universal DNN compression scheme compresses the 32-layer ResNet (trained on CIFAR-10) and the AlexNet (trained on ImageNet) with compression ratios of $47.1$ and $42.5$, respectively.

Compression of deep neural networks (DNNs) has been actively studied in deep learning to develop compact DNN models for memory-efficient and computation-efficient deployment.

Han et al. BID0 showed impressive compression results by weight pruning, k-means clustering, and Huffman coding.

It is further optimized in BID1 using Hessian-weighted k-means clustering.

Recently, it is shown how soft weight sharing or soft quantization can be employed for DNN weight quantization in BID2 BID3 .

On the other hand, weight pruning is also extensively studied, e.g., in BID4 BID5 BID6 BID7 BID8 .

In this paper, we focus on DNN weight quantization, which can be used together with weight pruning to generate compressed models.

Vector quantization reduces the gap to the rate-distortion bound by jointly quantizing multiple symbols.

Since conjectured by Gersho in BID9 , lattice quantization has been presumed to be the most efficient entropy coded vector quantization in the high resolution regime asymptotically, as the rate goes to infinity and the distortion diminishes BID10 .

Although lattice quantizers are simple and empirically shown to perform well even at finite rates, their efficiency depends on source statistics.

Thus, we consider universal quantization that provides near-optimal performance for any source distribution BID11 .

Of particular interest is randomized lattice quantization, where uniform random dithering makes the distortion independent of the source, and the gap of its rate from the rate-distortion bound at any distortion level is provably no more than 0.754 bits per sample for any finite dimension BID12 .From the classical lossy compression results, this paper establishes a universal DNN compression framework consisting of universal quantization and universal lossless source coding such as LempelZiv-Welch BID13 BID14 BID15 and the Burrows-Wheeler transform BID16 BID17 .

In order to recover any accuracy loss resulting from weight quantization, we furthermore propose a fine-tuning algorithm for vector quantized DNNs.

The gain of fine-tuning becomes larger as the vector dimension increases, due to the fact that the number of shared quantized values that are tunable (trainable) in a vector quantized model increases as the vector dimension increases.

For vector quantization, given N weights denoted by w 1 , . . .

, w N , we generate n-dimensional vectors v 1 , . . .

, v ???N/n??? by concatenating n distinct weights into one vector, e.g., as follows: DISPLAYFORM0 where w j = 0 for N + 1 ??? j ??? ???N/n???n, and ???x??? is the smallest integer larger than or equal to x. Vector quantization partitions these n-dimensional vectors into a finite number of clusters, and the vectors in each cluster share their quantized value, i.e., the cluster center.

Randomized lattice quantization: Randomized lattice quantization BID11 achieves universally good performance regardless of source statistics at any rates, and this leads us to the following universal DNN quantization method.??? We randomize the n-dimensional vectors in (1) by adding uniform random dithers as follows: DISPLAYFORM1 where each dithering vector u i consists of n repetitions of a single uniform random variable U i , and U 1 , . . .

U ???N/n??? are independent and identically distributed (i.i.d.) uniform random variables of support [??????/2, ???/2]; ??? is the quantization bin size.

In each dimension, dithering values are i.i.d.

uniform, which is sufficient to make the quantization error independent of source statistics.??? After dithering, uniform quantization in each dimension (i.e., lattice quantization) follows, i.e., DISPLAYFORM2 where q i is the quantized vector of??? i ; the rounding and the scaling are element-wise operations in (3), where the rounding yields the closest integer values of the input.

Remark 1.

Vector quantization theoretically provides a better rate-distortion trade-off.

However, in practice, for compression of a finite number of data, the gain of vector quantization is limited by the codebook overhead, which becomes more considerable as dimension n increases and becomes the dominant factor that degrades the compression ratio after some point (see FIG1 and compare the cases of n = 4 and n = 6).Fine-tuning of vector quantized DNNs: We fine-tune the vector quantization codebook to recover the loss after quantization.

Each element of a shared quantized vector in the codebook is fine-tuned separately.

That is, if we have k VQ clusters of n-dimensional vectors, we effectively divide weights into nk VQ groups and fine-tune their shared quantized values separately.

The average gradient of the network loss function with respect to weights is computed in each group and used to update their shared quantized value, as will be clarified below.

DISPLAYFORM3 T be the shared quantized vector present in the codebook for cluster i, and let I i,j be the index set of all weights that are quantized to the same value c i,j from (3), for 1 ??? i ??? k VQ and 1 ??? j ??? n. The gradient descent for the shared quantized value c i,j is then given by DISPLAYFORM4 where t is the iteration time, L is the network loss function, and ?? is the learning rate.

The individual quantized vectors from (3) are also updated following their shared quantized vectors in the codebook.

In randomized lattice quantization, we would like to clarify that the average gradients are computed from the network loss function evaluated at the quantized weights after canceling random dithers (see FORMULA5 ), while we fine-tune the shared values obtained before canceling random dithers (see (3)).

Universal source coding: Universal source coding algorithms are more convenient in practice than Huffman coding since they do not require us to estimate source statistics as in Huffman coding.

Moreover, they utilize dictionary-based coding, where the codebook (i.e., dictionary) is built from source symbols adaptively in encoding and decoding, and therefore the codebook overhead is smaller than Huffman coding (see Remark 1).For universal compression, the indices in the codebook of the lattice quantization output are passed as an input stream to a universal source coding scheme, which produces a compressed stream.

The decoder needs to deploy the codebook that contains the indices and their corresponding fine-tuned shared quantized values for decompression.

Decompression: Using randomized lattice quantization, the encoder and the decoder are assumed to share the information on random dithers or their random seed.

Under this assumption, the decoder decompresses the fine-tuned lattice quantization output from the compressed stream and then cancels the dithering vectors to obtainv DISPLAYFORM5 which yields the deployed weights of the universally quantized DNN at the inference step.

We evaluate the proposed DNN compression scheme first without pruning for the 32-layer ResNet BID18 (ResNet-32) model on CIFAR-10 dataset BID19 .

We consider two cases for (randomized) lattice quantization where the uniform boundaries in each dimension are set from (a) {i???, i ??? Z} and (b) {(2i + 1)???/2, i ??? Z}, respectively; Z is the set of integers.

The quantization bin size ??? is the same for both cases, but case (a) has the zero at a bin boundary while in case (b) the zero element is at the center of the middle bin.

For unpruned models, we often have a high volume of weights concentrated around zero, and thus case (b) that assigns one bin to include all the weights near zero is expected to outperform case (a), which is aligned with our lattice quantization results in FIG1 .

However, it is interesting to observe that randomized lattice quantization provides similarly good performance in both cases, which is the main benefit of randomizing the source by uniform dithering before quantization.

FIG1 also shows that vector quantization provides additional gain over scalar quantization particularly when the compression ratio is large.

Finally, TAB1 summarizes the compression ratios that we obtain from our universal DNN compression method for pruned ResNet-32 and AlexNet BID20 models.

The proposed universal DNN compression scheme with the bzip2 BID17 universal source coding algorithm yields 47.10?? and 42.46?? compression for ResNet-32 and AlexNet, respectively.

Compared with BID0 BID1 BID3 ] which need to optimize and/or calculate source statistics for compression, we achieved a better trade-off between rate (compression ratio) and distortion (loss in accuracy) through the universal compression of DNNs.

FORMULA0 ); here, uniform quantization corresponds to lattice quantization with dimension n = 1.

Given the vector dimension n, the weights from all layers of the pre-trained ResNet-32 model are vectorized as in FORMULA0 for vector quantization.

Then, lattice quantization or randomized lattice quantization follows.

In plain lattice quantization, no random dithering is added before quantization, i.e., we set u i = 0 for all i in (2).

We fine-tune the quantization codebook as explained in Section 2.

We simply use Huffman coding only in this experiment to get the compressed models.

The gain of randomized lattice quantization over lattice quantization can be found in FIG1 (a) in particular for n ??? 2 and large compression ratios.

We note that randomized lattice quantizers provide similarly good performance in both cases (a) and (b).

Lattice quantization performs well only in case (b), where the quantization bins are optimized for given weight distribution.

We emphasize that randomized lattice quantization is applicable for any network models blindly, regardless of their weight distribution and with no optimization, while it is guaranteed to yield a good rate-distortion trade-off close to the optimum within a fixed gap BID11 .

In FIG2 , we show the performance of universally quantized ResNet-32 models before and after fine-tuning the codebook.

The gain of vector quantization becomes more significant after fine-tuning, in particular, as the vector dimension n increases, since there are a more number of shared quantized values trainable in vector quantized models.

A.3 Contribution of pruning, quantization, and source coding TAB2 shows the incremental improvement of the compression ratio for the AlexNet model.

Given a pre-trained AlexNet model, we prune its 90.4% weights and fine-tune remaining unpruned weights, as suggested in BID21 , which yields the compression ratio of 9.11 with the top-1 accuracy of 57.28%.Using universal quantization and bzip2 universal source coding for the pruned AlexNet model, we achieve the compression ratio of 42.46 with the top-1 accuracy of 57.02%.

We also compare Huffman coding to the universal source coding algorithms, i.e., Lempel-Ziv-Welch (LZW) and bzip2 that deploys the Burrows-Wheeler transform BID16 .

Both LZW and bzip2 provide better compression ratios than Huffman coding in our experiments.

@highlight

We introduce the universal deep neural network compression scheme, which is applicable universally for compression of any models and can perform near-optimally regardless of their weight distribution.

@highlight

Introduces a pipeline for network compression that is similar to deep compression and uses randomized lattice quantization instead of the classical vector quantization, and uses universal source coding (bzip2) instead of Huffman coding.